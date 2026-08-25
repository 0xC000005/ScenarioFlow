#!/usr/bin/env python
"""
176b: Shared-law latent local-template mixture on top of the 173a backbone.

Motivation from 176a mechanistic analysis:
  - the full latent-residual experts did not specialize meaningfully
  - posterior assignments collapsed entirely to one expert
  - unused experts were simply bad experts, not alternative useful local laws

Keep 173a:
  - support-aware transformed-space density modeling
  - one-shot future block generation
  - exact joint likelihood
  - structured separable time/cell covariance
  - conditional residual flow
  - shared mean / shared global covariance / shared residual flow

Add:
  - discrete latent routing over small additive local log-variance templates
  - batch-level gate usage regularization
  - explicit shrinkage on both the shared local field and the component templates
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_172a_residual_flow_structured_joint_student_t import (
    ConditionalResidualFlow,
    ResidualFlowStructuredJointDecoder,
    ResidualFlowStructuredJointStudentTModel,
)
from experiments.backfill.block_ar.train_173a_local_var_residual_flow_structured_joint_student_t import (
    evaluate_joint_subset,
)


class SharedLocalTemplateMixtureDecoder(ResidualFlowStructuredJointDecoder):
    """173a shared law plus discrete local log-variance templates."""

    def __init__(
        self,
        n_frames: int = 30,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        time_rank: int = 6,
        cell_rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        init_diag: float = 0.05,
        init_scale: float = 0.10,
        flow_context_dim: int = 256,
        local_delta_clip: float = 0.35,
        n_components: int = 3,
    ):
        super().__init__(
            n_frames=n_frames,
            n_cells=n_cells,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            cond_dim=cond_dim,
            time_rank=time_rank,
            cell_rank=cell_rank,
            diag_floor=diag_floor,
            scale_floor=scale_floor,
            init_diag=init_diag,
            init_scale=init_scale,
            flow_context_dim=flow_context_dim,
        )
        self.local_delta_clip = local_delta_clip
        self.n_components = n_components

        self.local_logvar_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.local_logvar_head.weight)
        nn.init.zeros_(self.local_logvar_head.bias)

        self.local_template_bank = nn.Parameter(torch.zeros(n_components, n_frames, n_cells))
        nn.init.normal_(self.local_template_bank, mean=0.0, std=5e-3)

        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_components),
        )
        for module in self.gate_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_head[-1].weight)

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = prev_u.shape[0]
        n_frames, n_cells = self.n_frames, self.n_cells

        prev_rep = prev_u.unsqueeze(1).expand(batch, n_frames, n_cells)
        hidden = self.input_proj(prev_rep.unsqueeze(-1))
        hidden = hidden + self.cond_proj(cond).unsqueeze(1).unsqueeze(1)
        hidden = hidden + self.temporal_pos + self.spatial_pos

        for layer in self.layers:
            h_temp = hidden.permute(0, 2, 1, 3).reshape(batch * n_cells, n_frames, -1)
            h_norm = layer["temp_norm"](h_temp)
            attn_out, _ = layer["temp_attn"](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer["temp_ff"](layer["temp_ff_norm"](h_temp))
            hidden = h_temp.reshape(batch, n_cells, n_frames, -1).permute(0, 2, 1, 3)

            h_spat = hidden.reshape(batch * n_frames, n_cells, -1)
            h_norm = layer["spat_norm"](h_spat)
            attn_out, _ = layer["spat_attn"](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer["spat_ff"](layer["spat_ff_norm"](h_spat))
            hidden = h_spat.reshape(batch, n_frames, n_cells, -1)

        hidden = self.out_norm(hidden)
        pooled = hidden.mean(dim=(1, 2))
        time_summary = hidden.mean(dim=2)
        cell_summary = hidden.mean(dim=1)

        mu = prev_rep + self.mean_head(hidden).squeeze(-1)
        time_factor = self.time_factor_head(time_summary)
        time_diag = F.softplus(self.time_diag_head(time_summary).squeeze(-1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        flow_context = self.flow_context_head(torch.cat([cond, prev_u], dim=-1))

        base_local_delta_raw = self.local_logvar_head(hidden).squeeze(-1)
        base_local_delta = torch.tanh(base_local_delta_raw) * self.local_delta_clip
        base_local_delta = base_local_delta - base_local_delta.mean(dim=(1, 2), keepdim=True)

        local_delta_components = base_local_delta.unsqueeze(1) + self.local_template_bank.unsqueeze(0)
        local_delta_components = torch.tanh(local_delta_components) * self.local_delta_clip
        local_delta_components = local_delta_components - local_delta_components.mean(
            dim=(2, 3), keepdim=True
        )

        gate_logits = self.gate_head(pooled)

        return (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            local_delta_components,
            gate_logits,
        )


class SharedLocalTemplateMixtureStudentTModel(ResidualFlowStructuredJointStudentTModel):
    """173a backbone with discrete local-variance template routing."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
    ):
        base_decoder_config = dict(decoder_config)
        base_decoder_config.pop("local_delta_clip", None)
        base_decoder_config.pop("n_components", None)
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=base_decoder_config,
            flow_config=flow_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
        )
        self.decoder = SharedLocalTemplateMixtureDecoder(**decoder_config)
        self.decoder_config = decoder_config
        self.flow = ConditionalResidualFlow(**flow_config)
        self.flow_config = flow_config

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        return self.decoder(cond, prev_u)

    def mixture_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        flow_context: torch.Tensor,
        local_delta_components: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch, n_frames, n_cells = target_u.shape
        n_components = gate_logits.shape[1]

        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        local_scale = torch.exp(0.5 * local_delta_components)
        diff = (target_u.unsqueeze(1) - mu.unsqueeze(1)) / local_scale
        white_t = torch.linalg.solve_triangular(chol_t.unsqueeze(1), diff, upper=False)
        white = torch.linalg.solve_triangular(
            chol_c.unsqueeze(1), white_t.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        white_flat = white.reshape(batch * n_components, n_frames * n_cells)

        flow_context_rep = flow_context.unsqueeze(1).expand(batch, n_components, -1)
        z, flow_logdet = self.flow(white_flat, flow_context_rep.reshape(batch * n_components, -1))

        base_logprob = self._base_logprob(z).view(batch, n_components)
        flow_logdet = flow_logdet.view(batch, n_components)

        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c
        logdet_local = 2.0 * torch.log(local_scale).sum(dim=(2, 3))

        comp_log_prob = base_logprob + flow_logdet - 0.5 * (logdet_cov.unsqueeze(1) + logdet_local)
        log_prior = F.log_softmax(gate_logits, dim=-1)
        mix_log_prob = torch.logsumexp(log_prior + comp_log_prob, dim=1)
        posterior = F.softmax(log_prior + comp_log_prob, dim=-1)

        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "flow_logdet": flow_logdet,
            "white_std": white_flat.std(dim=-1).view(batch, n_components),
            "z_std": z.std(dim=-1).view(batch, n_components),
            "local_delta_rms": local_delta_components.pow(2).mean(dim=(2, 3)).sqrt(),
            "local_scale_min": local_scale.amin(dim=(2, 3)),
            "local_scale_max": local_scale.amax(dim=(2, 3)),
        }
        return mix_log_prob, posterior, aux

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            _base_local_delta,
            local_delta_components,
            gate_logits,
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        white_flat, _ = self.flow.inverse(z, ctx)
        white = white_flat.view(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)

        prior = F.softmax(gate_logits, dim=-1)
        component_idx = torch.multinomial(prior, num_samples=n_samples, replacement=True)
        batch_idx = torch.arange(batch, device=mu.device).unsqueeze(1)
        selected_local_scale = torch.exp(0.5 * local_delta_components)[batch_idx, component_idx]
        samples_u = mu.unsqueeze(1) + noise * selected_local_scale
        return samples_u, component_idx

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        history_01 = denormalize_iv(history)
        batch_size = history_01.shape[0]
        chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            samples_u, _ = self.sample_future_u(history_01, n_samples=k)
            samples_01 = unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)
            chunks.append(samples_01.view(batch_size, k, self.decoder.n_frames, 5, 5))
        return torch.cat(chunks, dim=1)


def joint_nll_loss(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
) -> tuple[torch.Tensor, dict]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        base_local_delta,
        local_delta_components,
        gate_logits,
    ) = model.forward_from_history(history_01)
    logprob, posterior, aux = model.mixture_log_prob(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        flow_context=flow_context,
        local_delta_components=local_delta_components,
        gate_logits=gate_logits,
    )

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    prior = F.softmax(gate_logits, dim=-1)
    usage = prior.mean(dim=0)
    target_usage = torch.full_like(usage, 1.0 / usage.numel())
    usage_penalty = (usage - target_usage).pow(2).mean()

    nll = (-logprob).mean()
    local_pen = base_local_delta.pow(2).mean()
    template_pen = model.decoder.local_template_bank.pow(2).mean()
    loss = nll + local_var_penalty * local_pen + template_penalty * template_pen + gate_balance_penalty * usage_penalty

    posterior_det = posterior.detach()
    weighted_local_delta_rms = (posterior_det * aux["local_delta_rms"]).sum(dim=1).mean()
    weighted_local_scale_min = (posterior_det * aux["local_scale_min"]).sum(dim=1).mean()
    weighted_local_scale_max = (posterior_det * aux["local_scale_max"]).sum(dim=1).mean()
    weighted_white_std = (posterior_det * aux["white_std"]).sum(dim=1).mean()
    weighted_z_std = (posterior_det * aux["z_std"]).sum(dim=1).mean()

    metrics = {
        "joint_nll": nll,
        "total_loss": loss,
        "local_var_penalty": local_pen,
        "template_penalty": template_pen,
        "gate_usage_penalty": usage_penalty,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": (posterior_det * aux["flow_logdet"]).sum(dim=1).mean(),
        "white_std_mean": weighted_white_std,
        "z_std_mean": weighted_z_std,
        "local_delta_rms": weighted_local_delta_rms,
        "local_scale_min": weighted_local_scale_min,
        "local_scale_max": weighted_local_scale_max,
        "prior_entropy": (-(prior * torch.log(prior.clamp_min(1e-8))).sum(dim=-1)).mean(),
        "posterior_entropy": (-(posterior_det * torch.log(posterior_det.clamp_min(1e-8))).sum(dim=-1)).mean(),
        "gate_max_prob": prior.max(dim=-1).values.mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: SharedLocalTemplateMixtureStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
) -> dict:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_joint_nll": 0.0,
        "val_local_var_penalty": 0.0,
        "val_template_penalty": 0.0,
        "val_gate_usage_penalty": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_flow_logdet_mean": 0.0,
        "val_white_std_mean": 0.0,
        "val_z_std_mean": 0.0,
        "val_local_delta_rms": 0.0,
        "val_local_scale_min": 0.0,
        "val_local_scale_max": 0.0,
        "val_prior_entropy": 0.0,
        "val_posterior_entropy": 0.0,
        "val_gate_max_prob": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        _loss, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            template_penalty=template_penalty,
            gate_balance_penalty=gate_balance_penalty,
        )
        batch_size = history_01.shape[0]
        totals["val_total_loss"] += metrics["total_loss"].item() * batch_size
        totals["val_joint_nll"] += metrics["joint_nll"].item() * batch_size
        totals["val_local_var_penalty"] += metrics["local_var_penalty"].item() * batch_size
        totals["val_template_penalty"] += metrics["template_penalty"].item() * batch_size
        totals["val_gate_usage_penalty"] += metrics["gate_usage_penalty"].item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_time_eff_rank"] += metrics["time_eff_rank"].item() * batch_size
        totals["val_cell_eff_rank"] += metrics["cell_eff_rank"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_flow_logdet_mean"] += metrics["flow_logdet_mean"].item() * batch_size
        totals["val_white_std_mean"] += metrics["white_std_mean"].item() * batch_size
        totals["val_z_std_mean"] += metrics["z_std_mean"].item() * batch_size
        totals["val_local_delta_rms"] += metrics["local_delta_rms"].item() * batch_size
        totals["val_local_scale_min"] += metrics["local_scale_min"].item() * batch_size
        totals["val_local_scale_max"] += metrics["local_scale_max"].item() * batch_size
        totals["val_prior_entropy"] += metrics["prior_entropy"].item() * batch_size
        totals["val_posterior_entropy"] += metrics["posterior_entropy"].item() * batch_size
        totals["val_gate_max_prob"] += metrics["gate_max_prob"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="176b: shared-law local-template mixture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--template_penalty", type=float, default=0.5)
    parser.add_argument("--gate_balance_penalty", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    hist_len = args.history_len
    future_len = args.future_len

    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
        n_components=args.n_components,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = SharedLocalTemplateMixtureStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("176b: shared-law local-template mixture")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")
    print(f"  Components={args.n_components} | local clip={args.local_delta_clip}")
    print(
        f"  Penalties: local={args.local_var_penalty} | template={args.template_penalty} "
        f"| gate_balance={args.gate_balance_penalty}"
    )
    print("  Shared mean / shared covariance / shared residual flow")
    print("  Latent routing only changes local uncertainty templates")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": args.lr_encoder,
                "weight_decay": args.weight_decay_encoder,
            },
            {
                "params": model.decoder.parameters(),
                "lr": args.lr_decoder,
                "weight_decay": args.weight_decay_decoder,
            },
            {
                "params": model.flow.parameters(),
                "lr": args.lr_flow,
                "weight_decay": args.weight_decay_flow,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {
            "train_total_loss": 0.0,
            "train_joint_nll": 0.0,
            "train_local_var_penalty": 0.0,
            "train_template_penalty": 0.0,
            "train_gate_usage_penalty": 0.0,
            "train_joint_mae": 0.0,
            "train_time_eff_rank": 0.0,
            "train_cell_eff_rank": 0.0,
            "train_scale_mean": 0.0,
            "train_flow_logdet_mean": 0.0,
            "train_white_std_mean": 0.0,
            "train_z_std_mean": 0.0,
            "train_local_delta_rms": 0.0,
            "train_local_scale_min": 0.0,
            "train_local_scale_max": 0.0,
            "train_prior_entropy": 0.0,
            "train_posterior_entropy": 0.0,
            "train_gate_max_prob": 0.0,
        }
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
                template_penalty=args.template_penalty,
                gate_balance_penalty=args.gate_balance_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key in totals:
                metric_key = key.replace("train_", "")
                totals[key] += metrics[metric_key].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}

        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
            template_penalty=args.template_penalty,
            gate_balance_penalty=args.gate_balance_penalty,
        )
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_total_loss"] < best_val
        if is_best:
            best_val = val_metrics["val_total_loss"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": best_val,
                    "config": {
                        "type": "shared_local_template_mixture_residual_flow_structured_joint_student_t_176b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "local_var_penalty": args.local_var_penalty,
                        "template_penalty": args.template_penalty,
                        "gate_balance_penalty": args.gate_balance_penalty,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"priorH={val_metrics['val_prior_entropy']:.3f}  "
            f"postH={val_metrics['val_posterior_entropy']:.3f}  "
            f"gate={val_metrics['val_gate_max_prob']:.3f}  "
            f"ld_rms={val_metrics['val_local_delta_rms']:.3f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "shared_local_template_mixture_residual_flow_structured_joint_student_t_176b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "local_var_penalty": args.local_var_penalty,
            "template_penalty": args.template_penalty,
            "gate_balance_penalty": args.gate_balance_penalty,
        },
        "best_val_total_loss": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    print(f"\nBest val total loss: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_support_violation_rate={best_metrics['joint_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
