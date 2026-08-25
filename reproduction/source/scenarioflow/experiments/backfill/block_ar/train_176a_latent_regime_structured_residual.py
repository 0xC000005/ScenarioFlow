#!/usr/bin/env python
"""
176a: shared-mean latent regime-conditioned structured residual model.

Keep from 173a:
  - support-aware transformed-space density modeling
  - one-shot future block generation
  - shared realism-first conditional mean path
  - structured separable time/cell covariance
  - conditional residual flow in whitened residual space

Change:
  - replace the single smooth residual law with K latent residual regimes
  - history-conditioned gating over residual regimes
  - regime-specific covariance / local variance / flow context
  - exact mixture likelihood over latent regimes
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    inverse_softplus,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
    eff_rank_np,
)
from experiments.backfill.block_ar.train_172a_residual_flow_structured_joint_student_t import (
    ConditionalResidualFlow,
)


class SharedMeanLatentRegimeStructuredResidualDecoder(nn.Module):
    """Shared mean path plus K latent residual regimes."""

    def __init__(
        self,
        n_frames: int = 30,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        n_components: int = 3,
        time_rank: int = 6,
        cell_rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        init_diag: float = 0.05,
        init_scale: float = 0.10,
        flow_context_dim: int = 256,
        local_delta_clip: float = 0.35,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.n_components = n_components
        self.time_rank = time_rank
        self.cell_rank = cell_rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.local_delta_clip = local_delta_clip
        self.flow_context_dim = flow_context_dim

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_cells, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "temp_norm": nn.LayerNorm(d_model),
                        "temp_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "temp_ff_norm": nn.LayerNorm(d_model),
                        "temp_ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                        "spat_norm": nn.LayerNorm(d_model),
                        "spat_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "spat_ff_norm": nn.LayerNorm(d_model),
                        "spat_ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.time_factor_head = nn.Linear(d_model, n_components * time_rank)
        self.time_diag_head = nn.Linear(d_model, n_components)
        self.cell_factor_head = nn.Linear(d_model, n_components * cell_rank)
        self.cell_diag_head = nn.Linear(d_model, n_components)
        self.scale_head = nn.Linear(d_model, n_components)
        self.local_logvar_head = nn.Linear(d_model, n_components)
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_components),
        )
        self.flow_context_head = nn.Sequential(
            nn.Linear(cond_dim + n_cells, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_components * flow_context_dim),
        )

        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in (
                        "mean_head",
                        "time_factor_head",
                        "time_diag_head",
                        "cell_factor_head",
                        "cell_diag_head",
                        "scale_head",
                        "local_logvar_head",
                        "gate_head",
                        "flow_context_head",
                    )
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.normal_(self.time_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.time_factor_head.bias, mean=0.0, std=1e-3)
        nn.init.normal_(self.cell_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.cell_factor_head.bias, mean=0.0, std=1e-3)

        nn.init.zeros_(self.time_diag_head.weight)
        nn.init.constant_(
            self.time_diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )
        nn.init.zeros_(self.cell_diag_head.weight)
        nn.init.constant_(
            self.cell_diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            inverse_softplus(max(init_scale - self.scale_floor, 1e-6)),
        )
        nn.init.zeros_(self.local_logvar_head.weight)
        nn.init.zeros_(self.local_logvar_head.bias)

        for module in self.gate_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_head[-1].weight)

        for module in self.flow_context_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

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
        n_frames, n_cells, n_components = self.n_frames, self.n_cells, self.n_components

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
        time_factor = self.time_factor_head(time_summary).view(batch, n_frames, n_components, self.time_rank).permute(0, 2, 1, 3)
        time_diag = F.softplus(self.time_diag_head(time_summary).permute(0, 2, 1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary).view(batch, n_cells, n_components, self.cell_rank).permute(0, 2, 1, 3)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).permute(0, 2, 1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled)) + self.scale_floor

        local_delta_raw = self.local_logvar_head(hidden).permute(0, 3, 1, 2)
        local_delta = torch.tanh(local_delta_raw) * self.local_delta_clip
        local_delta = local_delta - local_delta.mean(dim=(2, 3), keepdim=True)

        gate_logits = self.gate_head(pooled)
        flow_context = self.flow_context_head(torch.cat([cond, prev_u], dim=-1)).view(
            batch, n_components, self.flow_context_dim
        )

        return (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            local_delta,
            gate_logits,
            flow_context,
        )


class LatentRegimeStructuredResidualStudentTModel(nn.Module):
    """Shared mean path plus mixture of structured latent residual laws."""

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
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SharedMeanLatentRegimeStructuredResidualDecoder(**decoder_config)
        self.flow = ConditionalResidualFlow(**flow_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.flow_config = flow_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter
        self.base_nu = float(base_nu)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.encoder(history_norm)

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

    def normalized_components(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm

    def covariance_parts(
        self,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_factor_norm, time_diag_norm = self.normalized_components(time_factor, time_diag)
        cell_factor_norm, cell_diag_norm = self.normalized_components(cell_factor, cell_diag)

        cov_t = time_factor_norm @ time_factor_norm.transpose(-1, -2)
        cov_t = cov_t + torch.diag_embed(time_diag_norm.pow(2) + self.cov_jitter)
        cov_c = cell_factor_norm @ cell_factor_norm.transpose(-1, -2)
        cov_c = cov_c + torch.diag_embed(cell_diag_norm.pow(2) + self.cov_jitter)
        scale = scale.clamp_min(self.decoder.scale_floor)
        return cov_t * scale[:, :, None, None], cov_c

    def _base_logprob(self, z: torch.Tensor) -> torch.Tensor:
        nu = z.new_tensor(self.base_nu)
        log_norm = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * z.new_tensor(math.pi))
        )
        log_kernel = -0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
        return (log_norm + log_kernel).sum(dim=-1)

    def mixture_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        local_delta: torch.Tensor,
        gate_logits: torch.Tensor,
        flow_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch, n_frames, n_cells = target_u.shape
        n_components = gate_logits.shape[1]

        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        local_scale = torch.exp(0.5 * local_delta)
        diff = (target_u.unsqueeze(1) - mu.unsqueeze(1)) / local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white = torch.linalg.solve_triangular(
            chol_c, white_t.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        white_flat = white.reshape(batch * n_components, n_frames * n_cells)
        flow_context_flat = flow_context.reshape(batch * n_components, -1)

        z, flow_logdet = self.flow(white_flat, flow_context_flat)
        base_logprob = self._base_logprob(z).view(batch, n_components)
        flow_logdet = flow_logdet.view(batch, n_components)

        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c
        logdet_local = 2.0 * torch.log(local_scale).sum(dim=(2, 3))

        comp_log_prob = base_logprob + flow_logdet - 0.5 * (logdet_cov + logdet_local)
        log_mix = F.log_softmax(gate_logits, dim=-1)
        mix_log_prob = torch.logsumexp(log_mix + comp_log_prob, dim=1)
        posterior = F.softmax(log_mix + comp_log_prob, dim=-1)

        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "white_std": white_flat.std(dim=-1).view(batch, n_components),
            "z_std": z.std(dim=-1).view(batch, n_components),
            "flow_logdet": flow_logdet,
            "local_scale_min": local_scale.amin(dim=(2, 3)),
            "local_scale_max": local_scale.amax(dim=(2, 3)),
            "local_delta_rms": local_delta.pow(2).mean(dim=(2, 3)).sqrt(),
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
            local_delta,
            gate_logits,
            flow_context,
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        probs = F.softmax(gate_logits, dim=-1)
        n_components = probs.shape[1]

        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        local_scale = torch.exp(0.5 * local_delta)

        comp_idx = torch.distributions.Categorical(probs=probs).sample((n_samples,)).transpose(0, 1)
        batch_idx = torch.arange(batch, device=mu.device).unsqueeze(1)

        chol_t_sel = chol_t[batch_idx, comp_idx]
        chol_c_sel = chol_c[batch_idx, comp_idx]
        local_scale_sel = local_scale[batch_idx, comp_idx]
        flow_ctx_sel = flow_context[batch_idx, comp_idx]

        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        white_flat, _ = self.flow.inverse(z, flow_ctx_sel.reshape(batch * n_samples, -1))
        white = white_flat.view(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bsij,bsjk->bsik", chol_t_sel, white)
        noise = torch.einsum("bstj,bscj->bstc", temp, chol_c_sel)
        samples_u = mu.unsqueeze(1) + noise * local_scale_sel
        return samples_u, probs

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
    model: LatentRegimeStructuredResidualStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
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
        local_delta,
        gate_logits,
        flow_context,
    ) = model.forward_from_history(history_01)
    log_prob, posterior, aux = model.mixture_log_prob(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        local_delta=local_delta,
        gate_logits=gate_logits,
        flow_context=flow_context,
    )

    nll = (-log_prob).mean()
    local_pen = local_delta.pow(2).mean()

    prior = F.softmax(gate_logits, dim=-1)
    uniform = torch.full_like(prior.mean(dim=0), 1.0 / prior.shape[-1])
    usage_pen = (prior.mean(dim=0) - uniform).pow(2).mean()
    loss = nll + local_var_penalty * local_pen + gate_balance_penalty * usage_pen
    cov_t_mix = (prior[:, :, None, None] * aux["cov_t"]).sum(dim=1)
    cov_c_mix = (prior[:, :, None, None] * aux["cov_c"]).sum(dim=1)
    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    prior_entropy = -(prior * torch.log(prior.clamp_min(1e-8))).sum(dim=-1)
    post_entropy = -(posterior * torch.log(posterior.clamp_min(1e-8))).sum(dim=-1)
    max_prob = prior.max(dim=-1).values

    metrics = {
        "joint_nll": nll,
        "total_loss": loss,
        "local_var_penalty": local_pen,
        "gate_usage_penalty": usage_pen,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "time_eff_rank": effective_rank(cov_t_mix).mean(),
        "cell_eff_rank": effective_rank(cov_c_mix).mean(),
        "scale_mean": (prior * scale).sum(dim=-1).mean(),
        "prior_entropy": prior_entropy.mean(),
        "post_entropy": post_entropy.mean(),
        "gate_max_prob": max_prob.mean(),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
        "local_delta_rms": (prior * aux["local_delta_rms"]).sum(dim=-1).mean(),
        "local_scale_min": aux["local_scale_min"].amin(dim=-1).mean(),
        "local_scale_max": aux["local_scale_max"].amax(dim=-1).mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: LatentRegimeStructuredResidualStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    gate_balance_penalty: float,
) -> dict:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_joint_nll": 0.0,
        "val_local_var_penalty": 0.0,
        "val_gate_usage_penalty": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_prior_entropy": 0.0,
        "val_post_entropy": 0.0,
        "val_gate_max_prob": 0.0,
        "val_white_std_mean": 0.0,
        "val_z_std_mean": 0.0,
        "val_local_delta_rms": 0.0,
        "val_local_scale_min": 0.0,
        "val_local_scale_max": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        _, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            gate_balance_penalty=gate_balance_penalty,
        )
        batch_size = history_01.shape[0]
        totals["val_total_loss"] += metrics["total_loss"].item() * batch_size
        totals["val_joint_nll"] += metrics["joint_nll"].item() * batch_size
        totals["val_local_var_penalty"] += metrics["local_var_penalty"].item() * batch_size
        totals["val_gate_usage_penalty"] += metrics["gate_usage_penalty"].item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_time_eff_rank"] += metrics["time_eff_rank"].item() * batch_size
        totals["val_cell_eff_rank"] += metrics["cell_eff_rank"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_prior_entropy"] += metrics["prior_entropy"].item() * batch_size
        totals["val_post_entropy"] += metrics["post_entropy"].item() * batch_size
        totals["val_gate_max_prob"] += metrics["gate_max_prob"].item() * batch_size
        totals["val_white_std_mean"] += metrics["white_std_mean"].item() * batch_size
        totals["val_z_std_mean"] += metrics["z_std_mean"].item() * batch_size
        totals["val_local_delta_rms"] += metrics["local_delta_rms"].item() * batch_size
        totals["val_local_scale_min"] += metrics["local_scale_min"].item() * batch_size
        totals["val_local_scale_max"] += metrics["local_scale_max"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: LatentRegimeStructuredResidualStudentTModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
) -> dict:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_gate_entropy = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    all_sample_eff_rank = []

    for history_01, future_01 in val_loader:
        if total_count >= eval_limit:
            break
        if total_count + history_01.shape[0] > eval_limit:
            keep = eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(history_norm, n_samples=joint_val_samples)
        *_, gate_logits, _ = model.forward_from_history(history_01)
        mix_probs = F.softmax(gate_logits, dim=-1)
        gate_entropy = -(mix_probs * torch.log(mix_probs.clamp_min(1e-8))).sum(dim=-1).mean()

        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = ((samples < model.support_lo) | (samples > model.support_hi)).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        first_sample = samples[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        all_sample_eff_rank.append(eff_rank_np(corr))

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_gate_entropy += gate_entropy.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "joint_cov90": float("nan"),
            "joint_width90": float("nan"),
            "joint_mae": float("nan"),
            "joint_support_violation_rate": float("nan"),
            "joint_turb_calm_ratio": float("nan"),
            "joint_sample_eff_rank": float("nan"),
            "joint_gate_entropy": float("nan"),
        }

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    if calm_mask.any() and turb_mask.any():
        turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item()
    else:
        turb_calm_ratio = float("nan")

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
        "joint_gate_entropy": total_gate_entropy / total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="176a: latent regime structured residual model")
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
    parser.add_argument("--n_components", type=int, default=3)
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
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--gate_balance_penalty", type=float, default=20.0)
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
        n_components=args.n_components,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = LatentRegimeStructuredResidualStudentTModel(
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
    print("176a: Shared-mean latent regime structured residual model")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Components={args.n_components} | Time rank={args.time_rank} | Cell rank={args.cell_rank}")
    print(f"  Local penalty={args.local_var_penalty} | gate balance={args.gate_balance_penalty} | base nu={args.base_nu}")
    print("  Objective: exact latent-mixture likelihood + local variance shrinkage")
    print("  Shared mean path + regime-specific residual covariance/local scale/flow context")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
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
            "train_gate_usage_penalty": 0.0,
            "train_joint_mae": 0.0,
            "train_time_eff_rank": 0.0,
            "train_cell_eff_rank": 0.0,
            "train_scale_mean": 0.0,
            "train_prior_entropy": 0.0,
            "train_post_entropy": 0.0,
            "train_gate_max_prob": 0.0,
            "train_white_std_mean": 0.0,
            "train_z_std_mean": 0.0,
            "train_local_delta_rms": 0.0,
            "train_local_scale_min": 0.0,
            "train_local_scale_max": 0.0,
        }
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
                gate_balance_penalty=args.gate_balance_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key, value in metrics.items():
                totals[f"train_{key}"] += value.item()
            nb += 1

        scheduler.step()

        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
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
                        "type": "latent_regime_structured_residual_student_t_176a",
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
            f"H(pr/post)={val_metrics['val_prior_entropy']:.3f}/{val_metrics['val_post_entropy']:.3f}  "
            f"maxp={val_metrics['val_gate_max_prob']:.3f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "latent_regime_structured_residual_student_t_176a",
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
