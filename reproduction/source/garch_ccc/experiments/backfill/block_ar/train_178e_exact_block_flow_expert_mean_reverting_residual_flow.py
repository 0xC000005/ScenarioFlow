#!/usr/bin/env python
"""
178e_v0: Exact blockwise whitened residual-flow experts on top of the 177a/178d backbone.

Motivation from the focused 178d analysis:
  - the exact latent router is alive
  - the covariance-residual templates are genuinely distinct
  - but even the best forced covariance template still misses the worst turbulent slices
  - the remaining bottleneck is therefore the uncertainty-expert family itself

Keep 177a/178d:
  - support-aware transformed-space modeling
  - explicit mean-reverting drift anchored to history
  - shared spatiotemporal covariance backbone
  - exact blockwise latent-mixture semantics

Change only the uncertainty experts:
  - whiten residuals with the shared backbone
  - keep small blockwise covariance residual templates
  - replace covariance-only experts with exact blockwise residual-flow experts
"""

from __future__ import annotations

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

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170b_whitened_flow import ConditionalResidualFlow
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_172a_residual_flow_structured_joint_student_t import (
    ResidualFlowStructuredJointDecoder,
    ResidualFlowStructuredJointStudentTModel,
)
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import (
    aggregate_slope_ratio,
)


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


class ExactBlockFlowExpertMeanRevertingResidualFlowDecoder(ResidualFlowStructuredJointDecoder):
    """177a drift path plus exact blockwise covariance+flow expert routing context."""

    def __init__(
        self,
        *args,
        n_blocks: int = 5,
        n_templates: int = 3,
        local_delta_clip: float = 0.35,
        template_diag_clip: float = 0.30,
        template_offdiag_clip: float = 0.18,
        drift_strength_max: float = 0.75,
        equilibrium_offset_clip: float = 0.20,
        init_drift_strength: float = 0.20,
        block_context_dim: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.n_frames % n_blocks != 0:
            raise ValueError(f"Expected n_frames divisible by n_blocks, got {self.n_frames} and {n_blocks}")
        self.n_blocks = n_blocks
        self.block_len = self.n_frames // n_blocks
        self.n_templates = n_templates
        self.local_delta_clip = local_delta_clip
        self.template_diag_clip = template_diag_clip
        self.template_offdiag_clip = template_offdiag_clip
        self.drift_strength_max = drift_strength_max
        self.equilibrium_offset_clip = equilibrium_offset_clip

        d_model = self.mean_head.in_features
        self.equilibrium_offset_head = nn.Linear(d_model, 1)
        self.drift_strength_head = nn.Linear(d_model, 1)
        self.local_logvar_head = nn.Linear(d_model, 1)
        self.static_local_logvar = nn.Parameter(torch.zeros(self.n_frames, self.n_cells))
        self.block_cov_diag_bank = nn.Parameter(torch.zeros(n_blocks, n_templates, self.n_cells))
        self.block_cov_lower_bank = nn.Parameter(torch.zeros(n_blocks, n_templates, self.n_cells, self.n_cells))
        self.block_gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_templates),
        )
        self.block_context_head = nn.Sequential(
            nn.Linear(d_model + kwargs.get("cond_dim", 128), block_context_dim),
            nn.SiLU(),
            nn.Linear(block_context_dim, block_context_dim),
        )
        tril_mask = torch.tril(torch.ones(self.n_cells, self.n_cells), diagonal=-1)
        self.register_buffer("strict_tril_mask", tril_mask, persistent=False)

        nn.init.normal_(self.block_cov_diag_bank, mean=0.0, std=1e-3)
        nn.init.normal_(self.block_cov_lower_bank, mean=0.0, std=1e-3)
        nn.init.zeros_(self.equilibrium_offset_head.weight)
        nn.init.zeros_(self.equilibrium_offset_head.bias)
        nn.init.zeros_(self.drift_strength_head.weight)
        init_frac = min(max(init_drift_strength / max(drift_strength_max, 1e-6), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.drift_strength_head.bias, logit(init_frac))
        nn.init.zeros_(self.local_logvar_head.weight)
        nn.init.zeros_(self.local_logvar_head.bias)
        for module in self.block_gate_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.block_gate_head[-1].weight)
        for module in self.block_context_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def build_shared_local_delta(self, base_local_delta: torch.Tensor) -> torch.Tensor:
        local_delta_raw = base_local_delta + self.static_local_logvar.unsqueeze(0)
        local_delta = torch.tanh(local_delta_raw) * self.local_delta_clip
        return local_delta - local_delta.mean(dim=(1, 2), keepdim=True)

    def build_template_bank(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diag = torch.exp(torch.tanh(self.block_cov_diag_bank) * self.template_diag_clip)
        lower = torch.tanh(self.block_cov_lower_bank) * self.template_offdiag_clip
        lower = lower * self.strict_tril_mask
        factors = lower + torch.diag_embed(diag)
        logdet_cov = 2.0 * self.block_len * torch.log(diag).sum(dim=-1)
        offdiag_rms = lower.pow(2).mean(dim=(-1, -2)).sqrt()
        return factors, logdet_cov, offdiag_rms

    def forward(self, cond: torch.Tensor, prev_u: torch.Tensor, hist_mean_u: torch.Tensor):
        batch = prev_u.shape[0]
        n_frames, n_cells = self.n_frames, self.n_cells

        prev_rep = prev_u.unsqueeze(1).expand(batch, n_frames, n_cells)
        hist_mean_rep = hist_mean_u.unsqueeze(1).expand(batch, n_frames, n_cells)
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

        base_delta = self.mean_head(hidden).squeeze(-1)
        equilibrium_offset = torch.tanh(self.equilibrium_offset_head(hidden).squeeze(-1)) * self.equilibrium_offset_clip
        equilibrium = hist_mean_rep + equilibrium_offset
        drift_strength = torch.sigmoid(self.drift_strength_head(hidden).squeeze(-1)) * self.drift_strength_max
        drift_term = drift_strength * (equilibrium - prev_rep)
        mu = prev_rep + base_delta + drift_term

        time_factor = self.time_factor_head(time_summary)
        time_diag = F.softplus(self.time_diag_head(time_summary).squeeze(-1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor

        base_local_delta = torch.tanh(self.local_logvar_head(hidden).squeeze(-1)) * self.local_delta_clip
        block_summary = time_summary.view(batch, self.n_blocks, self.block_len, -1).mean(dim=2)
        block_logits = self.block_gate_head(block_summary)
        cond_rep = cond.unsqueeze(1).expand(-1, self.n_blocks, -1)
        block_context = self.block_context_head(torch.cat([block_summary, cond_rep], dim=-1))
        return (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            base_local_delta,
            block_logits,
            block_context,
        )


class ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel(
    ResidualFlowStructuredJointStudentTModel
):
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
        block_context_dim = base_decoder_config.pop("block_context_dim")
        n_blocks = base_decoder_config.pop("n_blocks")
        n_templates = base_decoder_config.pop("n_templates")
        for key in (
            "local_delta_clip",
            "template_diag_clip",
            "template_offdiag_clip",
            "drift_strength_max",
            "equilibrium_offset_clip",
            "init_drift_strength",
        ):
            base_decoder_config.pop(key, None)
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
        if hasattr(self, "flow"):
            del self.flow
        decoder_kwargs = dict(decoder_config)
        self.decoder = ExactBlockFlowExpertMeanRevertingResidualFlowDecoder(**decoder_kwargs)
        self.decoder_config = decoder_config
        self.block_context_dim = block_context_dim
        self.n_blocks = n_blocks
        self.n_templates = n_templates
        self.block_dim = decoder_config["n_cells"] * (decoder_config["n_frames"] // n_blocks)
        self.flow_experts = nn.ModuleList(
            [
                ConditionalResidualFlow(
                    dim=self.block_dim,
                    context_dim=block_context_dim,
                    hidden_dim=flow_config["hidden_dim"],
                    n_layers=flow_config["n_layers"],
                    scale_clip=flow_config["scale_clip"],
                )
                for _ in range(n_templates)
            ]
        )
        self.flow_config = dict(flow_config)
        self.flow_config["dim"] = self.block_dim
        self.flow_config["context_dim"] = block_context_dim

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        if "decoder.block_cov_diag_bank" in state and state["decoder.block_cov_diag_bank"].shape == (
            self.n_templates,
            self.n_blocks,
            self.decoder.n_cells,
        ):
            state["decoder.block_cov_diag_bank"] = state["decoder.block_cov_diag_bank"].permute(1, 0, 2).contiguous()
        if "decoder.block_cov_lower_bank" in state and state["decoder.block_cov_lower_bank"].shape == (
            self.n_templates,
            self.n_blocks,
            self.decoder.n_cells,
            self.decoder.n_cells,
        ):
            state["decoder.block_cov_lower_bank"] = state["decoder.block_cov_lower_bank"].permute(1, 0, 2, 3).contiguous()

        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Warm start loaded from {ckpt_path}")
        print(
            f"  Warm start missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | "
            f"shape-skipped: {len(skipped)}"
        )
        if "flow.layers.0.net.0.weight" in state:
            shared_flow_state = {k[len("flow.") :]: v for k, v in state.items() if k.startswith("flow.")}
            loaded_any = False
            for expert in self.flow_experts:
                expert_state = expert.state_dict()
                filtered_flow = {
                    k: v
                    for k, v in shared_flow_state.items()
                    if k in expert_state and expert_state[k].shape == v.shape
                }
                if filtered_flow:
                    expert.load_state_dict(filtered_flow, strict=False)
                    loaded_any = True
            if loaded_any:
                print("  Partially initialized block flow experts from shared 178d flow weights")
            else:
                print("  Shared 178d flow weights are dimension-mismatched; expert flows left freshly initialized")

    def forward_from_history(self, history_01: torch.Tensor):
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        hist_mean_01 = history_01.mean(dim=1).reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        hist_mean_u = iv_to_unconstrained(hist_mean_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        return self.decoder(cond, prev_u, hist_mean_u)

    def log_prob_future(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        base_local_delta: torch.Tensor,
        block_logits: torch.Tensor,
        block_context: torch.Tensor,
    ):
        batch, n_frames, n_cells = target_u.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        block_probs = F.softmax(block_logits, dim=-1)
        log_prior_blocks = F.log_softmax(block_logits, dim=-1)

        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        shared_local_scale = torch.exp(0.5 * shared_local_delta)
        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c
        logdet_local = 2.0 * torch.log(shared_local_scale).sum(dim=(1, 2))

        diff = (target_u - mu) / shared_local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white_shared = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_blocks = white_shared.view(batch, self.decoder.n_blocks, self.decoder.block_len, n_cells)

        factors_bank, logdet_template_cov, offdiag_rms_bank = self.decoder.build_template_bank()
        total = target_u.new_zeros(batch)
        expert_logdet_map = []
        white_std_map = []
        z_std_map = []
        map_assign = block_logits.argmax(dim=-1)
        map_diag_mean = []
        map_offdiag_rms = []

        for b in range(self.decoder.n_blocks):
            obs = white_blocks[:, b]
            ctx = block_context[:, b]
            rhs = obs.transpose(1, 2)
            comp_terms = []
            for k, expert in enumerate(self.flow_experts):
                factor = factors_bank[b, k].unsqueeze(0).expand(batch, -1, -1)
                base_block = torch.linalg.solve_triangular(factor, rhs, upper=False).transpose(1, 2)
                base_flat = base_block.reshape(batch, self.block_dim)
                z, flow_logdet = expert(base_flat, ctx)
                base_logprob = self._base_logprob(z)
                comp_terms.append(
                    log_prior_blocks[:, b, k]
                    + base_logprob
                    + flow_logdet
                    - 0.5 * logdet_template_cov[b, k]
                )
            total = total + torch.logsumexp(torch.stack(comp_terms, dim=-1), dim=-1)

            with torch.no_grad():
                map_k = map_assign[:, b]
                white_std_vals = []
                z_std_vals = []
                logdet_vals = []
                diag_vals = []
                offdiag_vals = []
                for k, expert in enumerate(self.flow_experts):
                    mask = map_k == k
                    if not mask.any():
                        continue
                    factor = factors_bank[b, k].unsqueeze(0).expand(mask.sum(), -1, -1)
                    base_block = torch.linalg.solve_triangular(factor, rhs[mask], upper=False).transpose(1, 2)
                    base_flat = base_block.reshape(mask.sum(), self.block_dim)
                    z, flow_logdet = expert(base_flat, ctx[mask])
                    white_std_vals.append(base_flat.std(dim=-1))
                    z_std_vals.append(z.std(dim=-1))
                    logdet_vals.append(flow_logdet)
                    diag_vals.append(torch.diagonal(factors_bank[b, k]).mean().expand(mask.sum()))
                    offdiag_vals.append(offdiag_rms_bank[b, k].expand(mask.sum()))
                if white_std_vals:
                    white_std_map.append(torch.cat(white_std_vals).mean())
                    z_std_map.append(torch.cat(z_std_vals).mean())
                    expert_logdet_map.append(torch.cat(logdet_vals).mean())
                    map_diag_mean.append(torch.cat(diag_vals).mean())
                    map_offdiag_rms.append(torch.cat(offdiag_vals).mean())

        logprob = total - 0.5 * (logdet_cov + logdet_local)
        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "white_std": torch.stack(white_std_map).mean().expand(batch) if white_std_map else white_shared.std(dim=(1, 2)),
            "z_std": torch.stack(z_std_map).mean().expand(batch) if z_std_map else white_shared.std(dim=(1, 2)),
            "flow_logdet": torch.stack(expert_logdet_map).mean().expand(batch) if expert_logdet_map else target_u.new_zeros(batch),
            "local_delta_rms": shared_local_delta.pow(2).mean(dim=(1, 2)).sqrt(),
            "local_scale_min": shared_local_scale.amin(dim=(1, 2)),
            "local_scale_max": shared_local_scale.amax(dim=(1, 2)),
            "template_diag_mean": torch.stack(map_diag_mean).mean().expand(batch) if map_diag_mean else target_u.new_ones(batch),
            "template_offdiag_rms": torch.stack(map_offdiag_rms).mean().expand(batch) if map_offdiag_rms else target_u.new_zeros(batch),
            "block_gate_entropy": (-(block_probs * torch.log(block_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "block_gate_max": block_probs.max(dim=-1).values.mean(dim=-1),
            "block_usage": block_probs.mean(dim=(0, 1)),
        }
        return logprob, aux

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int):
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            base_local_delta,
            block_logits,
            block_context,
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        block_probs = F.softmax(block_logits, dim=-1)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)

        sampled_blocks = []
        for b in range(self.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)

        base = torch.distributions.StudentT(df=self.base_nu)
        factors_bank, _logdet_template_cov, _offdiag_rms = self.decoder.build_template_bank()
        white_blocks = []
        for b in range(self.decoder.n_blocks):
            ctx_b = block_context[:, b].unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
            assign_b = sampled_assign[:, :, b].reshape(batch * n_samples)
            z = base.sample((batch * n_samples, self.block_dim)).to(device=mu.device, dtype=mu.dtype)
            base_flat = torch.zeros_like(z)
            for k, expert in enumerate(self.flow_experts):
                mask = assign_b == k
                if mask.any():
                    xk, _ = expert.inverse(z[mask], ctx_b[mask])
                    base_flat[mask] = xk
            base_block = base_flat.view(batch * n_samples, self.decoder.block_len, n_cells)
            factors = factors_bank[b][assign_b]
            obs = torch.matmul(factors, base_block.transpose(1, 2)).transpose(1, 2)
            white_blocks.append(obs.view(batch, n_samples, self.decoder.block_len, n_cells))
        routed_white = torch.cat(white_blocks, dim=2)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        return mu.unsqueeze(1) + noise * local_scale


def joint_nll_loss(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    gate_smooth_penalty: float,
):
    target_u = iv_to_unconstrained(future_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        base_local_delta,
        block_logits,
        block_context,
    ) = model.forward_from_history(history_01)
    logprob, aux = model.log_prob_future(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        base_local_delta=base_local_delta,
        block_logits=block_logits,
        block_context=block_context,
    )

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    local_pen = base_local_delta.pow(2).mean()
    template_l2 = model.decoder.block_cov_diag_bank.pow(2).mean() + model.decoder.block_cov_lower_bank.pow(2).mean()
    static_l2 = model.decoder.static_local_logvar.pow(2).mean()
    block_probs = F.softmax(block_logits, dim=-1)
    usage = aux["block_usage"]
    target_usage = torch.full_like(usage, 1.0 / usage.numel())
    usage_penalty = (usage - target_usage).pow(2).mean()
    smooth_pen = (block_probs[:, 1:] - block_probs[:, :-1]).pow(2).mean() if block_probs.shape[1] > 1 else block_probs.new_tensor(0.0)
    nll = (-logprob).mean()
    loss = (
        nll
        + local_var_penalty * (local_pen + 0.25 * static_l2)
        + template_penalty * template_l2
        + gate_balance_penalty * usage_penalty
        + gate_smooth_penalty * smooth_pen
    )

    det_mr_ratio = aggregate_slope_ratio(
        history_01[:, -1].reshape(history_01.shape[0], -1),
        future_01[:, 0].reshape(future_01.shape[0], -1),
        pred_01[:, 0],
    )
    metrics = {
        "total_loss": loss,
        "joint_nll": nll,
        "local_var_penalty": local_pen,
        "template_l2": template_l2,
        "gate_usage_penalty": usage_penalty,
        "smooth_penalty": smooth_pen,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "joint_det_mr_ratio": pred_01.new_tensor(det_mr_ratio),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": aux["flow_logdet"].mean(),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
        "local_delta_rms": aux["local_delta_rms"].mean(),
        "local_scale_min": aux["local_scale_min"].mean(),
        "local_scale_max": aux["local_scale_max"].mean(),
        "template_diag_mean": aux["template_diag_mean"].mean(),
        "template_offdiag_rms": aux["template_offdiag_rms"].mean(),
        "block_gate_entropy": aux["block_gate_entropy"].mean(),
        "block_gate_max": aux["block_gate_max"].mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    gate_smooth_penalty: float,
):
    model.eval()
    totals = {f"val_{k}": 0.0 for k in [
        "total_loss","joint_nll","local_var_penalty","template_l2","gate_usage_penalty","smooth_penalty",
        "joint_mae","joint_det_mr_ratio","time_eff_rank","cell_eff_rank","scale_mean",
        "flow_logdet_mean","white_std_mean","z_std_mean","local_delta_rms","local_scale_min",
        "local_scale_max","template_diag_mean","template_offdiag_rms","block_gate_entropy","block_gate_max"
    ]}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            template_penalty=template_penalty,
            gate_balance_penalty=gate_balance_penalty,
            gate_smooth_penalty=gate_smooth_penalty,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
):
    model.eval()
    total_cov = total_width = total_mae = total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    all_sample_eff_rank = []
    prev_chunks = []
    gt_next_chunks = []
    det_next_chunks = []
    sample_next_chunks = []

    for history_01, future_01 in val_loader:
        if total_count >= eval_limit:
            break
        if total_count + history_01.shape[0] > eval_limit:
            keep = eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples_u = model.sample_batched(history_norm, n_samples=joint_val_samples)
        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples_u.quantile(0.05, dim=1)
        hi = samples_u.quantile(0.95, dim=1)
        median = samples_u.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = ((samples_u < model.support_lo) | (samples_u > model.support_hi)).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        first_sample = samples_u[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 1e-10)
        probs = eigvals / eigvals.sum()
        all_sample_eff_rank.append(float(np.exp(-(probs * np.log(probs)).sum())))

        det_outputs = model.forward_from_history(denormalize_iv(history_norm))
        det_next = unconstrained_to_iv(det_outputs[0][:, 0], lo=model.support_lo, hi=model.support_hi).reshape(history_01.shape[0], 5, 5)
        prev_chunks.append(history_01[:, -1].detach().cpu())
        gt_next_chunks.append(future_01[:, 0].detach().cpu())
        det_next_chunks.append(det_next.detach().cpu())
        sample_next_chunks.append(samples_u[:, :, 0].mean(dim=1).detach().cpu())

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]
        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {k: float("nan") for k in [
            "joint_cov90","joint_width90","joint_mae","joint_support_violation_rate",
            "joint_turb_calm_ratio","joint_sample_eff_rank","joint_det_mr_ratio","joint_sample_mr_ratio"
        ]}

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item() if calm_mask.any() and turb_mask.any() else float("nan")

    prev = torch.cat(prev_chunks, dim=0)
    gt_next = torch.cat(gt_next_chunks, dim=0)
    det_next = torch.cat(det_next_chunks, dim=0)
    sample_next = torch.cat(sample_next_chunks, dim=0)
    det_mr_ratio = aggregate_slope_ratio(prev, gt_next, det_next)
    sample_mr_ratio = aggregate_slope_ratio(prev, gt_next, sample_next)

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
        "joint_det_mr_ratio": det_mr_ratio,
        "joint_sample_mr_ratio": sample_mr_ratio,
    }


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def main():
    parser = argparse.ArgumentParser(description="178e: exact block flow-expert mean-reverting residual flow")
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
    parser.add_argument("--block_context_dim", type=int, default=256)
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
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--template_penalty", type=float, default=0.10)
    parser.add_argument("--gate_balance_penalty", type=float, default=0.25)
    parser.add_argument("--gate_smooth_penalty", type=float, default=0.5)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
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

    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)
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
        flow_context_dim=args.block_context_dim,
        block_context_dim=args.block_context_dim,
        local_delta_clip=args.local_delta_clip,
        n_blocks=args.n_blocks,
        n_templates=args.n_templates,
        template_diag_clip=args.template_diag_clip,
        template_offdiag_clip=args.template_offdiag_clip,
        drift_strength_max=args.drift_strength_max,
        equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    block_dim = future_len // args.n_blocks * 25
    flow_config = dict(
        dim=block_dim,
        context_dim=args.block_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow_experts.parameters())
    print(f"\n{'=' * 64}")
    print("178e_v0: exact block flow-expert mean-reverting residual flow")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow-expert params: {n_flow:,}")
    print(f"  Total params: {n_enc + n_dec + n_flow:,}")
    if args.warm_start_path:
        print(f"  Warm start: {args.warm_start_path}")
        print(f"  Freeze backbone epochs: {args.freeze_backbone_epochs}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow_experts.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
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
        freeze_backbone = args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        set_requires_grad(model.encoder, not freeze_backbone)
        set_requires_grad(model.decoder, not freeze_backbone)
        set_requires_grad(model.flow_experts, True)
        totals = {f"train_{k}": 0.0 for k in [
            "total_loss","joint_nll","local_var_penalty","template_l2","gate_usage_penalty","smooth_penalty",
            "joint_mae","joint_det_mr_ratio","time_eff_rank","cell_eff_rank","scale_mean",
            "flow_logdet_mean","white_std_mean","z_std_mean","local_delta_rms","local_scale_min",
            "local_scale_max","template_diag_mean","template_offdiag_rms","block_gate_entropy","block_gate_max"
        ]}
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
                gate_smooth_penalty=args.gate_smooth_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += metrics[key.replace("train_", "")].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
            template_penalty=args.template_penalty,
            gate_balance_penalty=args.gate_balance_penalty,
            gate_smooth_penalty=args.gate_smooth_penalty,
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
                        "type": "exact_block_flow_expert_mean_reverting_residual_flow_structured_joint_student_t_178e",
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
                        "gate_smooth_penalty": args.gate_smooth_penalty,
                        "warm_start_path": args.warm_start_path,
                        "freeze_backbone_epochs": args.freeze_backbone_epochs,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))
        print(
            f"Ep {epoch:3d}  train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"mr_det={joint_metrics['joint_det_mr_ratio']:.3f}  mr_samp={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"gateH={val_metrics['val_block_gate_entropy']:.3f}  gateMax={val_metrics['val_block_gate_max']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "exact_block_flow_expert_mean_reverting_residual_flow_structured_joint_student_t_178e",
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
            "gate_smooth_penalty": args.gate_smooth_penalty,
            "warm_start_path": args.warm_start_path,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
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
            f"joint_det_mr_ratio={best_metrics['joint_det_mr_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
