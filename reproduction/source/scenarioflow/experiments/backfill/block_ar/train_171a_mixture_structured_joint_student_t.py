#!/usr/bin/env python
"""
171a: Regime-conditioned mixture of structured joint Student-t components

Keep 170d:
  - support-aware transformed-space density modeling
  - one-shot future block generation
  - proper joint likelihood
  - fixed scalar nu
  - structured separable time/cell covariance

Add:
  - K-component mixture
  - history-conditioned gating over components
  - exact mixture NLL
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


def build_multistep_windows(
    indices: np.ndarray,
    surf: torch.Tensor,
    hist_len: int,
    future_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.from_numpy(indices).long().to(surf.device)
    offsets_h = torch.arange(hist_len, device=surf.device).unsqueeze(0)
    offsets_f = torch.arange(future_len, device=surf.device).unsqueeze(0)
    hist_idx = idx.unsqueeze(1) + offsets_h
    fut_idx = idx.unsqueeze(1) + hist_len + offsets_f
    hist = surf[hist_idx]
    future = surf[fut_idx].reshape(len(indices), future_len, -1)
    return hist, future


class MixtureStructuredJointStudentTDecoder(nn.Module):
    """Factored temporal-spatial decoder with K structured Student-t components."""

    def __init__(
        self,
        n_frames: int = 30,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        n_components: int = 2,
        time_rank: int = 6,
        cell_rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.05,
        init_scale: float = 0.10,
        init_nu: float = 8.0,
        fixed_nu: float | None = 8.0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.n_components = n_components
        self.time_rank = time_rank
        self.cell_rank = cell_rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.nu_floor = nu_floor
        self.nu_max = nu_max
        self.fixed_nu = fixed_nu

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
        self.mean_head = nn.Linear(d_model, n_components)
        self.time_factor_head = nn.Linear(d_model, n_components * time_rank)
        self.time_diag_head = nn.Linear(d_model, n_components)
        self.cell_factor_head = nn.Linear(d_model, n_components * cell_rank)
        self.cell_diag_head = nn.Linear(d_model, n_components)
        self.scale_head = nn.Linear(d_model, n_components)
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_components),
        )
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, n_components)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_scale=init_scale, init_nu=init_nu)

    def _init_parameters(self, init_diag: float, init_scale: float, init_nu: float) -> None:
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
                        "gate_head",
                        "nu_head",
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
        for module in self.gate_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_head[-1].weight)

        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(
                self.nu_head.bias,
                inverse_softplus(max(init_nu - self.nu_floor, 1e-6)),
            )

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        mean_delta = self.mean_head(hidden).permute(0, 3, 1, 2)
        mu = prev_rep.unsqueeze(1) + mean_delta

        time_factor = self.time_factor_head(time_summary).view(batch, n_frames, n_components, self.time_rank).permute(0, 2, 1, 3)
        time_diag = F.softplus(self.time_diag_head(time_summary).permute(0, 2, 1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary).view(batch, n_cells, n_components, self.cell_rank).permute(0, 2, 1, 3)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).permute(0, 2, 1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled)) + self.scale_floor
        gate_logits = self.gate_head(pooled)

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(batch, n_components).to(hidden.dtype)

        return mu, time_factor, time_diag, cell_factor, cell_diag, scale, gate_logits, nu


class MixtureStructuredJointStudentTModel(nn.Module):
    """GRU history encoder + mixture of structured joint Student-t future blocks."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = MixtureStructuredJointStudentTDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.encoder(history_norm)

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return cov_t * scale.view(scale.shape[0], scale.shape[1], 1, 1), cov_c

    def component_log_probs(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_components, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        diff = target_u.unsqueeze(1) - mu
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white = torch.linalg.solve_triangular(chol_c, white_t.transpose(-1, -2), upper=False).transpose(-1, -2)
        mahal = white.pow(2).sum(dim=(2, 3))

        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet = n_cells * logdet_t + n_frames * logdet_c

        dim = n_frames * n_cells
        nu = nu.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + dim) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (dim * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + dim) * torch.log1p(mahal / nu)
        return log_norm + log_kernel, cov_t, cov_c

    def mixture_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        gate_logits: torch.Tensor,
        nu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        comp_log_prob, cov_t, cov_c = self.component_log_probs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            nu=nu,
        )
        log_mix = F.log_softmax(gate_logits, dim=-1)
        mixture_log_prob = torch.logsumexp(log_mix + comp_log_prob, dim=1)
        return -mixture_log_prob, cov_t, cov_c

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        mu, time_factor, time_diag, cell_factor, cell_diag, scale, gate_logits, nu = self.forward_from_history(history_01)
        batch, n_components, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        all_samples = []
        for k in range(n_components):
            z = torch.randn(batch, n_samples, n_frames, n_cells, device=mu.device, dtype=mu.dtype)
            temp = torch.einsum("bij,bsjk->bsik", chol_t[:, k], z)
            noise = torch.einsum("bstj,bcj->bstc", temp, chol_c[:, k])
            gamma = torch.distributions.Gamma(nu[:, k] / 2.0, nu[:, k] / 2.0)
            mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.device, mu.dtype).clamp_min(1e-6)
            t_scale = torch.rsqrt(mix).view(batch, n_samples, 1, 1)
            all_samples.append(mu[:, k].unsqueeze(1) + noise * t_scale)
        stacked = torch.stack(all_samples, dim=1)  # (B, K, S, T, C)

        probs = F.softmax(gate_logits, dim=-1)
        comp_idx = torch.distributions.Categorical(probs=probs).sample((n_samples,)).transpose(0, 1)  # (B, S)
        stacked_by_sample = stacked.permute(0, 2, 1, 3, 4)  # (B, S, K, T, C)
        gather_idx = comp_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch, n_samples, 1, n_frames, n_cells)
        chosen = torch.gather(stacked_by_sample, 2, gather_idx).squeeze(2)
        return chosen, probs

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


def eff_rank_np(corr: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    probs = eigvals / eigvals.sum()
    return float(np.exp(-(probs * np.log(probs)).sum()))


def joint_nll_loss(
    model: MixtureStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    mu, time_factor, time_diag, cell_factor, cell_diag, scale, gate_logits, nu = model.forward_from_history(history_01)
    nll, cov_t, cov_c = model.mixture_nll(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        gate_logits=gate_logits,
        nu=nu,
    )
    mix_probs = F.softmax(gate_logits, dim=-1)
    pred_u = (mix_probs[:, :, None, None] * mu).sum(dim=1)
    pred_01 = unconstrained_to_iv(pred_u, lo=model.support_lo, hi=model.support_hi)

    cov_t_mix = (mix_probs[:, :, None, None] * cov_t).sum(dim=1)
    cov_c_mix = (mix_probs[:, :, None, None] * cov_c).sum(dim=1)
    entropy = -(mix_probs * torch.log(mix_probs.clamp_min(1e-8))).sum(dim=-1)
    max_prob = mix_probs.max(dim=-1).values

    metrics = {
        "joint_nll": nll.mean(),
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "time_eff_rank": effective_rank(cov_t_mix).mean(),
        "cell_eff_rank": effective_rank(cov_c_mix).mean(),
        "scale_mean": (mix_probs * scale).sum(dim=-1).mean(),
        "gate_entropy": entropy.mean(),
        "gate_max_prob": max_prob.mean(),
        "nu_mean": (mix_probs * nu).sum(dim=-1).mean(),
    }
    return nll.mean(), metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: MixtureStructuredJointStudentTModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    totals = {
        "val_joint_nll": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_gate_entropy": 0.0,
        "val_gate_max_prob": 0.0,
        "val_nu_mean": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = joint_nll_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        totals["val_joint_nll"] += loss.item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_time_eff_rank"] += metrics["time_eff_rank"].item() * batch_size
        totals["val_cell_eff_rank"] += metrics["cell_eff_rank"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_gate_entropy"] += metrics["gate_entropy"].item() * batch_size
        totals["val_gate_max_prob"] += metrics["gate_max_prob"].item() * batch_size
        totals["val_nu_mean"] += metrics["nu_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: MixtureStructuredJointStudentTModel,
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
        _, _, _, _, _, _, gate_logits, _ = model.forward_from_history(history_01)
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
    parser = argparse.ArgumentParser(description="171a: mixture structured joint Student-t")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
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
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    model = MixtureStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 64}")
    print("171a: Mixture structured joint future Student-t")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Components={args.n_components} | Time rank={args.time_rank} | Cell rank={args.cell_rank}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: exact mixture joint Student-t NLL")
    print("  Component covariance: scale^2 * (Sigma_time kron Sigma_cell)")
    print(f"  Tail parameter: {'learned' if args.fixed_nu is None else f'fixed nu={args.fixed_nu}'}")

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
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        ep_mae = 0.0
        ep_time_rank = 0.0
        ep_cell_rank = 0.0
        ep_scale = 0.0
        ep_entropy = 0.0
        ep_max_prob = 0.0
        ep_nu = 0.0
        nbatches = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(model, history_01, future_01)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mae += metrics["joint_mae"].item()
            ep_time_rank += metrics["time_eff_rank"].item()
            ep_cell_rank += metrics["cell_eff_rank"].item()
            ep_scale += metrics["scale_mean"].item()
            ep_entropy += metrics["gate_entropy"].item()
            ep_max_prob += metrics["gate_max_prob"].item()
            ep_nu += metrics["nu_mean"].item()
            nbatches += 1

        scheduler.step()

        train_metrics = {
            "train_joint_nll": ep_loss / max(nbatches, 1),
            "train_joint_mae": ep_mae / max(nbatches, 1),
            "train_time_eff_rank": ep_time_rank / max(nbatches, 1),
            "train_cell_eff_rank": ep_cell_rank / max(nbatches, 1),
            "train_scale_mean": ep_scale / max(nbatches, 1),
            "train_gate_entropy": ep_entropy / max(nbatches, 1),
            "train_gate_max_prob": ep_max_prob / max(nbatches, 1),
            "train_nu_mean": ep_nu / max(nbatches, 1),
        }
        val_metrics = evaluate_teacher_forced(model, val_loader)
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_joint_nll"] < best_val
        if is_best:
            best_val = val_metrics["val_joint_nll"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_joint_nll": best_val,
                    "config": {
                        "type": "mixture_structured_joint_student_t_171a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "fixed_nu": args.fixed_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **joint_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_joint_nll']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"joint_cov90={joint_metrics['joint_cov90']:.4f}  "
            f"joint_tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"gate_H={val_metrics['val_gate_entropy']:.3f}  "
            f"gate_max={val_metrics['val_gate_max_prob']:.3f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_joint_nll": history[-1]["val_joint_nll"] if history else float("nan"),
        "config": {
            "type": "mixture_structured_joint_student_t_171a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "fixed_nu": args.fixed_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_val_joint_nll": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val joint NLL: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_gate_entropy={best_metrics['joint_gate_entropy']:.3f}, "
            f"joint_support_violation_rate={best_metrics['joint_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
