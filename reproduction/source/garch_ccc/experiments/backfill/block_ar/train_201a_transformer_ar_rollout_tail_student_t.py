#!/usr/bin/env python
"""
201a: Transformer AR innovation model with rollout-consistent training and tail-aware scoring.

Narrow branch after the AR sparse-case preparation phase:
  - keep AR recursion
  - keep explicit multivariate Student-t innovation modeling
  - replace the GRU history encoder with a temporal Transformer encoder
  - train with tail-aware teacher-forced NLL
  - add short reparameterized rollout-consistency loss on self-generated histories

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_201a_transformer_ar_rollout_tail_student_t.py \
        --output_dir models/backfill/transformer_ar_rollout_tail_student_t_201a --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    make_serializable,
    normalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    SpatialShapeScaleStudentTDecoder,
    build_multistep_windows,
)


def build_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def reshape_history(history_01: torch.Tensor) -> torch.Tensor:
    if history_01.dim() == 4:
        return history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
    return history_01


def cov_to_corr(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps))
    denom = std.unsqueeze(-1) * std.unsqueeze(-2)
    corr = cov / denom.clamp_min(eps)
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye) + eye
    return corr.clamp(-1.0, 1.0)


def corr_mse_per_example(corr_a: torch.Tensor, corr_b: torch.Tensor) -> torch.Tensor:
    n = corr_a.shape[-1]
    mask = 1.0 - torch.eye(n, device=corr_a.device, dtype=corr_a.dtype)
    diff = (corr_a - corr_b) * mask.unsqueeze(0)
    denom = mask.sum().clamp_min(1.0)
    return diff.pow(2).sum(dim=(-1, -2)) / denom


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[mask].mean())


def corr_from_cov(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


class TemporalTransformerHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        bottleneck_dim: int = 128,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("positional_encoding", build_sinusoidal_encoding(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.pool_proj = nn.Linear(d_model, 1)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, bottleneck_dim),
        )

    def forward(self, history_norm_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = history_norm_flat.shape[1]
        x = self.input_proj(history_norm_flat) + self.positional_encoding[:t].unsqueeze(0)
        x = self.dropout(x)
        h = self.encoder(x)
        attn_logits = self.pool_proj(h).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = (attn.unsqueeze(-1) * h).sum(dim=1)
        cond = self.bottleneck(pooled)
        return cond, attn


class TransformerRolloutTailStudentTARModel(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = TemporalTransformerHistoryEncoder(**encoder_config)
        self.decoder = SpatialShapeScaleStudentTDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(
        self, history_01: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        history_flat = reshape_history(history_01)
        history_norm = normalize_iv(history_flat)
        cond, attn = self.encoder(history_norm)
        if return_attention:
            return cond, attn
        return cond

    def forward_from_history(
        self, history_01: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        cond, attn = self.encode(history_01, return_attention=True)
        history_flat = reshape_history(history_01)
        prev_01 = history_flat[:, -1]
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        outputs = self.decoder(cond, prev_u)
        if return_attention:
            return (*outputs, attn)
        return outputs

    def normalized_components(
        self, factor: torch.Tensor, diag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm, avg_var

    def covariance(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def student_t_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        d = target_u.shape[-1]
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)

        diff = (target_u - mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        nu = nu.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return -(log_norm + log_kernel)

    def reparameterized_next_u(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        eps_lowrank = torch.randn(batch_size, rank, device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch_size, n_cells, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,br->bc", factor_norm, eps_lowrank)
        diag_noise = diag_norm * eps_diag
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.rsample().clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(-1)
        return mu + total_noise * t_scale

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        eps_lowrank = torch.randn(batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,bnr->bnc", factor_norm, eps_lowrank)
        diag_noise = diag_norm.unsqueeze(1) * eps_diag
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(1).unsqueeze(-1)
        return mu.unsqueeze(1) + total_noise * t_scale

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        samples_u = self.sample_next_u(history_01, n_samples=n_samples)
        return unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)

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
        history_flat = reshape_history(history_01)
        batch_size, hist_len = history_flat.shape[:2]

        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist_k = history_flat.unsqueeze(1).expand(batch_size, k, hist_len, 25)
            hist_k = hist_k.reshape(batch_size * k, hist_len, 25).clone()
            frames = []
            for _ in range(n_steps):
                next_iv = self.sample_next_iv(hist_k, n_samples=1).squeeze(1)
                frames.append(next_iv.view(batch_size, k, 5, 5))
                hist_k = torch.cat([hist_k[:, 1:], next_iv.unsqueeze(1)], dim=1)
            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def compute_step_tail_weight(
    prev_01: torch.Tensor,
    target_t: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
    tail_weight: float,
) -> torch.Tensor:
    delta_abs = (target_t - prev_01).abs()
    q95_frac = (delta_abs >= q95_threshold).float().mean(dim=-1)
    mean_excess = F.relu(delta_abs.mean(dim=-1) / max(q95_threshold, 1e-6) - 1.0)
    max_excess = F.relu(delta_abs.max(dim=-1).values / max(q99_threshold, 1e-6) - 1.0)
    severity = q95_frac + 0.5 * mean_excess + 0.5 * max_excess
    return 1.0 + tail_weight * severity


def maybe_load_decoder_warm_start(model: TransformerRolloutTailStudentTARModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        print("No decoder warm start")
        return
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Decoder warm start missing: {checkpoint_path}")
        return
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    decoder_state = {
        k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")
    }
    missing, unexpected = model.decoder.load_state_dict(decoder_state, strict=False)
    print(
        f"Loaded decoder warm start from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def teacher_forced_multistep_objective(
    model: TransformerRolloutTailStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_flat = reshape_history(history_01)
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    future_len = future_flat.shape[1]
    context = history_flat

    total_loss = 0.0
    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_scale = 0.0
    total_shape_var = 0.0
    total_nu = 0.0
    total_attn_top1 = 0.0
    total_tail_weight = 0.0

    for step in range(future_len):
        mu, factor, diag, scale, nu, attn = model.forward_from_history(context, return_attention=True)
        cov = model.covariance(factor, diag, scale)

        prev_01 = context[:, -1]
        target_t = future_flat[:, step]
        target_u = iv_to_unconstrained(target_t, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
        nll_t = model.student_t_nll(target_u, mu, factor, diag, scale, nu)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
        tail_w = compute_step_tail_weight(
            prev_01=prev_01,
            target_t=target_t,
            q95_threshold=objective_config["q95_threshold"],
            q99_threshold=objective_config["q99_threshold"],
            tail_weight=objective_config["tail_weight"],
        )
        weighted_nll = (tail_w * nll_t).mean()
        weighted_mae = (tail_w * (mu_iv - target_t).abs().mean(dim=-1)).mean()

        factor_norm, diag_norm, avg_var = model.normalized_components(factor, diag)
        total_loss = total_loss + weighted_nll + objective_config["tail_mae_weight"] * weighted_mae
        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(cov).mean()
        total_scale = total_scale + scale.mean()
        total_shape_var = total_shape_var + avg_var.mean()
        total_nu = total_nu + nu.mean()
        total_attn_top1 = total_attn_top1 + attn.max(dim=1).values.mean()
        total_tail_weight = total_tail_weight + tail_w.mean()

        context = torch.cat([context[:, 1:], target_t.unsqueeze(1)], dim=1)

    scale_fac = 1.0 / future_len
    metrics = {
        "teacher_total_loss": total_loss * scale_fac,
        "multistep_nll": total_nll * scale_fac,
        "multistep_mae": total_mae * scale_fac,
        "pred_eff_rank": total_rank * scale_fac,
        "scale_mean": total_scale * scale_fac,
        "shape_avg_var": total_shape_var * scale_fac,
        "nu_mean": total_nu * scale_fac,
        "attention_top1": total_attn_top1 * scale_fac,
        "tail_weight_mean": total_tail_weight * scale_fac,
    }
    return total_loss * scale_fac, metrics


def rollout_consistency_objective(
    model: TransformerRolloutTailStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_flat = reshape_history(history_01)
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    rollout_steps = int(min(objective_config["rollout_steps"], future_flat.shape[1]))

    tf_context = history_flat
    ro_context = history_flat
    total_loss = 0.0
    total_mean_cons = 0.0
    total_var_cons = 0.0
    total_corr_cons = 0.0
    total_tail_weight = 0.0

    for step in range(rollout_steps):
        with torch.no_grad():
            mu_tf, factor_tf, diag_tf, scale_tf, _nu_tf = model.forward_from_history(tf_context)
            cov_tf = model.covariance(factor_tf, diag_tf, scale_tf)
            corr_tf = cov_to_corr(cov_tf)
            logvar_tf = torch.log(torch.diagonal(cov_tf, dim1=-2, dim2=-1).clamp_min(1e-6))

        mu_ro, factor_ro, diag_ro, scale_ro, nu_ro = model.forward_from_history(ro_context)
        cov_ro = model.covariance(factor_ro, diag_ro, scale_ro)
        corr_ro = cov_to_corr(cov_ro)
        logvar_ro = torch.log(torch.diagonal(cov_ro, dim1=-2, dim2=-1).clamp_min(1e-6))

        target_t = future_flat[:, step]
        prev_01 = tf_context[:, -1]
        tail_w = compute_step_tail_weight(
            prev_01=prev_01,
            target_t=target_t,
            q95_threshold=objective_config["q95_threshold"],
            q99_threshold=objective_config["q99_threshold"],
            tail_weight=objective_config["tail_weight"],
        )

        mean_cons = F.smooth_l1_loss(mu_ro, mu_tf, reduction="none").mean(dim=-1)
        var_cons = F.smooth_l1_loss(logvar_ro, logvar_tf, reduction="none").mean(dim=-1)
        corr_cons = corr_mse_per_example(corr_ro, corr_tf)
        step_loss = (
            mean_cons
            + objective_config["rollout_var_weight"] * var_cons
            + objective_config["rollout_corr_weight"] * corr_cons
        )
        total_loss = total_loss + (tail_w * step_loss).mean()
        total_mean_cons = total_mean_cons + mean_cons.mean()
        total_var_cons = total_var_cons + var_cons.mean()
        total_corr_cons = total_corr_cons + corr_cons.mean()
        total_tail_weight = total_tail_weight + tail_w.mean()

        sample_u = model.reparameterized_next_u(mu_ro, factor_ro, diag_ro, scale_ro, nu_ro)
        sample_iv = unconstrained_to_iv(sample_u, lo=model.support_lo, hi=model.support_hi)

        tf_context = torch.cat([tf_context[:, 1:], target_t.unsqueeze(1)], dim=1)
        ro_context = torch.cat([ro_context[:, 1:], sample_iv.unsqueeze(1)], dim=1)

    scale_fac = 1.0 / max(rollout_steps, 1)
    metrics = {
        "rollout_consistency_loss": total_loss * scale_fac,
        "rollout_mean_consistency": total_mean_cons * scale_fac,
        "rollout_var_consistency": total_var_cons * scale_fac,
        "rollout_corr_consistency": total_corr_cons * scale_fac,
        "rollout_tail_weight_mean": total_tail_weight * scale_fac,
    }
    return total_loss * scale_fac, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: TransformerRolloutTailStudentTARModel,
    val_loader: DataLoader,
    objective_config: dict[str, float],
) -> dict[str, float]:
    model.eval()
    total = {
        "val_teacher_total_loss": 0.0,
        "val_multistep_nll": 0.0,
        "val_multistep_mae": 0.0,
        "val_pred_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_shape_avg_var": 0.0,
        "val_nu_mean": 0.0,
        "val_attention_top1": 0.0,
        "val_tail_weight_mean": 0.0,
    }
    total_count = 0
    for history_01, future_01 in val_loader:
        loss, metrics = teacher_forced_multistep_objective(model, history_01, future_01, objective_config)
        bs = history_01.shape[0]
        total["val_teacher_total_loss"] += loss.item() * bs
        total["val_multistep_nll"] += metrics["multistep_nll"].item() * bs
        total["val_multistep_mae"] += metrics["multistep_mae"].item() * bs
        total["val_pred_eff_rank"] += metrics["pred_eff_rank"].item() * bs
        total["val_scale_mean"] += metrics["scale_mean"].item() * bs
        total["val_shape_avg_var"] += metrics["shape_avg_var"].item() * bs
        total["val_nu_mean"] += metrics["nu_mean"].item() * bs
        total["val_attention_top1"] += metrics["attention_top1"].item() * bs
        total["val_tail_weight_mean"] += metrics["tail_weight_mean"].item() * bs
        total_count += bs
    return {k: v / max(total_count, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate_rollout_subset(
    model: TransformerRolloutTailStudentTARModel,
    val_loader: DataLoader,
    rollout_val_samples: int,
    rollout_eval_limit: int,
) -> dict[str, float]:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []

    gt_delta_sum = np.zeros(25, dtype=np.float64)
    gt_delta_outer = np.zeros((25, 25), dtype=np.float64)
    gt_delta_count = 0
    ro_delta_sum = np.zeros(25, dtype=np.float64)
    ro_delta_outer = np.zeros((25, 25), dtype=np.float64)
    ro_delta_count = 0

    for history_01, future_01 in val_loader:
        if total_count >= rollout_eval_limit:
            break
        if total_count + history_01.shape[0] > rollout_eval_limit:
            keep = rollout_eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(
            history_norm,
            n_samples=rollout_val_samples,
            n_steps=future_01.shape[1],
        )

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

        future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
        samples_flat = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)
        gt_delta = future_flat[:, 29, :] - future_flat[:, 28, :]
        gen_delta = samples_flat[:, :, 29, :] - samples_flat[:, :, 28, :]
        gen_flat = gen_delta.reshape(-1, 25)

        gt_np = gt_delta.detach().cpu().numpy()
        gen_np = gen_flat.detach().cpu().numpy()
        gt_delta_sum += gt_np.sum(axis=0)
        gt_delta_outer += gt_np.T @ gt_np
        gt_delta_count += gt_np.shape[0]
        ro_delta_sum += gen_np.sum(axis=0)
        ro_delta_outer += gen_np.T @ gen_np
        ro_delta_count += gen_np.shape[0]

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "rollout_cov90": float("nan"),
            "rollout_width90": float("nan"),
            "rollout_mae": float("nan"),
            "rollout_support_violation_rate": float("nan"),
            "rollout_turb_calm_ratio": float("nan"),
            "rollout_corr_ratio_h30": float("nan"),
            "rollout_rank_ratio_h30": float("nan"),
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

    gt_mean = gt_delta_sum / max(gt_delta_count, 1)
    gt_cov = gt_delta_outer / max(gt_delta_count, 1) - np.outer(gt_mean, gt_mean)
    ro_mean = ro_delta_sum / max(ro_delta_count, 1)
    ro_cov = ro_delta_outer / max(ro_delta_count, 1) - np.outer(ro_mean, ro_mean)
    gt_corr = mean_offdiag_corr(gt_cov)
    ro_corr = mean_offdiag_corr(ro_cov)
    gt_rank = eff_rank_from_matrix(corr_from_cov(gt_cov))
    ro_rank = eff_rank_from_matrix(corr_from_cov(ro_cov))

    return {
        "rollout_cov90": total_cov / total_count,
        "rollout_width90": total_width / total_count,
        "rollout_mae": total_mae / total_count,
        "rollout_support_violation_rate": total_support_viol / total_count,
        "rollout_turb_calm_ratio": turb_calm_ratio,
        "rollout_corr_ratio_h30": float(ro_corr / gt_corr) if abs(gt_corr) > 1e-8 else float("nan"),
        "rollout_rank_ratio_h30": float(ro_rank / gt_rank) if gt_rank > 1e-8 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="201a Transformer AR rollout-tail Student-t")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--enc_d_model", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_d_model", type=int, default=128)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--tail_weight", type=float, default=4.0)
    parser.add_argument("--tail_mae_weight", type=float, default=0.05)
    parser.add_argument("--rollout_weight", type=float, default=0.25)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--rollout_warmup_epochs", type=int, default=5)
    parser.add_argument("--rollout_ramp_epochs", type=int, default=5)
    parser.add_argument("--rollout_var_weight", type=float, default=0.5)
    parser.add_argument("--rollout_corr_weight", type=float, default=0.1)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--decoder_warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
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
    train_abs_delta = np.abs(np.diff(surfaces[:4511], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

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

    encoder_config = dict(
        input_dim=25,
        d_model=args.enc_d_model,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.enc_dropout,
        bottleneck_dim=128,
        max_len=max(hist_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.dec_d_model,
        n_heads=args.dec_heads,
        n_layers=args.dec_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    objective_config = dict(
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        tail_weight=args.tail_weight,
        tail_mae_weight=args.tail_mae_weight,
        rollout_steps=args.rollout_steps,
        rollout_var_weight=args.rollout_var_weight,
        rollout_corr_weight=args.rollout_corr_weight,
    )

    model = TransformerRolloutTailStudentTARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_decoder_warm_start(model, args.decoder_warm_start)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 72}")
    print("201a: Transformer AR Student-t with Rollout-Consistent Tail-Aware Training")
    print(f"{'=' * 72}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Encoder: temporal transformer d={args.enc_d_model}, layers={args.enc_layers}, heads={args.enc_heads}")
    print(f"  Decoder: spatial Student-t d={args.dec_d_model}, layers={args.dec_layers}, rank={args.rank}")
    print(f"  Tail thresholds: q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Rollout consistency: steps={args.rollout_steps}, weight={args.rollout_weight}")

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

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "teacher_total_loss": 0.0,
            "multistep_nll": 0.0,
            "multistep_mae": 0.0,
            "pred_eff_rank": 0.0,
            "scale_mean": 0.0,
            "shape_avg_var": 0.0,
            "nu_mean": 0.0,
            "attention_top1": 0.0,
            "tail_weight_mean": 0.0,
            "rollout_consistency_loss": 0.0,
            "rollout_mean_consistency": 0.0,
            "rollout_var_consistency": 0.0,
            "rollout_corr_consistency": 0.0,
        }
        nb = 0

        if epoch <= args.rollout_warmup_epochs:
            rollout_scale = 0.0
        else:
            progress = (epoch - args.rollout_warmup_epochs) / max(args.rollout_ramp_epochs, 1)
            rollout_scale = float(min(max(progress, 0.0), 1.0))

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            teacher_loss, teacher_metrics = teacher_forced_multistep_objective(model, history_01, future_01, objective_config)
            loss = teacher_loss
            rollout_metrics = {
                "rollout_consistency_loss": torch.tensor(0.0, device=history_01.device),
                "rollout_mean_consistency": torch.tensor(0.0, device=history_01.device),
                "rollout_var_consistency": torch.tensor(0.0, device=history_01.device),
                "rollout_corr_consistency": torch.tensor(0.0, device=history_01.device),
            }
            if rollout_scale > 0.0 and args.rollout_weight > 0.0 and args.rollout_steps > 0:
                rollout_loss, rollout_metrics = rollout_consistency_objective(model, history_01, future_01, objective_config)
                loss = loss + (args.rollout_weight * rollout_scale) * rollout_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in ep:
                if k in teacher_metrics:
                    ep[k] += teacher_metrics[k].item()
                elif k in rollout_metrics:
                    ep[k] += rollout_metrics[k].item()
            nb += 1

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep.items()}
        train_metrics["train_rollout_scale"] = rollout_scale

        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )

        selection_score = (
            val_metrics["val_teacher_total_loss"]
            + 0.25 * rollout_metrics["rollout_mae"]
            + 0.10 * rollout_metrics["rollout_width90"]
        )

        elapsed = time.time() - t0
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score
            best_metrics = {**val_metrics, **rollout_metrics, "selection_score": selection_score}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": {
                        "type": "transformer_ar_rollout_tail_student_t_201a",
                        "encoder": encoder_config,
                        "decoder": decoder_config,
                        "objective": objective_config,
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
            **rollout_metrics,
            "selection_score": selection_score,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_tf={train_metrics['train_teacher_total_loss']:.4f}  "
            f"val_tf={val_metrics['val_teacher_total_loss']:.4f}  "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f}  "
            f"roll_mae={rollout_metrics['rollout_mae']:.4f}  "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f}  "
            f"rank={val_metrics['val_pred_eff_rank']:.2f}  "
            f"attn={val_metrics['val_attention_top1']:.3f}  "
            f"rs={rollout_scale:.2f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "selection_score": history[-1]["selection_score"] if history else float("nan"),
        "config": {
            "type": "transformer_ar_rollout_tail_student_t_201a",
            "encoder": encoder_config,
            "decoder": decoder_config,
            "objective": objective_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "fixed_nu": args.fixed_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest selection score: {best_score:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"rollout_cov90={best_metrics['rollout_cov90']:.4f}, "
            f"rollout_width90={best_metrics['rollout_width90']:.4f}, "
            f"rollout_turb_calm_ratio={best_metrics['rollout_turb_calm_ratio']:.3f}, "
            f"rollout_rank_ratio_h30={best_metrics['rollout_rank_ratio_h30']:.3f}"
        )


if __name__ == "__main__":
    main()
