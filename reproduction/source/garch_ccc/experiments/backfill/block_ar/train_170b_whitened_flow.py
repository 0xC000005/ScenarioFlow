#!/usr/bin/env python
"""
170b: Conditionally Whitened Support-Aware Density Model

Literature-guided next step after 169c:
  1. Estimate conditional mean and covariance in transformed IV space.
  2. Whiten next-step residuals with the predicted covariance.
  3. Model the whitened residual law with a conditional normalizing flow.

This keeps the support-aware density framing while moving beyond simple
elliptical residual laws.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_170b_whitened_flow.py \
        --epochs 20 --batch_size 16 --rank 5 \
        --output_dir models/backfill/whitened_flow_170b --device cuda
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

import sys; sys.path.insert(0, ".")
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


class ConditionalAffineCoupling(nn.Module):
    """Simple conditional affine coupling layer for 25-d whitened residuals."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_clip: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.scale_clip = scale_clip
        self.register_buffer("mask", mask.float().view(1, dim))

        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        for module in self.net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _st(self, x_masked: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x_masked, context], dim=-1)
        log_s, t = self.net(h).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.scale_clip
        inv_mask = 1.0 - self.mask
        log_s = log_s * inv_mask
        t = t * inv_mask
        return log_s, t

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_s, t = self._st(x_masked, context)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_s) + t)
        logdet = log_s.sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        log_s, t = self._st(y_masked, context)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-log_s))
        logdet = -log_s.sum(dim=-1)
        return x, logdet


class ConditionalResidualFlow(nn.Module):
    """Stack of conditional affine coupling layers."""

    def __init__(
        self,
        dim: int = 25,
        context_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 4,
        scale_clip: float = 2.0,
    ):
        super().__init__()
        masks = []
        for li in range(n_layers):
            pattern = ((torch.arange(dim) + li) % 2 == 0).float()
            masks.append(pattern)
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask=mask,
                scale_clip=scale_clip,
            )
            for mask in masks
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, logdet = layer(z, context)
            total_logdet = total_logdet + logdet
        return z, total_logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, context)
            total_logdet = total_logdet + logdet
        return x, total_logdet


class SpatialWhitenedFlowDecoder(nn.Module):
    """Spatial transformer trunk with mean/covariance heads + flow context head."""

    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        init_diag: float = 0.10,
        init_scale: float = 0.10,
        flow_context_dim: int = 128,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.rank = rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.flow_context_dim = flow_context_dim

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)

        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ff_norm": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            }))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, rank)
        self.diag_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        self.flow_context_head = nn.Sequential(
            nn.Linear(cond_dim + n_cells, flow_context_dim),
            nn.SiLU(),
            nn.Linear(flow_context_dim, flow_context_dim),
        )
        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in (
                        "mean_head",
                        "factor_head",
                        "diag_head",
                        "scale_head",
                        "flow_context_head",
                    )
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.factor_head.bias, mean=0.0, std=1e-3)

        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(
            self.diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            inverse_softplus(max(init_scale - self.scale_floor, 1e-6)),
        )

        for module in self.flow_context_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(prev_u.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[2 * li]
            ls_f = self.ls_params[2 * li + 1]

            h_norm = layer["attn_norm"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer["ff"](layer["ff_norm"](h))

        h = self.out_norm(h)
        pooled = h.mean(dim=1)

        mu = prev_u + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        flow_context = self.flow_context_head(torch.cat([cond, prev_u], dim=-1))
        return mu, factor, diag, scale, flow_context


class WhitenedFlowARModel(nn.Module):
    """GRU encoder + mean/cov estimator + flow in whitened residual space."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SpatialWhitenedFlowDecoder(**decoder_config)
        self.flow = ConditionalResidualFlow(**flow_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.flow_config = flow_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.encoder(history_norm)

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def log_prob_step(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        flow_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)

        diff = (target_u - mu).unsqueeze(-1)
        white = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
        z, flow_logdet = self.flow(white, flow_context)

        log2pi = target_u.new_tensor(math.log(2.0 * math.pi))
        base_logprob = -0.5 * (z.pow(2) + log2pi).sum(dim=-1)
        logdet_cov = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        logprob = base_logprob + flow_logdet - logdet_cov

        _, _, avg_var = self.normalized_components(factor, diag)
        aux = {
            "cov": cov,
            "avg_var": avg_var,
            "flow_logdet": flow_logdet,
            "white_norm": white.norm(dim=-1),
        }
        return logprob, aux

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, flow_context = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)

        z = torch.randn(batch_size * n_samples, n_cells, device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch_size, n_samples, -1).reshape(batch_size * n_samples, -1)
        white, _ = self.flow.inverse(z, ctx)
        white = white.view(batch_size, n_samples, n_cells)

        samples = mu.unsqueeze(1) + torch.einsum("bij,bnj->bni", chol, white)
        return samples

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
        batch_size, hist_len = history_01.shape[:2]

        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist_k = history_01.unsqueeze(1).expand(batch_size, k, -1, -1, -1)
            hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()

            frames = []
            for _ in range(n_steps):
                next_iv = self.sample_next_iv(hist_k, n_samples=1).squeeze(1)
                frames.append(next_iv.reshape(batch_size, k, 5, 5))
                hist_k = torch.cat([hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)], dim=1)

            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def teacher_forced_multistep_loss(
    model: WhitenedFlowARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    batch_size, hist_len = history_01.shape[:2]
    future_len = future_01.shape[1]
    n_cells = future_01.shape[-1]

    hist_norm = normalize_iv(history_01).reshape(batch_size, hist_len, n_cells)
    gru_outputs, gru_state = model.encoder.gru(hist_norm)
    prev_01 = history_01[:, -1].reshape(batch_size, n_cells)

    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_shape_var = 0.0
    total_scale = 0.0
    total_flow_logdet = 0.0
    total_white_norm = 0.0

    for step in range(future_len):
        attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        cond = model.encoder.bottleneck(pooled)

        prev_u = iv_to_unconstrained(
            prev_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu, factor, diag, scale, flow_context = model.decoder(cond, prev_u)

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        logprob_t, aux = model.log_prob_step(target_u, mu, factor, diag, scale, flow_context)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        total_nll = total_nll - logprob_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(aux["cov"]).mean()
        total_shape_var = total_shape_var + aux["avg_var"].mean()
        total_scale = total_scale + scale.mean()
        total_flow_logdet = total_flow_logdet + aux["flow_logdet"].mean()
        total_white_norm = total_white_norm + aux["white_norm"].mean()

        next_norm = normalize_iv(target_t).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = target_t

    norm = 1.0 / future_len
    metrics = {
        "multistep_nll": total_nll * norm,
        "multistep_mae": total_mae * norm,
        "pred_eff_rank": total_rank * norm,
        "shape_avg_var": total_shape_var * norm,
        "scale_mean": total_scale * norm,
        "flow_logdet_mean": total_flow_logdet * norm,
        "white_norm_mean": total_white_norm * norm,
    }
    return total_nll * norm, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: WhitenedFlowARModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    totals = {
        "val_multistep_nll": 0.0,
        "val_multistep_mae": 0.0,
        "val_pred_eff_rank": 0.0,
        "val_shape_avg_var": 0.0,
        "val_scale_mean": 0.0,
        "val_flow_logdet_mean": 0.0,
        "val_white_norm_mean": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = teacher_forced_multistep_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        totals["val_multistep_nll"] += loss.item() * batch_size
        totals["val_multistep_mae"] += metrics["multistep_mae"].item() * batch_size
        totals["val_pred_eff_rank"] += metrics["pred_eff_rank"].item() * batch_size
        totals["val_shape_avg_var"] += metrics["shape_avg_var"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_flow_logdet_mean"] += metrics["flow_logdet_mean"].item() * batch_size
        totals["val_white_norm_mean"] += metrics["white_norm_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_rollout_subset(
    model: WhitenedFlowARModel,
    val_loader: DataLoader,
    rollout_val_samples: int,
    rollout_eval_limit: int,
) -> dict:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []

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
        support_viol = (
            (samples < model.support_lo) | (samples > model.support_hi)
        ).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

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
        "rollout_cov90": total_cov / total_count,
        "rollout_width90": total_width / total_count,
        "rollout_mae": total_mae / total_count,
        "rollout_support_violation_rate": total_support_viol / total_count,
        "rollout_turb_calm_ratio": turb_calm_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="170b: conditionally whitened support-aware density model")
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
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--flow_context_dim", type=int, default=128)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
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
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, hist_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, hist_len, future_len
    )

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
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
    )
    flow_config = dict(
        dim=25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = WhitenedFlowARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("170b: Conditionally Whitened Support-Aware Density Model")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Rank={args.rank} | d_model={args.d_model}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: multistep conditional flow NLL on whitened residuals")
    print("  Mean/covariance: learned in transformed space")
    print("  Residual law: conditional affine coupling flow in whitened space")

    optimizer = torch.optim.AdamW([
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
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        ep_mae = 0.0
        ep_rank = 0.0
        ep_shape_var = 0.0
        ep_scale = 0.0
        ep_flow_logdet = 0.0
        ep_white_norm = 0.0
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = teacher_forced_multistep_loss(model, history_01, future_01)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mae += metrics["multistep_mae"].item()
            ep_rank += metrics["pred_eff_rank"].item()
            ep_shape_var += metrics["shape_avg_var"].item()
            ep_scale += metrics["scale_mean"].item()
            ep_flow_logdet += metrics["flow_logdet_mean"].item()
            ep_white_norm += metrics["white_norm_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_multistep_nll": ep_loss / max(nb, 1),
            "train_multistep_mae": ep_mae / max(nb, 1),
            "train_pred_eff_rank": ep_rank / max(nb, 1),
            "train_shape_avg_var": ep_shape_var / max(nb, 1),
            "train_scale_mean": ep_scale / max(nb, 1),
            "train_flow_logdet_mean": ep_flow_logdet / max(nb, 1),
            "train_white_norm_mean": ep_white_norm / max(nb, 1),
        }
        val_metrics = evaluate_teacher_forced(model, val_loader)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_multistep_nll"] < best_val
        if is_best:
            best_val = val_metrics["val_multistep_nll"]
            best_metrics = {**val_metrics, **rollout_metrics}
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_multistep_nll": best_val,
                "config": {
                    "type": "whitened_flow_170b",
                    "encoder": vars(encoder_config),
                    "decoder": decoder_config,
                    "flow": flow_config,
                    "support_lo": args.support_lo,
                    "support_hi": args.support_hi,
                    "support_eps": args.support_eps,
                    "history_len": hist_len,
                    "future_len": future_len,
                    "train_windows": len(train_indices),
                    "val_windows": len(val_indices),
                },
                "best_metrics": best_metrics,
            }, f"{args.output_dir}/best_model.pt")

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_multistep_nll']:.4f}  "
            f"val_nll={val_metrics['val_multistep_nll']:.4f}  "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f}  "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f}  "
            f"rank={val_metrics['val_pred_eff_rank']:.2f}  "
            f"scale={val_metrics['val_scale_mean']:.4f}  "
            f"fdet={val_metrics['val_flow_logdet_mean']:.4f}  "
            f"viol={rollout_metrics['rollout_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_multistep_nll": history[-1]["val_multistep_nll"] if history else float("nan"),
        "config": {
            "type": "whitened_flow_170b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_val_multistep_nll": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val multistep NLL: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"rollout_cov90={best_metrics['rollout_cov90']:.4f}, "
            f"rollout_turb_calm_ratio={best_metrics['rollout_turb_calm_ratio']:.3f}, "
            f"rollout_support_violation_rate={best_metrics['rollout_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
