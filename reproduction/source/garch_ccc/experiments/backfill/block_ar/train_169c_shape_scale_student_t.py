#!/usr/bin/env python
"""
169c: Multi-Step Student-t Density Model with Shape/Scale-Separated Covariance

Narrow follow-up to 169b.

Keep fixed:
  - support-aware transform
  - multistep teacher-forced Student-t NLL
  - fixed scalar nu (recommended: 8.0)
  - GRU encoder and AR rollout interface

Change only:
  - separate covariance shape from total scale
  - Q_t = L L^T + diag(d^2)
  - Q_t is normalized to unit average variance
  - Sigma_t = s_t^2 * Q_t_norm

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_169c_shape_scale_student_t.py \
        --epochs 20 --batch_size 16 --rank 5 --fixed_nu 8.0 \
        --output_dir models/backfill/student_t_169c --device cuda
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


class SpatialShapeScaleStudentTDecoder(nn.Module):
    """Spatial transformer with shape / scale separated covariance heads."""

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
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.10,
        init_scale: float = 0.10,
        init_nu: float = 8.0,
        fixed_nu: float | None = 8.0,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.rank = rank
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
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, 1)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_scale=init_scale, init_nu=init_nu)

    def _init_parameters(self, init_diag: float, init_scale: float, init_nu: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in ("mean_head", "factor_head", "diag_head", "scale_head", "nu_head")
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

        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(
                self.nu_head.bias,
                inverse_softplus(max(init_nu - self.nu_floor, 1e-6)),
            )

    def forward(
        self, cond: torch.Tensor, prev_u: torch.Tensor
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

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)
        return mu, factor, diag, scale, nu


class ShapeScaleStudentTARModel(nn.Module):
    """GRU history encoder + shape/scale Student-t decoder."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SpatialShapeScaleStudentTDecoder(**decoder_config)
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

    def raw_covariance(self, factor: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        cov = factor @ factor.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag.pow(2) + self.cov_jitter)
        return cov

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

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]

        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        eps_lowrank = torch.randn(
            batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype
        )
        eps_diag = torch.randn(
            batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype
        )
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
    model: ShapeScaleStudentTARModel,
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
    total_nu = 0.0

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
        mu, factor, diag, scale, nu = model.decoder(cond, prev_u)

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        nll_t = model.student_t_nll(target_u, mu, factor, diag, scale, nu)
        cov_t = model.covariance(factor, diag, scale)
        _, _, avg_var = model.normalized_components(factor, diag)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(cov_t).mean()
        total_shape_var = total_shape_var + avg_var.mean()
        total_scale = total_scale + scale.mean()
        total_nu = total_nu + nu.mean()

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
        "nu_mean": total_nu * norm,
    }
    return total_nll * norm, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: ShapeScaleStudentTARModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_shape_var = 0.0
    total_scale = 0.0
    total_nu = 0.0
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = teacher_forced_multistep_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        total_nll += loss.item() * batch_size
        total_mae += metrics["multistep_mae"].item() * batch_size
        total_rank += metrics["pred_eff_rank"].item() * batch_size
        total_shape_var += metrics["shape_avg_var"].item() * batch_size
        total_scale += metrics["scale_mean"].item() * batch_size
        total_nu += metrics["nu_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {
            "val_multistep_nll": float("nan"),
            "val_multistep_mae": float("nan"),
            "val_pred_eff_rank": float("nan"),
            "val_shape_avg_var": float("nan"),
            "val_scale_mean": float("nan"),
            "val_nu_mean": float("nan"),
        }

    return {
        "val_multistep_nll": total_nll / total_count,
        "val_multistep_mae": total_mae / total_count,
        "val_pred_eff_rank": total_rank / total_count,
        "val_shape_avg_var": total_shape_var / total_count,
        "val_scale_mean": total_scale / total_count,
        "val_nu_mean": total_nu / total_count,
    }


@torch.no_grad()
def evaluate_rollout_subset(
    model: ShapeScaleStudentTARModel,
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
    parser = argparse.ArgumentParser(description="169c: shape/scale multistep Student-t prototype")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
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
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--rollout_eval_every", type=int, default=1)
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
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    model = ShapeScaleStudentTARModel(
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
    print("169c: Multi-Step Student-t with Shape/Scale-Separated Covariance")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Rank={args.rank} | d_model={args.d_model}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: teacher-forced multistep Student-t NLL")
    print("  Covariance: normalized shape x learned scalar scale")
    print(f"  Tail parameter: {'learned' if args.fixed_nu is None else f'fixed nu={args.fixed_nu}'}")

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
        ep_nu = 0.0
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
            ep_nu += metrics["nu_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_multistep_nll": ep_loss / max(nb, 1),
            "train_multistep_mae": ep_mae / max(nb, 1),
            "train_pred_eff_rank": ep_rank / max(nb, 1),
            "train_shape_avg_var": ep_shape_var / max(nb, 1),
            "train_scale_mean": ep_scale / max(nb, 1),
            "train_nu_mean": ep_nu / max(nb, 1),
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
                    "type": "multi_step_student_t_169c",
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
            f"shape_var={val_metrics['val_shape_avg_var']:.4f}  "
            f"viol={rollout_metrics['rollout_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_multistep_nll": history[-1]["val_multistep_nll"] if history else float("nan"),
        "config": {
            "type": "multi_step_student_t_169c",
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
