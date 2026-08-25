#!/usr/bin/env python
"""
170a: Joint Future Support-Aware Student-t Density Model

Minimal clean test of the H7 hypothesis after the 169c vs 170b mechanistic study:

  - Keep support-aware transformed-space density modeling
  - Remove autoregressive rollout from generation entirely
  - Predict the full 30x25 future jointly
  - Train with a proper joint Student-t likelihood

This is intentionally simpler than a full conditional flow. The point is to
test whether removing AR exposure bias is enough to improve the density branch.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_170a_joint_future_student_t.py \
        --epochs 20 --batch_size 16 --rank 16 \
        --output_dir models/backfill/joint_future_student_t_170a --device cuda
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


class JointFutureStudentTDecoder(nn.Module):
    """Factored temporal-spatial decoder for the full future block."""

    def __init__(
        self,
        n_frames: int = 30,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        rank: int = 16,
        diag_floor: float = 1e-3,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.05,
        init_nu: float = 8.0,
        fixed_nu: float | None = 8.0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.rank = rank
        self.diag_floor = diag_floor
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
            self.layers.append(nn.ModuleDict({
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
            }))

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, rank)
        self.diag_head = nn.Linear(d_model, 1)
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, 1)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_nu=init_nu)

    def _init_parameters(self, init_diag: float, init_nu: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(head_name in name for head_name in ("mean_head", "factor_head", "diag_head", "nu_head")):
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: (B, cond_dim)
            prev_u: (B, 25) last observed frame in unconstrained space
        Returns:
            mu: (B, T*C)
            factor: (B, T*C, rank)
            diag: (B, T*C)
            nu: (B,)
        """
        B = prev_u.shape[0]
        T, C = self.n_frames, self.n_cells

        prev_rep = prev_u.unsqueeze(1).expand(B, T, C)
        h = self.input_proj(prev_rep.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1).unsqueeze(1)
        h = h + self.temporal_pos + self.spatial_pos

        for layer in self.layers:
            h_temp = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
            h_norm = layer["temp_norm"](h_temp)
            attn_out, _ = layer["temp_attn"](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer["temp_ff"](layer["temp_ff_norm"](h_temp))
            h = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3)

            h_spat = h.reshape(B * T, C, -1)
            h_norm = layer["spat_norm"](h_spat)
            attn_out, _ = layer["spat_attn"](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer["spat_ff"](layer["spat_ff_norm"](h_spat))
            h = h_spat.reshape(B, T, C, -1)

        h = self.out_norm(h)
        mu = prev_rep + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h).reshape(B, T * C, self.rank)
        diag = F.softplus(self.diag_head(h).squeeze(-1)).reshape(B, T * C) + self.diag_floor

        if self.fixed_nu is None:
            pooled = h.mean(dim=(1, 2))
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(B).to(h.dtype)

        return mu.reshape(B, T * C), factor, diag, nu


class JointFutureStudentTModel(nn.Module):
    """GRU history encoder + joint future low-rank Student-t density decoder."""

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
        self.decoder = JointFutureStudentTDecoder(**decoder_config)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        return self.decoder(cond, prev_u)

    def student_t_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        diff = target_u - mu
        diag_sq = diag.pow(2) + self.cov_jitter
        inv_diag_sq = diag_sq.reciprocal()

        # Woodbury / determinant lemma for low-rank + diagonal covariance.
        ft_dinv = factor.transpose(-1, -2) * inv_diag_sq.unsqueeze(1)
        eye = torch.eye(factor.shape[-1], device=factor.device, dtype=factor.dtype).unsqueeze(0)
        middle = eye + torch.matmul(ft_dinv, factor)
        chol_mid = torch.linalg.cholesky(middle)

        logdet = torch.log(diag_sq).sum(dim=-1)
        logdet = logdet + 2.0 * torch.log(
            torch.diagonal(chol_mid, dim1=-2, dim2=-1)
        ).sum(dim=-1)

        dinv_diff = diff * inv_diag_sq
        v = torch.einsum("bdr,bd->br", factor, dinv_diff)
        tmp = torch.cholesky_solve(v.unsqueeze(-1), chol_mid).squeeze(-1)
        mahal = (diff * dinv_diff).sum(dim=-1) - (v * tmp).sum(dim=-1)

        d = target_u.shape[-1]
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
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, nu = self.forward_from_history(history_01)
        batch_size, dim = mu.shape
        rank = factor.shape[-1]

        eps_r = torch.randn(batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype)
        eps_d = torch.randn(batch_size, n_samples, dim, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bdr,bsr->bsd", factor, eps_r)
        diag_noise = diag.unsqueeze(1) * eps_d

        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)

        return mu.unsqueeze(1) + (lowrank_noise + diag_noise) * t_scale

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
        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            samples_u = self.sample_future_u(history_01, n_samples=k)
            samples_01 = unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)
            all_chunks.append(samples_01.view(batch_size, k, self.decoder.n_frames, 5, 5))
        return torch.cat(all_chunks, dim=1)


def joint_nll_loss(
    model: JointFutureStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    target_u = iv_to_unconstrained(
        future_01.reshape(future_01.shape[0], -1),
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    mu, factor, diag, nu = model.forward_from_history(history_01)
    nll = model.student_t_nll(target_u, mu, factor, diag, nu)
    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    pred_01 = pred_01.view_as(future_01)

    # Approximate per-frame rank from sampled covariance over the first frame block.
    cov_first = factor[:, :25] @ factor[:, :25].transpose(-1, -2) + torch.diag_embed(diag[:, :25].pow(2))
    metrics = {
        "joint_nll": nll.mean(),
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "first_frame_eff_rank": effective_rank(cov_first).mean(),
        "diag_mean": diag.mean(),
        "factor_norm_mean": factor.norm(dim=(1, 2)).mean(),
        "nu_mean": nu.mean(),
    }
    return nll.mean(), metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: JointFutureStudentTModel,
    val_loader: DataLoader,
) -> dict:
    model.eval()
    totals = {
        "val_joint_nll": 0.0,
        "val_joint_mae": 0.0,
        "val_first_frame_eff_rank": 0.0,
        "val_diag_mean": 0.0,
        "val_factor_norm_mean": 0.0,
        "val_nu_mean": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        loss, metrics = joint_nll_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        totals["val_joint_nll"] += loss.item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_first_frame_eff_rank"] += metrics["first_frame_eff_rank"].item() * batch_size
        totals["val_diag_mean"] += metrics["diag_mean"].item() * batch_size
        totals["val_factor_norm_mean"] += metrics["factor_norm_mean"].item() * batch_size
        totals["val_nu_mean"] += metrics["nu_mean"].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: JointFutureStudentTModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
) -> dict:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
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

        first_sample = samples[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        all_sample_eff_rank.append(
            eff_rank_np(corr)
        )

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
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
    }


def eff_rank_np(corr: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    probs = eigvals / eigvals.sum()
    return float(np.exp(-(probs * np.log(probs)).sum()))


def main():
    parser = argparse.ArgumentParser(description="170a: joint future support-aware Student-t")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--init_diag", type=float, default=0.05)
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
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

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
        rank=args.rank,
        diag_floor=args.diag_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    model = JointFutureStudentTModel(
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
    print("170a: Joint Future Support-Aware Student-t")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Joint dim: {future_len * 25} | rank={args.rank}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print("  Objective: joint future Student-t NLL")
    print("  Generation: one-shot future block, no AR rollout")
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
        ep_diag = 0.0
        ep_factor = 0.0
        ep_nu = 0.0
        nb = 0

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
            ep_rank += metrics["first_frame_eff_rank"].item()
            ep_diag += metrics["diag_mean"].item()
            ep_factor += metrics["factor_norm_mean"].item()
            ep_nu += metrics["nu_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_joint_nll": ep_loss / max(nb, 1),
            "train_joint_mae": ep_mae / max(nb, 1),
            "train_first_frame_eff_rank": ep_rank / max(nb, 1),
            "train_diag_mean": ep_diag / max(nb, 1),
            "train_factor_norm_mean": ep_factor / max(nb, 1),
            "train_nu_mean": ep_nu / max(nb, 1),
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_joint_nll": best_val,
                "config": {
                    "type": "joint_future_student_t_170a",
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
            **joint_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_joint_nll']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"joint_cov90={joint_metrics['joint_cov90']:.4f}  "
            f"joint_tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"rank={val_metrics['val_first_frame_eff_rank']:.2f}  "
            f"diag={val_metrics['val_diag_mean']:.4f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_joint_nll": history[-1]["val_joint_nll"] if history else float("nan"),
        "config": {
            "type": "joint_future_student_t_170a",
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
            f"joint_support_violation_rate={best_metrics['joint_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
