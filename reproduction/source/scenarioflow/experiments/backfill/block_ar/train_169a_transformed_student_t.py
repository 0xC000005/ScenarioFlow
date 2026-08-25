#!/usr/bin/env python
"""
169a: Support-Aware One-Step Conditional Student-t Prototype

Minimal likelihood prototype for the Block-AR research line:
  1. Keep the GRU history encoder.
  2. Keep the autoregressive "condition on previous frame" setup.
  3. Move IV from bounded support (0.01, 1.0) to unconstrained space.
  4. Predict a low-rank multivariate Student-t law for the next frame.

This is intentionally small and diagnostic:
  - one-step teacher-forced training only
  - low-rank + diagonal covariance in transformed space
  - Student-t NLL instead of afCRPS / VS / IS
  - no reflecting boundary, no post-hoc support fix

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_169a_transformed_student_t.py \
        --epochs 20 --batch_size 16 --rank 5 \
        --output_dir models/backfill/student_t_169a --device cuda
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


def normalize_iv(surfaces: torch.Tensor) -> torch.Tensor:
    """Normalize IV surfaces from [0, 1] to [-1, 1]."""
    return surfaces * 2.0 - 1.0


def denormalize_iv(surfaces: torch.Tensor) -> torch.Tensor:
    """Denormalize IV surfaces from [-1, 1] to [0, 1]."""
    return (surfaces + 1.0) / 2.0


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def inverse_softplus(x: float) -> float:
    """Stable inverse softplus for positive scalars."""
    return float(np.log(np.expm1(x)))


def iv_to_unconstrained(
    x: torch.Tensor, lo: float = 0.01, hi: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Map IV on (lo, hi) to unconstrained space via logit."""
    scaled = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)
    return torch.logit(scaled)


def unconstrained_to_iv(
    u: torch.Tensor, lo: float = 0.01, hi: float = 1.0
) -> torch.Tensor:
    """Map unconstrained state back to IV support."""
    return lo + (hi - lo) * torch.sigmoid(u)


def effective_rank(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy-based effective rank of covariance matrices."""
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
    probs = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.exp()


class SpatialStudentTDecoder(nn.Module):
    """Spatial transformer trunk with parametric Student-t output heads."""

    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        rank: int = 5,
        diag_floor: float = 1e-3,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.10,
        init_nu: float = 8.0,
        fixed_nu: float | None = None,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
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
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, 1)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_nu=init_nu)

    def _init_parameters(self, init_diag: float, init_nu: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in ("mean_head", "factor_head", "diag_head", "nu_head")
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

        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(
                self.nu_head.bias,
                inverse_softplus(max(init_nu - self.nu_floor, 1e-6)),
            )

    def forward(
        self, cond: torch.Tensor, prev_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: (B, cond_dim)
            prev_u: (B, 25) previous frame in unconstrained support-aware space
        Returns:
            mu: (B, 25)
            factor: (B, 25, rank)
            diag: (B, 25) positive diagonal scale
            nu: (B,) positive Student-t df
        """
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
        mu = prev_u + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor

        if self.fixed_nu is None:
            pooled = h.mean(dim=1)
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)
        return mu, factor, diag, nu


class OneStepStudentTARModel(nn.Module):
    """GRU history encoder + one-step spatial Student-t decoder."""

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
        self.decoder = SpatialStudentTDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        """Encode history in [0, 1] using the existing GRU encoder."""
        history_norm = normalize_iv(history_01)
        return self.encoder(history_norm)

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            history_01: (B, H, 5, 5) in [0, 1]
        Returns:
            mu, factor, diag, nu in transformed next-frame space
        """
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        return self.decoder(cond, prev_u)

    def covariance(self, factor: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        cov = factor @ factor.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag.pow(2) + self.cov_jitter)
        return cov

    def student_t_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """Dense multivariate Student-t NLL for D=25."""
        d = target_u.shape[-1]
        cov = self.covariance(factor, diag)
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

    def loss(
        self, history_01: torch.Tensor, target_01: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        mu, factor, diag, nu = self.forward_from_history(history_01)
        target_u = iv_to_unconstrained(
            target_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        nll = self.student_t_nll(target_u, mu, factor, diag, nu)
        mu_iv = unconstrained_to_iv(mu, lo=self.support_lo, hi=self.support_hi)
        cov = self.covariance(factor, diag)
        metrics = {
            "nll": nll.mean(),
            "mean_mae": (mu_iv - target_01).abs().mean(),
            "diag_mean": diag.mean(),
            "factor_norm": factor.norm(dim=1).mean(),
            "nu_mean": nu.mean(),
            "pred_eff_rank": effective_rank(cov).mean(),
        }
        return nll.mean(), metrics

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, nu = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        rank = factor.shape[-1]

        eps_lowrank = torch.randn(
            batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype
        )
        eps_diag = torch.randn(
            batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype
        )
        lowrank_noise = torch.einsum("bcr,bnr->bnc", factor, eps_lowrank)
        diag_noise = diag.unsqueeze(1) * eps_diag

        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
        scale = torch.rsqrt(mix).unsqueeze(-1)

        return mu.unsqueeze(1) + (lowrank_noise + diag_noise) * scale

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
        """
        Compatibility helper for later Block-AR evaluation.

        Args:
            history: (B, H, 5, 5) in [-1, 1]
        Returns:
            (B, n_samples, n_steps, 5, 5) in [0.01, 1.0]
        """
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


def build_one_step_windows(indices: np.ndarray, surf: torch.Tensor, hist_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.from_numpy(indices).long().to(surf.device)
    offsets_h = torch.arange(hist_len, device=surf.device).unsqueeze(0)
    hist_idx = idx.unsqueeze(1) + offsets_h
    hist = surf[hist_idx]
    target = surf[idx + hist_len].reshape(len(indices), -1)
    return hist, target


@torch.no_grad()
def evaluate_one_step(
    model: OneStepStudentTARModel,
    val_loader: DataLoader,
    val_samples: int,
    support_lo: float,
    support_hi: float,
    limit: int | None = None,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_mae = 0.0
    total_cov = 0.0
    total_width = 0.0
    total_pred_rank = 0.0
    total_sample_rank = 0.0
    total_support_viol = 0.0
    total_nu = 0.0
    total_diag = 0.0
    total_count = 0

    for history_01, target_01 in val_loader:
        if limit is not None and total_count >= limit:
            break
        if limit is not None and total_count + history_01.shape[0] > limit:
            keep = limit - total_count
            history_01 = history_01[:keep]
            target_01 = target_01[:keep]

        mu, factor, diag, nu = model.forward_from_history(history_01)
        target_u = iv_to_unconstrained(
            target_01,
            lo=support_lo,
            hi=support_hi,
            eps=model.support_eps,
        )
        nll = model.student_t_nll(target_u, mu, factor, diag, nu)
        mu_iv = unconstrained_to_iv(mu, lo=support_lo, hi=support_hi)
        cov = model.covariance(factor, diag)

        samples_u = model.sample_next_u(history_01, n_samples=val_samples)
        samples_iv = unconstrained_to_iv(samples_u, lo=support_lo, hi=support_hi)
        lo = samples_iv.quantile(0.05, dim=1)
        hi = samples_iv.quantile(0.95, dim=1)
        coverage = ((target_01 >= lo) & (target_01 <= hi)).float().mean()
        width = (hi - lo).mean()
        support_viol = ((samples_iv < support_lo) | (samples_iv > support_hi)).float().mean()

        centered = samples_u - samples_u.mean(dim=1, keepdim=True)
        denom = max(val_samples - 1, 1)
        sample_cov = torch.einsum("bnc,bnd->bcd", centered, centered) / denom

        batch_size = history_01.shape[0]
        total_nll += nll.sum().item()
        total_mae += (mu_iv - target_01).abs().sum().item() / target_01.shape[-1]
        total_cov += coverage.item() * batch_size
        total_width += width.item() * batch_size
        total_pred_rank += effective_rank(cov).sum().item()
        total_sample_rank += effective_rank(sample_cov).sum().item()
        total_support_viol += support_viol.item() * batch_size
        total_nu += nu.sum().item()
        total_diag += diag.mean(dim=-1).sum().item()
        total_count += batch_size

    if total_count == 0:
        return {
            "val_nll": float("nan"),
            "val_mae": float("nan"),
            "coverage_90": float("nan"),
            "width_90": float("nan"),
            "pred_eff_rank": float("nan"),
            "sample_eff_rank": float("nan"),
            "support_violation_rate": float("nan"),
            "nu_mean": float("nan"),
            "diag_mean": float("nan"),
        }

    return {
        "val_nll": total_nll / total_count,
        "val_mae": total_mae / total_count,
        "coverage_90": total_cov / total_count,
        "width_90": total_width / total_count,
        "pred_eff_rank": total_pred_rank / total_count,
        "sample_eff_rank": total_sample_rank / total_count,
        "support_violation_rate": total_support_viol / total_count,
        "nu_mean": total_nu / total_count,
        "diag_mean": total_diag / total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="169a: one-step transformed Student-t prototype")
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
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=None,
                        help="If set, disable learned nu head and use constant Student-t df")
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--val_samples", type=int, default=32)
    parser.add_argument("--val_eval_limit", type=int, default=256)
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
    n_total = surfaces.shape[0]
    hist_len = args.history_len

    test_start = 4511
    max_train_idx = test_start - hist_len - 30
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, hist_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, hist_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target),
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
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
    )
    model = OneStepStudentTARModel(
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
    print("169a: Support-Aware One-Step Conditional Student-t Prototype")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | rank={args.rank} | d_model={args.d_model}")
    print(f"  Support transform: logit(({args.support_lo}, {args.support_hi}))")
    print(f"  Objective: multivariate Student-t NLL")
    if args.fixed_nu is None:
        print(f"  Tail parameter: learned nu")
    else:
        print(f"  Tail parameter: fixed nu={args.fixed_nu}")

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
        ep_nu = 0.0
        nb = 0

        for history_01, target_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = model.loss(history_01, target_01)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mae += metrics["mean_mae"].item()
            ep_rank += metrics["pred_eff_rank"].item()
            ep_diag += metrics["diag_mean"].item()
            ep_nu += metrics["nu_mean"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_nll": ep_loss / max(nb, 1),
            "train_mae": ep_mae / max(nb, 1),
            "train_pred_eff_rank": ep_rank / max(nb, 1),
            "train_diag_mean": ep_diag / max(nb, 1),
            "train_nu_mean": ep_nu / max(nb, 1),
        }
        val_metrics = evaluate_one_step(
            model=model,
            val_loader=val_loader,
            val_samples=args.val_samples,
            support_lo=args.support_lo,
            support_hi=args.support_hi,
            limit=args.val_eval_limit,
        )
        elapsed = time.time() - t0
        is_best = val_metrics["val_nll"] < best_val
        if is_best:
            best_val = val_metrics["val_nll"]
            best_metrics = dict(val_metrics)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_nll": best_val,
                "config": {
                    "type": "one_step_student_t_169a",
                    "encoder": vars(encoder_config),
                    "decoder": decoder_config,
                    "support_lo": args.support_lo,
                    "support_hi": args.support_hi,
                    "support_eps": args.support_eps,
                    "fixed_nu": args.fixed_nu,
                    "history_len": hist_len,
                    "train_windows": len(train_indices),
                    "val_windows": len(val_indices),
                },
                "best_metrics": best_metrics,
            }, f"{args.output_dir}/best_model.pt")

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch:3d}  "
            f"train_nll={train_metrics['train_nll']:.4f}  "
            f"val_nll={val_metrics['val_nll']:.4f}  "
            f"cov90={val_metrics['coverage_90']:.4f}  "
            f"rank={val_metrics['sample_eff_rank']:.2f}  "
            f"nu={val_metrics['nu_mean']:.2f}  "
            f"diag={val_metrics['diag_mean']:.4f}  "
            f"viol={val_metrics['support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_nll": history[-1]["val_nll"] if history else float("nan"),
        "config": {
            "type": "one_step_student_t_169a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "fixed_nu": args.fixed_nu,
            "history_len": hist_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_val_nll": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val NLL: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"coverage_90={best_metrics['coverage_90']:.4f}, "
            f"sample_eff_rank={best_metrics['sample_eff_rank']:.2f}, "
            f"support_violation_rate={best_metrics['support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
