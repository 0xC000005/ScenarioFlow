#!/usr/bin/env python
"""
227a: End-to-end factor-structured AR scenario generator.

Combines:
- 226a's factor-decoupled architecture (correct cross-cell structure)
- 97a's end-to-end AR training (gradient through all N steps)
- Variogram Score (distribution-free cross-cell dependence)

Architecture:
  History → GRUEncoder → cond_0, local_scale_0
  AR loop (N steps, K members vectorized):
    factor_scores = FactorHead(prev, cond, z_f, pos)
    idio_residuals = IdioHead(prev, cond, z_i, pos)
    v = Λ(cond) @ f + D(cond) ⊙ ε
    delta = sinh(v) × local_scale
    next_iv = clamp(prev + delta, 0.001, 1.0)
    EWMA update local_scale
    GRUCell update cond
  Loss: Σ_t [ES + λ_vs × VS] on full trajectory
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
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    normalize_iv,
)


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for AR step index."""
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim

    def forward(self, t: int, batch_size: int, device: torch.device) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        args = torch.tensor([t], device=device, dtype=torch.float32) * freqs
        emb = torch.cat([args.sin(), args.cos()])  # (dim,)
        return emb.unsqueeze(0).expand(batch_size, -1)  # (B, dim)


class FactorARModel(nn.Module):
    """End-to-end factor-structured AR scenario generator."""

    def __init__(
        self,
        n_cells: int = 25,
        factor_rank: int = 6,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        decoder_hidden: int = 128,
        rho: float = 0.8,
        ewma_alpha: float = 0.20,
        scale_floor: float = 1e-4,
        include_scale_feature: bool = True,
        pos_embed_dim: int = 16,
        noise_skip: bool = False,
        d_scale: float = 1.0,
        cell_spread: bool = False,
        decoder_layers: int = 2,
        no_tanh: bool = False,
        use_scale_anchor: bool = False,
        scale_anchor_alpha: float = 0.50,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.factor_rank = factor_rank
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.ewma_alpha = ewma_alpha
        self.scale_floor = scale_floor
        self.include_scale_feature = include_scale_feature
        self.noise_skip = noise_skip
        self.d_scale = d_scale
        self.cell_spread_enabled = cell_spread
        self.no_tanh = no_tanh
        self.use_scale_anchor = use_scale_anchor
        self.scale_anchor_alpha = scale_anchor_alpha

        # History feature dim: 25 levels + 25 deltas + 25 log_scales = 75
        history_feat_dim = 75 if include_scale_feature else 50

        # GRU encoder for initial history encoding
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )

        # GRUCell for recurrent state update during AR loop
        self.gru_cell = nn.GRUCell(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
        )

        # Positional embedding
        self.pos_embed = SinusoidalPosEmbed(dim=pos_embed_dim)

        # Factor head: (prev_25, cond_128, z_f_6, pos_16) → 6
        factor_input_dim = n_cells + hidden_dim + factor_rank + pos_embed_dim
        factor_layers = [nn.Linear(factor_input_dim, decoder_hidden), nn.SiLU()]
        for _ in range(decoder_layers - 1):
            factor_layers += [nn.Linear(decoder_hidden, decoder_hidden), nn.SiLU()]
        if no_tanh:
            factor_layers += [nn.Linear(decoder_hidden, factor_rank)]
        else:
            factor_layers += [nn.Linear(decoder_hidden, factor_rank), nn.Tanh()]
        self.factor_head = nn.Sequential(*factor_layers)

        # Idiosyncratic head: (prev_25, cond_128, z_i_25, pos_16) → 25
        idio_input_dim = n_cells + hidden_dim + n_cells + pos_embed_dim
        idio_layers = [nn.Linear(idio_input_dim, decoder_hidden), nn.SiLU()]
        for _ in range(decoder_layers - 1):
            idio_layers += [nn.Linear(decoder_hidden, decoder_hidden), nn.SiLU()]
        if no_tanh:
            idio_layers += [nn.Linear(decoder_hidden, n_cells)]
        else:
            idio_layers += [nn.Linear(decoder_hidden, n_cells), nn.Tanh()]
        self.idio_head = nn.Sequential(*idio_layers)

        # Noise skip: direct noise → output bypass (like 97a's noise_skip_proj)
        if noise_skip:
            self.noise_skip_proj = nn.Linear(n_cells, n_cells)
            nn.init.normal_(self.noise_skip_proj.weight, std=0.01)
            nn.init.zeros_(self.noise_skip_proj.bias)

        # Cell spread: condition-dependent per-cell scaling (like 97a)
        if cell_spread:
            self.cell_spread_proj = nn.Linear(hidden_dim, n_cells)
            nn.init.zeros_(self.cell_spread_proj.weight)
            nn.init.constant_(self.cell_spread_proj.bias, 0.541)  # softplus(0.541) ≈ 1.0

        # Loading matrix Λ(cond) = Λ_base + MLP(cond)
        self.lambda_base = nn.Parameter(torch.zeros(n_cells, factor_rank))
        self.lambda_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_cells * factor_rank),
        )

        # Idiosyncratic scale D(cond) = softplus(D_bias + Linear(cond))
        self.d_bias = nn.Parameter(torch.zeros(n_cells))
        self.d_linear = nn.Linear(hidden_dim, n_cells)

        self._init_weights()

    def _init_weights(self):
        # Small-init decoder output layers (avoids dead start, like 97a)
        # [-2] is the last Linear (before Tanh)
        for head in [self.factor_head, self.idio_head]:
            for module in reversed(list(head.modules())):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
                    break
        # Zero-init Λ MLP final layer (Λ starts as Λ_base from PCA)
        nn.init.zeros_(self.lambda_mlp[2].weight)
        nn.init.zeros_(self.lambda_mlp[2].bias)
        # Zero-init D linear (D starts as PCA residual)
        nn.init.zeros_(self.d_linear.weight)
        nn.init.zeros_(self.d_linear.bias)

    def init_from_pca(self, lambda_init: np.ndarray, d_init: np.ndarray) -> None:
        """Initialize factor loadings and idiosyncratic scale from PCA."""
        with torch.no_grad():
            self.lambda_base.copy_(torch.from_numpy(lambda_init))
            d_target = torch.from_numpy(d_init).clamp_min(0.01) * self.d_scale
            self.d_bias.copy_(torch.log(torch.exp(d_target) - 1.0))

    def get_lambda(self, cond: torch.Tensor) -> torch.Tensor:
        """(B, hidden) → (B, 25, r)"""
        mlp_out = self.lambda_mlp(cond).view(cond.shape[0], self.n_cells, self.factor_rank)
        return self.lambda_base.unsqueeze(0) + mlp_out

    def get_d(self, cond: torch.Tensor) -> torch.Tensor:
        """(B, hidden) → (B, 25)"""
        return F.softplus(self.d_bias.unsqueeze(0) + self.d_linear(cond))

    def encode_history(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode history → (cond, local_scale)."""
        feat, local_scale = build_local_scale_history_features(
            history_01=history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        return h_n[-1], local_scale  # (B, 128), (B, 25)

    def _step_features(
        self, prev_01: torch.Tensor, next_01: torch.Tensor, local_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-step features for GRUCell update + update EWMA scale."""
        delta = next_01 - prev_01
        new_scale = (
            self.ewma_alpha * delta.abs().clamp_min(self.scale_floor)
            + (1.0 - self.ewma_alpha) * local_scale
        ).clamp_min(self.scale_floor)

        feat_parts = [
            next_01 * 2.0 - 1.0,  # normalized levels
            delta / new_scale,     # standardized deltas
        ]
        if self.include_scale_feature:
            feat_parts.append(torch.log(new_scale))
        feat = torch.cat(feat_parts, dim=-1)  # (B, 75)
        return feat, new_scale

    def forward(
        self,
        history_01: torch.Tensor,
        n_members: int,
        n_steps: int,
    ) -> torch.Tensor:
        """Generate trajectories: (B, K, N, 5, 5).

        Gradient flows through all N AR steps.
        """
        B = history_01.shape[0]
        device = history_01.device

        # Encode history
        cond, local_scale = self.encode_history(history_01)
        prev = history_01[:, -1].reshape(B, self.n_cells)

        # Capture scale anchor before BK expansion (for anchored EWMA)
        if self.use_scale_anchor:
            scale_anchor = local_scale.clone()  # (B, 25)

        # Vectorize K members into batch dim
        BK = B * n_members
        cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        if self.use_scale_anchor:
            scale_anchor_bk = scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        # Init AR(1) factor noise
        z_f = torch.randn(BK, self.factor_rank, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        frames = []
        for t in range(n_steps):
            # AR(1) noise update for factors
            if t > 0:
                z_f = self.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
            # iid noise for idiosyncratic
            z_i = torch.randn(BK, self.n_cells, device=device)

            # Positional embedding
            pos = self.pos_embed(t, BK, device)

            # Factor scores and idiosyncratic residuals
            factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
            f_scores = self.factor_head(factor_in)  # (BK, 6)

            idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
            i_resid = self.idio_head(idio_in)  # (BK, 25)

            # Combine: v = Λ(cond) @ f + D(cond) ⊙ ε + noise_skip
            Lambda = self.get_lambda(cond)  # (BK, 25, 6)
            D = self.get_d(cond)  # (BK, 25)
            v = torch.einsum("br,bcr->bc", f_scores, Lambda) + D * i_resid
            if self.noise_skip:
                v = v + torch.tanh(self.noise_skip_proj(z_i))

            # Cell spread: condition-dependent per-cell scaling
            if self.cell_spread_enabled:
                cs = F.softplus(self.cell_spread_proj(cond))  # (BK, 25)
                v = v * cs

            # Innovation → delta → next IV
            delta = torch.sinh(v) * local_scale
            next_iv = (prev + delta).clamp(0.001, 1.0)
            frames.append(next_iv)

            # Update state for next step
            feat, local_scale = self._step_features(prev, next_iv, local_scale)
            cond = self.gru_cell(feat, cond)
            prev = next_iv

            # Anchored EWMA: blend self-fed scale with initial anchor in log-space
            if self.use_scale_anchor:
                log_s = ((1.0 - self.scale_anchor_alpha)
                         * torch.log(local_scale.clamp_min(self.scale_floor))
                         + self.scale_anchor_alpha
                         * torch.log(scale_anchor_bk.clamp_min(self.scale_floor)))
                local_scale = torch.exp(log_s)

        # (N, BK, 25) → (B, K, N, 5, 5)
        trajectory = torch.stack(frames, dim=0)  # (N, BK, 25)
        trajectory = trajectory.permute(1, 0, 2)  # (BK, N, 25)
        trajectory = trajectory.view(B, n_members, n_steps, 5, 5)
        return trajectory

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Public API for evaluation. Returns (B, K, N, 5, 5) in [0,1]."""
        if history_is_normalized:
            from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv
            history_01 = denormalize_iv(history)
        else:
            history_01 = history

        chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            chunk = self.forward(history_01, n_members=k, n_steps=n_steps)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int = 1,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step sampling for compatibility with OneDayKernelRolloutWrapper."""
        return self.forward(history_01, n_members=n_samples, n_steps=1).squeeze(2).reshape(
            history_01.shape[0], n_samples, self.n_cells
        )


def afcrps_per_step(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Almost-fair CRPS for one time step.

    Args:
        samples: (B, K, D) generated samples
        target: (B, D) ground truth
    Returns:
        scalar loss (mean over batch)
    """
    # MAE: mean |sample - target| per cell, averaged over K members
    mae = (samples - target.unsqueeze(1)).abs().mean(dim=1).mean(dim=1)  # (B,)

    # Spread: mean |sample_i - sample_j| using random permutation
    K = samples.shape[1]
    idx = torch.randperm(K, device=samples.device)
    spread = (samples - samples[:, idx]).abs().mean(dim=(1, 2))  # (B,)

    # afCRPS: alpha * (mae - 0.5*spread) + (1-alpha) * mae
    crps = alpha * (mae - 0.5 * spread) + (1.0 - alpha) * mae
    return crps.mean()


def compute_trajectory_loss(
    trajectory: torch.Tensor,
    future_01: torch.Tensor,
    lambda_vs: float = 0.03,
    loss_type: str = "es",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-step loss summed over all time steps.

    Args:
        trajectory: (B, K, N, 5, 5) generated
        future_01: (B, N, 5, 5) ground truth
        loss_type: "es" for Energy Score, "afcrps" for almost-fair CRPS
    """
    B, K, N = trajectory.shape[:3]
    total_main = 0.0
    total_vs = 0.0

    for t in range(N):
        samples_t = trajectory[:, :, t].reshape(B, K, -1)  # (B, K, 25)
        gt_t = future_01[:, t].reshape(B, -1)  # (B, 25)
        if loss_type == "afcrps":
            total_main = total_main + afcrps_per_step(samples_t, gt_t)
        else:
            total_main = total_main + energy_score(samples_t, gt_t)
        total_vs = total_vs + variogram_score(samples_t, gt_t, p=0.5)

    loss = total_main + lambda_vs * total_vs
    metrics = {
        "main": float((total_main / N).detach().item()),
        "vs": float((total_vs / N).detach().item()),
        "loss": float((loss / N).detach().item()),
    }
    return loss, metrics


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[FactorARModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = FactorARModel(
        n_cells=cfg["n_cells"],
        factor_rank=cfg["factor_rank"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        decoder_hidden=cfg["decoder_hidden"],
        rho=cfg["rho"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
        pos_embed_dim=cfg["pos_embed_dim"],
        noise_skip=cfg.get("noise_skip", False),
        d_scale=cfg.get("d_scale", 1.0),
        cell_spread=cfg.get("cell_spread", False),
        decoder_layers=cfg.get("decoder_layers", 2),
        no_tanh=cfg.get("no_tanh", False),
        use_scale_anchor=cfg.get("use_scale_anchor", False),
        scale_anchor_alpha=cfg.get("scale_anchor_alpha", 0.50),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="227a: End-to-end factor-structured AR")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    parser.add_argument("--warmstart_encoder", type=str, default=None)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_vs", type=float, default=0.03)
    parser.add_argument("--factor_rank", type=int, default=6)
    parser.add_argument("--decoder_hidden", type=int, default=128)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--pos_embed_dim", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=48)
    parser.add_argument("--noise_skip", action="store_true",
                        help="Add noise skip connection (direct z_i → output bypass)")
    parser.add_argument("--d_scale", type=float, default=1.0,
                        help="Scale D_init by this factor (>1 = more idiosyncratic)")
    parser.add_argument("--cell_spread", action="store_true",
                        help="Condition-dependent per-cell scaling (like 97a)")
    parser.add_argument("--decoder_layers", type=int, default=2,
                        help="Number of hidden layers in factor/idio heads")
    parser.add_argument("--loss_type", type=str, default="es", choices=["es", "afcrps"],
                        help="Main loss: 'es' (Energy Score) or 'afcrps' (almost-fair CRPS)")
    parser.add_argument("--no_tanh", action="store_true",
                        help="Remove tanh from decoder output layers (unbounded outputs)")
    parser.add_argument("--use_scale_anchor", action="store_true",
                        help="Enable anchored EWMA: blend self-fed scale with initial anchor")
    parser.add_argument("--scale_anchor_alpha", type=float, default=0.50,
                        help="Anchor blend weight (0=pure self-fed, 1=pure anchor)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to fine-tune from (loads full model state)")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze factor/idio heads and noise_skip; only GRU and cell_spread learn")
    parser.add_argument("--checkpoint_every_early", type=int, default=3,
                        help="Checkpoint frequency for first --checkpoint_late_after epochs")
    parser.add_argument("--checkpoint_every_late", type=int, default=3,
                        help="Checkpoint frequency after --checkpoint_late_after epochs")
    parser.add_argument("--checkpoint_late_after", type=int, default=0,
                        help="Epoch after which to switch to late checkpoint frequency")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = FactorARModel(
        n_cells=25,
        factor_rank=args.factor_rank,
        hidden_dim=128,
        gru_layers=2,
        gru_dropout=0.1,
        decoder_hidden=args.decoder_hidden,
        rho=args.rho,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=True,
        pos_embed_dim=args.pos_embed_dim,
        noise_skip=args.noise_skip,
        d_scale=args.d_scale,
        cell_spread=args.cell_spread,
        decoder_layers=args.decoder_layers,
        no_tanh=args.no_tanh,
        use_scale_anchor=args.use_scale_anchor,
        scale_anchor_alpha=args.scale_anchor_alpha,
    ).to(device)

    # Initialize weights
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"])
        print(f"Resumed from {args.resume} (epoch {resume_payload.get('epoch', '?')})")
    else:
        # PCA init (only for fresh models)
        pca = np.load(args.pca_init)
        model.init_from_pca(pca["lambda_init"], pca["d_init"])
        print(f"PCA init: rank={pca['factor_rank']}, var={pca['explained_variance_ratio'][:int(pca['factor_rank'])].sum():.3f}")

        # Warm-start GRU encoder
        if args.warmstart_encoder:
            ws = torch.load(args.warmstart_encoder, map_location=device, weights_only=False)
            gru_keys = {k: v for k, v in ws["model_state_dict"].items() if k.startswith("gru.")}
            model.load_state_dict(gru_keys, strict=False)
            print(f"Warm-started GRU from {args.warmstart_encoder} ({len(gru_keys)} params)")

    # Freeze decoder if requested (only GRU/cell_spread learn)
    if args.freeze_decoder:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("gru.", "gru_cell.")) or name.startswith("cell_spread"):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen += 1
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Decoder frozen: {frozen} params frozen, {trainable} trainable")

    # Data: multi-step windows
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[:args.max_train_windows]

    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.n_steps
    )
    train_future = train_future.view(train_hist.shape[0], args.n_steps, 5, 5)

    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.n_steps
    )
    val_future = val_future.view(val_hist.shape[0], args.n_steps, 5, 5)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "type": "factor_ar_227a",
        "n_cells": 25,
        "factor_rank": args.factor_rank,
        "hidden_dim": 128,
        "gru_layers": 2,
        "gru_dropout": 0.1,
        "decoder_hidden": args.decoder_hidden,
        "rho": args.rho,
        "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor,
        "include_scale_feature": True,
        "pos_embed_dim": args.pos_embed_dim,
        "lambda_vs": args.lambda_vs,
        "n_steps": args.n_steps,
        "n_members": args.n_members,
        "noise_skip": args.noise_skip,
        "d_scale": args.d_scale,
        "cell_spread": args.cell_spread,
        "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type,
        "no_tanh": args.no_tanh,
        "use_scale_anchor": args.use_scale_anchor,
        "scale_anchor_alpha": args.scale_anchor_alpha,
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n227a Factor AR Model")
    print(f"  factor_rank={args.factor_rank}, decoder_hidden={args.decoder_hidden}")
    print(f"  rho={args.rho}, lambda_vs={args.lambda_vs}")
    print(f"  n_steps={args.n_steps}, n_members={args.n_members}, batch_size={args.batch_size}")
    print(f"  train={train_hist.shape[0]}, val={val_hist.shape[0]}")
    print(f"  params: {n_params:,}")

    history_log: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running = {"loss": 0.0, "main": 0.0, "vs": 0.0}
        count = 0

        for history_01, future_01 in train_loader:
            trajectory = model(history_01, n_members=args.n_members, n_steps=args.n_steps)
            loss, metrics = compute_trajectory_loss(trajectory, future_01, lambda_vs=args.lambda_vs, loss_type=args.loss_type)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = history_01.shape[0]
            running["loss"] += metrics["loss"] * b
            running["main"] += metrics["main"] * b
            running["vs"] += metrics["vs"] * b
            count += b

        train_metrics = {k: v / count for k, v in running.items()}

        # Validation
        model.eval()
        val_main_sum, val_vs_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt = model(vh, n_members=args.n_members, n_steps=args.n_steps)
                _, vm = compute_trajectory_loss(vt, vf, lambda_vs=args.lambda_vs, loss_type=args.loss_type)
                vb = vh.shape[0]
                val_main_sum += vm["main"] * vb
                val_vs_sum += vm["vs"] * vb
                val_count += vb

        val_main = val_main_sum / val_count
        val_vs = val_vs_sum / val_count
        val_loss = val_main + args.lambda_vs * val_vs

        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "val_main": val_main, "val_vs": val_vs, "val_loss": val_loss,
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")

        ckpt_freq = args.checkpoint_every_early if epoch <= args.checkpoint_late_after else args.checkpoint_every_late
        if epoch % ckpt_freq == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        if epoch % 2 == 0 or epoch == 1:
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={train_metrics['loss']:.4f} {args.loss_type}={train_metrics['main']:.4f} VS={train_metrics['vs']:.4f} "
                f"val={val_loss:.4f} "
                f"({elapsed:.1f}s)"
            )

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_loss": val_loss,
    }, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
