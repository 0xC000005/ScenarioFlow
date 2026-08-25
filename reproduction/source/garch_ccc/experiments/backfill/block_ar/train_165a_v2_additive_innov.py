#!/usr/bin/env python
"""
165a_v2: Additive Innovation Decomposition (RC21 H1v2)

Fixes 165a's fatal flaw: hard zero-mean centering killed member persistence (kurtosis
75→20). H1v2 preserves baseline additive recurrence while adding MSE centering signal.

Architecture (additive innovation — preserves prev_k carry):
  mean_delta = mean_head(cond, prev_mean)              # shared centering signal
  innov_k = tanh(decoder(cond_K, prev_k, z_k))         # per-member innovation
  frame_k = prev_k + mean_delta + innov_k               # additive carry preserved

Key difference from 165a:
  165a:    frame_k = mean_pred + (resid_k - resid_mean)  # re-anchored, kills persistence
  165a_v2: frame_k = prev_k + mean_delta + innov_k       # additive, preserves drift

No soft coupling — identifiability test. If decoder steals centering, add coupling in v3.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_165a_v2_additive_innov.py \
        --base_model models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt \
        --epochs 80 --batch_size 16 --n_members 16 --noise_dim 32 \
        --lambda_vs 0.5 --lambda_is 0.05 --is_warmup_epochs 10 --bptt_steps 5 \
        --lambda_floor 2.5 --floor_tau 0.005 --floor_warmup_epochs 10 \
        --lambda_mean 1.0 --mean_warmup_epochs 10 \
        --output_dir models/backfill/afcrps_165a --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig


def normalize_iv(surfaces):
    return surfaces * 2.0 - 1.0


def denormalize_iv(surfaces):
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


# ---- Per-Cell No-LN Conditional Norm (from 159a, FCN3 pattern) ----

class ConditionalNorm(nn.Module):
    """Per-cell No-LN CLN: y_c = (scale_c(z) + 1) * x_c + bias_c(z). Zero-init."""

    def __init__(self, d_model, noise_dim, n_cells=25):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.scale_proj = nn.Linear(noise_dim, n_cells * d_model)
        self.bias_proj = nn.Linear(noise_dim, n_cells * d_model)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, x, z):
        B = z.shape[0]
        scale = self.scale_proj(z).view(B, self.n_cells, self.d_model)
        bias = self.bias_proj(z).view(B, self.n_cells, self.d_model)
        return (scale + 1.0) * x + bias


# ---- Spatial Transformer Decoder ----

class SpatialTransformerDecoder(nn.Module):
    """Per-frame spatial transformer: attention over 25 cells.
    No LayerNorm (FCN3), ConditionalNorm, LayerScale 0.1, He init, zero-init output.
    """

    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.noise_dim = noise_dim

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.SiLU(), nn.Linear(d_model, noise_dim),
        )

        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cln': ConditionalNorm(d_model, noise_dim, n_cells),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'ff_cln': ConditionalNorm(d_model, noise_dim, n_cells),
            }))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.output_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._he_init()

    def _he_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'output_proj' in name or 'scale_proj' in name or 'bias_proj' in name:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cond, prev_frame, noise):
        B = cond.shape[0]
        h = self.input_proj(prev_frame.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos
        z = self.noise_proj(noise)

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[li * 2]
            ls_f = self.ls_params[li * 2 + 1]
            h_norm = layer['cln'](h, z)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer['ff'](layer['ff_cln'](h, z))

        delta = self.output_proj(h).squeeze(-1)
        return delta


# ---- Mean Head (RC21 H1) ----

class MeanHead(nn.Module):
    """Deterministic mean prediction: small MLP predicting per-cell delta.

    Input: condition (128-d) concatenated with previous mean frame (25-d)
    Output: mean_delta (25-d) -- expected change per cell
    Zero-init output so mean_pred starts as prev_mean (identity at init).
    """

    def __init__(self, cond_dim=128, n_cells=25, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + n_cells, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_cells),
        )
        # Zero-init output layer: mean_delta = 0 at init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, prev_mean):
        """
        cond: (B, cond_dim)
        prev_mean: (B, n_cells)
        Returns: mean_delta (B, n_cells)
        """
        return self.net(torch.cat([cond, prev_mean], dim=-1))


# ---- Loss functions ----

def afcrps_per_frame(samples, gt, alpha=0.95, spread_weight=0.5):
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()
    fcrps = mae - spread_weight * spread
    loss = alpha * fcrps + (1 - alpha) * mae
    return loss, mae, spread


def variogram_score_per_frame(samples, gt, p=0.5):
    B, K, C = samples.shape
    eps = 1e-8
    s_diff = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    s_mean = s_diff.mean(dim=1)
    g_diff = (gt.unsqueeze(-1) - gt.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    loss = (g_diff - s_mean).pow(2)
    mask = torch.triu(torch.ones(C, C, device=samples.device), diagonal=1).bool()
    return loss[:, mask].mean()


def interval_score(samples, gt, alpha=0.1):
    lo = samples.quantile(alpha / 2, dim=1)
    hi = samples.quantile(1 - alpha / 2, dim=1)
    width = hi - lo
    return (width + (2 / alpha) * (F.relu(lo - gt) + F.relu(gt - hi))).mean()


# ---- Reflecting boundary ----

def reflecting_boundary(x, lo=0.01, hi=1.0):
    below = x < lo
    above = x > hi
    x = torch.where(below, 2 * lo - x, x)
    x = torch.where(above, 2 * hi - x, x)
    return x.clamp(lo, hi)


# ---- Model ----

class ARMeanResidualModel(nn.Module):
    """GRU encoder + mean head + spatial transformer decoder with zero-mean residuals.

    The mean head predicts the deterministic expected trajectory.
    The decoder predicts noise-dependent residuals, centered to zero mean.
    Final frame = mean_pred + centered_residual.
    """

    def __init__(self, encoder_config, decoder_config, mean_head_config):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SpatialTransformerDecoder(**decoder_config)
        self.mean_head = MeanHead(**mean_head_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.mean_head_config = mean_head_config

    def encode(self, history):
        return self.encoder(history)

    def _compute_condition(self, gru_outputs):
        """Compute condition from GRU outputs via attention pooling + bottleneck."""
        attn_logits = self.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        return self.encoder.bottleneck(h_pooled)

    def sample_batched(self, history, n_samples=50, **kwargs):
        """Generate samples with mean+residual decomposition.

        Args:
            history: (B, T_hist, 5, 5) in [-1, 1] (normalized)
        Returns:
            samples: (B, n_samples, T_future, 5, 5) in [0, 1]
        """
        B = history.shape[0]
        T = 30
        device = history.device
        noise_dim = self.decoder.noise_dim
        C = 25
        CHUNK = 10

        with torch.no_grad():
            last_frame = denormalize_iv(history[:, -1]).reshape(B, C)
            hist_flat = history.reshape(B, history.shape[1], -1)
            gru_outputs_base, h_last_base = self.encoder.gru(hist_flat)

            # --- Mean path (deterministic, at B level) ---
            # Precompute mean_deltas for all T steps
            mean_cond = self._compute_condition(gru_outputs_base)
            mean_gru_state = h_last_base.contiguous()
            mean_gru_outs = gru_outputs_base.clone()
            prev_mean = last_frame.clone()
            mean_deltas = []  # (B, C) per step

            for t in range(T):
                if t > 0:
                    mean_cond = self._compute_condition(mean_gru_outs)

                mean_delta = self.mean_head(mean_cond, prev_mean)
                mean_deltas.append(mean_delta)
                mean_pred = prev_mean + mean_delta

                # GRU feedback for mean path
                mean_frame_norm = normalize_iv(mean_pred).unsqueeze(1)
                mean_gru_out, mean_gru_state = self.encoder.gru(
                    mean_frame_norm, mean_gru_state
                )
                mean_gru_outs = torch.cat([mean_gru_outs, mean_gru_out], dim=1)
                prev_mean = mean_pred

            # --- Member path (stochastic, per chunk, additive innovation) ---
            all_samples = []
            for start in range(0, n_samples, CHUNK):
                k = min(CHUNK, n_samples - start)
                Bk = B * k

                last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(Bk, C)
                gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                    B, k, -1, -1).reshape(Bk, -1, self.encoder_config.gru_hidden_dim)
                h_last_k = h_last_base.unsqueeze(2).expand(
                    1, B, k, -1).reshape(1, Bk, -1).contiguous()

                cond_k = self._compute_condition(gru_out_k)
                prev_k = last_k
                chunk_frames = []

                for t in range(T):
                    if t > 0:
                        cond_k = self._compute_condition(gru_out_k)

                    z_t = torch.randn(Bk, noise_dim, device=device)
                    innov = torch.tanh(self.decoder(cond_k, prev_k, z_t))  # (Bk, C)

                    # Additive innovation: prev_k + mean_delta + innov_k
                    mean_delta_t = mean_deltas[t].unsqueeze(1).expand(B, k, -1).reshape(Bk, C)
                    frame_flat = prev_k + mean_delta_t + innov  # additive carry
                    frame_flat = reflecting_boundary(frame_flat)

                    chunk_frames.append(frame_flat.reshape(B, k, C))

                    # GRU feedback
                    fn = normalize_iv(frame_flat).unsqueeze(1)
                    go, h_last_k = self.encoder.gru(fn, h_last_k)
                    gru_out_k = torch.cat([gru_out_k, go], dim=1)
                    prev_k = frame_flat

                chunk_frames = torch.stack(chunk_frames, dim=2)  # (B, k, T, C)
                all_samples.append(chunk_frames.reshape(B, k, T, 5, 5))

            samples = torch.cat(all_samples, dim=1)
        return samples


# ---- Training ----

def main():
    parser = argparse.ArgumentParser(description="165a: Mean+Residual Decomposition (RC21 H1)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Pretrained model to warm-start from (encoder + decoder)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_mean_head", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_members", type=int, default=16,
                        help="K ensemble members")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5)
    parser.add_argument("--lambda_vs", type=float, default=0.5)
    parser.add_argument("--lambda_is", type=float, default=0.05)
    parser.add_argument("--is_warmup_epochs", type=int, default=10)
    parser.add_argument("--bptt_steps", type=int, default=5)
    parser.add_argument("--lambda_floor", type=float, default=2.5)
    parser.add_argument("--floor_tau", type=float, default=0.005)
    parser.add_argument("--floor_warmup_epochs", type=int, default=10)
    # Mean head specific
    parser.add_argument("--lambda_mean", type=float, default=1.0,
                        help="Weight for Huber loss on mean_pred")
    parser.add_argument("--mean_warmup_epochs", type=int, default=10,
                        help="Phase 1: freeze decoder, train mean head MSE only")
    parser.add_argument("--mean_hidden", type=int, default=128,
                        help="Hidden dim for mean head MLP")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)
    val_indices = np.arange(max_train_idx - VAL_SIZE, max_train_idx)

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ---- Model ----
    encoder_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=C, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, cond_dim=128, noise_dim=args.noise_dim,
    )
    mean_head_config = dict(
        cond_dim=128, n_cells=C, hidden_dim=args.mean_hidden,
    )

    model = ARMeanResidualModel(encoder_config, decoder_config, mean_head_config).to(device)

    # ---- Load pretrained weights (encoder + decoder) ----
    if args.base_model:
        print(f"\nLoading pretrained weights from {args.base_model}")
        ckpt = torch.load(args.base_model, weights_only=False, map_location=device)
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys (expected — mean_head is new): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            mean_keys = [k for k in missing if 'mean_head' in k]
            other_keys = [k for k in missing if 'mean_head' not in k]
            print(f"  Mean head keys (zero-init): {len(mean_keys)}")
            if other_keys:
                print(f"  WARNING — non-mean-head missing keys: {other_keys}")

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_mean = sum(p.numel() for p in model.mean_head.parameters())
    print(f"\n{'='*60}")
    print(f"165a: Mean+Residual Decomposition (RC21 H1)")
    print(f"{'='*60}")
    print(f"  Encoder: {n_enc:,} params")
    print(f"  Decoder: {n_dec:,} params")
    print(f"  Mean head: {n_mean:,} params (NEW)")
    print(f"  Total: {n_enc + n_dec + n_mean:,} params")
    print(f"  K={args.n_members}, noise_dim={args.noise_dim}")
    print(f"  Phase 1 (ep 1-{args.mean_warmup_epochs}): decoder FROZEN, mean head MSE only")
    print(f"  Phase 2 (ep {args.mean_warmup_epochs+1}-{args.epochs}): joint MSE + CRPS + VS + IS + floor")
    print(f"  lambda_mean={args.lambda_mean}, lambda_floor={args.lambda_floor}")

    # ---- Data loading ----
    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    def build_windows(indices, surf):
        idx = torch.from_numpy(indices).long()
        offsets_h = torch.arange(H, device=surf.device).unsqueeze(0)
        offsets_f = torch.arange(T, device=surf.device).unsqueeze(0)
        hist_idx = idx.to(surf.device).unsqueeze(1) + offsets_h
        fut_idx = idx.to(surf.device).unsqueeze(1) + H + offsets_f
        hist = surf[hist_idx]
        future = surf[fut_idx].reshape(len(indices), T, C)
        last_frame = surf[idx.to(surf.device) + H - 1].reshape(len(indices), C)
        return hist, future, last_frame

    train_hist, train_future, train_last = build_windows(train_indices, surf_tensor)
    val_hist, val_future, val_last = build_windows(val_indices, surf_tensor)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future, train_last),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future, val_last),
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": 0.01},
        {"params": model.mean_head.parameters(), "lr": args.lr_mean_head, "weight_decay": 0.01},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    config_dict = {
        "type": "ar_spatial_transformer_165a_v2_additive_innov",
        "encoder": {
            "input_dim": 25, "gru_hidden_dim": 64,
            "bottleneck_dim": 128, "dropout": 0.1,
        },
        "decoder": decoder_config,
        "mean_head": mean_head_config,
        "n_members": args.n_members,
        "alpha": args.alpha, "spread_weight": args.spread_weight,
        "lambda_vs": args.lambda_vs, "lambda_is": args.lambda_is,
        "lambda_mean": args.lambda_mean,
        "n_frames": T, "n_cells": C,
        "train_windows": len(train_indices),
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        is_phase1 = epoch <= args.mean_warmup_epochs

        # Phase control: freeze/unfreeze decoder
        if is_phase1:
            model.decoder.requires_grad_(False)
        elif epoch == args.mean_warmup_epochs + 1:
            model.decoder.requires_grad_(True)
            # Save Phase 1 checkpoint before Phase 2 overwrites best_model
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch - 1, "val_loss": best_val,
                "config": config_dict,
                "phase": "phase1_final",
            }, f"{args.output_dir}/phase1_model.pt")
            best_val = float("inf")  # Reset: Phase 1 uses MSE, Phase 2 uses CRPS
            print(f"\n>>> Phase 2: decoder UNFROZEN at epoch {epoch}, best_val reset")

        ep_loss = 0; ep_mae = 0; ep_spread = 0; ep_vs = 0; ep_is = 0
        ep_floor = 0; ep_mean_mse = 0; nb = 0

        # Warmup schedules
        lambda_is_eff = args.lambda_is * min(1.0, epoch / max(1, args.is_warmup_epochs)) if args.is_warmup_epochs > 0 else args.lambda_is
        lambda_floor_eff = args.lambda_floor * min(1.0, epoch / max(1, args.floor_warmup_epochs)) if args.floor_warmup_epochs > 0 else args.lambda_floor

        for hist, gt_frames, last_frame in train_loader:
            B = hist.shape[0]
            K = args.n_members

            # Encode history (with gradients for E2E)
            hist_norm = normalize_iv(hist)
            hist_flat = hist_norm.reshape(B, H, C)
            gru_outputs, h_last = model.encoder.gru(hist_flat)

            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)  # (B, 128)

            # --- Mean path state (B level) ---
            prev_mean = last_frame.clone()  # (B, C)
            mean_h_gru = h_last.detach().contiguous()  # (1, B, 64)
            mean_gru_outs = gru_outputs.detach().clone()  # (B, H, 64)
            mean_cond = cond  # initial condition WITH gradients

            if not is_phase1:
                # --- Member path state (B*K level) ---
                cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
                prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)
                h_gru = h_last.detach().unsqueeze(2).expand(1, B, K, -1).reshape(1, B * K, -1).contiguous()
                gru_outs = gru_outputs.detach().unsqueeze(1).expand(B, K, H, -1).reshape(B * K, H, -1)

            N = args.bptt_steps
            optimizer.zero_grad()
            total_loss_val = 0; total_mae = 0; total_spread = 0; total_vs = 0
            window_loss = 0

            for t in range(T):
                # Recompute mean condition (B level)
                if t > 0:
                    mean_attn = F.softmax(model.encoder.attn_proj(mean_gru_outs).squeeze(-1), dim=1)
                    mean_h = (mean_attn.unsqueeze(-1) * mean_gru_outs).sum(dim=1)
                    mean_cond = model.encoder.bottleneck(mean_h)

                # Mean prediction
                mean_delta = model.mean_head(mean_cond, prev_mean)  # (B, C)
                mean_pred = prev_mean + mean_delta  # (B, C)

                # MSE loss on mean prediction
                gt_t = gt_frames[:, t, :]  # (B, C)
                loss_mean = F.huber_loss(mean_pred, gt_t)

                if is_phase1:
                    # Phase 1: only mean head MSE
                    step_loss = args.lambda_mean * loss_mean / T
                    window_loss = window_loss + step_loss
                    total_loss_val += step_loss.item()
                    ep_mean_mse += loss_mean.item()
                else:
                    # Phase 2: joint training
                    # Recompute member condition (B*K level)
                    if t > 0:
                        attn_logits_t = model.encoder.attn_proj(gru_outs).squeeze(-1)
                        attn_weights_t = F.softmax(attn_logits_t, dim=1)
                        h_pooled_t = (attn_weights_t.unsqueeze(-1) * gru_outs).sum(dim=1)
                        cond_K = model.encoder.bottleneck(h_pooled_t)

                    z_t = torch.randn(B * K, args.noise_dim, device=device)
                    innov = torch.tanh(model.decoder(cond_K, prev, z_t))  # (B*K, C)

                    # Additive innovation: frame_k = prev_k + mean_delta + innov_k
                    mean_delta_expand = mean_delta.unsqueeze(1).expand(B, K, C).reshape(B * K, C)
                    frame_t = prev + mean_delta_expand + innov  # additive carry
                    frame_BK = frame_t.reshape(B, K, C)

                    # Losses on frame_BK
                    loss_crps, mae_t, spread_t = afcrps_per_frame(
                        frame_BK, gt_t, alpha=args.alpha, spread_weight=args.spread_weight
                    )
                    is_t = interval_score(frame_BK, gt_t)
                    vs_t = variogram_score_per_frame(frame_BK, gt_t) if args.lambda_vs > 0 else torch.tensor(0.0, device=device)
                    floor_barrier = args.floor_tau * F.softplus(-frame_t / args.floor_tau).mean()

                    step_loss = (
                        args.lambda_mean * loss_mean
                        + loss_crps
                        + lambda_is_eff * is_t
                        + args.lambda_vs * vs_t
                        + lambda_floor_eff * floor_barrier
                    ) / T
                    window_loss = window_loss + step_loss

                    total_loss_val += step_loss.item()
                    total_mae += mae_t.item()
                    total_spread += spread_t.item()
                    total_vs += vs_t.item() if args.lambda_vs > 0 else 0
                    ep_is += is_t.item()
                    ep_floor += floor_barrier.item()
                    ep_mean_mse += loss_mean.item()

                # BPTT window boundary
                is_window_end = ((t + 1) % N == 0) or (t == T - 1)
                if is_window_end:
                    window_loss.backward()
                    window_loss = 0
                    prev_mean = mean_pred.detach()
                    mean_h_gru = mean_h_gru.detach()
                    mean_gru_outs = mean_gru_outs.detach()
                    if not is_phase1:
                        prev = frame_t.detach()
                        h_gru = h_gru.detach()
                        gru_outs = gru_outs.detach()
                else:
                    prev_mean = mean_pred  # keep gradient for BPTT
                    if not is_phase1:
                        prev = frame_t

                # GRU feedback for mean path
                mean_frame_norm = normalize_iv(mean_pred.detach()).unsqueeze(1)
                mean_gru_out, mean_h_gru = model.encoder.gru(mean_frame_norm, mean_h_gru)
                mean_gru_outs = torch.cat([mean_gru_outs, mean_gru_out], dim=1)

                # GRU feedback for member path
                if not is_phase1:
                    frame_norm = normalize_iv(prev.detach()).unsqueeze(1)
                    gru_out, h_gru = model.encoder.gru(frame_norm, h_gru)
                    gru_outs = torch.cat([gru_outs, gru_out], dim=1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += total_loss_val
            ep_mae += total_mae / T if not is_phase1 else 0
            ep_spread += total_spread / T if not is_phase1 else 0
            ep_vs += total_vs / T if not is_phase1 else 0
            nb += 1

        scheduler.step()
        tl = ep_loss / nb
        tm = ep_mae / nb
        ts = ep_spread / nb
        tvs = ep_vs / nb
        tis = ep_is / nb
        t_mean_mse = ep_mean_mse / nb
        elapsed = time.time() - t0

        # ---- Validation ----
        model.eval()
        vl = 0; v_mean = 0; nv = 0
        with torch.no_grad():
            for v_hist, v_gt, v_last in val_loader:
                B2 = v_hist.shape[0]; K2 = args.n_members
                vh_norm = normalize_iv(v_hist)
                vh_flat = vh_norm.reshape(B2, H, C)
                v_gru_outs, v_h = model.encoder.gru(vh_flat)
                v_cond = model._compute_condition(v_gru_outs)

                # Mean path
                v_prev_mean = v_last.clone()
                v_mean_h = v_h.contiguous()
                v_mean_gru = v_gru_outs.clone()
                v_mean_cond = v_cond

                if not is_phase1:
                    # Member path
                    v_cond_K = v_cond.unsqueeze(1).expand(B2, K2, -1).reshape(B2 * K2, -1)
                    v_prev = v_last.unsqueeze(1).expand(B2, K2, C).reshape(B2 * K2, C)
                    v_h_k = v_h.unsqueeze(2).expand(1, B2, K2, -1).reshape(1, B2*K2, -1).contiguous()
                    v_gru_k = v_gru_outs.unsqueeze(1).expand(B2, K2, H, -1).reshape(B2*K2, H, -1)

                v_crps = 0; v_mse = 0
                for t in range(T):
                    if t > 0:
                        v_mean_cond = model._compute_condition(v_mean_gru)

                    v_mean_delta = model.mean_head(v_mean_cond, v_prev_mean)
                    v_mean_pred = v_prev_mean + v_mean_delta
                    v_mse += F.mse_loss(v_mean_pred, v_gt[:, t]).item()

                    if not is_phase1:
                        if t > 0:
                            va = F.softmax(model.encoder.attn_proj(v_gru_k).squeeze(-1), dim=1)
                            v_cond_K = model.encoder.bottleneck((va.unsqueeze(-1) * v_gru_k).sum(dim=1))
                        z_t = torch.randn(B2 * K2, args.noise_dim, device=device)
                        innov = torch.tanh(model.decoder(v_cond_K, v_prev, z_t))

                        # Additive innovation
                        v_md_expand = v_mean_delta.unsqueeze(1).expand(B2, K2, C).reshape(B2*K2, C)
                        frame_flat = v_prev + v_md_expand + innov
                        frame_BK = frame_flat.reshape(B2, K2, C)

                        crps_t, _, _ = afcrps_per_frame(frame_BK, v_gt[:, t])
                        v_crps += crps_t.item()

                        # GRU feedback for members
                        fn = normalize_iv(frame_flat).unsqueeze(1)
                        go, v_h_k = model.encoder.gru(fn, v_h_k)
                        v_gru_k = torch.cat([v_gru_k, go], dim=1)
                        v_prev = frame_flat

                    # GRU feedback for mean path
                    mfn = normalize_iv(v_mean_pred).unsqueeze(1)
                    mgo, v_mean_h = model.encoder.gru(mfn, v_mean_h)
                    v_mean_gru = torch.cat([v_mean_gru, mgo], dim=1)
                    v_prev_mean = v_mean_pred

                if is_phase1:
                    vl += (v_mse / T) * B2
                else:
                    # Phase 2 val includes MSE (Codex: checkpoint selection must match training)
                    vl += (v_crps / T + args.lambda_mean * v_mse / T) * B2
                v_mean += (v_mse / T) * B2
                nv += B2

        val_loss = vl / max(nv, 1)
        val_mean_mse = v_mean / max(nv, 1)

        # ---- Save best model ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "config": config_dict,
            }, f"{args.output_dir}/best_model.pt")

        # ---- Success checks (H1v2 identifiability) ----
        mh_weight_norm = sum(p.norm().item() for p in model.mean_head.parameters())

        # ---- Logging ----
        phase_str = "P1:MSE" if is_phase1 else "P2:JOINT"
        log_line = (
            f"Ep {epoch:3d} [{phase_str}]  loss={tl:.4f}  val={val_loss:.4f}  "
            f"mean_mse={t_mean_mse:.4f}  val_mean_mse={val_mean_mse:.4f}  "
            f"mh_wnorm={mh_weight_norm:.3f}"
        )
        if not is_phase1:
            tfloor = ep_floor / nb
            log_line += (
                f"  mae={tm:.4f}  spread={ts:.4f}  vs={tvs:.6f}  "
                f"is={tis:.4f}  floor={tfloor:.6f}"
            )
        log_line += f"  ({elapsed:.1f}s)"
        if val_loss <= best_val:
            log_line += "  *best"
        print(log_line, flush=True)  # flush=True to avoid buffering issue from 165a

        history.append({
            "epoch": epoch, "phase": "P1" if is_phase1 else "P2",
            "train_loss": tl, "val_loss": val_loss,
            "mean_mse": t_mean_mse, "val_mean_mse": val_mean_mse,
            "mae": tm, "spread": ts, "vs": tvs,
            "mh_weight_norm": mh_weight_norm,
        })

    # ---- Save final ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs, "val_loss": val_loss,
        "config": config_dict,
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
