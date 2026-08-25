#!/usr/bin/env python
"""
155d: Factored Transformer + CLN + afCRPS — No ODE (RC17-H5)

Single-pass: (condition, noise_z) → residual via factored attention with CLN.
No ODE. Combines:
- Factored attention (153a's correlation structure)
- CLN noise per block (Latte design, multiplicative, survives attention)
- afCRPS loss (calibration gradient from 155a)
- Direct generation (no ODE contraction from 155c)

The hypothesis: CLN in factored attention produces diverse, correlated
predictions. Without ODE, CLN noise goes directly to output.

Literature: Latte (2401.03048), FGN (2506.10772), AIFS-CRPS (2412.15832)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_155d_cln_transformer.py \
        --base_model models/backfill/flow_153a/final_model.pt \
        --epochs 200 --batch_size 32 --n_members 8 --noise_dim 32 \
        --output_dir models/backfill/flow_155d --device cuda
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
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    load_encoder, normalize_iv, make_serializable
)


class ConditionalLayerNorm(nn.Module):
    """CLN: y = (scale(z) + 1) * LayerNorm(x) + bias(z). Zero-init."""

    def __init__(self, d_model, noise_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scale_proj = nn.Linear(noise_dim, d_model)
        self.bias_proj = nn.Linear(noise_dim, d_model)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, x, z):
        """x: (..., d_model), z: (B, noise_dim)"""
        h = self.norm(x)
        scale = self.scale_proj(z)
        bias = self.bias_proj(z)
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(-2)
            bias = bias.unsqueeze(-2)
        return (scale + 1.0) * h + bias


class CLNResidualTransformer(nn.Module):
    """Factored attention transformer with CLN for residual generation.

    Single-pass: condition → tokens → factored attention with CLN(z) → residual.
    No flow matching, no ODE. The condition creates structure, CLN creates diversity.
    """

    def __init__(self, n_frames=30, n_cells=25, d_model=128, n_heads=4,
                 n_layers=4, cond_dim=128, noise_dim=32):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.d_model = d_model
        self.noise_dim = noise_dim

        # Condition → per-token features (broadcast to all T*C positions)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Learned positional encodings
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_cells, d_model) * 0.02)

        # Noise embedding: noise_dim → internal noise representation
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.SiLU(), nn.Linear(d_model, noise_dim),
        )

        # Factored attention blocks with CLN (Latte design: independent CLN per block)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'temp_cln': ConditionalLayerNorm(d_model, noise_dim),
                'temp_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'temp_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'temp_ff_cln': ConditionalLayerNorm(d_model, noise_dim),
                'spat_cln': ConditionalLayerNorm(d_model, noise_dim),
                'spat_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'spat_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'spat_ff_cln': ConditionalLayerNorm(d_model, noise_dim),
            }))

        # Output: per-token → scalar residual
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        # Zero-init: at start, residual = 0 → combined = base_pred
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, cond, noise_z):
        """
        Args:
            cond: (B, cond_dim) conditioning vector
            noise_z: (B, noise_dim) noise for CLN
        Returns:
            residual: (B, T*C) predicted residual
        """
        B = cond.shape[0]
        T, C = self.n_frames, self.n_cells

        # Project condition to per-token features
        c_emb = self.cond_proj(cond)  # (B, d_model)
        h = c_emb[:, None, None, :].expand(B, T, C, -1).clone()  # (B, T, C, d_model)

        # Add positional encoding
        h = h + self.temporal_pos + self.spatial_pos

        # Project noise for CLN
        z = self.noise_proj(noise_z)  # (B, noise_dim)

        # Factored attention with CLN
        for layer in self.layers:
            # Temporal attention
            h_temp = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
            z_temp = z.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
            h_norm = layer['temp_cln'](h_temp, z_temp)
            attn_out, _ = layer['temp_attn'](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer['temp_ff'](layer['temp_ff_cln'](h_temp, z_temp))
            h = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3)

            # Spatial attention
            h_spat = h.reshape(B * T, C, -1)
            z_spat = z.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            h_norm = layer['spat_cln'](h_spat, z_spat)
            attn_out, _ = layer['spat_attn'](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer['spat_ff'](layer['spat_ff_cln'](h_spat, z_spat))
            h = h_spat.reshape(B, T, C, -1)

        # Output projection
        h = self.output_norm(h)
        residual = self.output_proj(h).squeeze(-1)  # (B, T, C)
        return residual.reshape(B, T * C)


def afcrps_loss(samples, gt, alpha=0.95, spread_weight=0.5):
    """afCRPS for (B, K, D) tensors."""
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()
    fcrps = mae - spread_weight * spread
    loss = alpha * fcrps + (1 - alpha) * mae
    return loss, mae, spread


def interval_score(samples, gt, alpha=0.9):
    """Interval score for CI calibration."""
    lo = samples.quantile(alpha / 2, dim=1)
    hi = samples.quantile(1 - alpha / 2, dim=1)
    width = hi - lo
    return (width + (2 / alpha) * (F.relu(lo - gt) + F.relu(gt - hi))).mean()


def variogram_score(samples, gt, p=0.5):
    """Variogram score for cross-cell dependency structure.
    Args: samples (B, K, T, C), gt (B, T, C). Returns scalar."""
    B, K, T, C = samples.shape
    eps = 1e-8
    s_diff = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    g_diff = (gt.unsqueeze(-1) - gt.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    s_mean = s_diff.mean(dim=1)
    loss = (g_diff - s_mean).pow(2)
    mask = torch.triu(torch.ones(C, C, device=samples.device), diagonal=1).bool()
    return loss[:, :, mask].sum(dim=(-2, -1)).mean()


def per_frame_energy_score(samples, gt):
    """Per-frame Energy Score at D=25 (proper multivariate scoring rule).

    ES_t = E[||X_t - y_t||_2] - 0.5 * E[||X_t - X'_t||_2]

    Computed per frame (D=25 cells), then averaged over frames and batch.
    This provides a direct gradient on cross-cell correlation structure.

    Args: samples (B, K, T, C), gt (B, T, C). Returns scalar.
    """
    B, K, T, C = samples.shape
    # mae term: mean over K of L2 norm per frame
    # samples: (B, K, T, C), gt: (B, 1, T, C)
    diff = samples - gt.unsqueeze(1)  # (B, K, T, C)
    mae_term = diff.norm(dim=-1).mean(dim=1)  # (B, T) — L2 over C, mean over K

    # spread term: mean over K pairs of L2 norm per frame
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    pair_diff = samples[:, idx_i] - samples[:, idx_j]  # (B, n_pairs, T, C)
    spread_term = pair_diff.norm(dim=-1).mean(dim=1)  # (B, T)

    es = mae_term - 0.5 * spread_term  # (B, T)
    return es.mean()


def evaluate_model(model, base_preds, gt_futures, conditions,
                   n_samples=50, device='cuda'):
    """Full evaluation."""
    model.eval()
    N = len(base_preds)
    T, C = 30, 25

    all_samples = []
    with torch.no_grad():
        for i in range(N):
            cond = torch.from_numpy(conditions[i:i+1]).float().to(device).expand(n_samples, -1)
            noise = torch.randn(n_samples, model.noise_dim, device=device)
            residual = model(cond, noise).cpu().numpy()  # (K, 750)
            combined = np.clip(base_preds[i] + residual, 0, 1)
            all_samples.append(combined.reshape(n_samples, T, C))

    samples = np.array(all_samples)
    gt = gt_futures.reshape(N, T, C)

    worst_ci = 1.0
    for c in range(C):
        lo = np.percentile(samples[:, :, :, c], 5, axis=1)
        hi = np.percentile(samples[:, :, :, c], 95, axis=1)
        cov = ((gt[:, :, c] >= lo) & (gt[:, :, c] <= hi)).mean()
        worst_ci = min(worst_ci, cov)

    ci_h_pass = 0
    for h in range(T):
        lo = np.percentile(samples[:, :, h], 5, axis=1)
        hi = np.percentile(samples[:, :, h], 95, axis=1)
        if ((gt[:, h] >= lo) & (gt[:, h] <= hi)).mean() >= 0.85:
            ci_h_pass += 1

    gen_ch = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_ch = np.diff(gt, axis=1).reshape(-1, C)
    ks = sum(1 for c2 in range(C) if ks_2samp(gen_ch[:, c2], gt_ch[:, c2])[0] < 0.15)
    kr = kurtosis(gen_ch.flatten()) / (kurtosis(gt_ch.flatten()) + 1e-6)
    gc = np.corrcoef(gen_ch.T); gtc = np.corrcoef(gt_ch.T)
    corr = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)

    def eff_rank(corr_mat):
        ev = np.linalg.eigvalsh(corr_mat)[::-1]; ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    ss = samples.std(axis=1).mean() / (np.abs(samples.mean(axis=1) - gt).mean() + 1e-8)

    return {
        "ci_worst": float(worst_ci), "ci_h_pass": ci_h_pass,
        "ks": ks, "kurt": float(kr), "corr": float(corr),
        "eff_rank_ratio": float(eff_rank(gc) / (eff_rank(gtc) + 1e-6)),
        "ss": float(ss),
        "spread_h1": float(samples[:, :, 0].std(axis=1).mean()),
        "spread_h30": float(samples[:, :, -1].std(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--lambda_is", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5,
                        help="Spread term weight in afCRPS (default 0.5, try 1.0)")
    parser.add_argument("--lambda_vs", type=float, default=0.0,
                        help="Variogram score weight for cross-cell correlation")
    parser.add_argument("--lambda_es", type=float, default=0.0,
                        help="Per-frame Energy Score weight (D=25 multivariate scoring rule)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750

    # Load cached base predictions
    cached = np.load("models/backfill/flow_154b/base_predictions.npz")
    val_preds = cached["val_preds"]
    val_gts = cached["val_gts"]
    train_preds = cached["train_preds"]
    train_gts = cached["train_gts"]

    print("Computing encoder conditions...")
    val_conds = []
    with torch.no_grad():
        for i in range(4040, 4040 + len(val_preds)):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            val_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
    val_conds = np.concatenate(val_conds)
    train_conds = []
    with torch.no_grad():
        for i in range(3540, 3540 + len(train_preds)):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            train_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
    train_conds = np.concatenate(train_conds)

    # Create model
    model = CLNResidualTransformer(
        n_frames=T, n_cells=25, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, cond_dim=cond_dim, noise_dim=args.noise_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"155d: CLN Residual Transformer + afCRPS (RC17-H5, no ODE)")
    print(f"{'='*60}")
    print(f"  Parameters: {n_params:,}")
    print(f"  K={args.n_members}, noise_dim={args.noise_dim}, d_model={args.d_model}")
    print(f"  n_layers={args.n_layers}, n_heads={args.n_heads}")

    # Pre-load tensors
    bp_train = torch.from_numpy(val_preds).float().to(device)
    gt_train = torch.from_numpy(val_gts).float().to(device)
    cd_train = torch.from_numpy(val_conds).float().to(device)
    bp_val = torch.from_numpy(train_preds).float().to(device)
    gt_val = torch.from_numpy(train_gts).float().to(device)
    cd_val = torch.from_numpy(train_conds).float().to(device)

    train_loader = DataLoader(TensorDataset(torch.arange(len(val_preds))),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.arange(len(train_preds))),
                            batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0; ep_mae = 0; ep_spread = 0; ep_is = 0; nb = 0

        for (idx,) in train_loader:
            B = idx.shape[0]
            K = args.n_members
            bp = bp_train[idx]
            gt = gt_train[idx]
            cd = cd_train[idx]

            # Generate K members: fold into batch dim
            noise = torch.randn(B * K, args.noise_dim, device=device)
            cond_K = cd.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

            residual = model(cond_K, noise).reshape(B, K, DIM)
            combined = (bp.unsqueeze(1) + residual).clamp(0, 1)

            # Reshape for loss: (B, K, T, C)
            combined_4d = combined.reshape(B, K, T, 25)
            gt_4d = gt.reshape(B, T, 25)

            crps, mae, spread = afcrps_loss(combined_4d, gt_4d, alpha=args.alpha, spread_weight=args.spread_weight)
            is_loss = interval_score(combined_4d, gt_4d)
            # VS for cross-cell correlation
            vs_loss = variogram_score(combined_4d, gt_4d) if args.lambda_vs > 0 else torch.tensor(0.0, device=device)
            # Per-frame Energy Score for multivariate structure
            es_loss = per_frame_energy_score(combined_4d, gt_4d) if args.lambda_es > 0 else torch.tensor(0.0, device=device)
            loss = crps + args.lambda_is * is_loss + args.lambda_vs * vs_loss + args.lambda_es * es_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item(); ep_mae += mae.item()
            ep_spread += spread.item(); ep_is += is_loss.item()
            nb += 1

        scheduler.step()
        tl = ep_loss/nb; tm = ep_mae/nb; ts = ep_spread/nb; ti = ep_is/nb
        elapsed = time.time() - t0

        # Validation
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for (idx,) in val_loader:
                B = idx.shape[0]; K = args.n_members
                noise = torch.randn(B * K, args.noise_dim, device=device)
                cond_K = cd_val[idx].unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
                res = model(cond_K, noise).reshape(B, K, DIM)
                comb = (bp_val[idx].unsqueeze(1) + res).clamp(0, 1).reshape(B, K, T, 25)
                crps, _, _ = afcrps_loss(comb, gt_val[idx].reshape(B, T, 25), alpha=args.alpha, spread_weight=args.spread_weight)
                vl += crps.item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "n_frames": T, "n_cells": 25, "d_model": args.d_model,
                    "n_heads": args.n_heads, "n_layers": args.n_layers,
                    "cond_dim": cond_dim, "noise_dim": args.noise_dim,
                    "n_members": args.n_members, "alpha": args.alpha,
                    "lambda_is": args.lambda_is, "spread_weight": args.spread_weight,
                    "lambda_vs": args.lambda_vs, "lambda_es": args.lambda_es,
                    "type": "cln_residual_transformer",
                },
            }, f"{args.output_dir}/best_model.pt")

        # CLN scale analysis
        scale_norms = []
        for layer in model.layers:
            for n in ['temp_cln', 'temp_ff_cln', 'spat_cln', 'spat_ff_cln']:
                scale_norms.append(layer[n].scale_proj.weight.detach().norm().item())
        avg_scale = np.mean(scale_norms)

        if epoch % 40 == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate_model(model, val_preds, val_gts, val_conds,
                                     n_samples=50, device=device)
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  IS={ti:.4f}  "
                  f"({elapsed:.1f}s)  CLN={avg_scale:.3f}")
            print(f"  -> CI={metrics['ci_worst']:.3f}  CI_h={metrics['ci_h_pass']}/30  "
                  f"KS={metrics['ks']}/25  kurt={metrics['kurt']:.3f}  "
                  f"corr={metrics['corr']:.3f}  SS={metrics['ss']:.3f}  "
                  f"ER={metrics['eff_rank_ratio']:.3f}")
            history.append({"epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                           "mae": tm, "spread": ts, "is": ti, "cln_scale": avg_scale,
                           **metrics})
        else:
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  IS={ti:.4f}  "
                  f"({elapsed:.1f}s)  CLN={avg_scale:.3f}")
            history.append({"epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                           "mae": tm, "spread": ts, "is": ti, "cln_scale": avg_scale})

    torch.save({
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "config": {
            "n_frames": T, "n_cells": 25, "d_model": args.d_model,
            "n_heads": args.n_heads, "n_layers": args.n_layers,
            "cond_dim": cond_dim, "noise_dim": args.noise_dim,
            "n_members": args.n_members, "alpha": args.alpha,
            "lambda_is": args.lambda_is, "spread_weight": args.spread_weight,
            "lambda_vs": args.lambda_vs, "lambda_es": args.lambda_es,
            "type": "cln_residual_transformer",
        },
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
