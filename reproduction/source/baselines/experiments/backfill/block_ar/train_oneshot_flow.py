#!/usr/bin/env python
"""
152e: One-Shot Flow Matching with Factored Transformer

Generates all (30, 5, 5) = 750 dims at once via factored temporal + spatial
attention velocity network. Same architecture principle as 133c (which worked
at 750-dim with CRPS), but with CFM loss instead.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_oneshot_flow.py \
        --epochs 300 --batch_size 64 --lr 5e-4 \
        --output_dir models/backfill/flow_152e --device cuda
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


class FactoredVelocityTransformer(nn.Module):
    """Velocity network with factored temporal + spatial attention.

    Processes (B, T*C) input with alternating:
    - Temporal attention: each cell attends across T=30 timesteps
    - Spatial attention: each frame attends across C=25 cells
    This factorization gives the model inductive bias for spatiotemporal data
    without requiring T*C = 750 fully-connected interactions.
    """

    def __init__(self, n_frames=30, n_cells=25, d_model=128, n_heads=4,
                 n_layers=4, time_dim=64):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.d_model = d_model
        self.time_dim = time_dim

        # Input projection: per-cell value → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Temporal position encoding (learned, T positions)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)
        # Spatial position encoding (learned, C positions)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_cells, d_model) * 0.02)

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Alternating temporal and spatial attention layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                # Temporal: (B*C, T, d_model)
                'temp_norm': nn.LayerNorm(d_model),
                'temp_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'temp_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'temp_ff_norm': nn.LayerNorm(d_model),
                # Spatial: (B*T, C, d_model)
                'spat_norm': nn.LayerNorm(d_model),
                'spat_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'spat_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'spat_ff_norm': nn.LayerNorm(d_model),
            }))

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        # Zero-init for stable start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def time_embed(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, t):
        """
        Args:
            x_t: (B, T*C) noisy state, T=30, C=25
            t: (B,) time in [0, 1]
        Returns:
            v: (B, T*C) predicted velocity
        """
        B = x_t.shape[0]
        T, C = self.n_frames, self.n_cells

        # Reshape to (B, T, C, 1) and project to d_model
        h = x_t.reshape(B, T, C, 1)
        h = self.input_proj(h)  # (B, T, C, d_model)

        # Add positional encodings
        h = h + self.temporal_pos + self.spatial_pos

        # Add time embedding (broadcast to all positions)
        t_emb = self.time_proj(self.time_embed(t))  # (B, d_model)
        h = h + t_emb[:, None, None, :]

        # Alternating temporal and spatial attention
        for layer in self.layers:
            # --- Temporal attention: each cell attends across time ---
            # Reshape: (B, T, C, d) → (B*C, T, d)
            h_temp = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
            h_norm = layer['temp_norm'](h_temp)
            attn_out, _ = layer['temp_attn'](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer['temp_ff'](layer['temp_ff_norm'](h_temp))
            h = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3)  # → (B, T, C, d)

            # --- Spatial attention: each frame attends across cells ---
            # Reshape: (B, T, C, d) → (B*T, C, d)
            h_spat = h.reshape(B * T, C, -1)
            h_norm = layer['spat_norm'](h_spat)
            attn_out, _ = layer['spat_attn'](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer['spat_ff'](layer['spat_ff_norm'](h_spat))
            h = h_spat.reshape(B, T, C, -1)  # → (B, T, C, d)

        # Output projection
        h = self.output_norm(h)
        v = self.output_proj(h).squeeze(-1)  # (B, T, C)
        return v.reshape(B, T * C)


def evaluate_samples(samples, gt_data, n_frames=30, n_cells=25):
    """Evaluate 750-dim samples against GT."""
    samples_3d = samples.reshape(-1, n_frames, n_cells)
    gt_3d = gt_data.reshape(-1, n_frames, n_cells)

    # Daily changes for cross-cell correlation
    gen_ch = np.diff(samples_3d, axis=1).reshape(-1, n_cells)
    gt_ch = np.diff(gt_3d, axis=1).reshape(-1, n_cells)

    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)

    def eff_rank(corr):
        ev = np.linalg.eigvalsh(corr)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    # PC alignment
    gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
    pc1 = abs(float(np.dot(gt_vecs[:, 0], gen_vecs[:, 0])))
    pc2 = abs(float(np.dot(gt_vecs[:, 1], gen_vecs[:, 1])))

    # KS on daily changes
    ks_pass = sum(1 for c in range(n_cells)
                  if ks_2samp(gen_ch[:, c], gt_ch[:, c])[0] < 0.15)

    # Kurtosis
    kurt_gen = kurtosis(gen_ch.flatten(), fisher=True)
    kurt_gt = kurtosis(gt_ch.flatten(), fisher=True)

    # Spread growth
    spreads = samples_3d.std(axis=0).mean(axis=1)  # (T,)
    spread_h1 = spreads[0]
    spread_h30 = spreads[-1]
    mono = all(spreads[i+1] >= spreads[i] * 0.99 for i in range(len(spreads)-1))

    return {
        'eff_rank': eff_rank(gen_corr), 'gt_eff_rank': eff_rank(gt_corr),
        'pc1': pc1, 'pc2': pc2,
        'frob': float(np.linalg.norm(gen_corr - gt_corr, 'fro')),
        'ks_pass': ks_pass, 'kurt_ratio': kurt_gen / (kurt_gt + 1e-6),
        'spread_h1': spread_h1, 'spread_h30': spread_h30, 'mono': mono,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, F_LEN, DIM = 30, 30, 750

    train_end = 4040
    futures = [surfaces[i+H:i+H+F_LEN].reshape(-1)
               for i in range(train_end - H - F_LEN + 1)]
    train_data = np.array(futures, dtype=np.float32)

    val_futures = [surfaces[i+H:i+H+F_LEN].reshape(-1)
                   for i in range(train_end - H - F_LEN + 1, 4540 - H - F_LEN + 1)]
    val_data = np.array(val_futures, dtype=np.float32)

    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True) + 1e-6
    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std

    print("=" * 60)
    print("152e: One-Shot Flow Matching — Factored Transformer (750-dim)")
    print("=" * 60)
    print(f"  Train: {train_norm.shape}, Val: {val_norm.shape}")

    # Model
    model = FactoredVelocityTransformer(
        n_frames=F_LEN, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_ds = TensorDataset(torch.from_numpy(train_norm))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0; nb = 0
        for (x1,) in train_loader:
            x1 = x1.to(device)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = model(x_t, t)
            loss = F.mse_loss(v, x1 - x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = epoch_loss / nb
        elapsed = time.time() - t0

        # Val
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for i in range(0, len(val_norm), args.batch_size):
                x1 = torch.from_numpy(val_norm[i:i+args.batch_size]).to(device)
                B = x1.shape[0]
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
                v = model(x_t, t)
                vl += F.mse_loss(v, x1 - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "train_mean": train_mean, "train_std": train_std,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": F_LEN, "n_cells": 25, "dim": DIM},
            }, f"{args.output_dir}/best_model.pt")

        # Evaluate every 30 epochs
        if epoch % 30 == 0 or epoch == 1:
            with torch.no_grad():
                # Generate in batches to avoid OOM
                all_samp = []
                eval_batch = 64
                n_eval = 512
                dt = 1.0 / args.n_steps
                for si in range(0, n_eval, eval_batch):
                    eb = min(eval_batch, n_eval - si)
                    x = torch.randn(eb, DIM, device=device)
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt, device=device)
                        x = x + model(x, tt) * dt
                    all_samp.append(x.cpu().numpy())
                samples = np.concatenate(all_samp)
                samples = samples * train_std + train_mean
                samples = np.clip(samples, 0, 1)
            m = evaluate_samples(samples, train_data)
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)  eff_rank={m['eff_rank']:.2f}(GT={m['gt_eff_rank']:.2f})  "
                  f"PC1={m['pc1']:.3f}  PC2={m['pc2']:.3f}  frob={m['frob']:.1f}  "
                  f"KS={m['ks_pass']}/25  kurt={m['kurt_ratio']:.3f}  "
                  f"spread h1={m['spread_h1']:.4f} h30={m['spread_h30']:.4f} "
                  f"mono={m['mono']}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, **m})
        else:
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss, "train_mean": train_mean, "train_std": train_std,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": F_LEN, "n_cells": 25, "dim": DIM},
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
