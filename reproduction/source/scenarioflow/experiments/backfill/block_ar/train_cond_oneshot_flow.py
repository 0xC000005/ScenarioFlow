#!/usr/bin/env python
"""
153a: Conditional One-Shot Flow Matching with Concatenation Conditioning

RC15-H1-S1: Add conditioning from pretrained GRU encoder to the factored
velocity transformer. Condition is projected and concatenated per token
(FMAP 2504.03463 design). Source = N(0,I) per H2 result.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_cond_oneshot_flow.py \
        --epochs 200 --batch_size 64 --lr 5e-4 \
        --encoder_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --output_dir models/backfill/flow_153a --device cuda
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
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


class ConditionalFactoredVelocityTransformer(nn.Module):
    """Velocity network with factored temporal + spatial attention + condition concatenation.

    Same as FactoredVelocityTransformer but each token gets an extra condition
    embedding concatenated to its features. The condition is projected from
    the encoder's 128-dim output to d_model and added as an additional input
    channel per token position.

    Per FMAP design: concatenation preserves attention's ability to learn
    cross-cell correlations (unlike FiLM which modulates globally).
    """

    def __init__(self, n_frames=30, n_cells=25, d_model=128, n_heads=4,
                 n_layers=4, time_dim=64, cond_dim=128):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.d_model = d_model
        self.time_dim = time_dim

        # Input projection: per-cell value → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Condition projection: encoder output → d_model
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

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

    def forward(self, x_t, t, cond=None):
        """
        Args:
            x_t: (B, T*C) noisy state
            t: (B,) time in [0, 1]
            cond: (B, cond_dim) conditioning vector from encoder, or None
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

        # Add condition embedding (broadcast to all positions)
        if cond is not None:
            c_emb = self.cond_proj(cond)  # (B, d_model)
            h = h + c_emb[:, None, None, :]

        # Alternating temporal and spatial attention
        for layer in self.layers:
            # Temporal attention
            h_temp = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
            h_norm = layer['temp_norm'](h_temp)
            attn_out, _ = layer['temp_attn'](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer['temp_ff'](layer['temp_ff_norm'](h_temp))
            h = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3)

            # Spatial attention
            h_spat = h.reshape(B * T, C, -1)
            h_norm = layer['spat_norm'](h_spat)
            attn_out, _ = layer['spat_attn'](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer['spat_ff'](layer['spat_ff_norm'](h_spat))
            h = h_spat.reshape(B, T, C, -1)

        # Output projection
        h = self.output_norm(h)
        v = self.output_proj(h).squeeze(-1)  # (B, T, C)
        return v.reshape(B, T * C)


def load_encoder(encoder_path, device):
    """Load pretrained GRU encoder from DDPM checkpoint."""
    ckpt = torch.load(encoder_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]

    enc_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=cfg.get("gru_hidden_dim", 64),
        bottleneck_dim=cfg.get("bottleneck_dim", 128),
        cond_aug_sigma=0.0,  # no augmentation at inference
        dropout=0.0,  # no dropout at inference
    )
    encoder = GRUEncoder(enc_config)

    # Extract encoder weights from the full model state dict
    enc_sd = {}
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith("encoder."):
            enc_sd[k[len("encoder."):]] = v
    encoder.load_state_dict(enc_sd)
    encoder.to(device).eval()

    return encoder, enc_config.bottleneck_dim


def normalize_iv(surfaces):
    """Normalize IV surfaces from [0,1] to [-1,1]."""
    return surfaces * 2.0 - 1.0


def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder (FROZEN)
    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded encoder: cond_dim={cond_dim}")

    # Data — need both history (for encoder) and future (for flow matching)
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, F_LEN, DIM = 30, 30, 750

    train_end = 4040
    train_histories = []
    train_futures = []
    for i in range(train_end - H - F_LEN + 1):
        history = surfaces[i:i+H]  # (30, 5, 5)
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)  # (750,)
        train_histories.append(history)
        train_futures.append(future)

    train_hist = np.array(train_histories, dtype=np.float32)  # (N, 30, 5, 5)
    train_data = np.array(train_futures, dtype=np.float32)  # (N, 750)

    val_histories = []
    val_futures = []
    for i in range(train_end - H - F_LEN + 1, 4540 - H - F_LEN + 1):
        history = surfaces[i:i+H]
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)
        val_histories.append(history)
        val_futures.append(future)

    val_hist = np.array(val_histories, dtype=np.float32)
    val_data = np.array(val_futures, dtype=np.float32)

    # Standardize futures
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True) + 1e-6
    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std

    # Pre-compute encoder conditions for ALL training data (frozen encoder, deterministic)
    print("Pre-computing encoder conditions...")
    train_conds = []
    with torch.no_grad():
        for i in range(0, len(train_hist), 256):
            batch_hist = torch.from_numpy(train_hist[i:i+256]).to(device)
            batch_hist_norm = normalize_iv(batch_hist)
            cond = encoder(batch_hist_norm)  # (B, 128)
            train_conds.append(cond.cpu().numpy())
    train_conds = np.concatenate(train_conds, axis=0)  # (N, 128)

    val_conds = []
    with torch.no_grad():
        for i in range(0, len(val_hist), 256):
            batch_hist = torch.from_numpy(val_hist[i:i+256]).to(device)
            batch_hist_norm = normalize_iv(batch_hist)
            cond = encoder(batch_hist_norm)
            val_conds.append(cond.cpu().numpy())
    val_conds = np.concatenate(val_conds, axis=0)

    # Encoder output statistics
    cond_var = np.var(train_conds, axis=0)
    print(f"  Encoder output variance: mean={cond_var.mean():.4f}, "
          f"min={cond_var.min():.4f}, max={cond_var.max():.4f}")
    print(f"  Non-zero variance dims (>0.01): {(cond_var > 0.01).sum()}/128")

    print("=" * 60)
    print("153a: Conditional One-Shot FM — Concatenation Conditioning")
    print("=" * 60)
    print(f"  Train: {train_norm.shape}, Val: {val_norm.shape}")
    print(f"  Condition: {train_conds.shape}")

    # Model — conditional version
    model = ConditionalFactoredVelocityTransformer(
        n_frames=F_LEN, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} (velocity net only, encoder frozen)")

    # Dataset: futures + conditions
    train_ds = TensorDataset(
        torch.from_numpy(train_norm),
        torch.from_numpy(train_conds)
    )
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
        for x1, cond in train_loader:
            x1 = x1.to(device)
            cond = cond.to(device)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)  # N(0,I) source — confirmed by H2
            t = torch.rand(B, device=device)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = model(x_t, t, cond=cond)
            loss = F.mse_loss(v, x1 - x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = epoch_loss / nb
        elapsed = time.time() - t0

        # Val loss
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for i in range(0, len(val_norm), args.batch_size):
                x1 = torch.from_numpy(val_norm[i:i+args.batch_size]).to(device)
                cond = torch.from_numpy(val_conds[i:i+args.batch_size]).to(device)
                B = x1.shape[0]
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
                v = model(x_t, t, cond=cond)
                vl += F.mse_loss(v, x1 - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "train_mean": train_mean, "train_std": train_std,
                "encoder_path": args.encoder_path,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                           "cond_dim": cond_dim, "source": "gaussian",
                           "conditioning": "concatenation"},
            }, f"{args.output_dir}/best_model.pt")

        # Evaluate every 40 epochs
        if epoch % 40 == 0 or epoch == 1:
            with torch.no_grad():
                # Generate samples using random training conditions
                all_samp = []
                eval_batch = 64
                n_eval = 512
                dt = 1.0 / args.n_steps
                for si in range(0, n_eval, eval_batch):
                    eb = min(eval_batch, n_eval - si)
                    x = torch.randn(eb, DIM, device=device)
                    # Random conditions from training set
                    idx = np.random.choice(len(train_conds), eb, replace=True)
                    c = torch.from_numpy(train_conds[idx]).to(device)
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt, device=device)
                        x = x + model(x, tt, cond=c) * dt
                    all_samp.append(x.cpu().numpy())
                samples = np.concatenate(all_samp)
                samples = samples * train_std + train_mean
                samples = np.clip(samples, 0, 1)
            m = evaluate_samples(samples, train_data)
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)  eff_rank={m['eff_rank']:.2f}(GT={m['gt_eff_rank']:.2f})  "
                  f"PC1={m['pc1']:.3f}  PC2={m['pc2']:.3f}  frob={m['frob']:.1f}  "
                  f"KS={m['ks_pass']}/25  kurt={m['kurt_ratio']:.3f}  "
                  f"spread h1={m['spread_h1']:.4f} h30={m['spread_h30']:.4f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, **m})
        else:
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "train_mean": train_mean, "train_std": train_std,
        "encoder_path": args.encoder_path,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                   "cond_dim": cond_dim, "source": "gaussian",
                   "conditioning": "concatenation"},
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
