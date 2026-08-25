#!/usr/bin/env python
"""
164a_v3_percell_bptt_gate: Per-Cell CLN + IS fix + Partial BPTT + Noise Gate

RC20.5: Add state-dependent noise gate to ConditionalNorm.
sigmoid(Linear(n_cells, n_cells)) on prev_frame gates the noise scale/bias.
When prev is near zero (data floor), gate reduces noise for that cell.
Only the noise path is gated — mean-reversion correction is preserved.

Root cause (Codex-verified): at prev=0.02 for cell (2,4), correction=+0.072
but noise std=0.110 (ratio 0.65). Good cells have ratio >1.5. The gate
targets this by dampening noise amplitude near the floor.

Architecture change from BPTT:
  - ConditionalNorm gains gate_proj = Linear(n_cells, n_cells)
  - forward(x, z, prev_frame) applies noise_gate = sigmoid(gate_proj(prev_frame))
  - scale and bias multiplied by noise_gate before modulating h
  - Gate bias init'd at +2.0 so sigmoid(2)=0.88 — near pass-through at start
  - 625 extra params (25x25 linear)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_164a_v3_percell_bptt_gate.py \
        --epochs 80 --batch_size 16 --n_members 16 --noise_dim 32 \
        --lambda_vs 0.5 --lambda_is 0.05 --is_warmup_epochs 10 --bptt_steps 5 \
        --output_dir models/backfill/afcrps_164a_v3_percell_bptt_gate --device cuda
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
    """Normalize IV surfaces from [0,1] to [-1,1]."""
    return surfaces * 2.0 - 1.0


def denormalize_iv(surfaces):
    """Denormalize IV surfaces from [-1,1] to [0,1]."""
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


# --- No-LN Conditional Norm with Noise Gate (RC20.5) -------------------------

class ConditionalNorm(nn.Module):
    """Per-cell No-LN CLN with state-dependent noise gate.

    y_c = (gate_c * scale_c(z) + 1) * x_c + gate_c * bias_c(z)

    gate_c = sigmoid(Linear(prev_frame))  -- when prev near 0, gate dampens noise
    Only the noise modulation (scale, bias from z) is gated.
    The identity path (the +1) is preserved, so mean-reversion correction flows freely.
    """

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

        # Noise gate: prev_frame (B, n_cells) -> gate (B, n_cells) in [0, 1]
        self.gate_proj = nn.Linear(n_cells, n_cells)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)  # sigmoid(2) ~ 0.88: near pass-through

    def forward(self, x, z, prev_frame=None):
        """x: (B, C, d_model), z: (B, noise_dim), prev_frame: (B, C) in [0,1]"""
        B = z.shape[0]
        scale = self.scale_proj(z).view(B, self.n_cells, self.d_model)
        bias = self.bias_proj(z).view(B, self.n_cells, self.d_model)

        if prev_frame is not None:
            noise_gate = torch.sigmoid(self.gate_proj(prev_frame))  # (B, n_cells)
            scale = scale * noise_gate.unsqueeze(-1)  # (B, C, d_model)
            bias = bias * noise_gate.unsqueeze(-1)

        return (scale + 1.0) * x + bias


# --- Spatial Transformer Decoder ---------------------------------------------

class SpatialTransformerDecoder(nn.Module):
    """Per-frame spatial transformer: attention over 25 cells.

    No LayerNorm (FCN3), ConditionalNorm with noise gate, LayerScale 0.1,
    He init, zero-init output. Each call generates ONE frame's delta.
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
                if 'output_proj' in name or 'scale_proj' in name or 'bias_proj' in name or 'gate_proj' in name:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cond, prev_frame, noise):
        """
        cond: (B, cond_dim)
        prev_frame: (B, 25) in [0, 1]
        noise: (B, noise_dim)
        Returns: delta (B, 25)
        """
        B = cond.shape[0]
        h = self.input_proj(prev_frame.unsqueeze(-1))  # (B, 25, d_model)
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos
        z = self.noise_proj(noise)  # (B, noise_dim)

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[li * 2]
            ls_f = self.ls_params[li * 2 + 1]

            # Pass prev_frame to CLN for noise gating
            h_norm = layer['cln'](h, z, prev_frame)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out

            h = h + ls_f * layer['ff'](layer['ff_cln'](h, z, prev_frame))

        delta = self.output_proj(h).squeeze(-1)  # (B, 25)
        return delta


# --- Loss functions -----------------------------------------------------------

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


def reflecting_boundary(x, lo=0.01, hi=1.0):
    below = x < lo
    above = x > hi
    x = torch.where(below, 2 * lo - x, x)
    x = torch.where(above, 2 * hi - x, x)
    return x.clamp(lo, hi)


# --- AR Spatial Transformer Model with Noise Gate ----------------------------

class ARSpatialTransformerModel(nn.Module):
    """Full model: GRU encoder + spatial transformer decoder (with noise gate) + AR generation."""

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SpatialTransformerDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

    def encode(self, history):
        return self.encoder(history)

    def ar_generate(self, condition, last_frame, n_steps=30, gru_state=None,
                    gru_outputs=None):
        B = condition.shape[0]
        device = condition.device
        noise_dim = self.decoder.noise_dim
        frames = []
        prev = last_frame

        for t in range(n_steps):
            z_t = torch.randn(B, noise_dim, device=device)

            if gru_state is not None and gru_outputs is not None:
                attn_logits = self.encoder.attn_proj(gru_outputs).squeeze(-1)
                attn_weights = F.softmax(attn_logits, dim=1)
                h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
                cond_t = self.encoder.bottleneck(h_pooled)
            else:
                cond_t = condition

            delta = self.decoder(cond_t, prev, z_t)
            frame_t = prev + torch.tanh(delta)
            frames.append(frame_t)

            if gru_state is not None:
                frame_norm = normalize_iv(frame_t).unsqueeze(1)
                gru_out, gru_state = self.encoder.gru(frame_norm, gru_state)
                gru_outputs = torch.cat([gru_outputs, gru_out], dim=1)

            prev = frame_t

        return torch.stack(frames, dim=1)

    def sample_batched(self, history, n_samples=50, **kwargs):
        B = history.shape[0]
        T = 30
        device = history.device
        CHUNK = 10

        with torch.no_grad():
            last_frame = denormalize_iv(history[:, -1]).reshape(B, 25)
            hist_flat = history.reshape(B, history.shape[1], -1)
            gru_outputs_base, h_last_base = self.encoder.gru(hist_flat)

            all_samples = []
            for start in range(0, n_samples, CHUNK):
                k = min(CHUNK, n_samples - start)
                last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(B * k, 25)
                gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                    B, k, -1, -1).reshape(B * k, -1, self.encoder_config.gru_hidden_dim)
                h_last_k = h_last_base.unsqueeze(2).expand(
                    1, B, k, -1).reshape(1, B * k, -1)

                attn_logits = self.encoder.attn_proj(gru_out_k).squeeze(-1)
                attn_weights = F.softmax(attn_logits, dim=1)
                h_pooled = (attn_weights.unsqueeze(-1) * gru_out_k).sum(dim=1)
                cond_init = self.encoder.bottleneck(h_pooled)

                frames = self.ar_generate(
                    cond_init, last_k, n_steps=T,
                    gru_state=h_last_k.contiguous(),
                    gru_outputs=gru_out_k,
                )
                frames = frames.reshape(B, k, T, 5, 5)
                all_samples.append(frames)

            samples = torch.cat(all_samples, dim=1)
        return samples


# --- Training ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="164a_v3_percell_bptt_gate: CLN Noise Gate")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_members", type=int, default=16,
                        help="K ensemble members (vectorized into batch dim)")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5)
    parser.add_argument("--lambda_vs", type=float, default=0.5)
    parser.add_argument("--lambda_is", type=float, default=0.05)
    parser.add_argument("--is_warmup_epochs", type=int, default=10)
    parser.add_argument("--bptt_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Data --
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)
    val_indices = np.arange(max_train_idx - VAL_SIZE, max_train_idx)
    test_indices = np.arange(TEST_START, N_total - H - T + 1)

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # -- Model --
    encoder_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=C, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=128, noise_dim=args.noise_dim,
    )

    model = ARSpatialTransformerModel(encoder_config, decoder_config).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_gate = sum(p.numel() for p in model.decoder.layers[0]['cln'].gate_proj.parameters()) * len(model.decoder.layers) * 2
    print(f"\n{'='*60}")
    print(f"164a_v3_percell_bptt_gate: Per-Cell CLN + Noise Gate (RC20.5)")
    print(f"{'='*60}")
    print(f"  Encoder (GRU, random init): {n_enc:,} params")
    print(f"  Decoder (spatial transformer): {n_dec:,} params")
    print(f"  Gate params (8 CLN layers x 25x25+25): ~{n_gate:,} params")
    print(f"  Total: {n_enc + n_dec:,} params")
    print(f"  K={args.n_members}, noise_dim={args.noise_dim}")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"  Loss: afCRPS + VS(l={args.lambda_vs}) + IS(l={args.lambda_is})")
    print(f"  BPTT steps: {args.bptt_steps}")
    print(f"  Noise gate: sigmoid(Linear(25,25)) on prev_frame, bias init +2.0")
    print(f"  Output: frame_t = prev_frame + tanh(delta)")
    print(f"  Boundary: NONE | GRU feedback: DURING TRAINING")

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
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0; ep_mae = 0; ep_spread = 0; ep_vs = 0; ep_is = 0; nb = 0

        if args.is_warmup_epochs > 0 and epoch <= args.is_warmup_epochs:
            lambda_is_eff = args.lambda_is * epoch / args.is_warmup_epochs
        else:
            lambda_is_eff = args.lambda_is

        for hist, gt_frames, last_frame in train_loader:
            B = hist.shape[0]
            K = args.n_members

            hist_norm = normalize_iv(hist)
            hist_flat = hist_norm.reshape(B, H, C)
            gru_outputs, h_last = model.encoder.gru(hist_flat)

            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)

            cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)

            h_gru = h_last.detach().unsqueeze(2).expand(1, B, K, -1).reshape(1, B * K, -1).contiguous()
            gru_outs = gru_outputs.detach().unsqueeze(1).expand(B, K, H, -1).reshape(B * K, H, -1)

            N = args.bptt_steps
            optimizer.zero_grad()
            total_mae = 0; total_spread = 0; total_vs = 0; total_loss_val = 0
            window_loss = 0

            for t in range(T):
                z_t = torch.randn(B * K, args.noise_dim, device=device)

                if t > 0:
                    attn_logits_t = model.encoder.attn_proj(gru_outs).squeeze(-1)
                    attn_weights_t = F.softmax(attn_logits_t, dim=1)
                    h_pooled_t = (attn_weights_t.unsqueeze(-1) * gru_outs).sum(dim=1)
                    cond_K = model.encoder.bottleneck(h_pooled_t)

                delta = model.decoder(cond_K, prev, z_t)
                frame_t = prev + torch.tanh(delta)

                frame_BK = frame_t.reshape(B, K, C)
                gt_t = gt_frames[:, t, :]

                loss_t, mae_t, spread_t = afcrps_per_frame(
                    frame_BK, gt_t, alpha=args.alpha, spread_weight=args.spread_weight
                )
                is_t = interval_score(frame_BK, gt_t)
                vs_t = variogram_score_per_frame(frame_BK, gt_t) if args.lambda_vs > 0 else torch.tensor(0.0, device=device)

                step_loss = (loss_t + lambda_is_eff * is_t + args.lambda_vs * vs_t) / T
                window_loss = window_loss + step_loss

                total_loss_val += step_loss.item()
                total_mae += mae_t.item()
                total_spread += spread_t.item()
                total_vs += vs_t.item() if args.lambda_vs > 0 else 0
                ep_is += is_t.item()

                is_window_end = ((t + 1) % N == 0) or (t == T - 1)
                if is_window_end:
                    window_loss.backward()
                    window_loss = 0
                    prev = frame_t.detach()
                    h_gru = h_gru.detach()
                    gru_outs = gru_outs.detach()
                else:
                    prev = frame_t  # NO detach — gradient flows within BPTT window

                frame_norm = normalize_iv(prev.detach()).unsqueeze(1)
                gru_out, h_gru = model.encoder.gru(frame_norm, h_gru)
                gru_outs = torch.cat([gru_outs, gru_out], dim=1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += total_loss_val
            ep_mae += total_mae / T
            ep_spread += total_spread / T
            ep_vs += total_vs / T
            nb += 1

        scheduler.step()
        tl = ep_loss / nb; tm = ep_mae / nb; ts = ep_spread / nb; tvs = ep_vs / nb; tis = ep_is / nb
        elapsed = time.time() - t0

        # -- Validation --
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for v_hist, v_gt, v_last in val_loader:
                B2 = v_hist.shape[0]; K2 = args.n_members
                vh_norm = normalize_iv(v_hist)
                vh_flat = vh_norm.reshape(B2, H, C)
                v_gru_outs, v_h = model.encoder.gru(vh_flat)
                v_attn = F.softmax(model.encoder.attn_proj(v_gru_outs).squeeze(-1), dim=1)
                v_cond = model.encoder.bottleneck((v_attn.unsqueeze(-1) * v_gru_outs).sum(dim=1))
                cond_K = v_cond.unsqueeze(1).expand(B2, K2, -1).reshape(B2 * K2, -1)
                prev = v_last.unsqueeze(1).expand(B2, K2, C).reshape(B2 * K2, C)
                v_h_k = v_h.unsqueeze(2).expand(1, B2, K2, -1).reshape(1, B2*K2, -1).contiguous()
                v_gru_k = v_gru_outs.unsqueeze(1).expand(B2, K2, H, -1).reshape(B2*K2, H, -1)
                vloss = 0
                for t in range(T):
                    z_t = torch.randn(B2 * K2, args.noise_dim, device=device)
                    if t > 0:
                        va = F.softmax(model.encoder.attn_proj(v_gru_k).squeeze(-1), dim=1)
                        cond_K = model.encoder.bottleneck((va.unsqueeze(-1) * v_gru_k).sum(dim=1))
                    delta = model.decoder(cond_K, prev, z_t)
                    frame_t = prev + torch.tanh(delta)
                    frame_BK = frame_t.reshape(B2, K2, C)
                    crps_t, _, _ = afcrps_per_frame(frame_BK, v_gt[:, t])
                    vloss += crps_t.item()
                    prev = frame_t
                    fn = normalize_iv(frame_t).unsqueeze(1)
                    go, v_h_k = model.encoder.gru(fn, v_h_k)
                    v_gru_k = torch.cat([v_gru_k, go], dim=1)
                vl += (vloss / T) * B2; nv += B2
        val_loss = vl / max(nv, 1)

        # -- Save best model --
        config_dict = {
            "type": "ar_spatial_transformer_164a_v3_percell_bptt_gate",
            "encoder": {
                "input_dim": 25, "gru_hidden_dim": 64,
                "bottleneck_dim": 128, "dropout": 0.1,
            },
            "decoder": decoder_config,
            "n_members": args.n_members,
            "alpha": args.alpha, "spread_weight": args.spread_weight,
            "lambda_vs": args.lambda_vs, "lambda_is": args.lambda_is,
            "n_frames": T, "n_cells": C,
            "train_windows": len(train_indices),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "config": config_dict,
            }, f"{args.output_dir}/best_model.pt")

        # -- Logging --
        # Log gate stats every 10 epochs
        gate_info = ""
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                # Check gate values for a low-IV input (floor region)
                dummy_low = torch.full((1, 25), 0.02, device=device)
                dummy_mid = torch.full((1, 25), 0.30, device=device)
                gate_low = torch.sigmoid(model.decoder.layers[0]['cln'].gate_proj(dummy_low)).mean().item()
                gate_mid = torch.sigmoid(model.decoder.layers[0]['cln'].gate_proj(dummy_mid)).mean().item()
                gate_info = f"  gate(0.02)={gate_low:.3f} gate(0.30)={gate_mid:.3f}"

        print(f"Ep {epoch:3d}  loss={tl:.4f}  val={val_loss:.4f}  "
              f"mae={tm:.4f}  spread={ts:.4f}  vs={tvs:.6f}  is={tis:.4f}  "
              f"lis={lambda_is_eff:.3f}  ({elapsed:.1f}s)"
              + (f"  *best" if val_loss <= best_val else "")
              + gate_info)

        history.append({
            "epoch": epoch, "train_loss": tl, "val_loss": val_loss,
            "mae": tm, "spread": ts, "vs": tvs,
        })

    # -- Save final --
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
