#!/usr/bin/env python
"""
167b_clean_isolation: Factorized Decoder Clean Isolation Test (Codex-designed)

Binary diagnostic: Can L@eps work when all confounds from 167a are removed?

4 fixes applied simultaneously:
1. CLN frozen to zero (remove competing diversity path)
2. wd=0 for load_head + cond_resid_film (remove 27x WD/grad kill mechanism)
3. FiLM uses (1+gamma)*h+beta (identity at init, not suppressive)
4. load_head simplified to Linear(d_model, n_factors) (remove 35,444x WD first layer)

If L wakes up (eff_rank >3, factor PC1 <0.8): 167a failure was fixable
If L stays rank-2: abandon factorized L@eps, need structure loss

Budget: 20 epochs (~35 min). Diagnostic only.

Usage:
    PYTHONPATH=. python -u experiments/backfill/block_ar/train_167b_clean_isolation.py \
        --epochs 20 --batch_size 16 --n_members 16 --noise_dim 32 \
        --n_factors 5 \
        --lambda_vs 1.0 --lambda_is 0.005 --is_warmup_epochs 10 --bptt_steps 5 \
        --lambda_floor 2.5 --floor_tau 0.005 --floor_warmup_epochs 10 \
        --output_dir models/backfill/afcrps_167b --device cuda
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
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    return obj


# --- ConditionalNorm (FROZEN to zero in this experiment) ---

class ConditionalNorm(nn.Module):
    """Per-cell CLN. In 167b: initialized to zero AND frozen."""

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


# --- Spatial Transformer Decoder (trunk) ---

class SpatialTransformerDecoder(nn.Module):
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

    def forward_trunk(self, cond, prev_frame, noise):
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
        return h


# --- Factorized Decoder (167b: simplified load_head, fixed FiLM) ---

class FactorizedDecoderClean(SpatialTransformerDecoder):
    """Clean isolation version: simplified load_head + fixed FiLM + CLN frozen."""

    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32, n_factors=5):
        super().__init__(n_cells=n_cells, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, cond_dim=cond_dim, noise_dim=noise_dim)
        self.n_factors = n_factors

        # Base head: zero-init (shared innovation)
        self.base_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)

        # FIX 4: Simplified load_head — single Linear, no hidden layer
        # Removes the 35,444x WD/grad ratio on the first layer
        # Small random init (Codex: NOT zeros)
        self.load_head = nn.Linear(d_model, n_factors)
        nn.init.normal_(self.load_head.weight, std=0.01)
        nn.init.zeros_(self.load_head.bias)

        # FIX 3: FiLM with (1+gamma)*h+beta (identity at init, not suppressive)
        self.cond_resid_film = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        # Zero-init the last layer so gamma=0, beta=0 at init → identity
        nn.init.zeros_(self.cond_resid_film[-1].weight)
        nn.init.zeros_(self.cond_resid_film[-1].bias)

        self.register_buffer('cond_ref', torch.zeros(cond_dim))

    def forward(self, cond, prev_frame, noise):
        h = self.forward_trunk(cond, prev_frame, noise)  # (B, 25, d_model)

        # Base innovation
        delta_base = self.base_head(h).squeeze(-1)  # (B, 25)

        # FIX 3: FiLM with (1+gamma) — identity at init
        cond_resid = cond - self.cond_ref
        film_params = self.cond_resid_film(cond_resid)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)  # (B, 25, d_model)

        # FIX 4: Simple linear load_head
        L = self.load_head(h_modulated)  # (B, 25, n_factors)

        return delta_base, L


# --- Loss functions ---

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


# --- Model ---

class ARFactorizedCleanModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_factors=5):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = FactorizedDecoderClean(**decoder_config, n_factors=n_factors)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_factors = n_factors

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

                # AR generation with factorized output
                frames = []
                prev = last_k
                gru_state = h_last_k.contiguous()
                gru_outs = gru_out_k

                for t in range(T):
                    z_t = torch.randn(B * k, self.decoder.noise_dim, device=device)
                    if t > 0:
                        al = self.encoder.attn_proj(gru_outs).squeeze(-1)
                        aw = F.softmax(al, dim=1)
                        hp = (aw.unsqueeze(-1) * gru_outs).sum(dim=1)
                        cond_t = self.encoder.bottleneck(hp)
                    else:
                        cond_t = cond_init

                    delta_base, L = self.decoder(cond_t, prev, z_t)
                    eps = torch.randn(B * k, self.n_factors, device=device)
                    delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
                    frame_t = prev + torch.tanh(delta)
                    frame_t = reflecting_boundary(frame_t)
                    frames.append(frame_t)

                    fn = normalize_iv(frame_t).unsqueeze(1)
                    go, gru_state = self.encoder.gru(fn, gru_state)
                    gru_outs = torch.cat([gru_outs, go], dim=1)
                    prev = frame_t

                frames = torch.stack(frames, dim=1).reshape(B, k, T, 5, 5)
                all_samples.append(frames)

            samples = torch.cat(all_samples, dim=1)
        return samples


# --- cond_ref computation ---

def compute_cond_ref(model, surf_tensor, indices, device, H=30, C=25):
    model.eval()
    cond_sum = torch.zeros(128, device=device)
    n = 0
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            end = min(start + 64, len(indices))
            idx = torch.from_numpy(indices[start:end]).long().to(device)
            hist = surf_tensor[idx.unsqueeze(1) + torch.arange(H, device=device).unsqueeze(0)]
            hist_flat = normalize_iv(hist).reshape(len(idx), H, C)
            gru_outputs, _ = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)
            cond_sum += cond.sum(dim=0)
            n += len(idx)
    return cond_sum / n


def main():
    parser = argparse.ArgumentParser(description="167b: Clean Isolation Test (Codex-designed)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--n_factors", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5)
    parser.add_argument("--lambda_vs", type=float, default=1.0)
    parser.add_argument("--lambda_is", type=float, default=0.005)
    parser.add_argument("--is_warmup_epochs", type=int, default=10)
    parser.add_argument("--bptt_steps", type=int, default=5)
    parser.add_argument("--lambda_floor", type=float, default=2.5)
    parser.add_argument("--floor_tau", type=float, default=0.005)
    parser.add_argument("--floor_warmup_epochs", type=int, default=10)
    parser.add_argument("--cond_ref_ema_decay", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)
    val_indices = np.arange(max_train_idx - VAL_SIZE, max_train_idx)

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}", flush=True)

    encoder_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=C, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, cond_dim=128, noise_dim=args.noise_dim,
    )

    model = ARFactorizedCleanModel(encoder_config, decoder_config, n_factors=args.n_factors).to(device)

    # FIX 1: Freeze ALL CLN parameters to zero
    cln_frozen = 0
    for name, param in model.decoder.named_parameters():
        if 'cln' in name or 'ff_cln' in name:
            param.requires_grad = False
            cln_frozen += param.numel()

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_load = sum(p.numel() for p in model.decoder.load_head.parameters())
    n_film = sum(p.numel() for p in model.decoder.cond_resid_film.parameters())

    print(f"\n{'='*60}", flush=True)
    print(f"167b: Clean Isolation Test (Codex-designed)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total params: {n_enc + n_dec:,} ({n_trainable:,} trainable)", flush=True)
    print(f"  CLN frozen: {cln_frozen:,} params (FIX 1)", flush=True)
    print(f"  load_head: {n_load:,} params (FIX 4: simple Linear, no hidden layer)", flush=True)
    print(f"  cond_resid_film: {n_film:,} params", flush=True)
    print(f"  FiLM: (1+gamma)*h+beta (FIX 3: identity at init)", flush=True)
    print(f"  K={args.n_members}, n_factors={args.n_factors}", flush=True)
    print(f"  Loss: afCRPS + VS(lambda={args.lambda_vs}) + IS(lambda={args.lambda_is})", flush=True)
    print(f"  Budget: {args.epochs} epochs (diagnostic)", flush=True)

    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    print("\nComputing initial cond_ref...", flush=True)
    cond_ref_init = compute_cond_ref(model, surf_tensor, train_indices, device, H, C)
    model.decoder.cond_ref.copy_(cond_ref_init)
    print(f"  cond_ref norm: {cond_ref_init.norm():.4f}", flush=True)

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

    # FIX 2: wd=0 for load_head and cond_resid_film
    load_film_params = set()
    for name, param in model.decoder.named_parameters():
        if 'load_head' in name or 'cond_resid_film' in name:
            load_film_params.add(id(param))

    trunk_params = [p for p in model.decoder.parameters()
                    if p.requires_grad and id(p) not in load_film_params]
    load_film_param_list = [p for p in model.decoder.parameters()
                            if p.requires_grad and id(p) in load_film_params]

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": trunk_params, "lr": args.lr_decoder, "weight_decay": 0.01},
        {"params": load_film_param_list, "lr": args.lr_decoder, "weight_decay": 0.0},  # FIX 2
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"  Optimizer: 3 groups (encoder wd=0.01, trunk wd=0.01, load+film wd=0.0)", flush=True)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0; ep_mae = 0; ep_spread = 0; ep_vs = 0; ep_is = 0; ep_floor = 0; nb = 0

        if args.is_warmup_epochs > 0 and epoch <= args.is_warmup_epochs:
            lambda_is_eff = args.lambda_is * epoch / args.is_warmup_epochs
        else:
            lambda_is_eff = args.lambda_is

        if args.floor_warmup_epochs > 0 and epoch <= args.floor_warmup_epochs:
            lambda_floor_eff = args.lambda_floor * epoch / args.floor_warmup_epochs
        else:
            lambda_floor_eff = args.lambda_floor

        if epoch > 1:
            cond_ref_new = compute_cond_ref(model, surf_tensor, train_indices, device, H, C)
            decay = args.cond_ref_ema_decay
            model.decoder.cond_ref.copy_(decay * model.decoder.cond_ref + (1 - decay) * cond_ref_new)
            model.train()

        for hist, gt_frames, last_frame in train_loader:
            B = hist.shape[0]; K = args.n_members
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
                    al_t = model.encoder.attn_proj(gru_outs).squeeze(-1)
                    aw_t = F.softmax(al_t, dim=1)
                    hp_t = (aw_t.unsqueeze(-1) * gru_outs).sum(dim=1)
                    cond_K = model.encoder.bottleneck(hp_t)

                delta_base, L = model.decoder(cond_K, prev, z_t)
                eps = torch.randn(B * K, args.n_factors, device=device)
                delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
                frame_t = prev + torch.tanh(delta)

                frame_BK = frame_t.reshape(B, K, C)
                gt_t = gt_frames[:, t, :]
                loss_t, mae_t, spread_t = afcrps_per_frame(frame_BK, gt_t, alpha=args.alpha, spread_weight=args.spread_weight)
                is_t = interval_score(frame_BK, gt_t)
                vs_t = variogram_score_per_frame(frame_BK, gt_t) if args.lambda_vs > 0 else torch.tensor(0.0, device=device)
                floor_barrier = args.floor_tau * F.softplus(-frame_t / args.floor_tau).mean()

                step_loss = (loss_t + lambda_is_eff * is_t + args.lambda_vs * vs_t + lambda_floor_eff * floor_barrier) / T
                window_loss = window_loss + step_loss

                total_loss_val += step_loss.item()
                total_mae += mae_t.item()
                total_spread += spread_t.item()
                total_vs += vs_t.item() if args.lambda_vs > 0 else 0
                ep_is += is_t.item()
                ep_floor += floor_barrier.item()

                is_window_end = ((t + 1) % N == 0) or (t == T - 1)
                if is_window_end:
                    window_loss.backward()
                    window_loss = 0
                    prev = frame_t.detach()
                    h_gru = h_gru.detach()
                    gru_outs = gru_outs.detach()
                else:
                    prev = frame_t

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
        tl = ep_loss / nb; tm = ep_mae / nb; ts = ep_spread / nb; tvs = ep_vs / nb
        elapsed = time.time() - t0

        # Validation
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
                    delta_base, L_v = model.decoder(cond_K, prev, z_t)
                    eps_v = torch.randn(B2 * K2, args.n_factors, device=device)
                    delta = delta_base + torch.einsum("bcr,br->bc", L_v, eps_v)
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

        # Checkpoints
        config_dict = {
            "type": "ar_spatial_transformer_167b_clean_isolation",
            "encoder": {"input_dim": 25, "gru_hidden_dim": 64, "bottleneck_dim": 128, "dropout": 0.1},
            "decoder": decoder_config,
            "n_members": args.n_members, "n_factors": args.n_factors,
            "alpha": args.alpha, "spread_weight": args.spread_weight,
            "lambda_vs": args.lambda_vs, "lambda_is": args.lambda_is,
            "n_frames": T, "n_cells": C, "train_windows": len(train_indices),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": config_dict},
                       f"{args.output_dir}/best_model.pt")
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": config_dict},
                       f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")

        # L diagnostics
        with torch.no_grad():
            sh = normalize_iv(train_hist[:4]).reshape(4, H, C)
            sg, _ = model.encoder.gru(sh)
            sa = F.softmax(model.encoder.attn_proj(sg).squeeze(-1), dim=1)
            sc = model.encoder.bottleneck((sa.unsqueeze(-1) * sg).sum(dim=1))
            sz = torch.randn(4, args.noise_dim, device=device)
            _, L_diag = model.decoder(sc, train_last[:4], sz)
            L_norm = L_diag.norm().item() / 4
            L_std = L_diag.std().item()
            # L effective rank
            L_flat = L_diag.reshape(-1, args.n_factors)  # (100, 5)
            _, sv, _ = torch.linalg.svd(L_flat)
            sv_norm = sv / sv.sum()
            L_eff_rank = torch.exp(-(sv_norm * torch.log(sv_norm + 1e-10)).sum()).item()

        tfloor = ep_floor / nb
        print(f"Ep {epoch:3d}  loss={tl:.4f}  val={val_loss:.4f}  "
              f"mae={tm:.4f}  spread={ts:.4f}  vs={tvs:.6f}  "
              f"L_norm={L_norm:.4f}  L_std={L_std:.4f}  L_rank={L_eff_rank:.2f}  ({elapsed:.1f}s)"
              + (f"  *best" if val_loss <= best_val else ""), flush=True)

        history.append({
            "epoch": epoch, "train_loss": tl, "val_loss": val_loss,
            "mae": tm, "spread": ts, "vs": tvs,
            "L_norm": L_norm, "L_std": L_std, "L_eff_rank": L_eff_rank,
        })

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs, "val_loss": val_loss, "config": config_dict},
               f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
