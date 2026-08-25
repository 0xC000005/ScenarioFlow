"""240a — Joint-chunk DiT denoiser, H1 Stage 1 smoke test.

Paradigm pivot from AR (229a–233a families) to joint-chunk denoising (Diffusion Policy,
Chi et al 2023, arXiv:2303.04137 / GenCast arXiv:2312.15796). The full (T=30, S=25) future
trajectory is denoised jointly in one chunk, conditioned on history via a GRU encoder.
No AR compounding, no anchor, no FiLM-per-step gate. Regime conditioning now shapes the
*joint* distribution over the chunk rather than composing step-by-step.

Stage 1 goal (2-day feasibility): verify architecture wiring — sampling shape, conditioning
gradient flow, no NaNs. Gate to Stage 2: ANY improvement over 229a@ep30 (3/7) on any single
suite metric at evaluate_220b_multihorizon_path_suite.py.

Objective: standard DDPM ε-prediction with MSE loss. DDIM 20-step sampling at inference.

Architecture:
    x_t ∈ (B, 30, 25), t ∈ {0..T_diff-1}, cond ∈ (B, 128)
      → tokenize each cell: Linear(1, d_model) gives (B, 30, 25, d_model)
      → add time_pos + spatial_pos embeddings
      → reshape to (B, 750, d_model)
      → 6 × DiTBlock with AdaLN-Zero conditioning from (time_emb + history_cond)
      → output Linear(d_model, 1) → (B, 750, 1) → (B, 30, 25) predicted ε

Encoder warm-started from models/backfill/block_ar_vol_scaled_30ep/best_model.pt (the same
encoder every AR model in this project has used). Keep it trainable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.single_pass_ar import denormalize_iv, normalize_iv
from diffusion.ddpm_scheduler import DDPMScheduler
from diffusion.time_embedding import SinusoidalTimeEmbedding
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# =============================================================================
# Architecture
# =============================================================================
@dataclass
class JointChunkConfig:
    future_len: int = 30
    surface_cells: int = 25
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    history_cond_dim: int = 128

    # Diffusion
    n_diffusion_steps: int = 100
    schedule: str = "cosine"

    # Data
    history_len: int = 30


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning (Peebles & Xie 2023)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN(cond)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * h
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        return x


class JointChunkDiT(nn.Module):
    """Denoiser that attends jointly over (T × S) tokens of the future chunk."""

    def __init__(self, cfg: JointChunkConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        T = cfg.future_len
        S = cfg.surface_cells

        self.input_proj = nn.Linear(1, d)
        self.time_pos = nn.Parameter(torch.zeros(1, T, 1, d))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, S, d))
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.spatial_pos, std=0.02)

        self.noise_emb = SinusoidalTimeEmbedding(dim=cfg.time_embed_dim)
        cond_in_dim = cfg.time_embed_dim + cfg.history_cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, 4 * d),
            nn.SiLU(),
            nn.Linear(4 * d, d),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio, cond_dim=d) for _ in range(cfg.n_layers)
        ])

        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.output_proj = nn.Linear(d, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, history_cond: torch.Tensor):
        """Predict epsilon.

        Args:
            x_t: noisy future (B, T, S=25).
            t: diffusion timestep (B,) long.
            history_cond: (B, history_cond_dim) from GRUEncoder.

        Returns:
            eps_pred: (B, T, S).
        """
        B, T, S = x_t.shape
        d = self.cfg.d_model

        tokens = self.input_proj(x_t.unsqueeze(-1))  # (B, T, S, d)
        tokens = tokens + self.time_pos + self.spatial_pos
        tokens = tokens.reshape(B, T * S, d)

        t_emb = self.noise_emb(t)  # (B, time_embed_dim)
        cond = self.cond_mlp(torch.cat([t_emb, history_cond], dim=-1))  # (B, d)

        for block in self.blocks:
            tokens = block(tokens, cond)

        h = self.final_norm(tokens)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        eps = self.output_proj(h).reshape(B, T, S)
        return eps


# =============================================================================
# Full model: encoder + denoiser + diffusion
# =============================================================================
class JointChunkFlowModel(nn.Module):
    def __init__(self, cfg: JointChunkConfig):
        super().__init__()
        self.cfg = cfg
        enc_cfg = EncoderConfig(
            input_dim=cfg.surface_cells,
            extra_features=0,
            gru_hidden_dim=64,
            bottleneck_dim=cfg.history_cond_dim,
            cond_aug_sigma=0.0,
            dropout=0.1,
        )
        self.encoder = GRUEncoder(enc_cfg)
        self.denoiser = JointChunkDiT(cfg)
        self.scheduler = DDPMScheduler(
            n_steps=cfg.n_diffusion_steps,
            schedule=cfg.schedule,
            clip_sample=True,
            clip_sample_range=1.0,
            device="cpu",
        )

    def _move_scheduler(self, device):
        for a in (
            "betas", "alphas", "alpha_bar", "alpha_bar_prev",
            "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar",
            "posterior_variance", "posterior_log_variance",
            "sqrt_recip_alpha_bar", "sqrt_recip_alpha_bar_minus_one",
        ):
            setattr(self.scheduler, a, getattr(self.scheduler, a).to(device))
        self.scheduler.device = device

    def compute_loss(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """history, future both in normalized [-1, 1] space, (B, T, 5, 5)."""
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells
        cond = self.encoder(history)  # (B, history_cond_dim)
        x_0 = future.reshape(B, T, S)
        t = torch.randint(0, self.cfg.n_diffusion_steps, (B,), device=x_0.device)
        x_t, noise = self.scheduler.q_sample(x_0, t)
        eps_pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,  # retained for interface; trajectory length is fixed
        chunk_size: Optional[int] = None,  # retained for interface; unused
        n_inference_steps: int = 20,
        max_parallel: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        """Returns (B, n_samples, T=30, 5, 5) in [0, 1] (denormalized).

        Uses internal micro-batching to bound peak memory. `max_parallel` caps the
        number of (batch*sample) rows run through the denoiser per DDIM step. With
        T*S=750 tokens and d_model=128, 64 is comfortable on 8 GB GPUs.
        """
        self.eval()
        device = next(self.parameters()).device
        self._move_scheduler(device)
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells

        cond = self.encoder(history)  # (B, C)
        cond_rep = cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)

        total_rows = B * n_samples
        x = torch.randn(total_rows, T, S, device=device)

        total_steps = self.cfg.n_diffusion_steps
        step_indices = torch.linspace(
            total_steps - 1, 0, n_inference_steps + 1, device=device
        ).round().long()
        for i in range(n_inference_steps):
            t_cur_scalar = step_indices[i]
            t_prev_scalar = step_indices[i + 1]
            for start in range(0, total_rows, max_parallel):
                end = min(start + max_parallel, total_rows)
                x_mb = x[start:end]
                cond_mb = cond_rep[start:end]
                t_cur_mb = t_cur_scalar.expand(end - start)
                eps_pred = self.denoiser(x_mb, t_cur_mb, cond_mb)
                x_0_pred = self.scheduler.predict_x0_from_noise(x_mb, t_cur_mb, eps_pred)
                alpha_bar_prev = self.scheduler.alpha_bar[t_prev_scalar].view(1, 1, 1)
                x[start:end] = (
                    torch.sqrt(alpha_bar_prev) * x_0_pred
                    + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * eps_pred
                )

        x_clean = x.reshape(B, n_samples, T, 5, 5)
        return denormalize_iv(x_clean).clamp(0.0, 1.0)


# =============================================================================
# Training
# =============================================================================
def warmstart_encoder(model: JointChunkFlowModel, ckpt_path: str):
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model_state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    print(f"[warmstart] encoder loaded from {ckpt_path}")
    if missing:
        print(f"[warmstart]   missing keys: {missing}")
    if unexpected:
        print(f"[warmstart]   unexpected keys: {unexpected}")


def train_loop(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    cfg = JointChunkConfig(
        future_len=args.future_len,
        surface_cells=25,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        n_diffusion_steps=args.n_diffusion_steps,
        schedule=args.schedule,
        history_len=args.history_len,
    )
    model = JointChunkFlowModel(cfg).to(device)
    model._move_scheduler(device)
    if args.warmstart_encoder:
        warmstart_encoder(model, args.warmstart_encoder)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters: {n_params:,}")

    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    test_start = args.test_start
    val_size = args.val_size

    # Same split geometry as evaluate_220b
    max_train_end = test_start - cfg.history_len - cfg.future_len
    val_start_idx = max_train_end - val_size
    train_ds = VolSurfaceDataset(surfaces, cfg.history_len, cfg.future_len, start_idx=0, end_idx=val_start_idx + cfg.history_len + cfg.future_len - 1)
    # VolSurfaceDataset builds windows up to len(surfaces[start:end]) - total_len + 1
    # We want train windows with start index in [0, val_start_idx)
    train_ds.valid_starts = [i for i in train_ds.valid_starts if i < val_start_idx]

    val_ds = VolSurfaceDataset(surfaces, cfg.history_len, cfg.future_len, start_idx=val_start_idx, end_idx=max_train_end + cfg.history_len + cfg.future_len - 1)
    val_ds.valid_starts = list(range(len(val_ds.surfaces) - val_ds.total_len + 1))

    print(f"[train] train windows: {len(train_ds.valid_starts)}")
    print(f"[train] val   windows: {len(val_ds.valid_starts)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine LR schedule with warmup
    total_steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, int(0.05 * total_steps))

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos((step - warmup) / max(1, total_steps - warmup) * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_fn)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history_log = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            loss = model.compute_loss(history, future)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        dt = time.time() - t0

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                loss = model.compute_loss(history, future)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        lr_now = optim.param_groups[0]["lr"]
        print(
            f"[ep {epoch+1:3d}/{args.epochs}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} lr={lr_now:.2e} time={dt:.1f}s"
        )
        history_log.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")
            print(f"[ep {epoch+1}] new best val_loss={val_loss:.6f} → saved best_model.pt")

    # Final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "val_loss": val_loss,
    }, out_dir / "final_model.pt")
    (out_dir / "training_history.json").write_text(json.dumps(history_log, indent=2))
    print(f"[train] done. best val_loss={best_val:.6f}")


# =============================================================================
# Model loading hook
# =============================================================================
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = JointChunkConfig(**ckpt["config"])
    model = JointChunkFlowModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model._move_scheduler(device)
    model.eval()
    return model, ckpt


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--future_len", type=int, default=30)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--n_diffusion_steps", type=int, default=100)
    p.add_argument("--schedule", default="cosine")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmstart_encoder", default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    p.add_argument("--output_dir", default="models/backfill/240a_joint_chunk_dit_s42")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_loop(args)


if __name__ == "__main__":
    main()
