"""240b — Joint-chunk DiT generator with Energy-Score + Variogram-Score (Stage 2).

Reuses 240a's DiT backbone but retasks it as a one-step conditional generator. At training
time we sample K noise vectors per window, push them through the denoiser (at a fixed
diffusion timestep, interpretable as a "high-noise" level), predict x_0 directly, and score
the K-ensemble against the ground-truth future with:

    L = λ_es · EnergyScore(samples, gt)
        + λ_vs · VariogramScore(samples, gt)
        + λ_mse · MSE(mean(samples), gt)   # small stabilising accuracy term

Energy Score is the multivariate generalization of CRPS (Gneiting & Raftery 2007) and
directly supervises the joint sample distribution across cells — the exact fix for the
corr_ratio=0.002 failure mode observed in 240a Stage 1. Variogram Score further supervises
spatial pair-wise variability structure (Scheuerer & Hamill 2015).

At inference, samples come from one denoiser pass from pure noise at the high-noise timestep
— no DDIM loop needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.single_pass_ar import (
    denormalize_iv,
    normalize_iv,
    energy_score,
    variogram_score,
)
from diffusion.time_embedding import SinusoidalTimeEmbedding
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# Reuse architecture from 240a verbatim
from experiments.backfill.block_ar.train_240a_joint_chunk_flow import (
    JointChunkConfig,
    JointChunkDiT,
)


# =============================================================================
# Full model: encoder + DiT used as a conditional one-step generator
# =============================================================================
class JointChunkCRPSModel(nn.Module):
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

    def _generate(self, history: torch.Tensor, n_samples: int) -> torch.Tensor:
        """One-step conditional generator: noise → predicted future.

        Returns predicted future in normalized [-1, 1] space, shape (B, K, T, S).
        """
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells
        device = history.device

        cond = self.encoder(history)  # (B, C)
        cond_rep = cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)

        # Noise at the "highest-noise" timestep — the model is conditioned with the
        # max timestep index and learns to map pure noise to clean data in one pass.
        z = torch.randn(B * n_samples, T, S, device=device)
        t = torch.full((B * n_samples,), self.cfg.n_diffusion_steps - 1, device=device, dtype=torch.long)

        # DiT output — parameterised as x_0-prediction directly (no ε subtraction).
        x0_pred = self.denoiser(z, t, cond_rep)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        return x0_pred.reshape(B, n_samples, T, S)

    def compute_loss(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        n_samples: int,
        lam_es: float,
        lam_vs: float,
        lam_mse: float,
    ):
        """history, future both in normalized [-1, 1] space, (B, T, 5, 5)."""
        B = history.shape[0]
        T = self.cfg.future_len
        x0_samples = self._generate(history, n_samples)  # (B, K, T, S=25)
        samples_2d = x0_samples.reshape(B, n_samples, T, 5, 5)
        gt_2d = future
        # ES and VS expect IV space (they use squared distances; shift doesn't matter for ES spread
        # but the accuracy term and GT alignment mean same-space matters for both sides — both are
        # computed in normalised [-1, 1] here for consistency; the bitter-lesson signal is the same).
        es = energy_score(samples_2d, gt_2d, spread_only=False)
        vs = variogram_score(samples_2d, gt_2d)
        mean_samp = samples_2d.mean(dim=1)
        mse = F.mse_loss(mean_samp, gt_2d)
        loss = lam_es * es + lam_vs * vs + lam_mse * mse
        return loss, {"es": es.item(), "vs": vs.item(), "mse": mse.item()}

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        max_parallel: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        """Returns (B, n_samples, T=30, 5, 5) in [0, 1]."""
        self.eval()
        device = next(self.parameters()).device
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells

        cond = self.encoder(history)  # (B, C)
        cond_rep = cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
        total_rows = B * n_samples

        # Micro-batch through the denoiser
        out = torch.empty(total_rows, T, S, device=device)
        t_scalar = torch.tensor(self.cfg.n_diffusion_steps - 1, device=device, dtype=torch.long)
        for start in range(0, total_rows, max_parallel):
            end = min(start + max_parallel, total_rows)
            z = torch.randn(end - start, T, S, device=device)
            t_mb = t_scalar.expand(end - start)
            x0 = self.denoiser(z, t_mb, cond_rep[start:end])
            out[start:end] = torch.clamp(x0, -1.0, 1.0)

        x_clean = out.reshape(B, n_samples, T, 5, 5)
        return denormalize_iv(x_clean).clamp(0.0, 1.0)


# =============================================================================
# Training
# =============================================================================
def warmstart_from_240a(model: JointChunkCRPSModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    den_sd = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    me, ue = model.encoder.load_state_dict(enc_sd, strict=False)
    md, ud = model.denoiser.load_state_dict(den_sd, strict=False)
    print(f"[warmstart] loaded encoder + denoiser from {ckpt_path}")
    for label, (missing, unexp) in (("encoder", (me, ue)), ("denoiser", (md, ud))):
        if missing:
            print(f"[warmstart]   {label} missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexp:
            print(f"[warmstart]   {label} unexpected: {unexp[:5]}")


def warmstart_encoder_only(model: JointChunkCRPSModel, ckpt_path: str):
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model_state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_sd, strict=False)
    print(f"[warmstart] encoder only loaded from {ckpt_path}")


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
    model = JointChunkCRPSModel(cfg).to(device)
    if args.warmstart_240a:
        warmstart_from_240a(model, args.warmstart_240a)
    elif args.warmstart_encoder:
        warmstart_encoder_only(model, args.warmstart_encoder)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters: {n_params:,}")

    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    test_start = args.test_start
    val_size = args.val_size
    max_train_end = test_start - cfg.history_len - cfg.future_len
    val_start_idx = max_train_end - val_size
    total_len = cfg.history_len + cfg.future_len

    train_ds = VolSurfaceDataset(surfaces, cfg.history_len, cfg.future_len, start_idx=0, end_idx=val_start_idx + total_len - 1)
    val_ds = VolSurfaceDataset(surfaces, cfg.history_len, cfg.future_len, start_idx=val_start_idx, end_idx=max_train_end + total_len - 1)
    print(f"[train] train windows: {len(train_ds.valid_starts)}")
    print(f"[train] val   windows: {len(val_ds.valid_starts)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    grad_accum = max(1, args.grad_accum)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        es_acc, vs_acc, mse_acc, n = 0.0, 0.0, 0.0, 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            loss, metrics = model.compute_loss(history, future, args.n_samples, args.lam_es, args.lam_vs, args.lam_mse)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad()
            es_acc += metrics["es"]; vs_acc += metrics["vs"]; mse_acc += metrics["mse"]; n += 1
        # Flush any remaining accumulated grads
        if n % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()
        train_es, train_vs, train_mse = es_acc / n, vs_acc / n, mse_acc / n
        dt = time.time() - t0

        model.eval()
        v_es, v_vs, v_mse, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                _, m = model.compute_loss(history, future, args.n_samples, args.lam_es, args.lam_vs, args.lam_mse)
                v_es += m["es"]; v_vs += m["vs"]; v_mse += m["mse"]; vn += 1
        val_es, val_vs, val_mse = v_es / vn, v_vs / vn, v_mse / vn
        val_loss = args.lam_es * val_es + args.lam_vs * val_vs + args.lam_mse * val_mse
        lr_now = optim.param_groups[0]["lr"]
        print(
            f"[ep {epoch+1:3d}/{args.epochs}] tr(es={train_es:.4f} vs={train_vs:.4f} mse={train_mse:.4f}) "
            f"va(es={val_es:.4f} vs={val_vs:.4f} mse={val_mse:.4f}) val_loss={val_loss:.5f} "
            f"lr={lr_now:.2e} time={dt:.1f}s"
        )
        history_log.append({
            "epoch": epoch + 1,
            "train_es": train_es, "train_vs": train_vs, "train_mse": train_mse,
            "val_es": val_es, "val_vs": val_vs, "val_mse": val_mse, "val_loss": val_loss, "lr": lr_now,
        })
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")
            print(f"[ep {epoch+1}] new best → saved")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "val_loss": val_loss,
    }, out_dir / "final_model.pt")
    (out_dir / "training_history.json").write_text(json.dumps(history_log, indent=2))
    print(f"[train] done. best val_loss={best_val:.5f}")


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = JointChunkConfig(**ckpt["config"])
    model = JointChunkCRPSModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


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
    p.add_argument("--n_samples", type=int, default=8, help="K ensemble size per batch for ES/VS")
    p.add_argument("--lam_es", type=float, default=1.0)
    p.add_argument("--lam_vs", type=float, default=0.1)
    p.add_argument("--lam_mse", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmstart_240a", default="models/backfill/240a_joint_chunk_dit_s42/best_model.pt",
                   help="Warm-start both encoder and DiT from 240a checkpoint")
    p.add_argument("--warmstart_encoder", default=None,
                   help="If no 240a ckpt, warm-start only encoder from this path")
    p.add_argument("--output_dir", default="models/backfill/240b_joint_chunk_crps_s42")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_loop(args)


if __name__ == "__main__":
    main()
