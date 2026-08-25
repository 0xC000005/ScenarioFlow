"""240c_iter — Iterative DDIM unroll + ES+VS (Stage 2.5, the SOTA recipe).

Matches the GenCast / AIFS-CRPS pattern: the DiT is trained inside its native diffusion
scaffold (multi-step iterative denoising is preserved), AND the K-ensemble is scored
with proper multivariate scoring rules at the end of the denoising loop.

Loss per batch:
    L = λ_fm · MSE(eps_pred(x_t, t, cond), noise)       # at a random t per window
        + λ_es · EnergyScore(samples_K, gt)              # K samples from full DDIM unroll
        + λ_vs · VariogramScore(samples_K, gt)

Gradient checkpointing across DDIM steps is essential to fit K=32 on an 8 GB GPU.
Training-time DDIM is short (8 steps); inference-time DDIM can use more (20+).

Relation to Stage 2 (240b single-step): single-step is the ablation. Stage 2.5 is the
principled version. If 240c_iter beats 240b on any non-trivial metric (corr, kurtosis,
max-jump), the iterative scaffold is load-bearing and this is the architecture we ship
to Stage 3 regime-conditioning. If it ties, single-step wasn't actually losing anything.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.single_pass_ar import (
    denormalize_iv,
    energy_score,
    variogram_score,
)
from diffusion.ddpm_scheduler import DDPMScheduler
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# Reuse architecture from 240a verbatim
from experiments.backfill.block_ar.train_240a_joint_chunk_flow import (
    JointChunkConfig,
    JointChunkDiT,
)


class JointChunkIterCRPSModel(nn.Module):
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

    def _ddim_step(self, x, t_cur, t_prev, cond):
        """Single DDIM update step. Wrapped for grad checkpointing."""
        eps = self.denoiser(x, t_cur, cond)
        x0_hat = self.scheduler.predict_x0_from_noise(x, t_cur, eps)
        alpha_bar_prev = self.scheduler.alpha_bar[t_prev].view(-1, 1, 1)
        return (
            torch.sqrt(alpha_bar_prev) * x0_hat
            + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * eps
        )

    def _unroll_ddim(self, cond_K: torch.Tensor, n_ddim: int, use_checkpoint: bool):
        """Unroll DDIM for (B*K) parallel samples, return final (B*K, T, S)."""
        device = cond_K.device
        BK = cond_K.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells
        x = torch.randn(BK, T, S, device=device)
        total_steps = self.cfg.n_diffusion_steps
        step_indices = torch.linspace(
            total_steps - 1, 0, n_ddim + 1, device=device
        ).round().long()
        for i in range(n_ddim):
            t_cur = step_indices[i].expand(BK)
            t_prev = step_indices[i + 1].expand(BK)
            if use_checkpoint and self.training:
                x = checkpoint_utils.checkpoint(
                    self._ddim_step, x, t_cur, t_prev, cond_K, use_reentrant=False
                )
            else:
                x = self._ddim_step(x, t_cur, t_prev, cond_K)
        return x

    def compute_loss(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        n_samples: int,
        n_ddim_train: int,
        lam_fm: float,
        lam_es: float,
        lam_vs: float,
        use_checkpoint: bool = True,
    ):
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells
        cond = self.encoder(history)  # (B, C)

        # (1) Flow-matching MSE at random t — keeps the diffusion training signal alive
        x_0 = future.reshape(B, T, S)
        t_fm = torch.randint(0, self.cfg.n_diffusion_steps, (B,), device=x_0.device)
        x_t, noise = self.scheduler.q_sample(x_0, t_fm)
        eps_pred = self.denoiser(x_t, t_fm, cond)
        loss_fm = F.mse_loss(eps_pred, noise)

        # (2) DDIM unroll for K samples with grad flow + proper scoring rules
        cond_K = cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
        x_final = self._unroll_ddim(cond_K, n_ddim_train, use_checkpoint)
        x_final = torch.clamp(x_final, -1.0, 1.0)
        samples = x_final.reshape(B, n_samples, T, 5, 5)
        es = energy_score(samples, future, spread_only=False)
        vs = variogram_score(samples, future)

        loss = lam_fm * loss_fm + lam_es * es + lam_vs * vs
        return loss, {"fm": loss_fm.item(), "es": es.item(), "vs": vs.item()}

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        n_inference_steps: int = 20,
        max_parallel: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        self._move_scheduler(device)
        B = history.shape[0]
        T = self.cfg.future_len
        S = self.cfg.surface_cells
        cond = self.encoder(history)
        cond_rep = cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
        total_rows = B * n_samples
        out = torch.empty(total_rows, T, S, device=device)
        for start in range(0, total_rows, max_parallel):
            end = min(start + max_parallel, total_rows)
            x = self._unroll_ddim(cond_rep[start:end], n_inference_steps, use_checkpoint=False)
            out[start:end] = torch.clamp(x, -1.0, 1.0)
        x_clean = out.reshape(B, n_samples, T, 5, 5)
        return denormalize_iv(x_clean).clamp(0.0, 1.0)


def warmstart_from_checkpoint(model: JointChunkIterCRPSModel, ckpt_path: str, label: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    den_sd = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    me, ue = model.encoder.load_state_dict(enc_sd, strict=False)
    md, ud = model.denoiser.load_state_dict(den_sd, strict=False)
    print(f"[warmstart:{label}] loaded encoder + denoiser from {ckpt_path}")
    if me or ue: print(f"  encoder missing={me[:3]} unexpected={ue[:3]}")
    if md or ud: print(f"  denoiser missing={md[:3]} unexpected={ud[:3]}")


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
    model = JointChunkIterCRPSModel(cfg).to(device)
    model._move_scheduler(device)

    # Warm-start priority: 240b (Stage 2 single-step) > 240a (MSE) > encoder-only
    if args.warmstart_240b and Path(args.warmstart_240b).exists():
        warmstart_from_checkpoint(model, args.warmstart_240b, "240b")
    elif args.warmstart_240a and Path(args.warmstart_240a).exists():
        warmstart_from_checkpoint(model, args.warmstart_240a, "240a")

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

    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    use_autocast = amp_dtype != torch.float32
    print(f"[train] precision={args.precision} (autocast={use_autocast})")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        fm_acc, es_acc, vs_acc, n = 0.0, 0.0, 0.0, 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                loss, metrics = model.compute_loss(
                    history, future, args.n_samples, args.n_ddim_train,
                    args.lam_fm, args.lam_es, args.lam_vs,
                )
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step(); scheduler.step(); optim.zero_grad()
            fm_acc += metrics["fm"]; es_acc += metrics["es"]; vs_acc += metrics["vs"]; n += 1
        if n % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step(); scheduler.step(); optim.zero_grad()
        train_fm, train_es, train_vs = fm_acc/n, es_acc/n, vs_acc/n
        dt = time.time() - t0

        model.eval()
        v_fm, v_es, v_vs, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                    _, m = model.compute_loss(
                        history, future, args.n_samples, args.n_ddim_train,
                        args.lam_fm, args.lam_es, args.lam_vs, use_checkpoint=False,
                    )
                v_fm += m["fm"]; v_es += m["es"]; v_vs += m["vs"]; vn += 1
        val_fm, val_es, val_vs = v_fm/vn, v_es/vn, v_vs/vn
        val_loss = args.lam_fm * val_fm + args.lam_es * val_es + args.lam_vs * val_vs
        lr_now = optim.param_groups[0]["lr"]
        print(
            f"[ep {epoch+1:3d}/{args.epochs}] "
            f"tr(fm={train_fm:.4f} es={train_es:.4f} vs={train_vs:.4f}) "
            f"va(fm={val_fm:.4f} es={val_es:.4f} vs={val_vs:.4f}) "
            f"val_loss={val_loss:.4f} lr={lr_now:.2e} time={dt:.1f}s"
        )
        history_log.append({
            "epoch": epoch+1, "train_fm": train_fm, "train_es": train_es, "train_vs": train_vs,
            "val_fm": val_fm, "val_es": val_es, "val_vs": val_vs, "val_loss": val_loss, "lr": lr_now,
        })
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch+1,
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
    print(f"[train] done. best val_loss={best_val:.4f}")


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = JointChunkConfig(**ckpt["config"])
    model = JointChunkIterCRPSModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model._move_scheduler(device)
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
    p.add_argument("--n_samples", type=int, default=32, help="K ensemble size")
    p.add_argument("--n_ddim_train", type=int, default=8, help="DDIM steps during training (gradient flows through these)")
    p.add_argument("--lam_fm", type=float, default=1.0)
    p.add_argument("--lam_es", type=float, default=1.0)
    p.add_argument("--lam_vs", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmstart_240b", default="models/backfill/240b_joint_chunk_crps_k128_s42/best_model.pt")
    p.add_argument("--warmstart_240a", default="models/backfill/240a_joint_chunk_dit_s42/best_model.pt")
    p.add_argument("--output_dir", default="models/backfill/240c_iter_ddim_crps_s42")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16",
                   help="Mixed precision. bf16 recommended (Ampere-native, safer than fp16 for sqrt-heavy ES/VS).")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_loop(args)


if __name__ == "__main__":
    main()
