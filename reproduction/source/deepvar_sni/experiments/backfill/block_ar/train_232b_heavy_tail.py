#!/usr/bin/env python
"""
232b: Heavy-tail Student-t innovation noise. Attacks max-jump KS.

Replaces Gaussian z_f with Student-t (learnable ν). Keeps everything else.
Warm-start from 229a@ep30. Inference still uses scale_anchor at 0.50.

Why Student-t instead of mixture-of-Gaussians: one extra scalar parameter ν,
closed-form sampling (z = normal / sqrt(chi2/df / ν)), differentiable, well-
understood as a heavier-tailed proxy for Gaussian.

Init ν=5.0 (moderate heaviness; ν→∞ recovers Gaussian; ν<2 is pathological).
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
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel,
    afcrps_per_step,
    compute_trajectory_loss,
)


# ---------------------------------------------------------------------------
# Student-t sampler. Differentiable via reparameterization (chi2 = sum of K
# gamma(0.5, 2) samples; we use torch.distributions.StudentT).
# ---------------------------------------------------------------------------

def sample_student_t(df: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sample from Student-t with learnable df. df must be > 2 for finite variance.

    df has the target shape; output matches df.shape. Uses reparameterized
    Chi2 sampling so gradients flow through df.

    Rescales to unit variance so replacing torch.randn by this is a drop-in
    swap of innovation distribution, not a scale change.
    """
    df_safe = df.clamp_min(2.1)
    chi2_dist = torch.distributions.Chi2(df_safe)
    chi2 = chi2_dist.rsample()  # shape matches df_safe
    normal = torch.randn_like(df_safe)
    z = normal / torch.sqrt(chi2 / df_safe)
    # rescale to unit variance: Var(Student-t) = df / (df - 2)
    unit_scale = torch.sqrt(df_safe / (df_safe - 2.0))
    return z / unit_scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeavyTailFactorARModel(FactorARModel):
    """229a + Student-t factor noise (learnable ν)."""

    def __init__(self, *, nu_init: float = 5.0, nu_min: float = 2.1, **base_kwargs):
        super().__init__(**base_kwargs)
        # Use unconstrained param: nu = nu_min + softplus(raw_nu)
        # raw_nu s.t. softplus(raw_nu) = nu_init - nu_min
        delta = max(nu_init - nu_min, 0.1)
        raw_init = math.log(math.exp(delta) - 1.0)  # inverse of softplus
        self.raw_nu = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.nu_min = float(nu_min)

    @property
    def nu(self) -> torch.Tensor:
        return self.nu_min + F.softplus(self.raw_nu)

    # ------------------------------------------------------------------
    # forward — same as 227a but factor noise is Student-t.
    # ------------------------------------------------------------------
    def forward(
        self, history_01: torch.Tensor, n_members: int, n_steps: int,
    ) -> torch.Tensor:
        B = history_01.shape[0]
        device = history_01.device

        cond, local_scale = self.encode_history(history_01)
        prev = history_01[:, -1].reshape(B, self.n_cells)
        scale_anchor = local_scale.clone() if self.use_scale_anchor else None

        BK = B * n_members
        cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
        if self.use_scale_anchor:
            scale_anchor_bk = scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

        # Student-t initial z_f (shape follows df via broadcast)
        df = self.nu.expand(BK, self.factor_rank)
        z_f = sample_student_t(df, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        frames = []
        for t in range(n_steps):
            if t > 0:
                # AR(1) update in Student-t space
                z_f_new = sample_student_t(df, device=device)
                z_f = self.rho * z_f + rho_sq_comp * z_f_new
            z_i = torch.randn(BK, self.n_cells, device=device)  # keep idio Gaussian
            pos = self.pos_embed(t, BK, device)

            factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
            f_scores = self.factor_head(factor_in)
            idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
            i_resid = self.idio_head(idio_in)

            Lambda = self.get_lambda(cond)
            D = self.get_d(cond)
            v = torch.einsum("br,bcr->bc", f_scores, Lambda) + D * i_resid
            if self.noise_skip:
                v = v + torch.tanh(self.noise_skip_proj(z_i))
            if self.cell_spread_enabled:
                cs = F.softplus(self.cell_spread_proj(cond))
                v = v * cs

            delta = torch.sinh(v) * local_scale
            next_iv = (prev + delta).clamp(0.001, 1.0)
            frames.append(next_iv)

            feat, local_scale_raw = self._step_features(prev, next_iv, local_scale)
            cond = self.gru_cell(feat, cond)
            prev = next_iv

            if self.use_scale_anchor:
                log_s = (
                    (1.0 - self.scale_anchor_alpha)
                    * torch.log(local_scale_raw.clamp_min(self.scale_floor))
                    + self.scale_anchor_alpha
                    * torch.log(scale_anchor_bk.clamp_min(self.scale_floor))
                )
                local_scale = torch.exp(log_s)
            else:
                local_scale = local_scale_raw

        trajectory = (
            torch.stack(frames, dim=0).permute(1, 0, 2).view(B, n_members, n_steps, 5, 5)
        )
        return trajectory


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str, device: torch.device,
) -> tuple[HeavyTailFactorARModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = HeavyTailFactorARModel(
        n_cells=cfg["n_cells"], factor_rank=cfg["factor_rank"],
        hidden_dim=cfg["hidden_dim"], gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"], decoder_hidden=cfg["decoder_hidden"],
        rho=cfg["rho"], ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"], include_scale_feature=cfg["include_scale_feature"],
        pos_embed_dim=cfg["pos_embed_dim"], noise_skip=cfg.get("noise_skip", False),
        d_scale=cfg.get("d_scale", 1.0), cell_spread=cfg.get("cell_spread", False),
        decoder_layers=cfg.get("decoder_layers", 2), no_tanh=cfg.get("no_tanh", False),
        use_scale_anchor=cfg.get("use_scale_anchor", False),
        scale_anchor_alpha=cfg.get("scale_anchor_alpha", 0.50),
        nu_init=cfg.get("nu_init", 5.0), nu_min=cfg.get("nu_min", 2.1),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


# ---------------------------------------------------------------------------
# Main (fine-tune from 229a)
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="232b: heavy-tail factor noise")
    p.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    p.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    p.add_argument("--warmstart_checkpoint", type=str, required=True)
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--max_train_windows", type=int, default=4010)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_members", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_vs", type=float, default=0.05)
    p.add_argument("--loss_type", type=str, default="es", choices=["es", "afcrps"])
    p.add_argument("--factor_rank", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=256)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--pos_embed_dim", type=int, default=16)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--ewma_alpha", type=float, default=0.20)
    p.add_argument("--scale_floor", type=float, default=1e-4)
    p.add_argument("--noise_skip", action="store_true")
    p.add_argument("--d_scale", type=float, default=3.0)
    p.add_argument("--no_tanh", action="store_true")
    p.add_argument("--nu_init", type=float, default=5.0)
    p.add_argument("--nu_min", type=float, default=2.1)
    p.add_argument("--checkpoint_every", type=int, default=3)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HeavyTailFactorARModel(
        n_cells=25, factor_rank=args.factor_rank,
        hidden_dim=args.hidden_dim, gru_layers=2, gru_dropout=0.1,
        decoder_hidden=args.decoder_hidden,
        rho=args.rho, ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor, include_scale_feature=True,
        pos_embed_dim=args.pos_embed_dim,
        noise_skip=args.noise_skip, d_scale=args.d_scale,
        cell_spread=False, decoder_layers=args.decoder_layers,
        no_tanh=args.no_tanh,
        use_scale_anchor=False, scale_anchor_alpha=0.50,
        nu_init=args.nu_init, nu_min=args.nu_min,
    ).to(device)

    # PCA init
    pca = np.load(args.pca_init)
    model.init_from_pca(pca["lambda_init"], pca["d_init"])

    # Warm-start from 229a
    ws = torch.load(args.warmstart_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ws["model_state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("raw_nu")]
    if real_missing:
        print(f"WARN warm-start missing: {real_missing[:5]}")
    print(f"Warm-started from {args.warmstart_checkpoint} "
          f"(ep={ws.get('epoch')}, val={ws.get('val_loss'):.4f})")
    print(f"Initial nu: {model.nu.item():.3f}")

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        "type": "heavy_tail_factor_ar_232b",
        "n_cells": 25, "factor_rank": args.factor_rank,
        "hidden_dim": args.hidden_dim, "gru_layers": 2, "gru_dropout": 0.1,
        "decoder_hidden": args.decoder_hidden,
        "rho": args.rho, "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor, "include_scale_feature": True,
        "pos_embed_dim": args.pos_embed_dim, "lambda_vs": args.lambda_vs,
        "n_steps": args.n_steps, "n_members": args.n_members,
        "noise_skip": args.noise_skip, "d_scale": args.d_scale,
        "cell_spread": False, "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type, "no_tanh": args.no_tanh,
        "use_scale_anchor": False, "scale_anchor_alpha": 0.50,
        "nu_init": args.nu_init, "nu_min": args.nu_min,
        "warmstart_checkpoint": args.warmstart_checkpoint,
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n232b Heavy-Tail Factor AR (ν learnable)")
    print(f"  nu_init={args.nu_init} nu_min={args.nu_min}")
    print(f"  decoder_hidden={args.decoder_hidden}")
    print(f"  n_members={args.n_members} batch={args.batch_size} lr={args.lr}")
    print(f"  train={train_hist.shape[0]} val={val_hist.shape[0]}")
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
            loss, metrics = compute_trajectory_loss(
                trajectory, future_01, lambda_vs=args.lambda_vs, loss_type=args.loss_type,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            b = history_01.shape[0]
            running["loss"] += metrics["loss"] * b
            running["main"] += metrics["main"] * b
            running["vs"] += metrics["vs"] * b
            count += b
        train_metrics = {k: v / count for k, v in running.items()}

        model.eval()
        val_main_sum, val_vs_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt = model(vh, n_members=args.n_members, n_steps=args.n_steps)
                _, vm = compute_trajectory_loss(
                    vt, vf, lambda_vs=args.lambda_vs, loss_type=args.loss_type,
                )
                vb = vh.shape[0]
                val_main_sum += vm["main"] * vb
                val_vs_sum += vm["vs"] * vb
                val_count += vb
        val_main = val_main_sum / val_count
        val_vs = val_vs_sum / val_count
        val_loss = val_main + args.lambda_vs * val_vs

        elapsed = time.time() - t0
        nu_val = model.nu.item()
        record = {
            "epoch": epoch, "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "val_main": val_main, "val_vs": val_vs, "val_loss": val_loss,
            "nu": nu_val,
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / "best_model.pt")
        if epoch % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        print(
            f"[{epoch:3d}/{args.epochs}] loss={train_metrics['loss']:.4f} "
            f"main={train_metrics['main']:.4f} vs={train_metrics['vs']:.4f} "
            f"val={val_loss:.4f} nu={nu_val:.3f} ({elapsed:.1f}s)"
        )

    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "epoch": args.epochs, "val_loss": val_loss,
    }, out_dir / "final_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final nu: {model.nu.item():.3f}")


if __name__ == "__main__":
    main()
