#!/usr/bin/env python
"""
226a: Factor-decoupled conditional flow with Variogram Score.

Architecture: v = Λ(cond) @ f + D(cond) ⊙ ε
  - Factor flow: 6-dim affine coupling flow → shared factors f
  - Idiosyncratic flow: 25-dim affine coupling flow → per-cell residuals ε
  - Learned loading matrix Λ(cond): (25, 6), conditioned on history
  - Learned idiosyncratic scale D(cond): (25,), conditioned on history

Loss: Energy Score + λ_vs × Variogram Score

Goal: Fix 212ai's cross-cell correlation (0.647 → >0.85) while preserving
excellent per-cell marginals (24/25 KS). Distribution-free — no Student-t.
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
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import (
    AffineCoupling,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)


class FactorDecoupledFlowModel(nn.Module):
    """Factor-decoupled conditional flow for IV surface scenario generation.

    Generates innovations in asinh-transformed space as:
        v = Λ(cond) @ f + D(cond) ⊙ ε

    where f comes from a small factor flow and ε from an idiosyncratic flow.
    """

    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 75,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        factor_rank: int = 6,
        factor_coupling_layers: int = 3,
        factor_hidden: int = 128,
        idio_coupling_layers: int = 4,
        idio_hidden: int = 192,
        ewma_alpha: float = 0.20,
        scale_floor: float = 1e-4,
        include_scale_feature: bool = True,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = n_cells  # compatibility with OneDayKernelRolloutWrapper
        self.hidden_dim = hidden_dim
        self.factor_rank = factor_rank
        self.ewma_alpha = float(ewma_alpha)
        self.scale_floor = float(scale_floor)
        self.include_scale_feature = bool(include_scale_feature)

        # --- GRU encoder (same as 212ai) ---
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )

        # --- Factor flow (r-dim) ---
        factor_masks = []
        base_f = torch.tensor([(i % 2) for i in range(factor_rank)], dtype=torch.float32)
        for i in range(factor_coupling_layers):
            factor_masks.append(base_f if i % 2 == 0 else 1.0 - base_f)
        self.factor_flow = nn.ModuleList(
            [AffineCoupling(factor_rank, hidden_dim, factor_hidden, m) for m in factor_masks]
        )

        # --- Idiosyncratic flow (25-dim) ---
        idio_masks = []
        base_i = torch.tensor([(i % 2) for i in range(n_cells)], dtype=torch.float32)
        for i in range(idio_coupling_layers):
            idio_masks.append(base_i if i % 2 == 0 else 1.0 - base_i)
        self.idio_flow = nn.ModuleList(
            [AffineCoupling(n_cells, hidden_dim, idio_hidden, m) for m in idio_masks]
        )

        # --- Loading matrix Λ(cond) = Λ_base + MLP(cond) ---
        self.lambda_base = nn.Parameter(torch.zeros(n_cells, factor_rank))
        self.lambda_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_cells * factor_rank),
        )
        # Zero-init MLP final layer so Λ starts as Λ_base
        nn.init.zeros_(self.lambda_mlp[2].weight)
        nn.init.zeros_(self.lambda_mlp[2].bias)

        # --- Idiosyncratic scale D(cond) = softplus(D_bias + Linear(cond)) ---
        self.d_bias = nn.Parameter(torch.zeros(n_cells))
        self.d_linear = nn.Linear(hidden_dim, n_cells)
        nn.init.zeros_(self.d_linear.weight)
        nn.init.zeros_(self.d_linear.bias)

    def init_from_pca(self, lambda_init: np.ndarray, d_init: np.ndarray) -> None:
        """Initialize factor loadings and idiosyncratic scale from PCA."""
        with torch.no_grad():
            self.lambda_base.copy_(torch.from_numpy(lambda_init))
            # Set d_bias so that softplus(d_bias) = d_init
            # softplus(x) = log(1 + exp(x)), inverse: x = log(exp(d) - 1)
            d_target = torch.from_numpy(d_init).clamp_min(0.01)
            self.d_bias.copy_(torch.log(torch.exp(d_target) - 1.0))

    def _expand_condition(self, state: torch.Tensor, n_samples: int) -> torch.Tensor:
        return state.unsqueeze(1).expand(state.shape[0], n_samples, self.hidden_dim)

    def _factor_flow_forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in self.factor_flow:
            x, _ = layer.forward_with_logdet(x, cond)
        return x

    def _idio_flow_forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in self.idio_flow:
            x, _ = layer.forward_with_logdet(x, cond)
        return x

    def get_lambda(self, cond: torch.Tensor) -> torch.Tensor:
        """Compute loading matrix Λ(cond). cond: (B, hidden_dim) → (B, 25, r)"""
        mlp_out = self.lambda_mlp(cond).view(cond.shape[0], self.n_cells, self.factor_rank)
        return self.lambda_base.unsqueeze(0) + mlp_out

    def get_d(self, cond: torch.Tensor) -> torch.Tensor:
        """Compute idiosyncratic scale D(cond). cond: (B, hidden_dim) → (B, 25)"""
        return F.softplus(self.d_bias.unsqueeze(0) + self.d_linear(cond))

    def encode_with_scale(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat, local_scale = build_local_scale_history_features(
            history_01=history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        return h_n[-1], local_scale

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        state, _ = self.encode_with_scale(history_01)
        return state

    def sample_transformed_innovation(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample v in asinh-transformed innovation space.

        Returns:
            v: (B, K, 25)
            local_scale: (B, 25)
            prev: (B, 25)
        """
        state, local_scale = self.encode_with_scale(history_01)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)

        # Expand condition for both flows
        cond = self._expand_condition(state, n_samples)  # (B, K, 128)

        # Factor branch: z_f → f
        z_f = torch.randn(batch, n_samples, self.factor_rank,
                          device=history_01.device, dtype=history_01.dtype)
        f = self._factor_flow_forward(z_f, cond)  # (B, K, r)

        # Idiosyncratic branch: z_eps → eps
        z_eps = torch.randn(batch, n_samples, self.n_cells,
                            device=history_01.device, dtype=history_01.dtype)
        eps = self._idio_flow_forward(z_eps, cond)  # (B, K, 25)

        # Combine: v = Λ(cond) @ f + D(cond) ⊙ eps
        Lambda = self.get_lambda(state)  # (B, 25, r)
        D = self.get_d(state)  # (B, 25)

        # Factor contribution: (B, K, r) × (B, r, 25) → (B, K, 25)
        factor_contrib = torch.einsum("bkr,bcr->bkc", f, Lambda)
        # Idiosyncratic contribution: (B, 1, 25) * (B, K, 25)
        idio_contrib = D.unsqueeze(1) * eps

        v = factor_contrib + idio_contrib
        return v, local_scale, prev

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v, local_scale, prev = self.sample_transformed_innovation(
            history_01, n_samples=n_samples, noise=noise
        )
        innovation = torch.sinh(v)
        delta = innovation * local_scale.unsqueeze(1)
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        return torch.maximum(torch.minimum(delta, upper), lower)

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        return (prev + self.sample_delta(history_01, n_samples=n_samples, noise=noise)).clamp(0.0, 1.0)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
        lambda_vs: float = 0.03,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        v_samples, local_scale, _ = self.sample_transformed_innovation(
            history_01, n_samples=n_samples
        )
        target_v = torch.asinh(target_delta / local_scale.clamp_min(self.scale_floor))

        es = energy_score(v_samples, target_v)
        vs = variogram_score(v_samples, target_v, p=0.5)
        loss = es + lambda_vs * vs

        # Diagnostics
        with torch.no_grad():
            state, _ = self.encode_with_scale(history_01)
            Lambda = self.get_lambda(state)
            D = self.get_d(state)

            # Factor/idio contribution ratio (on random subset)
            z_f_diag = torch.randn(min(8, history_01.shape[0]), 16, self.factor_rank,
                                   device=history_01.device)
            cond_diag = self._expand_condition(state[:min(8, history_01.shape[0])], 16)
            f_diag = self._factor_flow_forward(z_f_diag, cond_diag)
            factor_norm = torch.einsum("bkr,bcr->bkc", f_diag, Lambda[:min(8, history_01.shape[0])]).norm(dim=-1).mean()
            z_e_diag = torch.randn(min(8, history_01.shape[0]), 16, self.n_cells,
                                   device=history_01.device)
            eps_diag = self._idio_flow_forward(z_e_diag, cond_diag)
            idio_norm = (D[:min(8, history_01.shape[0])].unsqueeze(1) * eps_diag).norm(dim=-1).mean()

        sample_delta = torch.sinh(v_samples) * local_scale.unsqueeze(1)
        metrics = {
            "energy": es.detach(),
            "variogram": vs.detach(),
            "sample_v_std": v_samples.std(dim=1).mean().detach(),
            "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
            "factor_norm": factor_norm.detach(),
            "idio_norm": idio_norm.detach(),
            "factor_ratio": (factor_norm / (factor_norm + idio_norm + 1e-8)).detach(),
            "d_mean": D.mean().detach(),
            "lambda_base_norm": self.lambda_base.norm().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[FactorDecoupledFlowModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = FactorDecoupledFlowModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        factor_rank=cfg["factor_rank"],
        factor_coupling_layers=cfg["factor_coupling_layers"],
        factor_hidden=cfg["factor_hidden"],
        idio_coupling_layers=cfg["idio_coupling_layers"],
        idio_hidden=cfg["idio_hidden"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="226a factor-decoupled flow with ES+VS")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    parser.add_argument("--warmstart_encoder", type=str, default=None,
                        help="212ai checkpoint to warm-start GRU encoder from")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_factor", type=float, default=2e-3)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--lambda_vs", type=float, default=0.03)
    parser.add_argument("--factor_rank", type=int, default=6)
    parser.add_argument("--factor_coupling_layers", type=int, default=3)
    parser.add_argument("--factor_hidden", type=int, default=128)
    parser.add_argument("--idio_coupling_layers", type=int, default=4)
    parser.add_argument("--idio_hidden", type=int, default=192)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0,
                        help="Freeze GRU encoder for first N epochs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = FactorDecoupledFlowModel(
        n_cells=25,
        history_feat_dim=75,
        hidden_dim=128,
        gru_layers=2,
        gru_dropout=0.1,
        factor_rank=args.factor_rank,
        factor_coupling_layers=args.factor_coupling_layers,
        factor_hidden=args.factor_hidden,
        idio_coupling_layers=args.idio_coupling_layers,
        idio_hidden=args.idio_hidden,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=True,
    ).to(device)

    # PCA initialization
    pca = np.load(args.pca_init)
    model.init_from_pca(pca["lambda_init"], pca["d_init"])
    print(f"PCA init: factor_rank={pca['factor_rank']}, "
          f"top-{pca['factor_rank']} variance={pca['explained_variance_ratio'][:int(pca['factor_rank'])].sum():.3f}")

    # Warm-start GRU encoder from 212ai
    if args.warmstart_encoder:
        ws_payload = torch.load(args.warmstart_encoder, map_location=device, weights_only=False)
        ws_state = ws_payload["model_state_dict"]
        gru_keys = {k: v for k, v in ws_state.items() if k.startswith("gru.")}
        missing, unexpected = model.load_state_dict(gru_keys, strict=False)
        loaded = len(gru_keys) - len(unexpected)
        print(f"Warm-started GRU encoder: {loaded} params from {args.warmstart_encoder}")

    # Data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[:args.max_train_windows]
    val_indices = val_indices[:args.max_val_windows]

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[:args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target),
                            batch_size=args.batch_size, shuffle=False)

    # Optimizer with parameter groups
    factor_params = (
        list(model.factor_flow.parameters())
        + list(model.lambda_mlp.parameters())
        + [model.lambda_base, model.d_bias]
        + list(model.d_linear.parameters())
    )
    encoder_params = (
        list(model.gru.parameters())
        + list(model.idio_flow.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": factor_params, "lr": args.lr_factor},
        {"params": encoder_params, "lr": args.lr_encoder},
    ], weight_decay=args.weight_decay)

    config = {
        "type": "factor_decoupled_flow_226a",
        "n_cells": 25,
        "history_feat_dim": 75,
        "hidden_dim": 128,
        "gru_layers": 2,
        "gru_dropout": 0.1,
        "factor_rank": args.factor_rank,
        "factor_coupling_layers": args.factor_coupling_layers,
        "factor_hidden": args.factor_hidden,
        "idio_coupling_layers": args.idio_coupling_layers,
        "idio_hidden": args.idio_hidden,
        "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor,
        "include_scale_feature": True,
        "lambda_vs": args.lambda_vs,
        "lr_factor": args.lr_factor,
        "lr_encoder": args.lr_encoder,
        "warmstart_encoder": args.warmstart_encoder,
        "pca_init": args.pca_init,
    }

    print(f"\n226a Factor-Decoupled Flow")
    print(f"  factor_rank={args.factor_rank}, factor_flow={args.factor_coupling_layers}×{args.factor_hidden}")
    print(f"  idio_flow={args.idio_coupling_layers}×{args.idio_hidden}")
    print(f"  lambda_vs={args.lambda_vs}, lr_factor={args.lr_factor}, lr_encoder={args.lr_encoder}")
    print(f"  train={train_hist.shape[0]}, val={val_hist.shape[0]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")

    history_log: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Optionally freeze encoder
        if epoch <= args.freeze_encoder_epochs:
            for p in model.gru.parameters():
                p.requires_grad_(False)
        elif epoch == args.freeze_encoder_epochs + 1 and args.freeze_encoder_epochs > 0:
            for p in model.gru.parameters():
                p.requires_grad_(True)

        model.train()
        running = {
            "loss": 0.0, "energy": 0.0, "variogram": 0.0,
            "sample_v_std": 0.0, "factor_ratio": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(
                history_01, target_01,
                n_samples=args.train_samples,
                lambda_vs=args.lambda_vs,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = history_01.shape[0]
            running["loss"] += loss.item() * b
            running["energy"] += metrics["energy"].item() * b
            running["variogram"] += metrics["variogram"].item() * b
            running["sample_v_std"] += metrics["sample_v_std"].item() * b
            running["factor_ratio"] += metrics["factor_ratio"].item() * b
            count += b

        train_metrics = {k: v / count for k, v in running.items()}

        # Validation
        with torch.no_grad():
            val_metrics = evaluate_h1(
                model, val_loader, q95_threshold, q99_threshold, args.eval_samples
            )

        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "elapsed": elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
            "factor_norm": metrics["factor_norm"].item(),
            "idio_norm": metrics["idio_norm"].item(),
            "factor_ratio": metrics["factor_ratio"].item(),
            "d_mean": metrics["d_mean"].item(),
            "lambda_base_norm": metrics["lambda_base_norm"].item(),
        }
        history_log.append(make_serializable(record))

        # Model selection: coverage + shape + correlation awareness
        score = (
            max(0.0, 0.85 - val_metrics.get("val_coverage_90", 0.0))
            + max(0.0, 0.58 - val_metrics.get("val_realized_q99_coverage_90", 0.0))
            + max(0.0, 0.85 - val_metrics.get("val_h1_quiet_ratio", 0.0))
            + max(0.0, val_metrics.get("val_h1_shoulder_ratio", 1.0) - 1.10)
            + max(0.0, 0.65 - val_metrics.get("val_h1_kurtosis_ratio", 0.0))
        )

        if score < best_score:
            best_score = score
            payload = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "score": score,
                "val_metrics": make_serializable(val_metrics),
            }
            torch.save(payload, out_dir / "best_model.pt")

        if epoch % 2 == 0 or epoch == 1:
            cov = val_metrics.get("val_coverage_90", 0.0)
            es_val = train_metrics["energy"]
            vs_val = train_metrics["variogram"]
            fr = train_metrics["factor_ratio"]
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={train_metrics['loss']:.4f} ES={es_val:.4f} VS={vs_val:.4f} "
                f"cov90={cov:.3f} "
                f"f_ratio={fr:.2f} d={metrics['d_mean'].item():.3f} "
                f"score={score:.4f} "
                f"({elapsed:.1f}s)"
            )

    # Save final
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_metrics": make_serializable(val_metrics),
    }
    torch.save(payload, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"Best score: {best_score:.4f} (lower is better)")


if __name__ == "__main__":
    main()
