#!/usr/bin/env python
"""
212an: 212am + mild deterministic boundary-aware residual scaling.

This keeps the bounded-coordinate support-by-construction flow from 212am, but
reduces residual amplitude near the physical boundaries using a fixed
distance-to-bound multiplier:

  b(prev) = b_min + (1 - b_min) * 4 * prev * (1 - prev)

so:

  y_{t+1} = y_t + m(h_t) + b(prev_t) * s_t * sinh(v_t)

This is meant to preserve the good support behavior from 212am while reducing
the excess width and dependence degradation, without introducing a learned free
scale pathway.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from experiments.backfill.block_ar.train_212am_h1_conditional_flow_logit_drift_staged_nll import (
    ConditionalFlowLogitDriftModel,
)


class ConditionalFlowLogitDriftBoundaryScaleModel(ConditionalFlowLogitDriftModel):
    def __init__(self, *args: Any, boundary_scale_min: float = 0.5, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.boundary_scale_min = float(boundary_scale_min)

    def _boundary_multiplier(self, prev_01: torch.Tensor) -> torch.Tensor:
        interior = (4.0 * prev_01 * (1.0 - prev_01)).clamp(0.0, 1.0)
        return self.boundary_scale_min + (1.0 - self.boundary_scale_min) * interior

    def sample_transformed_residual(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v, local_scale, prev_y, mean_dy = super().sample_transformed_residual(
            history_01, n_samples=n_samples, noise=noise
        )
        prev_01 = self._prev_01(history_01)
        boundary_mult = self._boundary_multiplier(prev_01)
        return v, local_scale, prev_y, mean_dy, boundary_mult

    def sample_next_logit(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v, local_scale, prev_y, mean_dy, boundary_mult = self.sample_transformed_residual(
            history_01,
            n_samples=n_samples,
            noise=noise,
        )
        residual_dy = torch.sinh(v) * (local_scale * boundary_mult).unsqueeze(1)
        return prev_y.unsqueeze(1) + mean_dy.unsqueeze(1) + residual_dy

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
        nll_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev_01 = self._prev_01(history_01)
        prev_y = self._prev_y(history_01)
        target_01_flat = target_01.reshape(history_01.shape[0], self.n_cells)
        target_y = torch.log(target_01_flat.clamp(self.support_eps, 1.0 - self.support_eps)) - torch.log1p(
            -target_01_flat.clamp(self.support_eps, 1.0 - self.support_eps)
        )
        target_delta = target_01_flat - prev_01

        v_samples, local_scale, _prev_y, mean_dy, boundary_mult = self.sample_transformed_residual(
            history_01,
            n_samples=n_samples,
        )
        eff_scale = local_scale * boundary_mult
        residual_dy = torch.sinh(v_samples) * eff_scale.unsqueeze(1)
        sample_y = prev_y.unsqueeze(1) + mean_dy.unsqueeze(1) + residual_dy
        sample_next_01 = torch.sigmoid(sample_y)
        sample_delta = sample_next_01 - prev_01.unsqueeze(1)

        target_v = torch.asinh((target_y - prev_y - mean_dy) / eff_scale.clamp_min(self.scale_floor))
        es = energy_score(sample_delta, target_delta)
        log_prob = self.log_prob_transformed_residual(history_01, target_v)
        nll = -log_prob.mean() / self.n_cells
        loss = es + nll_weight * nll
        metrics = {
            "energy": es.detach(),
            "nll": nll.detach(),
            "sample_y_std": sample_y.std(dim=1).mean().detach(),
            "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
            "mean_abs_dy": mean_dy.abs().mean().detach(),
            "boundary_mult_mean": boundary_mult.mean().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ConditionalFlowLogitDriftBoundaryScaleModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_flow_212an_logit_drift_boundary_scale_staged_nll":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = ConditionalFlowLogitDriftBoundaryScaleModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        flow_hidden=cfg["flow_hidden"],
        n_coupling_layers=cfg["n_coupling_layers"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
        support_eps=cfg["support_eps"],
        boundary_scale_min=cfg["boundary_scale_min"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212an bounded-coordinate flow with boundary-aware residual scaling")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--flow_hidden", type=int, default=256)
    parser.add_argument("--n_coupling_layers", type=int, default=6)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--support_eps", type=float, default=1e-4)
    parser.add_argument("--boundary_scale_min", type=float, default=0.5)
    parser.add_argument("--include_scale_feature", action="store_true", default=True)
    parser.add_argument("--no_scale_feature", action="store_false", dest="include_scale_feature")
    parser.add_argument("--lambda_nll_max", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices = val_indices[: args.max_val_windows]

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    history_feat_dim = 25 + 25 + (25 if args.include_scale_feature else 0)
    model = ConditionalFlowLogitDriftBoundaryScaleModel(
        history_feat_dim=history_feat_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        flow_hidden=args.flow_hidden,
        n_coupling_layers=args.n_coupling_layers,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=args.include_scale_feature,
        support_eps=args.support_eps,
        boundary_scale_min=args.boundary_scale_min,
    ).to(device)
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_payload["model_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212an bounded-coordinate flow with boundary-aware residual scaling")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    if args.init_checkpoint:
        print(f"  init={args.init_checkpoint}")
    print(f"  train_samples={args.train_samples} lambda_nll_max={args.lambda_nll_max} bmin={args.boundary_scale_min}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lambda_nll = args.lambda_nll_max * (epoch / args.epochs)
        model.train()
        running = {
            "loss": 0.0,
            "energy": 0.0,
            "nll": 0.0,
            "sample_y_std": 0.0,
            "sample_delta_std": 0.0,
            "mean_abs_dy": 0.0,
            "boundary_mult_mean": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(
                history_01,
                target_01,
                n_samples=args.train_samples,
                nll_weight=lambda_nll,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["nll"] += float(metrics["nll"].item()) * batch
            running["sample_y_std"] += float(metrics["sample_y_std"].item()) * batch
            running["sample_delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["mean_abs_dy"] += float(metrics["mean_abs_dy"].item()) * batch
            running["boundary_mult_mean"] += float(metrics["boundary_mult_mean"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "lambda_nll": lambda_nll,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_212an_logit_drift_boundary_scale_staged_nll",
                "n_cells": 25,
                "history_feat_dim": history_feat_dim,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "flow_hidden": args.flow_hidden,
                "n_coupling_layers": args.n_coupling_layers,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": args.ewma_alpha,
                "scale_floor": args.scale_floor,
                "include_scale_feature": args.include_scale_feature,
                "support_eps": args.support_eps,
                "boundary_scale_min": args.boundary_scale_min,
                "lambda_nll_max": args.lambda_nll_max,
                "init_checkpoint": args.init_checkpoint,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"lambda={lambda_nll:.4f} "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"trainNLL={train_metrics['train_nll']:.4f}  "
            f"bmean={train_metrics['train_boundary_mult_mean']:.3f}  "
            f"valES={val_metrics['val_energy']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
