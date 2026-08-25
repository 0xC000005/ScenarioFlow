#!/usr/bin/env python
"""
212ak: staged exact-NLL fine-tuning of 212ai with a generic conditional
location-only residual decomposition.

This keeps the fixed causal local scale from 212ai and adds only a bounded
conditional mean path:

  delta = mean_delta(h) + s_t * sinh(v)

where:
  - mean_delta(h) = mean_mult * s_t * tanh(mean_head(h))
  - s_t is the existing causal local scale
  - v is modeled by the same conditional flow

The new mean head is zero-initialized and the model is warm-started from 212ai,
so the initial behavior is identical to 212ai.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import (
    ConditionalFlowLocalScaleAsinhModel,
)


class ConditionalFlowLocalScaleAsinhLocationModel(ConditionalFlowLocalScaleAsinhModel):
    def __init__(
        self,
        *args: Any,
        max_mean_mult: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.max_mean_mult = float(max_mean_mult)
        self.mean_head = nn.Linear(self.hidden_dim, self.n_cells)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def _mean_delta(
        self,
        state: torch.Tensor,
        local_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_raw = self.mean_head(state)
        mean_u = self.max_mean_mult * torch.tanh(mean_raw)
        mean_delta = mean_u * local_scale
        return mean_delta, mean_u

    def sample_transformed_residual(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, local_scale = self.encode_with_scale(history_01)
        mean_delta, _mean_u = self._mean_delta(state, local_scale)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)
        if noise is None:
            z = torch.randn(batch, n_samples, self.n_cells, device=history_01.device, dtype=history_01.dtype)
        else:
            if noise.shape[-1] != self.n_cells:
                raise ValueError(f"Expected noise last dim {self.n_cells}, got {noise.shape[-1]}")
            z = noise.to(device=history_01.device, dtype=history_01.dtype)
        cond = self._expand_condition(state, n_samples)
        v, _logdet = self._flow_forward(z, cond)
        return v, mean_delta, local_scale, prev

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v, mean_delta, local_scale, prev = self.sample_transformed_residual(
            history_01,
            n_samples=n_samples,
            noise=noise,
        )
        delta = mean_delta.unsqueeze(1) + torch.sinh(v) * local_scale.unsqueeze(1)
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        return torch.maximum(torch.minimum(delta, upper), lower)

    def log_prob_transformed_residual(
        self,
        history_01: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        return self.log_prob_transformed_innovation(history_01, target_v)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
        nll_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev

        state, local_scale = self.encode_with_scale(history_01)
        mean_delta, mean_u = self._mean_delta(state, local_scale)

        z = torch.randn(
            history_01.shape[0],
            n_samples,
            self.n_cells,
            device=history_01.device,
            dtype=history_01.dtype,
        )
        cond = self._expand_condition(state, n_samples)
        v_samples, _ = self._flow_forward(z, cond)

        target_v = torch.asinh((target_delta - mean_delta) / local_scale.clamp_min(self.scale_floor))
        es = energy_score(v_samples, target_v)
        log_prob = self.log_prob_transformed_residual(history_01, target_v)
        nll = -log_prob.mean() / self.n_cells
        loss = es + nll_weight * nll

        sample_delta = mean_delta.unsqueeze(1) + torch.sinh(v_samples) * local_scale.unsqueeze(1)
        metrics = {
            "energy": es.detach(),
            "nll": nll.detach(),
            "sample_v_std": v_samples.std(dim=1).mean().detach(),
            "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
            "mean_delta_std": mean_delta.std(dim=0).mean().detach(),
            "mean_u_abs": mean_u.abs().mean().detach(),
            "mean_u_std": mean_u.std(dim=0).mean().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ConditionalFlowLocalScaleAsinhLocationModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_flow_212ak_local_scale_asinh_location_staged_nll":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = ConditionalFlowLocalScaleAsinhLocationModel(
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
        max_mean_mult=cfg["max_mean_mult"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="212ak staged location-only flow fine-tuning from 212ai"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--lambda_nll_max", type=float, default=0.05)
    parser.add_argument("--max_mean_mult", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    cfg = init_payload["config"]
    model = ConditionalFlowLocalScaleAsinhLocationModel(
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
        max_mean_mult=args.max_mean_mult,
    ).to(device)

    init_state = init_payload["model_state_dict"]
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    expected_missing = {"mean_head.weight", "mean_head.bias"}
    if set(missing) != expected_missing:
        raise ValueError(f"Unexpected missing keys when loading init checkpoint: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected keys when loading init checkpoint: {unexpected}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212ak staged location-only flow fine-tuning from 212ai")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(
        f"  train_samples={args.train_samples} lambda_nll_max={args.lambda_nll_max} "
        f"max_mean_mult={args.max_mean_mult}"
    )

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
            "sample_v_std": 0.0,
            "sample_delta_std": 0.0,
            "mean_delta_std": 0.0,
            "mean_u_abs": 0.0,
            "mean_u_std": 0.0,
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
            for key in running:
                if key == "loss":
                    continue
                running[key] += float(metrics[key].item()) * batch
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
                "type": "minimal_h1_conditional_flow_212ak_local_scale_asinh_location_staged_nll",
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": cfg["history_len"],
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "lambda_nll_max": args.lambda_nll_max,
                "max_mean_mult": args.max_mean_mult,
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
            f"mean|u|={train_metrics['train_mean_u_abs']:.4f}  "
            f"valES={val_metrics['val_energy']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage90'] if 'val_realized_q99_coverage90' in val_metrics else val_metrics['val_realized_q99_coverage_90']:.3f}  "
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
