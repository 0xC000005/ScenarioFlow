#!/usr/bin/env python
"""
212r: H=1 asymmetric modulated direct stochastic delta with mean + CVaR energy.

Same architecture as 212e.
Only training objective changes:
  - compute per-window energy score
  - optimize mean(window ES) + lambda * CVaR_alpha(window ES)
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
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import (
    AsymmetricModulatedDirectDeltaModel,
    evaluate_h1,
)


def energy_score_per_window(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-window energy score.

    Args:
        samples: (B, K, D)
        target: (B, D)
    Returns:
        (B,)
    """
    target_dist = torch.linalg.vector_norm(samples - target.unsqueeze(1), dim=-1).mean(dim=1)
    pairwise = torch.cdist(samples, samples).mean(dim=(1, 2))
    return target_dist - 0.5 * pairwise


def empirical_cvar(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    losses = losses.reshape(-1)
    k = max(1, int(math.ceil(alpha * losses.numel())))
    topk = torch.topk(losses, k=k, largest=True, sorted=False).values
    return topk.mean()


def training_loss(
    model: AsymmetricModulatedDirectDeltaModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    n_samples: int,
    cvar_alpha: float,
    cvar_lambda: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    samples, aux = model.sample_delta(history_01, n_samples=n_samples)

    per_window_es = energy_score_per_window(samples, target_delta)
    mean_es = per_window_es.mean()
    cvar_es = empirical_cvar(per_window_es, alpha=cvar_alpha)
    loss = mean_es + cvar_lambda * cvar_es

    mean_delta = samples.mean(dim=1)
    metrics = {
        "mean_es": mean_es.detach(),
        "cvar_es": cvar_es.detach(),
        "mean_abs_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
        "sample_delta_std": samples.std(dim=1).mean().detach(),
        "center_abs_mean": aux["center"].abs().mean().detach(),
        "pos_scale_mean": aux["pos_scale"].mean().detach(),
        "neg_scale_mean": aux["neg_scale"].mean().detach(),
    }
    return loss, metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[AsymmetricModulatedDirectDeltaModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_asymmetric_modulated_direct_delta_212r":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = AsymmetricModulatedDirectDeltaModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        noise_dim=cfg["noise_dim"],
        noise_hidden=cfg["noise_hidden"],
        state_hidden=cfg["state_hidden"],
        delta_scale=torch.tensor(cfg["delta_scale"], dtype=torch.float32),
        center_scale_factor=cfg["center_scale_factor"],
        max_scale_factor=cfg["max_scale_factor"],
        init_scale_factor=cfg["init_scale_factor"],
        modulation_scale=cfg["modulation_scale"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212r H=1 asymmetric modulated direct delta with mean + CVaR energy")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--noise_hidden", type=int, default=128)
    parser.add_argument("--state_hidden", type=int, default=128)
    parser.add_argument("--center_scale_factor", type=float, default=0.5)
    parser.add_argument("--max_scale_factor", type=float, default=2.0)
    parser.add_argument("--init_scale_factor", type=float, default=0.15)
    parser.add_argument("--modulation_scale", type=float, default=0.5)
    parser.add_argument("--cvar_alpha", type=float, default=0.2)
    parser.add_argument("--cvar_lambda", type=float, default=1.0)
    parser.add_argument("--train_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=128)
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
    delta_scale = np.quantile(np.abs(train_delta_np), 0.99, axis=0).astype(np.float32)
    delta_scale = np.clip(delta_scale, 1e-3, None)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    model = AsymmetricModulatedDirectDeltaModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        noise_dim=args.noise_dim,
        noise_hidden=args.noise_hidden,
        state_hidden=args.state_hidden,
        delta_scale=torch.from_numpy(delta_scale),
        center_scale_factor=args.center_scale_factor,
        max_scale_factor=args.max_scale_factor,
        init_scale_factor=args.init_scale_factor,
        modulation_scale=args.modulation_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212r H=1 asymmetric modulated direct delta + mean/CVaR energy")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(
        f"  cvar_alpha={args.cvar_alpha:.3f} "
        f"cvar_lambda={args.cvar_lambda:.3f} "
        f"train_samples={args.train_samples}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "mean_es": 0.0,
            "cvar_es": 0.0,
            "mae": 0.0,
            "delta_std": 0.0,
            "center_abs_mean": 0.0,
            "pos_scale_mean": 0.0,
            "neg_scale_mean": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = training_loss(
                model=model,
                history_01=history_01,
                target_01=target_01,
                n_samples=args.train_samples,
                cvar_alpha=args.cvar_alpha,
                cvar_lambda=args.cvar_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["mean_es"] += float(metrics["mean_es"].item()) * batch
            running["cvar_es"] += float(metrics["cvar_es"].item()) * batch
            running["mae"] += float(metrics["mean_abs_delta_mae"].item()) * batch
            running["delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["center_abs_mean"] += float(metrics["center_abs_mean"].item()) * batch
            running["pos_scale_mean"] += float(metrics["pos_scale_mean"].item()) * batch
            running["neg_scale_mean"] += float(metrics["neg_scale_mean"].item()) * batch
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
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_asymmetric_modulated_direct_delta_212r",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "noise_dim": args.noise_dim,
                "noise_hidden": args.noise_hidden,
                "state_hidden": args.state_hidden,
                "delta_scale": delta_scale.tolist(),
                "center_scale_factor": args.center_scale_factor,
                "max_scale_factor": args.max_scale_factor,
                "init_scale_factor": args.init_scale_factor,
                "modulation_scale": args.modulation_scale,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "cvar_alpha": args.cvar_alpha,
                "cvar_lambda": args.cvar_lambda,
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
            f"loss={train_metrics['train_loss']:.4f}  "
            f"meanES={train_metrics['train_mean_es']:.4f}  "
            f"cvarES={train_metrics['train_cvar_es']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"ps={val_metrics['val_pos_scale_mean']:.4f}  "
            f"ns={val_metrics['val_neg_scale_mean']:.4f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
