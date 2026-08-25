#!/usr/bin/env python
"""
212g: H=1 asymmetric modulated direct stochastic delta model with scale-head supervision.

Same architecture as 212e.
Only training objective changes:
  - keep energy score on direct delta samples
  - add direct auxiliary supervision on pos_scale / neg_scale
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import (
    AsymmetricModulatedDirectDeltaModel,
    evaluate_h1,
)


def training_loss(
    model: AsymmetricModulatedDirectDeltaModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    n_samples: int,
    scale_aux_weight: float,
    huber_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    sample_delta, aux = model.sample_delta(history_01, n_samples=n_samples)

    es = energy_score(sample_delta, target_delta)

    norm = model.delta_scale.view(1, model.n_cells).clamp_min(1e-6)
    max_scale = model.max_scale_factor * model.delta_scale.view(1, model.n_cells)
    center_detached = aux["center"].detach()
    target_pos = torch.minimum(torch.relu(target_delta - center_detached), max_scale)
    target_neg = torch.minimum(torch.relu(center_detached - target_delta), max_scale)

    pos_loss = F.smooth_l1_loss(aux["pos_scale"] / norm, target_pos / norm, beta=huber_beta)
    neg_loss = F.smooth_l1_loss(aux["neg_scale"] / norm, target_neg / norm, beta=huber_beta)
    scale_aux = 0.5 * (pos_loss + neg_loss)
    loss = es + scale_aux_weight * scale_aux

    mean_delta = sample_delta.mean(dim=1)
    metrics = {
        "energy": es.detach(),
        "scale_aux": scale_aux.detach(),
        "pos_loss": pos_loss.detach(),
        "neg_loss": neg_loss.detach(),
        "mean_abs_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
        "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
        "center_abs_mean": aux["center"].abs().mean().detach(),
        "pos_scale_mean": aux["pos_scale"].mean().detach(),
        "neg_scale_mean": aux["neg_scale"].mean().detach(),
        "target_pos_mean": target_pos.mean().detach(),
        "target_neg_mean": target_neg.mean().detach(),
    }
    return loss, metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[AsymmetricModulatedDirectDeltaModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_asymmetric_modulated_direct_delta_212g":
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
    parser = argparse.ArgumentParser(description="212g H=1 asymmetric modulated direct delta with direct scale-head supervision")
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
    parser.add_argument("--scale_aux_weight", type=float, default=0.2)
    parser.add_argument("--huber_beta", type=float, default=0.25)
    parser.add_argument("--train_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=256)
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

    print("212g H=1 asymmetric modulated direct delta + scale-head supervision")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(
        f"  scale_aux_weight={args.scale_aux_weight:.3f} "
        f"huber_beta={args.huber_beta:.3f} train_samples={args.train_samples}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "energy": 0.0,
            "scale_aux": 0.0,
            "pos_loss": 0.0,
            "neg_loss": 0.0,
            "mae": 0.0,
            "delta_std": 0.0,
            "center_abs_mean": 0.0,
            "pos_scale_mean": 0.0,
            "neg_scale_mean": 0.0,
            "target_pos_mean": 0.0,
            "target_neg_mean": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = training_loss(
                model=model,
                history_01=history_01,
                target_01=target_01,
                n_samples=args.train_samples,
                scale_aux_weight=args.scale_aux_weight,
                huber_beta=args.huber_beta,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["scale_aux"] += float(metrics["scale_aux"].item()) * batch
            running["pos_loss"] += float(metrics["pos_loss"].item()) * batch
            running["neg_loss"] += float(metrics["neg_loss"].item()) * batch
            running["mae"] += float(metrics["mean_abs_delta_mae"].item()) * batch
            running["delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["center_abs_mean"] += float(metrics["center_abs_mean"].item()) * batch
            running["pos_scale_mean"] += float(metrics["pos_scale_mean"].item()) * batch
            running["neg_scale_mean"] += float(metrics["neg_scale_mean"].item()) * batch
            running["target_pos_mean"] += float(metrics["target_pos_mean"].item()) * batch
            running["target_neg_mean"] += float(metrics["target_neg_mean"].item()) * batch
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
                "type": "minimal_h1_asymmetric_modulated_direct_delta_212g",
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
                "scale_aux_weight": args.scale_aux_weight,
                "huber_beta": args.huber_beta,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
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
            f"es={train_metrics['train_energy']:.4f}  "
            f"aux={train_metrics['train_scale_aux']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"center={val_metrics['val_center_abs_mean']:.4f}  "
            f"ps={val_metrics['val_pos_scale_mean']:.4f}  "
            f"ns={val_metrics['val_neg_scale_mean']:.4f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
