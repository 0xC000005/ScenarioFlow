#!/usr/bin/env python
"""
220c: rollout-aware fine-tuning of the frozen-best 212ai one-day kernel.

Recipe:
  - initialize from 212ai
  - keep one-step ES+NLL anchor
  - add short-horizon sampled rollout ES on future levels
  - select on a small recursive-rollout validation probe
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

from experiments.backfill.block_ar._rollout_220_utils import (
    OneDayKernelRolloutWrapper,
    RolloutBatch,
    build_rollout_windows,
    evaluate_rollout_subset,
    make_serializable,
    rollout_energy_score_levels,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ConditionalFlowLocalScaleAsinhNLLModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "multiday_rollout_finetune_220c_from_212ai":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = ConditionalFlowLocalScaleAsinhNLLModel(
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
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def build_dataloaders(
    data_path: str,
    history_len: int,
    train_future_len: int,
    eval_future_len: int,
    test_start: int,
    val_size: int,
    max_train_windows: int,
    max_val_windows: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, RolloutBatch]:
    train_batch = build_rollout_windows(
        data_path=data_path,
        history_len=history_len,
        future_len=train_future_len,
        test_start=test_start,
        val_size=val_size,
        max_windows=max_train_windows,
        device=device,
        split="train",
    )
    val_batch = build_rollout_windows(
        data_path=data_path,
        history_len=history_len,
        future_len=eval_future_len,
        test_start=test_start,
        val_size=val_size,
        max_windows=max_val_windows,
        device=device,
        split="val",
    )
    train_loader = DataLoader(
        TensorDataset(train_batch.history_01, train_batch.future_01),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, val_batch


def parse_horizons(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="220c rollout-aware fine-tune from 212ai")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--train_future_len", type=int, default=10)
    parser.add_argument("--eval_future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--anchor_samples", type=int, default=64)
    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--rollout_chunk_size", type=int, default=4)
    parser.add_argument("--rollout_weight_max", type=float, default=0.15)
    parser.add_argument("--rollout_horizons", type=str, default="5,10")
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--val_rollout_samples", type=int, default=32)
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
    model = ConditionalFlowLocalScaleAsinhNLLModel(
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
    ).to(device)
    model.load_state_dict(init_payload["model_state_dict"])

    train_loader, val_batch = build_dataloaders(
        data_path=args.data_path,
        history_len=args.history_len,
        train_future_len=args.train_future_len,
        eval_future_len=args.eval_future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        batch_size=args.batch_size,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rollout_horizons = parse_horizons(args.rollout_horizons)

    print("220c rollout-aware fine-tuning from 212ai")
    print(f"  init={args.init_checkpoint}")
    print(
        f"  train_windows={len(train_loader.dataset)} val_windows={val_batch.history_01.shape[0]} "
        f"train_future_len={args.train_future_len} eval_future_len={args.eval_future_len}"
    )
    print(
        f"  anchor_samples={args.anchor_samples} rollout_samples={args.rollout_samples} "
        f"rollout_weight_max={args.rollout_weight_max} rollout_horizons={rollout_horizons}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        rollout_weight = args.rollout_weight_max * (epoch / args.epochs)
        model.train()
        running = {
            "loss": 0.0,
            "anchor": 0.0,
            "anchor_energy": 0.0,
            "anchor_nll": 0.0,
            "rollout_energy": 0.0,
            "rollout_terminal_std": 0.0,
        }
        count = 0

        for history_01, future_01 in train_loader:
            target_1 = future_01[:, 0].reshape(history_01.shape[0], -1)
            anchor_loss, anchor_metrics = model.training_loss(
                history_01,
                target_1,
                n_samples=args.anchor_samples,
                nll_weight=args.nll_weight,
            )
            rollout_loss, rollout_metrics = rollout_energy_score_levels(
                model=model,
                history_01=history_01,
                future_01=future_01,
                n_samples=args.rollout_samples,
                rollout_horizons=rollout_horizons,
                chunk_size=args.rollout_chunk_size,
            )
            loss = anchor_loss + rollout_weight * rollout_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["anchor"] += float(anchor_loss.item()) * batch
            running["anchor_energy"] += float(anchor_metrics["energy"].item()) * batch
            running["anchor_nll"] += float(anchor_metrics["nll"].item()) * batch
            running["rollout_energy"] += float(rollout_loss.item()) * batch
            running["rollout_terminal_std"] += float(rollout_metrics["rollout_terminal_std"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        wrapper = OneDayKernelRolloutWrapper(model).eval()
        val_metrics = evaluate_rollout_subset(
            wrapper=wrapper,
            history_norm=val_batch.history_norm,
            future_01=val_batch.future_01,
            batch_size=args.batch_size,
            n_samples=args.val_rollout_samples,
            n_steps=args.eval_future_len,
            chunk_size=min(args.rollout_chunk_size, args.val_rollout_samples),
        )

        score = (
            max(0.0, 0.75 - val_metrics.get("cov90_h10", 0.0))
            + max(0.0, 0.65 - val_metrics.get("cov90_h30", 0.0))
            + max(0.0, 1.15 - val_metrics.get("turb_calm_ratio_h30", 0.0))
            + 8.0 * max(0.0, val_metrics.get("at_floor_h30", 0.0) - 0.02)
            + 4.0 * max(0.0, val_metrics.get("at_ceiling_h30", 0.0) - 0.01)
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "rollout_weight": rollout_weight,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "multiday_rollout_finetune_220c_from_212ai",
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": args.history_len,
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "anchor_samples": args.anchor_samples,
                "rollout_samples": args.rollout_samples,
                "rollout_horizons": rollout_horizons,
                "rollout_weight_max": args.rollout_weight_max,
                "nll_weight": args.nll_weight,
                "init_checkpoint": args.init_checkpoint,
                "train_future_len": args.train_future_len,
                "eval_future_len": args.eval_future_len,
            },
            "metrics": history[-1],
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"rw={rollout_weight:.3f} "
            f"train_anchor={train_metrics['train_anchor']:.4f} "
            f"train_roll={train_metrics['train_rollout_energy']:.4f} "
            f"h10cov={val_metrics.get('cov90_h10', float('nan')):.3f} "
            f"h30cov={val_metrics.get('cov90_h30', float('nan')):.3f} "
            f"turb/calm h30={val_metrics.get('turb_calm_ratio_h30', float('nan')):.3f} "
            f"floor h30={val_metrics.get('at_floor_h30', float('nan')):.3%} "
            f"score={score:.3f}"
        )

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
