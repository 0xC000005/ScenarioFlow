#!/usr/bin/env python
"""
223a/223b: Generated-history fine-tuning of 212ai.

Core idea:
- keep the 212ai one-step conditional objective
- augment training with histories whose final 1 day is sampled from the model
- keep the target as the real next day after that history
- optionally add a small GT-anchored rollout regularizer

This is intentionally local:
- k=1 only
- sampled replacements are filtered to remain near-manifold
- no unconditional loss dominates the conditional objective
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
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)
from experiments.backfill.block_ar.train_222a_distributional_invariant_finetuning import (
    differentiable_free_rollout,
    gt_anchored_change_loss,
    gt_anchored_level_loss,
)


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ConditionalFlowLocalScaleAsinhNLLModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] not in {
        "minimal_h1_conditional_flow_212ai_local_scale_asinh_staged_nll",
        "minimal_h1_conditional_flow_223_generated_history",
    }:
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


def parse_curriculum(schedule_str: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in schedule_str.split(","):
        epoch_s, horizon_s = part.strip().split(":")
        pairs.append((int(epoch_s), int(horizon_s)))
    return sorted(pairs)


def get_curriculum_horizon(epoch: int, schedule: list[tuple[int, int]]) -> int:
    horizon = schedule[0][1]
    for ep, h in schedule:
        if epoch >= ep:
            horizon = h
    return horizon


@torch.no_grad()
def build_filtered_augmented_batch(
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    history_01: torch.Tensor,
    real_next_01: torch.Tensor,
    real_next2_01: torch.Tensor,
    n_candidates: int,
    max_std_mae: float,
    max_std_shock: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Construct k=1 generated-history examples.

    For each real history H_t and realized next day x_{t+1}, sample candidate x~_{t+1}
    from the model and keep only near-manifold candidates. The augmented history is
    [H_t[1:], x~_{t+1}] with target x_{t+2}.
    """
    batch = history_01.shape[0]
    prev = history_01[:, -1].reshape(batch, model.n_cells)
    _state, local_scale = model.encode_with_scale(history_01)
    local_scale = local_scale.clamp_min(model.scale_floor)

    candidates = model.sample_next_iv(history_01, n_samples=n_candidates)  # (B, K, 25)
    cand_delta = candidates - prev.unsqueeze(1)
    real_delta = real_next_01 - prev

    std_err = (
        (cand_delta - real_delta.unsqueeze(1)).abs() / local_scale.unsqueeze(1)
    ).mean(dim=-1)  # (B, K)
    std_shock = (cand_delta.abs() / local_scale.unsqueeze(1)).amax(dim=-1)  # (B, K)

    admissible = (std_err <= max_std_mae) & (std_shock <= max_std_shock)
    masked_score = std_err + (~admissible).float() * 1e6
    best_idx = masked_score.argmin(dim=1)
    valid = admissible.any(dim=1)

    if not valid.any():
        stats = {
            "valid_rate": 0.0,
            "mean_std_err": float(std_err.mean().item()),
            "mean_std_shock": float(std_shock.mean().item()),
        }
        return None, None, stats

    row_idx = torch.arange(batch, device=history_01.device)
    chosen = candidates[row_idx, best_idx]

    aug_history = torch.cat(
        [history_01[valid, 1:], chosen[valid].view(-1, 1, 5, 5)],
        dim=1,
    )
    aug_target = real_next2_01[valid]
    stats = {
        "valid_rate": float(valid.float().mean().item()),
        "mean_std_err": float(std_err[row_idx, best_idx].mean().item()),
        "mean_std_shock": float(std_shock[row_idx, best_idx].mean().item()),
    }
    return aug_history, aug_target, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="223a/223b generated-history fine-tuning of 212ai"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--aug_candidates", type=int, default=4)
    parser.add_argument("--max_std_mae", type=float, default=0.75)
    parser.add_argument("--max_std_shock", type=float, default=4.0)
    parser.add_argument("--aug_loss_weight", type=float, default=1.0)
    parser.add_argument("--lambda_rollout", type=float, default=0.0)
    parser.add_argument("--rollout_paths", type=int, default=4)
    parser.add_argument("--rollout_batch_cap", type=int, default=8)
    parser.add_argument("--rollout_bptt_interval", type=int, default=5)
    parser.add_argument("--rollout_curriculum", type=str, default="0:5,4:15,8:30")
    parser.add_argument("--invariant_ramp_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
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

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices = val_indices[: args.max_val_windows]

    train_hist2, train_future2 = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, 2
    )
    val_hist1, val_target1 = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(
        TensorDataset(train_hist2, train_future2),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist1, val_target1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    gt_level_pool = torch.from_numpy(surfaces[: args.test_start].reshape(-1, 25).astype(np.float32)).to(device)
    gt_change_pool = torch.from_numpy(
        np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0).astype(np.float32)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = parse_curriculum(args.rollout_curriculum)

    print("223 generated-history fine-tuning from 212ai")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist2.shape[0]} val_windows={val_hist1.shape[0]}")
    print(f"  train_samples={args.train_samples} aug_candidates={args.aug_candidates}")
    print(f"  max_std_mae={args.max_std_mae} max_std_shock={args.max_std_shock}")
    print(f"  aug_loss_weight={args.aug_loss_weight} lambda_rollout={args.lambda_rollout}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        T_curr = get_curriculum_horizon(epoch, schedule)
        invariant_ramp = min(1.0, epoch / max(args.invariant_ramp_epochs, 1))

        model.train()
        running = {
            "loss": 0.0,
            "clean_loss": 0.0,
            "aug_loss": 0.0,
            "rollout_loss": 0.0,
            "valid_rate": 0.0,
            "mean_std_err": 0.0,
            "mean_std_shock": 0.0,
        }
        count = 0

        for history_01, future2_01 in train_loader:
            target1_01 = future2_01[:, 0, :]
            target2_01 = future2_01[:, 1, :]

            clean_loss, _clean_metrics = model.training_loss(
                history_01,
                target1_01,
                n_samples=args.train_samples,
                nll_weight=args.nll_weight,
            )

            aug_history, aug_target, aug_stats = build_filtered_augmented_batch(
                model=model,
                history_01=history_01,
                real_next_01=target1_01,
                real_next2_01=target2_01,
                n_candidates=args.aug_candidates,
                max_std_mae=args.max_std_mae,
                max_std_shock=args.max_std_shock,
            )
            if aug_history is not None and aug_target is not None and aug_history.shape[0] > 0:
                aug_loss, _aug_metrics = model.training_loss(
                    aug_history,
                    aug_target,
                    n_samples=args.train_samples,
                    nll_weight=args.nll_weight,
                )
            else:
                aug_loss = torch.tensor(0.0, device=device)

            rollout_loss = torch.tensor(0.0, device=device)
            if args.lambda_rollout > 0.0:
                rollout_hist = history_01[: args.rollout_batch_cap]
                levels = differentiable_free_rollout(
                    model=model,
                    history_01=rollout_hist,
                    n_paths=args.rollout_paths,
                    n_steps=T_curr,
                    bptt_interval=args.rollout_bptt_interval,
                )
                rollout_loss = gt_anchored_level_loss(levels, gt_level_pool) + gt_anchored_change_loss(
                    levels, gt_change_pool
                )

            loss = clean_loss + args.aug_loss_weight * aug_loss
            if args.lambda_rollout > 0.0:
                loss = loss + args.lambda_rollout * invariant_ramp * rollout_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["clean_loss"] += float(clean_loss.item()) * batch
            running["aug_loss"] += float(aug_loss.item()) * batch
            running["rollout_loss"] += float(rollout_loss.item()) * batch
            running["valid_rate"] += aug_stats["valid_rate"] * batch
            running["mean_std_err"] += aug_stats["mean_std_err"] * batch
            running["mean_std_shock"] += aug_stats["mean_std_shock"] * batch
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
            "rollout_horizon": T_curr,
            "invariant_ramp": invariant_ramp,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_223_generated_history",
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
                "nll_weight": args.nll_weight,
                "aug_candidates": args.aug_candidates,
                "max_std_mae": args.max_std_mae,
                "max_std_shock": args.max_std_shock,
                "aug_loss_weight": args.aug_loss_weight,
                "lambda_rollout": args.lambda_rollout,
                "rollout_paths": args.rollout_paths,
                "rollout_curriculum": args.rollout_curriculum,
                "init_checkpoint": args.init_checkpoint,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        torch.save(payload, out_dir / f"checkpoint_ep{epoch:02d}.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f} "
            f"clean={train_metrics['train_clean_loss']:.4f} "
            f"aug={train_metrics['train_aug_loss']:.4f} "
            f"roll={train_metrics['train_rollout_loss']:.4f} "
            f"valid={train_metrics['train_valid_rate']:.3f} "
            f"stdErr={train_metrics['train_mean_std_err']:.3f} "
            f"cov90={val_metrics['val_coverage_90']:.3f} "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f} "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f} "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f} "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f} "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
