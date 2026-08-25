#!/usr/bin/env python
"""
222a: Fine-tune 212ai with distributional invariant losses for multi-day rollout.

Key idea: instead of per-step energy score (which has supervision mismatch under
scheduled sampling), use losses computed on the model's own free-run rollouts that
enforce statistical invariants — properties that should hold regardless of what
history the model saw.

Loss = lambda_nll * h1_NLL  (anchor: preserve one-step quality)
     + lambda_level * level_stationarity  (level distribution at h=k ≈ h=1)
     + lambda_change * change_stationarity  (change distribution at h=k ≈ h=1)
     + lambda_mr * mean_reversion  (paths that drift far come back)
     + lambda_corr * cross_cell_correlation  (correlation structure preserved)

For 222a: only NLL + level_stationarity enabled. Others added in 222b-d.
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
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)


# ---------------------------------------------------------------------------
# Differentiable free-run rollout
# ---------------------------------------------------------------------------

def differentiable_free_rollout(
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    history_01: torch.Tensor,
    n_paths: int,
    n_steps: int,
    bptt_interval: int = 5,
) -> torch.Tensor:
    """Generate a differentiable multi-step free-run rollout.

    Args:
        model: 212ai-class model (sample_next_iv is differentiable)
        history_01: (B, H, 5, 5) real history in [0,1]
        n_paths: K independent paths per window
        n_steps: T rollout horizon
        bptt_interval: detach history sliding window every N steps

    Returns:
        levels: (B*K, T+1, 25) — h=0 is last real day, h=1..T are generated
    """
    B, H = history_01.shape[0], history_01.shape[1]
    # Expand each window into K independent paths
    hist = (
        history_01.unsqueeze(1)
        .expand(B, n_paths, H, 5, 5)
        .reshape(B * n_paths, H, 5, 5)
        .clone()
    )

    prev = hist[:, -1].reshape(B * n_paths, 25)
    levels = [prev]

    for t in range(n_steps):
        # sample_next_iv: no @torch.no_grad — fully differentiable
        # n_samples=1 gives (B*K, 1, 25) → squeeze to (B*K, 25)
        next_iv = model.sample_next_iv(hist, n_samples=1).squeeze(1)
        levels.append(next_iv)

        # Slide history window
        next_frame = next_iv.view(B * n_paths, 1, 5, 5)
        hist = torch.cat([hist[:, 1:], next_frame], dim=1)

        # Truncate BPTT through the history chain every N steps.
        # The levels list stays on the computation graph —
        # invariant loss gradients still flow through each step's output.
        if (t + 1) % bptt_interval == 0:
            hist = hist.detach()

    return torch.stack(levels, dim=1)  # (B*K, T+1, 25)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def energy_distance_2sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two-sample energy distance. x: (N, D), y: (M, D) -> scalar."""
    cross = torch.cdist(x, y).mean()
    self_x = torch.cdist(x, x).mean()
    self_y = torch.cdist(y, y).mean()
    return 2 * cross - self_x - self_y


def energy_distance_1d_percell(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-cell 1D energy distance, averaged across cells.

    x: (N, C), y: (M, C) where C=25 cells.
    Computes 1D energy distance for each cell independently, then averages.
    Vectorized: no python loop over cells.
    """
    # 1D distance = |a - b|. For each cell, compute pairwise |x_i - y_j|.
    # x[:, c] is (N,), y[:, c] is (M,). |x_i - y_j| = cdist on (N,1) vs (M,1).
    # Vectorize: transpose to (C, N, 1) and (C, M, 1), batch cdist.
    N, C = x.shape
    M = y.shape[0]
    x_t = x.T.unsqueeze(2)  # (C, N, 1)
    y_t = y.T.unsqueeze(2)  # (C, M, 1)
    # Batch pairwise absolute differences
    cross = torch.cdist(x_t, y_t).mean(dim=(1, 2))  # (C,)
    self_x = torch.cdist(x_t, x_t).mean(dim=(1, 2))  # (C,)
    self_y = torch.cdist(y_t, y_t).mean(dim=(1, 2))  # (C,)
    per_cell_ed = 2 * cross - self_x - self_y  # (C,)
    return per_cell_ed.mean()


def level_stationarity_loss(levels: torch.Tensor) -> torch.Tensor:
    """Level distribution at horizon k should match horizon 1.

    levels: (B*K, T+1, 25). h=0 is real last day, h=1 is first generated step.
    Compares h=1 against h=5,10,15,...,T using energy distance.
    """
    T = levels.shape[1] - 1  # number of generated steps
    ref = levels[:, 1, :]  # (B*K, 25) — h=1, anchored by NLL
    eval_horizons = list(range(5, T + 1, 5))  # h=5,10,15,...
    if not eval_horizons:
        eval_horizons = [T] if T > 1 else []
    if not eval_horizons:
        return torch.tensor(0.0, device=levels.device)

    loss = torch.tensor(0.0, device=levels.device)
    for k in eval_horizons:
        loss = loss + energy_distance_2sample(ref, levels[:, k, :])
    return loss / len(eval_horizons)


def gt_anchored_level_loss(
    levels: torch.Tensor, gt_level_pool: torch.Tensor
) -> torch.Tensor:
    """Per-cell 1D energy distance: model's per-horizon levels vs GT unconditional.

    Args:
        levels: (B*K, T+1, 25). h=0 is real last day, h=1..T are generated.
        gt_level_pool: (N_gt, 25) — all training day levels (the unconditional population).
    """
    BK = levels.shape[0]
    T = levels.shape[1] - 1
    eval_horizons = list(range(5, T + 1, 5))  # align with BPTT window ends
    if not eval_horizons:
        eval_horizons = [T] if T >= 1 else [1]

    # Subsample GT to match model sample count (stochastic each call)
    idx = torch.randint(0, gt_level_pool.shape[0], (BK,), device=levels.device)
    gt_sample = gt_level_pool[idx]  # (BK, 25)

    loss = torch.tensor(0.0, device=levels.device)
    for k in eval_horizons:
        loss = loss + energy_distance_1d_percell(levels[:, k, :], gt_sample)
    return loss / len(eval_horizons)


def change_stationarity_loss(levels: torch.Tensor) -> torch.Tensor:
    """Daily change distribution at horizon k should match horizon 1.

    levels: (B*K, T+1, 25).
    """
    changes = levels[:, 1:, :] - levels[:, :-1, :]  # (B*K, T, 25)
    T = changes.shape[1]
    ref = changes[:, 0, :]  # first generated change
    eval_horizons = list(range(5, T, 5))
    if not eval_horizons:
        eval_horizons = [T - 1] if T > 1 else []
    if not eval_horizons:
        return torch.tensor(0.0, device=levels.device)

    loss = torch.tensor(0.0, device=levels.device)
    for k in eval_horizons:
        loss = loss + energy_distance_2sample(ref, changes[:, k, :])
    return loss / len(eval_horizons)


def gt_anchored_change_loss(
    levels: torch.Tensor, gt_change_pool: torch.Tensor
) -> torch.Tensor:
    """Per-cell 1D energy distance: model's per-horizon changes vs GT unconditional changes.

    Args:
        levels: (B*K, T+1, 25). h=0 is real last day, h=1..T are generated.
        gt_change_pool: (N_gt, 25) — all training day-to-day changes (unconditional).
    """
    changes = levels[:, 1:, :] - levels[:, :-1, :]  # (B*K, T, 25)
    BK, T, C = changes.shape
    eval_horizons = list(range(5, T + 1, 5))  # align with BPTT window ends
    if not eval_horizons:
        eval_horizons = [T] if T >= 1 else [1]

    idx = torch.randint(0, gt_change_pool.shape[0], (BK,), device=levels.device)
    gt_sample = gt_change_pool[idx]  # (BK, 25)

    loss = torch.tensor(0.0, device=levels.device)
    for k in eval_horizons:
        loss = loss + energy_distance_1d_percell(changes[:, k - 1, :], gt_sample)
    return loss / len(eval_horizons)


def mean_reversion_loss(
    levels: torch.Tensor, gt_mean_level: torch.Tensor
) -> torch.Tensor:
    """Penalize anti-mean-reverting behavior.

    If deviation from mean is positive and next change is also positive,
    that's anti-mean-reverting. Penalize positive covariance.
    """
    changes = levels[:, 1:, :] - levels[:, :-1, :]  # (B*K, T, 25)
    deviation = levels[:, :-1, :] - gt_mean_level.unsqueeze(0).unsqueeze(0)

    dev_flat = deviation.reshape(-1, 25)
    chg_flat = changes.reshape(-1, 25)

    # Per-cell covariance: should be negative for mean-reverting process
    cov_per_cell = (
        (dev_flat * chg_flat).mean(dim=0)
        - dev_flat.mean(dim=0) * chg_flat.mean(dim=0)
    )
    return torch.relu(cov_per_cell).mean()


def cross_cell_correlation_loss(
    levels: torch.Tensor, gt_cov: torch.Tensor
) -> torch.Tensor:
    """Frobenius distance between generated and GT covariance of daily changes."""
    changes = (levels[:, 1:, :] - levels[:, :-1, :]).reshape(-1, 25)
    centered = changes - changes.mean(dim=0, keepdim=True)
    gen_cov = centered.T @ centered / max(changes.shape[0] - 1, 1)
    return (gen_cov - gt_cov).pow(2).mean()


def h1_nll_loss(
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
) -> torch.Tensor:
    """Same NLL loss as 212ai training — anchor for one-step quality."""
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    state, local_scale = model.encode_with_scale(history_01)
    target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))
    log_prob = model.log_prob_transformed_innovation(history_01, target_v)
    return -log_prob.mean() / model.n_cells


def h1_es_loss(
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Energy score on h1 — additional anchor signal."""
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    v_samples, local_scale, _ = model.sample_transformed_innovation(
        history_01, n_samples=n_samples
    )
    target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))
    return energy_score(v_samples, target_v)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def parse_curriculum(schedule_str: str) -> list[tuple[int, int]]:
    """Parse 'epoch:horizon,...' into sorted list of (epoch, horizon)."""
    pairs = []
    for part in schedule_str.split(","):
        epoch_s, horizon_s = part.strip().split(":")
        pairs.append((int(epoch_s), int(horizon_s)))
    return sorted(pairs)


def get_curriculum_horizon(epoch: int, schedule: list[tuple[int, int]]) -> int:
    """Get rollout horizon for given epoch based on curriculum schedule."""
    horizon = schedule[0][1]
    for ep, h in schedule:
        if epoch >= ep:
            horizon = h
    return horizon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="222a: distributional invariant fine-tuning of 212ai"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_paths", type=int, default=16)
    parser.add_argument("--nll_train_samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_flow", type=float, default=None,
                        help="Flow LR (default: same as --lr). Set lower for differential LR.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bptt_interval", type=int, default=5)

    # Loss weights
    parser.add_argument("--lambda_nll", type=float, default=1.0)
    parser.add_argument("--lambda_level", type=float, default=0.1)
    parser.add_argument("--lambda_change", type=float, default=0.0)
    parser.add_argument("--lambda_mr", type=float, default=0.0)
    parser.add_argument("--lambda_corr", type=float, default=0.0)
    parser.add_argument("--invariant_ramp_epochs", type=int, default=3,
                        help="Ramp invariant loss weights from 0 to full over this many epochs")
    parser.add_argument("--gt_anchored", action="store_true",
                        help="Use GT-anchored per-cell ED instead of self-referential ED")

    # Curriculum
    parser.add_argument("--curriculum_schedule", type=str, default="0:5,9:15,19:30")

    # Eval
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--freeze_flow", action="store_true",
                        help="Freeze flow coupling layers, train only GRU + scale")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
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

    if args.freeze_flow:
        for name, param in model.named_parameters():
            if "layers" in name:  # affine coupling flow layers
                param.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in model.parameters())
        print(f"Froze {n_frozen}/{n_total} parameters (flow layers)")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
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

    train_loader = DataLoader(
        TensorDataset(train_hist, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Precomputed constants for invariant losses
    train_surfaces_flat = surfaces[: args.test_start].reshape(-1, 25)
    gt_mean_level = torch.from_numpy(train_surfaces_flat.mean(axis=0)).to(device)
    gt_level_pool = torch.from_numpy(train_surfaces_flat.astype(np.float32)).to(device)  # (~4500, 25)
    train_changes = np.diff(train_surfaces_flat, axis=0)
    gt_change_pool = torch.from_numpy(train_changes.astype(np.float32)).to(device)  # (~4499, 25)
    gt_cov = torch.from_numpy(np.cov(train_changes.T).astype(np.float32)).to(device)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    if args.lr_flow is not None and not args.freeze_flow:
        flow_params = list(model.layers.parameters())
        flow_ids = {id(p) for p in flow_params}
        other_params = [p for p in model.parameters() if id(p) not in flow_ids and p.requires_grad]
        param_groups = [
            {"params": flow_params, "lr": args.lr_flow},
            {"params": other_params, "lr": args.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"Differential LR: flow={args.lr_flow}, gru={args.lr}")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    curriculum = parse_curriculum(args.curriculum_schedule)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("222a: distributional invariant fine-tuning of 212ai")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  batch_size={args.batch_size} n_paths={args.n_paths}")
    print(f"  lambda_nll={args.lambda_nll} lambda_level={args.lambda_level}")
    print(f"  lambda_change={args.lambda_change} lambda_mr={args.lambda_mr} lambda_corr={args.lambda_corr}")
    print(f"  curriculum={args.curriculum_schedule}")
    print(f"  lr={args.lr} bptt_interval={args.bptt_interval}")

    history_log: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        T_curr = get_curriculum_horizon(epoch, curriculum)

        # Invariant loss weight ramp
        invariant_ramp = min(1.0, epoch / max(args.invariant_ramp_epochs, 1))

        model.train()
        running = {
            "loss": 0.0,
            "nll": 0.0,
            "level_stat": 0.0,
            "change_stat": 0.0,
            "mr": 0.0,
            "corr": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            B = history_01.shape[0]

            # Branch 1: h1 NLL anchor on real data
            nll = h1_nll_loss(model, history_01, target_01)

            # Branch 2: differentiable free-run rollout
            levels = differentiable_free_rollout(
                model, history_01, args.n_paths, T_curr, args.bptt_interval
            )

            # Invariant losses (each gated by lambda)
            if args.lambda_level > 0:
                if args.gt_anchored:
                    l_level = gt_anchored_level_loss(levels, gt_level_pool)
                else:
                    l_level = level_stationarity_loss(levels)
            else:
                l_level = torch.tensor(0.0, device=device)
            if args.lambda_change > 0:
                if args.gt_anchored:
                    l_change = gt_anchored_change_loss(levels, gt_change_pool)
                else:
                    l_change = change_stationarity_loss(levels)
            else:
                l_change = torch.tensor(0.0, device=device)
            l_mr = (
                mean_reversion_loss(levels, gt_mean_level) if args.lambda_mr > 0 else torch.tensor(0.0, device=device)
            )
            l_corr = (
                cross_cell_correlation_loss(levels, gt_cov) if args.lambda_corr > 0 else torch.tensor(0.0, device=device)
            )

            total = (
                args.lambda_nll * nll
                + invariant_ramp * args.lambda_level * l_level
                + invariant_ramp * args.lambda_change * l_change
                + invariant_ramp * args.lambda_mr * l_mr
                + invariant_ramp * args.lambda_corr * l_corr
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running["loss"] += float(total.item()) * B
            running["nll"] += float(nll.item()) * B
            running["level_stat"] += float(l_level.item()) * B
            running["change_stat"] += float(l_change.item()) * B
            running["mr"] += float(l_mr.item()) * B
            running["corr"] += float(l_corr.item()) * B
            count += B

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}

        # Validation: monitor h1 quality
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        # Selection score: prioritize h1 quality + level stationarity
        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
            + train_metrics["train_level_stat"] * 10  # penalize large stationarity gap
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "T_curr": T_curr,
            "invariant_ramp": invariant_ramp,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history_log.append(make_serializable(record))

        # Save checkpoint
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_212ai_local_scale_asinh_staged_nll",
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": cfg.get("history_len", args.history_len),
                "train_samples": args.nll_train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "init_checkpoint": args.init_checkpoint,
                "experiment": "222a_distributional_invariant",
                "lambda_nll": args.lambda_nll,
                "lambda_level": args.lambda_level,
                "lambda_change": args.lambda_change,
                "lambda_mr": args.lambda_mr,
                "lambda_corr": args.lambda_corr,
            },
            "metrics": history_log[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(payload, out_dir / f"checkpoint_ep{epoch}.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"T={T_curr:2d} ramp={invariant_ramp:.2f} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"nll={train_metrics['train_nll']:.4f} "
            f"lvl={train_metrics['train_level_stat']:.4f} "
            f"chg={train_metrics['train_change_stat']:.4f} "
            f"mr={train_metrics['train_mr']:.4f} "
            f"cor={train_metrics['train_corr']:.4f} "
            f"| cov90={val_metrics['val_coverage_90']:.3f} "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f} "
            f"score={score:.3f} "
            f"({time.time() - t0:.1f}s)"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history_log), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")
    print(f"\nDone. Best score: {best_score:.4f}")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
