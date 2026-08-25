#!/usr/bin/env python
"""
212al: location-only frozen-center calibration on top of 212ai.

This branch keeps the fixed causal local scale and the residual flow from 212ai,
adds the same bounded conditional mean path as 212ak, but calibrates that mean
path in a dedicated frozen stage before any tiny joint fine-tune.
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
from experiments.backfill.block_ar.train_212ak_h1_conditional_flow_local_scale_asinh_location_staged_nll import (
    ConditionalFlowLocalScaleAsinhLocationModel,
)


def _sample_h1_arrays(
    model: ConditionalFlowLocalScaleAsinhLocationModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    batch_size: int,
    eval_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        batch_hist = history_01[start:end]
        with torch.no_grad():
            samp = model.sample_next_iv(batch_hist, n_samples=eval_samples)
        samples.append(samp.detach().cpu().numpy())
    cond_samples = np.concatenate(samples, axis=0).reshape(history_01.shape[0], eval_samples, 1, 5, 5)
    ground_truth = target_01.detach().cpu().numpy().reshape(history_01.shape[0], 1, 5, 5)
    history = history_01.detach().cpu().numpy().reshape(history_01.shape[0], history_01.shape[1], 5, 5)
    return cond_samples, ground_truth, history


def _slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = np.mean((x - x_mean) ** 2)
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    slope = cov_xy / var_x if var_x > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    ss_res = np.mean((y - y_hat) ** 2)
    ss_tot = np.mean((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def _regime_summary(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, float]:
    lower = np.quantile(cond_samples, 0.05, axis=1)[:, 0]
    upper = np.quantile(cond_samples, 0.95, axis=1)[:, 0]
    covered = (ground_truth[:, 0] >= lower) & (ground_truth[:, 0] <= upper)

    mean_iv = history.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = float(np.quantile(vol_of_vol, 0.20))
    q80 = float(np.quantile(vol_of_vol, 0.80))
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    out: dict[str, float] = {}
    for name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        cell_cov = covered[mask].mean(axis=0)
        out[f"{name}_avg"] = float(covered[mask].mean())
        out[f"{name}_worst"] = float(cell_cov.min())
        out[f"{name}_best"] = float(cell_cov.max())
    return out


def _mean_reversion_summary(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    active_slope_threshold: float = 0.05,
) -> dict[str, float]:
    pred_mean = cond_samples.mean(axis=1)
    prev = history[:, -1]
    gt_delta = ground_truth[:, 0] - prev
    pred_delta = pred_mean[:, 0] - prev

    gt_slope, _gt_intercept, _gt_r2 = _slope_intercept_r2(prev, gt_delta)
    pred_slope, _pred_intercept, _pred_r2 = _slope_intercept_r2(prev, pred_delta)
    ratio = pred_slope / gt_slope if abs(gt_slope) > 1e-12 else float("nan")

    gt_cell_slopes = np.zeros((5, 5), dtype=np.float64)
    pred_cell_slopes = np.zeros((5, 5), dtype=np.float64)
    cell_ratio = np.full((5, 5), np.nan, dtype=np.float64)
    cell_sign_match = np.zeros((5, 5), dtype=bool)

    for i in range(5):
        for j in range(5):
            gt_sc, _i1, _r1 = _slope_intercept_r2(prev[:, i, j], gt_delta[:, i, j])
            pred_sc, _i2, _r2 = _slope_intercept_r2(prev[:, i, j], pred_delta[:, i, j])
            gt_cell_slopes[i, j] = gt_sc
            pred_cell_slopes[i, j] = pred_sc
            if abs(gt_sc) > 1e-12:
                cell_ratio[i, j] = pred_sc / gt_sc
            cell_sign_match[i, j] = np.sign(gt_sc) == np.sign(pred_sc)

    active_mask = np.abs(gt_cell_slopes) >= active_slope_threshold
    active_count = int(active_mask.sum())
    if active_count > 0:
        cell_pass = (
            active_mask
            & cell_sign_match
            & np.isfinite(cell_ratio)
            & (cell_ratio >= 0.50)
            & (cell_ratio <= 1.50)
        )
        active_pass_count = int(cell_pass.sum())
        active_pass_rate = float(active_pass_count / active_count)
    else:
        active_pass_count = 0
        active_pass_rate = 1.0

    return {
        "gt_slope": float(gt_slope),
        "pred_slope": float(pred_slope),
        "ratio": float(ratio),
        "active_pass_count": float(active_pass_count),
        "active_count": float(active_count),
        "active_pass_rate": float(active_pass_rate),
    }


def _selection_score(
    val_metrics: dict[str, float],
    regime: dict[str, float],
    mr: dict[str, float],
) -> float:
    score = 0.0
    score += max(0.0, 0.85 - val_metrics["val_coverage_90"])
    score += max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
    score += max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
    score += max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
    score += max(0.0, 0.80 - val_metrics["val_h1_kurtosis_ratio"])
    score += max(0.0, val_metrics["val_h1_kurtosis_ratio"] - 1.25)
    score += max(0.0, 0.70 - regime["calm_worst"])
    score += max(0.0, regime["calm_best"] - 0.95)
    score += max(0.0, 0.70 - regime["turb_worst"])
    score += max(0.0, regime["turb_best"] - 0.95)
    score += max(0.0, 0.70 - mr["ratio"])
    score += max(0.0, mr["ratio"] - 1.30)
    score += max(0.0, 0.70 - mr["active_pass_rate"])
    return float(score)


def _freeze_except_mean(model: ConditionalFlowLocalScaleAsinhLocationModel) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mean_head.parameters():
        param.requires_grad = True


def _unfreeze_all(model: ConditionalFlowLocalScaleAsinhLocationModel) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _calibration_loss(
    model: ConditionalFlowLocalScaleAsinhLocationModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    state, local_scale = model.encode_with_scale(history_01)
    mean_delta, mean_u = model._mean_delta(state, local_scale)
    target_v = torch.asinh((target_delta - mean_delta) / local_scale.clamp_min(model.scale_floor))
    log_prob = model.log_prob_transformed_residual(history_01, target_v)
    nll = -log_prob.mean() / model.n_cells
    metrics = {
        "nll": nll.detach(),
        "mean_delta_std": mean_delta.std(dim=0).mean().detach(),
        "mean_u_abs": mean_u.abs().mean().detach(),
        "mean_u_std": mean_u.std(dim=0).mean().detach(),
    }
    return nll, metrics


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ConditionalFlowLocalScaleAsinhLocationModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_flow_212al_local_scale_asinh_location_frozen_stage":
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
        description="212al frozen location calibration on top of 212ai"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=6)
    parser.add_argument("--phase2_epochs", type=int, default=2)
    parser.add_argument("--phase1_lr", type=float, default=2e-3)
    parser.add_argument("--phase2_lr", type=float, default=2e-5)
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

    total_epochs = args.phase1_epochs + args.phase2_epochs

    print("212al frozen location calibration from 212ai")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(
        f"  phase1_epochs={args.phase1_epochs} phase2_epochs={args.phase2_epochs} "
        f"phase1_lr={args.phase1_lr} phase2_lr={args.phase2_lr} "
        f"lambda_nll_max={args.lambda_nll_max} max_mean_mult={args.max_mean_mult}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}
    _freeze_except_mean(model)
    phase1_optimizer = torch.optim.AdamW(model.mean_head.parameters(), lr=args.phase1_lr, weight_decay=args.weight_decay)
    phase2_optimizer: torch.optim.Optimizer | None = None

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        in_phase1 = epoch <= args.phase1_epochs
        phase_name = "frozen_mean" if in_phase1 else "joint"

        if in_phase1:
            optimizer = phase1_optimizer
            lambda_nll = 1.0
        else:
            if phase2_optimizer is None:
                _unfreeze_all(model)
                phase2_optimizer = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=args.weight_decay)
            optimizer = phase2_optimizer
            phase2_epoch = epoch - args.phase1_epochs
            lambda_nll = args.lambda_nll_max * (phase2_epoch / max(args.phase2_epochs, 1))

        model.train()
        running = {
            "loss": 0.0,
            "nll": 0.0,
            "energy": 0.0,
            "sample_v_std": 0.0,
            "sample_delta_std": 0.0,
            "mean_delta_std": 0.0,
            "mean_u_abs": 0.0,
            "mean_u_std": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            if in_phase1:
                loss, metrics = _calibration_loss(model, history_01, target_01)
                metrics = {
                    "nll": metrics["nll"],
                    "energy": history_01.new_tensor(0.0),
                    "sample_v_std": history_01.new_tensor(0.0),
                    "sample_delta_std": history_01.new_tensor(0.0),
                    "mean_delta_std": metrics["mean_delta_std"],
                    "mean_u_abs": metrics["mean_u_abs"],
                    "mean_u_std": metrics["mean_u_std"],
                }
            else:
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
            running["nll"] += float(metrics["nll"].item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["sample_v_std"] += float(metrics["sample_v_std"].item()) * batch
            running["sample_delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["mean_delta_std"] += float(metrics["mean_delta_std"].item()) * batch
            running["mean_u_abs"] += float(metrics["mean_u_abs"].item()) * batch
            running["mean_u_std"] += float(metrics["mean_u_std"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        cond_samples, ground_truth, history_np = _sample_h1_arrays(
            model,
            val_hist,
            val_target,
            batch_size=args.batch_size,
            eval_samples=args.eval_samples,
        )
        regime = _regime_summary(cond_samples, ground_truth, history_np)
        mr = _mean_reversion_summary(cond_samples, ground_truth, history_np)
        score = _selection_score(val_metrics, regime, mr)

        record = {
            "epoch": epoch,
            "phase": phase_name,
            "elapsed_sec": time.time() - t0,
            "lambda_nll": lambda_nll,
            **train_metrics,
            **val_metrics,
            "val_regime_calm_avg": regime["calm_avg"],
            "val_regime_calm_worst": regime["calm_worst"],
            "val_regime_calm_best": regime["calm_best"],
            "val_regime_turb_avg": regime["turb_avg"],
            "val_regime_turb_worst": regime["turb_worst"],
            "val_regime_turb_best": regime["turb_best"],
            "val_mr_ratio": mr["ratio"],
            "val_mr_active_pass_count": mr["active_pass_count"],
            "val_mr_active_count": mr["active_count"],
            "val_mr_active_pass_rate": mr["active_pass_rate"],
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_212al_local_scale_asinh_location_frozen_stage",
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
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "phase1_lr": args.phase1_lr,
                "phase2_lr": args.phase2_lr,
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
            f"[{epoch:02d}/{total_epochs}] "
            f"phase={phase_name} "
            f"lambda={lambda_nll:.4f} "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"trainNLL={train_metrics['train_nll']:.4f}  "
            f"mean|u|={train_metrics['train_mean_u_abs']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"calmW={regime['calm_worst']:.3f}  "
            f"turbW={regime['turb_worst']:.3f}  "
            f"mr={mr['ratio']:.3f}  "
            f"active={mr['active_pass_rate']:.3f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
