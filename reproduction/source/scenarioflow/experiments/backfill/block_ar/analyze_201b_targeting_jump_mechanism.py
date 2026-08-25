#!/usr/bin/env python
"""
Mechanistic review of 201b conditional targeting, jump realism, and encoder discrimination.

Questions:
  1. Is weak S3/S7 targeting caused by a weak encoder, or by poor use of encoded information?
  2. Why does S11 still fail even when some local sparse-case metrics improved?
  3. Why is the generated law still too shoulder-heavy and not quiet enough?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import ShapeScaleStudentTARModel
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TransformerRolloutTailStudentTARModel,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def build_test_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(data_path)
    dataset = VolSurfaceDataset(
        data["surface"],
        history_len,
        future_len,
        start_idx=test_start,
    )
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    histories = []
    futures = []
    for i in range(n):
        item = dataset[i]
        histories.append(item["history"])
        futures.append(item["future"])
    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    return vov, calm, turb, q20, q80


def safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def safe_median(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.median(values[mask]))


def auc_roc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fit_binary_probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    epochs: int = 400,
    lr: float = 5e-2,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    order = rng.permutation(n)
    split = max(int(0.7 * n), 1)
    train_idx = order[:split]
    test_idx = order[split:]
    x = features.astype(np.float32)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx].astype(np.float32))
    x_test = torch.from_numpy(x[test_idx])
    y_test = labels[test_idx].astype(np.int64)

    probe = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        logits = probe(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_scores = probe(x_train).squeeze(-1).numpy()
        test_scores = probe(x_test).squeeze(-1).numpy()

    return {
        "train_auc": auc_roc_binary(labels[train_idx], train_scores),
        "test_auc": auc_roc_binary(y_test, test_scores),
        "train_pos_rate": float(labels[train_idx].mean()),
        "test_pos_rate": float(y_test.mean()) if y_test.size else float("nan"),
    }


def fit_multiclass_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    epochs: int = 500,
    lr: float = 5e-2,
    n_classes: int = 25,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    order = rng.permutation(n)
    split = max(int(0.7 * n), 1)
    train_idx = order[:split]
    test_idx = order[split:]
    x = features.astype(np.float32)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64))
    x_test = torch.from_numpy(x[test_idx])
    y_test = labels[test_idx].astype(np.int64)

    probe = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_logits = probe(x_train)
        test_logits = probe(x_test)
        train_pred = train_logits.argmax(dim=1).numpy()
        test_pred = test_logits.argmax(dim=1).numpy()
        train_top3 = train_logits.topk(k=min(3, n_classes), dim=1).indices.numpy()
        test_top3 = test_logits.topk(k=min(3, n_classes), dim=1).indices.numpy()

    majority = np.bincount(labels[train_idx], minlength=n_classes).argmax()
    return {
        "train_top1_acc": float((train_pred == labels[train_idx]).mean()),
        "test_top1_acc": float((test_pred == y_test).mean()) if y_test.size else float("nan"),
        "test_top3_acc": float(np.mean([y in top for y, top in zip(y_test, test_top3)])) if y_test.size else float("nan"),
        "chance_top1": 1.0 / float(n_classes),
        "majority_top1": float((y_test == majority).mean()) if y_test.size else float("nan"),
        "n_classes": int(n_classes),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
    }


def extract_features(model, history_01: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    feats = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        with torch.no_grad():
            enc = model.encode(hist)
            if isinstance(enc, tuple):
                enc = enc[0]
            feats.append(enc.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    model_type = raw_config["type"]
    if model_type == "multi_step_student_t_169c":
        model = ShapeScaleStudentTARModel(
            encoder_config=EncoderConfig(**raw_config["encoder"]),
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
            cov_jitter=raw_config.get("cov_jitter", 1e-4),
        )
    elif model_type in {
        "transformer_ar_rollout_tail_student_t_201a",
        "transformer_underfit_aware_selffed_rollout_student_t_201b",
    }:
        model = TransformerRolloutTailStudentTARModel(
            encoder_config=raw_config["encoder"],
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
            cov_jitter=raw_config.get("cov_jitter", 1e-4),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, model_type, payload


def sample_next_iv_batched(
    model,
    history_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        samp = model.sample_next_iv(hist, n_samples=n_samples).detach().cpu().numpy()
        outs.append(samp)
    return np.concatenate(outs, axis=0)


def sample_teacher_forced_horizon(
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    hidx: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        fut = future_01[start:end].to(device)
        if hidx > 0:
            tf_hist = torch.cat([hist[:, hidx:], fut[:, :hidx]], dim=1)
        else:
            tf_hist = hist
        samp = model.sample_next_iv(tf_hist, n_samples=n_samples).detach().cpu().numpy()
        outs.append(samp)
    return np.concatenate(outs, axis=0)


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(np.asarray(sample_a, dtype=np.float64))
    b = np.sort(np.asarray(sample_b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def summarize_jump_distribution(
    history_flat: np.ndarray,
    future_flat: np.ndarray,
    rollout_flat: np.ndarray,
    q95: float,
    q99: float,
    subsets: dict[str, np.ndarray],
) -> dict[str, Any]:
    prev = history_flat[:, -1:, :]
    gt_path = np.concatenate([prev, future_flat], axis=1)
    gt_abs_step = np.abs(np.diff(gt_path, axis=1))
    gt_max_jump = gt_abs_step.reshape(gt_abs_step.shape[0], -1).max(axis=1)
    gt_q95_count = (gt_abs_step >= q95).reshape(gt_abs_step.shape[0], -1).sum(axis=1)
    gt_q99_count = (gt_abs_step >= q99).reshape(gt_abs_step.shape[0], -1).sum(axis=1)
    gt_any_q95 = gt_q95_count > 0
    gt_any_q99 = gt_q99_count > 0
    gt_max_idx = gt_abs_step.reshape(gt_abs_step.shape[0], -1).argmax(axis=1)
    gt_max_h = gt_max_idx // gt_abs_step.shape[2]
    gt_max_cell = gt_max_idx % gt_abs_step.shape[2]

    prev_ro = np.repeat(prev[:, None, :, :], rollout_flat.shape[1], axis=1)
    ro_path = np.concatenate([prev_ro, rollout_flat], axis=2)
    ro_abs_step = np.abs(np.diff(ro_path, axis=2))
    ro_flat = ro_abs_step.reshape(ro_abs_step.shape[0], ro_abs_step.shape[1], -1)
    ro_max_jump = ro_flat.max(axis=2)
    ro_q95_count = (ro_flat >= q95).sum(axis=2)
    ro_q99_count = (ro_flat >= q99).sum(axis=2)
    ro_any_q95 = ro_q95_count > 0
    ro_any_q99 = ro_q99_count > 0
    ro_max_idx = ro_flat.argmax(axis=2)
    ro_max_h = ro_max_idx // gt_abs_step.shape[2]
    ro_max_cell = ro_max_idx % gt_abs_step.shape[2]

    out: dict[str, Any] = {}
    for name, mask in subsets.items():
        if not np.any(mask):
            out[name] = {}
            continue
        gt_vals = gt_max_jump[mask]
        ro_vals = ro_max_jump[mask].reshape(-1)
        gt_h = gt_max_h[mask]
        ro_h = ro_max_h[mask].reshape(-1)
        gt_c = gt_max_cell[mask]
        ro_c = ro_max_cell[mask].reshape(-1)
        out[name] = {
            "ks_stat": ks_statistic(gt_vals, ro_vals),
            "gt_median": float(np.median(gt_vals)),
            "ro_median": float(np.median(ro_vals)),
            "q90_ratio": float(np.quantile(ro_vals, 0.9) / max(np.quantile(gt_vals, 0.9), 1e-8)),
            "q99_ratio": float(np.quantile(ro_vals, 0.99) / max(np.quantile(gt_vals, 0.99), 1e-8)),
            "gt_any_q95": float(gt_any_q95[mask].mean()),
            "ro_any_q95": float(ro_any_q95[mask].mean()),
            "gt_any_q99": float(gt_any_q99[mask].mean()),
            "ro_any_q99": float(ro_any_q99[mask].mean()),
            "gt_q95_count_mean": float(gt_q95_count[mask].mean()),
            "ro_q95_count_mean": float(ro_q95_count[mask].mean()),
            "gt_q99_count_mean": float(gt_q99_count[mask].mean()),
            "ro_q99_count_mean": float(ro_q99_count[mask].mean()),
            "max_horizon_mode_gt": int(np.bincount(gt_h, minlength=future_flat.shape[1]).argmax()),
            "max_horizon_mode_ro": int(np.bincount(ro_h, minlength=future_flat.shape[1]).argmax()),
            "max_cell_mode_gt": int(np.bincount(gt_c, minlength=future_flat.shape[2]).argmax()),
            "max_cell_mode_ro": int(np.bincount(ro_c, minlength=future_flat.shape[2]).argmax()),
        }
    return out


def summarize_abs_delta_shape(
    history_flat: np.ndarray,
    future_flat: np.ndarray,
    rollout_flat: np.ndarray,
    subsets: dict[str, np.ndarray],
) -> dict[str, Any]:
    prev = history_flat[:, -1:, :]
    gt_abs = np.abs(np.diff(np.concatenate([prev, future_flat], axis=1), axis=1)).reshape(history_flat.shape[0], -1)
    prev_ro = np.repeat(prev[:, None, :, :], rollout_flat.shape[1], axis=1)
    ro_abs = np.abs(np.diff(np.concatenate([prev_ro, rollout_flat], axis=2), axis=2))
    ro_abs = ro_abs.reshape(history_flat.shape[0], rollout_flat.shape[1], -1)
    gt_global = gt_abs.reshape(-1)
    quantile_levels = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]

    out: dict[str, Any] = {}
    for name, mask in subsets.items():
        if not np.any(mask):
            out[name] = {}
            continue
        gt_vals = gt_abs[mask].reshape(-1)
        ro_vals = ro_abs[mask].reshape(-1)
        gt_q = {str(q): float(np.quantile(gt_vals, q)) for q in quantile_levels}
        ro_q = {str(q): float(np.quantile(ro_vals, q)) for q in quantile_levels}
        gt_top1 = gt_vals[gt_vals >= np.quantile(gt_vals, 0.99)]
        ro_top1 = ro_vals[ro_vals >= np.quantile(ro_vals, 0.99)]
        gt_top01 = gt_vals[gt_vals >= np.quantile(gt_vals, 0.999)]
        ro_top01 = ro_vals[ro_vals >= np.quantile(ro_vals, 0.999)]
        out[name] = {
            "quantiles_gt": gt_q,
            "quantiles_ro": ro_q,
            "quantile_ratios": {k: float(ro_q[k] / max(gt_q[k], 1e-8)) for k in gt_q},
            "quiet_ratio": float((ro_vals <= np.quantile(gt_global, 0.5)).mean() / max((gt_vals <= np.quantile(gt_global, 0.5)).mean(), 1e-8)),
            "shoulder_ratio": float((((ro_vals > np.quantile(gt_global, 0.5)) & (ro_vals <= np.quantile(gt_global, 0.95))).mean()) / max((((gt_vals > np.quantile(gt_global, 0.5)) & (gt_vals <= np.quantile(gt_global, 0.95))).mean()), 1e-8)),
            "extreme_ratio": float((ro_vals > np.quantile(gt_global, 0.99)).mean() / max((gt_vals > np.quantile(gt_global, 0.99)).mean(), 1e-8)),
            "top1_mean_amplitude_ratio": float(ro_top1.mean() / max(gt_top1.mean(), 1e-8)),
            "top01_mean_amplitude_ratio": float(ro_top01.mean() / max(gt_top01.mean(), 1e-8)),
        }
    return out


def summarize_targeting(
    future_flat: np.ndarray,
    tf_h30: np.ndarray,
    ro_h30: np.ndarray,
    hard_late: np.ndarray,
    q95: float,
    q99: float,
) -> dict[str, Any]:
    prev = future_flat[:, 28, :]
    target = future_flat[:, 29, :]
    target_abs_delta = np.abs(target - prev)
    realized_q95 = target_abs_delta >= q95
    realized_q99 = target_abs_delta >= q99
    top_target_cell = target_abs_delta.argmax(axis=1)
    tf_width = np.quantile(tf_h30, 0.95, axis=1) - np.quantile(tf_h30, 0.05, axis=1)
    ro_width = np.quantile(ro_h30, 0.95, axis=1) - np.quantile(ro_h30, 0.05, axis=1)
    tf_top = tf_width.argmax(axis=1)
    ro_top = ro_width.argmax(axis=1)
    tf_top3 = np.argsort(tf_width, axis=1)[:, -3:]
    ro_top3 = np.argsort(ro_width, axis=1)[:, -3:]
    tf_eff_support = np.exp(-(np.clip(tf_width / np.maximum(tf_width.sum(axis=1, keepdims=True), 1e-8), 1e-8, None) * np.log(np.clip(tf_width / np.maximum(tf_width.sum(axis=1, keepdims=True), 1e-8), 1e-8, None))).sum(axis=1))
    ro_eff_support = np.exp(-(np.clip(ro_width / np.maximum(ro_width.sum(axis=1, keepdims=True), 1e-8), 1e-8, None) * np.log(np.clip(ro_width / np.maximum(ro_width.sum(axis=1, keepdims=True), 1e-8), 1e-8, None))).sum(axis=1))

    hard_idx = np.where(hard_late)[0]
    if hard_idx.size == 0:
        return {}

    def mean_topk_hit(topk: np.ndarray, mask_2d: np.ndarray) -> float:
        hits = []
        for row, cells in zip(mask_2d, topk):
            hits.append(np.any(row[cells]))
        return float(np.mean(hits))

    return {
        "hard_windows": int(hard_idx.size),
        "teacher_forced": {
            "top1_matches_true_top1": float((tf_top[hard_late] == top_target_cell[hard_late]).mean()),
            "top3_hits_realized_q95": mean_topk_hit(tf_top3[hard_late], realized_q95[hard_late]),
            "top3_hits_realized_q99": mean_topk_hit(tf_top3[hard_late], realized_q99[hard_late]),
            "width_on_true_top1_cell": float(tf_width[hard_late, top_target_cell[hard_late]].mean()),
            "width_on_realized_q99_cells": safe_mean(realized_q99[hard_late], tf_width[hard_late]),
            "effective_support": float(tf_eff_support[hard_late].mean()),
        },
        "rollout": {
            "top1_matches_true_top1": float((ro_top[hard_late] == top_target_cell[hard_late]).mean()),
            "top3_hits_realized_q95": mean_topk_hit(ro_top3[hard_late], realized_q95[hard_late]),
            "top3_hits_realized_q99": mean_topk_hit(ro_top3[hard_late], realized_q99[hard_late]),
            "width_on_true_top1_cell": float(ro_width[hard_late, top_target_cell[hard_late]].mean()),
            "width_on_realized_q99_cells": safe_mean(realized_q99[hard_late], ro_width[hard_late]),
            "effective_support": float(ro_eff_support[hard_late].mean()),
        },
    }


def evaluate_model(
    name: str,
    checkpoint_path: str,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    q95: float,
    q99: float,
    calm: np.ndarray,
    turb: np.ndarray,
    hard_late: np.ndarray,
    clean_late_turb: np.ndarray,
    batch_size: int,
    h1_samples: int,
    h30_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    model, model_type, payload = load_model(checkpoint_path, device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1).numpy()
    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1).numpy()

    h1_samples_arr = sample_next_iv_batched(model, history_01, n_samples=h1_samples, batch_size=batch_size, device=device)
    tf_h30 = sample_teacher_forced_horizon(
        model,
        history_01,
        future_01,
        hidx=29,
        n_samples=h30_samples,
        batch_size=batch_size,
        device=device,
    )
    rollout = model.sample_batched(
        history_norm.to(device),
        n_samples=h30_samples,
        n_steps=30,
        chunk_size=min(batch_size, h30_samples),
    ).detach().cpu().numpy()
    ro_h30 = rollout[:, :, 29].reshape(history_01.shape[0], h30_samples, -1)

    features = extract_features(model, history_01, batch_size=batch_size, device=device)
    window_probe = fit_binary_probe_auc(features, hard_late.astype(np.int64))

    top_target_cell = np.abs(future_flat[:, 29, :] - future_flat[:, 28, :]).argmax(axis=1)
    if int(hard_late.sum()) >= 10:
        cell_probe = fit_multiclass_probe(features[hard_late], top_target_cell[hard_late], n_classes=25)
    else:
        cell_probe = {}

    subsets = {
        "all": np.ones(history_01.shape[0], dtype=bool),
        "calm": calm,
        "turb": turb,
        "hard_late": hard_late,
        "clean_late_turb": clean_late_turb,
    }

    out = {
        "model_name": name,
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "epoch": int(payload.get("epoch", -1)),
        "encoder_window_probe_hard_late": window_probe,
        "encoder_top_cell_probe_h30": cell_probe,
        "targeting_h30": summarize_targeting(future_flat, tf_h30, ro_h30, hard_late, q95, q99),
        "pathwise_jump_realism": summarize_jump_distribution(history_flat, future_flat, rollout.reshape(history_01.shape[0], h30_samples, 30, -1), q95, q99, subsets),
        "abs_delta_shape": summarize_abs_delta_shape(history_flat, future_flat, rollout.reshape(history_01.shape[0], h30_samples, 30, -1), subsets),
    }

    if model_type in {
        "transformer_ar_rollout_tail_student_t_201a",
        "transformer_underfit_aware_selffed_rollout_student_t_201b",
    }:
        with torch.no_grad():
            attns = []
            for start in range(0, history_01.shape[0], batch_size):
                end = min(start + batch_size, history_01.shape[0])
                _, attn = model.encode(history_01[start:end].to(device), return_attention=True)
                attns.append(attn.detach().cpu().numpy())
            attn = np.concatenate(attns, axis=0)
        out["attention"] = {
            "top1_mean": float(attn.max(axis=1).mean()),
            "entropy_mean": float((-(attn * np.log(np.clip(attn, 1e-8, None))).sum(axis=1)).mean()),
            "hard_late_top1": float(attn[hard_late].max(axis=1).mean()) if np.any(hard_late) else float("nan"),
            "non_hard_top1": float(attn[~hard_late].max(axis=1).mean()) if np.any(~hard_late) else float("nan"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanistic review of 201b targeting and jump realism")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--h1_samples", type=int, default=128)
    parser.add_argument("--h30_samples", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline_ckpt", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--rollout201a_ckpt", type=str, default="models/backfill/transformer_ar_rollout_tail_student_t_201a/best_model.pt")
    parser.add_argument("--rollout201b_ckpt", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--rollout201b_final_ckpt", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/final_model.pt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/201b_targeting_jump_mechanistic",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_norm, future_norm = build_test_subset(
        args.data_path,
        args.history_len,
        args.future_len,
        args.test_start,
        args.max_windows,
    )
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1).numpy()
    raw = np.load(args.data_path)
    train_surface = raw["surface"][: args.test_start].astype(np.float32)
    train_abs_delta = np.abs(np.diff(train_surface, axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    vov, calm, turb, q20_vov, q80_vov = regime_masks_from_history(history_norm)
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).numpy()
    target_h30 = future_flat[:, -1, :]
    h30_energy = np.abs(target_h30 - prev).sum(axis=1)
    q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8))
    hard_late = turb & (h30_energy >= q80_h30_turb)
    clean_late_turb = turb & ~hard_late

    summary = {
        "config": {
            "max_windows": int(history_norm.shape[0]),
            "h1_samples": args.h1_samples,
            "h30_samples": args.h30_samples,
            "batch_size": args.batch_size,
        },
        "thresholds": {
            "train_abs_q95": q95,
            "train_abs_q99": q99,
            "history_vov_q20": q20_vov,
            "history_vov_q80": q80_vov,
            "h30_energy_q80_within_turb": q80_h30_turb,
        },
        "subset_sizes": {
            "all": int(history_norm.shape[0]),
            "calm": int(calm.sum()),
            "turb": int(turb.sum()),
            "hard_late": int(hard_late.sum()),
            "clean_late_turb": int(clean_late_turb.sum()),
        },
        "models": {
            "169c_best": evaluate_model(
                "169c_best",
                args.baseline_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
            "201a_best": evaluate_model(
                "201a_best",
                args.rollout201a_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
            "201b_best": evaluate_model(
                "201b_best",
                args.rollout201b_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
            "201b_final": evaluate_model(
                "201b_final",
                args.rollout201b_final_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
        },
        "interpretation": {
            "read": (
                "If encoder hard-late AUC stays high but top-cell probing and width top-k overlap remain weak, "
                "then the bottleneck is not window-level discrimination but localized routing. "
                "If pathwise max-jump KS failure comes with inflated any-q99 path rate or q99-count-per-path, "
                "then S11 is being broken by scattered overactive tails rather than a complete lack of jumps."
            )
        },
    }

    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved mechanistic summary to {out_path}")


if __name__ == "__main__":
    main()
