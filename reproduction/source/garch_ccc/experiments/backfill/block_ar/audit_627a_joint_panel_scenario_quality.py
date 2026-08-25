#!/usr/bin/env python
"""627a: generic joint-panel scenario quality audit for native joint models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._factor_conditioning_525_utils import (
    official_train_val_indices,
)  # noqa: E402
from experiments.backfill.block_ar._panel_law_535_utils import (
    load_aligned_iv_factor_panel,
)  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    build_unified_increment_block,
    clean_nonpositive_log_level_factors,
    decode_state,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (
    set_seed,
)  # noqa: E402
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
    reconstruct_state_from_increments,
    state_panel_from_specs,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (
    select_scope,
)  # noqa: E402
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    select_increment_scope,
)
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    select_state_increment_scope,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, values, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def safe_corrcoef(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected 2d array")
    if x.shape[0] < 3:
        return np.eye(x.shape[1], dtype=np.float64)
    std = x.std(axis=0)
    keep = std > 1e-12
    out = np.eye(x.shape[1], dtype=np.float64)
    if np.sum(keep) >= 2:
        corr = np.corrcoef(x[:, keep], rowvar=False)
        out[np.ix_(keep, keep)] = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def upper_tri_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def corr_similarity(gt: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    gt_v = upper_tri_values(gt)
    gen_v = upper_tri_values(gen)
    if gt_v.size == 0:
        corr = float("nan")
    elif np.std(gt_v) < 1e-12 or np.std(gen_v) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(gt_v, gen_v)[0, 1])
    return {
        "upper_corr": corr,
        "mae": float(np.mean(np.abs(gt_v - gen_v))) if gt_v.size else float("nan"),
        "frobenius": float(np.linalg.norm(gt - gen) / max(np.linalg.norm(gt), 1e-12)),
        "gt_mean_abs": float(np.mean(np.abs(gt_v))) if gt_v.size else float("nan"),
        "gen_mean_abs": float(np.mean(np.abs(gen_v))) if gt_v.size else float("nan"),
    }


def _ordinal_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def state_block_alignment_diagnostics(
    panel: np.ndarray,
    block: Any,
    specs: list[Any],
    *,
    n_windows: int | None = None,
) -> dict[str, float | int]:
    n = int(block.history_state.shape[0] if n_windows is None else n_windows)
    n = min(n, int(block.history_state.shape[0]))
    if n == 0:
        return {
            "n_windows": 0,
            "history_max_abs_error": float("nan"),
            "future_max_abs_error": float("nan"),
            "history_mean_abs_error": float("nan"),
            "future_mean_abs_error": float("nan"),
        }
    block_history = np.asarray(block.history_state[:n], dtype=np.float64)
    block_future = np.asarray(block.future_state[:n], dtype=np.float64)
    if block_history.shape[-1] != len(specs):
        block_specs = getattr(block, "specs", None)
        if block_specs is None:
            raise ValueError("block has wider state than specs but does not expose block.specs")
        source_to_pos = {int(spec.source_index): idx for idx, spec in enumerate(block_specs)}
        try:
            positions = [source_to_pos[int(spec.source_index)] for spec in specs]
        except KeyError as exc:
            raise ValueError("selected specs are not present in block specs") from exc
        block_history = block_history[..., positions]
        block_future = block_future[..., positions]
    history_len = int(block.history_state.shape[1])
    future_len = int(block.future_state.shape[1])
    state_panel = state_panel_from_specs(panel, specs)
    expected_history = np.stack(
        [state_panel[int(idx) : int(idx) + history_len] for idx in block.indices[:n]],
        axis=0,
    )
    expected_future = np.stack(
        [
            state_panel[
                int(idx) + history_len : int(idx) + history_len + future_len
            ]
            for idx in block.indices[:n]
        ],
            axis=0,
        )
    history_err = np.abs(block_history - expected_history)
    future_err = np.abs(block_future - expected_future)
    return {
        "n_windows": int(n),
        "history_max_abs_error": float(np.max(history_err)) if n else float("nan"),
        "future_max_abs_error": float(np.max(future_err)) if n else float("nan"),
        "history_mean_abs_error": float(np.mean(history_err)) if n else float("nan"),
        "future_mean_abs_error": float(np.mean(future_err)) if n else float("nan"),
    }


def conditional_panel_diagnostics(
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    samples_raw: np.ndarray,
) -> dict[str, float]:
    sample_median = np.median(samples_raw, axis=1)
    rolled_median = np.roll(sample_median, shift=1, axis=0)
    conditional_mae = float(np.mean(np.abs(sample_median - future_raw)))
    rolled_mae = float(np.mean(np.abs(rolled_median - future_raw)))
    mae_reduction = (
        (rolled_mae - conditional_mae) / rolled_mae * 100.0
        if rolled_mae > 1e-12
        else 0.0
    )

    history_delta = np.diff(history_raw, axis=1)
    history_activity = np.mean(history_delta * history_delta, axis=(1, 2))
    lo = np.quantile(samples_raw, 0.05, axis=1)
    hi = np.quantile(samples_raw, 0.95, axis=1)
    sample_width = np.mean(hi - lo, axis=(1, 2))
    future_activity = np.mean(
        panel_daily_changes(history_raw, future_raw) ** 2,
        axis=(1, 2),
    )

    return {
        "conditional_median_mae": conditional_mae,
        "rolled_median_mae": rolled_mae,
        "median_mae_reduction_vs_rolled_pct": float(mae_reduction),
        "history_activity_width_spearman": _ordinal_spearman(
            history_activity,
            sample_width,
        ),
        "future_activity_width_spearman": _ordinal_spearman(
            future_activity,
            sample_width,
        ),
    }


HistoryInput = np.ndarray | tuple[np.ndarray, ...]


def build_history_future(
    args: argparse.Namespace, payload: dict[str, Any]
) -> tuple[HistoryInput, np.ndarray, list[Any], Any, dict[str, float | int]]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    positive_level_policy = payload.get(
        "positive_level_policy",
        payload.get("panel_metadata", {}).get(
            "positive_level_policy",
            getattr(args, "positive_level_policy", "reference_based"),
        ),
    )
    if args.clean_nonpositive_log_levels:
        panel, _cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
        )
    iv_transform = payload.get(
        "iv_transform",
        payload.get("normalization", {}).get(
            "iv_transform",
            payload.get("panel_metadata", {}).get(
                "iv_transform",
                getattr(args, "iv_transform", "log_level"),
            ),
        ),
    )
    iv_lower_bound = float(
        payload.get(
            "iv_lower_bound",
            payload.get("normalization", {}).get(
                "iv_lower_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_lower_bound",
                    getattr(args, "iv_lower_bound", 1e-4),
                ),
            ),
        )
    )
    iv_upper_bound = float(
        payload.get(
            "iv_upper_bound",
            payload.get("normalization", {}).get(
                "iv_upper_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_upper_bound",
                    getattr(args, "iv_upper_bound", 1.0),
                ),
            ),
        )
    )
    _train_indices, val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    if int(args.max_windows) > 0:
        val_indices = val_indices[: int(args.max_windows)]
    if payload.get("model_coordinate") in {
        "state_aware_normalized_innovation",
        "state_aware_normalized_innovation_score",
        "state_aware_normalized_innovation_hybrid_score",
        "state_aware_normalized_innovation_tail_asinh",
    } or args.model_type == "662a":
        block = build_increment_coordinate_block(
            panel,
            columns,
            val_indices,
            history_len=int(payload["config"]["history_len"]),
            future_len=int(payload["config"]["future_len"]),
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
            iv_transform=iv_transform,
            iv_lower_bound=iv_lower_bound,
            iv_upper_bound=iv_upper_bound,
        )
        norm_cfg = payload.get("normalization", {})
        scale_half_life = norm_cfg.get("scale_half_life", 0.0)
        if scale_half_life is not None and float(scale_half_life) <= 0.0:
            scale_half_life = None
        center_mode = norm_cfg.get("center_mode", "zero")
        drift_feature_mode = norm_cfg.get("drift_feature_mode", "none")
        (
            history_level,
            history_norm,
            _future_level,
            future_norm,
            center,
            scale,
            drift_feature,
            _history_state,
            specs,
        ) = select_normalized_innovation_scope(
            block,
            payload.get("state_scope", args.state_scope),
            int(args.iv_count),
            scale_half_life=scale_half_life,
            scale_floor=float(norm_cfg.get("scale_floor", 1e-4)),
            center_mode=center_mode,
            drift_feature_mode=drift_feature_mode,
        )
        expected = [spec["name"] for spec in payload.get("state_specs", [])]
        actual = [spec.name for spec in specs]
        if expected and expected != actual:
            raise RuntimeError(
                "checkpoint state specs do not match rebuilt validation specs"
            )
        alignment = state_block_alignment_diagnostics(panel, block, specs)
        return (
            (
                history_level.astype(np.float32),
                history_norm.astype(np.float32),
                center.astype(np.float32),
                scale.astype(np.float32),
                drift_feature.astype(np.float32),
            ),
            future_norm.astype(np.float32),
            specs,
            block,
            alignment,
        )

    if payload.get("model_coordinate") in {
        "state_conditioned_encoded_increment",
        "state_conditioned_level_score",
        "state_conditioned_mixed_coordinate",
        "mixed_coordinate_path",
    } or args.model_type in {"629a", "638a", "641a", "647a", "652a", "658a"}:
        block = build_increment_coordinate_block(
            panel,
            columns,
            val_indices,
            history_len=int(payload["config"]["history_len"]),
            future_len=int(payload["config"]["future_len"]),
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
            iv_transform=iv_transform,
            iv_lower_bound=iv_lower_bound,
            iv_upper_bound=iv_upper_bound,
        )
        scope = payload.get("state_scope", args.state_scope)
        (
            history_level,
            history_increment,
            _future_level,
            future_increment,
            _history_state,
            specs,
        ) = select_state_increment_scope(
            block,
            scope,
            int(args.iv_count),
        )
        expected = [spec["name"] for spec in payload.get("state_specs", [])]
        actual = [spec.name for spec in specs]
        if expected and expected != actual:
            raise RuntimeError(
                "checkpoint state specs do not match rebuilt validation specs"
            )
        alignment = state_block_alignment_diagnostics(panel, block, specs)
        return (
            (
                history_level.astype(np.float32),
                history_increment.astype(np.float32),
            ),
            future_increment.astype(np.float32),
            specs,
            block,
            alignment,
        )

    if (
        payload.get("model_coordinate") == "encoded_increment"
        or args.model_type == "628a"
    ):
        block = build_increment_coordinate_block(
            panel,
            columns,
            val_indices,
            history_len=int(payload["config"]["history_len"]),
            future_len=int(payload["config"]["future_len"]),
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
            iv_transform=iv_transform,
            iv_lower_bound=iv_lower_bound,
            iv_upper_bound=iv_upper_bound,
        )
        scope = payload.get("state_scope", args.state_scope)
        history, future, _history_state, _future_state, specs = select_increment_scope(
            block,
            scope,
            int(args.iv_count),
        )
        expected = [spec["name"] for spec in payload.get("state_specs", [])]
        actual = [spec.name for spec in specs]
        if expected and expected != actual:
            raise RuntimeError(
                "checkpoint state specs do not match rebuilt validation specs"
            )
        alignment = state_block_alignment_diagnostics(panel, block, specs)
        return (
            history.astype(np.float32),
            future.astype(np.float32),
            specs,
            block,
            alignment,
        )

    block = build_unified_increment_block(
        panel,
        columns,
        val_indices,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        iv_count=int(args.iv_count),
    )
    scope = payload.get("state_scope", args.state_scope)
    value_coordinate = payload.get("value_coordinate", args.value_coordinate)
    history, future, specs = select_scope(
        block, scope, int(args.iv_count), value_coordinate
    )
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in specs]
    if expected and expected != actual:
        raise RuntimeError(
            "checkpoint state specs do not match rebuilt validation specs"
        )
    alignment = state_block_alignment_diagnostics(panel, block, specs)
    return history.astype(np.float32), future.astype(np.float32), specs, block, alignment


def load_native_model(
    model_type: str, checkpoint: str, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    if model_type in {"609a", "628a", "661a"}:
        from diffusion.block_ar.generic_empirical_score_transition_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "629a":
        from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "638a":
        from diffusion.block_ar.generic_state_conditioned_level_score_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "641a":
        from diffusion.block_ar.generic_state_conditioned_mixed_coordinate_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "658a":
        from diffusion.block_ar.generic_multihead_state_conditioned_mixed_coordinate_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "662a":
        from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "647a":
        from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "652a":
        from diffusion.block_ar.generic_multihead_mixed_coordinate_path_flow_matching import (
            load_model,
        )

        return load_model(checkpoint, device)
    if model_type == "625a":
        from diffusion.block_ar.generic_realnvp_transition_law import load_model

        return load_model(checkpoint, device)
    raise ValueError(f"unknown model_type {model_type!r}")


@torch.no_grad()
def generate_panel_samples(
    model: Any,
    history: HistoryInput,
    specs: list[Any],
    raw_history: np.ndarray,
    *,
    samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    sample_temperature: float,
    value_coordinate: str,
    model_coordinate: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_history = int(
        history[0].shape[0] if isinstance(history, tuple) else history.shape[0]
    )
    for start in range(0, n_history, int(batch_size)):
        end = min(start + int(batch_size), n_history)
        if model_coordinate in {
            "state_conditioned_encoded_increment",
            "state_conditioned_level_score",
            "state_conditioned_mixed_coordinate",
            "mixed_coordinate_path",
        }:
            history_level, history_increment = history
            panel_samples = model.sample_batched(
                torch.from_numpy(history_level[start:end]).to(device),
                torch.from_numpy(history_increment[start:end]).to(device),
                n_samples=int(samples),
                n_steps=int(n_steps),
                chunk_size=int(chunk_size),
                temperature=float(sample_temperature),
            )
        elif model_coordinate in {
            "state_aware_normalized_innovation",
            "state_aware_normalized_innovation_score",
            "state_aware_normalized_innovation_hybrid_score",
            "state_aware_normalized_innovation_tail_asinh",
        }:
            history_level, history_norm, center, scale, drift_feature = history
            panel_samples = model.sample_batched(
                torch.from_numpy(history_level[start:end]).to(device),
                torch.from_numpy(history_norm[start:end]).to(device),
                torch.from_numpy(center[start:end]).to(device),
                torch.from_numpy(scale[start:end]).to(device),
                drift_feature=torch.from_numpy(drift_feature[start:end]).to(device),
                n_samples=int(samples),
                n_steps=int(n_steps),
                chunk_size=int(chunk_size),
                temperature=float(sample_temperature),
            )
        else:
            hist = torch.from_numpy(history[start:end]).to(device)
            panel_samples = model.sample_batched(
                hist,
                n_samples=int(samples),
                n_steps=int(n_steps),
                chunk_size=int(chunk_size),
                temperature=float(sample_temperature),
            )
        arr = panel_samples.detach().cpu().numpy()
        if model_coordinate in {
            "encoded_increment",
            "state_conditioned_encoded_increment",
            "state_conditioned_level_score",
            "state_conditioned_mixed_coordinate",
            "mixed_coordinate_path",
            "state_aware_normalized_innovation",
            "state_aware_normalized_innovation_score",
            "state_aware_normalized_innovation_hybrid_score",
            "state_aware_normalized_innovation_tail_asinh",
        }:
            arr = reconstruct_state_from_increments(
                raw_history[start:end, -1, :], arr, specs
            )
        elif value_coordinate == "encoded":
            arr = decode_state(arr, specs).astype(np.float32)
        chunks.append(arr.astype(np.float32))
        print(f"  generated windows {end}/{n_history}", flush=True)
    return np.concatenate(chunks, axis=0)


def panel_daily_changes(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :], future[:, :-1, :]], axis=1)
    return future - prev


def select_raw_state_scope(
    block: Any,
    scope: str,
    iv_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scope == "joint38":
        return block.history_state, block.future_state
    if scope == "iv_only":
        return block.history_state[..., :iv_count], block.future_state[..., :iv_count]
    if scope == "anchor_only":
        return block.history_state[..., iv_count:], block.future_state[..., iv_count:]
    raise ValueError(f"unknown state_scope {scope!r}")


def summarize_joint_quality(
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    samples_raw: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    gt_delta = panel_daily_changes(history_raw, future_raw)
    sample_prev = np.concatenate(
        [
            np.repeat(history_raw[:, None, -1:, :], samples_raw.shape[1], axis=1),
            samples_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    gen_delta = samples_raw - sample_prev
    factor_slice = slice(iv_count, samples_raw.shape[-1])
    gt_factor_delta = gt_delta[..., factor_slice]
    gen_factor_delta = gen_delta[..., factor_slice]
    gt_factor_level = future_raw[..., factor_slice]
    gen_factor_level = samples_raw[..., factor_slice]

    marginal_ks = []
    tail_ratios = []
    range_rows = []
    for idx, name in enumerate(factor_names):
        gt_d = gt_factor_delta[..., idx]
        gen_d = gen_factor_delta[..., idx]
        marginal_ks.append(ks_statistic(gt_d, gen_d))
        gt_q99 = float(np.quantile(np.abs(gt_d).reshape(-1), 0.99))
        gen_q99 = float(np.quantile(np.abs(gen_d).reshape(-1), 0.99))
        tail_ratios.append(gen_q99 / max(gt_q99, 1e-12))
        gt_l = gt_factor_level[..., idx]
        gen_l = gen_factor_level[..., idx]
        range_rows.append(
            {
                "name": name,
                "gt_min": float(np.nanmin(gt_l)),
                "gt_max": float(np.nanmax(gt_l)),
                "gen_min": float(np.nanmin(gen_l)),
                "gen_max": float(np.nanmax(gen_l)),
                "ks_delta": float(marginal_ks[-1]),
                "q99_abs_delta_ratio": float(tail_ratios[-1]),
            }
        )

    gt_factor_flat = gt_factor_delta.reshape(-1, gt_factor_delta.shape[-1])
    gen_factor_flat = gen_factor_delta.reshape(-1, gen_factor_delta.shape[-1])
    if iv_count > 0:
        gt_iv_flat = gt_delta[..., :iv_count].reshape(-1, iv_count)
        gen_iv_flat = gen_delta[..., :iv_count].reshape(-1, iv_count)
    else:
        gt_iv_flat = np.empty((gt_factor_flat.shape[0], 0), dtype=np.float64)
        gen_iv_flat = np.empty((gen_factor_flat.shape[0], 0), dtype=np.float64)
    gt_all_flat = np.concatenate([gt_iv_flat, gt_factor_flat], axis=1)
    gen_all_flat = np.concatenate([gen_iv_flat, gen_factor_flat], axis=1)
    gt_all_corr = safe_corrcoef(gt_all_flat)
    gen_all_corr = safe_corrcoef(gen_all_flat)
    gt_factor_corr = gt_all_corr[iv_count:, iv_count:]
    gen_factor_corr = gen_all_corr[iv_count:, iv_count:]
    gt_iv_factor = gt_all_corr[:iv_count, iv_count:]
    gen_iv_factor = gen_all_corr[:iv_count, iv_count:]

    if iv_count > 0 and gt_iv_factor.size > 0:
        iv_factor_mae = float(np.mean(np.abs(gt_iv_factor - gen_iv_factor)))
        iv_factor_corr = 0.0
        if (
            np.std(gt_iv_factor.reshape(-1)) > 1e-12
            and np.std(gen_iv_factor.reshape(-1)) > 1e-12
        ):
            iv_factor_corr = float(
                np.corrcoef(gt_iv_factor.reshape(-1), gen_iv_factor.reshape(-1))[0, 1]
            )
        iv_factor_gt_mean_abs = float(np.mean(np.abs(gt_iv_factor)))
        iv_factor_gen_mean_abs = float(np.mean(np.abs(gen_iv_factor)))
    else:
        iv_factor_mae = float("nan")
        iv_factor_corr = float("nan")
        iv_factor_gt_mean_abs = float("nan")
        iv_factor_gen_mean_abs = float("nan")

    return {
        "finite_rate": float(np.isfinite(samples_raw).mean()),
        "n_windows": int(samples_raw.shape[0]),
        "n_samples": int(samples_raw.shape[1]),
        "future_len": int(samples_raw.shape[2]),
        "n_factors": int(len(factor_names)),
        "factor_delta_ks_mean": float(np.nanmean(marginal_ks)),
        "factor_delta_ks_pass_020": int(np.sum(np.asarray(marginal_ks) < 0.20)),
        "factor_tail_q99_ratio_median": float(np.nanmedian(tail_ratios)),
        "factor_tail_q99_pass_05_20": int(
            np.sum((np.asarray(tail_ratios) >= 0.5) & (np.asarray(tail_ratios) <= 2.0))
        ),
        "factor_factor_corr": corr_similarity(gt_factor_corr, gen_factor_corr),
        "iv_factor_corr": {
            "matrix_corr": iv_factor_corr,
            "mae": iv_factor_mae,
            "gt_mean_abs": iv_factor_gt_mean_abs,
            "gen_mean_abs": iv_factor_gen_mean_abs,
        },
        "conditional_panel": conditional_panel_diagnostics(
            history_raw,
            future_raw,
            samples_raw,
        ),
        "per_factor": range_rows,
    }


def write_markdown(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- finite rate: `{summary['finite_rate']:.4f}`",
        f"- windows / samples / horizon: `{summary['n_windows']}` / `{summary['n_samples']}` / `{summary['future_len']}`",
        f"- factor delta KS mean: `{summary['factor_delta_ks_mean']:.3f}`",
        f"- factor delta KS pass <0.20: `{summary['factor_delta_ks_pass_020']}/{summary['n_factors']}`",
        f"- factor q99 abs-delta ratio median: `{summary['factor_tail_q99_ratio_median']:.3f}`",
        f"- factor q99 abs-delta pass [0.5,2.0]: `{summary['factor_tail_q99_pass_05_20']}/{summary['n_factors']}`",
        f"- factor-factor corr upper-triangle corr: `{summary['factor_factor_corr']['upper_corr']:.3f}`",
        f"- factor-factor corr MAE: `{summary['factor_factor_corr']['mae']:.3f}`",
        f"- IV-factor corr matrix corr: `{summary['iv_factor_corr']['matrix_corr']:.3f}`",
        f"- IV-factor corr MAE: `{summary['iv_factor_corr']['mae']:.3f}`",
        f"- conditional median MAE reduction vs rolled deck: `{summary['conditional_panel']['median_mae_reduction_vs_rolled_pct']:.2f}%`",
        f"- history-activity vs generated-width Spearman: `{summary['conditional_panel']['history_activity_width_spearman']:.3f}`",
        "",
        "| factor | KS(delta) | q99 abs-delta ratio | GT range | Gen range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_factor"]:
        lines.append(
            f"| {row['name']} | {row['ks_delta']:.3f} | {row['q99_abs_delta_ratio']:.3f} | "
            f"[{row['gt_min']:.4g}, {row['gt_max']:.4g}] | [{row['gen_min']:.4g}, {row['gen_max']:.4g}] |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        choices=[
            "609a",
            "625a",
            "628a",
            "629a",
            "638a",
            "641a",
            "647a",
            "652a",
            "658a",
            "661a",
            "662a",
        ],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--state_scope",
        choices=["iv_only", "anchor_only", "joint38"],
        default="joint38",
    )
    parser.add_argument("--value_coordinate", choices=["raw", "encoded"], default="raw")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=627)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, payload = load_native_model(args.model_type, args.checkpoint, device)
    history, future, specs, block, alignment = build_history_future(args, payload)
    if (
        alignment["history_max_abs_error"] > 1e-6
        or alignment["future_max_abs_error"] > 1e-6
    ):
        raise RuntimeError(f"627a/joint-panel alignment failed: {alignment}")
    value_coordinate = payload.get("value_coordinate", args.value_coordinate)
    model_coordinate = payload.get("model_coordinate", "state")
    history_n = int(
        history[0].shape[0] if isinstance(history, tuple) else history.shape[0]
    )
    n_windows = min(int(args.max_windows), history_n)
    if isinstance(history, tuple):
        history = tuple(item[:n_windows] for item in history)
    else:
        history = history[:n_windows]
    future = future[:n_windows]
    state_scope = payload.get("state_scope", args.state_scope)
    raw_history_full, raw_future_full = select_raw_state_scope(
        block,
        state_scope,
        int(args.iv_count),
    )
    raw_history = raw_history_full[:n_windows]
    raw_future = raw_future_full[:n_windows]
    if state_scope == "joint38":
        audit_iv_count = int(args.iv_count)
        factor_names = [spec.name for spec in specs[audit_iv_count:]]
    else:
        audit_iv_count = 0
        factor_names = [spec.name for spec in specs]

    t0 = time.time()
    samples_raw = generate_panel_samples(
        model,
        history,
        specs,
        raw_history,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        sample_temperature=float(args.sample_temperature),
        value_coordinate=value_coordinate,
        model_coordinate=model_coordinate,
    )
    summary = summarize_joint_quality(
        raw_history,
        raw_future,
        samples_raw,
        factor_names,
        iv_count=int(audit_iv_count),
    )
    result = {
        "summary": summary,
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
            "state_scope": payload.get("state_scope", args.state_scope),
            "value_coordinate": value_coordinate,
            "model_coordinate": model_coordinate,
            "audit_iv_count": int(audit_iv_count),
            "n_windows": int(n_windows),
            "samples": int(args.samples),
            "n_steps": int(args.n_steps),
            "sample_temperature": float(args.sample_temperature),
            "generation_time_s": float(time.time() - t0),
            "seed": int(args.seed),
            "alignment": alignment,
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(make_serializable(result), indent=2), encoding="utf-8"
    )
    write_markdown(out_md, "627a Joint-Panel Scenario Quality Audit", summary)
    print(json.dumps(make_serializable(result["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
