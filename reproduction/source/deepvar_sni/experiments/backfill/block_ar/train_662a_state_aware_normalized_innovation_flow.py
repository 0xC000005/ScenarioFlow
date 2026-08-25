#!/usr/bin/env python
"""662a: train a state-aware normalized-innovation AR flow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import encode_state  # noqa: E402
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    IncrementCoordinateBlock,
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.normalized_innovation_662_utils import (  # noqa: E402
    ewma_mean_center,
    normalize_increment_windows,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
    fit_empirical_quantiles,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    select_state_increment_scope,
)


def select_normalized_innovation_scope(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    *,
    scale_half_life: float | None,
    scale_floor: float,
    center_mode: str = "zero",
    drift_feature_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    if scope in {"joint38", "iv_only"}:
        history_level, history_increment, future_level, future_increment, history_raw, specs = (
            select_state_increment_scope(block, scope, iv_count)
        )
    elif scope == "anchor_only":
        specs = block.specs[iv_count:]
        history_raw = block.history_state[..., iv_count:]
        future_raw = block.future_state[..., iv_count:]
        history_level = encode_state(history_raw, specs).astype(np.float32)
        future_level = encode_state(future_raw, specs).astype(np.float32)
        history_increment = block.history_increment[..., iv_count:]
        future_increment = block.future_increment[..., iv_count:]
    else:
        raise ValueError(f"unknown state_scope {scope!r}")
    history_norm, future_norm, center, scale = normalize_increment_windows(
        history_increment,
        future_increment,
        half_life=scale_half_life,
        scale_floor=float(scale_floor),
        center_mode=center_mode,
    )
    if drift_feature_mode == "none":
        drift_feature = np.zeros_like(center, dtype=np.float32)
    elif drift_feature_mode == "ewma_mean":
        drift_feature = ewma_mean_center(history_increment, half_life=scale_half_life)
    else:
        raise ValueError("drift_feature_mode must be 'none' or 'ewma_mean'")
    return (
        history_level.astype(np.float32),
        history_norm.astype(np.float32),
        future_level.astype(np.float32),
        future_norm.astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
        drift_feature.astype(np.float32),
        history_raw.astype(np.float32),
        specs,
    )


def select_raw_scope_from_block(
    block: IncrementCoordinateBlock,
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


def panel_daily_changes(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :], future[:, :-1, :]], axis=1)
    return future - prev


def sticky_score_mask_from_block(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    *,
    zero_eps: float,
    zero_rate_gate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_history, raw_future = select_raw_scope_from_block(block, scope, iv_count)
    history_delta = np.diff(raw_history.astype(np.float64), axis=1)
    future_delta = panel_daily_changes(raw_history.astype(np.float64), raw_future.astype(np.float64))
    all_delta = np.concatenate([history_delta, future_delta], axis=1)
    zero_rate = np.mean(np.abs(all_delta) <= float(zero_eps), axis=(0, 1))
    mask = zero_rate >= float(zero_rate_gate)
    if scope == "joint38":
        specs = block.specs
    elif scope == "iv_only":
        specs = block.specs[:iv_count]
    else:
        specs = block.specs[iv_count:]
    rows = [
        {
            "name": spec.name,
            "selected": bool(mask[idx]),
            "zero_rate": float(zero_rate[idx]),
        }
        for idx, spec in enumerate(specs)
    ]
    return mask.astype(bool), {
        "policy": "train_raw_no_change_rate",
        "zero_eps": float(zero_eps),
        "zero_rate_gate": float(zero_rate_gate),
        "selected_names": [row["name"] for row in rows if row["selected"]],
        "rows": rows,
    }


def sticky_observation_weight_from_block(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    sticky_mask: np.ndarray,
    *,
    zero_eps: float,
    zero_weight: float,
) -> np.ndarray:
    raw_history, raw_future = select_raw_scope_from_block(block, scope, iv_count)
    raw_delta = panel_daily_changes(raw_history.astype(np.float64), raw_future.astype(np.float64))
    weight = np.ones(raw_delta.shape, dtype=np.float32)
    if bool(np.any(sticky_mask)):
        zero = np.abs(raw_delta) <= float(zero_eps)
        sticky = np.asarray(sticky_mask, dtype=bool)[None, None, :]
        weight = np.where(zero & sticky, float(zero_weight), 1.0).astype(np.float32)
    return weight


def no_update_target_from_block(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    *,
    zero_eps: float,
) -> np.ndarray:
    raw_history, raw_future = select_raw_scope_from_block(block, scope, iv_count)
    raw_delta = panel_daily_changes(raw_history.astype(np.float64), raw_future.astype(np.float64))
    return (np.abs(raw_delta) <= float(zero_eps)).astype(np.float32)


def tail_asinh_scale_from_future_norm(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    *,
    sticky_mask: np.ndarray,
    future_norm: np.ndarray,
    zero_eps: float,
    scale_quantile: float,
    min_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_history, raw_future = select_raw_scope_from_block(block, scope, iv_count)
    raw_delta = panel_daily_changes(raw_history.astype(np.float64), raw_future.astype(np.float64))
    future_norm_arr = np.asarray(future_norm, dtype=np.float64)
    sticky = np.asarray(sticky_mask, dtype=bool)
    if future_norm_arr.shape != raw_delta.shape:
        raise ValueError(
            f"future_norm shape {future_norm_arr.shape} must match raw daily changes {raw_delta.shape}"
        )
    if sticky.shape != (raw_delta.shape[-1],):
        raise ValueError(f"sticky_mask must have shape ({raw_delta.shape[-1]},), got {sticky.shape}")
    if scope == "joint38":
        specs = block.specs
    elif scope == "iv_only":
        specs = block.specs[:iv_count]
    else:
        specs = block.specs[iv_count:]
    q = float(scale_quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("tail-asinh scale quantile must be in [0, 1]")
    scales = np.ones(raw_delta.shape[-1], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        nonzero = (np.abs(raw_delta[..., idx]) > float(zero_eps)) & np.isfinite(future_norm_arr[..., idx])
        values = np.abs(future_norm_arr[..., idx][nonzero])
        if bool(sticky[idx]) and values.size:
            scale = max(float(np.quantile(values, q)), float(min_scale))
            scales[idx] = np.float32(scale)
        rows.append(
            {
                "name": spec.name,
                "selected": bool(sticky[idx]),
                "nonzero_count": int(values.size),
                "scale": float(scales[idx]),
            }
        )
    return scales, {
        "policy": "train_nonzero_normalized_abs_quantile_for_high_no_change_channels",
        "zero_eps": float(zero_eps),
        "scale_quantile": float(scale_quantile),
        "min_scale": float(min_scale),
        "selected_names": [row["name"] for row in rows if row["selected"]],
        "rows": rows,
    }


def model_coordinate_from_innovation_coordinate(innovation_coordinate: str) -> str:
    if innovation_coordinate == "score":
        return "state_aware_normalized_innovation_score"
    if innovation_coordinate == "hybrid_sticky_score":
        return "state_aware_normalized_innovation_hybrid_score"
    if innovation_coordinate == "hybrid_tail_asinh":
        return "state_aware_normalized_innovation_tail_asinh"
    return "state_aware_normalized_innovation"


def eval_loss(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    loader: DataLoader,
    device: torch.device,
    *,
    condition_contrast_weight: float,
    condition_contrast_margin: float,
    risk_state_weight: float,
    risk_state_rank_weight: float,
    mixed_support_weight: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for history_level, history_norm, future_level, future_norm, center, scale, drift_feature, future_weight, no_update_target in loader:
            loss, _metrics = model.training_loss(
                history_level.to(device),
                history_norm.to(device),
                future_level.to(device),
                future_norm.to(device),
                center.to(device),
                scale.to(device),
                drift_feature=drift_feature.to(device),
                future_element_weight=future_weight.to(device),
                future_no_update_target=no_update_target.to(device),
                condition_contrast_weight=float(condition_contrast_weight),
                condition_contrast_margin=float(condition_contrast_margin),
                risk_state_weight=float(risk_state_weight),
                risk_state_rank_weight=float(risk_state_rank_weight),
                mixed_support_weight=float(mixed_support_weight),
            )
            batch_n = int(history_level.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
    return total / max(count, 1)


@torch.no_grad()
def sample_smoke(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_state_raw: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    steps: int,
    chunk_size: int,
    device: torch.device,
    iv_count: int,
) -> dict[str, Any]:
    n = min(4, int(history_level.shape[0]))
    sampled_increment = model.sample_batched(
        torch.from_numpy(history_level[:n]).to(device),
        torch.from_numpy(history_norm[:n]).to(device),
        torch.from_numpy(center[:n]).to(device),
        torch.from_numpy(scale[:n]).to(device),
        drift_feature=torch.from_numpy(drift_feature[:n]).to(device),
        n_samples=int(samples),
        n_steps=int(steps),
        chunk_size=int(chunk_size),
    )
    increments = sampled_increment.detach().cpu().numpy()
    states = reconstruct_state_from_increments(history_state_raw[:n, -1, :], increments, specs)
    report: dict[str, Any] = {
        "sample_increment_shape": list(increments.shape),
        "sample_state_shape": list(states.shape),
        "finite_increment_rate": float(np.isfinite(increments).mean()),
        "finite_state_rate": float(np.isfinite(states).mean()),
        "state_min": float(np.nanmin(states)),
        "state_max": float(np.nanmax(states)),
    }
    if states.shape[-1] >= iv_count:
        report.update(
            {
                "iv_min": float(np.nanmin(states[..., :iv_count])),
                "iv_max": float(np.nanmax(states[..., :iv_count])),
            }
        )
    if states.shape[-1] > iv_count:
        report.update(
            {
                "factor_min": float(np.nanmin(states[..., iv_count:])),
                "factor_max": float(np.nanmax(states[..., iv_count:])),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument(
        "--innovation_coordinate",
        choices=["normalized", "score", "hybrid_sticky_score", "hybrid_tail_asinh"],
        default="normalized",
    )
    parser.add_argument("--sticky_score_zero_eps", type=float, default=1e-10)
    parser.add_argument("--sticky_score_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--tail_asinh_scale_quantile", type=float, default=0.5)
    parser.add_argument("--tail_asinh_min_scale", type=float, default=0.5)
    parser.add_argument("--sticky_observation_loss", choices=["none", "nonzero_mask"], default="none")
    parser.add_argument("--sticky_observation_zero_weight", type=float, default=0.0)
    parser.add_argument("--mixed_support_observation", choices=["none", "bernoulli_no_update"], default="none")
    parser.add_argument("--mixed_support_zero_eps", type=float, default=1e-10)
    parser.add_argument("--mixed_support_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--mixed_support_loss_weight", type=float, default=1.0)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.05)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--base_noise_rho", type=float, default=0.0)
    parser.add_argument("--conditional_base_noise_scale", action="store_true")
    parser.add_argument("--base_noise_scale_min", type=float, default=0.5)
    parser.add_argument("--base_noise_scale_max", type=float, default=2.0)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale", "scale_local", "scale_drift"], default="scale")
    parser.add_argument("--condition_contrast_weight", type=float, default=0.0)
    parser.add_argument("--condition_contrast_margin", type=float, default=0.0)
    parser.add_argument("--risk_state_dim", type=int, default=0)
    parser.add_argument("--risk_state_weight", type=float, default=0.0)
    parser.add_argument("--risk_state_rank_weight", type=float, default=0.0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=662)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_half_life = None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life)

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    (
        train_level,
        train_norm,
        train_future_level,
        train_future_norm,
        train_center,
        train_scale,
        train_drift,
        _train_raw,
        train_specs,
    ) = (
        select_normalized_innovation_scope(
            train_block,
            args.state_scope,
            int(args.iv_count),
            scale_half_life=scale_half_life,
            scale_floor=float(args.scale_floor),
            center_mode=args.center_mode,
            drift_feature_mode=args.drift_feature_mode,
        )
    )
    (
        val_level,
        val_norm,
        val_future_level,
        val_future_norm,
        val_center,
        val_scale,
        val_drift,
        val_raw,
        val_specs,
    ) = (
        select_normalized_innovation_scope(
            val_block,
            args.state_scope,
            int(args.iv_count),
            scale_half_life=scale_half_life,
            scale_floor=float(args.scale_floor),
            center_mode=args.center_mode,
            drift_feature_mode=args.drift_feature_mode,
        )
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    level_quantiles, quantile_levels = fit_empirical_quantiles(
        train_level,
        train_future_level,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    innovation_quantiles, innovation_quantile_levels = fit_empirical_quantiles(
        train_norm,
        train_future_norm,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        n_cells=int(train_level.shape[-1]),
        memory_dim=int(args.memory_dim),
        memory_layers=int(args.memory_layers),
        memory_heads=int(args.memory_heads),
        memory_ff=int(args.memory_ff),
        token_dim=int(args.token_dim),
        token_layers=int(args.token_layers),
        token_heads=int(args.token_heads),
        token_ff=int(args.token_ff),
        time_dim=int(args.time_dim),
        model_dropout=float(args.model_dropout),
        flow_steps=int(args.flow_steps),
        sample_temperature=float(args.sample_temperature),
        base_noise_rho=float(args.base_noise_rho),
        conditional_base_noise_scale=bool(args.conditional_base_noise_scale),
        base_noise_scale_min=float(args.base_noise_scale_min),
        base_noise_scale_max=float(args.base_noise_scale_max),
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
        prefix_feature_mode=args.prefix_feature_mode,
        innovation_coordinate=args.innovation_coordinate,
        risk_state_dim=int(args.risk_state_dim),
        mixed_support_observation=args.mixed_support_observation,
        conditioning_mode="prefix",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg).to(device)
    model.set_level_quantiles(
        torch.from_numpy(level_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
    sticky_score_report: dict[str, Any] = {
        "policy": "disabled",
        "selected_names": [],
    }
    tail_asinh_report: dict[str, Any] = {
        "policy": "disabled",
        "selected_names": [],
    }
    mixed_support_report: dict[str, Any] = {
        "policy": "disabled",
        "selected_names": [],
    }
    if args.innovation_coordinate in {"score", "hybrid_sticky_score"}:
        model.set_innovation_quantiles(
            torch.from_numpy(innovation_quantiles).to(device),
            torch.from_numpy(innovation_quantile_levels).to(device),
        )
    if args.innovation_coordinate == "hybrid_sticky_score":
        sticky_mask, sticky_score_report = sticky_score_mask_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.sticky_score_zero_eps),
            zero_rate_gate=float(args.sticky_score_zero_rate_gate),
        )
        if sticky_mask.shape != (int(train_level.shape[-1]),):
            raise RuntimeError("sticky score mask shape does not match selected state scope")
        model.set_innovation_score_mask(torch.from_numpy(sticky_mask).to(device))
    elif args.innovation_coordinate == "hybrid_tail_asinh":
        sticky_mask, sticky_score_report = sticky_score_mask_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.sticky_score_zero_eps),
            zero_rate_gate=float(args.sticky_score_zero_rate_gate),
        )
        if sticky_mask.shape != (int(train_level.shape[-1]),):
            raise RuntimeError("tail-asinh mask shape does not match selected state scope")
        tail_scales, tail_asinh_report = tail_asinh_scale_from_future_norm(
            train_block,
            args.state_scope,
            int(args.iv_count),
            sticky_mask=sticky_mask,
            future_norm=train_future_norm,
            zero_eps=float(args.sticky_score_zero_eps),
            scale_quantile=float(args.tail_asinh_scale_quantile),
            min_scale=float(args.tail_asinh_min_scale),
        )
        model.set_innovation_tail_asinh(
            torch.from_numpy(sticky_mask).to(device),
            torch.from_numpy(tail_scales).to(device),
        )
    elif args.sticky_observation_loss == "nonzero_mask":
        sticky_mask, sticky_score_report = sticky_score_mask_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.sticky_score_zero_eps),
            zero_rate_gate=float(args.sticky_score_zero_rate_gate),
        )
    else:
        sticky_mask = np.zeros(int(train_level.shape[-1]), dtype=bool)
    if args.mixed_support_observation == "bernoulli_no_update":
        mixed_mask, mixed_support_report = sticky_score_mask_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.mixed_support_zero_eps),
            zero_rate_gate=float(args.mixed_support_zero_rate_gate),
        )
        mixed_prior = np.array(
            [float(row["zero_rate"]) for row in mixed_support_report["rows"]],
            dtype=np.float32,
        )
        model.set_mixed_support_no_update(
            torch.from_numpy(mixed_mask).to(device),
            torch.from_numpy(mixed_prior).to(device),
        )
        train_no_update_target = no_update_target_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.mixed_support_zero_eps),
        )
        val_no_update_target = no_update_target_from_block(
            val_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=float(args.mixed_support_zero_eps),
        )
    else:
        train_no_update_target = np.zeros_like(train_future_norm, dtype=np.float32)
        val_no_update_target = np.zeros_like(val_future_norm, dtype=np.float32)
    if args.sticky_observation_loss == "nonzero_mask":
        train_future_weight = sticky_observation_weight_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            sticky_mask,
            zero_eps=float(args.sticky_score_zero_eps),
            zero_weight=float(args.sticky_observation_zero_weight),
        )
        val_future_weight = sticky_observation_weight_from_block(
            val_block,
            args.state_scope,
            int(args.iv_count),
            sticky_mask,
            zero_eps=float(args.sticky_score_zero_eps),
            zero_weight=float(args.sticky_observation_zero_weight),
        )
    else:
        train_future_weight = np.ones_like(train_future_norm, dtype=np.float32)
        val_future_weight = np.ones_like(val_future_norm, dtype=np.float32)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_level),
            torch.from_numpy(train_norm),
            torch.from_numpy(train_future_level),
            torch.from_numpy(train_future_norm),
            torch.from_numpy(train_center),
            torch.from_numpy(train_scale),
            torch.from_numpy(train_drift),
            torch.from_numpy(train_future_weight),
            torch.from_numpy(train_no_update_target),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_level),
            torch.from_numpy(val_norm),
            torch.from_numpy(val_future_level),
            torch.from_numpy(val_future_norm),
            torch.from_numpy(val_center),
            torch.from_numpy(val_scale),
            torch.from_numpy(val_drift),
            torch.from_numpy(val_future_weight),
            torch.from_numpy(val_no_update_target),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    t0 = time.time()
    normalization = {
        "coordinate": "encoded_increment",
        "center_mode": args.center_mode,
        "drift_feature_mode": args.drift_feature_mode,
        "scale_method": "ewma_rms",
        "scale_half_life": scale_half_life,
        "scale_floor": float(args.scale_floor),
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "innovation_coordinate": args.innovation_coordinate,
        "sticky_score": sticky_score_report,
        "tail_asinh": tail_asinh_report,
        "sticky_observation_loss": args.sticky_observation_loss,
        "sticky_observation_zero_weight": float(args.sticky_observation_zero_weight),
        "mixed_support": mixed_support_report,
        "mixed_support_observation": args.mixed_support_observation,
        "mixed_support_zero_eps": float(args.mixed_support_zero_eps),
        "mixed_support_zero_rate_gate": float(args.mixed_support_zero_rate_gate),
        "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        "train_future_weight_mean": float(np.mean(train_future_weight)),
        "val_future_weight_mean": float(np.mean(val_future_weight)),
        "train_no_update_target_mean": float(np.mean(train_no_update_target)),
        "val_no_update_target_mean": float(np.mean(val_no_update_target)),
    }
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": model_coordinate_from_innovation_coordinate(args.innovation_coordinate),
        "normalization": normalization,
        "training_objective": {
            "base": "flow_matching_mse",
            "condition_contrast_weight": float(args.condition_contrast_weight),
            "condition_contrast_margin": float(args.condition_contrast_margin),
            "risk_state_dim": int(args.risk_state_dim),
            "risk_state_weight": float(args.risk_state_weight),
            "risk_state_rank_weight": float(args.risk_state_rank_weight),
            "base_noise_rho": float(cfg.base_noise_rho),
            "conditional_base_noise_scale": bool(cfg.conditional_base_noise_scale),
            "base_noise_scale_min": float(cfg.base_noise_scale_min),
            "base_noise_scale_max": float(cfg.base_noise_scale_max),
            "flow_coordinate": args.innovation_coordinate,
            "sticky_score": sticky_score_report,
            "tail_asinh": tail_asinh_report,
            "sticky_observation_loss": args.sticky_observation_loss,
            "sticky_observation_zero_weight": float(args.sticky_observation_zero_weight),
            "mixed_support": mixed_support_report,
            "mixed_support_observation": args.mixed_support_observation,
            "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        },
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "positive_level_policy": args.positive_level_policy,
        "sticky_score": sticky_score_report,
        "tail_asinh": tail_asinh_report,
        "mixed_support": mixed_support_report,
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        metric_sums: dict[str, float] = {}
        for history_level, history_norm, future_level, future_norm, center, scale, drift_feature, future_weight, no_update_target in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.training_loss(
                history_level.to(device),
                history_norm.to(device),
                future_level.to(device),
                future_norm.to(device),
                center.to(device),
                scale.to(device),
                drift_feature=drift_feature.to(device),
                future_element_weight=future_weight.to(device),
                future_no_update_target=no_update_target.to(device),
                condition_contrast_weight=float(args.condition_contrast_weight),
                condition_contrast_margin=float(args.condition_contrast_margin),
                risk_state_weight=float(args.risk_state_weight),
                risk_state_rank_weight=float(args.risk_state_rank_weight),
                mixed_support_weight=float(args.mixed_support_loss_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_n = int(history_level.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * batch_n
        train_loss = total / max(count, 1)
        val_loss = eval_loss(
            model,
            val_loader,
            device,
            condition_contrast_weight=float(args.condition_contrast_weight),
            condition_contrast_margin=float(args.condition_contrast_margin),
            risk_state_weight=float(args.risk_state_weight),
            risk_state_rank_weight=float(args.risk_state_rank_weight),
            mixed_support_weight=float(args.mixed_support_loss_weight),
        )
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "elapsed_s": float(time.time() - t0),
        }
        for key, value in metric_sums.items():
            record[f"train_{key}"] = float(value / max(count, 1))
        history_records.append(record)
        is_best = val_loss < best_val
        if is_best:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            save_checkpoint(str(best_path), model, cfg, epoch, best_val, extra=extra)
        print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}{' best' if is_best else ''}", flush=True)

    final_path = output_dir / "final_model.pt"
    save_checkpoint(str(final_path), model, cfg, int(args.epochs), best_val, extra=extra)
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_norm[: int(args.sample_windows)],
        val_center[: int(args.sample_windows)],
        val_scale[: int(args.sample_windows)],
        val_drift[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "model_coordinate": model_coordinate_from_innovation_coordinate(args.innovation_coordinate),
        "normalization": normalization,
        "training_objective": {
            "base": "flow_matching_mse",
            "condition_contrast_weight": float(args.condition_contrast_weight),
            "condition_contrast_margin": float(args.condition_contrast_margin),
            "risk_state_dim": int(args.risk_state_dim),
            "risk_state_weight": float(args.risk_state_weight),
            "risk_state_rank_weight": float(args.risk_state_rank_weight),
            "base_noise_rho": float(cfg.base_noise_rho),
            "flow_coordinate": args.innovation_coordinate,
            "sticky_score": sticky_score_report,
            "tail_asinh": tail_asinh_report,
            "sticky_observation_loss": args.sticky_observation_loss,
            "sticky_observation_zero_weight": float(args.sticky_observation_zero_weight),
            "mixed_support": mixed_support_report,
            "mixed_support_observation": args.mixed_support_observation,
            "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        },
        "n_state_vars": int(train_level.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_level.shape),
        "val_shape": list(val_level.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(history_records[-1]["val_loss"] if history_records else float("nan")),
        "sample_smoke": smoke,
        "panel_metadata": panel_metadata,
        "sticky_score": sticky_score_report,
        "tail_asinh": tail_asinh_report,
        "mixed_support": mixed_support_report,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(json.dumps(make_serializable(history_records), indent=2), encoding="utf-8")
    (output_dir / "train_summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")
    (output_dir / "args.json").write_text(json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
