"""Frozen 39-variable panel construction and normalized-innovation selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..data import load_aligned_iv_factor_panel, official_train_val_indices
from ..state import (
    IncrementCoordinateBlock,
    build_increment_coordinate_block,
    clean_nonpositive_log_level_factors,
    encode_state,
    ewma_mean_center,
    normalize_increment_windows,
)


def prepare_panel_blocks(
    *,
    data_dir: str | Path,
    history_len: int = 30,
    future_len: int = 30,
    test_start: int = 4511,
    val_size: int = 441,
    iv_count: int = 25,
    max_train_windows: int = 0,
    clean_nonpositive_log_levels: bool = True,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "bounded_logit",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> tuple[
    list[str], dict[str, Any], IncrementCoordinateBlock, IncrementCoordinateBlock
]:
    """Build the exact pre-test training and validation blocks."""
    panel, columns, dates = load_aligned_iv_factor_panel(data_dir=data_dir)
    cleaning_report: dict[str, Any] = {"enabled": False}
    if clean_nonpositive_log_levels:
        panel, cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(iv_count),
            positive_level_policy=positive_level_policy,
        )
    train_indices, val_indices = official_train_val_indices(
        test_start=int(test_start),
        val_size=int(val_size),
        history_len=int(history_len),
        future_len=int(future_len),
    )
    if int(max_train_windows) > 0:
        train_indices = train_indices[-int(max_train_windows) :]
    kwargs = {
        "history_len": int(history_len),
        "future_len": int(future_len),
        "iv_count": int(iv_count),
        "positive_level_policy": positive_level_policy,
        "iv_transform": iv_transform,
        "iv_lower_bound": float(iv_lower_bound),
        "iv_upper_bound": float(iv_upper_bound),
    }
    train_block = build_increment_coordinate_block(
        panel, columns, train_indices, **kwargs
    )
    val_block = build_increment_coordinate_block(panel, columns, val_indices, **kwargs)
    metadata = {
        "dates_start": str(dates[0]) if len(dates) else None,
        "dates_end": str(dates[-1]) if len(dates) else None,
        "source_columns": columns,
        "positive_level_policy": positive_level_policy,
        "iv_transform": iv_transform,
        "iv_lower_bound": float(iv_lower_bound),
        "iv_upper_bound": float(iv_upper_bound),
        "cleaning_report": cleaning_report,
        "train_indices_start": int(train_indices[0]) if len(train_indices) else None,
        "train_indices_end": int(train_indices[-1]) if len(train_indices) else None,
        "val_indices_start": int(val_indices[0]) if len(val_indices) else None,
        "val_indices_end": int(val_indices[-1]) if len(val_indices) else None,
    }
    return columns, metadata, train_block, val_block


def build_blocks(
    args: argparse.Namespace,
    *,
    data_dir: str | Path,
) -> tuple[
    list[str], dict[str, Any], IncrementCoordinateBlock, IncrementCoordinateBlock
]:
    """Compatibility adapter for the frozen trainer argument namespace."""
    return prepare_panel_blocks(
        data_dir=data_dir,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        iv_count=int(args.iv_count),
        max_train_windows=int(args.max_train_windows),
        clean_nonpositive_log_levels=bool(args.clean_nonpositive_log_levels),
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
    )


def select_state_increment_scope(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    if scope == "joint38":
        specs = block.specs
        return (
            encode_state(block.history_state, specs).astype(np.float32),
            block.history_increment,
            encode_state(block.future_state, specs).astype(np.float32),
            block.future_increment,
            block.history_state,
            specs,
        )
    if scope == "iv_only":
        specs = block.specs[:iv_count]
        return (
            encode_state(block.history_state[..., :iv_count], specs).astype(np.float32),
            block.history_increment[..., :iv_count],
            encode_state(block.future_state[..., :iv_count], specs).astype(np.float32),
            block.future_increment[..., :iv_count],
            block.history_state[..., :iv_count],
            specs,
        )
    raise ValueError(f"unknown state_scope {scope!r}")


def select_normalized_innovation_scope(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
    *,
    scale_half_life: float | None,
    scale_floor: float,
    center_mode: str = "zero",
    drift_feature_mode: str = "none",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[Any],
]:
    if scope in {"joint38", "iv_only"}:
        (
            history_level,
            history_increment,
            future_level,
            future_increment,
            history_raw,
            specs,
        ) = select_state_increment_scope(block, scope, iv_count)
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
