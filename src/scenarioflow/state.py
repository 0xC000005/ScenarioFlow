"""State transforms and normalized-innovation reconstruction.

The numerical definitions in this module are extracted from the frozen 576a,
628a, and 662a donor implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class UnifiedVariableSpec:
    name: str
    source_column: str
    source_index: int
    transform: str
    reference_increment_column: str | None = None
    reference_increment_index: int | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class UnifiedIncrementBlock:
    history_state: np.ndarray
    future_state: np.ndarray
    future_increment: np.ndarray
    reconstructed_future_state: np.ndarray
    indices: np.ndarray
    specs: list[UnifiedVariableSpec]


@dataclass(frozen=True)
class IncrementCoordinateBlock:
    history_increment: np.ndarray
    future_increment: np.ndarray
    history_state: np.ndarray
    future_state: np.ndarray
    indices: np.ndarray
    specs: list[UnifiedVariableSpec]


def is_return_like_column(column: str) -> bool:
    return column.endswith("_logret") or column.endswith("_diff") or "return:" in column


def _base_factor_name(column: str) -> str:
    return column.removeprefix("factor:")


def build_unified_variable_specs(
    columns: list[str],
    *,
    panel: np.ndarray | None = None,
    iv_count: int = 25,
    eps: float = 1e-8,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "log_level",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> list[UnifiedVariableSpec]:
    """Build one state-variable list without duplicated return targets."""
    if len(columns) < iv_count:
        raise ValueError("columns shorter than iv_count")
    if positive_level_policy not in {"reference_based", "observed_positive"}:
        raise ValueError(
            "positive_level_policy must be 'reference_based' or 'observed_positive'"
        )
    if iv_transform not in {"log_level", "bounded_logit"}:
        raise ValueError("iv_transform must be 'log_level' or 'bounded_logit'")
    if iv_transform == "bounded_logit" and not iv_lower_bound < iv_upper_bound:
        raise ValueError("iv_lower_bound must be less than iv_upper_bound")
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    specs: list[UnifiedVariableSpec] = []
    for idx in range(iv_count):
        column = columns[idx]
        if not column.startswith("iv:"):
            raise ValueError(f"expected IV column at {idx}, got {column}")
        specs.append(
            UnifiedVariableSpec(
                name=column,
                source_column=column,
                source_index=idx,
                transform=iv_transform,
                lower_bound=(
                    float(iv_lower_bound) if iv_transform == "bounded_logit" else None
                ),
                upper_bound=(
                    float(iv_upper_bound) if iv_transform == "bounded_logit" else None
                ),
            )
        )
    for idx in range(iv_count, len(columns)):
        column = columns[idx]
        if is_return_like_column(column):
            continue
        base = _base_factor_name(column)
        logret_name = f"factor:{base}_logret"
        diff_name = f"factor:{base}_diff"
        values = None if panel is None else np.asarray(panel[:, idx], dtype=np.float64)
        observed_positive = (
            positive_level_policy == "observed_positive"
            and values is not None
            and np.nanmin(values) > float(eps)
        )
        if logret_name in name_to_idx or observed_positive:
            if values is not None and np.nanmin(values) < -float(eps):
                transform = "diff_level"
                reference_column = None
                reference_index = None
            else:
                transform = "log_level"
                reference_column = logret_name if logret_name in name_to_idx else None
                reference_index = name_to_idx.get(logret_name)
        elif diff_name in name_to_idx:
            transform = "diff_level"
            reference_column = diff_name
            reference_index = name_to_idx[diff_name]
        else:
            transform = "diff_level"
            reference_column = None
            reference_index = None
        specs.append(
            UnifiedVariableSpec(
                name=column,
                source_column=column,
                source_index=idx,
                transform=transform,
                reference_increment_column=reference_column,
                reference_increment_index=reference_index,
            )
        )
    return specs


def clean_nonpositive_log_level_factors(
    panel: np.ndarray,
    columns: list[str],
    *,
    iv_count: int = 25,
    eps: float = 1e-8,
    positive_level_policy: str = "reference_based",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Forward-fill nonpositive level series that use log coordinates."""
    if positive_level_policy not in {"reference_based", "observed_positive"}:
        raise ValueError(
            "positive_level_policy must be 'reference_based' or 'observed_positive'"
        )
    cleaned = np.asarray(panel, dtype=np.float32).copy()
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    cleaned_columns: dict[str, int] = {}
    diff_fallback_columns: list[str] = []
    for idx in range(iv_count, len(columns)):
        column = columns[idx]
        if is_return_like_column(column):
            continue
        base = _base_factor_name(column)
        has_log_reference = f"factor:{base}_logret" in name_to_idx
        if not has_log_reference and positive_level_policy != "observed_positive":
            continue
        values = cleaned[:, idx].astype(np.float64)
        if np.nanmin(values) < -float(eps):
            diff_fallback_columns.append(column)
            continue
        bad = ~np.isfinite(values) | (values <= float(eps))
        if not np.any(bad):
            continue
        good_idx = np.flatnonzero(~bad)
        if good_idx.size == 0:
            raise ValueError(f"cannot clean {column}: no positive finite values")
        filled = values.copy()
        first = int(good_idx[0])
        filled[:first] = filled[first]
        last_value = float(filled[first])
        for row in range(first + 1, filled.shape[0]):
            if np.isfinite(filled[row]) and filled[row] > float(eps):
                last_value = float(filled[row])
            else:
                filled[row] = last_value
        cleaned[:, idx] = filled.astype(np.float32)
        cleaned_columns[column] = int(np.sum(bad))
    return cleaned, {
        "enabled": True,
        "eps": float(eps),
        "cleaned_columns": cleaned_columns,
        "diff_fallback_columns": diff_fallback_columns,
        "n_cleaned_values": int(sum(cleaned_columns.values())),
        "n_cleaned_columns": int(len(cleaned_columns)),
        "n_diff_fallback_columns": int(len(diff_fallback_columns)),
    }


def encode_state(
    state: np.ndarray,
    specs: list[UnifiedVariableSpec],
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != len(specs):
        raise ValueError("state last dimension must match specs")
    encoded = np.empty_like(state, dtype=np.float64)
    for idx, spec in enumerate(specs):
        values = state[..., idx]
        if spec.transform == "log_level":
            encoded[..., idx] = np.log(np.maximum(values, eps))
        elif spec.transform == "bounded_logit":
            if spec.lower_bound is None or spec.upper_bound is None:
                raise ValueError(f"bounded_logit spec {spec.name} requires bounds")
            lower = float(spec.lower_bound)
            upper = float(spec.upper_bound)
            unit = (values - lower) / max(upper - lower, eps)
            unit = np.clip(unit, eps, 1.0 - eps)
            encoded[..., idx] = np.log(unit) - np.log1p(-unit)
        elif spec.transform == "diff_level":
            encoded[..., idx] = values
        else:
            raise ValueError(f"unknown transform {spec.transform}")
    return encoded


def decode_state(encoded: np.ndarray, specs: list[UnifiedVariableSpec]) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.shape[-1] != len(specs):
        raise ValueError("encoded last dimension must match specs")
    state = np.empty_like(encoded, dtype=np.float64)
    for idx, spec in enumerate(specs):
        values = encoded[..., idx]
        if spec.transform == "log_level":
            state[..., idx] = np.exp(values)
        elif spec.transform == "bounded_logit":
            if spec.lower_bound is None or spec.upper_bound is None:
                raise ValueError(f"bounded_logit spec {spec.name} requires bounds")
            lower = float(spec.lower_bound)
            upper = float(spec.upper_bound)
            unit = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
            state[..., idx] = lower + (upper - lower) * unit
        elif spec.transform == "diff_level":
            state[..., idx] = values
        else:
            raise ValueError(f"unknown transform {spec.transform}")
    return state


def state_panel_from_specs(
    panel: np.ndarray, specs: list[UnifiedVariableSpec]
) -> np.ndarray:
    panel = np.asarray(panel, dtype=np.float64)
    return np.stack([panel[:, spec.source_index] for spec in specs], axis=1).astype(
        np.float64
    )


def build_unified_increment_block(
    panel: np.ndarray,
    columns: list[str],
    indices: Iterable[int],
    *,
    history_len: int,
    future_len: int,
    iv_count: int = 25,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "log_level",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> UnifiedIncrementBlock:
    panel = np.asarray(panel, dtype=np.float64)
    specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=iv_count,
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    state_panel = state_panel_from_specs(panel, specs)
    encoded_panel = encode_state(state_panel, specs)
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    reconstructed: list[np.ndarray] = []
    index_array = np.asarray(list(indices), dtype=np.int64)
    for raw_idx in index_array:
        start = int(raw_idx)
        hist_end = start + int(history_len)
        fut_end = hist_end + int(future_len)
        if start < 0 or fut_end > panel.shape[0]:
            raise ValueError(f"window {start} exceeds panel bounds")
        history_state = state_panel[start:hist_end]
        future_state = state_panel[hist_end:fut_end]
        future_increment = np.diff(encoded_panel[hist_end - 1 : fut_end], axis=0)
        reconstructed_state = decode_state(
            encoded_panel[hist_end - 1][None, :] + np.cumsum(future_increment, axis=0),
            specs,
        )
        histories.append(history_state)
        futures.append(future_state)
        increments.append(future_increment)
        reconstructed.append(reconstructed_state)
    return UnifiedIncrementBlock(
        history_state=np.asarray(histories, dtype=np.float32),
        future_state=np.asarray(futures, dtype=np.float32),
        future_increment=np.asarray(increments, dtype=np.float32),
        reconstructed_future_state=np.asarray(reconstructed, dtype=np.float32),
        indices=index_array,
        specs=specs,
    )


def build_increment_coordinate_block(
    panel: np.ndarray,
    columns: list[str],
    indices: Iterable[int],
    *,
    history_len: int,
    future_len: int,
    iv_count: int = 25,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "log_level",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> IncrementCoordinateBlock:
    panel = np.asarray(panel, dtype=np.float64)
    specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=iv_count,
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    state_panel = state_panel_from_specs(panel, specs)
    encoded_panel = encode_state(state_panel, specs)
    history_increment: list[np.ndarray] = []
    future_increment: list[np.ndarray] = []
    history_state: list[np.ndarray] = []
    future_state: list[np.ndarray] = []
    index_array = np.asarray(list(indices), dtype=np.int64)
    for raw_idx in index_array:
        start = int(raw_idx)
        hist_end = start + int(history_len)
        fut_end = hist_end + int(future_len)
        if start < 0 or fut_end > panel.shape[0]:
            raise ValueError(f"window {start} exceeds panel bounds")
        if start > 0:
            history_changes = np.diff(encoded_panel[start - 1 : hist_end], axis=0)
        else:
            history_changes = np.zeros((int(history_len), len(specs)), dtype=np.float64)
            if int(history_len) > 1:
                history_changes[1:] = np.diff(encoded_panel[start:hist_end], axis=0)
        history_increment.append(history_changes)
        future_increment.append(np.diff(encoded_panel[hist_end - 1 : fut_end], axis=0))
        history_state.append(state_panel[start:hist_end])
        future_state.append(state_panel[hist_end:fut_end])
    return IncrementCoordinateBlock(
        history_increment=np.asarray(history_increment, dtype=np.float32),
        future_increment=np.asarray(future_increment, dtype=np.float32),
        history_state=np.asarray(history_state, dtype=np.float32),
        future_state=np.asarray(future_state, dtype=np.float32),
        indices=index_array,
        specs=specs,
    )


def reconstruct_state_from_increments(
    last_history_state: np.ndarray,
    future_increment: np.ndarray,
    specs: list[UnifiedVariableSpec],
) -> np.ndarray:
    increments = np.asarray(future_increment, dtype=np.float64)
    base = encode_state(np.asarray(last_history_state, dtype=np.float64), specs)
    while base.ndim < increments.ndim:
        base = np.expand_dims(base, axis=-2)
    encoded_future = base + np.cumsum(increments, axis=-2)
    return decode_state(encoded_future, specs).astype(np.float32)


def ewma_rms_scale(
    history_increment: np.ndarray,
    *,
    half_life: float | None = None,
    scale_floor: float = 1e-4,
) -> np.ndarray:
    increments = np.asarray(history_increment, dtype=np.float64)
    if increments.ndim != 3:
        raise ValueError(
            "history_increment must have shape [windows, history, channels]"
        )
    history_len = int(increments.shape[1])
    if history_len < 1:
        raise ValueError("history_increment must contain at least one history step")
    if half_life is None:
        half_life = max(float(history_len) / 3.0, 1.0)
    ages = np.arange(history_len - 1, -1, -1, dtype=np.float64)
    weights = np.power(0.5, ages / max(float(half_life), 1e-6))
    weights = weights / np.sum(weights)
    rms = np.sqrt(np.sum(weights[None, :, None] * increments * increments, axis=1))
    return np.maximum(rms, float(scale_floor)).astype(np.float32)


def ewma_mean_center(
    history_increment: np.ndarray, *, half_life: float | None = None
) -> np.ndarray:
    increments = np.asarray(history_increment, dtype=np.float64)
    if increments.ndim != 3:
        raise ValueError(
            "history_increment must have shape [windows, history, channels]"
        )
    history_len = int(increments.shape[1])
    if history_len < 1:
        raise ValueError("history_increment must contain at least one history step")
    if half_life is None:
        half_life = max(float(history_len) / 3.0, 1.0)
    ages = np.arange(history_len - 1, -1, -1, dtype=np.float64)
    weights = np.power(0.5, ages / max(float(half_life), 1e-6))
    weights = weights / np.sum(weights)
    return np.sum(weights[None, :, None] * increments, axis=1).astype(np.float32)


def normalize_increment_windows(
    history_increment: np.ndarray,
    future_increment: np.ndarray,
    *,
    half_life: float | None = None,
    scale_floor: float = 1e-4,
    center_mode: str = "zero",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = np.asarray(history_increment, dtype=np.float32)
    future = np.asarray(future_increment, dtype=np.float32)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history and future increments must be rank-3 arrays")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history and future increment dimensions are inconsistent")
    scale = ewma_rms_scale(history, half_life=half_life, scale_floor=scale_floor)
    if center_mode == "zero":
        center = np.zeros((history.shape[0], history.shape[2]), dtype=np.float32)
    elif center_mode == "ewma_mean":
        center = ewma_mean_center(history, half_life=half_life)
    else:
        raise ValueError("center_mode must be 'zero' or 'ewma_mean'")
    return (
        ((history - center[:, None, :]) / scale[:, None, :]).astype(np.float32),
        ((future - center[:, None, :]) / scale[:, None, :]).astype(np.float32),
        center,
        scale,
    )


def unnormalize_increment_windows(
    normalized_increment: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    normalized = np.asarray(normalized_increment, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    while center_arr.ndim < normalized.ndim:
        center_arr = np.expand_dims(center_arr, axis=-2)
        scale_arr = np.expand_dims(scale_arr, axis=-2)
    return (normalized * scale_arr + center_arr).astype(np.float32)


def reconstruct_state_from_normalized_increments(
    last_history_state: np.ndarray,
    future_normalized_increment: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    specs: list[UnifiedVariableSpec],
) -> np.ndarray:
    increments = unnormalize_increment_windows(
        future_normalized_increment, center, scale
    )
    return reconstruct_state_from_increments(last_history_state, increments, specs)
