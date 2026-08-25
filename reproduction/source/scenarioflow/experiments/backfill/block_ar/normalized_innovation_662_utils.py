"""662a helpers for state-aware normalized-innovation scenario models."""

from __future__ import annotations

import numpy as np

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    UnifiedVariableSpec,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (
    reconstruct_state_from_increments,
)


def ewma_rms_scale(
    history_increment: np.ndarray,
    *,
    half_life: float | None = None,
    scale_floor: float = 1e-4,
) -> np.ndarray:
    """History-only EWMA/RMS scale for encoded increments.

    The most recent history increment gets the largest weight. The returned
    shape is `[n_windows, n_channels]`.
    """
    increments = np.asarray(history_increment, dtype=np.float64)
    if increments.ndim != 3:
        raise ValueError("history_increment must have shape [windows, history, channels]")
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
    history_increment: np.ndarray,
    *,
    half_life: float | None = None,
) -> np.ndarray:
    """History-only signed EWMA mean for encoded increments."""
    increments = np.asarray(history_increment, dtype=np.float64)
    if increments.ndim != 3:
        raise ValueError("history_increment must have shape [windows, history, channels]")
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
    """Normalize encoded increments using history-only local statistics."""
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
    history_norm = (history - center[:, None, :]) / scale[:, None, :]
    future_norm = (future - center[:, None, :]) / scale[:, None, :]
    return (
        history_norm.astype(np.float32),
        future_norm.astype(np.float32),
        center,
        scale,
    )


def unnormalize_increment_windows(
    normalized_increment: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
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
    increments = unnormalize_increment_windows(future_normalized_increment, center, scale)
    return reconstruct_state_from_increments(last_history_state, increments, specs)
