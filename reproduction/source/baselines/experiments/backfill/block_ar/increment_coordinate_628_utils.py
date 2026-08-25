"""628a helpers for training joint scenario models in daily-change coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    UnifiedVariableSpec,
    build_unified_variable_specs,
    decode_state,
    encode_state,
)


@dataclass(frozen=True)
class IncrementCoordinateBlock:
    history_increment: np.ndarray
    future_increment: np.ndarray
    history_state: np.ndarray
    future_state: np.ndarray
    indices: np.ndarray
    specs: list[UnifiedVariableSpec]


def state_panel_from_specs(
    panel: np.ndarray, specs: list[UnifiedVariableSpec]
) -> np.ndarray:
    """Extract the unique raw state panel represented by `specs`."""
    panel = np.asarray(panel, dtype=np.float64)
    state = np.stack([panel[:, spec.source_index] for spec in specs], axis=1)
    return state.astype(np.float64)


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
    """Build windows whose model coordinate is encoded daily changes.

    Raw levels are retained for reconstruction and evaluation. The generated
    object is the daily change in encoded coordinates: log changes for positive
    level-like variables and arithmetic differences for diff-level variables.
    """
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
            hist_encoded = encoded_panel[start - 1 : hist_end]
            hist_increment = np.diff(hist_encoded, axis=0)
        else:
            hist_encoded = encoded_panel[start:hist_end]
            hist_increment = np.zeros((int(history_len), len(specs)), dtype=np.float64)
            if int(history_len) > 1:
                hist_increment[1:] = np.diff(hist_encoded, axis=0)

        fut_encoded = encoded_panel[hist_end - 1 : fut_end]
        history_increment.append(hist_increment)
        future_increment.append(np.diff(fut_encoded, axis=0))
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
    """Integrate encoded future increments and decode them back to raw states."""
    increments = np.asarray(future_increment, dtype=np.float64)
    base = encode_state(np.asarray(last_history_state, dtype=np.float64), specs)
    while base.ndim < increments.ndim:
        base = np.expand_dims(base, axis=-2)
    encoded_future = base + np.cumsum(increments, axis=-2)
    return decode_state(encoded_future, specs).astype(np.float32)
