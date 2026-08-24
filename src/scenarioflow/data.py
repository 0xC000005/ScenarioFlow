"""Aligned IV/factor panel loading and window preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PanelPaths:
    iv_npz: Path
    iv_parquet: Path
    factor_levels_parquet: Path
    factor_returns_parquet: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "PanelPaths":
        root = Path(data_dir)
        return cls(
            iv_npz=root / "vol_surface_with_ret.npz",
            iv_parquet=root / "spx_vol_surface_history_full_data_fixed.parquet",
            factor_levels_parquet=root / "multi_factor_levels.parquet",
            factor_returns_parquet=root / "multi_factor_returns.parquet",
        )


@dataclass(frozen=True)
class PanelBlock:
    history_panel: torch.Tensor
    future_panel: torch.Tensor
    indices: np.ndarray
    columns: list[str]


def load_aligned_iv_factor_panel(
    paths: PanelPaths | None = None,
    *,
    data_dir: str | Path = "data",
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Load the date-aligned 25-IV plus level/return factor panel."""
    resolved = paths if paths is not None else PanelPaths.from_data_dir(data_dir)
    with np.load(resolved.iv_npz, allow_pickle=False) as iv_data:
        surface = iv_data["surface"].astype(np.float32)
        iv = surface.reshape(surface.shape[0], -1)
    iv_frame = pd.read_parquet(resolved.iv_parquet)
    iv_dates = pd.DatetimeIndex(pd.to_datetime(iv_frame["date"]))
    factor_levels = pd.read_parquet(resolved.factor_levels_parquet).reindex(iv_dates)
    factor_returns = pd.read_parquet(resolved.factor_returns_parquet).reindex(iv_dates)
    factor = pd.concat([factor_levels, factor_returns], axis=1)
    factor = factor.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    if factor.isna().any().any():
        raise ValueError("Factor panel still contains NaNs after alignment")
    if iv.shape[0] != factor.shape[0]:
        raise ValueError(
            f"IV/factor length mismatch: {iv.shape[0]} vs {factor.shape[0]}"
        )
    panel = np.concatenate([iv, factor.to_numpy(dtype=np.float32)], axis=1).astype(
        np.float32
    )
    columns = [f"iv:{idx:02d}" for idx in range(iv.shape[1])] + [
        f"factor:{column}" for column in factor.columns
    ]
    return panel, columns, iv_dates


def official_train_val_indices(
    *,
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen pre-test train/validation window split."""
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    if max_train_idx < int(val_size):
        raise ValueError("test_start is too early for the requested window lengths")
    train_indices = np.arange(0, max_train_idx - int(val_size), dtype=np.int64)
    val_indices = np.arange(
        max_train_idx - int(val_size), max_train_idx, dtype=np.int64
    )
    return train_indices, val_indices


def build_panel_block(
    panel: np.ndarray,
    columns: list[str],
    indices: np.ndarray,
    history_len: int,
    future_len: int,
    device: torch.device | str = "cpu",
) -> PanelBlock:
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    for raw_idx in np.asarray(indices, dtype=np.int64):
        start = int(raw_idx)
        end = start + int(history_len) + int(future_len)
        if start < 0 or end > panel.shape[0]:
            raise ValueError(f"window {start} exceeds panel bounds")
        histories.append(panel[start : start + history_len])
        futures.append(panel[start + history_len : end])
    return PanelBlock(
        history_panel=torch.from_numpy(np.asarray(histories, dtype=np.float32)).to(
            device
        ),
        future_panel=torch.from_numpy(np.asarray(futures, dtype=np.float32)).to(device),
        indices=np.asarray(indices, dtype=np.int64),
        columns=list(columns),
    )
