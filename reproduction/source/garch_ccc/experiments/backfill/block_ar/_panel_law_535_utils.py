from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PanelBlock:
    history_panel: torch.Tensor
    future_panel: torch.Tensor
    indices: np.ndarray
    columns: list[str]


def load_aligned_iv_factor_panel(
    iv_path: str = "data/vol_surface_with_ret.npz",
    iv_parquet: str = "data/spx_vol_surface_history_full_data_fixed.parquet",
    factor_levels_parquet: str = "data/multi_factor_levels.parquet",
    factor_returns_parquet: str = "data/multi_factor_returns.parquet",
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    iv_npz = np.load(iv_path)
    iv = iv_npz["surface"].astype(np.float32).reshape(iv_npz["surface"].shape[0], -1)
    iv_frame = pd.read_parquet(iv_parquet)
    iv_dates = pd.DatetimeIndex(pd.to_datetime(iv_frame["date"]))
    factor_levels = pd.read_parquet(factor_levels_parquet).reindex(iv_dates)
    factor_returns = pd.read_parquet(factor_returns_parquet).reindex(iv_dates)
    factor = pd.concat([factor_levels, factor_returns], axis=1)
    factor = factor.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    if factor.isna().any().any():
        raise ValueError("Factor panel still contains NaNs after alignment")
    if iv.shape[0] != factor.shape[0]:
        raise ValueError(f"IV/factor length mismatch: {iv.shape[0]} vs {factor.shape[0]}")
    panel = np.concatenate([iv, factor.to_numpy(dtype=np.float32)], axis=1).astype(np.float32)
    columns = [f"iv:{idx:02d}" for idx in range(iv.shape[1])] + [
        f"factor:{col}" for col in factor.columns
    ]
    return panel, columns, iv_dates


def build_panel_block(
    panel: np.ndarray,
    columns: list[str],
    indices: np.ndarray,
    history_len: int,
    future_len: int,
    device: torch.device,
) -> PanelBlock:
    histories = []
    futures = []
    for idx in indices:
        start = int(idx)
        histories.append(panel[start : start + history_len])
        futures.append(panel[start + history_len : start + history_len + future_len])
    return PanelBlock(
        history_panel=torch.from_numpy(np.asarray(histories, dtype=np.float32)).to(device),
        future_panel=torch.from_numpy(np.asarray(futures, dtype=np.float32)).to(device),
        indices=indices,
        columns=columns,
    )


def panel_summary(block: PanelBlock) -> dict[str, object]:
    return {
        "n_windows": int(block.history_panel.shape[0]),
        "history_shape": list(block.history_panel.shape),
        "future_shape": list(block.future_panel.shape),
        "index_start": int(block.indices[0]),
        "index_end": int(block.indices[-1]),
        "n_columns": len(block.columns),
        "columns_head": block.columns[:8],
        "columns_tail": block.columns[-8:],
    }
