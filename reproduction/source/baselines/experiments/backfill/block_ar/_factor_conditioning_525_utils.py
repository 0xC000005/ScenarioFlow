from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


@dataclass(frozen=True)
class FactorHistoryBlock:
    history_01: torch.Tensor
    future_01: torch.Tensor
    factor_history: torch.Tensor
    indices: np.ndarray
    factor_mean: np.ndarray
    factor_std: np.ndarray
    factor_columns: list[str]


def official_train_val_indices(
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    train_indices = np.arange(0, max_train_idx - int(val_size), dtype=np.int64)
    val_indices = np.arange(max_train_idx - int(val_size), max_train_idx, dtype=np.int64)
    return train_indices, val_indices


def recent_prevalidation_indices(
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
    adaptation_windows: int,
) -> np.ndarray:
    train_indices, _ = official_train_val_indices(
        test_start=test_start,
        val_size=val_size,
        history_len=history_len,
        future_len=future_len,
    )
    n = min(int(adaptation_windows), int(train_indices.shape[0]))
    return train_indices[-n:]


def load_factor_panel() -> tuple[np.ndarray, list[str]]:
    data = load_aligned_38d_data()
    returns = data["factor_returns_13"].astype(np.float32)
    levels = data["factor_levels_13"].astype(np.float32)
    features = np.concatenate([returns, levels], axis=1)
    columns = (
        [f"ret:{name}" for name in data["factor_return_columns"]]
        + [f"level:{name}" for name in data["factor_level_columns"]]
    )
    return features, columns


def factor_histories(
    factor_features: np.ndarray,
    indices: np.ndarray,
    history_len: int,
) -> np.ndarray:
    histories = [factor_features[s : s + int(history_len)] for s in indices]
    return np.asarray(histories, dtype=np.float32)


def fit_factor_standardizer(
    factor_features: np.ndarray,
    fit_indices: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit = factor_histories(factor_features, fit_indices, history_len)
    flat = fit.reshape(-1, fit.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def build_factor_history_block(
    data_path: str,
    indices: np.ndarray,
    history_len: int,
    future_len: int,
    device: torch.device,
    factor_mean: np.ndarray | None = None,
    factor_std: np.ndarray | None = None,
    fit_indices: np.ndarray | None = None,
) -> FactorHistoryBlock:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, future_01_flat = build_multistep_windows(
        indices,
        surf_tensor,
        history_len,
        future_len,
    )
    future_01 = future_01_flat.view(history_01.shape[0], future_len, 5, 5)

    factor_features, factor_columns = load_factor_panel()
    if factor_features.shape[0] != surfaces.shape[0]:
        raise ValueError(
            f"Factor/surface length mismatch: {factor_features.shape[0]} vs {surfaces.shape[0]}"
        )
    if factor_mean is None or factor_std is None:
        if fit_indices is None:
            fit_indices = indices
        factor_mean, factor_std = fit_factor_standardizer(
            factor_features,
            fit_indices,
            history_len,
        )
    factor_hist = factor_histories(factor_features, indices, history_len)
    factor_hist = (factor_hist - factor_mean[None, None, :]) / factor_std[None, None, :]
    return FactorHistoryBlock(
        history_01=history_01,
        future_01=future_01,
        factor_history=torch.from_numpy(factor_hist).to(device),
        indices=indices,
        factor_mean=factor_mean.astype(np.float32),
        factor_std=factor_std.astype(np.float32),
        factor_columns=factor_columns,
    )


def tensor_dict_summary(block: FactorHistoryBlock) -> dict[str, Any]:
    return {
        "n_windows": int(block.history_01.shape[0]),
        "history_shape": list(block.history_01.shape),
        "future_shape": list(block.future_01.shape),
        "factor_history_shape": list(block.factor_history.shape),
        "index_start": int(block.indices[0]),
        "index_end": int(block.indices[-1]),
        "factor_dim": int(block.factor_history.shape[-1]),
    }
