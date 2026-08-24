"""Numerical building blocks for narrative-conditioned ScenarioFlow rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

KEY_FACTOR_NAMES = {
    "SPX": "factor:spx",
    "US2Y": "factor:us2y",
    "US10Y": "factor:us10y",
    "BBB_OAS": "factor:bbb_oas",
    "AAA_OAS": "factor:aaa_oas",
    "USDJPY": "factor:usdjpy",
    "DXY": "factor:dxy",
    "GOLD": "factor:gold",
    "CRUDE_OIL": "factor:crude_oil",
    "VIX": "factor:vix",
}
SELECTED_IV_CELLS = (
    ("IV_ATM_3M", 7, "IV ATM 3M (K=1.00)"),
    ("IV_ATM_1Y", 17, "IV ATM 1Y (K=1.00)"),
    ("IV_OTM_PUT_1Y", 15, "IV OTM Put 1Y (K=0.70)"),
    ("IV_WING_6M_K130", 14, "IV Wing 6M (K=1.30)"),
)


@dataclass(frozen=True, slots=True)
class PrefixFeatureLayout:
    history_len: int
    n_cells: int
    level_size: int
    norm_size: int
    center_size: int
    log_scale_size: int
    drift_size: int

    @property
    def input_dim(self) -> int:
        return (
            self.level_size
            + self.norm_size
            + self.center_size
            + self.log_scale_size
            + self.drift_size
        )


@dataclass(frozen=True, slots=True)
class SupportSelection:
    mode: str
    window_indices: tuple[int, ...]
    weights: tuple[float, ...]
    candidates: tuple[Mapping[str, Any], ...]


class NarrativeAdapter(nn.Module):
    """Map a normalized text embedding into ScenarioFlow's memory space."""

    def __init__(
        self, embedding_dim: int, condition_dim: int, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or min(512, max(condition_dim * 2, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(condition_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class MemoryStartPrefixDecoder(nn.Module):
    """Construct a full numerical history from memory and the requested day zero."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(slots=True)
class StateAlignment:
    model: MemoryStartPrefixDecoder
    memory_mean: np.ndarray
    memory_std: np.ndarray
    start_mean: np.ndarray
    start_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    loss_first: float
    loss_last: float
    device: torch.device

    def predict_features(
        self, memory: np.ndarray, start_state: np.ndarray
    ) -> np.ndarray:
        values = np.concatenate(
            [
                (np.asarray(memory, dtype=np.float32) - self.memory_mean)
                / self.memory_std,
                (np.asarray(start_state, dtype=np.float32) - self.start_mean)
                / self.start_std,
            ],
            axis=1,
        ).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(torch.from_numpy(values).to(self.device))
        return (
            predicted.detach().cpu().numpy().astype(np.float32) * self.target_std
            + self.target_mean
        ).astype(np.float32)


def _safe_std(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    return np.maximum(
        np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True), floor
    ).astype(np.float32)


def fit_state_alignment(
    memory: np.ndarray,
    start_state: np.ndarray,
    features: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    hidden_dim: int = 256,
    steps: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 791,
    device: str | torch.device = "cpu",
) -> StateAlignment:
    """Fit the frozen-demo state-alignment architecture deterministically."""
    memory_values = np.asarray(memory, dtype=np.float32)
    starts = np.asarray(start_state, dtype=np.float32)
    targets = np.asarray(features, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if memory_values.ndim != 2 or starts.ndim != 2 or targets.ndim != 2:
        raise ValueError("memory, start_state, and features must be two-dimensional")
    if not (memory_values.shape[0] == starts.shape[0] == targets.shape[0]):
        raise ValueError("memory, start_state, and features must have matching rows")
    if train.size == 0 or test.size == 0:
        raise ValueError("train_indices and test_indices must be non-empty")
    memory_mean = memory_values[train].mean(axis=0, keepdims=True).astype(np.float32)
    memory_std = _safe_std(memory_values[train])
    start_mean = starts[train].mean(axis=0, keepdims=True).astype(np.float32)
    start_std = _safe_std(starts[train])
    inputs = np.concatenate(
        [
            (memory_values - memory_mean) / memory_std,
            (starts - start_mean) / start_std,
        ],
        axis=1,
    ).astype(np.float32)
    target_mean = targets[train].mean(axis=0, keepdims=True).astype(np.float32)
    target_std = _safe_std(targets[train])
    standardized_targets = ((targets - target_mean) / target_std).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    resolved_device = torch.device(device)
    model = MemoryStartPrefixDecoder(
        inputs.shape[1], targets.shape[1], int(hidden_dim)
    ).to(resolved_device)
    x = torch.from_numpy(inputs).to(resolved_device)
    y = torch.from_numpy(standardized_targets).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    batch = max(1, min(int(batch_size), int(train.size)))
    with torch.no_grad():
        loss_first = float(torch.mean((model(x[train]) - y[train]) ** 2).cpu())
    loss_last = loss_first
    model.train()
    for _ in range(int(steps)):
        positions = rng.choice(train, size=batch, replace=train.size < batch)
        loss = torch.mean((model(x[positions]) - y[positions]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_last = float(loss.detach().cpu())
    model.eval()
    return StateAlignment(
        model=model,
        memory_mean=memory_mean,
        memory_std=memory_std,
        start_mean=start_mean,
        start_std=start_std,
        target_mean=target_mean,
        target_std=target_std,
        loss_first=loss_first,
        loss_last=loss_last,
        device=resolved_device,
    )


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def build_prefix_feature_matrix(
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
) -> tuple[np.ndarray, PrefixFeatureLayout]:
    """Flatten one conditioning history relative to its final state."""
    level = np.asarray(history_level, dtype=np.float32)
    norm = np.asarray(history_norm, dtype=np.float32)
    ctr = np.asarray(center, dtype=np.float32)
    scl = np.asarray(scale, dtype=np.float32)
    drift = np.asarray(drift_feature, dtype=np.float32)
    if level.ndim != 3:
        raise ValueError("history_level must have shape [N,T,C]")
    n_windows, history_len, n_cells = level.shape
    _require_shape("history_norm", norm, (n_windows, history_len, n_cells))
    _require_shape("center", ctr, (n_windows, n_cells))
    _require_shape("scale", scl, (n_windows, n_cells))
    _require_shape("drift_feature", drift, (n_windows, n_cells))
    if np.any(~np.isfinite(scl)) or np.any(scl <= 0.0):
        raise ValueError("scale must be finite and positive")
    level_delta = level - level[:, -1:, :]
    feature = np.concatenate(
        [
            level_delta.reshape(n_windows, -1),
            norm.reshape(n_windows, -1),
            ctr,
            np.log(np.maximum(scl, 1e-8)),
            drift,
        ],
        axis=1,
    ).astype(np.float32)
    layout = PrefixFeatureLayout(
        history_len=history_len,
        n_cells=n_cells,
        level_size=history_len * n_cells,
        norm_size=history_len * n_cells,
        center_size=n_cells,
        log_scale_size=n_cells,
        drift_size=n_cells,
    )
    return feature, layout


def reconstruct_prefix_from_features(
    features: np.ndarray,
    *,
    start_state: np.ndarray,
    layout: PrefixFeatureLayout,
) -> dict[str, np.ndarray]:
    """Restore the conditioning arrays and pin their last state to day zero."""
    values = np.asarray(features, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != layout.input_dim:
        raise ValueError(f"features must have shape [N,{layout.input_dim}]")
    _require_shape("start_state", start, (values.shape[0], layout.n_cells))
    cursor = 0
    level_delta = values[:, cursor : cursor + layout.level_size].reshape(
        values.shape[0], layout.history_len, layout.n_cells
    )
    cursor += layout.level_size
    history_norm = values[:, cursor : cursor + layout.norm_size].reshape(
        values.shape[0], layout.history_len, layout.n_cells
    )
    cursor += layout.norm_size
    center = values[:, cursor : cursor + layout.center_size]
    cursor += layout.center_size
    log_scale = values[:, cursor : cursor + layout.log_scale_size]
    cursor += layout.log_scale_size
    drift = values[:, cursor : cursor + layout.drift_size]
    level_delta = level_delta.copy()
    level_delta[:, -1, :] = 0.0
    return {
        "history_level": (start[:, None, :] + level_delta).astype(np.float32),
        "history_norm": history_norm.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": np.exp(log_scale).astype(np.float32),
        "drift_feature": drift.astype(np.float32),
    }


def _cosine(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    rows = np.asarray(targets, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != q.shape[1]:
        raise ValueError("query and memory targets must have the same width")
    denominator = np.maximum(np.linalg.norm(rows, axis=1) * np.linalg.norm(q), 1e-8)
    return (np.sum(rows * q, axis=1) / denominator).astype(np.float32)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    scaled = (scores - float(np.max(scores))) / max(float(temperature), 1e-6)
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _direction_matches(
    history: np.ndarray,
    implications: Sequence[Mapping[str, Any]],
    spec_names: Sequence[str],
) -> bool:
    if not implications:
        return True
    index = {name: position for position, name in enumerate(spec_names)}
    checked = 0
    for implication in implications:
        factor = KEY_FACTOR_NAMES.get(str(implication.get("market", "")).upper())
        direction = str(implication.get("direction", "")).lower()
        if factor not in index or direction not in {"up", "down", "wider", "tighter"}:
            continue
        delta = float(history[-1, index[factor]] - history[0, index[factor]])
        expected_positive = direction in {"up", "wider"}
        checked += 1
        if (delta >= 0.0) != expected_positive:
            return False
    return checked > 0


def retrieve_support(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    train_indices: np.ndarray,
    query_start_state: np.ndarray,
    mode: str,
    top_k: int = 8,
    temperature: float = 0.2,
    min_index_gap: int = 30,
    implications: Sequence[Mapping[str, Any]] = (),
    spec_names: Sequence[str] = (),
) -> SupportSelection:
    """Select historical support by narrative memory or by day-zero state."""
    if mode not in {"narrative", "start_only"}:
        raise ValueError("mode must be 'narrative' or 'start_only'")
    memory = np.asarray(memory_targets, dtype=np.float32)
    history = np.asarray(history_level, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    start = history[:, -1, :]
    fit_mean = start[train].mean(axis=0, keepdims=True)
    fit_std = np.maximum(start[train].std(axis=0, keepdims=True), 1e-6)
    query_z = (
        np.asarray(query_start_state, dtype=np.float32)[None, :] - fit_mean
    ) / fit_std
    distances = np.linalg.norm((start[train] - fit_mean) / fit_std - query_z, axis=1)
    cosines = _cosine(query_memory, memory)[train]
    rows: list[dict[str, Any]] = []
    for position, window_index in enumerate(train):
        distance = float(distances[position])
        score = -distance
        if mode == "narrative":
            score = float(cosines[position]) - 0.02 * distance
            score -= 0.02 * max(distance - 15.0, 0.0)
        rows.append(
            {
                "window_index": int(window_index),
                "memory_support_cosine": float(cosines[position]),
                "start_distance_z": distance,
                "score": score,
                "direction_match": _direction_matches(
                    history[int(window_index)], implications, spec_names
                ),
            }
        )
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    if mode == "narrative" and implications:
        passing = [row for row in ordered if row["direction_match"]]
        if passing:
            ordered = passing
    selected: list[dict[str, Any]] = []
    if mode == "narrative" and ordered:
        anchor = ordered[0]
        selected.append(anchor)
        anchor_index = int(anchor["window_index"])
        remaining = [
            row for row in ordered[1:] if int(row["window_index"]) != anchor_index
        ]
        if remaining:
            remaining_indices = np.asarray(
                [int(row["window_index"]) for row in remaining], dtype=np.int64
            )
            family_cosine = _cosine(memory[anchor_index], memory[remaining_indices])
            remaining = [
                row
                for row, _ in sorted(
                    zip(remaining, family_cosine, strict=True),
                    key=lambda pair: (float(pair[1]), float(pair[0]["score"])),
                    reverse=True,
                )
            ]
    else:
        remaining = ordered
    for row in remaining:
        index = int(row["window_index"])
        if mode == "narrative" and any(
            abs(index - int(existing["window_index"])) < int(min_index_gap)
            for existing in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= max(1, int(top_k)):
            break
    selected = selected[: max(1, int(top_k))]
    scores = np.asarray([float(row["score"]) for row in selected], dtype=np.float32)
    weights = _softmax(scores, temperature)
    return SupportSelection(
        mode=mode,
        window_indices=tuple(int(row["window_index"]) for row in selected),
        weights=tuple(float(value) for value in weights),
        candidates=tuple(selected),
    )


def allocate_weighted_samples(weights: np.ndarray, total_samples: int) -> np.ndarray:
    """Allocate an exact sample budget across weighted support components."""
    total = int(total_samples)
    if total < 1:
        raise ValueError("total_samples must be positive")
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("weights must be non-empty")
    values = np.maximum(values, 0.0)
    if not np.all(np.isfinite(values)) or float(np.sum(values)) <= 0.0:
        values = np.ones_like(values)
    values /= float(np.sum(values))
    expected = values * total
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        for position in np.argsort(-(expected - counts))[:remainder]:
            counts[int(position)] += 1
    return counts


def _market_arrays(
    values: np.ndarray, spec_names: Sequence[str]
) -> list[tuple[str, str, np.ndarray]]:
    index = {name: position for position, name in enumerate(spec_names)}
    rows: list[tuple[str, str, np.ndarray]] = [
        ("IV_SURFACE", "IV surface average", np.nanmean(values[..., :25], axis=-1))
    ]
    rows.extend(
        (market, display, values[..., column])
        for market, column, display in SELECTED_IV_CELLS
    )
    rows.extend(
        (market, market, values[..., index[spec]])
        for market, spec in KEY_FACTOR_NAMES.items()
        if spec in index
    )
    return rows


def _market_start(start: np.ndarray, market: str, spec_names: Sequence[str]) -> float:
    if market == "IV_SURFACE":
        return float(np.nanmean(start[:25]))
    for name, column, _display in SELECTED_IV_CELLS:
        if name == market:
            return float(start[column])
    spec = KEY_FACTOR_NAMES[market]
    return float(start[list(spec_names).index(spec)])


def _representative_paths(values: np.ndarray, max_paths: int) -> list[list[float]]:
    count = min(int(max_paths), int(values.shape[0]))
    if count <= 0:
        return []
    order = np.argsort(values[:, -1])
    positions = np.unique(np.round(np.linspace(0, len(order) - 1, count)).astype(int))
    return [
        values[int(order[position])].astype(float).tolist() for position in positions
    ]


def path_quantiles(
    generated_states: np.ndarray,
    start_state: np.ndarray,
    spec_names: Sequence[str],
    *,
    max_paths: int = 6,
) -> list[dict[str, Any]]:
    """Build the fifteen fan-chart rows consumed by the React demo."""
    states = np.asarray(generated_states, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    if states.ndim != 3 or start.shape != (states.shape[-1],):
        raise ValueError("generated_states must be [S,T,C] with one [C] start")
    rows: list[dict[str, Any]] = []
    for market, display, series in _market_arrays(states, spec_names):
        rows.append(
            {
                "market": market,
                "display_name": display,
                "value_kind": "raw_level",
                "start_level": _market_start(start, market, spec_names),
                "p10": np.nanquantile(series, 0.10, axis=0).astype(float).tolist(),
                "p50": np.nanquantile(series, 0.50, axis=0).astype(float).tolist(),
                "p90": np.nanquantile(series, 0.90, axis=0).astype(float).tolist(),
                "mean": np.nanmean(series, axis=0).astype(float).tolist(),
                "sample_paths": _representative_paths(series, max_paths),
            }
        )
    return rows


def terminal_delta_summary(
    generated_states: np.ndarray,
    start_state: np.ndarray,
    spec_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize day-30 changes for the paper-style comparison table."""
    states = np.asarray(generated_states, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    if states.ndim != 3 or start.shape != (states.shape[-1],):
        raise ValueError("generated_states must be [S,T,C] with one [C] start")
    selected = [
        row
        for row in _market_arrays(states, spec_names)
        if not row[0].startswith("IV_") or row[0] == "IV_SURFACE"
    ]
    rows: list[dict[str, Any]] = []
    for market, _display, series in selected:
        delta = series[:, -1] - _market_start(start, market, spec_names)
        sigma = float(np.nanstd(delta))
        mean = float(np.nanmean(delta))
        rows.append(
            {
                "market": market,
                "mean_terminal_delta": mean,
                "p10": float(np.nanquantile(delta, 0.10)),
                "p50": float(np.nanquantile(delta, 0.50)),
                "p90": float(np.nanquantile(delta, 0.90)),
                "terminal_sample_count": int(np.sum(np.isfinite(delta))),
                "terminal_probability_up": float(np.nanmean(delta > 0.0)),
                "terminal_probability_down": float(np.nanmean(delta < 0.0)),
                "terminal_probability_flat": float(np.nanmean(delta == 0.0)),
                "terminal_mean_sigma": mean / sigma if sigma > 1e-8 else 0.0,
            }
        )
    return rows
