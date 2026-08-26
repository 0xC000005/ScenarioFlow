"""Deck-free per-window score artifacts for final benchmark evaluations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "fmts.window_scores.v1"
HORIZONS = np.asarray([1, 7, 14, 30], dtype=np.int64)
COVERAGE_LEVELS = np.asarray([0.5, 0.8, 0.9, 0.95], dtype=np.float64)


def add_window_score_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save-window-scores", action="store_true")


def _score_contributions(
    samples_delta: np.ndarray,
    ground_truth_delta: np.ndarray,
    train_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, samples, future, variables = samples_delta.shape
    horizons = HORIZONS[HORIZONS <= future]
    crps = np.empty((windows, len(horizons), variables), dtype=np.float64)
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    samples_z = (samples_delta - mean[None, None, None, :]) / std[None, None, None, :]
    truth_z = (ground_truth_delta - mean[None, None, :]) / std[None, None, :]
    energy = np.empty((windows, len(horizons)), dtype=np.float64)
    for position, horizon in enumerate(horizons):
        time = int(horizon) - 1
        values = samples_delta[:, :, time]
        truth = ground_truth_delta[:, time]
        point = np.abs(values - truth[:, None]).mean(axis=1)
        pairwise = np.zeros((windows, variables), dtype=np.float64)
        for index in range(samples):
            pairwise += np.abs(values[:, index : index + 1] - values).sum(axis=1)
        crps[:, position] = point - 0.5 * pairwise / float(samples * samples)
        values_z = samples_z[:, :, time]
        truth_time_z = truth_z[:, time]
        point_energy = np.linalg.norm(values_z - truth_time_z[:, None], axis=2).mean(
            axis=1
        )
        pair_energy = np.zeros(windows, dtype=np.float64)
        for index in range(samples):
            pair_energy += np.linalg.norm(
                values_z[:, index : index + 1] - values_z, axis=2
            ).sum(axis=1)
        energy[:, position] = point_energy - 0.5 * pair_energy / float(
            samples * samples
        )
    time = int(horizons[-1]) - 1
    values_z = samples_z[:, :, time]
    truth_time_z = truth_z[:, time]
    sample_variogram = np.mean(
        np.abs(values_z[:, :, :, None] - values_z[:, :, None, :]) ** 0.5,
        axis=1,
    )
    truth_variogram = np.abs(truth_time_z[:, :, None] - truth_time_z[:, None, :]) ** 0.5
    upper = np.triu(np.ones((variables, variables), dtype=bool), k=1)
    variogram = ((sample_variogram - truth_variogram) ** 2)[:, upper].mean(axis=1)
    return crps, energy, variogram


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if left.size < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(
        np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]
    )


def build_window_score_artifact(
    *,
    samples_delta: Any,
    ground_truth_delta: Any,
    train_delta: Any,
    conditional_iv_samples: Any,
    ground_truth_iv: Any,
    history_iv: Any,
    raw_history: Any,
    sample_raw: Any,
    window_indices: Any,
    variable_names: Sequence[str],
    existing_result: Mapping[str, Any],
    iv_count: int = 25,
) -> dict[str, np.ndarray]:
    """Build the final window artifact and verify its proper-score aggregates."""
    samples_array = np.asarray(samples_delta, dtype=np.float64)
    truth_array = np.asarray(ground_truth_delta, dtype=np.float64)
    train_array = np.asarray(train_delta, dtype=np.float64)
    iv_samples = np.asarray(conditional_iv_samples, dtype=np.float64)
    iv_truth = np.asarray(ground_truth_iv, dtype=np.float64)
    iv_history = np.asarray(history_iv, dtype=np.float64)
    raw_history_array = np.asarray(raw_history, dtype=np.float64)
    sample_raw_array = np.asarray(sample_raw, dtype=np.float64)
    if samples_array.ndim != 4 or samples_array.shape[-1] != len(variable_names):
        raise ValueError("samples_delta must be [window, sample, day, variable]")
    crps, energy, variogram = _score_contributions(
        samples_array, truth_array, train_array
    )
    expected = existing_result["panel_b_crps"]["overall"]
    if not np.isclose(crps.mean(), expected, rtol=1e-6, atol=1e-8):
        raise ValueError("CRPS aggregate does not match result JSON")
    levels = COVERAGE_LEVELS
    coverage = np.empty((samples_array.shape[0], len(levels)))
    for position, level in enumerate(levels):
        alpha = (1.0 - float(level)) / 2.0
        lower = np.quantile(iv_samples, alpha, axis=1)
        upper = np.quantile(iv_samples, 1.0 - alpha, axis=1)
        coverage[:, position] = ((iv_truth >= lower) & (iv_truth <= upper)).mean(
            axis=(1, 2, 3)
        )
    iv_activity = (np.diff(iv_history, axis=1) ** 2).mean(axis=(1, 2, 3))
    iv_width = (
        np.quantile(iv_samples, 0.95, axis=1) - np.quantile(iv_samples, 0.05, axis=1)
    ).mean(axis=(1, 2, 3))
    panel_activity = (np.diff(raw_history_array, axis=1) ** 2).mean(axis=(1, 2))
    panel_width = (
        np.quantile(sample_raw_array, 0.95, axis=1)
        - np.quantile(sample_raw_array, 0.05, axis=1)
    ).mean(axis=(1, 2))
    _rank_correlation(iv_activity, iv_width)
    _rank_correlation(panel_activity, panel_width)
    horizons = HORIZONS[HORIZONS <= samples_array.shape[2]]
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "window_indices": np.asarray(window_indices, dtype=np.int64),
        "horizons": horizons,
        "variable_names": np.asarray(list(variable_names), dtype=np.str_),
        "crps_contributions": crps,
        "energy_contributions": energy,
        "variogram_h30_contributions": variogram,
        "coverage_levels": levels,
        "coverage_by_window": coverage,
        "iv_history_activity": iv_activity,
        "iv_deck_width": iv_width,
        "full_panel_history_activity": panel_activity,
        "full_panel_deck_width": panel_width,
    }


def maybe_write_window_score_artifact(
    *, enabled: bool, output_path: str | Path, **inputs: Any
) -> Path | None:
    if not enabled:
        return None
    output = Path(output_path)
    artifact = build_window_score_artifact(**inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **artifact)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
