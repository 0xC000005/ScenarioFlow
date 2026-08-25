"""Audited, deck-free per-window score artifacts for FMTS evaluations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "fmts.window_scores.v1"
HORIZONS = np.asarray([1, 7, 14, 30], dtype=np.int64)
COVERAGE_LEVELS = np.asarray([0.5, 0.8, 0.9, 0.95], dtype=np.float64)
IV_VARIABLE_COUNT = 25
PANEL_VARIABLE_COUNT = 39
_RTOL = 1e-6
_ATOL = 1e-8


def add_window_score_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common opt-in flag used by every current-panel evaluator."""
    parser.add_argument(
        "--save-window-scores",
        action="store_true",
        help=(
            "write deck-free per-window scoring contributions to "
            "window_scores.npz under each candidate output directory"
        ),
    )


def _as_float_array(name: str, value: Any, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _assert_close(label: str, actual: Any, expected: Any) -> None:
    try:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=_RTOL,
            atol=_ATOL,
        )
    except AssertionError as exc:
        raise ValueError(
            f"{label} aggregate does not match results_current_panel.json"
        ) from exc


def _mapping_value(mapping: Mapping[Any, Any], key: int | float) -> Any:
    for candidate in (key, str(key), f"{float(key):g}"):
        if candidate in mapping:
            return mapping[candidate]
    raise ValueError(f"missing aggregate key {key!r} in results_current_panel.json")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _iv_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Match the tie-aware IV risk-state diagnostic without importing it."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return 0.0
    x = x[keep]
    y = y[keep]
    x_scale = max(float(np.max(np.abs(x))), 1.0)
    y_scale = max(float(np.max(np.abs(y))), 1.0)
    if float(np.ptp(x)) / x_scale < 1e-6 or float(np.ptp(y)) / y_scale < 1e-6:
        return 0.0
    xr = _average_ranks(x)
    yr = _average_ranks(y)
    if float(np.std(xr)) < 1e-12 or float(np.std(yr)) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(xr, yr)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _full_panel_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Match the ordinal full-panel diagnostic without importing it."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def _score_contributions(
    samples_delta: np.ndarray,
    ground_truth_delta: np.ndarray,
    train_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_windows, n_samples, future_len, n_variables = samples_delta.shape
    crps = np.empty((n_windows, len(HORIZONS), n_variables), dtype=np.float64)

    train_mean = train_delta.mean(axis=0)
    train_std = train_delta.std(axis=0) + 1e-8
    standardized_samples = (
        samples_delta - train_mean[None, None, None, :]
    ) / train_std[None, None, None, :]
    standardized_truth = (ground_truth_delta - train_mean[None, None, :]) / train_std[
        None, None, :
    ]
    energy = np.empty((n_windows, len(HORIZONS)), dtype=np.float64)

    for horizon_position, horizon in enumerate(HORIZONS):
        if int(horizon) > future_len:
            raise ValueError(
                f"future length {future_len} does not contain required horizon {horizon}"
            )
        time_index = int(horizon) - 1
        samples_h = samples_delta[:, :, time_index, :]
        truth_h = ground_truth_delta[:, time_index, :]
        point_error = np.abs(samples_h - truth_h[:, None, :]).mean(axis=1)
        pairwise = np.zeros((n_windows, n_variables), dtype=np.float64)
        for sample_index in range(n_samples):
            pairwise += np.abs(
                samples_h[:, sample_index : sample_index + 1, :] - samples_h
            ).sum(axis=1)
        pairwise /= float(n_samples * n_samples)
        crps[:, horizon_position, :] = point_error - 0.5 * pairwise

        samples_z = standardized_samples[:, :, time_index, :]
        truth_z = standardized_truth[:, time_index, :]
        energy_point = np.linalg.norm(samples_z - truth_z[:, None, :], axis=2).mean(
            axis=1
        )
        energy_pairwise = np.zeros(n_windows, dtype=np.float64)
        for sample_index in range(n_samples):
            energy_pairwise += np.linalg.norm(
                samples_z[:, sample_index : sample_index + 1, :] - samples_z,
                axis=2,
            ).sum(axis=1)
        energy_pairwise /= float(n_samples * n_samples)
        energy[:, horizon_position] = energy_point - 0.5 * energy_pairwise

    horizon_index = int(HORIZONS[-1]) - 1
    samples_h30 = (
        samples_delta[:, :, horizon_index, :] - train_mean[None, None, :]
    ) / train_std[None, None, :]
    truth_h30 = (
        ground_truth_delta[:, horizon_index, :] - train_mean[None, :]
    ) / train_std[None, :]
    sample_variogram = np.mean(
        np.abs(samples_h30[:, :, :, None] - samples_h30[:, :, None, :]) ** 0.5,
        axis=1,
    )
    truth_variogram = np.abs(truth_h30[:, :, None] - truth_h30[:, None, :]) ** 0.5
    upper_triangle = np.triu(np.ones((n_variables, n_variables), dtype=bool), k=1)
    variogram_h30 = ((sample_variogram - truth_variogram) ** 2)[:, upper_triangle].mean(
        axis=1
    )
    return crps, energy, variogram_h30


def _validate_score_aggregates(
    crps: np.ndarray,
    energy: np.ndarray,
    variogram_h30: np.ndarray,
    existing_result: Mapping[str, Any],
    *,
    iv_count: int,
) -> None:
    expected_crps = existing_result["panel_b_crps"]
    for horizon_position, horizon in enumerate(HORIZONS):
        _assert_close(
            f"CRPS h{horizon}",
            crps[:, horizon_position, :].mean(axis=0),
            _mapping_value(expected_crps["per_horizon"], int(horizon)),
        )
    per_dimension = crps.mean(axis=(0, 1))
    _assert_close("CRPS per dimension", per_dimension, expected_crps["per_dimension"])
    _assert_close("CRPS overall", per_dimension.mean(), expected_crps["overall"])
    _assert_close("CRPS IV", per_dimension[:iv_count].mean(), expected_crps["iv_crps"])
    _assert_close(
        "CRPS factors", per_dimension[iv_count:].mean(), expected_crps["factor_crps"]
    )

    expected_energy = existing_result["panel_b_energy_score"]
    for horizon_position, horizon in enumerate(HORIZONS):
        _assert_close(
            f"Energy h{horizon}",
            energy[:, horizon_position].mean(),
            _mapping_value(expected_energy["per_horizon"], int(horizon)),
        )
    _assert_close("Energy overall", energy.mean(), expected_energy["overall"])
    _assert_close(
        "Variogram h30",
        variogram_h30.mean(),
        existing_result["panel_b_variogram_score"]["variogram_score"],
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
    iv_count: int = IV_VARIABLE_COUNT,
) -> dict[str, np.ndarray]:
    """Build and audit the shared per-window artifact without retaining decks."""
    samples_delta_arr = _as_float_array("samples_delta", samples_delta, 4)
    ground_truth_delta_arr = _as_float_array(
        "ground_truth_delta", ground_truth_delta, 3
    )
    train_delta_arr = _as_float_array("train_delta", train_delta, 2)
    conditional_iv_arr = _as_float_array(
        "conditional_iv_samples", conditional_iv_samples, 5
    )
    ground_truth_iv_arr = _as_float_array("ground_truth_iv", ground_truth_iv, 4)
    history_iv_arr = _as_float_array("history_iv", history_iv, 4)
    raw_history_arr = _as_float_array("raw_history", raw_history, 3)
    sample_raw_arr = _as_float_array("sample_raw", sample_raw, 4)

    n_windows, n_samples, future_len, n_variables = samples_delta_arr.shape
    if n_variables != PANEL_VARIABLE_COUNT:
        raise ValueError(
            f"current-panel artifact requires 39 variables, got {n_variables}"
        )
    if int(iv_count) != IV_VARIABLE_COUNT:
        raise ValueError(f"current-panel artifact requires iv_count=25, got {iv_count}")
    if ground_truth_delta_arr.shape != (n_windows, future_len, n_variables):
        raise ValueError("ground_truth_delta shape does not match samples_delta")
    if train_delta_arr.shape[1] != n_variables:
        raise ValueError("train_delta variable count does not match samples_delta")
    if sample_raw_arr.shape != samples_delta_arr.shape:
        raise ValueError("sample_raw shape does not match samples_delta")
    if raw_history_arr.shape[0] != n_windows or raw_history_arr.shape[2] != n_variables:
        raise ValueError("raw_history shape does not match samples_delta")
    expected_iv_sample_shape = (n_windows, n_samples, future_len, 5, 5)
    if conditional_iv_arr.shape != expected_iv_sample_shape:
        raise ValueError(
            "conditional_iv_samples must have shape "
            f"{expected_iv_sample_shape}, got {conditional_iv_arr.shape}"
        )
    if ground_truth_iv_arr.shape != (n_windows, future_len, 5, 5):
        raise ValueError("ground_truth_iv shape does not match conditional_iv_samples")
    if history_iv_arr.shape[0] != n_windows or history_iv_arr.shape[2:] != (5, 5):
        raise ValueError("history_iv shape does not match conditional_iv_samples")

    indices = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] != n_windows:
        raise ValueError(
            f"window_indices length {indices.shape[0]} does not match {n_windows} windows"
        )
    if indices.size > 1 and not np.all(np.diff(indices) > 0):
        raise ValueError("window_indices must be strictly increasing chronologically")

    names = np.asarray([str(name) for name in variable_names], dtype=np.str_)
    if names.shape[0] != n_variables:
        raise ValueError(
            f"variable_names length {names.shape[0]} does not match {n_variables} variables"
        )

    crps, energy, variogram_h30 = _score_contributions(
        samples_delta_arr, ground_truth_delta_arr, train_delta_arr
    )
    _validate_score_aggregates(
        crps, energy, variogram_h30, existing_result, iv_count=int(iv_count)
    )

    coverage_by_window = np.empty(
        (n_windows, COVERAGE_LEVELS.shape[0]), dtype=np.float64
    )
    expected_coverage = existing_result["current_panel_iv"]["coverage"]["overall"]
    for level_position, level in enumerate(COVERAGE_LEVELS):
        alpha = (1.0 - float(level)) / 2.0
        lower = np.quantile(conditional_iv_arr, alpha, axis=1)
        upper = np.quantile(conditional_iv_arr, 1.0 - alpha, axis=1)
        covered = (ground_truth_iv_arr >= lower) & (ground_truth_iv_arr <= upper)
        coverage_by_window[:, level_position] = covered.mean(axis=(1, 2, 3))
        _assert_close(
            f"coverage {level:g}",
            coverage_by_window[:, level_position].mean(),
            _mapping_value(expected_coverage, float(level)),
        )

    iv_history_activity = (np.diff(history_iv_arr, axis=1) ** 2).mean(axis=(1, 2, 3))
    iv_lower = np.quantile(conditional_iv_arr, 0.05, axis=1)
    iv_upper = np.quantile(conditional_iv_arr, 0.95, axis=1)
    iv_deck_width = (iv_upper - iv_lower).mean(axis=(1, 2, 3))

    # The authoritative IV risk-state aggregate explicitly promotes its
    # inputs before computing activity and quantiles.  Use that precision for
    # the audit while retaining native-dtype per-window values in the shared
    # artifact, matching artifacts already produced under this protocol.
    iv_history_for_risk = history_iv_arr.astype(np.float64, copy=False)
    conditional_iv_for_risk = conditional_iv_arr.astype(np.float64, copy=False)
    iv_history_activity_for_audit = (np.diff(iv_history_for_risk, axis=1) ** 2).mean(
        axis=(1, 2, 3)
    )
    iv_lower_for_audit = np.quantile(conditional_iv_for_risk, 0.05, axis=1)
    iv_upper_for_audit = np.quantile(conditional_iv_for_risk, 0.95, axis=1)
    iv_deck_width_for_audit = (iv_upper_for_audit - iv_lower_for_audit).mean(
        axis=(1, 2, 3)
    )
    _assert_close(
        "IV history-width Spearman",
        _iv_spearman(iv_history_activity_for_audit, iv_deck_width_for_audit),
        existing_result["current_panel_iv"]["risk_state_allocation"][
            "history_width_spearman"
        ],
    )

    full_panel_history_activity = (np.diff(raw_history_arr, axis=1) ** 2).mean(
        axis=(1, 2)
    )
    full_lower = np.quantile(sample_raw_arr, 0.05, axis=1)
    full_upper = np.quantile(sample_raw_arr, 0.95, axis=1)
    full_panel_deck_width = (full_upper - full_lower).mean(axis=(1, 2))
    _assert_close(
        "full-panel history-width Spearman",
        _full_panel_spearman(full_panel_history_activity, full_panel_deck_width),
        existing_result["joint_panel_quality"]["conditional_panel"][
            "history_activity_width_spearman"
        ],
    )

    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "window_indices": indices,
        "horizons": HORIZONS.copy(),
        "variable_names": names,
        "crps_contributions": crps,
        "energy_contributions": energy,
        "variogram_h30_contributions": variogram_h30,
        "coverage_levels": COVERAGE_LEVELS.copy(),
        "coverage_by_window": coverage_by_window,
        "iv_history_activity": iv_history_activity,
        "iv_deck_width": iv_deck_width,
        "full_panel_history_activity": full_panel_history_activity,
        "full_panel_deck_width": full_panel_deck_width,
    }


def maybe_write_window_score_artifact(
    *,
    enabled: bool,
    output_path: str | Path,
    **artifact_inputs: Any,
) -> Path | None:
    """Write the audited artifact atomically when explicitly enabled."""
    if not enabled:
        return None
    output = Path(output_path)
    artifact = build_window_score_artifact(**artifact_inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **artifact)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
