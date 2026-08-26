"""Shared final-panel evaluation and result serialization."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..reproduction.metrics import (
    compute_correlation_score,
    compute_crps,
    compute_energy_score,
    compute_factor_ks,
    compute_variogram_score,
    panel_daily_changes,
    summarize_joint_quality,
)
from ..state import reconstruct_state_from_increments
from .window_scores import maybe_write_window_score_artifact


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-12))


def _lag_one(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, values.shape[-1])
    rows = []
    for channel in range(flattened.shape[1]):
        left = flattened[:-1, channel]
        right = flattened[1:, channel]
        rows.append(
            0.0
            if np.std(left) < 1e-12 or np.std(right) < 1e-12
            else float(np.corrcoef(left, right)[0, 1])
        )
    return np.asarray(rows)


def iv_diagnostics(
    history_iv: np.ndarray, future_iv: np.ndarray, sample_iv: np.ndarray
) -> tuple[dict[str, Any], dict[str, float]]:
    levels = (0.5, 0.8, 0.9, 0.95)
    coverage: dict[str, float] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(sample_iv, alpha, axis=1)
        upper = np.quantile(sample_iv, 1.0 - alpha, axis=1)
        coverage[str(level)] = float(
            ((future_iv >= lower) & (future_iv <= upper)).mean()
        )
    history_activity = (np.diff(history_iv, axis=1) ** 2).mean(axis=(1, 2, 3))
    width = (
        np.quantile(sample_iv, 0.95, axis=1) - np.quantile(sample_iv, 0.05, axis=1)
    ).mean(axis=(1, 2, 3))
    width_rho = (
        float(stats.spearmanr(history_activity, width).statistic)
        if history_activity.size >= 3
        else 0.0
    )
    if not np.isfinite(width_rho):
        width_rho = 0.0
    ks = [
        float(
            stats.ks_2samp(
                sample_iv[..., row, column].reshape(-1),
                future_iv[..., row, column].reshape(-1),
            ).statistic
        )
        for row in range(5)
        for column in range(5)
    ]
    above = (sample_iv > future_iv[:, None]).mean(axis=1)
    generated_delta = np.diff(
        np.concatenate(
            [
                np.repeat(history_iv[:, None, -1:], sample_iv.shape[1], axis=1),
                sample_iv,
            ],
            axis=2,
        ),
        axis=2,
    )
    truth_delta = np.diff(
        np.concatenate([history_iv[:, -1:], future_iv], axis=1), axis=1
    )
    generated_flat = generated_delta.reshape(-1)
    truth_flat = truth_delta.reshape(-1)
    generated_jump = np.max(np.abs(generated_delta), axis=(2, 3, 4)).reshape(-1)
    truth_jump = np.max(np.abs(truth_delta), axis=(1, 2, 3)).reshape(-1)
    generated_acf = _lag_one(generated_delta.reshape(-1, generated_delta.shape[2], 25))
    truth_acf = _lag_one(truth_delta.reshape(-1, truth_delta.shape[1], 25))
    acf_corr = (
        float(np.corrcoef(generated_acf, truth_acf)[0, 1])
        if np.std(generated_acf) > 1e-12 and np.std(truth_acf) > 1e-12
        else 0.0
    )
    metrics = {
        "cov90": coverage["0.9"],
        "calibration_error": float(
            np.mean([abs(coverage[str(level)] - level) for level in levels])
        ),
        "level_ks_median": float(np.median(ks)),
        "median_abs_dev": float(np.median(np.abs(above - 0.5))),
        "width_rho": width_rho,
        "mr_ratio": _safe_ratio(
            float(np.mean(np.abs(generated_delta))),
            float(np.mean(np.abs(truth_delta))),
        ),
        "economic_ratio": _safe_ratio(
            float(np.std(sample_iv.mean(axis=(-1, -2)))),
            float(np.std(future_iv.mean(axis=(-1, -2)))),
        ),
        "acf_corr": acf_corr,
        "kurtosis_ratio": _safe_ratio(
            float(stats.kurtosis(generated_flat, fisher=False)),
            float(stats.kurtosis(truth_flat, fisher=False)),
        ),
        "jump_ks": float(stats.ks_2samp(generated_jump, truth_jump).statistic),
    }
    nested = {
        "coverage": {
            "overall": coverage,
            "calibration_error": metrics["calibration_error"],
        },
        "distributional_fidelity": {
            "ks_level_test": {"median_stat": metrics["level_ks_median"]},
            "median_bias": {"above_frac": above.tolist()},
        },
        "risk_state_allocation": {"history_width_spearman": width_rho},
    }
    return nested, metrics


def evaluate_samples(
    *,
    candidate: str,
    train_block: Any,
    validation_block: Any,
    sample_raw: np.ndarray,
    output_dir: Path,
    save_window_scores: bool,
) -> dict[str, Any]:
    raw_history = validation_block.history_state.astype(np.float32)
    raw_future = validation_block.future_state.astype(np.float32)
    truth_delta = panel_daily_changes(raw_history, raw_future)
    previous = np.concatenate(
        [
            np.repeat(raw_history[:, None, -1:], sample_raw.shape[1], axis=1),
            sample_raw[:, :, :-1],
        ],
        axis=2,
    )
    sample_delta = sample_raw - previous
    train_delta = panel_daily_changes(
        train_block.history_state, train_block.future_state
    ).reshape(-1, len(train_block.specs))
    factor_names = [spec.name for spec in validation_block.specs[25:]]
    current_iv, current_iv_metrics = iv_diagnostics(
        raw_history[..., :25].reshape(raw_history.shape[0], raw_history.shape[1], 5, 5),
        raw_future[..., :25].reshape(raw_future.shape[0], raw_future.shape[1], 5, 5),
        sample_raw[..., :25].reshape(
            sample_raw.shape[0], sample_raw.shape[1], sample_raw.shape[2], 5, 5
        ),
    )
    result = {
        "baseline_name": candidate,
        "mode": "current_realvix_39_state_panel",
        "n_windows": int(raw_history.shape[0]),
        "n_samples": int(sample_raw.shape[1]),
        "state_specs": [asdict(spec) for spec in validation_block.specs],
        "current_panel_iv": current_iv,
        "current_panel_iv_metrics": current_iv_metrics,
        "joint_panel_quality": summarize_joint_quality(
            raw_history, raw_future, sample_raw, factor_names, iv_count=25
        ),
        "panel_b_crps": compute_crps(sample_delta, truth_delta, iv_count=25),
        "panel_b_energy_score": compute_energy_score(
            sample_delta, truth_delta, train_delta
        ),
        "panel_b_variogram_score": compute_variogram_score(
            sample_delta, truth_delta, train_delta
        ),
        "panel_b_correlation": compute_correlation_score(
            sample_delta, truth_delta, iv_count=25
        ),
        "panel_b_factor_ks": compute_factor_ks(
            sample_delta, truth_delta, factor_names, iv_count=25
        ),
    }
    candidate_dir = output_dir / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    maybe_write_window_score_artifact(
        enabled=save_window_scores,
        output_path=candidate_dir / "window_scores.npz",
        samples_delta=sample_delta,
        ground_truth_delta=truth_delta,
        train_delta=train_delta,
        conditional_iv_samples=sample_raw[..., :25].reshape(
            sample_raw.shape[0], sample_raw.shape[1], sample_raw.shape[2], 5, 5
        ),
        ground_truth_iv=raw_future[..., :25].reshape(
            raw_future.shape[0], raw_future.shape[1], 5, 5
        ),
        history_iv=raw_history[..., :25].reshape(
            raw_history.shape[0], raw_history.shape[1], 5, 5
        ),
        raw_history=raw_history,
        sample_raw=sample_raw,
        window_indices=validation_block.indices,
        variable_names=[spec.name for spec in validation_block.specs],
        existing_result=result,
        iv_count=25,
    )
    (candidate_dir / "results_current_panel.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def sample_and_evaluate(
    *,
    candidate: str,
    model: Any,
    train_block: Any,
    validation_block: Any,
    samples: int,
    batch_size: int,
    output_dir: Path,
    save_window_scores: bool,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(int(seed))
    chunks = []
    for start in range(0, validation_block.history_increment.shape[0], int(batch_size)):
        end = min(start + int(batch_size), validation_block.history_increment.shape[0])
        kwargs: dict[str, Any] = {"rng": generator}
        if candidate == "historical_sim":
            kwargs["history_state"] = validation_block.history_state[start:end]
        chunks.append(
            model.sample_joint(
                validation_block.history_increment[start:end],
                n_samples=int(samples),
                **kwargs,
            )
        )
    sample_increment = np.concatenate(chunks)
    sample_raw = reconstruct_state_from_increments(
        validation_block.history_state[:, -1],
        sample_increment,
        validation_block.specs,
    ).astype(np.float32)
    return evaluate_samples(
        candidate=candidate,
        train_block=train_block,
        validation_block=validation_block,
        sample_raw=sample_raw,
        output_dir=output_dir,
        save_window_scores=save_window_scores,
    )
