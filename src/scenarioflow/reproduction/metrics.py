"""Frozen current-panel numerical metrics and practitioner screens."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats


def panel_daily_changes(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :], future[:, :-1, :]], axis=1)
    return future - prev


def compute_crps(
    samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int
) -> dict[str, Any]:
    horizons = [1, 7, 14, 30]
    n, k, t_len, d = samples_delta.shape
    per_h: dict[str, list[float]] = {}
    for h in horizons:
        if h > t_len:
            continue
        t = h - 1
        samples = samples_delta[:, :, t, :]
        truth = gt_delta[:, t, :]
        mae = np.abs(samples - truth[:, None, :]).mean(axis=1)
        spread = np.zeros((n, d), dtype=np.float64)
        for i in range(k):
            spread += np.abs(samples[:, i : i + 1, :] - samples).sum(axis=1)
        spread /= float(k * k)
        per_h[str(h)] = (mae - 0.5 * spread).mean(axis=0).tolist()
    per_dimension = np.asarray(list(per_h.values()), dtype=np.float64).mean(axis=0)
    return {
        "per_horizon": per_h,
        "per_dimension": per_dimension.tolist(),
        "overall": float(per_dimension.mean()),
        "iv_crps": float(per_dimension[:iv_count].mean()),
        "factor_crps": float(per_dimension[iv_count:].mean()),
    }


def compute_energy_score(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    train_delta: np.ndarray,
) -> dict[str, Any]:
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    samples_z = (samples_delta - mean[None, None, None, :]) / std[None, None, None, :]
    truth_z = (gt_delta - mean[None, None, :]) / std[None, None, :]
    n, k, t_len, _d = samples_delta.shape
    per_h: dict[str, float] = {}
    for h in [1, 7, 14, 30]:
        if h > t_len:
            continue
        t = h - 1
        samples = samples_z[:, :, t, :]
        truth = truth_z[:, t, :]
        term1 = np.linalg.norm(samples - truth[:, None, :], axis=2).mean(axis=1)
        spread = np.zeros(n, dtype=np.float64)
        for i in range(k):
            spread += np.linalg.norm(samples[:, i : i + 1, :] - samples, axis=2).sum(
                axis=1
            )
        spread /= float(k * k)
        per_h[str(h)] = float((term1 - 0.5 * spread).mean())
    return {"per_horizon": per_h, "overall": float(np.mean(list(per_h.values())))}


def compute_variogram_score(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    train_delta: np.ndarray,
    *,
    p: float = 0.5,
) -> dict[str, Any]:
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    t = min(samples_delta.shape[2], 30) - 1
    samples = (samples_delta[:, :, t, :] - mean[None, None, :]) / std[None, None, :]
    truth = (gt_delta[:, t, :] - mean[None, :]) / std[None, :]
    sample_variogram = np.mean(
        np.abs(samples[:, :, :, None] - samples[:, :, None, :]) ** p, axis=1
    )
    truth_variogram = np.abs(truth[:, :, None] - truth[:, None, :]) ** p
    upper = np.triu(np.ones((samples.shape[-1], samples.shape[-1]), dtype=bool), k=1)
    return {
        "variogram_score": float(
            ((sample_variogram - truth_variogram) ** 2)[:, upper].mean()
        ),
        "p": float(p),
        "horizon": int(t + 1),
    }


def compute_correlation_score(
    samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int
) -> dict[str, Any]:
    truth = gt_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    generated = samples_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    truth_corr = np.nan_to_num(np.corrcoef(truth.T), nan=0.0)
    generated_corr = np.nan_to_num(np.corrcoef(generated.T), nan=0.0)
    frobenius = float(np.linalg.norm(generated_corr - truth_corr, ord="fro"))
    ratio = frobenius / max(float(np.linalg.norm(truth_corr, ord="fro")), 1e-12)
    return {
        "frobenius_error": frobenius,
        "frobenius_ratio": ratio,
        "corr_score": float(1.0 - ratio),
    }


def compute_factor_ks(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    statistics = []
    for j, name in enumerate(factor_names):
        generated = samples_delta[..., iv_count + j].reshape(-1)
        truth = gt_delta[..., iv_count + j].reshape(-1)
        statistic, p_value = sp_stats.ks_2samp(generated, truth)
        statistics.append(float(statistic))
        rows[name] = {"statistic": float(statistic), "p_value": float(p_value)}
    return {
        "per_factor": rows,
        "mean_statistic": float(np.mean(statistics)) if statistics else None,
        "median_statistic": float(np.median(statistics)) if statistics else None,
        "worst_statistic": float(np.max(statistics)) if statistics else None,
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, values, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _ordinal_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def conditional_panel_diagnostics(
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    samples_raw: np.ndarray,
) -> dict[str, float]:
    sample_median = np.median(samples_raw, axis=1)
    rolled_median = np.roll(sample_median, shift=1, axis=0)
    conditional_mae = float(np.mean(np.abs(sample_median - future_raw)))
    rolled_mae = float(np.mean(np.abs(rolled_median - future_raw)))
    mae_reduction = (
        (rolled_mae - conditional_mae) / rolled_mae * 100.0
        if rolled_mae > 1e-12
        else 0.0
    )
    history_delta = np.diff(history_raw, axis=1)
    history_activity = np.mean(history_delta * history_delta, axis=(1, 2))
    low = np.quantile(samples_raw, 0.05, axis=1)
    high = np.quantile(samples_raw, 0.95, axis=1)
    sample_width = np.mean(high - low, axis=(1, 2))
    future_activity = np.mean(
        panel_daily_changes(history_raw, future_raw) ** 2, axis=(1, 2)
    )
    return {
        "conditional_median_mae": conditional_mae,
        "rolled_median_mae": rolled_mae,
        "median_mae_reduction_vs_rolled_pct": float(mae_reduction),
        "history_activity_width_spearman": _ordinal_spearman(
            history_activity, sample_width
        ),
        "future_activity_width_spearman": _ordinal_spearman(
            future_activity, sample_width
        ),
    }


def summarize_joint_quality(
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    samples_raw: np.ndarray,
    factor_names: list[str],
    *,
    iv_count: int,
) -> dict[str, Any]:
    gt_delta = panel_daily_changes(history_raw, future_raw)
    sample_previous = np.concatenate(
        [
            np.repeat(history_raw[:, None, -1:, :], samples_raw.shape[1], axis=1),
            samples_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    generated_delta = samples_raw - sample_previous
    gt_factor_delta = gt_delta[..., iv_count:]
    generated_factor_delta = generated_delta[..., iv_count:]
    marginal_ks = []
    tail_ratios = []
    for idx, _name in enumerate(factor_names):
        observed = gt_factor_delta[..., idx]
        generated = generated_factor_delta[..., idx]
        marginal_ks.append(ks_statistic(observed, generated))
        observed_q99 = float(np.quantile(np.abs(observed).reshape(-1), 0.99))
        generated_q99 = float(np.quantile(np.abs(generated).reshape(-1), 0.99))
        tail_ratios.append(generated_q99 / max(observed_q99, 1e-12))
    return {
        "finite_rate": float(np.isfinite(samples_raw).mean()),
        "n_windows": int(samples_raw.shape[0]),
        "n_samples": int(samples_raw.shape[1]),
        "future_len": int(samples_raw.shape[2]),
        "n_factors": int(len(factor_names)),
        "factor_delta_ks_mean": float(np.nanmean(marginal_ks)),
        "factor_delta_ks_pass_020": int(np.sum(np.asarray(marginal_ks) < 0.20)),
        "factor_tail_q99_ratio_median": float(np.nanmedian(tail_ratios)),
        "factor_tail_q99_pass_05_20": int(
            np.sum((np.asarray(tail_ratios) >= 0.5) & (np.asarray(tail_ratios) <= 2.0))
        ),
        "conditional_panel": conditional_panel_diagnostics(
            history_raw, future_raw, samples_raw
        ),
    }


def evaluate_deck(
    *,
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    sample_raw: np.ndarray,
    train_block: Any,
    factor_names: list[str],
    iv_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen eight metrics and four screen decisions."""
    ground_truth_delta = panel_daily_changes(raw_history, raw_future).astype(np.float32)
    sample_previous = np.concatenate(
        [
            np.repeat(raw_history[:, None, -1:, :], sample_raw.shape[1], axis=1),
            sample_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    sample_delta = sample_raw - sample_previous
    train_delta = panel_daily_changes(
        train_block.history_state, train_block.future_state
    ).reshape(-1, raw_history.shape[-1])
    crps = compute_crps(sample_delta, ground_truth_delta, iv_count=iv_count)
    energy = compute_energy_score(sample_delta, ground_truth_delta, train_delta)
    variogram = compute_variogram_score(sample_delta, ground_truth_delta, train_delta)
    correlation = compute_correlation_score(
        sample_delta, ground_truth_delta, iv_count=iv_count
    )
    factor_ks = compute_factor_ks(
        sample_delta, ground_truth_delta, factor_names, iv_count=iv_count
    )
    quality = summarize_joint_quality(
        raw_history, raw_future, sample_raw, factor_names, iv_count=iv_count
    )
    metrics = np.asarray(
        [
            crps["overall"],
            crps["iv_crps"],
            crps["factor_crps"],
            energy["overall"],
            variogram["variogram_score"],
            correlation["corr_score"],
            factor_ks["mean_statistic"],
            quality["factor_tail_q99_ratio_median"],
        ],
        dtype=np.float32,
    )
    conditional = quality["conditional_panel"]
    screens = np.asarray(
        [
            quality["finite_rate"] == 1.0,
            quality["factor_delta_ks_pass_020"] == quality["n_factors"],
            quality["factor_tail_q99_pass_05_20"] == quality["n_factors"],
            conditional["conditional_median_mae"] <= conditional["rolled_median_mae"],
        ],
        dtype=np.int8,
    )
    return metrics, screens
