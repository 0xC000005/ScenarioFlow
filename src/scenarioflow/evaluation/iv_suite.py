"""Portable frozen current-panel IV diagnostic suite.

This is the evaluator-only extraction of the 14-key suite used by the
published current-panel baseline runner.  It consumes an already frozen deck;
it has no model, worktree, or artifact-discovery dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) < 1e-12 or np.std(y[keep]) < 1e-12:
        return 0.0
    value = float(np.corrcoef(x[keep], y[keep])[0, 1])
    return value if np.isfinite(value) else 0.0


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) < 1e-12 or np.ptp(y[keep]) < 1e-12:
        return 0.0
    value = float(sp_stats.spearmanr(x[keep], y[keep]).statistic)
    return value if np.isfinite(value) else 0.0


def _ratio(numerator: float, denominator: float, *, zero_value: float = 1.0) -> float:
    if abs(denominator) <= 1e-12:
        return zero_value
    return float(numerator / denominator)


def _slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean, y_mean = float(x.mean()), float(y.mean())
    variance = float(np.mean((x - x_mean) ** 2))
    slope = (
        float(np.mean((x - x_mean) * (y - y_mean)) / variance)
        if variance > 1e-12
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    residual = y - (intercept + slope * x)
    total = float(np.mean((y - y_mean) ** 2))
    r2 = 1.0 - float(np.mean(residual**2)) / total if total > 1e-12 else 0.0
    return slope, intercept, r2


def _ewma_volatility(
    returns: np.ndarray, warmup: np.ndarray, *, decay: float = 0.94
) -> np.ndarray:
    if warmup.size:
        variance = float(warmup[0] ** 2)
        for value in warmup[1:]:
            variance = decay * variance + (1.0 - decay) * float(value**2)
    elif returns.size:
        variance = float(returns[0] ** 2)
    else:
        variance = 0.0
    path = []
    for value in returns:
        variance = decay * variance + (1.0 - decay) * float(value**2)
        path.append(np.sqrt(variance * 252.0))
    return np.asarray(path, dtype=np.float64)


def _aligned_ewma(
    returns: np.ndarray,
    window_indices: np.ndarray,
    *,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, valid = [], []
    values = np.asarray(returns, dtype=np.float64)
    for position, start in enumerate(np.asarray(window_indices, dtype=np.int64)):
        future_start = int(start) + history_len
        if future_start + future_len > values.size:
            continue
        rows.append(
            _ewma_volatility(
                values[future_start : future_start + future_len],
                values[int(start) : future_start],
            )
        )
        valid.append(position)
    return np.asarray(rows, dtype=np.float64), np.asarray(valid, dtype=np.int64)


def _cointegration(
    samples: np.ndarray,
    truth: np.ndarray,
    returns: np.ndarray,
    window_indices: np.ndarray,
) -> dict[str, Any]:
    ewma, valid = _aligned_ewma(
        returns, window_indices, history_len=30, future_len=truth.shape[1]
    )
    h, w = truth.shape[-2:]
    if valid.size == 0:
        return {"overall_pass": False, "n_valid_windows": 0, "gen_gt_ratio_legacy": 0.0}
    generated = np.median(samples, axis=1)
    counts = {
        name: np.zeros((h, w), dtype=np.float64)
        for name in ("gen", "gt", "gen_legacy", "gt_legacy")
    }
    gen_r2 = np.zeros((h, w), dtype=np.float64)
    gt_r2 = np.zeros((h, w), dtype=np.float64)

    def one(iv: np.ndarray, vol: np.ndarray) -> tuple[bool, bool, float]:
        if np.std(iv) < 1e-8 or np.std(vol) < 1e-8:
            return False, False, 0.0
        try:
            _statistic, p_value, _critical = coint(
                iv, vol, trend="c", maxlag=3, autolag=None
            )
            fit = OLS(iv, add_constant(vol)).fit()
            legacy_p = float(adfuller(fit.resid, maxlag=3, regression="c")[1])
            return bool(p_value < 0.10), bool(legacy_p < 0.10), float(fit.rsquared)
        except (ValueError, np.linalg.LinAlgError):
            return False, False, 0.0

    for row, position in enumerate(valid):
        for i in range(h):
            for j in range(w):
                for prefix, path, accumulator in (
                    ("gen", generated[position, :, i, j], gen_r2),
                    ("gt", truth[position, :, i, j], gt_r2),
                ):
                    proper, legacy, r2 = one(path, ewma[row])
                    counts[prefix][i, j] += proper
                    counts[prefix + "_legacy"][i, j] += legacy
                    accumulator[i, j] += r2
    rates = {name: value / valid.size for name, value in counts.items()}
    gen_rate, gt_rate = float(rates["gen"].mean()), float(rates["gt"].mean())
    gen_legacy = float(rates["gen_legacy"].mean())
    gt_legacy = float(rates["gt_legacy"].mean())
    ratio_grid = np.ones_like(rates["gt"])
    np.divide(rates["gen"], rates["gt"], out=ratio_grid, where=rates["gt"] > 0)
    ratio_grid[(rates["gt"] <= 0) & (rates["gen"] > 0)] = np.inf
    worst = np.unravel_index(np.argmin(ratio_grid), ratio_grid.shape)
    ratio = _ratio(gen_rate, gt_rate, zero_value=0.0)
    legacy_ratio = _ratio(gen_legacy, gt_legacy, zero_value=0.0)
    return {
        "gen_pass_rate": gen_rate,
        "gt_pass_rate": gt_rate,
        "gen_gt_ratio": ratio,
        "gen_pass_rate_legacy": gen_legacy,
        "gt_pass_rate_legacy": gt_legacy,
        "gen_gt_ratio_legacy": legacy_ratio,
        "gen_mean_rsq": float((gen_r2 / valid.size).mean()),
        "gt_mean_rsq": float((gt_r2 / valid.size).mean()),
        "gen_pass_rates_grid": rates["gen"].tolist(),
        "gt_pass_rates_grid": rates["gt"].tolist(),
        "gen_pass_rates_grid_legacy": rates["gen_legacy"].tolist(),
        "gt_pass_rates_grid_legacy": rates["gt_legacy"].tolist(),
        "per_cell_ratio_grid": ratio_grid.tolist(),
        "worst_cell_ratio": float(ratio_grid[worst]),
        "worst_cell_idx": list(worst),
        "worst_cell_pass": bool(ratio_grid[worst] >= 0.25),
        "n_valid_windows": int(valid.size),
        "adf_alpha": 0.10,
        "adf_lags": 3,
        "overall_pass": bool(ratio >= 0.5 and ratio_grid[worst] >= 0.25),
    }


def _economic_link(
    samples: np.ndarray,
    truth: np.ndarray,
    returns: np.ndarray,
    window_indices: np.ndarray,
) -> dict[str, Any]:
    ewma, valid = _aligned_ewma(
        returns, window_indices, history_len=30, future_len=truth.shape[1]
    )
    if valid.size == 0:
        return {
            "overall_pass": False,
            "n_valid_windows": 0,
            "methodology": "pooled_iv_ewma_relationship_strength_not_engle_granger",
        }
    generated = np.median(samples, axis=1)[valid]
    observed = truth[valid]
    grid = np.broadcast_to(ewma[:, :, None, None], observed.shape)
    gt_slope, gt_intercept, gt_r2 = _slope_intercept_r2(grid, observed)
    gen_slope, gen_intercept, gen_r2 = _slope_intercept_r2(grid, generated)
    gt_rho, gen_rho = _safe_spearman(grid, observed), _safe_spearman(grid, generated)
    slope_ratio = _ratio(gen_slope, gt_slope, zero_value=float("nan"))
    r2_ratio = _ratio(gen_r2, gt_r2, zero_value=float("nan"))
    rho_ratio = _ratio(gen_rho, gt_rho, zero_value=float("nan"))
    cell_ratios = np.full((5, 5), np.nan)
    cell_pass = np.zeros((5, 5), dtype=bool)
    for i in range(5):
        for j in range(5):
            gt_cell, _, _ = _slope_intercept_r2(ewma, observed[:, :, i, j])
            gen_cell, _, _ = _slope_intercept_r2(ewma, generated[:, :, i, j])
            if abs(gt_cell) > 1e-12:
                cell_ratios[i, j] = gen_cell / gt_cell
                cell_pass[i, j] = gt_cell <= 0 or (
                    gen_cell > 0 and 0.25 <= cell_ratios[i, j] <= 4.0
                )
    valid_cells = np.isfinite(cell_ratios)
    pass_count = int((cell_pass & valid_cells).sum())
    valid_count = int(valid_cells.sum())
    pass_rate = pass_count / valid_count if valid_count else 1.0
    checks = (
        gt_slope <= 0 or gen_slope > 0,
        np.isfinite(slope_ratio) and 0.5 <= slope_ratio <= 2.0,
        gt_r2 < 0.02 or (np.isfinite(r2_ratio) and 0.35 <= r2_ratio <= 3.0),
        gt_rho < 0.10
        or (gen_rho >= 0.05 and np.isfinite(rho_ratio) and rho_ratio >= 0.5),
        pass_rate >= 0.8,
    )
    return {
        "methodology": "pooled_iv_ewma_relationship_strength_not_engle_granger",
        "n_valid_windows": int(valid.size),
        "gt_slope": gt_slope,
        "gen_slope": gen_slope,
        "slope_ratio": slope_ratio,
        "slope_sign_pass": bool(checks[0]),
        "slope_ratio_pass": bool(checks[1]),
        "gt_intercept": gt_intercept,
        "gen_intercept": gen_intercept,
        "gt_r2": gt_r2,
        "gen_r2": gen_r2,
        "r2_ratio": r2_ratio,
        "r2_ratio_pass": bool(checks[2]),
        "gt_spearman": gt_rho,
        "gen_spearman": gen_rho,
        "spearman_ratio": rho_ratio,
        "spearman_pass": bool(checks[3]),
        "per_cell_slope_ratio": cell_ratios.tolist(),
        "per_cell_slope_pass_count": pass_count,
        "per_cell_slope_valid_count": valid_count,
        "per_cell_slope_pass_rate": float(pass_rate),
        "per_cell_slope_pass": bool(checks[4]),
        "overall_pass": bool(all(checks)),
    }


def run_full_iv_suite(
    conditional_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    *,
    shuffled_samples: np.ndarray,
    conditionality_batch_size: int = 32,
    conditionality_samples: int = 32,
    returns: np.ndarray,
    window_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    samples = np.asarray(conditional_samples, dtype=np.float64).reshape(
        conditional_samples.shape[0],
        conditional_samples.shape[1],
        conditional_samples.shape[2],
        5,
        5,
    )
    shuffled_input = np.asarray(shuffled_samples, dtype=np.float64)
    shuffled = shuffled_input.reshape(
        shuffled_input.shape[0],
        shuffled_input.shape[1],
        shuffled_input.shape[2],
        5,
        5,
    )
    truth = np.asarray(ground_truth, dtype=np.float64).reshape(
        ground_truth.shape[0], ground_truth.shape[1], 5, 5
    )
    history_arr = np.asarray(history, dtype=np.float64).reshape(
        history.shape[0], history.shape[1], 5, 5
    )
    levels = (0.5, 0.8, 0.9, 0.95)
    overall: dict[float, float] = {}
    per_horizon: dict[int, dict[float, float]] = {}
    per_cell_coverage: dict[int, np.ndarray] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower, upper = np.quantile(samples, alpha, axis=1), np.quantile(
            samples, 1 - alpha, axis=1
        )
        overall[level] = float(((truth >= lower) & (truth <= upper)).mean())
    for horizon in (1, 7, 14, 30):
        if horizon <= truth.shape[1]:
            per_horizon[horizon] = {}
            for level in levels:
                alpha = (1.0 - level) / 2.0
                lo = np.quantile(samples[:, :, horizon - 1], alpha, axis=1)
                hi = np.quantile(samples[:, :, horizon - 1], 1 - alpha, axis=1)
                covered = (truth[:, horizon - 1] >= lo) & (truth[:, horizon - 1] <= hi)
                per_horizon[horizon][level] = float(
                    (
                        (truth[:, horizon - 1] >= lo) & (truth[:, horizon - 1] <= hi)
                    ).mean()
                )
                if level == 0.9:
                    per_cell_coverage[horizon] = covered.mean(axis=0)
    calibration_nominal = [round(float(value), 1) for value in np.linspace(0.1, 0.9, 9)]
    calibration_empirical = []
    for level in calibration_nominal:
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(samples, alpha, axis=1)
        upper = np.quantile(samples, 1.0 - alpha, axis=1)
        calibration_empirical.append(
            float(((truth >= lower) & (truth <= upper)).mean())
        )
    calibration_error = float(
        np.mean(np.abs(np.asarray(calibration_nominal) - calibration_empirical))
    )
    horizon_targets = {1: 0.80, 7: 0.75, 14: 0.70, 30: 0.65}
    horizon_pass = {
        horizon: bool(per_horizon[horizon][0.9] > horizon_targets[horizon])
        for horizon in per_horizon
    }
    worst = {
        horizon: float(value.min()) for horizon, value in per_cell_coverage.items()
    }
    best = {horizon: float(value.max()) for horizon, value in per_cell_coverage.items()}
    worst_cell_pass = all(
        worst[horizon] >= 0.70 and best[horizon] <= 0.95
        for horizon in per_cell_coverage
    )
    coverage = {
        "overall": overall,
        "per_horizon": per_horizon,
        "per_cell_coverage": {
            horizon: value.tolist() for horizon, value in per_cell_coverage.items()
        },
        "worst_cell_per_horizon": worst,
        "best_cell_per_horizon": best,
        "worst_cell_pass": bool(worst_cell_pass),
        "calibration": {
            "nominal": calibration_nominal,
            "empirical": calibration_empirical,
        },
        "calibration_error": calibration_error,
        "horizon_pass": horizon_pass,
        "overall_pass": bool(all(horizon_pass.values()) and worst_cell_pass),
    }

    all_samples = samples.reshape(-1, samples.shape[2], 5, 5)
    finite = float(np.isfinite(all_samples).mean())
    high_explosion = (all_samples > 1.0).any(axis=(1, 2, 3))
    low_explosion = (all_samples < 0.0).any(axis=(1, 2, 3))
    explosion_rate = float((high_explosion | low_explosion).mean())
    tenors = np.asarray([1, 3, 6, 12, 24], dtype=np.float64)

    def calendar_statistics(values: np.ndarray) -> tuple[float, float, list[float]]:
        violations, per_strike = [], [[] for _ in range(5)]
        for horizon in range(values.shape[1]):
            total_variance = values[:, horizon] ** 2 * tenors[:, None]
            for tenor in range(4):
                violation = (
                    total_variance[:, tenor] > total_variance[:, tenor + 1] * 1.001
                )
                violations.append(float(violation.mean()))
                for strike in range(5):
                    per_strike[strike].append(float(violation[:, strike].mean()))
        rates = [float(np.mean(row)) for row in per_strike]
        return float(np.mean(violations)), float(np.max(violations)), rates

    calendar_average, calendar_maximum, calendar_per_strike = calendar_statistics(
        all_samples
    )
    gt_calendar_average, _gt_calendar_maximum, gt_calendar_per_strike = (
        calendar_statistics(truth)
    )
    butterfly_by_horizon = []
    butterfly_per_tenor = [[] for _ in range(5)]
    for horizon in range(all_samples.shape[1]):
        curvature = (
            all_samples[:, horizon, :, :-2]
            - 2 * all_samples[:, horizon, :, 1:-1]
            + all_samples[:, horizon, :, 2:]
        )
        violation = curvature < -0.005
        butterfly_by_horizon.append(float(violation.mean()))
        for tenor in range(5):
            butterfly_per_tenor[tenor].append(float(violation[:, tenor].mean()))
    butterfly_rates = [float(np.mean(row)) for row in butterfly_per_tenor]
    butterfly_average = float(np.mean(butterfly_by_horizon))
    calendar_worst = max(calendar_per_strike)
    calendar_gate = max(gt_calendar_per_strike) + 0.10
    butterfly_worst = max(butterfly_rates)
    surface = {
        "explosion": {
            "explosion_high_rate": float(high_explosion.mean()),
            "explosion_low_rate": float(low_explosion.mean()),
            "explosion_total_rate": explosion_rate,
            "max_iv_observed": float(np.nanmax(all_samples)),
            "min_iv_observed": float(np.nanmin(all_samples)),
            "pass": bool(finite == 1.0 and explosion_rate < 0.05),
        },
        "calendar": {
            "calendar_avg_violation_rate": calendar_average,
            "calendar_max_violation_rate": calendar_maximum,
            "per_strike_rates": calendar_per_strike,
            "worst_strike_rate": calendar_worst,
            "worst_strike_pass": bool(calendar_worst < calendar_gate),
            "gt_worst_strike_rate": max(gt_calendar_per_strike),
            "gt_avg_violation_rate": gt_calendar_average,
            "worst_strike_gate": calendar_gate,
            "pass": bool(calendar_average < 0.15 and calendar_worst < calendar_gate),
        },
        "butterfly": {
            "butterfly_avg_violation_rate": butterfly_average,
            "butterfly_max_violation_rate": float(np.max(butterfly_by_horizon)),
            "per_tenor_rates": butterfly_rates,
            "worst_tenor_rate": butterfly_worst,
            "worst_tenor_pass": bool(butterfly_worst < 0.50),
            "pass": bool(butterfly_average < 0.40 and butterfly_worst < 0.50),
        },
    }
    surface["overall_pass"] = bool(all(value["pass"] for value in surface.values()))

    median = np.median(samples, axis=1)
    cond_limit = min(samples.shape[0], int(conditionality_batch_size) * 8)
    sample_limit = min(samples.shape[1], int(conditionality_samples))
    cond_samples = samples[:cond_limit, :sample_limit]
    cond_truth = truth[:cond_limit]
    cond_history = history_arr[:cond_limit]
    cond_median = median[:cond_limit]
    shuffled_truth = truth[: shuffled.shape[0]]
    shuffled = shuffled[:, :sample_limit]
    rolled = np.median(shuffled, axis=1)
    conditional_mae = float(np.abs(cond_median - cond_truth).mean())
    rolled_mae = float(np.abs(rolled - shuffled_truth).mean())
    conditional_width = np.quantile(cond_samples, 0.95, axis=1) - np.quantile(
        cond_samples, 0.05, axis=1
    )
    rolled_width = np.quantile(shuffled, 0.95, axis=1) - np.quantile(
        shuffled, 0.05, axis=1
    )
    width_by_horizon = conditional_width.mean(axis=(0, 2, 3))
    avg_cond_width = float(conditional_width.mean())
    avg_uncond_width = float(rolled_width.mean())
    width_ratio = _ratio(avg_cond_width, avg_uncond_width)
    cond_cell_width = conditional_width.mean(axis=(0, 1))
    uncond_cell_width = rolled_width.mean(axis=(0, 1))
    cell_width_ratio = np.divide(
        cond_cell_width,
        np.maximum(uncond_cell_width, 1e-8),
    )
    cond_cell_mae = np.abs(cond_median - cond_truth).mean(axis=(0, 1))
    uncond_cell_mae = np.abs(rolled - shuffled_truth).mean(axis=(0, 1))
    cell_mae_reduction = np.zeros_like(uncond_cell_mae)
    np.divide(
        (uncond_cell_mae - cond_cell_mae) * 100.0,
        uncond_cell_mae,
        out=cell_mae_reduction,
        where=uncond_cell_mae > 1e-8,
    )
    per_horizon_conditionality = {
        str(horizon): {
            "width_ratio": _ratio(
                float(conditional_width[:, horizon - 1].mean()),
                float(rolled_width[:, horizon - 1].mean()),
            ),
            "mae_reduction_pct": float(
                (
                    np.abs(
                        rolled[:, horizon - 1] - shuffled_truth[:, horizon - 1]
                    ).mean()
                    - np.abs(
                        cond_median[:, horizon - 1] - cond_truth[:, horizon - 1]
                    ).mean()
                )
                / max(
                    float(
                        np.abs(
                            rolled[:, horizon - 1] - shuffled_truth[:, horizon - 1]
                        ).mean()
                    ),
                    1e-12,
                )
                * 100.0
            ),
            "cond_width": float(conditional_width[:, horizon - 1].mean()),
            "uncond_width": float(rolled_width[:, horizon - 1].mean()),
        }
        for horizon in (1, 7, 14, 30)
        if horizon <= truth.shape[1]
    }
    vov = np.mean(np.diff(cond_history, axis=1) ** 2, axis=(1, 2, 3))
    low, high = np.quantile(vov, [0.2, 0.8])
    regimes = {}
    for name, mask in (("calm", vov <= low), ("turb", vov >= high)):
        if not mask.any():
            continue
        regime_width = conditional_width[mask].mean(axis=(0, 1))
        regime_mae = np.abs(cond_median[mask] - cond_truth[mask]).mean(axis=(0, 1))
        regimes[name] = {
            "avg_width_ratio": float(
                np.mean(regime_width / np.maximum(uncond_cell_width, 1e-8))
            ),
            "worst_cell_width_ratio": float(
                np.max(regime_width / np.maximum(uncond_cell_width, 1e-8))
            ),
            "avg_mae_reduction_pct": float(
                np.mean(
                    (uncond_cell_mae - regime_mae)
                    / np.maximum(uncond_cell_mae, 1e-8)
                    * 100.0
                )
            ),
            "worst_cell_mae_reduction_pct": float(
                np.min(
                    (uncond_cell_mae - regime_mae)
                    / np.maximum(uncond_cell_mae, 1e-8)
                    * 100.0
                )
            ),
            "per_cell_width_ratio": (
                regime_width / np.maximum(uncond_cell_width, 1e-8)
            ).tolist(),
            "per_cell_mae_reduction": (
                (uncond_cell_mae - regime_mae)
                / np.maximum(uncond_cell_mae, 1e-8)
                * 100.0
            ).tolist(),
            "n_windows": int(mask.sum()),
            "avg_width": float(regime_width.mean()),
        }
    turb_calm_ratio = _ratio(
        float(conditional_width[vov >= high].mean()),
        float(conditional_width[vov <= low].mean()),
    )
    mae_reduction = (rolled_mae - conditional_mae) / max(rolled_mae, 1e-12) * 100.0
    conditionality = {
        "width_ratio": width_ratio,
        "avg_cond_width": avg_cond_width,
        "avg_uncond_width": avg_uncond_width,
        "width_pass": bool(width_ratio < 0.95),
        "turb_calm_ratio": turb_calm_ratio,
        "turb_calm_pass": bool(turb_calm_ratio > 1.15),
        "turb_calm_policy_target": 1.15,
        "turb_calm_informational": True,
        "mae_reduction_pct": float(mae_reduction),
        "avg_cond_mae": conditional_mae,
        "avg_uncond_mae": rolled_mae,
        "mae_pass": bool(mae_reduction > 5.0),
        "per_horizon_var": {
            str(horizon): float(cond_samples[:, :, horizon - 1].var(axis=1).mean())
            for horizon in (1, 10, 20, 30)
            if horizon <= samples.shape[2]
        },
        "per_horizon_conditionality": per_horizon_conditionality,
        "growing_uncertainty_monotonic": bool(
            np.mean(np.diff(width_by_horizon) >= 0) >= 0.5
        ),
        "per_cell_width_ratio": cell_width_ratio.tolist(),
        "worst_cell_width_ratio": float(cell_width_ratio.max()),
        "worst_cell_wr_pass": bool(cell_width_ratio.max() < 1.20),
        "per_cell_mae_reduction": cell_mae_reduction.tolist(),
        "worst_cell_mae_reduction": float(cell_mae_reduction.min()),
        "worst_cell_mae_pass": bool(cell_mae_reduction.min() > -10.0),
        "per_regime_conditionality": regimes,
    }
    conditionality["overall_pass"] = bool(
        conditionality["mae_pass"]
        and conditionality["worst_cell_mae_pass"]
        and conditionality["worst_cell_wr_pass"]
    )

    anchor = history_arr[:, -1]
    truth_change = np.diff(np.concatenate([anchor[:, None], truth], axis=1), axis=1)
    sample_anchor = np.repeat(anchor[:, None, None], samples.shape[1], axis=1)
    sample_change = np.diff(np.concatenate([sample_anchor, samples], axis=2), axis=2)
    gt_flat, gen_flat = truth_change.reshape(-1, 25), sample_change.reshape(-1, 25)
    gt_diff = np.diff(truth, axis=1)
    gen_diff = np.diff(samples, axis=2)
    gt_changes = gt_diff.reshape(-1)
    generated_changes = [
        gen_diff[:, index].reshape(-1) for index in range(min(10, samples.shape[1]))
    ]
    gt_kurtosis = float(sp_stats.kurtosis(gt_changes, fisher=True))
    gen_kurtosis = float(
        np.median(
            [sp_stats.kurtosis(value, fisher=True) for value in generated_changes]
        )
    )
    kurtosis_ratio = _ratio(gen_kurtosis, gt_kurtosis)
    gt_skewness = float(sp_stats.skew(gt_changes))
    gen_skewness = float(
        np.median([sp_stats.skew(value) for value in generated_changes])
    )
    skewness_ratio = _ratio(gen_skewness, gt_skewness)

    def acf_curve(values: np.ndarray, max_lag: int) -> np.ndarray:
        centered = values - values.mean()
        variance = float(np.dot(centered, centered))
        if variance <= 1e-12:
            return np.zeros(max_lag + 1)
        return np.asarray(
            [
                float(
                    np.dot(centered[: len(centered) - lag], centered[lag:]) / variance
                )
                for lag in range(max_lag + 1)
            ]
        )

    max_lag = min(20, truth.shape[1] - 2)
    gt_curves = [
        acf_curve(row, max_lag)
        for row in np.diff(truth[:, :, 2, 2], axis=1)
        if np.std(row) > 1e-10
    ]
    gen_curves = [
        acf_curve(row, max_lag)
        for sample_index in range(min(5, samples.shape[1]))
        for row in np.diff(samples[:, sample_index, :, 2, 2], axis=1)
        if np.std(row) > 1e-10
    ]
    gt_acf = np.mean(gt_curves, axis=0) if gt_curves else np.zeros(max_lag + 1)
    gen_acf = np.mean(gen_curves, axis=0) if gen_curves else np.zeros(max_lag + 1)
    acf_corr = _safe_corr(gt_acf, gen_acf)
    acf_mae = float(np.mean(np.abs(gt_acf - gen_acf)))
    per_cell_kurtosis = np.zeros((5, 5))
    per_cell_skewness = np.zeros((5, 5))
    per_cell_tail = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            observed = gt_diff[..., i, j].reshape(-1)
            generated = gen_diff[:, 0, ..., i, j].reshape(-1)
            per_cell_kurtosis[i, j] = _ratio(
                float(sp_stats.kurtosis(generated, fisher=True)),
                float(sp_stats.kurtosis(observed, fisher=True)),
            )
            per_cell_skewness[i, j] = _ratio(
                float(sp_stats.skew(generated)), float(sp_stats.skew(observed))
            )
            observed_q99 = float(np.quantile(np.abs(observed), 0.99))
            generated_q99 = float(
                np.median(
                    [
                        np.quantile(np.abs(gen_diff[:, index, ..., i, j]), 0.99)
                        for index in range(min(10, samples.shape[1]))
                    ]
                )
            )
            if observed_q99 > 1e-12:
                per_cell_tail[i, j] = generated_q99 / observed_q99
    tail_pass_grid = (
        np.isfinite(per_cell_tail) & (per_cell_tail >= 0.5) & (per_cell_tail <= 2.0)
    )
    finite_tail = per_cell_tail[np.isfinite(per_cell_tail)]
    move_profile: dict[str, Any] = {"gate_lo": 0.9, "gate_hi": 1.1}
    for key, threshold in (
        ("very_small_moves", 0.005),
        ("small_moves", 0.010),
        ("moderate_moves", 0.020),
        ("large_moves", 0.050),
    ):
        gt_share = float((np.abs(gt_changes) <= threshold).mean())
        gen_share = float((np.abs(generated_changes[0]) <= threshold).mean())
        ratio = _ratio(gen_share, gt_share)
        move_profile[key] = {
            "threshold": threshold,
            "gt_share": gt_share,
            "gen_share": gen_share,
            "ratio": ratio,
            "pass": bool(0.9 <= ratio <= 1.1),
        }
    move_profile["pass"] = bool(
        all(
            value["pass"]
            for key, value in move_profile.items()
            if key.endswith("moves")
        )
    )
    time_series = {
        "acf": {
            "acf_correlation": acf_corr,
            "acf_mae": acf_mae,
            "gt_acf": gt_acf.tolist(),
            "gen_acf": gen_acf.tolist(),
            "pass": bool(acf_corr > 0.5),
        },
        "kurtosis": {
            "gt_kurtosis": gt_kurtosis,
            "gen_kurtosis": gen_kurtosis,
            "kurtosis_ratio": kurtosis_ratio,
            "gate_lo": 0.8,
            "gate_hi": 1.25,
            "gt_skewness": gt_skewness,
            "gen_skewness": gen_skewness,
            "skewness_ratio": skewness_ratio,
            "skewness_pass": bool(skewness_ratio >= 0.25),
            "per_cell_ratio": per_cell_kurtosis.tolist(),
            "worst_cell_ratio": float(np.nanmin(per_cell_kurtosis)),
            "best_cell_ratio": float(np.nanmax(per_cell_kurtosis)),
            "per_cell_skew_ratio": per_cell_skewness.tolist(),
            "worst_cell_skew_ratio": float(np.nanmin(per_cell_skewness)),
            "best_cell_skew_ratio": float(np.nanmax(per_cell_skewness)),
            "pass": bool(0.8 <= kurtosis_ratio <= 1.25),
        },
        "tail_scale": {
            "per_cell_q99_ratio": per_cell_tail.tolist(),
            "n_pass": int(tail_pass_grid.sum()),
            "pass": bool(tail_pass_grid.sum() >= 20),
            "worst_cell_ratio": (
                float(finite_tail.min()) if finite_tail.size else float("nan")
            ),
            "best_cell_ratio": (
                float(finite_tail.max()) if finite_tail.size else float("nan")
            ),
            "gate_lo": 0.5,
            "gate_hi": 2.0,
        },
        "move_size_profile": move_profile,
    }
    time_series["overall_pass"] = bool(
        time_series["acf"]["pass"]
        and time_series["kurtosis"]["pass"]
        and time_series["tail_scale"]["pass"]
        and move_profile["pass"]
    )

    trajectories = samples[:, 0]
    frame_jump = np.abs(np.diff(trajectories, axis=1)).mean(axis=(0, 2, 3))
    boundary_indices = [index for index in (9, 19) if index < frame_jump.size]
    interior_indices = [
        index for index in range(frame_jump.size) if index not in boundary_indices
    ]
    boundary_jump = (
        float(frame_jump[boundary_indices].mean()) if boundary_indices else 0.0
    )
    interior_jump = (
        float(frame_jump[interior_indices].mean()) if interior_indices else 0.0
    )
    boundary_ratio = _ratio(boundary_jump, interior_jump)
    horizon_variance = samples.var(axis=1).mean(axis=(0, 2, 3))
    key_horizons = [value for value in (1, 10, 20, 30) if value <= samples.shape[2]]
    key_variance = [float(horizon_variance[value - 1]) for value in key_horizons]
    monotonic = all(
        key_variance[index] < key_variance[index + 1]
        for index in range(len(key_variance) - 1)
    )
    block_ar = {
        "boundary_smoothness": {
            "avg_boundary_jump": boundary_jump,
            "avg_intra_jump": interior_jump,
            "boundary_ratio": boundary_ratio,
            "pass": bool(boundary_ratio < 2.0),
        },
        "growing_uncertainty": {
            "key_horizon_vars": dict(zip(key_horizons, key_variance, strict=True)),
            "monotonic": bool(monotonic),
            "pass": bool(monotonic),
        },
        "overall_pass": bool(boundary_ratio < 2.0),
    }
    cointegration = _cointegration(samples, truth, returns, window_indices)
    economic = _economic_link(samples, truth, returns, window_indices)

    history_activity = np.mean(np.diff(history_arr, axis=1) ** 2, axis=(1, 2, 3))
    future_activity = np.mean(truth_change**2, axis=(1, 2, 3))
    width = (
        np.quantile(samples, 0.95, axis=1) - np.quantile(samples, 0.05, axis=1)
    ).mean(axis=(1, 2, 3))
    generated_activity = np.mean(sample_change**2, axis=(1, 2, 3, 4))

    def bucket_summary(
        activity: np.ndarray,
    ) -> tuple[list[dict[str, Any]], float, float]:
        rows = []
        valid = np.where(np.isfinite(activity))[0]
        if valid.size < 16:
            return rows, 0.0, 1.0
        bucket_count = min(5, max(2, valid.size // 8))
        ordered = valid[np.argsort(activity[valid], kind="mergesort")]
        for bucket, positions in enumerate(np.array_split(ordered, bucket_count)):
            rows.append(
                {
                    "bucket": bucket,
                    "n_windows": int(positions.size),
                    "sort_metric_mean": float(activity[positions].mean()),
                    "history_activity_mean": float(history_activity[positions].mean()),
                    "future_activity_mean": float(future_activity[positions].mean()),
                    "generated_width_mean": float(width[positions].mean()),
                }
            )
        if len(rows) < 2:
            return rows, 0.0, 1.0
        widths = np.asarray([row["generated_width_mean"] for row in rows])
        tolerance = max(float(np.max(np.abs(widths))), 1.0) * 1e-6
        monotonicity = float(np.mean(np.diff(widths) > tolerance))
        ratio = float(widths[-1] / max(widths[0], 1e-12))
        return rows, monotonicity, 1.0 if abs(ratio - 1.0) < 1e-5 else ratio

    history_buckets, history_monotonicity, history_ratio = bucket_summary(
        history_activity
    )
    future_buckets, future_monotonicity, future_ratio = bucket_summary(future_activity)
    history_future_rho = _safe_spearman(history_activity, future_activity)
    history_width_rho = _safe_spearman(history_activity, width)
    future_width_rho = _safe_spearman(future_activity, width)
    future_generated_rho = _safe_spearman(future_activity, generated_activity)
    positive_signal = bool(history_future_rho >= 0.10)
    history_width_pass = bool(history_width_rho >= 0.10)
    future_width_pass = bool(future_width_rho >= 0.15)
    history_bucket_pass = bool(history_monotonicity >= 0.50)
    future_bucket_pass = bool(future_monotonicity >= 0.50)
    history_ratio_pass = bool(history_ratio >= 1.05)
    future_ratio_pass = bool(future_ratio >= 1.05)
    observable_pass = history_width_pass and history_bucket_pass and history_ratio_pass
    oracle_pass = future_width_pass and future_bucket_pass and future_ratio_pass
    risk = {
        "history_future_activity_spearman": history_future_rho,
        "observable_signal_present": positive_signal,
        "positive_future_signal_present": positive_signal,
        "observable_signal_floor": 0.10,
        "history_width_spearman": history_width_rho,
        "history_width_spearman_target": 0.10,
        "history_width_pass": history_width_pass,
        "future_width_spearman": future_width_rho,
        "future_width_spearman_target": 0.15,
        "future_width_pass": future_width_pass,
        "future_generated_activity_spearman": future_generated_rho,
        "history_bucket_width_monotonicity": history_monotonicity,
        "future_bucket_width_monotonicity": future_monotonicity,
        "bucket_monotonicity_target": 0.50,
        "history_bucket_pass": history_bucket_pass,
        "future_bucket_pass": future_bucket_pass,
        "history_low_high_width_ratio": history_ratio,
        "future_low_high_width_ratio": future_ratio,
        "low_high_width_ratio_target": 1.05,
        "history_ratio_pass": history_ratio_pass,
        "future_ratio_pass": future_ratio_pass,
        "observable_state_response_pass": observable_pass,
        "oracle_future_alignment_pass": oracle_pass,
        "overall_pass_rule": (
            "observable_state_response_and_oracle_future_alignment"
            if positive_signal
            else "observable_state_response_only_future_signal_absent"
        ),
        "history_buckets": history_buckets,
        "future_buckets": future_buckets,
        "n_windows": int(samples.shape[0]),
        "n_buckets": 5,
        "min_bucket_size": 8,
        "informational": True,
    }
    risk["overall_pass"] = bool(
        observable_pass and (oracle_pass if positive_signal else True)
    )
    mean_iv_history = history_arr.mean(axis=(2, 3))
    vol_of_vol = np.diff(mean_iv_history, axis=1).std(axis=1)
    quantiles = np.quantile(vol_of_vol, [0.2, 0.8])
    regime_masks = {
        "calm": vol_of_vol <= quantiles[0],
        "turb": vol_of_vol >= quantiles[1],
    }
    layer1, layer2, path_bias, width_turb_calm, width_vs_vov = {}, {}, {}, {}, {}
    selected_regime_horizons = [
        value for value in (1, 7, 14, 30) if value <= truth.shape[1]
    ]
    covered_all = (truth >= np.quantile(samples, 0.05, axis=1)) & (
        truth <= np.quantile(samples, 0.95, axis=1)
    )
    for name, mask in regime_masks.items():
        layer1[name], layer2[name] = {}, {}
        for horizon in selected_regime_horizons:
            coverage_grid = covered_all[mask, horizon - 1].mean(axis=0)
            coverage_value = float(coverage_grid.mean())
            lower = np.quantile(samples[mask, :, horizon - 1], 0.05, axis=1)
            upper = np.quantile(samples[mask, :, horizon - 1], 0.95, axis=1)
            observed = truth[mask, horizon - 1]
            regime_median = median[mask, horizon - 1]
            layer1[name][horizon] = {
                "coverage": coverage_value,
                "gt_above_pct": float((observed > upper).mean()),
                "gt_below_pct": float((observed < lower).mean()),
                "median_above_gt_pct": float((regime_median > observed).mean()),
            }
            worst_index = np.unravel_index(
                np.argmin(coverage_grid), coverage_grid.shape
            )
            best_index = np.unravel_index(np.argmax(coverage_grid), coverage_grid.shape)
            layer2[name][horizon] = {
                "grid": coverage_grid.tolist(),
                "worst": float(coverage_grid.min()),
                "worst_cell": list(worst_index),
                "best": float(coverage_grid.max()),
                "best_cell": list(best_index),
            }
        median_mean = median[mask].mean(axis=(2, 3))
        truth_mean = truth[mask].mean(axis=(2, 3))
        below_fraction = (median_mean < truth_mean).mean(axis=1)
        path_bias[name] = {
            "persistent_low_pct": float((below_fraction > 0.80).mean()),
            "persistent_high_pct": float(((1.0 - below_fraction) > 0.80).mean()),
            "mean_frac_median_below_gt": float(below_fraction.mean()),
        }
    calm_width = conditional_width[regime_masks["calm"]]
    turbulent_width = conditional_width[regime_masks["turb"]]
    for horizon in selected_regime_horizons:
        calm_value = float(calm_width[:, horizon - 1].mean())
        turbulent_value = float(turbulent_width[:, horizon - 1].mean())
        width_turb_calm[horizon] = {
            "calm_width": calm_value,
            "turb_width": turbulent_value,
            "width_turb_calm_ratio": _ratio(turbulent_value, calm_value),
        }
        horizon_width = conditional_width[:, horizon - 1].mean(axis=(1, 2))
        if np.ptp(vol_of_vol) <= 1e-12 or np.ptp(horizon_width) <= 1e-12:
            rho_statistic = rho_pvalue = float("nan")
        else:
            rho = sp_stats.spearmanr(vol_of_vol, horizon_width)
            rho_statistic = float(rho.statistic)
            rho_pvalue = float(rho.pvalue)
        q10, q90 = np.quantile(vol_of_vol, [0.1, 0.9])
        p10_width = float(horizon_width[vol_of_vol <= q10].mean())
        p90_width = float(horizon_width[vol_of_vol >= q90].mean())
        width_vs_vov[horizon] = {
            "spearman_rho": rho_statistic,
            "spearman_pval": rho_pvalue,
            "p90_p10_width_ratio": _ratio(p90_width, p10_width),
            "p10_mean_width": p10_width,
            "p90_mean_width": p90_width,
        }
    layer1_pass = bool(
        all(
            row["coverage"] > 0.65
            for regime in layer1.values()
            for horizon, row in regime.items()
        )
    )
    layer2_pass = bool(
        all(
            row["worst"] >= 0.70 and row["best"] <= 0.95
            for regime in layer2.values()
            for row in regime.values()
        )
    )
    n_layer2_total = sum(len(regime) for regime in layer2.values())
    n_layer2_passing = sum(
        row["worst"] >= 0.70 and row["best"] <= 0.95
        for regime in layer2.values()
        for row in regime.values()
    )
    catastrophic = covered_all.mean(axis=1) < 0.30
    catastrophic_rate = float(catastrophic.mean())
    layer3_pass = bool(catastrophic_rate < 0.05)
    regime_coverage = {
        "layer1_regime_horizon": layer1,
        "layer1_pass": layer1_pass,
        "path_bias": path_bias,
        "width_turb_calm": {str(key): value for key, value in width_turb_calm.items()},
        "width_vs_vov": width_vs_vov,
        "layer2_regime_cell": layer2,
        "layer2_pass": layer2_pass,
        "layer2_n_passing": int(n_layer2_passing),
        "layer2_n_total": n_layer2_total,
        "layer3_catastrophic_rate": catastrophic_rate,
        "layer3_n_catastrophic": int(catastrophic.sum()),
        "layer3_pass": layer3_pass,
        "n_calm": int(regime_masks["calm"].sum()),
        "n_turb": int(regime_masks["turb"].sum()),
        "vol_of_vol_q20": float(quantiles[0]),
        "vol_of_vol_q80": float(quantiles[1]),
        "overall_pass": bool(layer1_pass and layer2_pass and layer3_pass),
    }

    level_ks = [
        float(
            sp_stats.ks_2samp(
                samples[..., i, j].reshape(-1), truth[..., i, j].reshape(-1)
            ).statistic
        )
        for i in range(5)
        for j in range(5)
    ]
    change_ks = [
        float(
            sp_stats.ks_2samp(
                sample_change[..., i, j].reshape(-1),
                truth_change[..., i, j].reshape(-1),
            ).statistic
        )
        for i in range(5)
        for j in range(5)
    ]
    above = (median >= truth).mean(axis=(0, 1))
    mean_bias = (median - truth).mean(axis=(0, 1))
    coverage_lo = np.quantile(samples, 0.05, axis=1)
    coverage_hi = np.quantile(samples, 0.95, axis=1)
    window_coverage = ((truth >= coverage_lo) & (truth <= coverage_hi)).mean(
        axis=(1, 2, 3)
    )
    ceiling_per_cell = (samples >= 1.0).mean(axis=(0, 1, 2))
    floor_per_cell = (samples <= 0.0).mean(axis=(0, 1, 2))
    cell_mae = np.abs(median - truth).mean(axis=(0, 1))
    distributional = {
        "ks_test": {
            "ks_grid": np.asarray(change_ks).reshape(5, 5).tolist(),
            "ks_gate": 0.20,
            "n_pass": int(np.sum(np.asarray(change_ks) < 0.20)),
            "worst_stat": float(np.max(change_ks)),
            "median_stat": float(np.median(change_ks)),
            "pass": bool(np.sum(np.asarray(change_ks) < 0.20) >= 15),
        },
        "ks_level_test": {
            "ks_grid": np.asarray(level_ks).reshape(5, 5).tolist(),
            "ks_gate": 0.20,
            "n_pass": int(np.sum(np.asarray(level_ks) < 0.20)),
            "worst_stat": float(np.max(level_ks)),
            "median_stat": float(np.median(level_ks)),
            "pass": bool(np.sum(np.asarray(level_ks) < 0.20) >= 15),
        },
        "median_bias": {
            "above_frac": above.reshape(-1).tolist(),
            "mean_bias": mean_bias.tolist(),
            "n_pass": int(np.sum(np.abs(above - 0.5) <= 0.20)),
            "frac_pass": bool(np.sum(np.abs(above - 0.5) <= 0.20) >= 20),
            "n_mag_pass": int(np.sum(np.abs(mean_bias) < 0.05)),
            "mag_pass": bool(np.sum(np.abs(mean_bias) < 0.05) >= 20),
            "mag_gate_ivpts": 5.0,
            "pass": bool(
                np.sum(np.abs(above - 0.5) <= 0.20) >= 20
                and np.sum(np.abs(mean_bias) < 0.05) >= 20
            ),
        },
        "window_floor": {
            "n_bad_windows": int(np.sum(window_coverage < 0.50)),
            "pct_bad": float(np.mean(window_coverage < 0.50)),
            "worst_window_cov": float(window_coverage.min()),
            "p10_cov": float(np.quantile(window_coverage, 0.10)),
            "pass": bool(np.mean(window_coverage < 0.50) < 0.10),
        },
        "explosion": {
            "at_ceiling": float((samples >= 1.0).mean()),
            "at_floor": float((samples <= 0.0).mean()),
            "ceiling_per_cell": ceiling_per_cell.tolist(),
            "worst_cell_ceiling": float(ceiling_per_cell.max()),
            "worst_cell_floor": float(floor_per_cell.max()),
            "agg_pass": bool(
                (samples >= 1.0).mean() < 0.05 and (samples <= 0.0).mean() < 0.05
            ),
            "percell_pass": bool(
                ceiling_per_cell.max() < 0.10 and floor_per_cell.max() < 0.10
            ),
            "pass": bool(
                (samples >= 1.0).mean() < 0.05
                and (samples <= 0.0).mean() < 0.05
                and ceiling_per_cell.max() < 0.10
                and floor_per_cell.max() < 0.10
            ),
        },
        "cell_mae": {
            "mae_grid": cell_mae.tolist(),
            "n_pass": int(np.sum(cell_mae < 0.10)),
            "worst_mae": float(cell_mae.max()),
            "pass": bool(np.sum(cell_mae < 0.10) >= 20),
        },
    }
    distributional["overall_pass"] = bool(
        distributional["ks_test"]["pass"]
        and distributional["ks_level_test"]["pass"]
        and distributional["median_bias"]["pass"]
        and distributional["window_floor"]["pass"]
        and distributional["explosion"]["pass"]
        and distributional["cell_mae"]["pass"]
    )

    gt_corr = np.nan_to_num(np.corrcoef(gt_flat.T), nan=0.0)
    gen_corr = np.nan_to_num(np.corrcoef(gen_flat.T), nan=0.0)
    upper = np.triu_indices(25, 1)
    corr_ratio = _ratio(float(gen_corr[upper].mean()), float(gt_corr[upper].mean()))
    gt_eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(gt_corr), 0.0))[::-1]
    gen_eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(gen_corr), 0.0))[::-1]

    def effective_rank(values: np.ndarray) -> float:
        probabilities = values / max(float(values.sum()), 1e-12)
        probabilities = probabilities[probabilities > 0]
        return float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    gt_rank, gen_rank = effective_rank(gt_eigenvalues), effective_rank(gen_eigenvalues)
    rank_ratio = _ratio(gen_rank, gt_rank)
    cross_cell = {
        "gt_mean_corr": float(gt_corr[upper].mean()),
        "gen_mean_corr": float(gen_corr[upper].mean()),
        "corr_ratio": corr_ratio,
        "corr_pass": bool(0.5 <= corr_ratio <= 2.0),
        "gt_eff_rank": gt_rank,
        "gen_eff_rank": gen_rank,
        "rank_ratio": rank_ratio,
        "rank_pass": bool(0.5 <= rank_ratio <= 3.0),
        "frob_dist": float(np.linalg.norm(gen_corr - gt_corr, ord="fro")),
        "gt_pc1_var": float(gt_eigenvalues[0] / max(gt_eigenvalues.sum(), 1e-12)),
        "gen_pc1_var": float(gen_eigenvalues[0] / max(gen_eigenvalues.sum(), 1e-12)),
        "overall_pass": bool(0.5 <= corr_ratio <= 2.0 and 0.5 <= rank_ratio <= 3.0),
    }

    predicted_mean = samples.mean(axis=1)
    gt_first_delta = truth[:, 0] - anchor
    gen_first_delta = predicted_mean[:, 0] - anchor
    gt_slope, gt_intercept, gt_r2 = _slope_intercept_r2(anchor, gt_first_delta)
    gen_slope, gen_intercept, gen_r2 = _slope_intercept_r2(anchor, gen_first_delta)
    mr_ratio = _ratio(gen_slope, gt_slope, zero_value=float("nan"))
    gt_cell_slopes = np.zeros((5, 5))
    gen_cell_slopes = np.zeros((5, 5))
    cell_ratio = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            gt_cell_slopes[i, j] = _slope_intercept_r2(
                anchor[:, i, j], gt_first_delta[:, i, j]
            )[0]
            gen_cell_slopes[i, j] = _slope_intercept_r2(
                anchor[:, i, j], gen_first_delta[:, i, j]
            )[0]
            if abs(gt_cell_slopes[i, j]) > 1e-12:
                cell_ratio[i, j] = gen_cell_slopes[i, j] / gt_cell_slopes[i, j]
    active_mask = np.abs(gt_cell_slopes) >= 0.05
    sign_match = np.sign(gt_cell_slopes) == np.sign(gen_cell_slopes)
    cell_pass = (
        active_mask
        & sign_match
        & np.isfinite(cell_ratio)
        & (cell_ratio >= 0.5)
        & (cell_ratio <= 1.5)
    )
    active_count = int(active_mask.sum())
    active_pass_count = int(cell_pass.sum())
    active_pass_rate = active_pass_count / active_count if active_count else 1.0
    slope_corr = (
        _safe_corr(gt_cell_slopes[active_mask], gen_cell_slopes[active_mask])
        if active_count >= 2
        else 1.0
    )
    selected_horizons = [value for value in (1, 7, 14, 30) if value <= truth.shape[1]]
    horizon_metrics = {}
    active_rates, active_corrs, aggregate_ratios = [], [], []
    for horizon in selected_horizons:
        gt_delta_h = truth[:, horizon - 1] - anchor
        gen_delta_h = predicted_mean[:, horizon - 1] - anchor
        gt_slope_h = _slope_intercept_r2(anchor, gt_delta_h)[0]
        gen_slope_h = _slope_intercept_r2(anchor, gen_delta_h)[0]
        ratio_h = _ratio(gen_slope_h, gt_slope_h, zero_value=float("nan"))
        aggregate_ratios.append(ratio_h)
        gt_cells_h = np.zeros((5, 5))
        gen_cells_h = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                gt_cells_h[i, j] = _slope_intercept_r2(
                    anchor[:, i, j], gt_delta_h[:, i, j]
                )[0]
                gen_cells_h[i, j] = _slope_intercept_r2(
                    anchor[:, i, j], gen_delta_h[:, i, j]
                )[0]
        active_h = np.abs(gt_cells_h) >= 0.05
        count_h = int(active_h.sum())
        ratios_h = np.full((5, 5), np.nan)
        np.divide(
            gen_cells_h, gt_cells_h, out=ratios_h, where=np.abs(gt_cells_h) > 1e-12
        )
        pass_h = (
            active_h
            & (np.sign(gt_cells_h) == np.sign(gen_cells_h))
            & np.isfinite(ratios_h)
            & (ratios_h >= 0.5)
            & (ratios_h <= 1.5)
        )
        rate_h = float(pass_h.sum() / count_h) if count_h else 1.0
        corr_h = (
            _safe_corr(gt_cells_h[active_h], gen_cells_h[active_h])
            if count_h >= 2
            else 1.0
        )
        active_rates.append(rate_h)
        active_corrs.append(corr_h)
        horizon_metrics[horizon] = {
            "gt_aggregate_slope": gt_slope_h,
            "gen_aggregate_slope": gen_slope_h,
            "ratio": ratio_h,
            "aggregate_pass": bool(0.70 <= ratio_h <= 1.30),
            "active_pass_rate": rate_h,
            "active_slope_corr": corr_h,
            "active_cell_count": count_h,
        }
    aggregate_profile_pass = bool(
        all(np.isfinite(value) and 0.70 <= value <= 1.30 for value in aggregate_ratios)
    )
    active_profile_pass = bool(
        (np.mean(active_rates) >= 0.65 if active_rates else True)
        and (np.mean(active_corrs) >= 0.65 if active_corrs else True)
    )
    terminal_pass = bool(
        horizon_metrics[selected_horizons[-1]]["aggregate_pass"]
        if selected_horizons
        else True
    )
    full_horizon_pass = aggregate_profile_pass and active_profile_pass and terminal_pass
    aggregate_pass = bool(0.70 <= mr_ratio <= 1.35)
    active_pass = bool(active_pass_rate >= 0.70)
    corr_pass = bool(slope_corr >= 0.65)
    mean_reversion = {
        "methodology": "sampled_first_step_mean_delta_vs_prev_frame",
        "active_slope_threshold": 0.05,
        "gt_aggregate_slope": gt_slope,
        "gt_aggregate_intercept": gt_intercept,
        "gt_aggregate_r2": gt_r2,
        "gen_aggregate_slope": gen_slope,
        "gen_aggregate_intercept": gen_intercept,
        "gen_aggregate_r2": gen_r2,
        "mr_gt_ratio": mr_ratio,
        "aggregate_pass": aggregate_pass,
        "gt_cell_slopes": gt_cell_slopes.tolist(),
        "gen_cell_slopes": gen_cell_slopes.tolist(),
        "cell_ratio": cell_ratio.tolist(),
        "active_cell_mask": active_mask.tolist(),
        "active_cell_pass": cell_pass.tolist(),
        "active_pass_count": active_pass_count,
        "active_cell_count": active_count,
        "active_pass_rate": active_pass_rate,
        "active_pass": active_pass,
        "active_cell_slope_corr": slope_corr,
        "corr_pass": corr_pass,
        "full_horizon": {
            "selected_horizons": selected_horizons,
            "per_horizon": horizon_metrics,
            "aggregate_profile_pass": aggregate_profile_pass,
            "mean_active_pass_rate": (
                float(np.mean(active_rates)) if active_rates else 1.0
            ),
            "mean_active_slope_corr": (
                float(np.mean(active_corrs)) if active_corrs else 1.0
            ),
            "active_profile_pass": active_profile_pass,
            "terminal_pass": terminal_pass,
            "overall_pass": full_horizon_pass,
        },
        "worst_active_cells": [],
        "overall_pass": bool(
            aggregate_pass and active_pass and corr_pass and full_horizon_pass
        ),
    }

    gt_path_max = np.max(np.abs(gt_diff), axis=(1, 2, 3))
    gen_path_max = np.max(np.abs(gen_diff), axis=(2, 3, 4)).reshape(-1)
    jump_ks = float(sp_stats.ks_2samp(gt_path_max, gen_path_max).statistic)
    jump_q90 = _ratio(
        float(np.quantile(gen_path_max, 0.90)), float(np.quantile(gt_path_max, 0.90))
    )
    jump_q99 = _ratio(
        float(np.quantile(gen_path_max, 0.99)), float(np.quantile(gt_path_max, 0.99))
    )
    per_cell_q99 = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            observed_q99 = float(np.quantile(np.abs(gt_diff[..., i, j]), 0.99))
            generated_q99 = float(
                np.median(
                    [
                        np.quantile(np.abs(gen_diff[:, index, ..., i, j]), 0.99)
                        for index in range(min(10, samples.shape[1]))
                    ]
                )
            )
            if observed_q99 > 1e-12:
                per_cell_q99[i, j] = generated_q99 / observed_q99
    per_cell_pass = (
        np.isfinite(per_cell_q99) & (per_cell_q99 >= 0.5) & (per_cell_q99 <= 2.0)
    )
    global_q99 = float(np.quantile(np.abs(gt_diff), 0.99))
    gt_extreme = (np.abs(gt_diff) >= global_q99).any(axis=(1, 2, 3))
    gen_extreme = (np.abs(gen_diff) >= global_q99).any(axis=(2, 3, 4)).reshape(-1)
    gt_extreme_rate = float(gt_extreme.mean())
    gen_extreme_rate = float(gen_extreme.mean())
    incidence_ratio = _ratio(gen_extreme_rate, gt_extreme_rate)
    pathwise = {
        "pathwise_max_jump": {
            "ks_stat": jump_ks,
            "ks_gate": 0.5,
            "q90_ratio": jump_q90,
            "q99_ratio": jump_q99,
            "pass": bool(jump_ks < 0.5 and 0.5 <= jump_q99 <= 2.0),
        },
        "per_cell_q99": {
            "ratio_grid": per_cell_q99.tolist(),
            "n_pass": int(per_cell_pass.sum()),
            "pass": bool(per_cell_pass.sum() >= 20),
            "gate_lo": 0.5,
            "gate_hi": 2.0,
        },
        "window_extreme_incidence": {
            "gt_rate": gt_extreme_rate,
            "gen_rate": gen_extreme_rate,
            "ratio": incidence_ratio,
            "pass": bool(0.5 <= incidence_ratio <= 2.0),
        },
    }
    pathwise["overall_pass"] = bool(
        pathwise["pathwise_max_jump"]["pass"]
        and pathwise["per_cell_q99"]["pass"]
        and pathwise["window_extreme_incidence"]["pass"]
    )

    suite = {
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "iv_ewma_economic_link": economic,
        "regime_coverage": regime_coverage,
        "risk_state_allocation": risk,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    counted = (
        "surface",
        "coverage",
        "conditionality",
        "time_series",
        "block_ar",
        "cointegration",
        "regime_coverage",
        "distributional_fidelity",
        "cross_cell_correlation",
        "mean_reversion",
        "pathwise_jump_realism",
    )
    failed = [name for name in counted if suite[name].get("overall_pass") is not True]
    suite["summary"] = {
        "n_pass": len(counted) - len(failed),
        "n_total": 11,
        "failed_suites": failed,
    }
    metrics = {
        "cov90": overall[0.9],
        "calibration_error": calibration_error,
        "level_ks_median": distributional["ks_level_test"]["median_stat"],
        "median_abs_dev": float(np.median(np.abs(above - 0.5))),
        "width_rho": risk["history_width_spearman"],
        "mr_ratio": mr_ratio,
        "economic_ratio": float(cointegration["gen_gt_ratio_legacy"]),
        "jump_ks": jump_ks,
        "kurtosis_ratio": kurtosis_ratio,
        "acf_corr": acf_corr,
    }
    return suite, metrics
