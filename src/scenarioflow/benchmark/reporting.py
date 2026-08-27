"""Threshold and canonical-versus-replay reporting for final benchmark results."""

from __future__ import annotations

import math
from typing import Any, Mapping

TABLE_THRESHOLD_RULES: dict[str, dict[str, float | str]] = {
    "Cov90": {"kind": "band", "lo": 0.75, "hi": 0.98},
    "CalErr": {"kind": "upper", "hi": 0.20},
    "Level KS": {"kind": "upper", "hi": 0.20},
    "Width rho": {"kind": "lower", "lo": 0.20},
    "MR ratio": {"kind": "band", "lo": 0.25, "hi": 1.75},
    "Econ ratio": {"kind": "band", "lo": 0.75, "hi": 1.30},
    "Jump KS": {"kind": "upper", "hi": 0.50},
    "Kurtosis": {"kind": "band", "lo": 0.50, "hi": 1.50},
    "Corr score": {"kind": "lower", "lo": 0.30},
    "Factor KS": {"kind": "upper", "hi": 0.15},
    "Tail q99": {"kind": "band", "lo": 0.50, "hi": 1.50},
}

METRIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CRPS", ("panel_b_crps", "overall")),
    ("IV CRPS", ("panel_b_crps", "iv_crps")),
    ("Factor CRPS", ("panel_b_crps", "factor_crps")),
    ("Energy Score", ("panel_b_energy_score", "overall")),
    ("Variogram", ("panel_b_variogram_score", "variogram_score")),
    ("Corr score", ("panel_b_correlation", "corr_score")),
    ("Factor KS", ("panel_b_factor_ks", "mean_statistic")),
    ("Tail q99", ("joint_panel_quality", "factor_tail_q99_ratio_median")),
    ("Cov90", ("current_panel_iv_metrics", "cov90")),
    ("CalErr", ("current_panel_iv_metrics", "calibration_error")),
    ("Level KS", ("current_panel_iv_metrics", "level_ks_median")),
    ("Median dev", ("current_panel_iv_metrics", "median_abs_dev")),
    ("Width rho", ("current_panel_iv_metrics", "width_rho")),
    ("MR ratio", ("current_panel_iv_metrics", "mr_ratio")),
    ("Econ ratio", ("current_panel_iv_metrics", "economic_ratio")),
    ("Jump KS", ("current_panel_iv_metrics", "jump_ks")),
    ("Kurtosis", ("current_panel_iv_metrics", "kurtosis_ratio")),
    ("ACF corr", ("current_panel_iv_metrics", "acf_corr")),
)


def metric_values(result: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, path in METRIC_PATHS:
        value: Any = result
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"missing metric {name}: {'.'.join(path)}")
            value = value[key]
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {name} is non-finite")
        values[name] = number
    return values


def threshold_status(value: float, rule: Mapping[str, Any]) -> bool:
    kind = rule["kind"]
    if kind == "upper":
        return value <= float(rule["hi"])
    if kind == "lower":
        return value >= float(rule["lo"])
    if kind == "band":
        return float(rule["lo"]) <= value <= float(rule["hi"])
    raise ValueError(f"unknown threshold kind {kind!r}")
