"""Threshold and canonical-versus-replay reporting for final benchmark results."""

from __future__ import annotations

import json
import math
from pathlib import Path
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


def compare_reproduction_runs(
    canonical: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two explicitly supplied result objects without guessing paths."""
    canonical_metrics = metric_values(canonical)
    replay_metrics = metric_values(replay)
    rows = []
    for name in canonical_metrics:
        left = canonical_metrics[name]
        right = replay_metrics[name]
        rows.append(
            {
                "metric": name,
                "canonical": left,
                "replay": right,
                "absolute_difference": abs(right - left),
                "relative_difference": abs(right - left) / max(abs(left), 1e-12),
            }
        )
    return {
        "schema_version": "scenarioflow.reproduction-comparison.v1",
        "structural_invariants": {
            "mode": canonical.get("mode") == replay.get("mode"),
            "n_windows": canonical.get("n_windows") == replay.get("n_windows"),
            "n_samples": canonical.get("n_samples") == replay.get("n_samples"),
            "state_specs": canonical.get("state_specs") == replay.get("state_specs"),
        },
        "metrics": rows,
    }


def write_reproduction_report(
    report: Mapping[str, Any], output_dir: str | Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Reproduction comparison", ""]
    for name, passed in report["structural_invariants"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "| Metric | Canonical | Replay | Relative difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["metrics"]:
        lines.append(
            f"| {row['metric']} | {row['canonical']:.6g} | {row['replay']:.6g} | {row['relative_difference']:.3%} |"
        )
    (output / "reproduction_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
