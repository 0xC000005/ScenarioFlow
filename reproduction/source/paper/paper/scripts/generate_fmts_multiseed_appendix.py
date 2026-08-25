"""Generate the FMTS multi-seed appendix table from the frozen aggregate.

This module deliberately accepts no alternate artifact at the command line.
The source is the completed aggregate, whose schema and seed contract are
validated before any LaTeX is written.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    REPOSITORY_ROOT / "results/fmts_multiseed_20260805/aggregate/summary.json"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "paper/camera_ready/neurips_workshop/main_v4_fmts_multiseed_table.tex"
)

ICAIF_CANDIDATES = (
    "scenarioflow",
    "scenarioflow_graph",
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
    "deepvar",
    "timegrad",
    "path_diffusion",
)
REPORT_CANDIDATES = tuple(
    candidate for candidate in ICAIF_CANDIDATES if candidate != "scenarioflow_graph"
)
EXCLUDED_CANDIDATES = ("deepvar_sni",)
SOURCE_CANDIDATES = frozenset((*ICAIF_CANDIDATES, *EXCLUDED_CANDIDATES))

LEARNED_CANDIDATES = frozenset(
    {"scenarioflow", "scenarioflow_graph", "deepvar", "timegrad", "path_diffusion"}
)
CLASSICAL_CANDIDATES = frozenset(ICAIF_CANDIDATES) - LEARNED_CANDIDATES

CANDIDATE_LABELS = {
    "scenarioflow": r"\scenarioflow{}",
    "random_walk": "Random walk",
    "bootstrap": "Bootstrap",
    "historical_sim": "Historical sim.",
    "pca_var": "PCA-VAR",
    "garch_ccc": "GARCH-CCC",
    "filtered_hs": "Filtered hist. sim.",
    "deepvar": "DeepVAR",
    "timegrad": "TimeGrad",
    "path_diffusion": "CSDI",
}

# source metric, LaTex heading, decimal places, preferred direction, target
METRIC_SPECS = {
    "CRPS": (r"CRPS$\downarrow$", 4, "min", None),
    "IV CRPS": (r"IV CRPS$\downarrow$", 4, "min", None),
    "Factor CRPS": (r"Factor CRPS$\downarrow$", 4, "min", None),
    "Energy Score": (r"Energy$\downarrow$", 3, "min", None),
    "Variogram": (r"Variogram$\downarrow$", 4, "min", None),
    "Corr score": (r"Corr.$\uparrow$", 3, "max", None),
    "Factor KS": (r"Factor KS$\downarrow$", 3, "min", None),
    "Tail q99": (r"Tail q99$\to1$", 3, "target", 1.0),
    "Cov90": (r"Cov90$\to0.90$", 3, "target", 0.90),
    "CalErr": (r"CalErr$\downarrow$", 3, "min", None),
    "Level KS": (r"Level KS$_{\rm med}\downarrow$", 3, "min", None),
    "Median dev": (r"Med. dev.$\downarrow$", 3, "min", None),
    "Width rho": (r"Width $\rho\uparrow$", 3, "max", None),
    "MR ratio": (r"MR$\to1$", 3, "target", 1.0),
    "Econ ratio": (r"Econ.$\to1$", 3, "target", 1.0),
    "Jump KS": (r"Jump KS$\downarrow$", 3, "min", None),
    "Kurtosis": (r"Kurt.$\to1$", 3, "target", 1.0),
    "ACF corr": (r"ACF$\uparrow$", 3, "max", None),
}
PANEL_A_METRICS = (
    "CRPS",
    "IV CRPS",
    "Factor CRPS",
    "Energy Score",
    "Variogram",
    "Corr score",
    "Factor KS",
    "Tail q99",
)
PANEL_B_METRICS = (
    "Cov90",
    "CalErr",
    "Level KS",
    "Median dev",
    "Width rho",
    "MR ratio",
    "Econ ratio",
    "Jump KS",
    "Kurtosis",
    "ACF corr",
)
REPORT_METRICS = PANEL_A_METRICS + PANEL_B_METRICS
REPORT_BLOCKS = (
    ("(a1) Distributional scores", PANEL_A_METRICS[:4]),
    ("(a2) Dependence, marginals, and tails", PANEL_A_METRICS[4:]),
    ("(b1) Calibration and historical response", PANEL_B_METRICS[:5]),
    ("(b2) Path dynamics", PANEL_B_METRICS[5:]),
)

THRESHOLD_SPECS = {
    "Corr score": ("lower", 0.30, None),
    "Factor KS": ("upper", None, 0.15),
    "Tail q99": ("band", 0.50, 1.50),
    "Cov90": ("band", 0.75, 0.98),
    "CalErr": ("upper", None, 0.20),
    "Level KS": ("upper", None, 0.20),
    "Width rho": ("lower", 0.20, None),
    "MR ratio": ("band", 0.25, 1.75),
    "Econ ratio": ("band", 0.75, 1.30),
    "Jump KS": ("upper", None, 0.50),
    "Kurtosis": ("band", 0.50, 1.50),
}

EXPECTED_TRAINING_SEEDS = (1001, 1002, 1003)
EXPECTED_ROLLOUT_SEEDS = (2001, 2002, 2003, 2004, 2005)
SUMMARY_ROOT_KEYS = {
    "bootstrap_contract",
    "candidate_summaries",
    "contract",
    "generated_at_utc",
    "paired_comparisons",
    "per_runs",
    "protocol_dag_sha256",
    "protocol_manifest",
    "schema_version",
    "seed_schedule",
}
STAT_KEYS = {"max", "mean", "min", "n", "std"}


def load_summary(path: Path = SUMMARY_PATH) -> dict[str, Any]:
    """Read the one frozen aggregate that is authorized for this table."""
    if path != SUMMARY_PATH:
        raise ValueError("only the frozen FMTS aggregate summary is permitted")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen aggregate summary: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_metric_stats(metrics: Any, expected_n: int, context: str) -> None:
    _require(isinstance(metrics, dict), f"{context}: metrics must be an object")
    _require(
        set(metrics) == set(METRIC_SPECS),
        f"{context}: metric contract does not contain exactly the 18 ICAIF metrics",
    )
    for metric, stats in metrics.items():
        _require(
            isinstance(stats, dict) and set(stats) == STAT_KEYS,
            f"{context}: invalid {metric} statistics",
        )
        _require(
            stats["n"] == expected_n, f"{context}: {metric} has the wrong sample count"
        )
        _require(
            all(_is_number(stats[key]) for key in STAT_KEYS - {"n"}),
            f"{context}: {metric} has non-finite statistics",
        )


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _matches_stats(actual: dict[str, Any], values: list[float], context: str) -> None:
    expected = _summary_stats(values)
    _require(actual["n"] == expected["n"], f"{context}: unexpected sample count")
    for key in ("mean", "std", "min", "max"):
        _require(
            math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-12),
            f"{context}: {key} does not match its frozen run-level values",
        )


def _threshold_passes(
    value: float, kind: str, lower: float | None, upper: float | None
) -> bool:
    if kind == "lower":
        return value >= lower  # type: ignore[operator]
    if kind == "upper":
        return value <= upper  # type: ignore[operator]
    if kind == "band":
        return lower <= value <= upper  # type: ignore[operator]
    raise ValueError(f"unknown threshold kind: {kind}")


def _validate_per_runs(
    per_runs: Any,
) -> dict[tuple[str, int | None, int], dict[str, Any]]:
    _require(isinstance(per_runs, list), "per_runs must be a list")
    expected_run_count = len(CLASSICAL_CANDIDATES) * len(EXPECTED_ROLLOUT_SEEDS) + len(
        LEARNED_CANDIDATES | set(EXCLUDED_CANDIDATES)
    ) * len(EXPECTED_TRAINING_SEEDS) * len(EXPECTED_ROLLOUT_SEEDS)
    _require(
        len(per_runs) == expected_run_count, "per_runs has an unexpected seed-grid size"
    )
    by_seed: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    for run in per_runs:
        _require(isinstance(run, dict), "per_runs contains a non-object run")
        _require(
            set(run["metrics"]) == set(METRIC_SPECS),
            "per_runs has an incomplete metric contract",
        )
        candidate = run.get("candidate")
        rollout_seed = run.get("rollout_seed")
        training_seed = run.get("training_seed")
        _require(
            candidate in SOURCE_CANDIDATES, "per_runs contains an unexpected candidate"
        )
        _require(
            rollout_seed in EXPECTED_ROLLOUT_SEEDS,
            "per_runs contains an unexpected rollout seed",
        )
        if candidate in LEARNED_CANDIDATES or candidate in EXCLUDED_CANDIDATES:
            _require(
                training_seed in EXPECTED_TRAINING_SEEDS,
                "learned run has an unexpected training seed",
            )
        else:
            _require(
                training_seed is None, "classical run must not have a training seed"
            )
        key = (candidate, training_seed, rollout_seed)
        _require(key not in by_seed, "per_runs contains a duplicate candidate/seed run")
        thresholds = run.get("thresholds")
        _require(
            isinstance(thresholds, dict) and set(thresholds) == set(THRESHOLD_SPECS),
            "per_runs has an unexpected threshold contract",
        )
        for metric, (kind, lower, upper) in THRESHOLD_SPECS.items():
            threshold = thresholds[metric]
            _require(
                isinstance(threshold, dict), f"{metric} threshold must be an object"
            )
            _require(threshold.get("kind") == kind, f"{metric} threshold kind changed")
            _require(
                threshold.get("lower") == lower and threshold.get("upper") == upper,
                f"{metric} threshold bound changed",
            )
            value = run["metrics"][metric]
            _require(_is_number(value), f"{metric} run value is not finite")
            _require(
                threshold.get("value") == value,
                f"{metric} threshold value disagrees with run metric",
            )
            _require(
                threshold.get("passed") == _threshold_passes(value, kind, lower, upper),
                f"{metric} threshold pass status is inconsistent",
            )
        by_seed[key] = run
    for candidate in SOURCE_CANDIDATES:
        training_seeds = (
            EXPECTED_TRAINING_SEEDS
            if candidate in LEARNED_CANDIDATES or candidate in EXCLUDED_CANDIDATES
            else (None,)
        )
        for training_seed in training_seeds:
            for rollout_seed in EXPECTED_ROLLOUT_SEEDS:
                _require(
                    (candidate, training_seed, rollout_seed) in by_seed,
                    "per_runs is missing a required candidate/seed run",
                )
    return by_seed


def validate_summary(summary: dict[str, Any]) -> None:
    """Fail closed unless the frozen aggregate matches the reporting contract."""
    _require(isinstance(summary, dict), "summary must be an object")
    _require(
        set(summary) == SUMMARY_ROOT_KEYS,
        "summary schema does not match fmts.multiseed_aggregate.v1",
    )
    _require(
        summary.get("schema_version") == "fmts.multiseed_aggregate.v1",
        "unsupported summary schema version",
    )
    _require(
        summary.get("seed_schedule")
        == {
            "training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "rollout_seeds": list(EXPECTED_ROLLOUT_SEEDS),
        },
        "training or rollout seed schedule changed",
    )
    candidate_summaries = summary.get("candidate_summaries")
    _require(
        isinstance(candidate_summaries, dict), "candidate_summaries must be an object"
    )
    _require(
        set(candidate_summaries) == SOURCE_CANDIDATES,
        "candidate contract changed or deepvar_sni was omitted",
    )
    by_seed = _validate_per_runs(summary.get("per_runs"))

    for candidate in SOURCE_CANDIDATES:
        candidate_summary = candidate_summaries[candidate]
        _require(
            isinstance(candidate_summary, dict),
            f"{candidate}: candidate summary must be an object",
        )
        _require(
            candidate_summary.get("candidate") == candidate,
            f"{candidate}: candidate identifier mismatch",
        )
        if candidate in LEARNED_CANDIDATES or candidate in EXCLUDED_CANDIDATES:
            sensitivity = candidate_summary.get("training_seed_sensitivity")
            _require(
                isinstance(sensitivity, dict),
                f"{candidate}: missing training_seed_sensitivity",
            )
            _require(
                sensitivity.get("training_seeds") == list(EXPECTED_TRAINING_SEEDS),
                f"{candidate}: training seed contract changed",
            )
            _require(
                sensitivity.get("n_training_seeds") == len(EXPECTED_TRAINING_SEEDS),
                f"{candidate}: wrong training-seed count",
            )
            _require(
                sensitivity.get("unit") == "per_training_seed_rollout_mean",
                f"{candidate}: wrong training-sensitivity unit",
            )
            _validate_metric_stats(
                sensitivity.get("metrics"), len(EXPECTED_TRAINING_SEEDS), candidate
            )
            for metric, stats in sensitivity["metrics"].items():
                checkpoint_means = [
                    statistics.fmean(
                        by_seed[(candidate, seed, rollout)]["metrics"][metric]
                        for rollout in EXPECTED_ROLLOUT_SEEDS
                    )
                    for seed in EXPECTED_TRAINING_SEEDS
                ]
                _matches_stats(stats, checkpoint_means, f"{candidate} {metric}")
        else:
            sensitivity = candidate_summary.get("sampling_sensitivity")
            _require(
                isinstance(sensitivity, dict),
                f"{candidate}: missing sampling_sensitivity",
            )
            _require(
                sensitivity.get("rollout_seeds") == list(EXPECTED_ROLLOUT_SEEDS),
                f"{candidate}: rollout seed contract changed",
            )
            _require(
                sensitivity.get("n_rollout_seeds") == len(EXPECTED_ROLLOUT_SEEDS),
                f"{candidate}: wrong rollout-seed count",
            )
            _validate_metric_stats(
                sensitivity.get("metrics"), len(EXPECTED_ROLLOUT_SEEDS), candidate
            )
            for metric, stats in sensitivity["metrics"].items():
                rollout_values = [
                    by_seed[(candidate, None, seed)]["metrics"][metric]
                    for seed in EXPECTED_ROLLOUT_SEEDS
                ]
                _matches_stats(stats, rollout_values, f"{candidate} {metric}")


def select_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Select fixed-order workshop rows using their correct variability unit."""
    validate_summary(summary)
    rows: dict[str, dict[str, Any]] = {}
    for candidate in REPORT_CANDIDATES:
        candidate_summary = summary["candidate_summaries"][candidate]
        if candidate in LEARNED_CANDIDATES:
            unit = "training_seed_sensitivity"
            metrics = candidate_summary[unit]["metrics"]
            n = candidate_summary[unit]["n_training_seeds"]
        else:
            unit = "sampling_sensitivity"
            metrics = candidate_summary[unit]["metrics"]
            n = candidate_summary[unit]["n_rollout_seeds"]
        rows[candidate] = {
            "label": CANDIDATE_LABELS[candidate],
            "metrics": metrics,
            "n": n,
            "unit": unit,
        }
    return rows


def _best_candidates(rows: dict[str, dict[str, Any]], metric: str) -> set[str]:
    _, _, direction, target = METRIC_SPECS[metric]
    values = {
        candidate: row["metrics"][metric]["mean"] for candidate, row in rows.items()
    }
    if direction == "min":
        best_value = min(values.values())
        return {
            candidate
            for candidate, value in values.items()
            if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
        }
    if direction == "max":
        best_value = max(values.values())
        return {
            candidate
            for candidate, value in values.items()
            if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
        }
    distances = {candidate: abs(value - target) for candidate, value in values.items()}
    best_distance = min(distances.values())
    return {
        candidate
        for candidate, value in distances.items()
        if math.isclose(value, best_distance, rel_tol=1e-12, abs_tol=1e-12)
    }


def _confidence_interval(stats: dict[str, Any], n: int) -> tuple[float, float]:
    """Return a two-sided 95% Student-t interval for the reported mean."""
    if n < 2:
        raise ValueError("a confidence interval requires at least two observations")
    half_width = float(student_t.ppf(0.975, n - 1)) * stats["std"] / math.sqrt(n)
    return stats["mean"] - half_width, stats["mean"] + half_width


def _interval_decimals(half_width: float, mean_decimals: int) -> int:
    """Keep enough precision to show a nonzero confidence interval."""
    if half_width == 0:
        return mean_decimals
    exponent = math.floor(math.log10(abs(half_width)))
    return max(mean_decimals, 1 - exponent)


def _format_cell(stats: dict[str, Any], metric: str, best: bool, n: int) -> str:
    _, decimals, _, _ = METRIC_SPECS[metric]
    low, high = _confidence_interval(stats, n)
    interval_decimals = _interval_decimals((high - low) / 2.0, decimals)
    mean = stats["mean"]
    formatted_mean = f"{mean:.{interval_decimals}f}"
    if best:
        formatted_mean = rf"\textbf{{{formatted_mean}}}"
    return (
        rf"{formatted_mean}\,[{low:.{interval_decimals}f}, "
        rf"{high:.{interval_decimals}f}]"
    )


def _render_panel(
    rows: dict[str, dict[str, Any]], *, title: str, metrics: tuple[str, ...]
) -> str:
    headers = " & ".join(METRIC_SPECS[metric][0] for metric in metrics)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'c' * len(metrics)}@{{}}}}",
        r"\toprule",
        rf"\multicolumn{{{len(metrics) + 1}}}{{l}}{{\textbf{{{title}}}}} \\",
        r"\midrule",
        f"Model & {headers} " + r"\\",
        r"\midrule",
    ]
    best_by_metric = {
        metric: _best_candidates(rows, metric) for metric in metrics
    }
    for candidate, row in rows.items():
        cells = [
            _format_cell(
                row["metrics"][metric],
                metric,
                candidate in best_by_metric[metric],
                row["n"],
            )
            for metric in metrics
        ]
        lines.append(f"{row['label']} & {' & '.join(cells)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def render_table(summary: dict[str, Any]) -> str:
    """Render one two-panel table containing every full-benchmark metric."""
    rows = select_rows(summary)
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Multi-seed sensitivity across the full distributional-fidelity, calibration, and path-realism panels. Each cell reports the mean and its 95\% Student-$t$ confidence interval. Learned-model intervals use three training-run means (each averaged over five sampling seeds); classical-model intervals use five sampling-seed values. The intervals therefore describe training-run variability for learned models and sampling variability for classical models. Bold marks the best aggregate mean.}",
            r"\label{tab:fmts-multiseed-sensitivity}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.80}",
            *[
                item
                for block_index, (title, metrics) in enumerate(REPORT_BLOCKS)
                for item in (
                    *(() if block_index == 0 else (r"\vspace{3pt}",)),
                    _render_panel(rows, title=title, metrics=metrics),
                )
            ],
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    summary = load_summary()
    table = render_table(summary)
    OUTPUT_PATH.write_text(table)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
