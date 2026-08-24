"""Render the frozen FMTS multi-seed sensitivity table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t

from ..evaluation.current_panel import METRIC_PATHS, TABLE_THRESHOLD_RULES
from ..workflows.multiseed import (
    CANDIDATE_IDS,
    CLASSICAL_CANDIDATES,
    ROLLOUT_SEEDS,
    TRAINED_CANDIDATES,
    TRAIN_SEEDS,
)

FROZEN_SUMMARY_BYTES = 692_525
FROZEN_SUMMARY_SHA256 = (
    "4ebb4d26666cf3d05f8f85c03f678584c23be799479e94f328078fd6783eb3fb"
)
FROZEN_SUMMARY_SCHEMA = "fmts.multiseed_aggregate.v1"

REPORT_CANDIDATES = tuple(
    candidate
    for candidate in CANDIDATE_IDS
    if candidate not in {"scenarioflow_graph", "deepvar_sni"}
)
SOURCE_CANDIDATES = frozenset(CANDIDATE_IDS)
LEARNED_CANDIDATES = frozenset(TRAINED_CANDIDATES)
CLASSICAL_CANDIDATE_SET = frozenset(CLASSICAL_CANDIDATES)

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

# metric name -> (LaTeX heading, base decimal places, preferred direction, target)
METRIC_SPECS: dict[str, tuple[str, int, str, float | None]] = {
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
PANEL_A_METRICS = tuple(name for name, _path in METRIC_PATHS[:8])
PANEL_B_METRICS = tuple(name for name, _path in METRIC_PATHS[8:])
REPORT_BLOCKS = (
    ("(a1) Distributional scores", PANEL_A_METRICS[:4]),
    ("(a2) Dependence, marginals, and tails", PANEL_A_METRICS[4:]),
    ("(b1) Calibration and historical response", PANEL_B_METRICS[:5]),
    ("(b2) Path dynamics", PANEL_B_METRICS[5:]),
)
STAT_KEYS = frozenset({"max", "mean", "min", "n", "std"})
SUMMARY_ROOT_KEYS = frozenset(
    {
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
)


class MultiseedExportError(ValueError):
    """Raised when the frozen aggregate or requested output is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MultiseedExportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_summary(path: Path) -> dict[str, Any]:
    source = Path(path)
    try:
        if source.is_symlink() or source.stat().st_size != FROZEN_SUMMARY_BYTES:
            raise MultiseedExportError("multi-seed summary authority identity changed")
        if _sha256(source) != FROZEN_SUMMARY_SHA256:
            raise MultiseedExportError("multi-seed summary authority identity changed")
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MultiseedExportError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiseedExportError("multi-seed summary is unreadable") from error
    if not isinstance(document, dict):
        raise MultiseedExportError("multi-seed summary must be a JSON object")
    return document


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiseedExportError(message)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _task3_threshold(metric: str) -> tuple[str, float | None, float | None]:
    rule = TABLE_THRESHOLD_RULES[metric]
    kind = str(rule["kind"])
    return (
        kind,
        float(rule["lo"]) if "lo" in rule else None,
        float(rule["hi"]) if "hi" in rule else None,
    )


def _threshold_passes(
    value: float, kind: str, lower: float | None, upper: float | None
) -> bool:
    if kind == "lower":
        return value >= float(lower)
    if kind == "upper":
        return value <= float(upper)
    if kind == "band":
        return float(lower) <= value <= float(upper)
    raise MultiseedExportError(f"unknown threshold kind: {kind}")


def _summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _validate_metric_stats(metrics: Any, *, expected_n: int, context: str) -> None:
    _require(isinstance(metrics, dict), f"{context}: metrics must be an object")
    _require(
        set(metrics) == {name for name, _path in METRIC_PATHS},
        f"{context}: metrics must contain exactly the 18 Task-3 metrics",
    )
    for metric, stats in metrics.items():
        _require(
            isinstance(stats, dict) and set(stats) == STAT_KEYS,
            f"{context}: invalid {metric} statistics",
        )
        _require(stats["n"] == expected_n, f"{context}: wrong {metric} count")
        _require(
            all(_is_number(stats[key]) for key in STAT_KEYS - {"n"}),
            f"{context}: non-finite {metric} statistics",
        )


def _matches_stats(
    actual: Mapping[str, Any], values: Sequence[float], context: str
) -> None:
    expected = _summary_stats(values)
    _require(actual["n"] == expected["n"], f"{context}: wrong sample count")
    for key in ("mean", "std", "min", "max"):
        _require(
            math.isclose(
                float(actual[key]),
                float(expected[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"{context}: {key} disagrees with frozen runs",
        )


def _validate_per_runs(
    per_runs: Any,
) -> dict[tuple[str, int | None, int], dict[str, Any]]:
    _require(isinstance(per_runs, list), "per_runs must be a list")
    expected_count = len(CLASSICAL_CANDIDATE_SET) * len(ROLLOUT_SEEDS) + len(
        LEARNED_CANDIDATES
    ) * len(TRAIN_SEEDS) * len(ROLLOUT_SEEDS)
    _require(len(per_runs) == expected_count, "per_runs seed grid changed")
    metric_names = {name for name, _path in METRIC_PATHS}
    threshold_names = tuple(TABLE_THRESHOLD_RULES)
    by_seed: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    for run in per_runs:
        _require(isinstance(run, dict), "per_runs contains a non-object")
        _require(
            set(run.get("metrics", {})) == metric_names,
            "per_runs metric contract changed",
        )
        candidate = run.get("candidate")
        training_seed = run.get("training_seed")
        rollout_seed = run.get("rollout_seed")
        _require(candidate in SOURCE_CANDIDATES, "unexpected multi-seed candidate")
        _require(rollout_seed in ROLLOUT_SEEDS, "unexpected rollout seed")
        if candidate in LEARNED_CANDIDATES:
            _require(training_seed in TRAIN_SEEDS, "unexpected training seed")
        else:
            _require(training_seed is None, "classical run has a training seed")
        key = (str(candidate), training_seed, int(rollout_seed))
        _require(key not in by_seed, "duplicate candidate/seed run")
        thresholds = run.get("thresholds")
        _require(
            isinstance(thresholds, dict) and set(thresholds) == set(threshold_names),
            "per_runs threshold contract changed",
        )
        for metric in threshold_names:
            threshold = thresholds[metric]
            kind, lower, upper = _task3_threshold(metric)
            value = run["metrics"][metric]
            _require(_is_number(value), f"{metric} run value is non-finite")
            _require(
                isinstance(threshold, dict)
                and threshold.get("kind") == kind
                and threshold.get("lower") == lower
                and threshold.get("upper") == upper
                and threshold.get("value") == value
                and threshold.get("passed")
                == _threshold_passes(float(value), kind, lower, upper),
                f"{metric} threshold record changed",
            )
        by_seed[key] = run
    for candidate in CANDIDATE_IDS:
        training_seeds: Sequence[int | None] = (
            TRAIN_SEEDS if candidate in LEARNED_CANDIDATES else (None,)
        )
        for training_seed in training_seeds:
            for rollout_seed in ROLLOUT_SEEDS:
                _require(
                    (candidate, training_seed, rollout_seed) in by_seed,
                    "per_runs is missing a required seed run",
                )
    return by_seed


def _validated_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require(set(summary) == SUMMARY_ROOT_KEYS, "multi-seed root schema changed")
    _require(
        summary.get("schema_version") == FROZEN_SUMMARY_SCHEMA,
        "unsupported multi-seed schema",
    )
    _require(
        summary.get("seed_schedule")
        == {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "multi-seed schedule changed",
    )
    contract = summary.get("contract")
    _require(
        isinstance(contract, dict)
        and contract.get("mode") == "current_realvix_39_state_panel"
        and contract.get("n_evaluation_windows") == 441
        and contract.get("n_scenarios_per_window") == 64
        and contract.get("n_market_variables") == 39
        and contract.get("history_length") == 30
        and contract.get("future_length") == 30,
        "multi-seed evaluation contract changed",
    )
    candidates = summary.get("candidate_summaries")
    _require(
        isinstance(candidates, dict) and set(candidates) == SOURCE_CANDIDATES,
        "multi-seed candidate set changed",
    )
    by_seed = _validate_per_runs(summary.get("per_runs"))
    rows: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_IDS:
        candidate_summary = candidates[candidate]
        _require(
            isinstance(candidate_summary, dict)
            and candidate_summary.get("candidate") == candidate,
            f"{candidate}: candidate summary changed",
        )
        if candidate in LEARNED_CANDIDATES:
            sensitivity = candidate_summary.get("training_seed_sensitivity")
            _require(
                isinstance(sensitivity, dict)
                and sensitivity.get("training_seeds") == list(TRAIN_SEEDS)
                and sensitivity.get("n_training_seeds") == len(TRAIN_SEEDS)
                and sensitivity.get("unit") == "per_training_seed_rollout_mean",
                f"{candidate}: training sensitivity contract changed",
            )
            metrics = sensitivity.get("metrics")
            _validate_metric_stats(
                metrics, expected_n=len(TRAIN_SEEDS), context=candidate
            )
            for metric, stats in metrics.items():
                training_means = [
                    statistics.fmean(
                        by_seed[(candidate, train_seed, rollout_seed)]["metrics"][
                            metric
                        ]
                        for rollout_seed in ROLLOUT_SEEDS
                    )
                    for train_seed in TRAIN_SEEDS
                ]
                _matches_stats(stats, training_means, f"{candidate} {metric}")
            n = len(TRAIN_SEEDS)
            unit = "training_seed_sensitivity"
        else:
            sensitivity = candidate_summary.get("sampling_sensitivity")
            _require(
                isinstance(sensitivity, dict)
                and sensitivity.get("rollout_seeds") == list(ROLLOUT_SEEDS)
                and sensitivity.get("n_rollout_seeds") == len(ROLLOUT_SEEDS),
                f"{candidate}: sampling sensitivity contract changed",
            )
            metrics = sensitivity.get("metrics")
            _validate_metric_stats(
                metrics, expected_n=len(ROLLOUT_SEEDS), context=candidate
            )
            for metric, stats in metrics.items():
                values = [
                    by_seed[(candidate, None, rollout_seed)]["metrics"][metric]
                    for rollout_seed in ROLLOUT_SEEDS
                ]
                _matches_stats(stats, values, f"{candidate} {metric}")
            n = len(ROLLOUT_SEEDS)
            unit = "sampling_sensitivity"
        if candidate in REPORT_CANDIDATES:
            rows[candidate] = {
                "label": CANDIDATE_LABELS[candidate],
                "metrics": metrics,
                "n": n,
                "unit": unit,
            }
    return rows


def _best_candidates(rows: Mapping[str, Mapping[str, Any]], metric: str) -> set[str]:
    _heading, _decimals, direction, target = METRIC_SPECS[metric]
    values = {
        candidate: float(row["metrics"][metric]["mean"])
        for candidate, row in rows.items()
    }
    if direction == "min":
        best = min(values.values())
        distances = {candidate: value - best for candidate, value in values.items()}
    elif direction == "max":
        best = max(values.values())
        distances = {candidate: best - value for candidate, value in values.items()}
    else:
        distances = {
            candidate: abs(value - float(target)) for candidate, value in values.items()
        }
    best_distance = min(distances.values())
    return {
        candidate
        for candidate, distance in distances.items()
        if math.isclose(distance, best_distance, rel_tol=1e-12, abs_tol=1e-12)
    }


def _confidence_interval(stats: Mapping[str, Any], n: int) -> tuple[float, float]:
    if n < 2:
        raise MultiseedExportError("a confidence interval needs two observations")
    half_width = float(student_t.ppf(0.975, n - 1)) * float(stats["std"]) / math.sqrt(n)
    mean = float(stats["mean"])
    return mean - half_width, mean + half_width


def _interval_decimals(half_width: float, mean_decimals: int) -> int:
    if half_width == 0:
        return mean_decimals
    exponent = math.floor(math.log10(abs(half_width)))
    return max(mean_decimals, 1 - exponent)


def _format_cell(stats: Mapping[str, Any], *, metric: str, best: bool, n: int) -> str:
    _heading, decimals, _direction, _target = METRIC_SPECS[metric]
    low, high = _confidence_interval(stats, n)
    interval_decimals = _interval_decimals((high - low) / 2.0, decimals)
    mean = f"{float(stats['mean']):.{interval_decimals}f}"
    if best:
        mean = rf"\textbf{{{mean}}}"
    return rf"{mean}\,[{low:.{interval_decimals}f}, " rf"{high:.{interval_decimals}f}]"


def _render_panel(
    rows: Mapping[str, Mapping[str, Any]], *, title: str, metrics: tuple[str, ...]
) -> str:
    headers = " & ".join(METRIC_SPECS[metric][0] for metric in metrics)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'c' * len(metrics)}@{{}}}}",
        r"\toprule",
        rf"\multicolumn{{{len(metrics) + 1}}}{{l}}{{\textbf{{{title}}}}} " + r"\\",
        r"\midrule",
        f"Model & {headers} " + r"\\",
        r"\midrule",
    ]
    best_by_metric = {metric: _best_candidates(rows, metric) for metric in metrics}
    for candidate, row in rows.items():
        cells = [
            _format_cell(
                row["metrics"][metric],
                metric=metric,
                best=candidate in best_by_metric[metric],
                n=int(row["n"]),
            )
            for metric in metrics
        ]
        lines.append(f"{row['label']} & {' & '.join(cells)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def _render_table(summary: dict[str, Any]) -> str:
    rows = _validated_rows(summary)
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


def _assert_output_is_safe(path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    golden = (repository / "paper_assets/tables/golden").resolve()
    frozen = (repository / "results/frozen").resolve()
    resolved = path.expanduser().resolve()
    if resolved == golden or golden in resolved.parents:
        raise MultiseedExportError("refusing to overwrite protected golden tables")
    if resolved == frozen or frozen in resolved.parents:
        raise MultiseedExportError("refusing to overwrite frozen evidence")


def render_multiseed_table(summary_path: Path) -> str:
    """Validate the frozen aggregate and render the reader-facing table."""
    return _render_table(_load_summary(Path(summary_path)))


def export_multiseed_table(summary_path: Path, output_path: Path) -> Path:
    """Validate the sole frozen aggregate and render its exact paper table."""
    destination = Path(output_path)
    try:
        _assert_output_is_safe(destination)
    except (OSError, RuntimeError) as error:
        raise MultiseedExportError("multi-seed output path is unusable") from error
    text = render_multiseed_table(summary_path)
    temporary: Path | None = None
    output_error: OSError | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_dir():
            raise IsADirectoryError
        if (
            (destination.exists() or destination.is_symlink())
            and not destination.is_file()
            and not destination.is_symlink()
        ):
            raise OSError
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(destination)
        temporary = None
    except OSError as error:
        output_error = error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                if output_error is None:
                    output_error = error
    if output_error is not None:
        raise MultiseedExportError(
            "multi-seed output path is unusable"
        ) from output_error
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scenarioflow.paper.multiseed", description=__doc__
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        path = export_multiseed_table(args.summary, args.output)
    except MultiseedExportError as error:
        parser.error(str(error))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
