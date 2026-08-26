"""Validate and aggregate a completed public multi-seed benchmark rerun."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .executor import completion_is_valid, sha256_file
from .protocol import (
    CANDIDATE_IDS,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    executable_dag_sha256,
    topological_nodes,
)
from .reporting import (
    METRIC_PATHS,
    TABLE_THRESHOLD_RULES,
    metric_values,
    threshold_status,
)

SUMMARY_SCHEMA = "scenarioflow.multiseed-semantic-rerun.v1"
ARTIFACT_KIND = "semantic_rerun_output"
CLASSICAL_CANDIDATES = (
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
)
LEARNED_CANDIDATES = tuple(
    candidate for candidate in CANDIDATE_IDS if candidate not in CLASSICAL_CANDIDATES
)
PROPER_WINDOW_METRICS = ("CRPS", "Energy Score", "Variogram")
PAIRED_COMPARISONS = (
    ("deepvar_vs_deepvar_support", "deepvar", "deepvar_sni"),
    ("scenarioflow_vs_graph", "scenarioflow", "scenarioflow_graph"),
)
DEFAULT_BLOCK_LENGTH = 30
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260805
DEFAULT_CI_LEVEL = 0.95


class AggregationContractError(ValueError):
    """Raised before output when the completed grid violates its contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregationContractError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise AggregationContractError(f"{label} must contain a JSON object: {path}")
    return value


def _descriptive_stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise AggregationContractError("cannot summarize empty or non-finite values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _student_t_summary(
    values: Iterable[float], *, ci_level: float
) -> dict[str, float | int]:
    result = _descriptive_stats(values)
    count = int(result["n"])
    mean = float(result["mean"])
    if count < 2:
        low = high = mean
    else:
        critical = float(student_t.ppf((1.0 + ci_level) / 2.0, count - 1))
        half_width = critical * float(result["std"]) / math.sqrt(count)
        low, high = mean - half_width, mean + half_width
    result.update(
        {
            "ci_level": float(ci_level),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    )
    return result


def moving_block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> dict[str, Any]:
    """Summarize one chronological paired series with moving blocks."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size < 1 or not np.isfinite(series).all():
        raise AggregationContractError("bootstrap values must be finite and non-empty")
    if block_length < 1 or n_bootstrap < 1:
        raise AggregationContractError(
            "block length and bootstrap count must be positive"
        )
    if not 0.0 < ci_level < 1.0:
        raise AggregationContractError(
            "ci_level must lie strictly between zero and one"
        )
    effective_length = min(int(block_length), int(series.size))
    block_count = int(math.ceil(series.size / effective_length))
    generator = np.random.default_rng(int(seed))
    starts = generator.integers(
        0,
        series.size - effective_length + 1,
        size=(int(n_bootstrap), block_count),
    )
    offsets = np.arange(effective_length, dtype=np.int64)
    draws = series[(starts[:, :, None] + offsets[None, None, :])].reshape(
        int(n_bootstrap), -1
    )[:, : series.size]
    means = draws.mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "n_windows": int(series.size),
        "mean_delta": float(series.mean()),
        "ci_level": float(ci_level),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "block_length": effective_length,
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


def _expected_evaluation_keys() -> set[tuple[str, int | None, int]]:
    expected = {
        (candidate, None, rollout_seed)
        for candidate in CLASSICAL_CANDIDATES
        for rollout_seed in ROLLOUT_SEEDS
    }
    expected.update(
        (candidate, training_seed, rollout_seed)
        for candidate in LEARNED_CANDIDATES
        for training_seed in TRAIN_SEEDS
        for rollout_seed in ROLLOUT_SEEDS
    )
    return expected


def _validate_protocol_manifest(
    path: Path, output_dir: Path
) -> tuple[dict[str, Any], Mapping[str, Any], list[dict[str, Any]], Path]:
    document = _read_json(path, "protocol manifest")
    protocol = document.get("protocol")
    commands = document.get("commands")
    if not isinstance(protocol, Mapping) or not isinstance(commands, list):
        raise AggregationContractError(
            "protocol manifest has no protocol or command graph"
        )
    if tuple(protocol.get("train_seeds", ())) != TRAIN_SEEDS:
        raise AggregationContractError("training seed schedule drifted")
    if tuple(protocol.get("rollout_seeds", ())) != ROLLOUT_SEEDS:
        raise AggregationContractError("rollout seed schedule drifted")
    if tuple(protocol.get("candidate_ids", ())) != CANDIDATE_IDS:
        raise AggregationContractError("candidate schedule drifted")
    if len(commands) != 141 or document.get("command_count") != 141:
        raise AggregationContractError("final protocol must contain 141 commands")
    dag_hash = protocol.get("executable_dag_sha256")
    if not isinstance(dag_hash, str) or dag_hash != executable_dag_sha256(commands):
        raise AggregationContractError(
            "executable protocol hash does not match commands"
        )
    try:
        ordered = topological_nodes(document)
    except (KeyError, TypeError, ValueError) as error:
        raise AggregationContractError("protocol command graph is invalid") from error
    if sum(node.get("action") == "train" for node in ordered) != 21:
        raise AggregationContractError("final protocol must contain 21 training nodes")
    evaluations = [node for node in ordered if node.get("action") == "evaluate"]
    if len(evaluations) != 120:
        raise AggregationContractError("final protocol must contain 120 evaluations")
    actual_keys = {
        (
            str(node.get("candidate")),
            (
                int(node["training_seed"])
                if node.get("training_seed") is not None
                else None
            ),
            int(node["rollout_seed"]),
        )
        for node in evaluations
    }
    if actual_keys != _expected_evaluation_keys() or len(actual_keys) != len(
        evaluations
    ):
        raise AggregationContractError(
            "evaluation seed grid does not match the authority"
        )

    result_value = protocol.get("result_root")
    if not isinstance(result_value, str) or not result_value:
        raise AggregationContractError("protocol has no result root")
    result_root = Path(result_value).expanduser().resolve()
    expected_manifest = result_root / "protocol_manifest.json"
    if path.resolve() != expected_manifest:
        raise AggregationContractError(
            "aggregation requires the executor protocol manifest under result_root"
        )
    if output_dir.resolve() != result_root / "aggregate":
        raise AggregationContractError(
            "semantic aggregate output must be result_root/aggregate"
        )
    return document, protocol, evaluations, result_root


def _artifact_path(node: Mapping[str, Any], filename: str, result_root: Path) -> Path:
    matches = [
        Path(str(value)).expanduser().resolve()
        for value in node.get("resume", {}).get("completion_artifacts", [])
        if Path(str(value)).name == filename
    ]
    if len(matches) != 1:
        raise AggregationContractError(
            f"node {node.get('id')} requires exactly one {filename}"
        )
    path = matches[0]
    if result_root not in path.parents:
        raise AggregationContractError(
            f"node {node.get('id')} artifact escapes result_root: {path}"
        )
    return path


def _candidate_result(document: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    baselines = document.get("baselines")
    if isinstance(baselines, Mapping):
        result = baselines.get(candidate)
        if not isinstance(result, Mapping):
            raise AggregationContractError(
                f"result summary does not contain candidate {candidate}"
            )
        return result
    return document


def _load_window_scores(
    path: Path, *, metrics: Mapping[str, float]
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, KeyError) as error:
        raise AggregationContractError(
            f"invalid window-score artifact: {path}"
        ) from error
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise AggregationContractError(
                f"window-score field {name} contains non-finite values: {path}"
            )
    indices = np.asarray(arrays["window_indices"], dtype=np.int64)
    if indices.size > 1 and not np.all(np.diff(indices) > 0):
        raise AggregationContractError(f"window indices are not chronological: {path}")
    checks = {
        "CRPS": float(np.asarray(arrays["crps_contributions"]).mean()),
        "Energy Score": float(np.asarray(arrays["energy_contributions"]).mean()),
        "Variogram": float(np.asarray(arrays["variogram_h30_contributions"]).mean()),
    }
    for metric, actual in checks.items():
        if not math.isclose(actual, metrics[metric], rel_tol=1e-6, abs_tol=1e-8):
            raise AggregationContractError(
                f"window-score {metric} does not match result JSON: {path}"
            )
    return arrays


def _threshold_assessment(value: float, rule: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(rule["kind"])
    lower = float(rule["lo"]) if "lo" in rule else None
    upper = float(rule["hi"]) if "hi" in rule else None
    if kind == "lower":
        margin = value - float(lower)
    elif kind == "upper":
        margin = float(upper) - value
    elif kind == "band":
        margin = min(value - float(lower), float(upper) - value)
    else:
        raise AggregationContractError(f"unknown threshold kind: {kind}")
    passed = threshold_status(value, rule)
    return {
        "value": value,
        "kind": kind,
        "lower": lower,
        "upper": upper,
        "signed_margin": float(margin),
        "status": "pass" if passed else "fail",
        "passed": passed,
    }


def _load_run(
    node: Mapping[str, Any], payload: Mapping[str, Any], result_root: Path
) -> dict[str, Any]:
    if not completion_is_valid(node, payload):
        raise AggregationContractError(
            f"node {node.get('id')} is not a current executor-validated completion"
        )
    candidate = str(node["candidate"])
    result_path = _artifact_path(node, "results_current_panel.json", result_root)
    window_path = _artifact_path(node, "window_scores.npz", result_root)
    document = _read_json(result_path, "result JSON")
    result = _candidate_result(document, candidate)
    try:
        metrics = metric_values(result)
    except (TypeError, ValueError) as error:
        raise AggregationContractError(
            f"node {node.get('id')} result metric contract is invalid"
        ) from error
    thresholds = {
        name: _threshold_assessment(metrics[name], rule)
        for name, rule in TABLE_THRESHOLD_RULES.items()
    }
    breach_count = sum(not item["passed"] for item in thresholds.values())
    window_scores = _load_window_scores(window_path, metrics=metrics)
    sidecar = Path(str(node["sidecar_manifest"]["path"])).expanduser().resolve()
    return {
        "node_id": str(node["id"]),
        "candidate": candidate,
        "display_name": str(node.get("display_name", candidate)),
        "training_seed": node.get("training_seed"),
        "rollout_seed": int(node["rollout_seed"]),
        "metrics": metrics,
        "thresholds": thresholds,
        "breach_count": int(breach_count),
        "pass_all": breach_count == 0,
        "artifacts": {
            "run_manifest": str(sidecar),
            "result": str(result_path),
            "window_scores": str(window_path),
            "sha256": {
                "run_manifest": sha256_file(sidecar),
                "result": sha256_file(result_path),
                "window_scores": sha256_file(window_path),
            },
        },
        "_window_scores": window_scores,
    }


def _metric_summary(
    runs: Sequence[Mapping[str, Any]], *, ci_level: float | None
) -> dict[str, Any]:
    summarize = (
        (lambda values: _student_t_summary(values, ci_level=ci_level))
        if ci_level is not None
        else _descriptive_stats
    )
    return {
        name: summarize(run["metrics"][name] for run in runs)
        for name, _path in METRIC_PATHS
    }


def _pass_profile(
    runs: Sequence[Mapping[str, Any]], *, learned: bool
) -> dict[str, Any]:
    passing = sum(bool(run["pass_all"]) for run in runs)
    result: dict[str, Any] = {
        "n_runs": len(runs),
        "n_pass_all": passing,
        "pass_all_rate": float(passing / len(runs)),
        "breach_count": _descriptive_stats(run["breach_count"] for run in runs),
    }
    if learned:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for run in runs:
            grouped[int(run["training_seed"])].append(run)
        passing_seeds = [
            seed
            for seed, seed_runs in grouped.items()
            if all(run["pass_all"] for run in seed_runs)
        ]
        result.update(
            {
                "n_training_seeds": len(grouped),
                "n_training_seeds_passing_every_rollout": len(passing_seeds),
                "training_seeds_passing_every_rollout": sorted(passing_seeds),
            }
        )
    return result


def _candidate_summary(
    candidate: str, runs: Sequence[Mapping[str, Any]], *, ci_level: float
) -> dict[str, Any]:
    ordered = sorted(
        runs,
        key=lambda run: (
            -1 if run["training_seed"] is None else int(run["training_seed"]),
            int(run["rollout_seed"]),
        ),
    )
    learned = candidate in LEARNED_CANDIDATES
    common: dict[str, Any] = {
        "candidate": candidate,
        "display_name": ordered[0]["display_name"],
        "pass_profile": _pass_profile(ordered, learned=learned),
    }
    if not learned:
        if len(ordered) != len(ROLLOUT_SEEDS):
            raise AggregationContractError(f"{candidate} does not have five seed runs")
        common.update(
            {
                "variation_type": "rollout_sampling_sensitivity",
                "sampling_sensitivity": {
                    "n_rollout_seeds": len(ordered),
                    "rollout_seeds": [run["rollout_seed"] for run in ordered],
                    "unit": "per_sampling_seed",
                    "metrics": _metric_summary(ordered, ci_level=ci_level),
                },
            }
        )
        return common

    by_training: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for run in ordered:
        by_training[int(run["training_seed"])].append(run)
    if set(by_training) != set(TRAIN_SEEDS) or any(
        {int(run["rollout_seed"]) for run in seed_runs} != set(ROLLOUT_SEEDS)
        for seed_runs in by_training.values()
    ):
        raise AggregationContractError(
            f"{candidate} does not have the complete 3x5 grid"
        )
    rollout_summaries = []
    training_metric_means: dict[str, list[float]] = defaultdict(list)
    for training_seed in TRAIN_SEEDS:
        seed_runs = sorted(
            by_training[training_seed], key=lambda run: int(run["rollout_seed"])
        )
        metrics = _metric_summary(seed_runs, ci_level=None)
        rollout_summaries.append(
            {
                "training_seed": training_seed,
                "n_rollout_seeds": len(seed_runs),
                "rollout_seeds": [run["rollout_seed"] for run in seed_runs],
                "unit": "per_sampling_seed_within_training_model",
                "metrics": metrics,
                "n_pass_all": sum(bool(run["pass_all"]) for run in seed_runs),
                "pass_all_rate": float(
                    np.mean([bool(run["pass_all"]) for run in seed_runs])
                ),
            }
        )
        for metric, _path in METRIC_PATHS:
            training_metric_means[metric].append(float(metrics[metric]["mean"]))
    common.update(
        {
            "variation_type": "training_then_rollout_sensitivity",
            "rollout_sensitivity_by_training_seed": rollout_summaries,
            "training_seed_sensitivity": {
                "n_training_seeds": len(TRAIN_SEEDS),
                "training_seeds": list(TRAIN_SEEDS),
                "unit": "per_training_seed_rollout_mean",
                "metrics": {
                    metric: _student_t_summary(values, ci_level=ci_level)
                    for metric, values in training_metric_means.items()
                },
            },
        }
    )
    return common


def _window_series(run: Mapping[str, Any], metric: str) -> np.ndarray:
    artifact = run["_window_scores"]
    if metric == "CRPS":
        return np.asarray(artifact["crps_contributions"], dtype=np.float64).mean(
            axis=(1, 2)
        )
    if metric == "Energy Score":
        return np.asarray(artifact["energy_contributions"], dtype=np.float64).mean(
            axis=1
        )
    if metric == "Variogram":
        return np.asarray(artifact["variogram_h30_contributions"], dtype=np.float64)
    raise AggregationContractError(f"no per-window evidence for {metric}")


def _paired_comparison(
    *,
    name: str,
    left_candidate: str,
    right_candidate: str,
    runs_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    block_length: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    ci_level: float,
) -> dict[str, Any]:
    def key(run: Mapping[str, Any]) -> tuple[int, int]:
        return int(run["training_seed"]), int(run["rollout_seed"])

    left = {key(run): run for run in runs_by_candidate[left_candidate]}
    right = {key(run): run for run in runs_by_candidate[right_candidate]}
    expected = {
        (training_seed, rollout_seed)
        for training_seed in TRAIN_SEEDS
        for rollout_seed in ROLLOUT_SEEDS
    }
    if set(left) != expected or set(right) != expected:
        raise AggregationContractError(f"{name} does not have all paired seed runs")
    ordered_keys = sorted(expected)
    run_deltas = {
        metric: _descriptive_stats(
            right[seed_pair]["metrics"][metric] - left[seed_pair]["metrics"][metric]
            for seed_pair in ordered_keys
        )
        for metric, _path in METRIC_PATHS
    }
    window_uncertainty = {}
    for metric in PROPER_WINDOW_METRICS:
        paired_window_deltas = []
        reference_indices: np.ndarray | None = None
        for seed_pair in ordered_keys:
            left_indices = np.asarray(
                left[seed_pair]["_window_scores"]["window_indices"]
            )
            right_indices = np.asarray(
                right[seed_pair]["_window_scores"]["window_indices"]
            )
            if not np.array_equal(left_indices, right_indices):
                raise AggregationContractError(
                    f"{name} window indices differ for seed pair {seed_pair}"
                )
            if reference_indices is None:
                reference_indices = left_indices
            elif not np.array_equal(reference_indices, left_indices):
                raise AggregationContractError(
                    f"{name} window schedule differs across seed pairs"
                )
            paired_window_deltas.append(
                _window_series(right[seed_pair], metric)
                - _window_series(left[seed_pair], metric)
            )
        seed_averaged = np.mean(np.stack(paired_window_deltas, axis=0), axis=0)
        window_uncertainty[metric] = moving_block_bootstrap_mean(
            seed_averaged,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            ci_level=ci_level,
        )
    return {
        "name": name,
        "left_candidate": left_candidate,
        "right_candidate": right_candidate,
        "delta_orientation": "right_minus_left",
        "n_seed_pairs": len(ordered_keys),
        "paired_training_seeds": list(TRAIN_SEEDS),
        "paired_rollout_seeds": list(ROLLOUT_SEEDS),
        "run_level_metric_deltas": run_deltas,
        "run_level_breach_count_delta": _descriptive_stats(
            right[seed_pair]["breach_count"] - left[seed_pair]["breach_count"]
            for seed_pair in ordered_keys
        ),
        "window_uncertainty": window_uncertainty,
    }


def _serializable_run(run: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _per_run_metric_rows(runs: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for run in runs:
        for metric, value in run["metrics"].items():
            yield {
                "node_id": run["node_id"],
                "candidate": run["candidate"],
                "training_seed": (
                    "" if run["training_seed"] is None else run["training_seed"]
                ),
                "rollout_seed": run["rollout_seed"],
                "metric": metric,
                "value": value,
                "run_breach_count": run["breach_count"],
                "run_pass_all": run["pass_all"],
            }


def _threshold_rows(runs: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for run in runs:
        for metric, entry in run["thresholds"].items():
            yield {
                "node_id": run["node_id"],
                "candidate": run["candidate"],
                "training_seed": (
                    "" if run["training_seed"] is None else run["training_seed"]
                ),
                "rollout_seed": run["rollout_seed"],
                "metric": metric,
                "value": entry["value"],
                "rule_kind": entry["kind"],
                "lower": "" if entry["lower"] is None else entry["lower"],
                "upper": "" if entry["upper"] is None else entry["upper"],
                "signed_margin": entry["signed_margin"],
                "status": entry["status"],
                "run_breach_count": run["breach_count"],
                "run_pass_all": run["pass_all"],
            }


def _metric_summary_rows(summaries: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for candidate in CANDIDATE_IDS:
        summary = summaries[candidate]
        if summary["variation_type"] == "rollout_sampling_sensitivity":
            sources = [
                (
                    "rollout_sampling",
                    "",
                    "sampling_seed",
                    summary["sampling_sensitivity"]["metrics"],
                )
            ]
        else:
            sources = [
                (
                    "within_training_rollout",
                    item["training_seed"],
                    "sampling_seed_within_training_model",
                    item["metrics"],
                )
                for item in summary["rollout_sensitivity_by_training_seed"]
            ]
            sources.append(
                (
                    "across_training_seed_means",
                    "",
                    "training_seed_rollout_mean",
                    summary["training_seed_sensitivity"]["metrics"],
                )
            )
        for variation, training_seed, independent_unit, metrics in sources:
            for metric, stats in metrics.items():
                yield {
                    "candidate": candidate,
                    "variation": variation,
                    "training_seed": training_seed,
                    "independent_unit": independent_unit,
                    "metric": metric,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "ci_level": stats.get("ci_level", ""),
                    "ci_low": stats.get("ci_low", ""),
                    "ci_high": stats.get("ci_high", ""),
                }


def _paired_rows(comparisons: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for name, comparison in comparisons.items():
        for metric, stats in comparison["run_level_metric_deltas"].items():
            yield {
                "comparison": name,
                "level": "paired_run_seed_pairs_descriptive",
                "metric": metric,
                "mean_delta": stats["mean"],
                "std_delta": stats["std"],
                "min_delta": stats["min"],
                "max_delta": stats["max"],
                "ci_low": "",
                "ci_high": "",
                "n": stats["n"],
                "block_length": "",
                "bootstrap_seed": "",
            }
        for metric, interval in comparison["window_uncertainty"].items():
            yield {
                "comparison": name,
                "level": "paired_window_bootstrap",
                "metric": metric,
                "mean_delta": interval["mean_delta"],
                "std_delta": "",
                "min_delta": "",
                "max_delta": "",
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "n": interval["n_windows"],
                "block_length": interval["block_length"],
                "bootstrap_seed": interval["bootstrap_seed"],
            }


def _candidate_inference_stats(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary["variation_type"] == "rollout_sampling_sensitivity":
        return summary["sampling_sensitivity"]["metrics"]["CRPS"]
    return summary["training_seed_sensitivity"]["metrics"]["CRPS"]


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ScenarioFlow final multi-seed semantic rerun aggregate",
        "",
        "This is a semantic rerun output, not a historical byte-identical artifact.",
        "Thresholds are evaluated per run before averaging. Learned-model intervals",
        "use three trained-model means; classical intervals use five sampling seeds.",
        "",
        "## Candidate sensitivity",
        "",
        "| Candidate | Independent units | CRPS mean | 95% Student-t interval | Pass rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATE_IDS:
        candidate_summary = summary["candidate_summaries"][candidate]
        stats = _candidate_inference_stats(candidate_summary)
        profile = candidate_summary["pass_profile"]
        lines.append(
            f"| {candidate_summary['display_name']} | {stats['n']} | "
            f"{stats['mean']:.6g} | [{stats['ci_low']:.6g}, {stats['ci_high']:.6g}] | "
            f"{profile['pass_all_rate']:.1%} |"
        )
    lines.extend(["", "## Paired controls", ""])
    for name, comparison in summary["paired_comparisons"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "Deltas are right minus left and use paired window evidence.",
                "",
                "| Proper score | Mean delta | 95% moving-block interval |",
                "| --- | ---: | ---: |",
            ]
        )
        for metric in PROPER_WINDOW_METRICS:
            interval = comparison["window_uncertainty"][metric]
            lines.append(
                f"| {metric} | {interval['mean_delta']:.6g} | "
                f"[{interval['ci_low']:.6g}, {interval['ci_high']:.6g}] |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(output_dir: Path, summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "summary.json", summary)
    markdown = output_dir / "summary.md"
    temporary = markdown.with_name(f".{markdown.name}.{os.getpid()}.tmp")
    temporary.write_text(_render_markdown(summary), encoding="utf-8")
    temporary.replace(markdown)
    _write_csv_atomic(
        output_dir / "per-run-metrics.csv",
        (
            "node_id",
            "candidate",
            "training_seed",
            "rollout_seed",
            "metric",
            "value",
            "run_breach_count",
            "run_pass_all",
        ),
        _per_run_metric_rows(summary["per_runs"]),
    )
    _write_csv_atomic(
        output_dir / "per-run-thresholds.csv",
        (
            "node_id",
            "candidate",
            "training_seed",
            "rollout_seed",
            "metric",
            "value",
            "rule_kind",
            "lower",
            "upper",
            "signed_margin",
            "status",
            "run_breach_count",
            "run_pass_all",
        ),
        _threshold_rows(summary["per_runs"]),
    )
    _write_csv_atomic(
        output_dir / "metric-summaries.csv",
        (
            "candidate",
            "variation",
            "training_seed",
            "independent_unit",
            "metric",
            "n",
            "mean",
            "std",
            "min",
            "max",
            "ci_level",
            "ci_low",
            "ci_high",
        ),
        _metric_summary_rows(summary["candidate_summaries"]),
    )
    _write_csv_atomic(
        output_dir / "paired-comparisons.csv",
        (
            "comparison",
            "level",
            "metric",
            "mean_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "ci_low",
            "ci_high",
            "n",
            "block_length",
            "bootstrap_seed",
        ),
        _paired_rows(summary["paired_comparisons"]),
    )


def aggregate_study(
    *,
    protocol_manifest: str | Path,
    output_dir: str | Path,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> dict[str, Any]:
    """Validate the complete grid and write stable semantic-rerun artifacts."""
    protocol_path = Path(protocol_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if block_length != DEFAULT_BLOCK_LENGTH:
        raise AggregationContractError("the frozen block length is 30 windows")
    if n_bootstrap < 1:
        raise AggregationContractError("n_bootstrap must be positive")
    if not 0.0 < ci_level < 1.0:
        raise AggregationContractError(
            "ci_level must lie strictly between zero and one"
        )
    payload, protocol, nodes, result_root = _validate_protocol_manifest(
        protocol_path, output
    )
    runs = [
        _load_run(node, payload, result_root)
        for node in sorted(nodes, key=lambda item: str(item["id"]))
    ]
    reference_indices = np.asarray(runs[0]["_window_scores"]["window_indices"])
    reference_names = np.asarray(runs[0]["_window_scores"]["variable_names"])
    for run in runs[1:]:
        if not np.array_equal(
            reference_indices, run["_window_scores"]["window_indices"]
        ):
            raise AggregationContractError(
                "chronological window schedule differs across runs"
            )
        if not np.array_equal(reference_names, run["_window_scores"]["variable_names"]):
            raise AggregationContractError("market-variable order differs across runs")

    runs_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_candidate[run["candidate"]].append(run)
    summaries = {
        candidate: _candidate_summary(
            candidate, runs_by_candidate[candidate], ci_level=ci_level
        )
        for candidate in CANDIDATE_IDS
    }
    comparisons = {
        name: _paired_comparison(
            name=name,
            left_candidate=left,
            right_candidate=right,
            runs_by_candidate=runs_by_candidate,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
        )
        for name, left, right in PAIRED_COMPARISONS
    }
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "historical_byte_identity_claimed": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_manifest": str(protocol_path),
        "protocol_dag_sha256": protocol["executable_dag_sha256"],
        "contract": dict(protocol.get("contract", {})),
        "seed_schedule": {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "methodology_contract": {
            "learned_model_independent_unit": "training_seed_mean_over_five_sampling_seeds",
            "learned_model_unit_count": 3,
            "classical_model_independent_unit": "sampling_seed",
            "classical_model_unit_count": 5,
            "confidence_interval": "two_sided_95_percent_student_t",
            "paired_window_method": "moving_block_bootstrap_of_seed_averaged_paired_differences",
        },
        "bootstrap_contract": {
            "method": "paired_moving_block_bootstrap_of_seed_averaged_window_differences",
            "block_length": block_length,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "ci_level": ci_level,
        },
        "per_runs": [_serializable_run(run) for run in runs],
        "candidate_summaries": summaries,
        "paired_comparisons": comparisons,
    }
    _write_outputs(output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL)
    args = parser.parse_args(argv)
    aggregate_study(
        protocol_manifest=args.protocol_manifest,
        output_dir=args.output_dir,
        block_length=args.block_length,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
    )


if __name__ == "__main__":
    main()
