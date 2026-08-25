#!/usr/bin/env python
"""Fail-closed statistical aggregation for the frozen FMTS seed study.

The aggregator is deliberately read-only with respect to experiment outputs. It
accepts the archived executable DAG, verifies every completed evaluation
sidecar and its hashed result artifacts, and only then writes paper-review
summaries to a separate aggregate directory.

Three sources of variation remain separate:

* each result is thresholded before any averaging;
* rollout-seed sensitivity is summarized within a fitted checkpoint;
* training-seed sensitivity is summarized across checkpoint-level means; and
* chronological window uncertainty is reported with a paired 30-window
  moving-block bootstrap over per-window proper-score differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis_code.fmts_reproduction_report import METRIC_PATHS, TABLE_THRESHOLD_RULES
from analysis_code.window_score_artifact import SCHEMA_VERSION as WINDOW_SCHEMA_VERSION


RUN_MANIFEST_SCHEMA = "fmts.node_run.v1"
STATUS_SCHEMA = "fmts.node_status.v1"
SUMMARY_SCHEMA = "fmts.multiseed_aggregate.v1"
TRAIN_SEEDS = (1001, 1002, 1003)
ROLLOUT_SEEDS = (2001, 2002, 2003, 2004, 2005)
LEARNED_CANDIDATES = (
    "scenarioflow",
    "scenarioflow_graph",
    "deepvar",
    "timegrad",
    "path_diffusion",
    "deepvar_sni",
)
CLASSICAL_CANDIDATES = (
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
)
EXPECTED_CANDIDATES = frozenset((*LEARNED_CANDIDATES, *CLASSICAL_CANDIDATES))
METRIC_NAMES = tuple(name for name, _ in METRIC_PATHS)
PROPER_WINDOW_METRICS = ("CRPS", "Energy Score", "Variogram")
DEFAULT_BOOTSTRAP_SEED = 20260805
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_BLOCK_LENGTH = 30
DEFAULT_CI_LEVEL = 0.95
_HASH_LENGTH = 64
_MISSING = object()


class AggregationContractError(ValueError):
    """Raised before output when an input violates the frozen study contract."""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationContractError(f"missing required artifact: {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregationContractError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AggregationContractError(f"{label} must contain a JSON object: {path}")
    return value


def _require_hash(value: object, label: str) -> str:
    text = str(value)
    if len(text) != _HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise AggregationContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> object:
    value: object = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationContractError(f"{label} is missing or non-numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AggregationContractError(f"{label} is non-finite")
    return number


def _candidate_result(document: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    baselines = document.get("baselines")
    if isinstance(baselines, Mapping):
        result = baselines.get(candidate)
        if not isinstance(result, Mapping):
            raise AggregationContractError(
                f"result summary does not contain candidate {candidate!r}"
            )
        return result
    return document


def _metric_values(result: Mapping[str, Any], candidate: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, path in METRIC_PATHS:
        values[name] = _finite_number(
            _path_value(result, path), f"{candidate} metric {name}"
        )
    return values


def _threshold_assessment(value: float, rule: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(rule.get("kind"))
    lower = float(rule["lo"]) if "lo" in rule else None
    upper = float(rule["hi"]) if "hi" in rule else None
    if kind == "lower" and lower is not None:
        margin = value - lower
    elif kind == "upper" and upper is not None:
        margin = upper - value
    elif kind == "band" and lower is not None and upper is not None:
        margin = min(value - lower, upper - value)
    else:
        raise AggregationContractError(f"unsupported threshold rule: {rule!r}")
    passed = margin >= 0.0
    return {
        "value": value,
        "kind": kind,
        "lower": lower,
        "upper": upper,
        "signed_margin": float(margin),
        "status": "pass" if passed else "fail",
        "passed": passed,
    }


def _threshold_profile(metrics: Mapping[str, float]) -> tuple[dict[str, Any], int]:
    profile = {
        name: _threshold_assessment(metrics[name], rule)
        for name, rule in TABLE_THRESHOLD_RULES.items()
    }
    return profile, sum(not entry["passed"] for entry in profile.values())


def _summary_stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise AggregationContractError("cannot summarize an empty or non-finite series")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _verify_artifact_hash(
    artifact_hashes: Mapping[str, Any], path: Path, label: str
) -> str:
    resolved = path.resolve()
    expected = artifact_hashes.get(str(resolved))
    if expected is None:
        raise AggregationContractError(
            f"completed status omits {label} hash: {resolved}"
        )
    expected_hash = _require_hash(expected, f"{label} hash")
    actual_hash = _file_hash(resolved)
    if actual_hash != expected_hash:
        raise AggregationContractError(
            f"SHA-256 mismatch for {label}: {resolved}; "
            f"status={expected_hash}, actual={actual_hash}"
        )
    return actual_hash


def _validate_contract(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    contract = protocol.get("contract")
    if not isinstance(contract, dict):
        raise AggregationContractError("protocol evaluation contract is missing")
    expected_scalars = {
        "mode": "current_realvix_39_state_panel",
        "n_market_variables": 39,
        "n_evaluation_windows": 441,
        "n_scenarios_per_window": 64,
        "history_length": 30,
        "future_length": 30,
    }
    for key, expected in expected_scalars.items():
        if contract.get(key) != expected:
            raise AggregationContractError(
                f"protocol contract field {key!r} must equal {expected!r}"
            )
    threshold_hash = _require_hash(
        contract.get("threshold_contract_hash"), "threshold contract hash"
    )
    if threshold_hash != _canonical_hash(TABLE_THRESHOLD_RULES):
        raise AggregationContractError(
            "threshold contract hash does not match the fixed reporting rules"
        )
    _require_hash(contract.get("state_specification_hash"), "state specification hash")
    dag_hash = _require_hash(
        protocol.get("executable_dag_sha256"), "executable protocol DAG hash"
    )
    return contract, dag_hash


def _expected_eval_keys() -> set[tuple[str, int | None, int]]:
    keys: set[tuple[str, int | None, int]] = set()
    for candidate in CLASSICAL_CANDIDATES:
        keys.update((candidate, None, rollout) for rollout in ROLLOUT_SEEDS)
    for candidate in LEARNED_CANDIDATES:
        keys.update(
            (candidate, training, rollout)
            for training in TRAIN_SEEDS
            for rollout in ROLLOUT_SEEDS
        )
    return keys


def _validate_protocol_schedule(
    document: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if tuple(protocol.get("train_seeds", ())) != TRAIN_SEEDS:
        raise AggregationContractError(
            f"training seed schedule must equal {list(TRAIN_SEEDS)}"
        )
    if tuple(protocol.get("rollout_seeds", ())) != ROLLOUT_SEEDS:
        raise AggregationContractError(
            f"rollout seed schedule must equal {list(ROLLOUT_SEEDS)}"
        )
    candidates = protocol.get("candidate_ids")
    if not isinstance(candidates, list) or set(candidates) != EXPECTED_CANDIDATES:
        raise AggregationContractError(
            "candidate schedule does not match the 12-model study"
        )
    commands = document.get("commands")
    if not isinstance(commands, list):
        raise AggregationContractError("protocol command DAG is missing")
    eval_nodes = [
        node
        for node in commands
        if isinstance(node, Mapping) and node.get("action") == "evaluate"
    ]
    keys: list[tuple[str, int | None, int]] = []
    for node in eval_nodes:
        candidate = str(node.get("candidate"))
        training = node.get("training_seed")
        rollout = node.get("rollout_seed")
        if training is not None and (
            isinstance(training, bool) or not isinstance(training, int)
        ):
            raise AggregationContractError(
                f"invalid training seed in node {node.get('id')}"
            )
        if isinstance(rollout, bool) or not isinstance(rollout, int):
            raise AggregationContractError(
                f"missing rollout seed in node {node.get('id')}"
            )
        keys.append((candidate, training, rollout))
    expected = _expected_eval_keys()
    if (
        len(keys) != len(expected)
        or set(keys) != expected
        or len(keys) != len(set(keys))
    ):
        raise AggregationContractError(
            "evaluation nodes do not match the frozen candidate and seed schedules"
        )
    return eval_nodes


def _completion_paths(node: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    destination = Path(str(node.get("destination", ""))).resolve()
    manifest_metadata = node.get("sidecar_manifest")
    if not isinstance(manifest_metadata, Mapping) or not manifest_metadata.get("path"):
        raise AggregationContractError(
            f"node {node.get('id')} has no run_manifest path"
        )
    manifest = Path(str(manifest_metadata["path"])).resolve()
    completion = _path_value(node, ("resume", "completion_artifacts"))
    if not isinstance(completion, list):
        raise AggregationContractError(
            f"node {node.get('id')} has no completion artifacts"
        )
    result_candidates = [
        Path(str(path)).resolve()
        for path in completion
        if Path(str(path)).name == "results_current_panel.json"
    ]
    window_candidates = [
        Path(str(path)).resolve()
        for path in completion
        if Path(str(path)).name == "window_scores.npz"
    ]
    if len(result_candidates) != 1 or len(window_candidates) != 1:
        raise AggregationContractError(
            f"node {node.get('id')} must declare one result and one window-score artifact"
        )
    for path in (manifest, result_candidates[0], window_candidates[0]):
        if destination != path.parent and destination not in path.parents:
            raise AggregationContractError(
                f"node {node.get('id')} artifact escapes its destination: {path}"
            )
    return (
        manifest,
        destination / "status.json",
        result_candidates[0],
        window_candidates[0],
    )


def _validate_manifest_and_status(
    *,
    node: Mapping[str, Any],
    manifest_path: Path,
    status_path: Path,
    result_path: Path,
    window_path: Path,
    contract: Mapping[str, Any],
    dag_hash: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    for label, path in (
        ("run_manifest.json", manifest_path),
        ("status.json", status_path),
        ("results_current_panel.json", result_path),
        ("window_scores.npz", window_path),
    ):
        if not path.is_file():
            raise AggregationContractError(
                f"missing required artifact: {label}: {path}"
            )
    manifest = _read_json(manifest_path, "run manifest")
    status = _read_json(status_path, "run status")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise AggregationContractError(f"run manifest schema mismatch: {manifest_path}")
    if status.get("schema_version") != STATUS_SCHEMA:
        raise AggregationContractError(f"run status schema mismatch: {status_path}")
    if status.get("state") != "completed" or status.get("returncode") != 0:
        raise AggregationContractError(
            f"incomplete node {node.get('id')}: {status.get('state')}"
        )
    exact_fields = (
        ("node_id", node.get("id")),
        ("candidate", node.get("candidate")),
        ("action", "evaluate"),
        ("rollout_seed", node.get("rollout_seed")),
    )
    for field, expected in exact_fields:
        if manifest.get(field) != expected:
            raise AggregationContractError(
                f"run manifest {field.replace('_', ' ')} mismatch for node {node.get('id')}"
            )
    expected_training = node.get("training_seed", _MISSING)
    actual_training = manifest.get("training_seed", _MISSING)
    if actual_training != expected_training:
        raise AggregationContractError(
            f"run manifest training seed mismatch for node {node.get('id')}"
        )
    if manifest.get("contract") != contract:
        raise AggregationContractError(
            f"run manifest contract mismatch for node {node.get('id')}"
        )
    for payload, label in ((manifest, "run manifest"), (status, "run status")):
        if payload.get("protocol_dag_sha256") != dag_hash:
            raise AggregationContractError(f"{label} protocol DAG hash mismatch")
    for digest_field in ("command_sha256", "configuration_sha256"):
        manifest_digest = _require_hash(
            manifest.get(digest_field), f"manifest {digest_field}"
        )
        if node.get(digest_field) != manifest_digest:
            label = digest_field.removesuffix("_sha256").replace("_", " ")
            raise AggregationContractError(
                f"run manifest {label} hash does not match the executable DAG"
            )
        if status.get(digest_field) != manifest_digest:
            raise AggregationContractError(f"status {digest_field} mismatch")
    if status.get("node_id") != node.get("id"):
        raise AggregationContractError("status node ID mismatch")
    artifact_hashes = status.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise AggregationContractError("completed status has no artifact hashes")
    verified = {
        "run_manifest": _verify_artifact_hash(
            artifact_hashes, manifest_path, "run manifest"
        ),
        "result": _verify_artifact_hash(artifact_hashes, result_path, "result"),
        "window_scores": _verify_artifact_hash(
            artifact_hashes, window_path, "window scores"
        ),
    }
    return manifest, verified


def _validate_result_contract(
    result: Mapping[str, Any], contract: Mapping[str, Any], candidate: str
) -> None:
    if result.get("mode") != contract["mode"]:
        raise AggregationContractError(f"{candidate} result panel mode mismatch")
    if result.get("n_windows") != contract["n_evaluation_windows"]:
        raise AggregationContractError(f"{candidate} result window count mismatch")
    if result.get("n_samples") != contract["n_scenarios_per_window"]:
        raise AggregationContractError(f"{candidate} result scenario count mismatch")
    state_specs = result.get("state_specs")
    if (
        not isinstance(state_specs, list)
        or len(state_specs) != contract["n_market_variables"]
    ):
        raise AggregationContractError(
            f"{candidate} result state specification mismatch"
        )
    if _canonical_hash(state_specs) != contract["state_specification_hash"]:
        raise AggregationContractError(
            f"{candidate} result state specification hash mismatch"
        )
    joint = result.get("joint_panel_quality")
    if not isinstance(joint, Mapping):
        raise AggregationContractError(
            f"{candidate} result joint-panel contract is missing"
        )
    if joint.get("future_len") != contract["future_length"]:
        raise AggregationContractError(f"{candidate} result future horizon mismatch")


def _load_window_scores(
    path: Path,
    *,
    contract: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, KeyError) as error:
        raise AggregationContractError(
            f"invalid window-score artifact {path}: {error}"
        ) from error
    expected_shapes = {
        "window_indices": (441,),
        "horizons": (4,),
        "variable_names": (39,),
        "crps_contributions": (441, 4, 39),
        "energy_contributions": (441, 4),
        "variogram_h30_contributions": (441,),
        "coverage_levels": (4,),
        "coverage_by_window": (441, 4),
        "iv_history_activity": (441,),
        "iv_deck_width": (441,),
        "full_panel_history_activity": (441,),
        "full_panel_deck_width": (441,),
    }
    schema = arrays.get("schema_version")
    if (
        schema is None
        or np.asarray(schema).shape != ()
        or str(np.asarray(schema).item()) != WINDOW_SCHEMA_VERSION
    ):
        raise AggregationContractError(f"window-score schema mismatch: {path}")
    for name, expected in expected_shapes.items():
        value = arrays.get(name)
        if value is None or np.asarray(value).shape != expected:
            actual = None if value is None else np.asarray(value).shape
            raise AggregationContractError(
                f"{name} shape mismatch in {path}: expected {expected}, got {actual}"
            )
        if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all():
            raise AggregationContractError(f"{name} contains non-finite values: {path}")
    if contract["n_evaluation_windows"] != 441 or contract["n_market_variables"] != 39:
        raise AggregationContractError(
            "window-score shape contract is not the frozen 441x39 panel"
        )
    if not np.array_equal(arrays["horizons"], np.asarray([1, 7, 14, 30])):
        raise AggregationContractError(f"window-score horizons mismatch: {path}")
    if not np.array_equal(arrays["coverage_levels"], np.asarray([0.5, 0.8, 0.9, 0.95])):
        raise AggregationContractError(f"window-score coverage levels mismatch: {path}")
    indices = np.asarray(arrays["window_indices"], dtype=np.int64)
    if not np.all(np.diff(indices) > 0):
        raise AggregationContractError(f"window indices are not chronological: {path}")
    checks = {
        "CRPS": float(np.asarray(arrays["crps_contributions"]).mean()),
        "Energy Score": float(np.asarray(arrays["energy_contributions"]).mean()),
        "Variogram": float(np.asarray(arrays["variogram_h30_contributions"]).mean()),
    }
    for name, actual in checks.items():
        if not math.isclose(actual, metrics[name], rel_tol=1e-6, abs_tol=1e-8):
            raise AggregationContractError(
                f"window-score {name} aggregate does not match result JSON in {path}"
            )
    return arrays


def _load_run(
    *, node: Mapping[str, Any], contract: Mapping[str, Any], dag_hash: str
) -> dict[str, Any]:
    manifest_path, status_path, result_path, window_path = _completion_paths(node)
    manifest, verified_hashes = _validate_manifest_and_status(
        node=node,
        manifest_path=manifest_path,
        status_path=status_path,
        result_path=result_path,
        window_path=window_path,
        contract=contract,
        dag_hash=dag_hash,
    )
    document = _read_json(result_path, "result JSON")
    candidate = str(node["candidate"])
    result = _candidate_result(document, candidate)
    _validate_result_contract(result, contract, candidate)
    metrics = _metric_values(result, candidate)
    thresholds, breach_count = _threshold_profile(metrics)
    window_scores = _load_window_scores(window_path, contract=contract, metrics=metrics)
    return {
        "node_id": str(node["id"]),
        "candidate": candidate,
        "display_name": str(
            manifest.get("display_name", node.get("display_name", candidate))
        ),
        "training_seed": node.get("training_seed"),
        "rollout_seed": int(node["rollout_seed"]),
        "metrics": metrics,
        "thresholds": thresholds,
        "breach_count": int(breach_count),
        "pass_all": breach_count == 0,
        "artifacts": {
            "run_manifest": str(manifest_path),
            "result": str(result_path),
            "window_scores": str(window_path),
            "sha256": verified_hashes,
        },
        "_window_scores": window_scores,
    }


def _metrics_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: _summary_stats(run["metrics"][name] for run in runs)
        for name in METRIC_NAMES
    }


def _pass_profile(
    runs: Sequence[Mapping[str, Any]], *, learned: bool
) -> dict[str, Any]:
    n_pass = sum(bool(run["pass_all"]) for run in runs)
    result: dict[str, Any] = {
        "n_runs": len(runs),
        "n_pass_all": n_pass,
        "pass_all_rate": float(n_pass / len(runs)),
        "breach_count": _summary_stats(run["breach_count"] for run in runs),
    }
    if learned:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for run in runs:
            grouped[int(run["training_seed"])].append(run)
        passing = [
            seed
            for seed, seed_runs in grouped.items()
            if all(run["pass_all"] for run in seed_runs)
        ]
        result.update(
            {
                "n_training_seeds": len(grouped),
                "n_training_seeds_passing_every_rollout": len(passing),
                "training_seeds_passing_every_rollout": sorted(passing),
            }
        )
    return result


def _candidate_summary(
    candidate: str, runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(
        runs, key=lambda run: (run["training_seed"] or -1, run["rollout_seed"])
    )
    learned = candidate in LEARNED_CANDIDATES
    common = {
        "candidate": candidate,
        "display_name": ordered[0]["display_name"],
        "pass_profile": _pass_profile(ordered, learned=learned),
    }
    if not learned:
        common.update(
            {
                "variation_type": "rollout_sampling_sensitivity",
                "sampling_sensitivity": {
                    "n_rollout_seeds": len(ordered),
                    "rollout_seeds": [run["rollout_seed"] for run in ordered],
                    "metrics": _metrics_summary(ordered),
                },
            }
        )
        return common

    by_training: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for run in ordered:
        by_training[int(run["training_seed"])].append(run)
    rollout_summaries: list[dict[str, Any]] = []
    training_metric_means: dict[str, list[float]] = defaultdict(list)
    for training_seed in sorted(by_training):
        seed_runs = sorted(
            by_training[training_seed], key=lambda run: run["rollout_seed"]
        )
        metric_summary = _metrics_summary(seed_runs)
        for metric, stats in metric_summary.items():
            training_metric_means[metric].append(float(stats["mean"]))
        rollout_summaries.append(
            {
                "training_seed": training_seed,
                "n_rollout_seeds": len(seed_runs),
                "rollout_seeds": [run["rollout_seed"] for run in seed_runs],
                "metrics": metric_summary,
                "n_pass_all": sum(bool(run["pass_all"]) for run in seed_runs),
                "pass_all_rate": float(
                    sum(bool(run["pass_all"]) for run in seed_runs) / len(seed_runs)
                ),
            }
        )
    common.update(
        {
            "variation_type": "training_then_rollout_sensitivity",
            "rollout_sensitivity_by_training_seed": rollout_summaries,
            "training_seed_sensitivity": {
                "n_training_seeds": len(by_training),
                "training_seeds": sorted(by_training),
                "unit": "per_training_seed_rollout_mean",
                "metrics": {
                    metric: _summary_stats(values)
                    for metric, values in training_metric_means.items()
                },
            },
        }
    )
    return common


def moving_block_bootstrap_mean(
    series: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return moving-block-bootstrap means for one chronological series."""
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise AggregationContractError("bootstrap series must be finite and non-empty")
    if n_bootstrap <= 0:
        raise AggregationContractError("n_bootstrap must be positive")
    if block_length <= 0:
        raise AggregationContractError("block_length must be positive")
    length = min(int(block_length), int(values.size))
    block_count = int(math.ceil(values.size / length))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(
        0, values.size - length + 1, size=(int(n_bootstrap), block_count)
    )
    offsets = np.arange(length, dtype=np.int64)
    samples = values[(starts[:, :, None] + offsets[None, None, :])].reshape(
        int(n_bootstrap), -1
    )[:, : values.size]
    return samples.mean(axis=1)


def _bootstrap_summary(
    series: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    if not (0.0 < ci_level < 1.0):
        raise AggregationContractError(
            "ci_level must lie strictly between zero and one"
        )
    boot = moving_block_bootstrap_mean(
        series, block_length=block_length, n_bootstrap=n_bootstrap, seed=seed
    )
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "n_windows": int(np.asarray(series).size),
        "mean_delta": float(np.asarray(series).mean()),
        "ci_level": float(ci_level),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "block_length": int(min(block_length, np.asarray(series).size)),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


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
    raise AggregationContractError(f"no per-window series for metric {metric!r}")


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
    expected = {(train, rollout) for train in TRAIN_SEEDS for rollout in ROLLOUT_SEEDS}
    if set(left) != expected or set(right) != expected:
        raise AggregationContractError(f"{name} does not have all paired seed runs")
    ordered_keys = sorted(expected)
    metric_deltas = {
        metric: _summary_stats(
            right[seed_pair]["metrics"][metric] - left[seed_pair]["metrics"][metric]
            for seed_pair in ordered_keys
        )
        for metric in METRIC_NAMES
    }
    breach_deltas = _summary_stats(
        right[seed_pair]["breach_count"] - left[seed_pair]["breach_count"]
        for seed_pair in ordered_keys
    )
    window_uncertainty: dict[str, Any] = {}
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
        window_uncertainty[metric] = _bootstrap_summary(
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
        "run_level_metric_deltas": metric_deltas,
        "run_level_breach_count_delta": breach_deltas,
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
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


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


def _per_run_metric_rows(
    runs: Sequence[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
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


def _metric_summary_rows(
    candidate_summaries: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    for candidate, summary in candidate_summaries.items():
        if summary["variation_type"] == "rollout_sampling_sensitivity":
            sources = [
                ("rollout_sampling", "", summary["sampling_sensitivity"]["metrics"])
            ]
        else:
            sources = [
                ("within_training_rollout", item["training_seed"], item["metrics"])
                for item in summary["rollout_sensitivity_by_training_seed"]
            ]
            sources.append(
                (
                    "across_training_seed_means",
                    "",
                    summary["training_seed_sensitivity"]["metrics"],
                )
            )
        for variation, training_seed, metrics in sources:
            for metric, stats in metrics.items():
                yield {
                    "candidate": candidate,
                    "variation": variation,
                    "training_seed": training_seed,
                    "metric": metric,
                    **stats,
                }


def _paired_rows(comparisons: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for name, comparison in comparisons.items():
        for metric, stats in comparison["run_level_metric_deltas"].items():
            yield {
                "comparison": name,
                "level": "run_seed_pairs",
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


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FMTS multi-seed benchmark aggregate",
        "",
        "Threshold passing is evaluated for each run before any seed averaging.",
        "Training, rollout, and chronological-window variation are reported separately.",
        "",
        "## Pass-all profile",
        "",
        "| Candidate | Runs passing | Pass rate | Training seeds passing every rollout |",
        "|---|---:|---:|---:|",
    ]
    for candidate in sorted(summary["candidate_summaries"]):
        profile = summary["candidate_summaries"][candidate]["pass_profile"]
        training = profile.get("n_training_seeds_passing_every_rollout", "n/a")
        lines.append(
            f"| {summary['candidate_summaries'][candidate]['display_name']} "
            f"| {profile['n_pass_all']}/{profile['n_runs']} "
            f"| {profile['pass_all_rate']:.3f} | {training} |"
        )
    lines.extend(["", "## Paired controls", ""])
    for name, comparison in summary["paired_comparisons"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "Deltas are right minus left; negative proper-score deltas favor the right candidate."
        )
        lines.append("")
        lines.append("| Proper score | Mean run delta | 95% moving-block interval |")
        lines.append("|---|---:|---:|")
        for metric in PROPER_WINDOW_METRICS:
            run_delta = comparison["run_level_metric_deltas"][metric]["mean"]
            interval = comparison["window_uncertainty"][metric]
            lines.append(
                f"| {metric} | {run_delta:.6g} | "
                f"[{interval['ci_low']:.6g}, {interval['ci_high']:.6g}] |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(output_dir: Path, summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "summary.json", summary)
    markdown_path = output_dir / "summary.md"
    temporary_markdown = markdown_path.with_name(
        f".{markdown_path.name}.{os.getpid()}.tmp"
    )
    temporary_markdown.write_text(_render_markdown(summary), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
    _write_csv_atomic(
        output_dir / "per_run_thresholds.csv",
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
        output_dir / "per_run_metrics.csv",
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
        output_dir / "metric_summaries.csv",
        (
            "candidate",
            "variation",
            "training_seed",
            "metric",
            "n",
            "mean",
            "std",
            "min",
            "max",
        ),
        _metric_summary_rows(summary["candidate_summaries"]),
    )
    _write_csv_atomic(
        output_dir / "paired_comparisons.csv",
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
    """Validate all runs, aggregate them, and atomically write review artifacts."""
    protocol_path = Path(protocol_manifest).expanduser().resolve()
    document = _read_json(protocol_path, "protocol manifest")
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping):
        raise AggregationContractError("protocol manifest has no protocol object")
    contract, dag_hash = _validate_contract(protocol)
    nodes = _validate_protocol_schedule(document, protocol)
    result_root = Path(str(protocol.get("result_root", ""))).resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == result_root or result_root not in output.parents:
        raise AggregationContractError(
            "aggregate output directory must be below result_root"
        )
    if block_length != 30:
        raise AggregationContractError("the frozen moving-block length is 30 windows")
    if n_bootstrap <= 0:
        raise AggregationContractError("n_bootstrap must be positive")

    runs = [
        _load_run(node=node, contract=contract, dag_hash=dag_hash)
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
    candidate_summaries = {
        candidate: _candidate_summary(candidate, runs_by_candidate[candidate])
        for candidate in sorted(EXPECTED_CANDIDATES)
    }
    paired_comparisons = {
        "deepvar_vs_deepvar_support": _paired_comparison(
            name="deepvar_vs_deepvar_support",
            left_candidate="deepvar",
            right_candidate="deepvar_sni",
            runs_by_candidate=runs_by_candidate,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
        ),
        "scenarioflow_vs_graph": _paired_comparison(
            name="scenarioflow_vs_graph",
            left_candidate="scenarioflow",
            right_candidate="scenarioflow_graph",
            runs_by_candidate=runs_by_candidate,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_manifest": str(protocol_path),
        "protocol_dag_sha256": dag_hash,
        "contract": dict(contract),
        "seed_schedule": {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "bootstrap_contract": {
            "method": "paired_moving_block_bootstrap_of_seed_averaged_window_differences",
            "block_length": int(block_length),
            "ci_level": float(ci_level),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "per_runs": [_serializable_run(run) for run in runs],
        "candidate_summaries": candidate_summaries,
        "paired_comparisons": paired_comparisons,
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
