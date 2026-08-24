"""Native, resumable three-training-seed/five-rollout-seed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..evaluation.current_panel import (
    METRIC_PATHS,
    MODE,
    PUBLISHED_VALIDATION_PROTOCOL,
    TABLE_THRESHOLD_RULES,
)

TRAIN_SEEDS = (1001, 1002, 1003)
ROLLOUT_SEEDS = (2001, 2002, 2003, 2004, 2005)
CANDIDATE_IDS = (
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
TRAINED_CANDIDATES = tuple(
    candidate for candidate in CANDIDATE_IDS if candidate not in CLASSICAL_CANDIDATES
)
PROTOCOL_SCHEMA = "scenarioflow.multiseed-protocol.v1"
AGGREGATE_SCHEMA = "scenarioflow.multiseed-aggregate.v1"
_CONTROL_FILENAME = ".scenarioflow-output-contract.json"


class ProtocolValidationError(ValueError):
    """Raised when frozen schedule or resume safety has drifted."""


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def prepare_output_root(
    output_root: str | Path,
    *,
    protected_roots: Iterable[str | Path] = (),
    contract_hash: str | None = None,
) -> Path:
    """Create or resume only an isolated root bound to the same protocol hash."""
    output = Path(output_root).expanduser().resolve()
    for supplied in protected_roots:
        protected = Path(supplied).expanduser().resolve()
        if _overlaps(output, protected):
            raise ProtocolValidationError(
                f"output root {output} overlaps protected path {protected}"
            )
    if contract_hash is not None and (
        len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash)
    ):
        raise ProtocolValidationError("contract_hash must be a lowercase SHA-256")
    output.mkdir(parents=True, exist_ok=True)
    control = output / _CONTROL_FILENAME
    if control.exists():
        try:
            existing = json.loads(control.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "output root has an unreadable contract"
            ) from error
        if contract_hash is not None and existing.get("contract_hash") != contract_hash:
            raise ProtocolValidationError("output root belongs to a different contract")
    elif any(output.iterdir()):
        raise ProtocolValidationError(
            "refusing to adopt a nonempty unsealed output root"
        )
    elif contract_hash is not None:
        control.write_text(
            json.dumps(
                {"schema_version": PROTOCOL_SCHEMA, "contract_hash": contract_hash},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return output


def _evaluation_node(
    *,
    candidate: str,
    output_root: Path,
    data_dir: Path,
    checkpoint_root: Path,
    artifact_root: Path,
    rollout_seed: int,
    train_seed: int | None,
    contract_hash: str,
    max_windows: int = 441,
    n_samples: int = 64,
) -> dict[str, Any]:
    run_id = (
        f"{candidate}/eval_seed_{rollout_seed}"
        if train_seed is None
        else f"{candidate}/train_seed_{train_seed}/eval_seed_{rollout_seed}"
    )
    destination = output_root / run_id
    node_id = "evaluate:" + run_id.replace("/", ":")
    if candidate == "garch_ccc":
        command = [
            "python",
            "-m",
            "scenarioflow.evaluation.garch_ccc_contract",
            "--summary",
            str(
                artifact_root
                / "results/fmts_reproduction_20260805/checkpoint_replay/garch_ccc/summary.json"
            ),
            "--result",
            str(
                artifact_root
                / "results/fmts_reproduction_20260805/checkpoint_replay/garch_ccc/garch_ccc/results_current_panel.json"
            ),
            "--manifest",
            str(
                artifact_root
                / "results/fmts_reproduction_20260805/checkpoint_replay/garch_ccc/manifest.json"
            ),
            "--table",
            str(artifact_root / "paper/generated_tables/table_baseline_comparison.tex"),
            "--output-dir",
            str(destination),
            "--protocol",
            str(output_root / "protocol_manifest.json"),
            "--node-id",
            node_id,
        ]
    else:
        command = [
            "python",
            "-m",
            "scenarioflow.evaluation.current_panel",
            "--baseline",
            candidate,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(destination),
            "--n-samples",
            str(n_samples),
            "--max-windows",
            str(max_windows),
            "--batch-size",
            "32",
            "--seed",
            str(rollout_seed),
            "--save-window-scores",
            "--protocol",
            str(output_root / "protocol_manifest.json"),
            "--node-id",
            node_id,
        ]
        if train_seed is not None:
            if candidate == "scenarioflow":
                checkpoint = (
                    output_root
                    / "generated_checkpoints"
                    / candidate
                    / f"train_seed_{train_seed}"
                    / "stage2.pt"
                )
                dependency = f"train:scenarioflow:stage2:{train_seed}"
            elif candidate == "scenarioflow_graph":
                checkpoint = (
                    output_root
                    / "generated_checkpoints"
                    / candidate
                    / f"train_seed_{train_seed}"
                    / "graph.pt"
                )
                dependency = f"train:scenarioflow_graph:graph:{train_seed}"
            else:
                checkpoint = (
                    output_root
                    / "generated_checkpoints"
                    / candidate
                    / f"train_seed_{train_seed}.pt"
                )
                dependency = f"train:{candidate}:{train_seed}"
            command += [
                "--checkpoint",
                str(checkpoint),
            ]
    return {
        "node_id": node_id,
        "action": "authority" if candidate == "garch_ccc" else "evaluate",
        "candidate": candidate,
        "train_seed": train_seed,
        "rollout_seed": rollout_seed,
        "destination": str(destination),
        "command": command,
        "depends_on": [] if train_seed is None else [dependency],
        "protocol_contract_hash": contract_hash,
    }


def _training_node(
    *,
    candidate: str,
    data_dir: Path,
    output_root: Path,
    artifact_root: Path,
    train_seed: int,
    contract_hash: str,
) -> dict[str, Any]:
    destination = (
        output_root
        / "generated_checkpoints"
        / candidate
        / f"train_seed_{train_seed}.pt"
    )
    command = [
        "python",
        "-m",
        "scenarioflow.workflows.reproduction",
        "train-baseline",
        "--candidate",
        candidate,
        "--data-dir",
        str(data_dir),
        "--checkpoint-out",
        str(destination),
        "--train-seed",
        str(train_seed),
        "--epochs",
        "10",
        "--batch-size",
        "64",
        "--n-diffusion-steps",
        "40",
        "--sample-steps",
        "8",
        "--weight-decay",
        "0.00001",
        "--clip-grad",
        "1.0",
    ]
    return {
        "node_id": f"train:{candidate}:{train_seed}",
        "action": "train",
        "candidate": candidate,
        "train_seed": train_seed,
        "rollout_seed": None,
        "destination": str(destination),
        "command": command,
        "depends_on": [],
        "protocol_contract_hash": contract_hash,
    }


def _scenarioflow_training_nodes(
    *, output_root: Path, data_dir: Path, train_seed: int, contract_hash: str
) -> list[dict[str, Any]]:
    stage1 = (
        output_root
        / "generated_checkpoints"
        / "scenarioflow"
        / f"train_seed_{train_seed}"
        / "stage1.pt"
    )
    stage2 = (
        output_root
        / "generated_checkpoints"
        / "scenarioflow"
        / f"train_seed_{train_seed}"
        / "stage2.pt"
    )
    graph = (
        output_root
        / "generated_checkpoints"
        / "scenarioflow_graph"
        / f"train_seed_{train_seed}"
        / "graph.pt"
    )
    specifications = (
        (
            "train:scenarioflow:stage1",
            "scenarioflow",
            "stage1",
            stage1,
            None,
            8,
            None,
        ),
        (
            "train:scenarioflow:stage2",
            "scenarioflow",
            "stage2",
            stage2,
            stage1,
            3,
            None,
        ),
        (
            "train:scenarioflow_graph:graph",
            "scenarioflow_graph",
            "graph",
            graph,
            stage2,
            2,
            None,
        ),
    )
    nodes = []
    previous: str | None = None
    for (
        prefix,
        candidate,
        stage,
        destination,
        checkpoint,
        epochs,
        _unused,
    ) in specifications:
        node_id = f"{prefix}:{train_seed}"
        command = [
            "python",
            "-m",
            "scenarioflow.training",
            "--stage",
            stage,
            "--data-dir",
            str(data_dir),
            "--output",
            str(destination),
            "--epochs",
            str(epochs),
            "--seed",
            str(train_seed),
            "--batch-size",
            "32" if stage == "stage1" else "8",
            "--max-train-windows",
            "0" if stage == "stage2" else "2048",
            "--lr",
            "0.0007" if stage == "stage1" else "0.00002",
            "--weight-decay",
            "0.00001",
            "--clip-grad",
            "1.0",
        ]
        if stage != "stage1":
            command += [
                "--train-sample-count",
                "2",
                "--rollout-flow-steps",
                "4",
                "--energy-weight",
                "0.2",
                "--channel-level-energy-weight",
                "0.05",
            ]
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
        if stage == "graph":
            command += [
                "--free-running-fm-weight",
                "0.2",
                "--free-running-fm-prefix-steps",
                "5",
                "--graph-k",
                "8",
            ]
        nodes.append(
            {
                "node_id": node_id,
                "action": "train",
                "stage": stage,
                "candidate": candidate,
                "train_seed": train_seed,
                "rollout_seed": None,
                "destination": str(destination),
                "command": command,
                "depends_on": [] if previous is None else [previous],
                "protocol_contract_hash": contract_hash,
            }
        )
        previous = node_id
    return nodes


def build_protocol(
    *,
    output_root: str | Path,
    source_manifest: str | Path,
    artifact_manifest: str | Path,
    data_dir: str | Path,
    checkpoint_root: str | Path,
    artifact_root: str | Path,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    rollout_seeds: Sequence[int] = ROLLOUT_SEEDS,
    candidates: Sequence[str] = CANDIDATE_IDS,
    smoke: bool = False,
    max_windows: int = 441,
    n_samples: int = 64,
) -> dict[str, Any]:
    if not smoke and tuple(train_seeds) != TRAIN_SEEDS:
        raise ProtocolValidationError(f"training seeds must remain {list(TRAIN_SEEDS)}")
    if not smoke and tuple(rollout_seeds) != ROLLOUT_SEEDS:
        raise ProtocolValidationError(
            f"rollout seeds must remain {list(ROLLOUT_SEEDS)}"
        )
    selected_candidates = tuple(candidates)
    if not selected_candidates or any(
        item not in CANDIDATE_IDS for item in selected_candidates
    ):
        raise ProtocolValidationError(
            "candidate selection contains an unknown candidate"
        )
    if smoke and any(
        item not in CLASSICAL_CANDIDATES or item == "garch_ccc"
        for item in selected_candidates
    ):
        raise ProtocolValidationError(
            "smoke execution supports generated classical rows only"
        )
    if smoke and (max_windows <= 0 or n_samples <= 0):
        raise ProtocolValidationError("smoke deck dimensions must be positive")
    source = Path(source_manifest).expanduser().resolve()
    artifacts = Path(artifact_manifest).expanduser().resolve()
    data = Path(data_dir).expanduser().resolve()
    checkpoints = Path(checkpoint_root).expanduser().resolve()
    artifact_directory = Path(artifact_root).expanduser().resolve()
    if not source.is_file() or not artifacts.is_file():
        raise FileNotFoundError(
            "explicit frozen source and artifact manifests are required"
        )
    if not data.is_dir() or not checkpoints.is_dir() or not artifact_directory.is_dir():
        raise FileNotFoundError(
            "explicit data, checkpoint, and artifact roots are required"
        )
    root = Path(output_root).expanduser().resolve()
    contract = {
        "schema_version": PROTOCOL_SCHEMA,
        "benchmark_split": (
            "published_validation_prefix_smoke" if smoke else "published_validation"
        ),
        "mode": MODE,
        "candidate_ids": list(selected_candidates),
        "train_seeds": list(train_seeds),
        "rollout_seeds": list(rollout_seeds),
        "validation_protocol": {
            key: (
                max_windows
                if smoke and key == "val_size"
                else n_samples if smoke and key == "n_samples" else value
            )
            for key, value in PUBLISHED_VALIDATION_PROTOCOL.__dict__.items()
        },
        "thresholds": TABLE_THRESHOLD_RULES,
        "source_manifest": {"path": str(source), "sha256": _file_hash(source)},
        "artifact_manifest": {"path": str(artifacts), "sha256": _file_hash(artifacts)},
        "data_dir": str(data),
        "checkpoint_root": str(checkpoints),
        "artifact_root": str(artifact_directory),
    }
    contract_hash = _canonical_hash(contract)
    prepare_output_root(
        root,
        protected_roots=(source, artifacts, data, checkpoints, artifact_directory),
        contract_hash=contract_hash,
    )
    nodes: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        if candidate in {"scenarioflow", "scenarioflow_graph"}:
            if candidate == "scenarioflow":
                for train_seed in train_seeds:
                    nodes.extend(
                        _scenarioflow_training_nodes(
                            output_root=root,
                            data_dir=data,
                            train_seed=train_seed,
                            contract_hash=contract_hash,
                        )
                    )
        elif candidate not in CLASSICAL_CANDIDATES:
            nodes.extend(
                _training_node(
                    candidate=candidate,
                    data_dir=data,
                    output_root=root,
                    artifact_root=artifact_directory,
                    train_seed=train_seed,
                    contract_hash=contract_hash,
                )
                for train_seed in train_seeds
            )
        if candidate == "garch_ccc":
            seed_pairs = ((None, 772),)
        elif candidate in CLASSICAL_CANDIDATES:
            seed_pairs = ((None, rollout) for rollout in rollout_seeds)
        else:
            seed_pairs = (
                (train_seed, rollout_seed)
                for train_seed in train_seeds
                for rollout_seed in rollout_seeds
            )
        nodes.extend(
            _evaluation_node(
                candidate=candidate,
                output_root=root,
                data_dir=data,
                checkpoint_root=checkpoints,
                artifact_root=artifact_directory,
                train_seed=train_seed,
                rollout_seed=rollout_seed,
                contract_hash=contract_hash,
                max_windows=max_windows,
                n_samples=n_samples,
            )
            for train_seed, rollout_seed in seed_pairs
        )
    payload = {
        **contract,
        "contract_hash": contract_hash,
        "output_root": str(root),
        "nodes": nodes,
    }
    destination = root / "protocol_manifest.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise ProtocolValidationError(
            "existing protocol manifest differs from requested protocol"
        )
    destination.write_text(serialized, encoding="utf-8")
    return payload


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _metric_values(result: Mapping[str, Any], candidate: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, path in METRIC_PATHS:
        raw = _path_value(result, path)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProtocolValidationError(
                f"{candidate} metric {name} is missing or non-numeric"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ProtocolValidationError(f"{candidate} metric {name} is non-finite")
        values[name] = value
    return values


def _threshold_assessment(value: float, rule: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(rule["kind"])
    lower = float(rule["lo"]) if "lo" in rule else None
    upper = float(rule["hi"]) if "hi" in rule else None
    if kind == "lower" and lower is not None:
        margin = value - lower
    elif kind == "upper" and upper is not None:
        margin = upper - value
    elif kind == "band" and lower is not None and upper is not None:
        margin = min(value - lower, upper - value)
    else:
        raise ProtocolValidationError(f"unsupported threshold rule {rule!r}")
    return {
        "value": value,
        "kind": kind,
        "lower": lower,
        "upper": upper,
        "signed_margin": float(margin),
        "status": "pass" if margin >= 0 else "fail",
        "passed": bool(margin >= 0),
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
        raise ProtocolValidationError("cannot summarize empty or non-finite values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _expected_keys(candidate: str) -> set[tuple[int | None, int]]:
    if candidate == "garch_ccc":
        return {(None, 772)}
    if candidate in CLASSICAL_CANDIDATES:
        return {(None, rollout) for rollout in ROLLOUT_SEEDS}
    return {
        (training, rollout) for training in TRAIN_SEEDS for rollout in ROLLOUT_SEEDS
    }


def _metrics_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: _summary_stats(run["metrics"][name] for run in runs)
        for name, _ in METRIC_PATHS
    }


def _pass_profile(
    runs: Sequence[Mapping[str, Any]], *, learned: bool
) -> dict[str, Any]:
    n_pass = sum(bool(run["pass_all"]) for run in runs)
    profile: dict[str, Any] = {
        "n_runs": len(runs),
        "n_pass_all": n_pass,
        "pass_all_rate": float(n_pass / len(runs)),
        "breach_count": _summary_stats(run["breach_count"] for run in runs),
    }
    if learned:
        passing = []
        for seed in TRAIN_SEEDS:
            seed_runs = [run for run in runs if run["training_seed"] == seed]
            if all(run["pass_all"] for run in seed_runs):
                passing.append(seed)
        profile.update(
            {
                "n_training_seeds": 3,
                "n_training_seeds_passing_every_rollout": len(passing),
                "training_seeds_passing_every_rollout": passing,
            }
        )
    return profile


def _candidate_summary(
    candidate: str, runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    learned = candidate not in CLASSICAL_CANDIDATES
    ordered = sorted(
        runs, key=lambda run: (run["training_seed"] or -1, run["rollout_seed"])
    )
    common: dict[str, Any] = {
        "candidate": candidate,
        "pass_profile": _pass_profile(ordered, learned=learned),
    }
    if candidate == "garch_ccc":
        common.update(
            {
                "variation_type": "frozen_published_authority",
                "authority_seed": 772,
                "metrics": _metrics_summary(ordered),
            }
        )
        return common
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
    rollout_summaries = []
    training_means: dict[str, list[float]] = {name: [] for name, _ in METRIC_PATHS}
    for seed in TRAIN_SEEDS:
        seed_runs = [run for run in ordered if run["training_seed"] == seed]
        metrics = _metrics_summary(seed_runs)
        for name in training_means:
            training_means[name].append(float(metrics[name]["mean"]))
        rollout_summaries.append(
            {
                "training_seed": seed,
                "n_rollout_seeds": len(seed_runs),
                "rollout_seeds": [run["rollout_seed"] for run in seed_runs],
                "metrics": metrics,
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
                "n_training_seeds": 3,
                "training_seeds": list(TRAIN_SEEDS),
                "unit": "per_training_seed_rollout_mean",
                "metrics": {
                    name: _summary_stats(values)
                    for name, values in training_means.items()
                },
            },
        }
    )
    return common


def moving_block_bootstrap_mean(
    series: np.ndarray, *, block_length: int, n_bootstrap: int, seed: int
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or n_bootstrap <= 0
        or block_length <= 0
    ):
        raise ProtocolValidationError("invalid moving-block bootstrap inputs")
    length = min(int(block_length), values.size)
    block_count = int(math.ceil(values.size / length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, values.size - length + 1, size=(n_bootstrap, block_count))
    offsets = np.arange(length)
    samples = values[starts[:, :, None] + offsets].reshape(n_bootstrap, -1)[
        :, : values.size
    ]
    return samples.mean(axis=1)


def _window_series(run: Mapping[str, Any], metric: str) -> np.ndarray:
    scores = run["_window_scores"]
    if metric == "CRPS":
        return scores["crps_contributions"].mean(axis=(1, 2))
    if metric == "Energy Score":
        return scores["energy_contributions"].mean(axis=1)
    return scores["variogram_h30_contributions"]


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
    if not 0 < ci_level < 1:
        raise ProtocolValidationError("ci_level must lie between zero and one")
    key = lambda run: (int(run["training_seed"]), int(run["rollout_seed"]))
    left = {key(run): run for run in runs_by_candidate[left_candidate]}
    right = {key(run): run for run in runs_by_candidate[right_candidate]}
    expected = {(train, rollout) for train in TRAIN_SEEDS for rollout in ROLLOUT_SEEDS}
    if set(left) != expected or set(right) != expected:
        raise ProtocolValidationError(f"{name} does not have all paired seed runs")
    ordered = sorted(expected)
    metric_deltas = {
        metric: _summary_stats(
            right[pair]["metrics"][metric] - left[pair]["metrics"][metric]
            for pair in ordered
        )
        for metric, _ in METRIC_PATHS
    }
    uncertainty: dict[str, Any] = {}
    for metric in ("CRPS", "Energy Score", "Variogram"):
        deltas = []
        reference = None
        for pair in ordered:
            li = left[pair]["_window_scores"]["window_indices"]
            ri = right[pair]["_window_scores"]["window_indices"]
            if not np.array_equal(li, ri) or (
                reference is not None and not np.array_equal(reference, li)
            ):
                raise ProtocolValidationError(f"{name} window chronology differs")
            reference = li
            deltas.append(
                _window_series(right[pair], metric) - _window_series(left[pair], metric)
            )
        series = np.stack(deltas).mean(axis=0)
        boot = moving_block_bootstrap_mean(
            series,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        alpha = (1 - ci_level) / 2
        low, high = np.quantile(boot, [alpha, 1 - alpha])
        uncertainty[metric] = {
            "n_windows": len(series),
            "mean_delta": float(series.mean()),
            "ci_level": ci_level,
            "ci_low": float(low),
            "ci_high": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "block_length": min(block_length, len(series)),
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
        }
    return {
        "name": name,
        "left_candidate": left_candidate,
        "right_candidate": right_candidate,
        "delta_orientation": "right_minus_left",
        "n_seed_pairs": 15,
        "paired_training_seeds": list(TRAIN_SEEDS),
        "paired_rollout_seeds": list(ROLLOUT_SEEDS),
        "run_level_metric_deltas": metric_deltas,
        "window_uncertainty": uncertainty,
    }


def aggregate_results(
    runs: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[str] = CANDIDATE_IDS,
    paired_comparisons: Sequence[tuple[str, str, str]] = (),
    block_length: int = 30,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260805,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Screen each complete run, then preserve training and rollout variation."""
    processed_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        selected = [run for run in runs if run.get("candidate") == candidate]
        keys = [(run.get("train_seed"), run.get("rollout_seed")) for run in selected]
        if len(keys) != len(set(keys)):
            raise ProtocolValidationError(f"{candidate} has a duplicate seed pair")
        actual = set(keys)
        if actual != _expected_keys(candidate):
            raise ProtocolValidationError(
                f"{candidate} does not have a complete seed deck"
            )
        processed: list[dict[str, Any]] = []
        for run in selected:
            result = run.get("result")
            if not isinstance(result, Mapping):
                raise ProtocolValidationError(
                    f"{candidate} run result must be an object"
                )
            if result.get("baseline_name") != candidate or result.get("mode") != MODE:
                raise ProtocolValidationError(f"{candidate} run identity mismatch")
            if result.get("n_windows") != 441 or result.get("n_samples") != 64:
                raise ProtocolValidationError(
                    f"{candidate} run is not a published validation deck"
                )
            if (
                not isinstance(result.get("state_specs"), list)
                or len(result["state_specs"]) != 39
            ):
                raise ProtocolValidationError(
                    f"{candidate} run must contain 39 state specs"
                )
            metrics = _metric_values(result, candidate)
            thresholds, breach_count = _threshold_profile(metrics)
            if candidate == "garch_ccc":
                processed.append(
                    {
                        "candidate": candidate,
                        "training_seed": None,
                        "rollout_seed": 772,
                        "metrics": metrics,
                        "thresholds": thresholds,
                        "breach_count": breach_count,
                        "pass_all": breach_count == 0,
                    }
                )
                continue
            window_scores = run.get("window_scores")
            if not isinstance(window_scores, Mapping):
                raise ProtocolValidationError(
                    f"{candidate} run is missing window scores"
                )
            indices = np.asarray(window_scores.get("window_indices"))
            if indices.shape != (441,) or not np.all(np.diff(indices) == 1):
                raise ProtocolValidationError(
                    f"{candidate} window chronology is invalid"
                )
            required_shapes = {
                "crps_contributions": (441, 4, 39),
                "energy_contributions": (441, 4),
                "variogram_h30_contributions": (441,),
            }
            arrays: dict[str, np.ndarray] = {"window_indices": indices}
            for name, shape in required_shapes.items():
                array = np.asarray(window_scores.get(name), dtype=np.float64)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ProtocolValidationError(
                        f"{candidate} window score {name} is invalid"
                    )
                arrays[name] = array
            reconciled = {
                "CRPS": float(arrays["crps_contributions"].mean()),
                "Energy Score": float(arrays["energy_contributions"].mean()),
                "Variogram": float(arrays["variogram_h30_contributions"].mean()),
            }
            for name, value in reconciled.items():
                if not np.isclose(value, metrics[name], rtol=1e-6, atol=1e-8):
                    raise ProtocolValidationError(
                        f"{candidate} window score {name} does not reconcile"
                    )
            processed.append(
                {
                    "candidate": candidate,
                    "training_seed": run.get("train_seed"),
                    "rollout_seed": run.get("rollout_seed"),
                    "metrics": metrics,
                    "thresholds": thresholds,
                    "breach_count": breach_count,
                    "pass_all": breach_count == 0,
                    "_window_scores": arrays,
                }
            )
        processed_by_candidate[candidate] = processed
    candidate_summaries = {
        candidate: _candidate_summary(candidate, selected)
        for candidate, selected in processed_by_candidate.items()
    }
    comparisons = {
        name: _paired_comparison(
            name=name,
            left_candidate=left,
            right_candidate=right,
            runs_by_candidate=processed_by_candidate,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
        )
        for name, left, right in paired_comparisons
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "benchmark_split": "published_validation",
        "protocol": {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "candidates": candidate_summaries,
        "paired_comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)
    payload = build_protocol(
        output_root=args.output_root,
        source_manifest=args.source_manifest,
        artifact_manifest=args.artifact_manifest,
        data_dir=args.data_dir,
        checkpoint_root=args.checkpoint_root,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "contract_hash": payload["contract_hash"],
                "node_count": len(payload["nodes"]),
            }
        )
    )


if __name__ == "__main__":
    main()
