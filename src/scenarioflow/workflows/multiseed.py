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
    RESULT_SCHEMA_VERSION,
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


def _node(
    *, candidate: str, output_root: Path, rollout_seed: int, train_seed: int | None
) -> dict[str, Any]:
    run_id = (
        f"{candidate}/eval_seed_{rollout_seed}"
        if train_seed is None
        else f"{candidate}/train_seed_{train_seed}/eval_seed_{rollout_seed}"
    )
    command = [
        "python",
        "-m",
        "scenarioflow.workflows.reproduction",
        "run-node",
        "--candidate",
        candidate,
        "--rollout-seed",
        str(rollout_seed),
        "--result-input",
        str(output_root / "inputs" / run_id / "results_current_panel.json"),
        "--output-dir",
        str(output_root / run_id),
    ]
    if train_seed is not None:
        command += ["--train-seed", str(train_seed)]
    if candidate == "garch_ccc":
        command += [
            "--garch-contract-report",
            str(output_root / "inputs" / "garch_ccc_contract_report.json"),
        ]
    return {
        "run_id": run_id,
        "candidate": candidate,
        "train_seed": train_seed,
        "rollout_seed": rollout_seed,
        "command": command,
        "completion_artifacts": [
            str(output_root / run_id / "results_current_panel.json"),
            str(output_root / run_id / "run_manifest.json"),
        ],
    }


def build_protocol(
    *,
    output_root: str | Path,
    source_manifest: str | Path,
    artifact_manifest: str | Path,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    rollout_seeds: Sequence[int] = ROLLOUT_SEEDS,
) -> dict[str, Any]:
    if tuple(train_seeds) != TRAIN_SEEDS:
        raise ProtocolValidationError(f"training seeds must remain {list(TRAIN_SEEDS)}")
    if tuple(rollout_seeds) != ROLLOUT_SEEDS:
        raise ProtocolValidationError(
            f"rollout seeds must remain {list(ROLLOUT_SEEDS)}"
        )
    source = Path(source_manifest).expanduser().resolve()
    artifacts = Path(artifact_manifest).expanduser().resolve()
    if not source.is_file() or not artifacts.is_file():
        raise FileNotFoundError(
            "explicit frozen source and artifact manifests are required"
        )
    root = Path(output_root).expanduser().resolve()
    contract = {
        "schema_version": PROTOCOL_SCHEMA,
        "benchmark_split": "published_validation",
        "mode": MODE,
        "candidate_ids": list(CANDIDATE_IDS),
        "train_seeds": list(TRAIN_SEEDS),
        "rollout_seeds": list(ROLLOUT_SEEDS),
        "validation_protocol": {
            key: value for key, value in PUBLISHED_VALIDATION_PROTOCOL.__dict__.items()
        },
        "thresholds": TABLE_THRESHOLD_RULES,
        "source_manifest": {"path": str(source), "sha256": _file_hash(source)},
        "artifact_manifest": {"path": str(artifacts), "sha256": _file_hash(artifacts)},
    }
    contract_hash = _canonical_hash(contract)
    prepare_output_root(
        root, protected_roots=(source, artifacts), contract_hash=contract_hash
    )
    nodes = []
    for candidate in CANDIDATE_IDS:
        if candidate in CLASSICAL_CANDIDATES:
            nodes.extend(
                _node(
                    candidate=candidate,
                    output_root=root,
                    train_seed=None,
                    rollout_seed=rollout_seed,
                )
                for rollout_seed in ROLLOUT_SEEDS
            )
        else:
            nodes.extend(
                _node(
                    candidate=candidate,
                    output_root=root,
                    train_seed=train_seed,
                    rollout_seed=rollout_seed,
                )
                for train_seed in TRAIN_SEEDS
                for rollout_seed in ROLLOUT_SEEDS
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


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    value = float(current)
    return value if math.isfinite(value) else None


def _expected_keys(candidate: str) -> set[tuple[int | None, int]]:
    if candidate in CLASSICAL_CANDIDATES:
        return {(None, rollout) for rollout in ROLLOUT_SEEDS}
    return {
        (training, rollout) for training in TRAIN_SEEDS for rollout in ROLLOUT_SEEDS
    }


def aggregate_results(
    runs: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = CANDIDATE_IDS
) -> dict[str, Any]:
    """Aggregate only complete frozen schedules; never average an incomplete resume."""
    candidate_summaries: dict[str, Any] = {}
    for candidate in candidates:
        selected = [run for run in runs if run.get("candidate") == candidate]
        actual = {(run.get("train_seed"), run.get("rollout_seed")) for run in selected}
        if actual != _expected_keys(candidate):
            raise ProtocolValidationError(
                f"{candidate} does not have a complete seed deck"
            )
        metric_values: dict[str, list[float]] = {name: [] for name, _ in METRIC_PATHS}
        for run in selected:
            result = run.get("result")
            if not isinstance(result, Mapping):
                raise ProtocolValidationError(
                    f"{candidate} run result must be an object"
                )
            if result.get("schema_version") != RESULT_SCHEMA_VERSION:
                raise ProtocolValidationError(
                    f"{candidate} run has wrong result schema"
                )
            if result.get("baseline_name") != candidate or result.get("mode") != MODE:
                raise ProtocolValidationError(f"{candidate} run identity mismatch")
            if result.get("n_windows") != 441 or result.get("n_samples") != 64:
                raise ProtocolValidationError(
                    f"{candidate} run is not a published validation deck"
                )
            for name, path in METRIC_PATHS:
                value = _path_value(result, path)
                if value is not None:
                    metric_values[name].append(value)
        metrics = {
            name: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in metric_values.items()
            if values
        }
        candidate_summaries[candidate] = {
            "run_count": len(selected),
            "metrics": metrics,
        }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "benchmark_split": "published_validation",
        "protocol": {
            "training_seeds": list(TRAIN_SEEDS),
            "rollout_seeds": list(ROLLOUT_SEEDS),
        },
        "candidates": candidate_summaries,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    args = parser.parse_args(argv)
    payload = build_protocol(
        output_root=args.output_root,
        source_manifest=args.source_manifest,
        artifact_manifest=args.artifact_manifest,
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
