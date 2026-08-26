"""Fail-closed serial executor and CLI for the compact benchmark protocol."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .guard import assert_isolated_output_root
from .protocol import (
    DEFAULT_PROTOCOL,
    SmokeOverrides,
    load_protocol,
    materialize_protocol,
    select_nodes,
    topological_nodes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROTECTED_RELEASE_PATHS = (
    ".github",
    ".superpowers",
    "results/frozen",
    "data",
    "artifacts",
    "deploy",
    "foundation_envs",
    "metadata",
    "paper_assets",
    "src",
    "scripts",
    "reproduction",
    "tests",
    "web",
)


class BenchmarkExecutionError(RuntimeError):
    """Raised when a node fails or claims an invalid completion artifact."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_option(node: Mapping[str, Any], *names: str) -> str | None:
    command = list(map(str, node.get("command", [])))
    for name in names:
        if name in command:
            position = command.index(name)
            return command[position + 1] if position + 1 < len(command) else None
    return None


def _node_contract(node: Mapping[str, Any]) -> Mapping[str, Any]:
    configuration = node.get("configuration")
    if not isinstance(configuration, Mapping):
        return {}
    contract = configuration.get("contract")
    return contract if isinstance(contract, Mapping) else {}


def _valid_result(path: Path, node: Mapping[str, Any]) -> bool:
    document = json.loads(path.read_text(encoding="utf-8"))
    contract = _node_contract(node)
    expected_windows = int(
        _command_option(node, "--max_windows")
        or contract.get("n_evaluation_windows", 0)
    )
    expected_samples = int(
        _command_option(node, "--n_samples", "--samples")
        or contract.get("n_scenarios_per_window", 0)
    )
    expected_variables = int(contract.get("n_market_variables", 0))
    required_metrics = {
        "current_panel_iv_metrics",
        "joint_panel_quality",
        "panel_b_crps",
        "panel_b_energy_score",
        "panel_b_variogram_score",
        "panel_b_correlation",
        "panel_b_factor_ks",
    }
    return (
        isinstance(document, dict)
        and document.get("baseline_name") == node.get("candidate")
        and document.get("mode") == contract.get("mode")
        and document.get("n_windows") == expected_windows
        and document.get("n_samples") == expected_samples
        and isinstance(document.get("state_specs"), list)
        and len(document["state_specs"]) == expected_variables
        and required_metrics.issubset(document)
    )


def _valid_window_scores(path: Path, node: Mapping[str, Any]) -> bool:
    contract = _node_contract(node)
    expected_windows = int(
        _command_option(node, "--max_windows")
        or contract.get("n_evaluation_windows", 0)
    )
    expected_variables = int(contract.get("n_market_variables", 0))
    expected_horizons = np.asarray([1, 7, 14, 30], dtype=np.int64)
    expected_coverage_levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    required = {
        "schema_version",
        "window_indices",
        "horizons",
        "variable_names",
        "crps_contributions",
        "energy_contributions",
        "variogram_h30_contributions",
        "coverage_levels",
        "coverage_by_window",
        "iv_history_activity",
        "iv_deck_width",
        "full_panel_history_activity",
        "full_panel_deck_width",
    }
    with np.load(path, allow_pickle=False) as document:
        if not required.issubset(document.files):
            return False
        horizons = document["horizons"]
        return (
            document["schema_version"].shape == ()
            and str(document["schema_version"].item()) == "fmts.window_scores.v1"
            and np.array_equal(horizons, expected_horizons)
            and document["window_indices"].shape == (expected_windows,)
            and document["variable_names"].shape == (expected_variables,)
            and document["crps_contributions"].shape
            == (expected_windows, len(expected_horizons), expected_variables)
            and document["energy_contributions"].shape
            == (expected_windows, len(expected_horizons))
            and document["variogram_h30_contributions"].shape == (expected_windows,)
            and document["coverage_levels"].shape == (4,)
            and np.array_equal(document["coverage_levels"], expected_coverage_levels)
            and document["coverage_by_window"].shape == (expected_windows, 4)
            and document["iv_history_activity"].shape == (expected_windows,)
            and document["iv_deck_width"].shape == (expected_windows,)
            and document["full_panel_history_activity"].shape == (expected_windows,)
            and document["full_panel_deck_width"].shape == (expected_windows,)
        )


def _valid_scenarioflow_checkpoint(
    document: Mapping[str, Any],
    node: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_variables: int,
) -> bool:
    configuration = node.get("configuration")
    if not isinstance(configuration, Mapping):
        return False
    normalization = document.get("normalization")
    state_specs = document.get("state_specs")
    model_type = document.get("model_type")
    if (
        model_type not in {None, "scenarioflow"}
        or configuration.get("implementation_name") != node.get("candidate")
        or config.get("conditioning_mode") != "prefix"
        or document.get("model_coordinate") != "state_aware_normalized_innovation"
        or document.get("state_scope")
        != (_command_option(node, "--state_scope") or "joint38")
        or not isinstance(normalization, Mapping)
        or normalization.get("coordinate") != "encoded_increment"
        or normalization.get("innovation_coordinate") != "normalized"
        or not isinstance(state_specs, list)
        or len(state_specs) != expected_variables
    ):
        return False
    stage = configuration.get("stage")
    if stage == "stage1":
        objective = document.get("training_objective")
        return isinstance(objective, Mapping) and bool(objective)
    objective = document.get("finetune_objective")
    if not isinstance(objective, Mapping) or not objective:
        return False
    if not isinstance(document.get("source_checkpoint"), str):
        return False
    if stage == "stage2":
        return node.get("candidate") == "scenarioflow"
    if stage == "graph":
        return (
            node.get("candidate") == "scenarioflow_graph"
            and config.get("velocity_mixer_mode") == "adaptive_graph_residual"
        )
    return False


def _valid_checkpoint(path: Path, node: Mapping[str, Any]) -> bool:
    document = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(document, Mapping):
        return False
    state = document.get("model_state_dict")
    config = document.get("config")
    contract = _node_contract(node)
    if not isinstance(state, Mapping) or not state or not isinstance(config, Mapping):
        return False
    expected_variables = int(contract.get("n_market_variables", 0))
    dimension = config.get("input_dim", config.get("n_cells"))
    if int(dimension or 0) != expected_variables:
        return False
    if int(config.get("history_len", 0)) != int(contract.get("history_length", 0)):
        return False
    if int(config.get("future_len", 0)) != int(contract.get("future_length", 0)):
        return False
    expected_model_type = {
        "deepvar": "deepvar",
        "timegrad": "timegrad",
        "path_diffusion": "path_diffusion",
        "deepvar_sni": "deepvar",
    }.get(str(node.get("candidate")))
    if expected_model_type is not None:
        return document.get("model_type") == expected_model_type
    if node.get("candidate") in {"scenarioflow", "scenarioflow_graph"}:
        return _valid_scenarioflow_checkpoint(
            document, node, config, expected_variables
        )
    return False


def _valid_artifact(path: Path, node: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        if path.name == "results_current_panel.json":
            return _valid_result(path, node)
        if path.name == "window_scores.npz":
            return _valid_window_scores(path, node)
        if path.name == "best_model.pt":
            return _valid_checkpoint(path, node)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        EOFError,
        RuntimeError,
    ):
        return False
    return False


def _artifacts_are_valid(node: Mapping[str, Any]) -> bool:
    resolved = _completion_paths(node)
    return bool(resolved) and all(_valid_artifact(path, node) for path in resolved)


def _completion_paths(node: Mapping[str, Any]) -> list[Path]:
    return [
        Path(path) for path in node.get("resume", {}).get("completion_artifacts", [])
    ]


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def completion_is_valid(node: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Validate artifacts plus their exact current node/protocol completion record."""
    if not _artifacts_are_valid(node):
        return False
    sidecar = Path(str(node.get("sidecar_manifest", {}).get("path", "")))
    status_path = Path(str(node.get("destination", ""))) / "status.json"
    manifest = _read_json_object(sidecar)
    status = _read_json_object(status_path)
    if manifest is None or status is None:
        return False
    identity = {
        "node_id": node.get("id"),
        "protocol_dag_sha256": payload.get("protocol", {}).get("executable_dag_sha256"),
        "command_sha256": node.get("command_sha256"),
        "configuration_sha256": node.get("configuration_sha256"),
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    if any(status.get(key) != value for key, value in identity.items()):
        return False
    if (
        manifest.get("schema_version") != "fmts.node_run.v1"
        or status.get("schema_version") != "fmts.node_status.v1"
        or manifest.get("candidate") != node.get("candidate")
        or manifest.get("display_name") != node.get("display_name")
        or manifest.get("action") != node.get("action")
        or manifest.get("training_seed") != node.get("training_seed")
        or manifest.get("rollout_seed") != node.get("rollout_seed")
        or manifest.get("contract") != payload.get("protocol", {}).get("contract")
        or status.get("state") != "completed"
        or status.get("returncode") != 0
    ):
        return False
    artifacts = _completion_paths(node)
    expected_paths = {str(path.resolve()) for path in artifacts} | {
        str(sidecar.resolve())
    }
    recorded_hashes = status.get("artifact_hashes")
    if (
        not isinstance(recorded_hashes, Mapping)
        or set(recorded_hashes) != expected_paths
    ):
        return False
    return all(
        Path(path).is_file() and sha256_file(path) == expected
        for path, expected in recorded_hashes.items()
    )


def validate_output_roots(
    model_root: str | Path, result_root: str | Path
) -> tuple[Path, Path]:
    """Resolve output roots and reject overlap with authority, inputs, or source."""
    models = Path(model_root).expanduser().resolve()
    results = Path(result_root).expanduser().resolve()
    models = assert_isolated_output_root(
        REPOSITORY_ROOT, (*PROTECTED_RELEASE_PATHS, results), models
    )
    results = assert_isolated_output_root(
        REPOSITORY_ROOT, (*PROTECTED_RELEASE_PATHS, models), results
    )
    return models, results


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ensure_protocol_manifest(payload: Mapping[str, Any]) -> Path:
    """Persist the exact runnable graph consumed by the aggregator."""
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or not protocol.get("result_root"):
        raise BenchmarkExecutionError("protocol has no result root")
    path = Path(str(protocol["result_root"])) / "protocol_manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkExecutionError(
                f"existing protocol manifest is invalid: {path}"
            ) from error
        if existing != payload:
            raise BenchmarkExecutionError(
                f"protocol manifest does not match this run: {path}"
            )
    else:
        _write_json(path, payload)
    return path


def _record_completion(node: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    artifacts = _completion_paths(node)
    sidecar = Path(node["sidecar_manifest"]["path"])
    manifest = {
        "schema_version": "fmts.node_run.v1",
        "node_id": node["id"],
        "candidate": node["candidate"],
        "display_name": node["display_name"],
        "action": node["action"],
        "training_seed": node.get("training_seed"),
        "rollout_seed": node.get("rollout_seed"),
        "contract": payload["protocol"]["contract"],
        "protocol_dag_sha256": payload["protocol"]["executable_dag_sha256"],
        "command_sha256": node["command_sha256"],
        "configuration_sha256": node["configuration_sha256"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(sidecar, manifest)
    hashes = {str(path.resolve()): sha256_file(path) for path in artifacts}
    hashes[str(sidecar.resolve())] = sha256_file(sidecar)
    status = {
        "schema_version": "fmts.node_status.v1",
        "node_id": node["id"],
        "state": "completed",
        "returncode": 0,
        "protocol_dag_sha256": payload["protocol"]["executable_dag_sha256"],
        "command_sha256": node["command_sha256"],
        "configuration_sha256": node["configuration_sha256"],
        "artifact_hashes": hashes,
    }
    _write_json(Path(node["destination"]) / "status.json", status)


def execute_nodes(
    payload: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Execute dependency-ordered nodes, skipping only valid completions."""
    completed: set[str] = set()
    result: dict[str, Any] = {"completed": [], "skipped": []}
    for node in nodes:
        missing = set(map(str, node.get("depends_on", []))) - completed
        if missing:
            raise BenchmarkExecutionError(
                f"dependencies not completed for {node['id']}: {sorted(missing)}"
            )
        if completion_is_valid(node, payload):
            completed.add(str(node["id"]))
            result["skipped"].append(str(node["id"]))
            continue
        destination = Path(node["destination"])
        destination.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(node["command"], cwd=REPOSITORY_ROOT, check=False)
        if process.returncode != 0:
            raise BenchmarkExecutionError(
                f"node {node['id']} failed with exit code {process.returncode}"
            )
        if not _artifacts_are_valid(node):
            raise BenchmarkExecutionError(
                f"node {node['id']} omitted or corrupted a completion artifact"
            )
        _record_completion(node, payload)
        if not completion_is_valid(node, payload):
            raise BenchmarkExecutionError(
                f"node {node['id']} completion record failed validation"
            )
        completed.add(str(node["id"]))
        result["completed"].append(str(node["id"]))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--node-id")
    parser.add_argument("--resume-all", action="store_true")
    parser.add_argument(
        "--model-root", type=Path, default=REPOSITORY_ROOT / "build/reproduction/models"
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPOSITORY_ROOT / "build/reproduction/results",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-max-windows", type=int)
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-max-train-windows", type=int)
    parser.add_argument("--smoke-max-val-windows", type=int)
    parser.add_argument("--smoke-batch-size", type=int)
    parser.add_argument("--smoke-sample-steps", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.node_id and args.resume_all:
        raise SystemExit("choose --node-id or --resume-all, not both")
    if not args.dry_run and not (args.node_id or args.resume_all):
        raise SystemExit("execution requires --node-id or --resume-all")
    smoke_values = {
        "max_windows": args.smoke_max_windows,
        "samples": args.smoke_samples,
        "epochs": args.smoke_epochs,
        "max_train_windows": args.smoke_max_train_windows,
        "max_val_windows": args.smoke_max_val_windows,
        "batch_size": args.smoke_batch_size,
        "sample_steps": args.smoke_sample_steps,
    }
    smoke = (
        SmokeOverrides(**smoke_values)
        if any(value is not None for value in smoke_values.values())
        else None
    )
    model_root, result_root = validate_output_roots(args.model_root, args.result_root)
    payload = materialize_protocol(
        load_protocol(args.protocol),
        model_root=model_root,
        result_root=result_root,
        device=args.device,
        smoke=smoke,
    )
    nodes = (
        select_nodes(payload, args.node_id, include_dependencies=True)
        if args.node_id
        else topological_nodes(payload)
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "node_count": len(nodes),
                    "protocol_dag_sha256": payload["protocol"]["executable_dag_sha256"],
                    "nodes": [
                        {
                            "id": node["id"],
                            "depends_on": node.get("depends_on", []),
                            "command": node["command"],
                        }
                        for node in nodes
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = ensure_protocol_manifest(payload)
    completion = execute_nodes(payload, nodes)
    if args.resume_all:
        from .aggregate import aggregate_study

        aggregate_path = result_root / "aggregate"
        aggregate_study(
            protocol_manifest=manifest,
            output_dir=aggregate_path,
        )
        completion["aggregate_path"] = str(aggregate_path)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
