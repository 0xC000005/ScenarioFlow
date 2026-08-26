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

from .protocol import (
    DEFAULT_PROTOCOL,
    SmokeOverrides,
    load_protocol,
    materialize_protocol,
    select_nodes,
    topological_nodes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class BenchmarkExecutionError(RuntimeError):
    """Raised when a node fails or claims an invalid completion artifact."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_artifact(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == ".json":
            return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return bool(payload.files)
        if path.suffix == ".pt":
            return isinstance(
                torch.load(path, map_location="cpu", weights_only=False), Mapping
            )
    except (OSError, ValueError, json.JSONDecodeError, EOFError, RuntimeError):
        return False
    return True


def completion_is_valid(paths: Sequence[str | Path]) -> bool:
    """Return true only when every declared completion artifact can be loaded."""
    resolved = [Path(path) for path in paths]
    return bool(resolved) and all(_valid_artifact(path) for path in resolved)


def _completion_paths(node: Mapping[str, Any]) -> list[Path]:
    return [
        Path(path) for path in node.get("resume", {}).get("completion_artifacts", [])
    ]


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
) -> dict[str, list[str]]:
    """Execute dependency-ordered nodes, skipping only valid completions."""
    completed: set[str] = set()
    result: dict[str, list[str]] = {"completed": [], "skipped": []}
    for node in nodes:
        missing = set(map(str, node.get("depends_on", []))) - completed
        if missing:
            raise BenchmarkExecutionError(
                f"dependencies not completed for {node['id']}: {sorted(missing)}"
            )
        artifacts = _completion_paths(node)
        if completion_is_valid(artifacts):
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
        if not completion_is_valid(artifacts):
            raise BenchmarkExecutionError(
                f"node {node['id']} omitted or corrupted a completion artifact"
            )
        _record_completion(node, payload)
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
    payload = materialize_protocol(
        load_protocol(args.protocol),
        model_root=args.model_root,
        result_root=args.result_root,
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
    ensure_protocol_manifest(payload)
    print(json.dumps(execute_nodes(payload, nodes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
