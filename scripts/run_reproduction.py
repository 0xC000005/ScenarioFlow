#!/usr/bin/env python3
"""Run the archived paper DAG with its original executor and preserved sources."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = ROOT / "results/frozen/multiseed/command-dag.json"
SOURCE_MANIFEST = ROOT / "reproduction/SOURCE_MANIFEST.tsv"
EXECUTOR_ROOT = ROOT / "reproduction/source/paper"
RESULT_ROOT = ROOT / "build/reproduction/results"
PATH_REPLACEMENTS = (
    (
        "frozen-source://models/backfill/fmts_multiseed_20260805",
        str(ROOT / "build/reproduction/checkpoints"),
    ),
    (
        "frozen-source://results/fmts_multiseed_20260805",
        str(RESULT_ROOT),
    ),
    (
        "legacy-worktree://fmts-window-scores-scenarioflow-5d13",
        str(ROOT / "reproduction/source/scenarioflow"),
    ),
    (
        "legacy-worktree://fmts-window-scores-paper-25b962",
        str(ROOT / "reproduction/source/baselines"),
    ),
    (
        "legacy-worktree://fmts-window-scores-garch-da91172",
        str(ROOT / "reproduction/source/garch_ccc"),
    ),
    ("frozen-source://", str(ROOT / "reproduction/source/deepvar_sni") + "/"),
)

sys.path.insert(0, str(EXECUTOR_ROOT))
from experiments.backfill.baselines.fmts_dag_executor import (  # noqa: E402
    ExecutionPolicy,
    canonical_json_sha256,
    executable_dag_sha256,
    execute_command_dag,
    seal_node_integrity,
)


class ReproductionError(RuntimeError):
    """Raised when a preserved source or archived command fails validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> int:
    count = 0
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = ROOT / "reproduction/source" / row["snapshot"] / row["path"]
            if not path.is_file():
                raise ReproductionError(f"preserved source is missing: {path}")
            if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                raise ReproductionError(f"preserved source changed: {path}")
            count += 1
    return count


def map_path(value: str) -> str:
    mapped = value
    for old, new in PATH_REPLACEMENTS:
        mapped = mapped.replace(old, new)
    return mapped


def map_paths(value: Any) -> Any:
    if isinstance(value, str):
        return map_path(value)
    if isinstance(value, list):
        return [map_paths(item) for item in value]
    if isinstance(value, dict):
        return {map_path(str(key)): map_paths(item) for key, item in value.items()}
    return value


def _release_source_contract(node: dict[str, Any]) -> None:
    source = copy.deepcopy(node["source_verification"])
    kind = source.get("kind")
    if kind == "file_hashes":
        files = {
            map_path(str(path)): sha256(Path(map_path(str(path))))
            for path in source["files"]
        }
        source["files"] = files
        source["manifest_sha256"] = canonical_json_sha256(files)
    elif kind == "overlay_registry":
        hashes = dict(source["expected_source_hashes"])
        worktree = Path(node["working_directory"])
        evaluator = (
            "evaluate_sni_current_panel_metrics.py"
            if source["overlay_name"] == "scenarioflow"
            else "evaluate_current_panel_baselines.py"
        )
        hashes["patched_evaluator_sha256"] = sha256(
            worktree / "experiments/backfill/baselines" / evaluator
        )
        hashes["window_score_helper_sha256"] = sha256(
            worktree / "analysis_code/window_score_artifact.py"
        )
        source["expected_source_hashes"] = hashes
        source["manifest_path"] = str(SOURCE_MANIFEST)
        source["manifest_sha256"] = sha256(SOURCE_MANIFEST)
    else:
        raise ReproductionError(
            f"unsupported archived source contract for {node['id']}: {kind!r}"
        )
    node["source_verification"] = source


def _preflight_context(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    overlays: dict[str, Any] = {}
    live_files: dict[str, str] = {}
    for node in nodes:
        source = node["source_verification"]
        if source["kind"] == "overlay_registry":
            name = source["overlay_name"]
            candidate = {
                "base_revision": source["base_revision"],
                "worktree": node["working_directory"],
                "source_hashes": source["expected_source_hashes"],
                "unexpected_changes": [],
                "cli_accepts_save_window_scores": True,
            }
            previous = overlays.setdefault(name, candidate)
            if previous != candidate:
                raise ReproductionError(f"inconsistent source contract for {name}")
        elif node.get("candidate") == "deepvar_sni":
            live_files.update(source["files"])
    manifest_hash = sha256(SOURCE_MANIFEST)
    return {
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "overlay_manifest_sha256": manifest_hash,
        "overlay_manifest_path": str(SOURCE_MANIFEST),
        "overlays": overlays,
        "live_source_manifest": {
            "files": live_files,
            "manifest_sha256": canonical_json_sha256(live_files),
        },
        "live_source_manifest_sha256": canonical_json_sha256(live_files),
        "live_source_manifest_location": "reproduction/SOURCE_MANIFEST.tsv",
    }


def materialize_payload() -> dict[str, Any]:
    archived = json.loads(DAG_PATH.read_text(encoding="utf-8"))
    payload = map_paths(archived)
    nodes: list[dict[str, Any]] = []
    for archived_node in payload["commands"]:
        node = dict(archived_node)
        if node["command"][0] == "python":
            node["command"][0] = sys.executable
        _release_source_contract(node)
        nodes.append(seal_node_integrity(node))
    payload["commands"] = nodes
    payload["command_count"] = len(nodes)
    payload["mode"] = "release"
    payload["execution_supported"] = True
    payload["protocol"]["executable_dag_sha256"] = executable_dag_sha256(nodes)
    payload["protocol_dag_sha256"] = payload["protocol"]["executable_dag_sha256"]
    payload["source_preflight"] = _preflight_context(nodes)
    return payload


def topological_order(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ReproductionError("archived DAG has duplicate node IDs")
    complete: set[str] = set()
    ordered: list[dict[str, Any]] = []
    while len(ordered) < len(nodes):
        ready = [
            node
            for node in nodes
            if node["id"] not in complete
            and set(node.get("depends_on", [])).issubset(complete)
        ]
        if not ready:
            raise ReproductionError("archived DAG has a cycle or missing dependency")
        for node in ready:
            complete.add(node["id"])
            ordered.append(node)
    return ordered


def selected_nodes(
    ordered: list[dict[str, Any]], node_id: str | None
) -> list[dict[str, Any]]:
    if node_id is None:
        return ordered
    by_id = {node["id"]: node for node in ordered}
    if node_id not in by_id:
        raise ReproductionError(f"unknown archived node: {node_id}")
    required: set[str] = set()

    def add(identifier: str) -> None:
        if identifier in required:
            return
        for dependency in by_id[identifier].get("depends_on", []):
            add(dependency)
        required.add(identifier)

    add(node_id)
    return [node for node in ordered if node["id"] in required]


def ensure_command_dag(payload: dict[str, Any]) -> Path:
    path = RESULT_ROOT / "command_dag.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ReproductionError(f"existing release DAG does not match: {path}")
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return path


def print_commands(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        print(f"{node['id']}: {' '.join(node['command'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id")
    parser.add_argument("--resume-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--max-parallel-jobs", type=int, default=1)
    parser.add_argument("--max-cpu-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.node_id and args.resume_all:
        parser.error("choose --node-id or --resume-all")
    count = verify_sources()
    if args.verify_sources and not args.node_id and not args.resume_all:
        print(f"verified {count} preserved source files")
        return 0
    if not args.node_id and not args.resume_all:
        parser.error("choose --node-id, --resume-all, or --verify-sources")
    payload = materialize_payload()
    ordered = topological_order(payload["commands"])
    selected = selected_nodes(ordered, args.node_id)
    if args.dry_run:
        print_commands(selected)
        return 0
    ensure_command_dag(payload)
    policy = ExecutionPolicy(
        max_parallel_jobs=args.max_parallel_jobs,
        max_cpu_jobs=args.max_cpu_jobs,
        max_gpu_jobs=1,
    )
    if args.node_id:
        for node in selected:
            execute_command_dag(
                payload,
                node_id=node["id"],
                policy=policy,
                preflight_context=payload["source_preflight"],
            )
    else:
        execute_command_dag(
            payload,
            resume_all=True,
            policy=policy,
            preflight_context=payload["source_preflight"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
