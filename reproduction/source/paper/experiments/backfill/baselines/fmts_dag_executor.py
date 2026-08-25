#!/usr/bin/env python
"""Fail-closed, resumable execution primitives for the FMTS command DAG."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


class DagExecutionError(RuntimeError):
    """Base class for fail-closed DAG execution errors."""


class NodeIntegrityError(DagExecutionError):
    """Raised before execution when a sealed node or input has drifted."""


class ExistingDestinationError(DagExecutionError):
    """Raised when an output directory is not a verified completed run."""


class DependencyNotCompletedError(DagExecutionError):
    """Raised when a selected node's prerequisites are not complete."""


class NodeExecutionError(DagExecutionError):
    """Raised when a command fails or omits a required artifact."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Bounded local scheduling policy; one GPU process is always the maximum."""

    max_parallel_jobs: int = 1
    max_cpu_jobs: int = 1
    max_gpu_jobs: int = 1

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_parallel_jobs) <= 4:
            raise ValueError("max_parallel_jobs must be between 1 and 4")
        if not 1 <= int(self.max_cpu_jobs) <= 4:
            raise ValueError("max_cpu_jobs must be between 1 and 4")
        if int(self.max_gpu_jobs) != 1:
            raise ValueError("max_gpu_jobs must be exactly 1")
        if int(self.max_cpu_jobs) > int(self.max_parallel_jobs):
            raise ValueError("max_cpu_jobs cannot exceed max_parallel_jobs")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def executable_dag_sha256(commands: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(list(commands))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_node_integrity(node: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying independent command/configuration/source hashes."""
    sealed = copy.deepcopy(dict(node))
    command = sealed.get("command")
    configuration = sealed.get("configuration")
    source = sealed.get("source_verification")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("node command must be a list of strings")
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError("node configuration must be a nonempty object")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("node source_verification must be a nonempty object")
    sealed["command_sha256"] = canonical_json_sha256(command)
    sealed["configuration_sha256"] = canonical_json_sha256(configuration)
    sealed["source_contract_sha256"] = canonical_json_sha256(source)
    return sealed


def _verify_node_integrity(node: Mapping[str, Any]) -> None:
    if node.get("command_sha256") != canonical_json_sha256(node.get("command")):
        raise NodeIntegrityError(f"command hash drifted for {node.get('id')}")
    if node.get("configuration_sha256") != canonical_json_sha256(
        node.get("configuration")
    ):
        raise NodeIntegrityError(f"configuration hash drifted for {node.get('id')}")
    if node.get("source_contract_sha256") != canonical_json_sha256(
        node.get("source_verification")
    ):
        raise NodeIntegrityError(f"source contract hash drifted for {node.get('id')}")


def _verify_source(
    node: Mapping[str, Any], preflight_context: Mapping[str, Any] | None
) -> None:
    source = node["source_verification"]
    kind = source.get("kind")
    if kind == "file_hashes":
        files = source.get("files")
        if not isinstance(files, Mapping) or not files:
            raise NodeIntegrityError(f"source hash manifest is empty for {node['id']}")
        for supplied_path, expected_hash in files.items():
            path = Path(str(supplied_path))
            if not path.is_file():
                raise NodeIntegrityError(f"source hash input is missing: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise NodeIntegrityError(
                    f"source hash drifted for {node['id']}: {path}"
                )
        if node.get("candidate") == "deepvar_sni":
            if not isinstance(preflight_context, Mapping):
                raise NodeIntegrityError(
                    f"transitive source preflight is unresolved for {node['id']}"
                )
            transitive = preflight_context.get("live_source_manifest")
            if not isinstance(transitive, Mapping):
                raise NodeIntegrityError(
                    f"transitive source manifest is missing for {node['id']}"
                )
            transitive_files = transitive.get("files")
            if (
                not isinstance(transitive_files, Mapping)
                or not transitive_files
                or canonical_json_sha256(transitive_files)
                != transitive.get("manifest_sha256")
            ):
                raise NodeIntegrityError(
                    f"transitive source manifest is invalid for {node['id']}"
                )
            for supplied_path, expected_hash in transitive_files.items():
                path = Path(str(supplied_path))
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise NodeIntegrityError(
                        f"transitive source hash drifted for {node['id']}: {path}"
                    )
        return
    if kind == "overlay_registry":
        if not isinstance(preflight_context, Mapping):
            raise NodeIntegrityError(
                f"overlay preflight is unresolved for {node['id']}"
            )
        if preflight_context.get("manifest_sha256") != source.get("manifest_sha256"):
            raise NodeIntegrityError(f"overlay manifest hash drifted for {node['id']}")
        overlay_name = source.get("overlay_name")
        overlay = (preflight_context.get("overlays") or {}).get(overlay_name)
        if not isinstance(overlay, Mapping):
            raise NodeIntegrityError(
                f"overlay preflight is missing {overlay_name!r} for {node['id']}"
            )
        if overlay.get("base_revision") != source.get("base_revision"):
            raise NodeIntegrityError(f"overlay base revision drifted for {node['id']}")
        if overlay.get("worktree") != node.get("working_directory"):
            raise NodeIntegrityError(f"overlay worktree drifted for {node['id']}")
        if overlay.get("source_hashes") != source.get("expected_source_hashes"):
            raise NodeIntegrityError(f"overlay source hashes drifted for {node['id']}")
        if overlay.get("unexpected_changes"):
            raise NodeIntegrityError(f"overlay has unexpected changes for {node['id']}")
        if not overlay.get("cli_accepts_save_window_scores"):
            raise NodeIntegrityError(f"overlay CLI preflight failed for {node['id']}")
        return
    raise NodeIntegrityError(
        f"unsupported or unresolved source verification for {node['id']}: {kind!r}"
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExistingDestinationError(
            f"existing destination is not a verified completion: {path.parent}"
        ) from error
    if not isinstance(payload, dict):
        raise ExistingDestinationError(
            f"existing destination is not a verified completion: {path.parent}"
        )
    return payload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_worktree_state(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    head = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        return {
            "path": str(resolved),
            "git_available": False,
            "error": head.stderr.strip() or "not a Git worktree",
        }
    status = subprocess.run(
        [
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise NodeIntegrityError(
            f"cannot capture dirty-worktree state for {resolved}: {status.stderr.strip()}"
        )
    porcelain = status.stdout
    return {
        "path": str(resolved),
        "git_available": True,
        "head": head.stdout.strip(),
        "dirty": bool(porcelain),
        "status_porcelain_v1": porcelain,
        "status_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def _runtime_environment() -> dict[str, Any]:
    torch_metadata: dict[str, Any] = {
        "version": None,
        "cuda_runtime_version": None,
        "cuda_available": False,
        "cudnn_version": None,
        "cuda_devices": [],
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        torch_metadata = {
            "version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_devices": devices,
        }
    except Exception as error:  # environment capture must also work on CPU hosts
        torch_metadata["import_error"] = repr(error)

    nvidia_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            nvidia_command,
            check=False,
            capture_output=True,
            text=True,
        )
        gpu_rows = []
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                values = [value.strip() for value in line.split(",", maxsplit=4)]
                if len(values) == 5:
                    gpu_rows.append(
                        {
                            "index": values[0],
                            "uuid": values[1],
                            "name": values[2],
                            "driver_version": values[3],
                            "memory_total_mib": values[4],
                        }
                    )
        nvidia_smi = {
            "command": nvidia_command,
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "gpus": gpu_rows,
        }
        if completed.returncode != 0:
            nvidia_smi["error"] = completed.stderr.strip()
    except OSError as error:
        nvidia_smi = {
            "command": nvidia_command,
            "available": False,
            "error": repr(error),
            "gpus": [],
        }

    return {
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "torch": torch_metadata,
        "nvidia_smi": nvidia_smi,
    }


def _ensure_before_run_snapshot(payload: Mapping[str, Any]) -> dict[str, str]:
    protocol = payload["protocol"]
    result_root = Path(str(protocol["result_root"])).resolve()
    path = result_root / "runtime_environment_before.json"
    if path.exists():
        snapshot = _read_json(path)
        if (
            snapshot.get("schema_version") != "fmts.runtime_environment.v1"
            or snapshot.get("protocol_dag_sha256") != protocol["executable_dag_sha256"]
        ):
            raise ExistingDestinationError(
                f"runtime environment snapshot does not match this DAG: {path}"
            )
    else:
        commands = payload["commands"]
        worktree_paths = {Path.cwd().resolve()}
        worktree_paths.update(
            Path(str(node["working_directory"])).resolve() for node in commands
        )
        snapshot = {
            "schema_version": "fmts.runtime_environment.v1",
            "captured_at_utc": _utc_now(),
            "captured_at_monotonic_ns": time.monotonic_ns(),
            "protocol_dag_sha256": protocol["executable_dag_sha256"],
            "runtime": _runtime_environment(),
            "worktrees": {
                str(worktree): _git_worktree_state(worktree)
                for worktree in sorted(worktree_paths, key=str)
            },
        }
        result_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, snapshot)
    return {
        "before_snapshot_path": str(path),
        "before_snapshot_sha256": sha256_file(path),
    }


def _requested_device(node: Mapping[str, Any]) -> str | None:
    command = node["command"]
    for index, argument in enumerate(command):
        if argument == "--device" and index + 1 < len(command):
            return str(command[index + 1])
        if argument.startswith("--device="):
            return argument.partition("=")[2]
    return None


def _protocol_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = payload["protocol"]
    source_preflight = copy.deepcopy(payload.get("source_preflight"))
    if isinstance(source_preflight, dict):
        overlays = source_preflight.get("overlays")
        if isinstance(overlays, dict):
            for overlay in overlays.values():
                if isinstance(overlay, dict):
                    overlay.pop("runtime_source_hashes", None)
                    overlay.pop("source_amendment", None)
    return {
        "schema_version": "fmts.protocol_run.v1",
        "protocol_dag_sha256": protocol["executable_dag_sha256"],
        "command_count": payload["command_count"],
        "protocol": protocol,
        "source_preflight": source_preflight,
    }


def _ensure_protocol_manifest(payload: Mapping[str, Any]) -> Path:
    result_root = Path(str(payload["protocol"]["result_root"]))
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "protocol_manifest.json"
    expected = _protocol_manifest(payload)
    if path.exists():
        if _read_json(path) != expected:
            raise ExistingDestinationError(
                f"protocol manifest does not match this DAG: {path}"
            )
    else:
        _atomic_json(path, expected)
    return path


def _status_path(node: Mapping[str, Any]) -> Path:
    return Path(str(node["destination"])) / "status.json"


def _run_manifest_path(node: Mapping[str, Any]) -> Path:
    return Path(str(node["destination"])) / "run_manifest.json"


def _completed_status(
    node: Mapping[str, Any], protocol_digest: str, *, verify_artifacts: bool
) -> dict[str, Any]:
    destination = Path(str(node["destination"]))
    status_path = _status_path(node)
    manifest_path = _run_manifest_path(node)
    if (
        not destination.is_dir()
        or not status_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ExistingDestinationError(
            f"existing destination is not a verified completion: {destination}"
        )
    status = _read_json(status_path)
    manifest = _read_json(manifest_path)
    expected = {
        "node_id": node["id"],
        "protocol_dag_sha256": protocol_digest,
        "command_sha256": node["command_sha256"],
        "configuration_sha256": node["configuration_sha256"],
    }
    if status.get("state") != "completed" or any(
        status.get(key) != value for key, value in expected.items()
    ):
        raise ExistingDestinationError(
            f"existing destination is not a verified completion: {destination}"
        )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ExistingDestinationError(
            f"run manifest does not match completed node: {destination}"
        )
    artifacts = status.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ExistingDestinationError(
            f"completed status has no artifact hashes: {destination}"
        )
    if verify_artifacts:
        for supplied_path, expected_hash in artifacts.items():
            path = Path(str(supplied_path))
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise NodeIntegrityError(
                    f"completed artifact hash drifted for {node['id']}: {path}"
                )
    return status


def _dependency_statuses(
    node: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    protocol_digest: str,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for dependency in node.get("depends_on", []):
        dependency_node = nodes.get(dependency)
        if dependency_node is None:
            raise DependencyNotCompletedError(
                f"unknown dependency {dependency!r} for {node['id']}"
            )
        try:
            statuses[dependency] = _completed_status(
                dependency_node, protocol_digest, verify_artifacts=False
            )
        except ExistingDestinationError as error:
            raise DependencyNotCompletedError(
                f"dependency {dependency} is not completed for {node['id']}"
            ) from error
    return statuses


def _verify_input_checkpoint(
    node: Mapping[str, Any], dependency_statuses: Mapping[str, Mapping[str, Any]]
) -> dict[str, str] | None:
    supplied = node.get("input_checkpoint")
    if supplied is None:
        return None
    checkpoint = Path(str(supplied))
    expected_hash: str | None = None
    for status in dependency_statuses.values():
        value = (status.get("artifact_hashes") or {}).get(str(checkpoint))
        if isinstance(value, str):
            expected_hash = value
            break
    if expected_hash is None:
        raise NodeIntegrityError(
            f"checkpoint hash is absent from dependencies for {node['id']}: {checkpoint}"
        )
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
        raise NodeIntegrityError(
            f"checkpoint hash drifted for {node['id']}: {checkpoint}"
        )
    return {"path": str(checkpoint), "sha256": expected_hash}


def _run_manifest(
    node: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    started_at: str,
    input_checkpoint: Mapping[str, str] | None,
    provenance_reference: Mapping[str, str],
    runtime_source_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "fmts.node_run.v1",
        "node_id": node["id"],
        "candidate": node["candidate"],
        "display_name": node.get("display_name"),
        "action": node["action"],
        "protocol_dag_sha256": payload["protocol"]["executable_dag_sha256"],
        "command": node["command"],
        "command_sha256": node["command_sha256"],
        "configuration": node["configuration"],
        "configuration_sha256": node["configuration_sha256"],
        "source_verification": node["source_verification"],
        "source_contract_sha256": node["source_contract_sha256"],
        "working_directory": node["working_directory"],
        "source_revision": node["source_revision"],
        "contract": payload["protocol"].get("contract"),
        "resource": node["resource"],
        "requested_device": _requested_device(node),
        "runtime_provenance": {
            **dict(provenance_reference),
            "worktree": str(Path(str(node["working_directory"])).resolve()),
        },
        "started_at_utc": started_at,
    }
    for key in (
        "training_seed",
        "rollout_seed",
        "stage",
        "pair_id",
        "produced_checkpoint",
    ):
        if key in node:
            manifest[key] = node[key]
    if input_checkpoint is not None:
        manifest["input_checkpoint"] = dict(input_checkpoint)
    if runtime_source_verification is not None:
        manifest["runtime_source_verification"] = dict(runtime_source_verification)
    return manifest


def _runtime_source_verification(
    node: Mapping[str, Any], preflight_context: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Expose a narrowly approved runtime source amendment in each new run."""
    source = node.get("source_verification")
    if not isinstance(source, Mapping) or source.get("kind") != "overlay_registry":
        return None
    if not isinstance(preflight_context, Mapping):
        return None
    overlays = preflight_context.get("overlays")
    if not isinstance(overlays, Mapping):
        return None
    overlay_name = source.get("overlay_name")
    overlay = overlays.get(overlay_name)
    if not isinstance(overlay, Mapping):
        return None
    amendment = overlay.get("source_amendment")
    runtime_hashes = overlay.get("runtime_source_hashes")
    if amendment is None and runtime_hashes is None:
        return None
    if not isinstance(amendment, Mapping) or not isinstance(runtime_hashes, Mapping):
        raise NodeIntegrityError(
            f"runtime source amendment is malformed for {node.get('id')}"
        )
    if amendment.get("expected_sha256") != source.get("expected_source_hashes", {}).get(
        "window_score_helper_sha256"
    ):
        raise NodeIntegrityError(
            f"runtime source amendment base hash drifted for {node.get('id')}"
        )
    if amendment.get("actual_sha256") != runtime_hashes.get(
        "window_score_helper_sha256"
    ):
        raise NodeIntegrityError(
            f"runtime source amendment hash drifted for {node.get('id')}"
        )
    return {
        "overlay_name": overlay_name,
        "runtime_source_hashes": dict(runtime_hashes),
        "source_amendment": dict(amendment),
    }


def _execute_or_skip(
    node: Mapping[str, Any],
    payload: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    *,
    preflight_context: Mapping[str, Any] | None,
    provenance_reference: Mapping[str, str],
) -> str:
    protocol_digest = str(payload["protocol"]["executable_dag_sha256"])
    destination = Path(str(node["destination"]))
    if destination.exists():
        _completed_status(node, protocol_digest, verify_artifacts=True)
        return "skipped"

    _verify_node_integrity(node)
    _verify_source(node, preflight_context)
    runtime_source_verification = _runtime_source_verification(node, preflight_context)
    dependencies = _dependency_statuses(node, nodes, protocol_digest)
    input_checkpoint = _verify_input_checkpoint(node, dependencies)

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ExistingDestinationError(
            f"existing destination is not a verified completion: {destination}"
        ) from error

    started_monotonic_ns = time.monotonic_ns()
    started_at = _utc_now()
    manifest_path = _run_manifest_path(node)
    status_path = _status_path(node)
    log_path = destination / "stdout_stderr.log"
    log_temporary = destination / "stdout_stderr.log.tmp"
    manifest = _run_manifest(
        node,
        payload,
        started_at=started_at,
        input_checkpoint=input_checkpoint,
        provenance_reference=provenance_reference,
        runtime_source_verification=runtime_source_verification,
    )
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        status_path,
        {
            "schema_version": "fmts.node_status.v1",
            "node_id": node["id"],
            "state": "running",
            "protocol_dag_sha256": protocol_digest,
            "command_sha256": node["command_sha256"],
            "configuration_sha256": node["configuration_sha256"],
            "started_at_utc": started_at,
        },
    )

    returncode: int | None = None
    execution_error: Exception | None = None
    try:
        environment = os.environ.copy()
        environment.update(
            {str(key): str(value) for key, value in node.get("environment", {}).items()}
        )
        with log_temporary.open("wb") as log_handle:
            completed = subprocess.run(
                list(node["command"]),
                cwd=str(node["working_directory"]),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = int(completed.returncode)
            log_handle.flush()
            os.fsync(log_handle.fileno())
    except Exception as error:  # preserve a durable failed status for launch errors
        execution_error = error
    finally:
        if log_temporary.exists():
            log_temporary.replace(log_path)

    ended_at = _utc_now()
    wall_clock_seconds = (time.monotonic_ns() - started_monotonic_ns) / 1_000_000_000
    manifest["ended_at_utc"] = ended_at
    manifest["wall_clock_seconds"] = wall_clock_seconds
    manifest["returncode"] = returncode
    if execution_error is not None:
        manifest["launch_error"] = repr(execution_error)
    _atomic_json(manifest_path, manifest)

    if execution_error is not None or returncode != 0:
        failure = {
            "schema_version": "fmts.node_status.v1",
            "node_id": node["id"],
            "state": "failed",
            "protocol_dag_sha256": protocol_digest,
            "command_sha256": node["command_sha256"],
            "configuration_sha256": node["configuration_sha256"],
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "wall_clock_seconds": wall_clock_seconds,
            "returncode": returncode,
            "error": repr(execution_error) if execution_error else None,
        }
        _atomic_json(status_path, failure)
        if execution_error is not None:
            raise NodeExecutionError(
                f"node {node['id']} failed to launch: {execution_error}"
            ) from execution_error
        raise NodeExecutionError(
            f"node {node['id']} exited with exit code {returncode}"
        )

    completion_artifacts = [
        Path(str(path)) for path in node["resume"]["completion_artifacts"]
    ]
    missing = [path for path in completion_artifacts if not path.is_file()]
    if missing:
        _atomic_json(
            status_path,
            {
                "schema_version": "fmts.node_status.v1",
                "node_id": node["id"],
                "state": "failed",
                "protocol_dag_sha256": protocol_digest,
                "command_sha256": node["command_sha256"],
                "configuration_sha256": node["configuration_sha256"],
                "started_at_utc": started_at,
                "ended_at_utc": ended_at,
                "wall_clock_seconds": wall_clock_seconds,
                "returncode": 0,
                "error": "missing completion artifacts: "
                + ", ".join(str(path) for path in missing),
            },
        )
        raise NodeExecutionError(
            f"node {node['id']} omitted completion artifacts: {missing}"
        )

    hashed_paths = [*completion_artifacts, manifest_path]
    if log_path.is_file():
        hashed_paths.append(log_path)
    artifact_hashes = {str(path): sha256_file(path) for path in hashed_paths}
    _atomic_json(
        status_path,
        {
            "schema_version": "fmts.node_status.v1",
            "node_id": node["id"],
            "state": "completed",
            "protocol_dag_sha256": protocol_digest,
            "command_sha256": node["command_sha256"],
            "configuration_sha256": node["configuration_sha256"],
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "wall_clock_seconds": wall_clock_seconds,
            "returncode": 0,
            "artifact_hashes": artifact_hashes,
        },
    )
    return "completed"


def _topological_order(nodes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(node["id"]): node for node in nodes}
    if len(by_id) != len(nodes):
        raise NodeIntegrityError("command DAG has duplicate node IDs")
    remaining = set(by_id)
    completed: set[str] = set()
    order: list[Mapping[str, Any]] = []
    while remaining:
        ready = [
            node_id
            for node_id in by_id
            if node_id in remaining
            and set(by_id[node_id].get("depends_on", [])).issubset(completed)
        ]
        if not ready:
            raise NodeIntegrityError(
                "command DAG contains a cycle or unknown dependency"
            )
        for node_id in ready:
            remaining.remove(node_id)
            completed.add(node_id)
            order.append(by_id[node_id])
    return order


def execute_command_dag(
    payload: Mapping[str, Any],
    *,
    node_id: str | None = None,
    resume_all: bool = False,
    policy: ExecutionPolicy | None = None,
    preflight_context: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Execute one node or resume the DAG; default scheduling is fully serial."""
    policy = policy or ExecutionPolicy()
    if (node_id is None) == (not resume_all):
        raise ValueError("choose exactly one of node_id or resume_all")
    commands = payload.get("commands")
    protocol = payload.get("protocol")
    if not isinstance(commands, list) or not isinstance(protocol, Mapping):
        raise NodeIntegrityError("execution payload is missing protocol or commands")
    if protocol.get("executable_dag_sha256") != executable_dag_sha256(commands):
        raise NodeIntegrityError("executable DAG digest does not match its commands")
    for node in commands:
        _verify_node_integrity(node)
        resource = node.get("resource")
        if resource not in {"cpu", "gpu"}:
            raise NodeIntegrityError(f"invalid resource class for {node.get('id')}")
    provenance_reference = _ensure_before_run_snapshot(payload)
    _ensure_protocol_manifest(payload)

    nodes = {str(node["id"]): node for node in commands}
    if len(nodes) != len(commands):
        raise NodeIntegrityError("command DAG has duplicate node IDs")
    selected: list[Mapping[str, Any]]
    if node_id is not None:
        if node_id not in nodes:
            raise KeyError(f"unknown node ID: {node_id}")
        selected = [nodes[node_id]]
    else:
        selected = _topological_order(commands)

    result = {"completed": [], "skipped": []}
    if node_id is not None or policy.max_parallel_jobs == 1:
        for node in selected:
            state = _execute_or_skip(
                node,
                payload,
                nodes,
                preflight_context=preflight_context,
                provenance_reference=provenance_reference,
            )
            result[state].append(str(node["id"]))
        return result

    pending = {str(node["id"]): node for node in selected}
    finished: set[str] = set()
    running: dict[Future[str], Mapping[str, Any]] = {}
    running_cpu = 0
    running_gpu = 0
    first_error: BaseException | None = None

    with ThreadPoolExecutor(max_workers=policy.max_parallel_jobs) as pool:
        while pending or running:
            if first_error is None:
                for selected_node in selected:
                    selected_id = str(selected_node["id"])
                    if selected_id not in pending:
                        continue
                    dependencies = set(selected_node.get("depends_on", []))
                    if not dependencies.issubset(finished):
                        continue
                    resource = str(selected_node["resource"])
                    if len(running) >= policy.max_parallel_jobs:
                        break
                    if resource == "gpu" and running_gpu >= policy.max_gpu_jobs:
                        continue
                    if resource == "cpu" and running_cpu >= policy.max_cpu_jobs:
                        continue
                    future = pool.submit(
                        _execute_or_skip,
                        selected_node,
                        payload,
                        nodes,
                        preflight_context=preflight_context,
                        provenance_reference=provenance_reference,
                    )
                    running[future] = selected_node
                    pending.pop(selected_id)
                    if resource == "gpu":
                        running_gpu += 1
                    else:
                        running_cpu += 1

            if not running:
                if first_error is not None:
                    raise first_error
                raise NodeIntegrityError(
                    "bounded scheduler cannot make progress on the command DAG"
                )

            completed_futures, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                completed_node = running.pop(future)
                resource = str(completed_node["resource"])
                if resource == "gpu":
                    running_gpu -= 1
                else:
                    running_cpu -= 1
                try:
                    state = future.result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    continue
                completed_id = str(completed_node["id"])
                finished.add(completed_id)
                result[state].append(completed_id)

        if first_error is not None:
            raise first_error
    return result
