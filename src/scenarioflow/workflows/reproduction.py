"""Single native entry point for planning, resuming, and aggregating reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..baselines.deep import build_model, save_checkpoint
from ..evaluation.current_panel import (
    METRIC_PATHS,
    ResultContractError,
    prepare_validation_blocks,
    validate_result_schema,
)
from ..state import normalize_increment_windows

from ..evaluation.garch_ccc_contract import (
    EXPECTED_SOURCE_BLOB,
    EXPECTED_SOURCE_REVISION,
    FROZEN_SUMMARY_SHA256,
    FROZEN_REPLAY_MANIFEST_SHA256,
    FROZEN_REPLAY_RESULT_SHA256,
    FROZEN_TABLE_SHA256,
    REPORT_SCHEMA,
    SOURCE_TAG,
)

from .multiseed import (
    CANDIDATE_IDS,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    ProtocolValidationError,
    FROZEN_ARTIFACT_ROWS,
    FROZEN_SOURCE_ROWS,
    _validate_manifest_rows,
    aggregate_results,
    build_protocol,
)

RUN_MANIFEST_SCHEMA = "scenarioflow.multiseed-node.v1"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _result_json(payload: Mapping[str, Any]) -> str:
    portable = json.loads(
        json.dumps(
            payload,
            allow_nan=True,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else value.tolist()
            ),
        )
    )
    return json.dumps(portable, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_identity(
    *,
    candidate: str,
    train_seed: int | None,
    rollout_seed: int,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    candidate_ids = tuple(protocol.get("candidate_ids", CANDIDATE_IDS))
    rollout_seeds = tuple(protocol.get("rollout_seeds", ROLLOUT_SEEDS))
    train_seeds = tuple(protocol.get("train_seeds", TRAIN_SEEDS))
    if candidate not in candidate_ids:
        raise ProtocolValidationError(f"unknown candidate {candidate!r}")
    if rollout_seed not in rollout_seeds and not (
        candidate == "garch_ccc" and rollout_seed == 772
    ):
        raise ProtocolValidationError(
            f"rollout seed must be one of {list(ROLLOUT_SEEDS)}"
        )
    expects_training = candidate not in {
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "garch_ccc",
        "filtered_hs",
    }
    if expects_training and train_seed not in train_seeds:
        raise ProtocolValidationError(
            f"training seed must be one of {list(TRAIN_SEEDS)}"
        )
    if not expects_training and train_seed is not None:
        raise ProtocolValidationError(f"{candidate} must not invent a training seed")
    if result.get("baseline_name") != candidate:
        raise ProtocolValidationError("result candidate identity mismatch")
    if result.get("mode") != "current_realvix_39_state_panel":
        raise ProtocolValidationError("result is not the published validation protocol")
    validation = protocol.get("validation_protocol", {})
    expected_windows = int(validation.get("val_size", 441))
    expected_samples = int(validation.get("n_samples", 64))
    if (
        result.get("n_windows") != expected_windows
        or result.get("n_samples") != expected_samples
    ):
        raise ProtocolValidationError(
            "result deck dimensions do not match the protocol"
        )


def _verify_protocol_authorities(protocol: Mapping[str, Any]) -> None:
    contract = {
        key: value
        for key, value in protocol.items()
        if key not in {"contract_hash", "output_root", "nodes"}
    }
    if protocol.get("contract_hash") != _digest_text(_canonical_json(contract)):
        raise ProtocolValidationError("protocol contract hash drifted")
    for key, expected, label in (
        ("source_manifest", FROZEN_SOURCE_ROWS, "source"),
        ("artifact_manifest", FROZEN_ARTIFACT_ROWS, "artifact"),
    ):
        authority = protocol.get(key)
        if not isinstance(authority, Mapping):
            raise ProtocolValidationError(f"protocol has no {label} manifest authority")
        path = Path(str(authority.get("path", "")))
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != authority.get("sha256"):
            raise ProtocolValidationError(f"{label} manifest checksum drifted")
        verified = _validate_manifest_rows(path, expected, label=label)
        if protocol.get(f"{label}_authority") != verified:
            raise ProtocolValidationError(f"protocol {label} authority drifted")


def _validate_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    node: Mapping[str, Any],
) -> None:
    validation = protocol["validation_protocol"]
    if manifest.get("mode") != protocol.get("mode"):
        raise ProtocolValidationError("evaluation manifest mode drifted")
    if manifest.get("benchmark_split") != protocol.get("benchmark_split"):
        raise ProtocolValidationError("evaluation manifest benchmark split drifted")
    if manifest.get("n_val_windows") != validation.get("val_size"):
        raise ProtocolValidationError("evaluation manifest validation count drifted")
    if manifest.get("n_samples") != validation.get("n_samples"):
        raise ProtocolValidationError("evaluation manifest sample count drifted")
    if manifest.get("seed") != node.get("rollout_seed"):
        raise ProtocolValidationError("evaluation manifest seed drifted")
    columns = manifest.get("columns")
    metadata = manifest.get("metadata")
    if (
        not isinstance(columns, list)
        or not columns
        or not isinstance(metadata, Mapping)
    ):
        raise ProtocolValidationError("evaluation panel metadata is malformed")
    if metadata.get("source_columns") != columns:
        raise ProtocolValidationError("evaluation source columns drifted")
    if not all(
        isinstance(metadata.get(key), str) and metadata.get(key)
        for key in ("dates_start", "dates_end")
    ):
        raise ProtocolValidationError("evaluation panel dates are missing")
    if (
        metadata.get("positive_level_policy") != "reference_based"
        or metadata.get("iv_transform") != "bounded_logit"
        or metadata.get("iv_lower_bound") != 1e-4
        or metadata.get("iv_upper_bound") != 1.0
        or not isinstance(metadata.get("cleaning_report"), Mapping)
        or metadata["cleaning_report"].get("enabled") is not True
    ):
        raise ProtocolValidationError("evaluation cleaning/bounds authority drifted")
    alignment = manifest.get("iv_alignment")
    if not isinstance(alignment, Mapping):
        raise ProtocolValidationError("evaluation manifest has no IV alignment")
    for key in ("history_max_abs_error", "future_max_abs_error"):
        try:
            value = float(alignment[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolValidationError(
                "evaluation alignment is malformed"
            ) from error
        if not np.isfinite(value) or value > 1e-6:
            raise ProtocolValidationError("evaluation alignment exceeds 1e-6")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ProtocolValidationError("evaluation manifest has no provenance")
    recorded = provenance.get("data_artifacts")
    expected_data = {
        Path(path).name: digest for path, digest in FROZEN_ARTIFACT_ROWS.items()
    }
    if recorded != expected_data:
        raise ProtocolValidationError("evaluation data artifact authority drifted")
    evaluator = provenance.get("evaluator")
    command_config = _command_configuration(list(node["command"]))
    expected_evaluator = {
        "baseline": node["candidate"],
        "batch_size": int(command_config["batch_size"]),
        "max_windows": int(command_config["max_windows"]),
        "n_samples": int(command_config["n_samples"]),
        "seed": int(command_config["seed"]),
    }
    if evaluator != expected_evaluator:
        raise ProtocolValidationError("evaluation command provenance drifted")


def _valid_garch_contract_report(report: Mapping[str, Any] | None) -> bool:
    expected_authority = {
        "summary_sha256": FROZEN_SUMMARY_SHA256,
        "replay_result_sha256": FROZEN_REPLAY_RESULT_SHA256,
        "replay_manifest_sha256": FROZEN_REPLAY_MANIFEST_SHA256,
        "table_sha256": FROZEN_TABLE_SHA256,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "source_blob": EXPECTED_SOURCE_BLOB,
        "source_tag": SOURCE_TAG,
    }
    return bool(
        isinstance(report, Mapping)
        and report.get("schema_version") == REPORT_SCHEMA
        and report.get("candidate") == "garch_ccc"
        and report.get("overall_pass") is True
        and report.get("immutable_authority") is True
        and report.get("authority") == expected_authority
        and {
            key: value.get("sha256")
            for key, value in report.get("source_artifacts", {}).items()
            if isinstance(value, Mapping)
        }
        == {
            "summary": FROZEN_SUMMARY_SHA256,
            "replay_result": FROZEN_REPLAY_RESULT_SHA256,
            "replay_manifest": FROZEN_REPLAY_MANIFEST_SHA256,
            "table": FROZEN_TABLE_SHA256,
        }
        and isinstance(report.get("structural_checks"), list)
        and isinstance(report.get("exact_headline_checks"), list)
        and isinstance(report.get("display_rounded_checks"), list)
        and all(
            check.get("passed") is True
            for key in (
                "structural_checks",
                "exact_headline_checks",
                "display_rounded_checks",
            )
            for check in report.get(key, [])
        )
    )


def _validate_window_scores(path: Path, *, n_windows: int) -> None:
    if not path.is_file():
        raise ProtocolValidationError(f"completed run has no window scores: {path}")
    try:
        with np.load(path, allow_pickle=False) as deck:
            expected_shapes = {
                "window_indices": (n_windows,),
                "crps_contributions": (n_windows, 4, 39),
                "energy_contributions": (n_windows, 4),
                "variogram_h30_contributions": (n_windows,),
            }
            for name, shape in expected_shapes.items():
                values = np.asarray(deck[name])
                if values.shape != shape or not np.isfinite(values).all():
                    raise ProtocolValidationError(f"window score {name} is malformed")
            indices = np.asarray(deck["window_indices"], dtype=np.int64)
            if indices.size > 1 and not np.all(np.diff(indices) > 0):
                raise ProtocolValidationError("window score chronology drifted")
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ProtocolValidationError):
            raise
        raise ProtocolValidationError("window score artifact is unreadable") from error


def _validate_completed_evaluation(
    root: Path,
    protocol: Mapping[str, Any],
    node: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_protocol_authorities(protocol)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ProtocolValidationError(
            "completed evaluation run manifest schema drifted"
        )
    if manifest.get("protocol_contract_hash") != protocol.get("contract_hash"):
        raise ProtocolValidationError("completed evaluation protocol hash drifted")
    if manifest.get("node_id") != node.get("node_id"):
        raise ProtocolValidationError("completed evaluation node identity drifted")
    expected_command = list(node["command"])
    if manifest.get("command") != expected_command or manifest.get(
        "command_sha256"
    ) != _digest_text(json.dumps(expected_command, separators=(",", ":"))):
        raise ProtocolValidationError("completed evaluation command drifted")
    expected_configuration = dict(protocol["validation_protocol"])
    if manifest.get("configuration") != expected_configuration or manifest.get(
        "configuration_sha256"
    ) != _digest_text(_canonical_json(expected_configuration)):
        raise ProtocolValidationError("completed evaluation configuration drifted")
    expected_sources = {
        "source_manifest": dict(protocol["source_manifest"]),
        "artifact_manifest": dict(protocol["artifact_manifest"]),
        "source_authority": dict(protocol["source_authority"]),
        "artifact_authority": dict(protocol["artifact_authority"]),
    }
    if manifest.get("source_verification") != expected_sources:
        raise ProtocolValidationError("completed evaluation source authority drifted")
    result_path = root / str(manifest.get("result_path", ""))
    if not result_path.is_file() or hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest() != manifest.get("result_sha256"):
        raise ProtocolValidationError("completed result checksum mismatch")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result_schema(
            result,
            canonical_counts=protocol.get("benchmark_split") == "published_validation",
        )
    except (OSError, json.JSONDecodeError, ResultContractError) as error:
        raise ProtocolValidationError("completed result contract drifted") from error
    _validate_identity(
        candidate=str(node["candidate"]),
        train_seed=node.get("train_seed"),
        rollout_seed=int(node["rollout_seed"]),
        result=result,
        protocol=protocol,
    )
    if node.get("candidate") != "garch_ccc":
        candidate_dir = root / str(node["candidate"])
        evaluator_path = candidate_dir / "manifest.json"
        if not evaluator_path.is_file() or hashlib.sha256(
            evaluator_path.read_bytes()
        ).hexdigest() != manifest.get("evaluation_manifest_sha256"):
            raise ProtocolValidationError("evaluation manifest checksum drifted")
        try:
            evaluator_manifest = json.loads(evaluator_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "evaluation manifest is unreadable"
            ) from error
        if evaluator_manifest != manifest.get("evaluation_manifest"):
            raise ProtocolValidationError("evaluation manifest payload drifted")
        _validate_evaluation_manifest(evaluator_manifest, protocol=protocol, node=node)
        if evaluator_manifest["provenance"].get("checkpoint") != manifest.get(
            "input_checkpoint"
        ):
            raise ProtocolValidationError("evaluation checkpoint provenance drifted")
        candidate_result = candidate_dir / "results_current_panel.json"
        if not candidate_result.is_file() or hashlib.sha256(
            candidate_result.read_bytes()
        ).hexdigest() != manifest.get("candidate_result_sha256"):
            raise ProtocolValidationError("candidate result checksum drifted")
        try:
            if _result_json(
                json.loads(candidate_result.read_text(encoding="utf-8"))
            ) != _result_json(result):
                raise ProtocolValidationError("candidate result payload drifted")
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError("candidate result is unreadable") from error
        score_path = candidate_dir / "window_scores.npz"
        if not score_path.is_file() or hashlib.sha256(
            score_path.read_bytes()
        ).hexdigest() != manifest.get("window_scores_sha256"):
            raise ProtocolValidationError("window scores checksum drifted")
        _validate_window_scores(score_path, n_windows=int(result["n_windows"]))
    else:
        report = manifest.get("garch_contract_report")
        if not _valid_garch_contract_report(report) or _digest_text(
            _canonical_json(report)
        ) != manifest.get("garch_contract_sha256"):
            raise ProtocolValidationError("garch authority report drifted")
    checkpoint = manifest.get("input_checkpoint")
    if checkpoint is not None:
        path = Path(str(checkpoint.get("path", "")))
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != checkpoint.get("sha256"):
            raise ProtocolValidationError("evaluation input checkpoint drifted")
        dependencies = list(node.get("depends_on", []))
        dependency_nodes = [
            item
            for item in protocol.get("nodes", [])
            if item.get("node_id") in dependencies and item.get("action") == "train"
        ]
        if len(dependency_nodes) != 1:
            raise ProtocolValidationError("evaluation checkpoint dependency drifted")
        dependency = dependency_nodes[0]
        seal_path = path.with_suffix(path.suffix + ".node_manifest.json")
        try:
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "evaluation checkpoint seal is missing"
            ) from error
        if seal != _training_node_manifest(dependency, protocol, path):
            raise ProtocolValidationError("evaluation checkpoint seal drifted")
    return result


def record_completed_node(
    *,
    output_dir: str | Path,
    protocol: Mapping[str, Any],
    node_id: str,
    result: Mapping[str, Any],
    garch_contract_report: Mapping[str, Any] | None = None,
    evaluation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically seal a node; exact repeats resume and any drift is rejected."""
    contract = {
        key: value
        for key, value in protocol.items()
        if key not in {"contract_hash", "output_root", "nodes"}
    }
    contract_hash = _digest_text(_canonical_json(contract))
    if protocol.get("contract_hash") != contract_hash:
        raise ProtocolValidationError("protocol contract hash mismatch")
    matching = [
        node
        for node in protocol.get("nodes", [])
        if isinstance(node, Mapping) and node.get("node_id") == node_id
    ]
    if len(matching) != 1:
        raise ProtocolValidationError("node ID is absent or duplicated in protocol")
    node = matching[0]
    if node.get("action") not in {"evaluate", "authority"}:
        raise ProtocolValidationError(
            "only evaluation/authority nodes produce result manifests"
        )
    candidate = str(node["candidate"])
    train_seed = node.get("train_seed")
    rollout_seed = int(node["rollout_seed"])
    _validate_identity(
        candidate=candidate,
        train_seed=train_seed,
        rollout_seed=rollout_seed,
        result=result,
        protocol=protocol,
    )
    root = Path(output_dir).expanduser().resolve()
    destination = Path(str(node["destination"])).expanduser().resolve()
    if root != destination:
        raise ProtocolValidationError(
            "output directory is not the node's declared destination"
        )
    output_root = Path(str(protocol["output_root"])).expanduser().resolve()
    if output_root not in root.parents:
        raise ProtocolValidationError("node output escapes the isolated output root")
    for field in ("data_dir", "checkpoint_root", "artifact_root"):
        protected = Path(str(protocol[field])).expanduser().resolve()
        if root == protected or root in protected.parents or protected in root.parents:
            raise ProtocolValidationError(f"node output overlaps protected {field}")
    if node.get("protocol_contract_hash") != contract_hash:
        raise ProtocolValidationError("node protocol contract hash mismatch")
    if candidate == "garch_ccc":
        if not _valid_garch_contract_report(garch_contract_report):
            raise ProtocolValidationError(
                "garch_ccc requires the full immutable published-row contract"
            )
    elif evaluation_manifest is None:
        raise ProtocolValidationError("evaluation manifest is required")
    _verify_protocol_authorities(protocol)
    result_text = (
        json.dumps(
            dict(result),
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else value.tolist()
            ),
        )
        + "\n"
    )
    result_sha256 = _digest_text(result_text)
    authority_hash = (
        None
        if garch_contract_report is None
        else _digest_text(_canonical_json(garch_contract_report))
    )
    candidate_dir = root / candidate
    evaluator_path = candidate_dir / "manifest.json"
    candidate_result_path = candidate_dir / "results_current_panel.json"
    window_scores_path = candidate_dir / "window_scores.npz"
    evaluator_hash = None
    candidate_result_hash = None
    window_scores_hash = None
    if evaluation_manifest is not None:
        if not evaluator_path.is_file():
            raise ProtocolValidationError("evaluation manifest artifact is missing")
        try:
            stored_evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "evaluation manifest artifact is unreadable"
            ) from error
        if stored_evaluator != dict(evaluation_manifest):
            raise ProtocolValidationError(
                "evaluation manifest argument differs from artifact"
            )
        _validate_evaluation_manifest(evaluation_manifest, protocol=protocol, node=node)
        if not candidate_result_path.is_file():
            raise ProtocolValidationError("candidate result artifact is missing")
        try:
            stored_result_text = _result_json(
                json.loads(candidate_result_path.read_text(encoding="utf-8"))
            )
            supplied_result_text = _result_json(result)
            if stored_result_text != supplied_result_text:
                for offset, (left, right) in enumerate(
                    zip(stored_result_text, supplied_result_text)
                ):
                    if left != right:
                        break
                else:
                    offset = min(len(stored_result_text), len(supplied_result_text))
                raise ProtocolValidationError(
                    "candidate result differs from sealed result near "
                    f"{offset}: {stored_result_text[offset:offset+80]!r} != "
                    f"{supplied_result_text[offset:offset+80]!r}"
                )
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "candidate result artifact is unreadable"
            ) from error
        _validate_window_scores(window_scores_path, n_windows=int(result["n_windows"]))
        evaluator_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
        candidate_result_hash = hashlib.sha256(
            candidate_result_path.read_bytes()
        ).hexdigest()
        window_scores_hash = hashlib.sha256(window_scores_path.read_bytes()).hexdigest()
    checkpoint_path = next(
        (
            Path(node["command"][index + 1])
            for index, value in enumerate(node["command"][:-1])
            if value == "--checkpoint"
        ),
        None,
    )
    checkpoint_provenance = None
    if checkpoint_path is not None:
        if evaluation_manifest is not None and not checkpoint_path.is_file():
            raise ProtocolValidationError("evaluation checkpoint is missing")
        if checkpoint_path.is_file():
            checkpoint_provenance = {
                "path": str(checkpoint_path.resolve()),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
    manifest_core = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "protocol_contract_hash": contract_hash,
        "node_id": node_id,
        "candidate": candidate,
        "train_seed": train_seed,
        "rollout_seed": rollout_seed,
        "benchmark_split": protocol["benchmark_split"],
        "result_path": "results_current_panel.json",
        "result_sha256": result_sha256,
        "garch_contract_sha256": authority_hash,
        "garch_contract_report": (
            None if garch_contract_report is None else dict(garch_contract_report)
        ),
        "command_sha256": _digest_text(
            json.dumps(node["command"], separators=(",", ":"))
        ),
        "command": list(node["command"]),
        "configuration": dict(protocol["validation_protocol"]),
        "configuration_sha256": _digest_text(
            _canonical_json(protocol["validation_protocol"])
        ),
        "source_verification": {
            "source_manifest": dict(protocol["source_manifest"]),
            "artifact_manifest": dict(protocol["artifact_manifest"]),
            "source_authority": dict(protocol["source_authority"]),
            "artifact_authority": dict(protocol["artifact_authority"]),
        },
        "runtime_provenance": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "input_checkpoint": checkpoint_provenance,
        "evaluation_manifest": (
            None if evaluation_manifest is None else dict(evaluation_manifest)
        ),
        "evaluation_manifest_sha256": evaluator_hash,
        "candidate_result_sha256": candidate_result_hash,
        "window_scores_sha256": window_scores_hash,
    }
    result_path = root / "results_current_panel.json"
    manifest_path = root / "run_manifest.json"
    if result_path.exists() or manifest_path.exists():
        if not result_path.is_file() or not manifest_path.is_file():
            raise ProtocolValidationError("partial completed node cannot be resumed")
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                "completed node manifest is unreadable"
            ) from error
        existing_result_text = result_path.read_text(encoding="utf-8")
        if existing_manifest != manifest_core or existing_result_text != result_text:
            raise ProtocolValidationError(
                "refusing to replace a different completed node"
            )
        _validate_completed_evaluation(root, protocol, node, existing_manifest)
        return {**manifest_core, "resumed": True}
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(result_path, result_text)
    _write_atomic(
        manifest_path, json.dumps(manifest_core, indent=2, sort_keys=True) + "\n"
    )
    _validate_completed_evaluation(root, protocol, node, manifest_core)
    return {**manifest_core, "resumed": False}


def collect_completed_runs(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    protocol_path = root / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise ProtocolValidationError("output root has no protocol manifest")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = protocol.get("contract_hash")
    nodes = {
        node["node_id"]: node
        for node in protocol.get("nodes", [])
        if isinstance(node, Mapping) and node.get("action") in {"evaluate", "authority"}
    }
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        node = nodes.get(manifest.get("node_id"))
        if (
            node is None
            or Path(str(node["destination"])).resolve()
            != manifest_path.parent.resolve()
        ):
            raise ProtocolValidationError(
                f"copied or misplaced run manifest: {manifest_path}"
            )
        result = _validate_completed_evaluation(
            manifest_path.parent, protocol, node, manifest
        )
        window_scores = None
        if manifest["candidate"] != "garch_ccc":
            window_path = (
                manifest_path.parent / manifest["candidate"] / "window_scores.npz"
            )
            if not window_path.is_file():
                raise ProtocolValidationError(
                    f"completed run has no window scores: {window_path}"
                )
            with np.load(window_path, allow_pickle=False) as deck:
                window_scores = {
                    name: np.asarray(deck[name])
                    for name in (
                        "window_indices",
                        "crps_contributions",
                        "energy_contributions",
                        "variogram_h30_contributions",
                    )
                }
        runs.append(
            {
                "candidate": manifest["candidate"],
                "train_seed": manifest["train_seed"],
                "rollout_seed": manifest["rollout_seed"],
                "result": result,
                **({} if window_scores is None else {"window_scores": window_scores}),
            }
        )
    return runs


def _run_command(command: list[str]) -> None:
    portable = [sys.executable, *command[1:]] if command[0] == "python" else command
    completed = subprocess.run(portable, check=False, text=True)
    if completed.returncode != 0:
        raise ProtocolValidationError(
            f"node command failed with return code {completed.returncode}"
        )


def _command_configuration(command: list[str]) -> dict[str, Any]:
    configuration: dict[str, Any] = {}
    index = 3
    while index < len(command):
        item = command[index]
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                configuration[key] = command[index + 1]
                index += 2
                continue
            configuration[key] = True
        index += 1
    return configuration


def _training_node_manifest(
    node: Mapping[str, Any], protocol: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    _verify_protocol_authorities(protocol)
    configuration = _command_configuration(list(node["command"]))
    checkpoint_input = configuration.get("checkpoint")
    input_provenance = None
    if checkpoint_input is not None:
        path = Path(str(checkpoint_input))
        if not path.is_file():
            raise ProtocolValidationError("training input checkpoint is missing")
        input_provenance = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "protocol_contract_hash": protocol["contract_hash"],
        "node_id": node["node_id"],
        "command": list(node["command"]),
        "command_sha256": _digest_text(
            json.dumps(node["command"], separators=(",", ":"))
        ),
        "configuration": configuration,
        "configuration_sha256": _digest_text(_canonical_json(configuration)),
        "source_verification": {
            "source_manifest": dict(protocol["source_manifest"]),
            "artifact_manifest": dict(protocol["artifact_manifest"]),
            "source_authority": dict(protocol["source_authority"]),
            "artifact_authority": dict(protocol["artifact_authority"]),
        },
        "runtime_provenance": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "input_checkpoint": input_provenance,
        "produced_checkpoint": {
            "path": str(destination.resolve()),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        },
    }


def execute_protocol(protocol: Mapping[str, Any]) -> dict[str, int]:
    """Run a protocol DAG in dependency order and resume hash-bound outputs."""
    completed: set[str] = set()
    ran = resumed = 0
    contract_hash = str(protocol["contract_hash"])
    for node in protocol["nodes"]:
        node_id = str(node["node_id"])
        missing = [item for item in node["depends_on"] if item not in completed]
        if missing:
            raise ProtocolValidationError(
                f"node dependencies are incomplete: {missing}"
            )
        destination = Path(str(node["destination"]))
        if node["action"] in {"evaluate", "authority"}:
            manifest_path = destination / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                _validate_completed_evaluation(
                    destination.resolve(), protocol, node, manifest
                )
                resumed += 1
            else:
                _run_command(list(node["command"]))
                if not manifest_path.is_file():
                    raise ProtocolValidationError(
                        "evaluation did not seal its run manifest"
                    )
                ran += 1
        else:
            seal = destination.with_suffix(destination.suffix + ".node_manifest.json")
            if seal.is_file():
                payload = json.loads(seal.read_text(encoding="utf-8"))
                if not destination.is_file() or payload != _training_node_manifest(
                    node, protocol, destination
                ):
                    raise ProtocolValidationError("training node resume seal drifted")
                resumed += 1
            else:
                _run_command(list(node["command"]))
                if not destination.is_file():
                    raise ProtocolValidationError(
                        "training node produced no checkpoint"
                    )
                payload = _training_node_manifest(node, protocol, destination)
                _write_atomic(
                    seal, json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                ran += 1
        completed.add(node_id)
    return {"ran": ran, "resumed": resumed, "completed": len(completed)}


def aggregate_smoke_results(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in sorted({str(run["candidate"]) for run in runs}):
        selected = [run for run in runs if run["candidate"] == candidate]
        metrics: dict[str, float] = {}
        for name, path in METRIC_PATHS:
            values = []
            for run in selected:
                value: Any = run["result"]
                for key in path:
                    value = value[key]
                values.append(float(value))
            metrics[name] = float(np.mean(values))
        candidates[candidate] = {"n_runs": len(selected), "metric_means": metrics}
    return {
        "schema_version": "scenarioflow.multiseed-smoke-aggregate.v1",
        "benchmark_split": "published_validation_prefix_smoke",
        "candidates": candidates,
    }


def train_baseline(
    *,
    candidate: str,
    data_dir: str | Path,
    checkpoint_out: str | Path,
    train_seed: int,
    epochs: int = 10,
    batch_size: int = 64,
    n_diffusion_steps: int = 40,
    sample_steps: int = 8,
    weight_decay: float = 1e-5,
    clip_grad: float = 1.0,
) -> Path:
    """Fit a package neural baseline from the committed pre-test window block."""
    if candidate not in {"deepvar", "timegrad", "path_diffusion", "deepvar_sni"}:
        raise ProtocolValidationError(
            f"{candidate} uses scenarioflow.training with an explicit batch/config"
        )
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    _panel, _columns, train, validation = prepare_validation_blocks(data_dir=data_dir)
    history = train.history_increment.astype(np.float32)
    future = train.future_increment.astype(np.float32)
    val_history = validation.history_increment[:256].astype(np.float32)
    val_future = validation.future_increment[:256].astype(np.float32)
    normalization: dict[str, Any] | None = None
    model_type = "deepvar" if candidate == "deepvar_sni" else candidate
    if candidate == "deepvar_sni":
        history, future, _center, _scale = normalize_increment_windows(
            history, future, half_life=None, scale_floor=1e-4, center_mode="zero"
        )
        val_history, val_future, _center, _scale = normalize_increment_windows(
            val_history,
            val_future,
            half_life=None,
            scale_floor=1e-4,
            center_mode="zero",
        )
        normalization = {
            "scale_half_life": None,
            "scale_floor": 1e-4,
            "center_mode": "zero",
        }
        mean = np.zeros(history.shape[-1], dtype=np.float32)
        std = np.ones(history.shape[-1], dtype=np.float32)
    else:
        daily = np.concatenate([history, future], axis=1).reshape(-1, history.shape[-1])
        mean = daily.mean(axis=0).astype(np.float32)
        std = (daily.std(axis=0) + 1e-6).astype(np.float32)
        history = (history - mean[None, None]) / std[None, None]
        future = (future - mean[None, None]) / std[None, None]
        val_history = (val_history - mean[None, None]) / std[None, None]
        val_future = (val_future - mean[None, None]) / std[None, None]
    config = {
        "model_type": model_type,
        "input_dim": int(history.shape[-1]),
        "future_len": int(future.shape[1]),
        "hidden_dim": 128,
        "num_layers": 2,
        "rank": 5,
        "n_diffusion_steps": int(n_diffusion_steps),
        "cond_dim": 128,
        "sample_steps": int(sample_steps),
    }
    model = build_model(
        model_type,
        input_dim=history.shape[-1],
        future_len=future.shape[1],
        hidden_dim=128,
        num_layers=2,
        rank=5,
        n_diffusion_steps=int(n_diffusion_steps),
        cond_dim=128,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(train_seed)
    best_val = float("inf")
    best_epoch = -1
    destination = Path(checkpoint_out)

    def save_best() -> None:
        if candidate == "deepvar_sni":
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalization": normalization,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                destination,
            )
        else:
            save_checkpoint(
                destination,
                model,
                model_type=model_type,
                mean=mean,
                std=std,
                config=config,
                best_val_loss=best_val,
            )

    for epoch in range(epochs):
        model.train()
        indices = rng.permutation(history.shape[0])
        for start in range(0, history.shape[0], batch_size):
            positions = indices[start : start + batch_size]
            h = torch.from_numpy(history[positions])
            f = torch.from_numpy(future[positions])
            optimizer.zero_grad(set_to_none=True)
            loss = model(h, f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_history.shape[0], batch_size):
                h = torch.from_numpy(val_history[start : start + batch_size])
                f = torch.from_numpy(val_future[start : start + batch_size])
                loss = model(h, f)
                size = int(h.shape[0])
                total += float(loss) * size
                count += size
        validation_loss = total / count
        if np.isfinite(validation_loss) and validation_loss < best_val:
            best_val = validation_loss
            best_epoch = epoch
            save_best()
    if best_epoch < 0:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser("plan", help="create or resume the frozen local DAG")
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--source-manifest", required=True)
    plan.add_argument("--artifact-manifest", required=True)
    plan.add_argument("--data-dir", required=True)
    plan.add_argument("--checkpoint-root", required=True)
    plan.add_argument("--artifact-root", required=True)
    reproduce = subcommands.add_parser(
        "reproduce", help="plan, run/resume, and aggregate the DAG"
    )
    for option in (
        "output-root",
        "source-manifest",
        "artifact-manifest",
        "data-dir",
        "checkpoint-root",
        "artifact-root",
        "output",
    ):
        reproduce.add_argument(f"--{option}", required=True)
    reproduce.add_argument("--smoke", action="store_true")
    reproduce.add_argument("--candidate", action="append")
    reproduce.add_argument("--max-windows", type=int, default=441)
    reproduce.add_argument("--n-samples", type=int, default=64)
    reproduce.add_argument("--rollout-seed", action="append", type=int)
    reproduce.add_argument("--train-seed", action="append", type=int)
    node = subcommands.add_parser(
        "run-node", help="seal an explicitly produced package result"
    )
    node.add_argument("--protocol", required=True)
    node.add_argument("--node-id", required=True)
    node.add_argument("--result-input", required=True)
    node.add_argument("--garch-contract-report")
    node.add_argument("--evaluation-manifest")
    node.add_argument("--output-dir", required=True)
    training = subcommands.add_parser(
        "train-baseline", help="fit a package neural baseline"
    )
    training.add_argument("--candidate", required=True)
    training.add_argument("--data-dir", required=True)
    training.add_argument("--checkpoint-out", required=True)
    training.add_argument("--train-seed", required=True, type=int)
    training.add_argument("--epochs", type=int, required=True)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--n-diffusion-steps", type=int, default=40)
    training.add_argument("--sample-steps", type=int, default=8)
    training.add_argument("--weight-decay", type=float, default=1e-5)
    training.add_argument("--clip-grad", type=float, default=1.0)
    aggregate = subcommands.add_parser(
        "aggregate", help="aggregate a complete resumed DAG"
    )
    aggregate.add_argument("--output-root", required=True)
    aggregate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
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
        return
    if args.command == "reproduce":
        rollout_seeds = (
            tuple(args.rollout_seed) if args.rollout_seed is not None else ROLLOUT_SEEDS
        )
        train_seeds = (
            tuple(args.train_seed)
            if args.train_seed is not None
            else ((TRAIN_SEEDS[0],) if args.smoke else TRAIN_SEEDS)
        )
        candidates = tuple(args.candidate) if args.candidate else CANDIDATE_IDS
        payload = build_protocol(
            output_root=args.output_root,
            source_manifest=args.source_manifest,
            artifact_manifest=args.artifact_manifest,
            data_dir=args.data_dir,
            checkpoint_root=args.checkpoint_root,
            artifact_root=args.artifact_root,
            rollout_seeds=rollout_seeds,
            train_seeds=train_seeds,
            candidates=candidates,
            smoke=bool(args.smoke),
            max_windows=int(args.max_windows),
            n_samples=int(args.n_samples),
        )
        execution = execute_protocol(payload)
        runs = collect_completed_runs(args.output_root)
        if args.smoke:
            summary = aggregate_smoke_results(runs)
        else:
            summary = aggregate_results(
                runs,
                paired_comparisons=(
                    ("deepvar_vs_deepvar_support", "deepvar", "deepvar_sni"),
                    ("scenarioflow_vs_graph", "scenarioflow", "scenarioflow_graph"),
                ),
            )
        output = Path(args.output)
        _write_atomic(output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({**execution, "aggregate": str(output)}))
        return
    if args.command == "run-node":
        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        result = json.loads(Path(args.result_input).read_text(encoding="utf-8"))
        authority = (
            None
            if args.garch_contract_report is None
            else json.loads(
                Path(args.garch_contract_report).read_text(encoding="utf-8")
            )
        )
        evaluation_manifest = (
            None
            if args.evaluation_manifest is None
            else json.loads(Path(args.evaluation_manifest).read_text(encoding="utf-8"))
        )
        manifest = record_completed_node(
            output_dir=args.output_dir,
            protocol=protocol,
            node_id=args.node_id,
            result=result,
            garch_contract_report=authority,
            evaluation_manifest=evaluation_manifest,
        )
        print(json.dumps(manifest, sort_keys=True))
        return
    if args.command == "train-baseline":
        print(
            train_baseline(
                candidate=args.candidate,
                data_dir=args.data_dir,
                checkpoint_out=args.checkpoint_out,
                train_seed=args.train_seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                n_diffusion_steps=args.n_diffusion_steps,
                sample_steps=args.sample_steps,
                weight_decay=args.weight_decay,
                clip_grad=args.clip_grad,
            )
        )
        return
    summary = aggregate_results(
        collect_completed_runs(args.output_root),
        paired_comparisons=(
            ("deepvar_vs_deepvar_support", "deepvar", "deepvar_sni"),
            ("scenarioflow_vs_graph", "scenarioflow", "scenarioflow_graph"),
        ),
    )
    output = Path(args.output)
    _write_atomic(output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
