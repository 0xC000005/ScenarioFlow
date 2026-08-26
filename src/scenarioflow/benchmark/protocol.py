"""Translate the frozen 141-node protocol to the compact public entrypoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "results/frozen/multiseed/command-dag.json"
FROZEN_MODEL_ROOT = "frozen-source://models/backfill/fmts_multiseed_20260805"
FROZEN_RESULT_ROOT = "frozen-source://results/fmts_multiseed_20260805"
FROZEN_PROTOCOL_HASH = (
    "c6d570106975a79750d9fc47be96b07b71ea09930812c72154116a54e3a37d72"
)
FROZEN_PROJECTED_PAYLOAD_HASH = (
    "ecf90958d519226e280c3f35cfc10bfbed067e068fb5f381b2edb4c343e83844"
)
_HISTORICAL_SOURCE_ROOT = "/" + "/".join(
    ("home", "max", "Documents", "vol-surface-vae-pub")
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

_ENTRYPOINTS = {
    "experiments/backfill/baselines/evaluate_current_panel_baselines.py": (
        "scenarioflow.benchmark.evaluate_baselines"
    ),
    "experiments/backfill/baselines/train_current_panel_deep_baselines.py": (
        "scenarioflow.benchmark.train_baselines"
    ),
    "experiments/backfill/baselines/train_g2_deepvar_sni_interface.py": (
        "scenarioflow.benchmark.train_deepvar_sni"
    ),
    "experiments/backfill/baselines/evaluate_g2_deepvar_sni_interface.py": (
        "scenarioflow.benchmark.evaluate_deepvar_sni"
    ),
    "experiments/backfill/block_ar/train_662a_state_aware_normalized_innovation_flow.py": (
        "scenarioflow.reproduction.train_next_step"
    ),
    "experiments/backfill/block_ar/train_666a_normalized_innovation_rollout_energy_finetune.py": (
        "scenarioflow.reproduction.finetune_rollouts"
    ),
    "experiments/backfill/baselines/evaluate_sni_current_panel_metrics.py": (
        "scenarioflow.benchmark.evaluate_scenarioflow"
    ),
}

_OBSOLETE_OPTIONS = {
    "scenarioflow.reproduction.train_next_step": {
        "--sample_windows",
        "--sample_count",
        "--sample_steps",
        "--chunk_size",
    },
    "scenarioflow.reproduction.finetune_rollouts": {
        "--sample_windows",
        "--sample_count",
        "--sample_steps",
        "--chunk_size",
        "--prefix_feature_mode",
    },
}


class ProtocolError(ValueError):
    """Raised when the frozen command graph no longer matches its public contract."""


@dataclass(frozen=True)
class SmokeOverrides:
    """Explicit resource bounds for local verification; omitted values stay frozen."""

    max_windows: int | None = None
    samples: int | None = None
    epochs: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    batch_size: int | None = None
    sample_steps: int | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def executable_dag_sha256(commands: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(list(commands))


def _historical_hash_value(value: Any) -> Any:
    """Reverse only the public source-root projection used before node sealing."""
    if isinstance(value, str):
        if value == "frozen-source://":
            return _HISTORICAL_SOURCE_ROOT
        if value.startswith("frozen-source://"):
            return (
                _HISTORICAL_SOURCE_ROOT + "/" + value.removeprefix("frozen-source://")
            )
        return value
    if isinstance(value, list):
        return [_historical_hash_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _historical_hash_value(key): _historical_hash_value(item)
            for key, item in value.items()
        }
    return value


def _validate_frozen_node_hashes(commands: Sequence[Mapping[str, Any]]) -> None:
    for node in commands:
        node_id = node.get("id")
        command = node.get("command")
        configuration = node.get("configuration")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ProtocolError(f"node {node_id} has no command list")
        if not isinstance(configuration, Mapping):
            raise ProtocolError(f"node {node_id} has no configuration object")
        if node.get("command_sha256") != canonical_json_sha256(
            _historical_hash_value(command)
        ):
            raise ProtocolError(f"frozen node command hash drifted: {node_id}")
        if node.get("configuration_sha256") != canonical_json_sha256(
            _historical_hash_value(dict(configuration))
        ):
            raise ProtocolError(f"frozen node configuration hash drifted: {node_id}")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Load and validate the immutable released protocol description."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    commands = document.get("commands")
    protocol = document.get("protocol")
    if not isinstance(commands, list) or not isinstance(protocol, dict):
        raise ProtocolError("protocol must contain command and protocol objects")
    if len(commands) != 141 or document.get("command_count") != 141:
        raise ProtocolError("the released final protocol must contain 141 commands")
    if tuple(protocol.get("train_seeds", ())) != TRAIN_SEEDS:
        raise ProtocolError("training seed grid drifted")
    if tuple(protocol.get("rollout_seeds", ())) != ROLLOUT_SEEDS:
        raise ProtocolError("sampling seed grid drifted")
    if tuple(protocol.get("candidate_ids", ())) != CANDIDATE_IDS:
        raise ProtocolError("final candidate set drifted")
    # The released JSON is a path-portable projection of the original sealed DAG,
    # so its historical digest authenticates the pre-projection command graph.
    if protocol.get("executable_dag_sha256") != FROZEN_PROTOCOL_HASH:
        raise ProtocolError("frozen executable DAG authority hash drifted")
    _validate_frozen_node_hashes(commands)
    if canonical_json_sha256(document) != FROZEN_PROJECTED_PAYLOAD_HASH:
        raise ProtocolError("frozen projected payload digest drifted")
    return document


def _replace_frozen_path(value: str, model_root: Path, result_root: Path) -> str:
    if value == FROZEN_MODEL_ROOT or value.startswith(FROZEN_MODEL_ROOT + "/"):
        suffix = value[len(FROZEN_MODEL_ROOT) :].lstrip("/")
        return str(model_root / suffix)
    if value == FROZEN_RESULT_ROOT or value.startswith(FROZEN_RESULT_ROOT + "/"):
        suffix = value[len(FROZEN_RESULT_ROOT) :].lstrip("/")
        return str(result_root / suffix)
    return value


def _replace_option(command: list[str], option: str, value: object) -> None:
    if option in command:
        position = command.index(option)
        if position + 1 >= len(command):
            raise ProtocolError(f"missing value for {option}")
        command[position + 1] = str(value)
    else:
        command.extend([option, str(value)])


def _remove_options(command: list[str], options: set[str]) -> list[str]:
    clean: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in options:
            index += 2
            continue
        clean.append(item)
        index += 1
    return clean


def _apply_smoke(command: list[str], module: str, smoke: SmokeOverrides) -> None:
    if smoke.epochs is not None and "train" in module:
        _replace_option(command, "--epochs", smoke.epochs)
    if smoke.batch_size is not None:
        _replace_option(command, "--batch_size", smoke.batch_size)
    if smoke.max_train_windows is not None and "train" in module:
        _replace_option(command, "--max_train_windows", smoke.max_train_windows)
    if smoke.max_val_windows is not None and module in {
        "scenarioflow.benchmark.train_baselines",
        "scenarioflow.benchmark.train_deepvar_sni",
    }:
        _replace_option(command, "--max_val_windows", smoke.max_val_windows)
    if smoke.max_val_windows is not None and module == (
        "scenarioflow.reproduction.finetune_rollouts"
    ):
        _replace_option(command, "--max_val_batches", smoke.max_val_windows)
    if smoke.max_windows is not None and "evaluate" in module:
        _replace_option(command, "--max_windows", smoke.max_windows)
    if smoke.samples is not None and "evaluate" in module:
        option = (
            "--samples"
            if module == "scenarioflow.benchmark.evaluate_scenarioflow"
            else "--n_samples"
        )
        _replace_option(command, option, smoke.samples)
    if smoke.sample_steps is not None:
        if module == "scenarioflow.benchmark.evaluate_baselines":
            _replace_option(command, "--deep_sample_steps", smoke.sample_steps)
        elif module == "scenarioflow.reproduction.finetune_rollouts":
            _replace_option(command, "--rollout_flow_steps", smoke.sample_steps)


def _materialize_node(
    source: Mapping[str, Any],
    *,
    model_root: Path,
    result_root: Path,
    device: str,
    smoke: SmokeOverrides | None,
) -> dict[str, Any]:
    node = copy.deepcopy(dict(source))
    frozen_command = node.get("command")
    if not isinstance(frozen_command, list) or len(frozen_command) < 2:
        raise ProtocolError(f"node {node.get('id')} has no executable command")
    legacy_entrypoint = str(frozen_command[1])
    try:
        module = _ENTRYPOINTS[legacy_entrypoint]
    except KeyError as error:
        raise ProtocolError(
            f"unmapped final entrypoint: {legacy_entrypoint}"
        ) from error
    tail = [
        _replace_frozen_path(str(value), model_root, result_root)
        for value in frozen_command[2:]
    ]
    command = [sys.executable, "-m", module, *tail]
    command = _remove_options(command, _OBSOLETE_OPTIONS.get(module, set()))
    if module == "scenarioflow.benchmark.evaluate_scenarioflow":
        _replace_option(command, "--candidate", node["candidate"])
    _replace_option(command, "--device", device)
    if smoke is not None:
        _apply_smoke(command, module, smoke)
    node["frozen_command_sha256"] = node.get("command_sha256")
    node["command"] = command
    node["command_text"] = " ".join(command)
    node["command_sha256"] = canonical_json_sha256(command)
    for field in ("destination", "input_checkpoint", "produced_checkpoint"):
        if isinstance(node.get(field), str):
            node[field] = _replace_frozen_path(node[field], model_root, result_root)
    resume = node.get("resume")
    if isinstance(resume, dict):
        resume["completion_artifacts"] = [
            _replace_frozen_path(str(path), model_root, result_root)
            for path in resume.get("completion_artifacts", [])
        ]
    sidecar = node.get("sidecar_manifest")
    if isinstance(sidecar, dict) and isinstance(sidecar.get("path"), str):
        sidecar["path"] = _replace_frozen_path(sidecar["path"], model_root, result_root)
    configuration = node.get("configuration")
    if isinstance(configuration, dict):
        for field in ("destination",):
            if isinstance(configuration.get(field), str):
                configuration[field] = _replace_frozen_path(
                    configuration[field], model_root, result_root
                )
        configuration["public_module"] = module
        configuration["device"] = device
        node["configuration_sha256"] = canonical_json_sha256(configuration)
    return node


def materialize_protocol(
    frozen: Mapping[str, Any],
    *,
    model_root: str | Path,
    result_root: str | Path,
    device: str,
    smoke: SmokeOverrides | None = None,
) -> dict[str, Any]:
    """Return a runnable copy of the frozen graph with only path/module rewrites."""
    models = Path(model_root).expanduser().resolve()
    results = Path(result_root).expanduser().resolve()
    commands = [
        _materialize_node(
            source,
            model_root=models,
            result_root=results,
            device=device,
            smoke=smoke,
        )
        for source in frozen["commands"]
    ]
    if len(commands) != 141:
        raise ProtocolError("materialized protocol must contain 141 nodes")
    document = copy.deepcopy(dict(frozen))
    document["commands"] = commands
    document["command_count"] = len(commands)
    protocol = document["protocol"]
    protocol["frozen_executable_dag_sha256"] = protocol["executable_dag_sha256"]
    protocol["model_root"] = str(models)
    protocol["result_root"] = str(results)
    protocol["executable_dag_sha256"] = executable_dag_sha256(commands)
    document["mode"] = "public-module-execution"
    return document


def select_nodes(
    payload: Mapping[str, Any],
    node_id: str,
    *,
    include_dependencies: bool = True,
) -> list[dict[str, Any]]:
    """Return one node and, by default, its transitive dependency closure."""
    commands = payload.get("commands")
    if not isinstance(commands, list):
        raise ProtocolError("protocol has no commands")
    by_id = {str(node["id"]): node for node in commands}
    if len(by_id) != len(commands):
        raise ProtocolError("duplicate node ID")
    if node_id not in by_id:
        raise KeyError(f"unknown node ID: {node_id}")
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(current: str) -> None:
        if current in complete:
            return
        if current in visiting:
            raise ProtocolError("command graph contains a dependency cycle")
        visiting.add(current)
        node = by_id.get(current)
        if node is None:
            raise ProtocolError(f"unknown dependency: {current}")
        if include_dependencies:
            for dependency in node.get("depends_on", []):
                visit(str(dependency))
        visiting.remove(current)
        complete.add(current)
        ordered.append(node)

    visit(node_id)
    return ordered


def topological_nodes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every command in dependency-safe order."""
    commands = payload.get("commands")
    if not isinstance(commands, list):
        raise ProtocolError("protocol has no commands")
    by_id = {str(node["id"]): node for node in commands}
    ordered: list[dict[str, Any]] = []
    complete: set[str] = set()
    while len(ordered) < len(commands):
        ready = [
            node
            for node in commands
            if str(node["id"]) not in complete
            and set(map(str, node.get("depends_on", []))).issubset(complete)
        ]
        if not ready:
            raise ProtocolError("command graph contains a cycle or unknown dependency")
        for node in ready:
            complete.add(str(node["id"]))
            ordered.append(node)
    if set(by_id) != complete:
        raise ProtocolError("command graph contains duplicate nodes")
    return ordered
