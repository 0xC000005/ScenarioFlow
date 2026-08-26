"""Public contract for the compact final numerical benchmark layer."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scenarioflow.benchmark.baselines.classical import (
    JointBootstrap,
    JointPCAVAR,
    JointRandomWalk,
)
from scenarioflow.benchmark.baselines.deep import build_model
from scenarioflow.benchmark.executor import (
    BenchmarkExecutionError,
    _record_completion,
    completion_is_valid,
    ensure_protocol_manifest,
    execute_nodes,
    validate_output_roots,
)
from scenarioflow.benchmark.guard import UnsafeOutputRoot, tracked_numbered_experiments
from scenarioflow.benchmark.protocol import (
    CANDIDATE_IDS,
    FROZEN_PROJECTED_PAYLOAD_HASH,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    ProtocolError,
    SmokeOverrides,
    canonical_json_sha256,
    load_protocol,
    materialize_protocol,
    select_nodes,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DAG = ROOT / "results/frozen/multiseed/command-dag.json"


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_protocol(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _evaluation_completion(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    destination = tmp_path / "run"
    result = destination / "random_walk/results_current_panel.json"
    scores = destination / "random_walk/window_scores.npz"
    command = [
        sys.executable,
        "-c",
        "pass",
        "--max_windows",
        "1",
        "--n_samples",
        "2",
    ]
    configuration = {
        "contract": {
            "mode": "current_realvix_39_state_panel",
            "n_evaluation_windows": 441,
            "n_market_variables": 39,
            "n_scenarios_per_window": 64,
            "history_length": 30,
            "future_length": 30,
        }
    }
    node = {
        "id": "eval:random_walk:rollout_2001",
        "candidate": "random_walk",
        "display_name": "Random walk",
        "action": "evaluate",
        "training_seed": None,
        "rollout_seed": 2001,
        "depends_on": [],
        "destination": str(destination),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "resume": {"completion_artifacts": [str(result), str(scores)]},
        "sidecar_manifest": {"path": str(destination / "run_manifest.json")},
    }
    payload = {
        "protocol": {
            "contract": configuration["contract"],
            "executable_dag_sha256": "a" * 64,
        }
    }
    return payload, node, result, scores


def _write_valid_evaluation(result: Path, scores: Path) -> None:
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "baseline_name": "random_walk",
                "mode": "current_realvix_39_state_panel",
                "n_windows": 1,
                "n_samples": 2,
                "state_specs": [{"name": f"state:{index}"} for index in range(39)],
                "current_panel_iv_metrics": {},
                "joint_panel_quality": {},
                "panel_b_crps": {},
                "panel_b_energy_score": {},
                "panel_b_variogram_score": {},
                "panel_b_correlation": {},
                "panel_b_factor_ks": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        scores,
        schema_version=np.asarray("fmts.window_scores.v1"),
        window_indices=np.asarray([4511]),
        horizons=np.asarray([1, 7, 14, 30]),
        variable_names=np.asarray([f"state:{index}" for index in range(39)]),
        crps_contributions=np.zeros((1, 4, 39)),
        energy_contributions=np.zeros((1, 4)),
        variogram_h30_contributions=np.zeros(1),
        coverage_levels=np.asarray([0.5, 0.8, 0.9, 0.95]),
        coverage_by_window=np.zeros((1, 4)),
        iv_history_activity=np.zeros(1),
        iv_deck_width=np.zeros(1),
        full_panel_history_activity=np.zeros(1),
        full_panel_deck_width=np.zeros(1),
    )


def test_frozen_protocol_materializes_all_clean_final_commands(tmp_path: Path) -> None:
    """Changing the final seed/candidate grid or leaking a legacy path is a bug."""
    frozen = load_protocol(FROZEN_DAG)
    materialized = materialize_protocol(
        frozen,
        model_root=tmp_path / "models",
        result_root=tmp_path / "results",
        device="cpu",
    )

    assert frozen["protocol"]["executable_dag_sha256"] == (
        "c6d570106975a79750d9fc47be96b07b71ea09930812c72154116a54e3a37d72"
    )
    assert FROZEN_PROJECTED_PAYLOAD_HASH == (
        "ecf90958d519226e280c3f35cfc10bfbed067e068fb5f381b2edb4c343e83844"
    )
    assert TRAIN_SEEDS == (1001, 1002, 1003)
    assert ROLLOUT_SEEDS == (2001, 2002, 2003, 2004, 2005)
    assert tuple(frozen["protocol"]["candidate_ids"]) == CANDIDATE_IDS
    assert len(materialized["commands"]) == 141
    assert sum(node["action"] == "train" for node in materialized["commands"]) == 21
    assert sum(node["action"] == "evaluate" for node in materialized["commands"]) == 120

    for node in materialized["commands"]:
        command = node["command"]
        assert command[:2] == [sys.executable, "-m"]
        assert "experiments/" not in " ".join(command)
        assert "frozen-source://" not in " ".join(command)
        assert _option(command, "--device") == "cpu"
        assert not any(
            "vol-surface-vae-pub" in part or "legacy-worktree://" in part
            for part in command
        )
        module = importlib.import_module(command[2])
        module.build_parser().parse_args(command[3:])

    by_id = {node["id"]: node for node in materialized["commands"]}
    assert by_id["train:scenarioflow:stage1:1001"]["command"][2] == (
        "scenarioflow.reproduction.train_next_step"
    )
    assert by_id["train:scenarioflow:stage2:1001"]["command"][2] == (
        "scenarioflow.reproduction.finetune_rollouts"
    )
    assert by_id["eval:scenarioflow:train_1001:rollout_2001"]["command"][2] == (
        "scenarioflow.benchmark.evaluate_scenarioflow"
    )
    assert by_id["eval:random_walk:rollout_2001"]["command"][2] == (
        "scenarioflow.benchmark.evaluate_baselines"
    )


def test_protocol_rejects_projected_payload_and_per_node_hash_drift(
    tmp_path: Path,
) -> None:
    """A changed projected command, config, or non-node field must fail closed."""
    original = json.loads(FROZEN_DAG.read_text(encoding="utf-8"))

    command_drift = json.loads(json.dumps(original))
    command_drift["commands"][0]["command"][-1] = "--not-the-frozen-command"
    path = tmp_path / "command-drift.json"
    _write_protocol(path, command_drift)
    with pytest.raises(ProtocolError, match="command hash drifted"):
        load_protocol(path)

    configuration_drift = json.loads(json.dumps(original))
    configuration_drift["commands"][0]["configuration"]["rollout_seed"] = 9999
    path = tmp_path / "configuration-drift.json"
    _write_protocol(path, configuration_drift)
    with pytest.raises(ProtocolError, match="configuration hash drifted"):
        load_protocol(path)

    payload_drift = json.loads(json.dumps(original))
    payload_drift["commands"][0]["display_name"] = "Changed label"
    path = tmp_path / "payload-drift.json"
    _write_protocol(path, payload_drift)
    with pytest.raises(ProtocolError, match="projected payload digest drifted"):
        load_protocol(path)


def test_selected_node_includes_dependencies_and_smoke_overrides(
    tmp_path: Path,
) -> None:
    """A selected evaluation must not run before its exact training chain."""
    payload = materialize_protocol(
        load_protocol(FROZEN_DAG),
        model_root=tmp_path / "models",
        result_root=tmp_path / "results",
        device="cpu",
        smoke=SmokeOverrides(
            max_windows=1,
            samples=2,
            epochs=1,
            max_train_windows=2,
            max_val_windows=1,
            batch_size=1,
            sample_steps=2,
        ),
    )
    selected = select_nodes(
        payload,
        "eval:scenarioflow_graph:train_1001:rollout_2001",
        include_dependencies=True,
    )
    assert [node["id"] for node in selected] == [
        "train:scenarioflow:stage1:1001",
        "train:scenarioflow:stage2:1001",
        "train:scenarioflow_graph:graph:1001",
        "eval:scenarioflow_graph:train_1001:rollout_2001",
    ]
    assert _option(selected[-1]["command"], "--max_windows") == "1"
    assert _option(selected[-1]["command"], "--samples") == "2"
    assert _option(selected[-1]["command"], "--n_steps") == "30"
    assert _option(selected[0]["command"], "--epochs") == "1"
    assert _option(selected[0]["command"], "--max_train_windows") == "2"


def test_resume_requires_schema_identity_and_current_artifact_hashes(
    tmp_path: Path,
) -> None:
    """Weak artifacts or stale node identity must never count as completed."""
    payload, node, result, scores = _evaluation_completion(tmp_path)
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    np.savez_compressed(scores, schema_version=np.asarray("fmts.window_scores.v1"))
    _record_completion(node, payload)
    assert not completion_is_valid(node, payload)

    _write_valid_evaluation(result, scores)
    _record_completion(node, payload)
    assert completion_is_valid(node, payload)

    status = Path(node["destination"]) / "status.json"
    status_payload = json.loads(status.read_text(encoding="utf-8"))
    status_payload["schema_version"] = "fmts.node_status.obsolete"
    status.write_text(json.dumps(status_payload) + "\n", encoding="utf-8")
    assert not completion_is_valid(node, payload)

    _record_completion(node, payload)
    changed_payload = json.loads(json.dumps(payload))
    changed_payload["protocol"]["contract"]["mode"] = "wrong-panel"
    assert not completion_is_valid(node, changed_payload)

    changed = json.loads(result.read_text(encoding="utf-8"))
    changed["baseline_name"] = "bootstrap"
    result.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    assert not completion_is_valid(node, payload)

    _write_valid_evaluation(result, scores)
    _record_completion(node, payload)
    status_payload = json.loads(status.read_text(encoding="utf-8"))
    status_payload["command_sha256"] = "b" * 64
    status.write_text(json.dumps(status_payload) + "\n", encoding="utf-8")
    assert not completion_is_valid(node, payload)


@pytest.mark.parametrize(
    "field",
    (
        "coverage_levels",
        "iv_history_activity",
        "iv_deck_width",
        "full_panel_history_activity",
        "full_panel_deck_width",
    ),
)
def test_resume_rejects_incomplete_required_npz_shapes(
    tmp_path: Path, field: str
) -> None:
    """Every required window-score field must carry its exact released shape."""
    payload, node, result, scores = _evaluation_completion(tmp_path)
    _write_valid_evaluation(result, scores)
    with np.load(scores, allow_pickle=False) as loaded:
        artifact = {name: loaded[name] for name in loaded.files}
    artifact[field] = np.zeros(2)
    np.savez_compressed(scores, **artifact)
    _record_completion(node, payload)
    assert not completion_is_valid(node, payload)


def test_resume_rejects_weak_checkpoint_schema(tmp_path: Path) -> None:
    """An empty state dict is not a completed training node."""
    checkpoint = tmp_path / "train/best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    configuration = {
        "implementation_name": "scenarioflow",
        "stage": "stage1",
        "contract": {
            "mode": "current_realvix_39_state_panel",
            "n_market_variables": 39,
            "history_length": 30,
            "future_length": 30,
        },
    }
    node = {
        "id": "train:scenarioflow:stage1:1001",
        "candidate": "scenarioflow",
        "display_name": "ScenarioFlow",
        "action": "train",
        "training_seed": 1001,
        "rollout_seed": None,
        "destination": str(checkpoint.parent),
        "command": [sys.executable, "-c", "pass"],
        "command_sha256": "c" * 64,
        "configuration": configuration,
        "configuration_sha256": "d" * 64,
        "resume": {"completion_artifacts": [str(checkpoint)]},
        "sidecar_manifest": {"path": str(checkpoint.parent / "run_manifest.json")},
    }
    payload = {
        "protocol": {
            "contract": configuration["contract"],
            "executable_dag_sha256": "e" * 64,
        }
    }
    torch.save({"model_state_dict": {}}, checkpoint)
    _record_completion(node, payload)
    assert not completion_is_valid(node, payload)

    torch.save(
        {
            "model_type": "deepvar",
            "model_state_dict": {"weight": torch.ones(1)},
            "config": {"input_dim": 39, "history_len": 30, "future_len": 30},
        },
        checkpoint,
    )
    _record_completion(node, payload)
    assert not completion_is_valid(node, payload)

    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "config": {
                "n_cells": 39,
                "history_len": 30,
                "future_len": 30,
                "conditioning_mode": "prefix",
            },
            "state_scope": "joint38",
            "model_coordinate": "state_aware_normalized_innovation",
            "normalization": {
                "coordinate": "encoded_increment",
                "innovation_coordinate": "normalized",
            },
            "state_specs": [{"name": f"state:{index}"} for index in range(39)],
            "training_objective": {"name": "next_step_flow_matching"},
        },
        checkpoint,
    )
    _record_completion(node, payload)
    assert completion_is_valid(node, payload)


def test_resume_skips_loadable_completion_and_persists_exact_protocol(
    tmp_path: Path,
) -> None:
    """A replay is resumable and leaves the full aggregation input behind."""
    payload = materialize_protocol(
        load_protocol(FROZEN_DAG),
        model_root=tmp_path / "models",
        result_root=tmp_path / "results",
        device="cpu",
    )
    resume_payload, node, result, scores = _evaluation_completion(tmp_path)
    payload["protocol"]["contract"] = resume_payload["protocol"]["contract"]
    _write_valid_evaluation(result, scores)
    _record_completion(node, payload)
    assert execute_nodes(payload, [node]) == {
        "completed": [],
        "skipped": ["eval:random_walk:rollout_2001"],
    }

    manifest = ensure_protocol_manifest(payload)
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload
    assert ensure_protocol_manifest(payload) == manifest
    changed = dict(payload)
    changed["command_count"] = 140
    with pytest.raises(BenchmarkExecutionError, match="does not match"):
        ensure_protocol_manifest(changed)


@pytest.mark.parametrize(
    ("root_kind", "protected"),
    (
        ("model", "artifacts/scratch-models"),
        ("result", "results/frozen/scratch-results"),
        ("model", "data/generated-models"),
        ("result", "src/generated-results"),
        ("result", "tests/generated-results"),
    ),
)
def test_output_roots_cannot_overlap_release_authority_or_source_paths(
    tmp_path: Path, root_kind: str, protected: str
) -> None:
    """Benchmark output must not overwrite released evidence, inputs, or code."""
    model_root = tmp_path / "models"
    result_root = tmp_path / "results"
    if root_kind == "model":
        model_root = ROOT / protected
    else:
        result_root = ROOT / protected
    with pytest.raises(UnsafeOutputRoot, match="overlaps protected"):
        validate_output_roots(model_root, result_root)

    assert validate_output_roots(
        tmp_path / "safe-models", tmp_path / "safe-results"
    ) == (
        (tmp_path / "safe-models").resolve(),
        (tmp_path / "safe-results").resolve(),
    )


def test_numbered_experiment_guard_detects_tracked_files(tmp_path: Path) -> None:
    """A tracked numbered experiment must fail the same guard used by installation."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    stale = tmp_path / "experiments/train_999_removed_branch.py"
    stale.parent.mkdir()
    stale.write_text("raise SystemExit\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", stale.relative_to(tmp_path)], cwd=tmp_path, check=True
    )
    assert tracked_numbered_experiments(tmp_path) == [
        Path("experiments/train_999_removed_branch.py")
    ]


def test_numbered_experiment_guard_detects_source_archive_files(tmp_path: Path) -> None:
    """An exported source archive must remain checkable without Git metadata."""
    stale = tmp_path / "experiments/train_999_removed_branch.py"
    stale.parent.mkdir()
    stale.write_text("raise SystemExit\n", encoding="utf-8")

    assert tracked_numbered_experiments(tmp_path) == [
        Path("experiments/train_999_removed_branch.py")
    ]


def test_numbered_experiment_guard_falls_back_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported source archive must not require a Git executable."""
    stale = tmp_path / "experiments/train_999_removed_branch.py"
    stale.parent.mkdir()
    stale.write_text("raise SystemExit\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "")

    assert tracked_numbered_experiments(tmp_path) == [
        Path("experiments/train_999_removed_branch.py")
    ]


def test_final_baseline_implementations_smoke_on_synthetic_panel() -> None:
    """The public classical and deep implementations must generate real trajectories."""
    rng = np.random.default_rng(17)
    train = rng.normal(scale=0.02, size=(96, 6)).astype(np.float32)
    history = rng.normal(scale=0.02, size=(2, 5, 6)).astype(np.float32)
    for model in (
        JointRandomWalk(train, future_len=3),
        JointBootstrap(train, future_len=3),
        JointPCAVAR(train, future_len=3, n_components=3),
    ):
        samples = model.sample_joint(history, n_samples=2, rng=rng)
        assert samples.shape == (2, 2, 3, 6)
        assert np.isfinite(samples).all()

    for name in ("deepvar", "timegrad", "path_diffusion"):
        model = build_model(
            name,
            input_dim=6,
            future_len=3,
            hidden_dim=8,
            cond_dim=8,
            num_layers=1,
            rank=2,
            n_diffusion_steps=4,
        )
        loss = model(
            torch.zeros(2, 5, 6),
            torch.zeros(2, 3, 6),
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_all_public_benchmark_entrypoints_import() -> None:
    """Deleting a final trainer, evaluator, aggregator, or reporter is a release bug."""
    for module in (
        "scenarioflow.benchmark.protocol",
        "scenarioflow.benchmark.executor",
        "scenarioflow.benchmark.guard",
        "scenarioflow.benchmark.window_scores",
        "scenarioflow.benchmark.aggregate",
        "scenarioflow.benchmark.reporting",
        "scenarioflow.benchmark.train_baselines",
        "scenarioflow.benchmark.evaluate_baselines",
        "scenarioflow.benchmark.train_deepvar_sni",
        "scenarioflow.benchmark.evaluate_deepvar_sni",
        "scenarioflow.benchmark.evaluate_scenarioflow",
    ):
        importlib.import_module(module)


def test_reproduce_launcher_exposes_full_grid_and_selected_dry_runs(
    tmp_path: Path,
) -> None:
    """The single launcher must describe the full graph and one dependency closure."""
    launcher = ROOT / "scripts/reproduce.sh"
    full = subprocess.run(
        [
            str(launcher),
            "--dry-run",
            "--model-root",
            str(tmp_path / "models"),
            "--result-root",
            str(tmp_path / "results"),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert full.returncode == 0, full.stderr
    document = json.loads(full.stdout)
    assert document["node_count"] == 141

    selected = subprocess.run(
        [
            str(launcher),
            "--dry-run",
            "--node-id",
            "eval:deepvar:train_1001:rollout_2001",
            "--model-root",
            str(tmp_path / "models"),
            "--result-root",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--smoke-max-windows",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert selected.returncode == 0, selected.stderr
    selected_document = json.loads(selected.stdout)
    assert [node["id"] for node in selected_document["nodes"]] == [
        "train:deepvar:1001",
        "eval:deepvar:train_1001:rollout_2001",
    ]

    protected = subprocess.run(
        [
            str(launcher),
            "--dry-run",
            "--model-root",
            str(tmp_path / "models"),
            "--result-root",
            str(ROOT / "results/frozen/forbidden-replay"),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert protected.returncode != 0
    assert "overlaps protected" in protected.stderr


def test_compact_provenance_pins_final_sources_and_data() -> None:
    """A reproduction claim without protocol/data hashes and the overlay caveat is invalid."""
    provenance = json.loads(
        (ROOT / "reproduction/final-benchmark-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["protocol"]["command_count"] == 141
    assert provenance["protocol"]["executable_dag_sha256"] == (
        "c6d570106975a79750d9fc47be96b07b71ea09930812c72154116a54e3a37d72"
    )
    assert provenance["sources"]["paper_baselines_revision"].startswith("25b9623")
    assert provenance["sources"]["gaussian_ccc_revision"].startswith("da91172")
    assert provenance["sources"]["scenarioflow_revision"].startswith("5d13f4a")
    assert len(provenance["data_sha256"]) == 5
    assert provenance["limitations"]["window_score_overlay_available"] is False
    mappings = provenance["implementation_mapping"]
    assert {mapping["candidate_id"] for mapping in mappings} == set(CANDIDATE_IDS)
    assert {mapping["display_name"] for mapping in mappings} == set(
        load_protocol(FROZEN_DAG)["protocol"]["display_names"].values()
    )
    assert all(mapping["clean_implementation"] for mapping in mappings)
