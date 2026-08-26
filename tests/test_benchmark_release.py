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
    completion_is_valid,
    ensure_protocol_manifest,
    execute_nodes,
)
from scenarioflow.benchmark.protocol import (
    CANDIDATE_IDS,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    SmokeOverrides,
    load_protocol,
    materialize_protocol,
    select_nodes,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DAG = ROOT / "results/frozen/multiseed/command-dag.json"


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


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


def test_resume_requires_loadable_completion_artifacts(tmp_path: Path) -> None:
    """A truncated or malformed artifact must be rerun instead of silently skipped."""
    result = tmp_path / "results_current_panel.json"
    scores = tmp_path / "window_scores.npz"
    result.write_text("{}\n", encoding="utf-8")
    np.savez_compressed(scores, schema_version=np.asarray("fmts.window_scores.v1"))
    assert completion_is_valid([result, scores])

    scores.write_bytes(b"truncated")
    assert not completion_is_valid([result, scores])

    checkpoint = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    assert completion_is_valid([checkpoint])
    checkpoint.write_bytes(b"")
    assert not completion_is_valid([checkpoint])


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
    completed = tmp_path / "already-complete.json"
    completed.write_text("{}\n", encoding="utf-8")
    node = {
        "id": "test:completed",
        "depends_on": [],
        "resume": {"completion_artifacts": [str(completed)]},
        "command": [sys.executable, "-c", "raise SystemExit(99)"],
    }
    assert execute_nodes(payload, [node]) == {
        "completed": [],
        "skipped": ["test:completed"],
    }

    manifest = ensure_protocol_manifest(payload)
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload
    assert ensure_protocol_manifest(payload) == manifest
    changed = dict(payload)
    changed["command_count"] = 140
    with pytest.raises(BenchmarkExecutionError, match="does not match"):
        ensure_protocol_manifest(changed)


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
    assert set(provenance["implementation_mapping"]) == set(CANDIDATE_IDS)
