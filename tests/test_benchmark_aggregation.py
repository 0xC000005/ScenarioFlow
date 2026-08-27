"""End-to-end contracts for semantic benchmark rerun aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from scenarioflow.benchmark.aggregate import (
    AggregationContractError,
    aggregate_study,
)
from scenarioflow.benchmark.executor import (
    _record_completion,
    completion_is_valid,
    ensure_protocol_manifest,
)
from scenarioflow.benchmark.protocol import (
    SmokeOverrides,
    load_protocol,
    materialize_protocol,
    topological_nodes,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DAG = ROOT / "results/frozen/multiseed/command-dag.json"
WINDOW_INDICES = np.asarray([4511, 4512], dtype=np.int64)
VARIABLE_NAMES = np.asarray([f"state:{index}" for index in range(39)])
ROLLOUT_ADJUSTMENTS = {2001: 0.0, 2002: 2.0, 2003: 4.0, 2004: 6.0, 2005: 8.0}


def _artifact(node: dict, filename: str) -> Path:
    matches = [
        Path(path)
        for path in node["resume"]["completion_artifacts"]
        if Path(path).name == filename
    ]
    assert len(matches) == 1
    return matches[0]


def _command_option(node: dict, name: str, default: str) -> str:
    command = node["command"]
    if name not in command:
        return default
    return command[command.index(name) + 1]


def _crps_value(node: dict) -> float:
    candidate = node["candidate"]
    rollout_seed = int(node["rollout_seed"])
    rollout = ROLLOUT_ADJUSTMENTS[rollout_seed]
    if candidate in {"scenarioflow", "scenarioflow_graph"}:
        training = {1001: 10.0, 1002: 20.0, 1003: 40.0}[int(node["training_seed"])]
        return training + rollout + (1.0 if candidate == "scenarioflow_graph" else 0.0)
    if candidate in {"deepvar", "deepvar_sni"}:
        training = {1001: 5.0, 1002: 7.0, 1003: 9.0}[int(node["training_seed"])]
        return training + rollout / 10.0 + (-0.5 if candidate == "deepvar_sni" else 0.0)
    if node.get("training_seed") is not None:
        learned_offset = {
            "timegrad": 12.0,
            "path_diffusion": 14.0,
        }[candidate]
        return learned_offset + (int(node["training_seed"]) - 1000) + rollout / 10.0
    classical_offset = {
        "random_walk": 0.0,
        "bootstrap": 10.0,
        "historical_sim": 20.0,
        "pca_var": 30.0,
        "garch_ccc": 40.0,
        "filtered_hs": 50.0,
    }[candidate]
    return classical_offset + float(rollout_seed - 2000)


def _window_pattern(candidate: str, scale: float) -> np.ndarray:
    if candidate == "scenarioflow_graph":
        return np.asarray([-3.0 * scale, 3.0 * scale])
    return np.asarray([-scale, scale])


def _result_document(node: dict) -> tuple[dict, dict[str, np.ndarray]]:
    candidate = str(node["candidate"])
    crps = _crps_value(node)
    energy = crps + 0.5
    variogram = 0.2 + crps / 1_000.0
    result = {
        "baseline_name": candidate,
        "mode": "current_realvix_39_state_panel",
        "n_windows": 2,
        "n_samples": 2,
        "state_specs": [{"name": name} for name in VARIABLE_NAMES.tolist()],
        "current_panel_iv_metrics": {
            "cov90": 0.9,
            "calibration_error": 0.1,
            "level_ks_median": 0.1,
            "median_abs_dev": 0.1,
            "width_rho": 0.5,
            "mr_ratio": 1.0,
            "economic_ratio": 1.0,
            "jump_ks": 0.1,
            "kurtosis_ratio": 1.0,
            "acf_corr": 0.8,
        },
        "joint_panel_quality": {
            "future_len": 30,
            "factor_tail_q99_ratio_median": 1.0,
        },
        "panel_b_crps": {
            "overall": crps,
            "iv_crps": crps / 100.0,
            "factor_crps": crps * 2.0,
        },
        "panel_b_energy_score": {"overall": energy},
        "panel_b_variogram_score": {"variogram_score": variogram},
        "panel_b_correlation": {"corr_score": 0.5},
        "panel_b_factor_ks": {"mean_statistic": 0.1},
    }
    crps_by_window = crps + _window_pattern(candidate, 0.25)
    energy_by_window = energy + _window_pattern(candidate, 0.1)
    variogram_by_window = variogram + _window_pattern(candidate, 0.001)
    scores = {
        "schema_version": np.asarray("fmts.window_scores.v1"),
        "window_indices": WINDOW_INDICES,
        "horizons": np.asarray([1, 7, 14, 30], dtype=np.int64),
        "variable_names": VARIABLE_NAMES,
        "crps_contributions": np.broadcast_to(
            crps_by_window[:, None, None], (2, 4, 39)
        ).copy(),
        "energy_contributions": np.broadcast_to(
            energy_by_window[:, None], (2, 4)
        ).copy(),
        "variogram_h30_contributions": variogram_by_window,
        "coverage_levels": np.asarray([0.5, 0.8, 0.9, 0.95]),
        "coverage_by_window": np.zeros((2, 4)),
        "iv_history_activity": np.zeros(2),
        "iv_deck_width": np.zeros(2),
        "full_panel_history_activity": np.zeros(2),
        "full_panel_deck_width": np.zeros(2),
    }
    return result, scores


def _write_evaluation_completion(node: dict, payload: dict) -> None:
    result_path = _artifact(node, "results_current_panel.json")
    score_path = _artifact(node, "window_scores.npz")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result, scores = _result_document(node)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(score_path, **scores)
    _record_completion(node, payload)


def _write_training_completion(node: dict, payload: dict) -> None:
    checkpoint = _artifact(node, "best_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    candidate = str(node["candidate"])
    config = {
        "input_dim": 39,
        "history_len": 30,
        "future_len": 30,
    }
    document: dict = {
        "model_state_dict": {"weight": torch.ones(1)},
        "config": config,
    }
    if candidate in {"scenarioflow", "scenarioflow_graph"}:
        config["conditioning_mode"] = "prefix"
        if candidate == "scenarioflow_graph":
            config["velocity_mixer_mode"] = "adaptive_graph_residual"
        document.update(
            {
                "model_coordinate": "state_aware_normalized_innovation",
                "state_scope": _command_option(node, "--state_scope", "joint38"),
                "normalization": {
                    "coordinate": "encoded_increment",
                    "innovation_coordinate": "normalized",
                },
                "state_specs": [{"name": name} for name in VARIABLE_NAMES.tolist()],
            }
        )
        if node["configuration"]["stage"] == "stage1":
            document["training_objective"] = {"name": "next_step_flow_matching"}
        else:
            document["finetune_objective"] = {"name": "rollout_energy"}
            document["source_checkpoint"] = "synthetic-authority-checkpoint.pt"
    else:
        document["model_type"] = {
            "deepvar": "deepvar",
            "deepvar_sni": "deepvar",
            "timegrad": "timegrad",
            "path_diffusion": "path_diffusion",
        }[candidate]
    torch.save(document, checkpoint)
    _record_completion(node, payload)


def _completed_grid(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    model_root = tmp_path / "models"
    result_root = tmp_path / "results"
    payload = materialize_protocol(
        load_protocol(FROZEN_DAG),
        model_root=model_root,
        result_root=result_root,
        device="cpu",
        smoke=SmokeOverrides(max_windows=2, samples=2),
    )
    for node in topological_nodes(payload):
        if node["action"] == "train":
            _write_training_completion(node, payload)
        else:
            _write_evaluation_completion(node, payload)
    assert all(
        completion_is_valid(node, payload) for node in topological_nodes(payload)
    )
    manifest = ensure_protocol_manifest(payload)
    return payload, model_root, result_root, manifest


@pytest.fixture
def completed_grid(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    return _completed_grid(tmp_path)


def test_semantic_aggregate_uses_independent_seed_units(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    payload, _model_root, result_root, manifest = completed_grid
    output = result_root / "aggregate"

    summary = aggregate_study(
        protocol_manifest=manifest,
        output_dir=output,
    )

    assert summary["artifact_kind"] == "semantic_rerun_output"
    assert summary["historical_byte_identity_claimed"] is False
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "summary.md",
        "per-run-metrics.csv",
        "per-run-thresholds.csv",
        "metric-summaries.csv",
    }

    learned = summary["candidate_summaries"]["scenarioflow"]
    learned_crps = learned["training_seed_sensitivity"]["metrics"]["CRPS"]
    assert learned["training_seed_sensitivity"]["unit"] == (
        "per_training_seed_rollout_mean"
    )
    assert learned_crps == pytest.approx(
        {
            "n": 3,
            "mean": 27.333333333333332,
            "std": 15.275252316519467,
            "min": 14.0,
            "max": 44.0,
            "ci_level": 0.95,
            "ci_low": -10.612497002634267,
            "ci_high": 65.27916366930093,
        }
    )

    classical = summary["candidate_summaries"]["random_walk"]
    classical_crps = classical["sampling_sensitivity"]["metrics"]["CRPS"]
    assert classical_crps == pytest.approx(
        {
            "n": 5,
            "mean": 3.0,
            "std": 1.5811388300841898,
            "min": 1.0,
            "max": 5.0,
            "ci_level": 0.95,
            "ci_low": 1.0367568385224428,
            "ci_high": 4.963243161477557,
        }
    )

    with (output / "metric-summaries.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        item
        for item in rows
        if item["candidate"] == "scenarioflow"
        and item["variation"] == "across_training_seed_means"
        and item["metric"] == "CRPS"
    )
    assert row["n"] == "3"
    assert row["independent_unit"] == "training_seed_rollout_mean"
    assert len(summary["per_runs"]) == 120
    assert (
        summary["protocol_dag_sha256"] == payload["protocol"]["executable_dag_sha256"]
    )


def test_semantic_aggregate_labels_the_requested_confidence_level(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    _payload, _model_root, result_root, manifest = completed_grid
    output = result_root / "aggregate"

    summary = aggregate_study(
        protocol_manifest=manifest,
        output_dir=output,
        ci_level=0.90,
    )

    assert summary["methodology_contract"]["confidence_interval"] == (
        "two_sided_90_percent_student_t"
    )
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "90% Student-t interval" in markdown
    assert "moving-block" not in markdown
    assert "95%" not in markdown


@pytest.mark.parametrize("tamper", ("missing_sidecar", "stale_hash"))
def test_aggregate_rejects_noncurrent_executor_completions(
    completed_grid: tuple[dict, Path, Path, Path], tamper: str
) -> None:
    payload, _model_root, result_root, manifest = completed_grid
    node = next(
        item
        for item in payload["commands"]
        if item["id"] == "eval:random_walk:rollout_2001"
    )
    if tamper == "missing_sidecar":
        Path(node["sidecar_manifest"]["path"]).unlink()
    else:
        result = _artifact(node, "results_current_panel.json")
        result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not completion_is_valid(node, payload)

    with pytest.raises(AggregationContractError, match="executor-validated"):
        aggregate_study(
            protocol_manifest=manifest,
            output_dir=result_root / "aggregate",
        )
    assert not (result_root / "aggregate").exists()


def test_aggregate_rejects_hashed_npz_that_disagrees_with_result_contract(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    payload, _model_root, result_root, manifest = completed_grid
    node = next(
        item
        for item in payload["commands"]
        if item["id"] == "eval:random_walk:rollout_2001"
    )
    scores = _artifact(node, "window_scores.npz")
    with np.load(scores, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["crps_contributions"] = arrays["crps_contributions"] + 1.0
    np.savez_compressed(scores, **arrays)
    _record_completion(node, payload)
    assert completion_is_valid(node, payload)

    with pytest.raises(AggregationContractError, match="does not match result JSON"):
        aggregate_study(
            protocol_manifest=manifest,
            output_dir=result_root / "aggregate",
        )
    assert not (result_root / "aggregate").exists()


def test_aggregate_accepts_float_recomputation_rounding(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    payload, _model_root, result_root, manifest = completed_grid
    node = next(
        item
        for item in payload["commands"]
        if item["id"] == "eval:random_walk:rollout_2001"
    )
    scores = _artifact(node, "window_scores.npz")
    with np.load(scores, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["energy_contributions"] *= 1.0 + 5e-5
    arrays["variogram_h30_contributions"] *= 1.0 - 5e-5
    np.savez_compressed(scores, **arrays)
    _record_completion(node, payload)

    summary = aggregate_study(
        protocol_manifest=manifest,
        output_dir=result_root / "aggregate",
    )

    assert summary["artifact_kind"] == "semantic_rerun_output"


def test_resume_all_skips_complete_fixture_and_writes_stable_aggregate(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    _payload, model_root, result_root, _manifest = completed_grid
    result = subprocess.run(
        [
            str(ROOT / "scripts/reproduce.sh"),
            "--resume-all",
            "--model-root",
            str(model_root),
            "--result-root",
            str(result_root),
            "--device",
            "cpu",
            "--smoke-max-windows",
            "2",
            "--smoke-samples",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    completion = json.loads(result.stdout)
    assert completion["completed"] == []
    assert len(completion["skipped"]) == 141
    assert completion["aggregate_path"] == str(result_root / "aggregate")
    assert (result_root / "aggregate/summary.json").is_file()


def test_selected_node_run_does_not_aggregate(
    completed_grid: tuple[dict, Path, Path, Path],
) -> None:
    _payload, model_root, result_root, _manifest = completed_grid
    result = subprocess.run(
        [
            str(ROOT / "scripts/reproduce.sh"),
            "--node-id",
            "eval:random_walk:rollout_2001",
            "--model-root",
            str(model_root),
            "--result-root",
            str(result_root),
            "--device",
            "cpu",
            "--smoke-max-windows",
            "2",
            "--smoke-samples",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    completion = json.loads(result.stdout)
    assert completion == {
        "completed": [],
        "skipped": ["eval:random_walk:rollout_2001"],
    }
    assert not (result_root / "aggregate").exists()


def test_public_workflow_documents_semantic_rerun_aggregate() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "build/reproduction/results/aggregate" in readme
    assert "semantic rerun output" in readme
    assert "historical byte-identical artifact" in readme
