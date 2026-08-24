from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scenarioflow.workflows.multiseed import (
    CANDIDATE_IDS,
    ROLLOUT_SEEDS,
    TRAIN_SEEDS,
    ProtocolValidationError,
    aggregate_results,
    build_protocol,
    prepare_output_root,
)
from scenarioflow.workflows.reproduction import record_completed_node


def test_multiseed_candidate_and_seed_order_is_frozen() -> None:
    assert CANDIDATE_IDS == (
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
    assert TRAIN_SEEDS == (1001, 1002, 1003)
    assert ROLLOUT_SEEDS == (2001, 2002, 2003, 2004, 2005)


def test_protocol_is_native_manifest_bound_and_resume_idempotent(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.tsv"
    artifact_manifest = tmp_path / "artifact.tsv"
    source_manifest.write_text("source\n")
    artifact_manifest.write_text("artifact\n")
    data_dir = tmp_path / "data"
    checkpoint_root = tmp_path / "checkpoints"
    artifact_root = tmp_path / "artifacts"
    data_dir.mkdir()
    checkpoint_root.mkdir()
    artifact_root.mkdir()
    output = tmp_path / "runs"
    first = build_protocol(
        output_root=output,
        source_manifest=source_manifest,
        artifact_manifest=artifact_manifest,
        data_dir=data_dir,
        checkpoint_root=checkpoint_root,
        artifact_root=artifact_root,
    )
    second = build_protocol(
        output_root=output,
        source_manifest=source_manifest,
        artifact_manifest=artifact_manifest,
        data_dir=data_dir,
        checkpoint_root=checkpoint_root,
        artifact_root=artifact_root,
    )
    assert first == second
    assert first["benchmark_split"] == "published_validation"
    assert first["train_seeds"] == [1001, 1002, 1003]
    assert first["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert all("working_directory" not in node for node in first["nodes"])
    evaluation_nodes = [node for node in first["nodes"] if node["action"] == "evaluate"]
    assert all(
        node["command"][:3] == ["python", "-m", "scenarioflow.evaluation.current_panel"]
        for node in evaluation_nodes
        if node["candidate"] != "garch_ccc"
    )
    assert any(node["action"] == "train" for node in first["nodes"])
    assert all(
        node["protocol_contract_hash"] == first["contract_hash"]
        for node in first["nodes"]
    )


def test_output_root_protection_rejects_overlap_and_mismatched_resume(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(ProtocolValidationError, match="overlaps"):
        prepare_output_root(protected / "child", protected_roots=(protected,))

    output = tmp_path / "output"
    prepare_output_root(output, protected_roots=(protected,), contract_hash="a" * 64)
    assert (
        prepare_output_root(
            output, protected_roots=(protected,), contract_hash="a" * 64
        )
        == output.resolve()
    )
    with pytest.raises(ProtocolValidationError, match="different contract"):
        prepare_output_root(
            output, protected_roots=(protected,), contract_hash="b" * 64
        )


def _result(candidate: str, value: float) -> dict:
    return {
        "baseline_name": candidate,
        "mode": "current_realvix_39_state_panel",
        "n_windows": 441,
        "n_samples": 64,
        "state_specs": [{}] * 39,
        "panel_b_crps": {"overall": value, "iv_crps": value, "factor_crps": value},
        "panel_b_energy_score": {"overall": value},
        "panel_b_variogram_score": {"variogram_score": value},
        "panel_b_correlation": {"corr_score": 0.5},
        "panel_b_factor_ks": {"mean_statistic": 0.1},
        "joint_panel_quality": {"factor_tail_q99_ratio_median": 1.0},
        "current_panel_iv_metrics": {
            "cov90": 0.9,
            "calibration_error": 0.1,
            "level_ks_median": 0.1,
            "median_abs_dev": 0.1,
            "width_rho": 0.3,
            "mr_ratio": 1.0,
            "economic_ratio": 1.0,
            "jump_ks": 0.2,
            "kurtosis_ratio": 1.0,
            "acf_corr": 0.9,
        },
    }


def _window_scores(value: float) -> dict[str, np.ndarray]:
    return {
        "window_indices": np.arange(4010, 4451),
        "crps_contributions": np.full((441, 4, 39), value),
        "energy_contributions": np.full((441, 4), value),
        "variogram_h30_contributions": np.full(441, value),
    }


def test_aggregation_preserves_training_then_rollout_hierarchy_and_screens() -> None:
    runs = []
    for train_seed in TRAIN_SEEDS:
        for rollout_seed in ROLLOUT_SEEDS:
            value = float((train_seed - 1000) * 10 + (rollout_seed - 2000))
            runs.append(
                {
                    "candidate": "deepvar",
                    "train_seed": train_seed,
                    "rollout_seed": rollout_seed,
                    "result": _result("deepvar", value),
                    "window_scores": _window_scores(value),
                }
            )
    summary = aggregate_results(runs, candidates=("deepvar",))
    assert summary["protocol"]["training_seeds"] == [1001, 1002, 1003]
    assert summary["protocol"]["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]
    deepvar = summary["candidates"]["deepvar"]
    assert deepvar["variation_type"] == "training_then_rollout_sensitivity"
    assert len(deepvar["rollout_sensitivity_by_training_seed"]) == 3
    assert deepvar["training_seed_sensitivity"]["metrics"]["CRPS"]["n"] == 3
    assert deepvar["training_seed_sensitivity"]["metrics"]["CRPS"]["mean"] == 23.0
    assert deepvar["pass_profile"]["n_runs"] == 15

    with pytest.raises(ProtocolValidationError, match="complete"):
        aggregate_results(runs[:-1], candidates=("deepvar",))
    with pytest.raises(ProtocolValidationError, match="duplicate"):
        aggregate_results(runs + [runs[0]], candidates=("deepvar",))
    missing_metric = [dict(run) for run in runs]
    missing_metric[0] = dict(
        missing_metric[0], result=dict(missing_metric[0]["result"])
    )
    del missing_metric[0]["result"]["panel_b_energy_score"]
    with pytest.raises(ProtocolValidationError, match="Energy Score"):
        aggregate_results(missing_metric, candidates=("deepvar",))


def test_paired_comparison_preserves_seed_pairs_and_chronological_bootstrap() -> None:
    runs = []
    for candidate, shift in (("deepvar", 0.0), ("deepvar_sni", -0.25)):
        for train_seed in TRAIN_SEEDS:
            for rollout_seed in ROLLOUT_SEEDS:
                value = 1.0 + shift
                runs.append(
                    {
                        "candidate": candidate,
                        "train_seed": train_seed,
                        "rollout_seed": rollout_seed,
                        "result": _result(candidate, value),
                        "window_scores": _window_scores(value),
                    }
                )
    summary = aggregate_results(
        runs,
        candidates=("deepvar", "deepvar_sni"),
        paired_comparisons=(("interface", "deepvar", "deepvar_sni"),),
        n_bootstrap=100,
    )
    paired = summary["paired_comparisons"]["interface"]
    assert paired["n_seed_pairs"] == 15
    assert paired["window_uncertainty"]["CRPS"]["mean_delta"] == -0.25


def test_node_completion_is_hash_bound_and_resumable(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.tsv"
    artifact_manifest = tmp_path / "artifact.tsv"
    source_manifest.write_text("source\n")
    artifact_manifest.write_text("artifact\n")
    for name in ("data", "checkpoints", "artifacts"):
        (tmp_path / name).mkdir()
    protocol = build_protocol(
        output_root=tmp_path / "runs",
        source_manifest=source_manifest,
        artifact_manifest=artifact_manifest,
        data_dir=tmp_path / "data",
        checkpoint_root=tmp_path / "checkpoints",
        artifact_root=tmp_path / "artifacts",
    )
    node = next(
        item
        for item in protocol["nodes"]
        if item["action"] == "evaluate"
        and item["candidate"] == "deepvar"
        and item["train_seed"] == 1001
        and item["rollout_seed"] == 2001
    )
    result = _result("deepvar", 1.0)
    output = Path(node["destination"])
    first = record_completed_node(
        output_dir=output,
        protocol=protocol,
        node_id=node["node_id"],
        result=result,
    )
    second = record_completed_node(
        output_dir=output,
        protocol=protocol,
        node_id=node["node_id"],
        result=result,
    )
    assert first["resumed"] is False
    assert second["resumed"] is True
    assert {key: value for key, value in first.items() if key != "resumed"} == {
        key: value for key, value in second.items() if key != "resumed"
    }

    changed = dict(result, generation_time_s=1.0)
    with pytest.raises(ProtocolValidationError, match="different completed node"):
        record_completed_node(
            output_dir=output,
            protocol=protocol,
            node_id=node["node_id"],
            result=changed,
        )
    copied = tmp_path / "runs" / "copied"
    copied.mkdir()
    with pytest.raises(ProtocolValidationError, match="declared destination"):
        record_completed_node(
            output_dir=copied,
            protocol=protocol,
            node_id=node["node_id"],
            result=result,
        )
