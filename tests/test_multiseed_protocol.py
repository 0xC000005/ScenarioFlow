from __future__ import annotations

import json
from pathlib import Path

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
    output = tmp_path / "runs"
    first = build_protocol(
        output_root=output,
        source_manifest=source_manifest,
        artifact_manifest=artifact_manifest,
    )
    second = build_protocol(
        output_root=output,
        source_manifest=source_manifest,
        artifact_manifest=artifact_manifest,
    )
    assert first == second
    assert first["benchmark_split"] == "published_validation"
    assert first["train_seeds"] == [1001, 1002, 1003]
    assert first["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert all("working_directory" not in node for node in first["nodes"])
    assert all(
        node["command"][:3] == ["python", "-m", "scenarioflow.workflows.reproduction"]
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


def test_aggregation_requires_complete_three_by_five_schema() -> None:
    runs = []
    for train_seed in TRAIN_SEEDS:
        for rollout_seed in ROLLOUT_SEEDS:
            runs.append(
                {
                    "candidate": "deepvar",
                    "train_seed": train_seed,
                    "rollout_seed": rollout_seed,
                    "result": {
                        "schema_version": "scenarioflow.current-panel-result.v1",
                        "baseline_name": "deepvar",
                        "mode": "current_realvix_39_state_panel",
                        "n_windows": 441,
                        "n_samples": 64,
                        "panel_b_crps": {"overall": float(train_seed + rollout_seed)},
                    },
                }
            )
    summary = aggregate_results(runs, candidates=("deepvar",))
    assert summary["protocol"]["training_seeds"] == [1001, 1002, 1003]
    assert summary["protocol"]["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert summary["candidates"]["deepvar"]["run_count"] == 15
    assert summary["candidates"]["deepvar"]["metrics"]["CRPS"]["count"] == 15

    with pytest.raises(ProtocolValidationError, match="complete"):
        aggregate_results(runs[:-1], candidates=("deepvar",))


def test_node_completion_is_hash_bound_and_resumable(tmp_path: Path) -> None:
    result = {
        "schema_version": "scenarioflow.current-panel-result.v1",
        "baseline_name": "deepvar",
        "mode": "current_realvix_39_state_panel",
        "n_windows": 441,
        "n_samples": 64,
    }
    output = tmp_path / "node"
    first = record_completed_node(
        output_dir=output,
        candidate="deepvar",
        train_seed=1001,
        rollout_seed=2001,
        result=result,
    )
    second = record_completed_node(
        output_dir=output,
        candidate="deepvar",
        train_seed=1001,
        rollout_seed=2001,
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
            candidate="deepvar",
            train_seed=1001,
            rollout_seed=2001,
            result=changed,
        )
