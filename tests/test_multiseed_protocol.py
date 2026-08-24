from __future__ import annotations

import json
import hashlib
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
import scenarioflow.workflows.reproduction as reproduction


SOURCE_AUTHORITY_ROWS = {
    "experiments/backfill/baselines/evaluate_current_panel_baselines.py": "7224278d74ee8bfd86c81eb2d1ddc485a1bf5fd1a681f14f22f41749b1cb0147",
    "experiments/backfill/baselines/fmts_dag_executor.py": "9a6254b59b58db27cc184d6848a001cef489a2df48c77ff5d7b73953b8ea97ff",
    "experiments/backfill/baselines/run_fmts_multiseed_benchmark.py": "1d362bb5bce8f97267ad2afe1976b1da1d970e5741e6830d2184903f0f48c2ab",
    "experiments/backfill/baselines/train_current_panel_deep_baselines.py": "5091da02d11870d3721ce8f5af25f53137b550bae2f87bcdb38bb12e46ab0e63",
    "experiments/backfill/baselines/train_g2_deepvar_sni_interface.py": "be8e1eee5816b9cf1d6314db53ee71f0fad0eb24e58c5fee44b09e4dce7bd1d0",
    "experiments/backfill/block_ar/train_666a_normalized_innovation_rollout_energy_finetune.py": "39181197d48357746c9ffb73a49e156f3011ed48634507e8bac24a62895b484a",
    "paper/generated_tables/table_baseline_comparison.tex": "1492cf31ee50c99ded8c068c714c98fb0c793a0dce4f49137c20d37cccfd2589",
}
ARTIFACT_AUTHORITY_ROWS = {
    "data/multi_factor_levels.parquet": "3ac3aeb04fd227fb471bc026f5f4191f5655c4cf3e09443b102e4d6d0019c75f",
    "data/multi_factor_returns.parquet": "77970f0b506c1ea6bd0ab45b539c621e2cc4199b10b223f2a07c6d4a2003f21d",
    "data/spx_vol_surface_history_full_data_fixed.parquet": "d657a9cf50ba513fa1806aaf8b1a30a7646438bda09ebeabc5f6804f6e0f704a",
    "data/vol_surface_with_ret.npz": "3e5d05a4cc72cfbd60646aab6f491aac763d4667144409dd72f7969e40b4d50f",
}


def _write_authority_manifests(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.tsv"
    source.write_text(
        "decision\tcategory\tpath\tbytes\tsha256\tgit_state\tpurpose\n"
        + "".join(
            f"include\tsource\t{path}\t1\t{digest}\ttracked\ttest\n"
            for path, digest in SOURCE_AUTHORITY_ROWS.items()
        )
    )
    artifacts = tmp_path / "artifacts.tsv"
    artifacts.write_text(
        "decision\tcategory\tpath\tbytes\tsha256\tpurpose\trelease_location\tgit_state\n"
        + "".join(
            f"include\tdata\t{path}\t1\t{digest}\ttest\tretained\tignored\n"
            for path, digest in ARTIFACT_AUTHORITY_ROWS.items()
        )
    )
    return source, artifacts


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
    source_manifest, artifact_manifest = _write_authority_manifests(tmp_path)
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
    retired_input_slug = "training_" + "inputs"
    assert all(
        retired_input_slug not in " ".join(node["command"]) for node in first["nodes"]
    )
    by_id = {node["node_id"]: node for node in first["nodes"]}
    for seed in TRAIN_SEEDS:
        stage1 = by_id[f"train:scenarioflow:stage1:{seed}"]
        stage2 = by_id[f"train:scenarioflow:stage2:{seed}"]
        graph = by_id[f"train:scenarioflow_graph:graph:{seed}"]
        assert stage2["depends_on"] == [stage1["node_id"]]
        assert graph["depends_on"] == [stage2["node_id"]]
        assert stage1["command"][stage1["command"].index("--epochs") + 1] == "8"
        assert stage2["command"][stage2["command"].index("--epochs") + 1] == "3"
        assert graph["command"][graph["command"].index("--epochs") + 1] == "2"
        assert stage1["command"][stage1["command"].index("--batch-size") + 1] == "32"
        assert (
            stage1["command"][stage1["command"].index("--max-train-windows") + 1]
            == "2048"
        )
        for fine_tune in (stage2, graph):
            assert (
                fine_tune["command"][fine_tune["command"].index("--batch-size") + 1]
                == "8"
            )
            assert (
                fine_tune["command"][
                    fine_tune["command"].index("--train-sample-count") + 1
                ]
                == "2"
            )
            assert (
                fine_tune["command"][
                    fine_tune["command"].index("--rollout-flow-steps") + 1
                ]
                == "4"
            )
            assert (
                fine_tune["command"][fine_tune["command"].index("--weight-decay") + 1]
                == "0.00001"
            )
            assert (
                fine_tune["command"][fine_tune["command"].index("--clip-grad") + 1]
                == "1.0"
            )
            assert "--drop-last" in fine_tune["command"]
            assert (
                fine_tune["command"][
                    fine_tune["command"].index("--channel-level-energy-coordinate") + 1
                ]
                == "level"
            )
        assert (
            graph["command"][graph["command"].index("--free-running-fm-weight") + 1]
            == "0.2"
        )
        assert (
            graph["command"][
                graph["command"].index("--free-running-fm-prefix-steps") + 1
            ]
            == "5"
        )
        assert (
            stage2["command"][stage2["command"].index("--max-train-windows") + 1] == "0"
        )
        assert (
            graph["command"][graph["command"].index("--max-train-windows") + 1]
            == "2048"
        )
    for node in first["nodes"]:
        if node["action"] == "train" and node["candidate"] in {
            "deepvar",
            "timegrad",
            "path_diffusion",
            "deepvar_sni",
        }:
            assert node["command"][node["command"].index("--epochs") + 1] == "10"
            assert (
                node["command"][node["command"].index("--n-diffusion-steps") + 1]
                == "40"
            )
            assert node["command"][node["command"].index("--sample-steps") + 1] == "8"
            assert (
                node["command"][node["command"].index("--weight-decay") + 1]
                == "0.00001"
            )
            assert node["command"][node["command"].index("--clip-grad") + 1] == "1.0"
    assert any(node["action"] == "train" for node in first["nodes"])
    assert all(
        node["protocol_contract_hash"] == first["contract_hash"]
        for node in first["nodes"]
    )


def test_protocol_rejects_forged_frozen_manifest_row(tmp_path: Path) -> None:
    source, artifacts = _write_authority_manifests(tmp_path)
    source.write_text(
        source.read_text().replace(next(iter(SOURCE_AUTHORITY_ROWS.values())), "0" * 64)
    )
    for name in ("data", "checkpoints", "artifact-root"):
        (tmp_path / name).mkdir()
    with pytest.raises(ProtocolValidationError, match="source authority"):
        build_protocol(
            output_root=tmp_path / "runs",
            source_manifest=source,
            artifact_manifest=artifacts,
            data_dir=tmp_path / "data",
            checkpoint_root=tmp_path / "checkpoints",
            artifact_root=tmp_path / "artifact-root",
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
        "generation_time_s": 0.0,
        "n_windows": 441,
        "n_samples": 64,
        "state_specs": [{}] * 39,
        "current_panel_iv": {},
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


def _write_evaluation_sidecars(
    output: Path, candidate: str, result: dict, value: float = 1.0
) -> dict:
    candidate_dir = output / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "results_current_panel.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.savez(candidate_dir / "window_scores.npz", **_window_scores(value))
    manifest = {
        "mode": "current_realvix_39_state_panel",
        "benchmark_split": "published_validation",
        "columns": [f"column-{index}" for index in range(39)],
        "metadata": {
            "dates_start": "2000-01-03 00:00:00",
            "dates_end": "2023-02-24 00:00:00",
            "source_columns": [f"column-{index}" for index in range(39)],
            "positive_level_policy": "reference_based",
            "iv_transform": "bounded_logit",
            "iv_lower_bound": 1e-4,
            "iv_upper_bound": 1.0,
            "cleaning_report": {"enabled": True},
        },
        "state_specs": [{}] * 39,
        "n_train_windows": 4010,
        "n_val_windows": 441,
        "n_samples": 64,
        "seed": 2001,
        "iv_alignment": {
            "history_max_abs_error": 0.0,
            "future_max_abs_error": 0.0,
        },
        "provenance": {
            "data_artifacts": {
                Path(path).name: digest
                for path, digest in ARTIFACT_AUTHORITY_ROWS.items()
            },
            "evaluator": {
                "baseline": candidate,
                "batch_size": 32,
                "max_windows": 441,
                "n_samples": 64,
                "seed": 2001,
            },
        },
    }
    (candidate_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


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
    assert {
        item["bootstrap_seed"] for item in paired["window_uncertainty"].values()
    } == {20260805}


def test_aggregate_cli_always_requests_both_frozen_paired_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    monkeypatch.setattr(reproduction, "collect_completed_runs", lambda _root: [])

    def fake_aggregate(_runs, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(reproduction, "aggregate_results", fake_aggregate)
    reproduction.main(
        [
            "aggregate",
            "--output-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "summary.json"),
        ]
    )
    assert captured["paired_comparisons"] == (
        ("deepvar_vs_deepvar_support", "deepvar", "deepvar_sni"),
        ("scenarioflow_vs_graph", "scenarioflow", "scenarioflow_graph"),
    )


def test_node_completion_is_hash_bound_and_resumable(tmp_path: Path) -> None:
    source_manifest, artifact_manifest = _write_authority_manifests(tmp_path)
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
        and item["candidate"] == "bootstrap"
        and item["train_seed"] is None
        and item["rollout_seed"] == 2001
    )
    result = _result("bootstrap", 1.0)
    output = Path(node["destination"])
    evaluation_manifest = _write_evaluation_sidecars(output, "bootstrap", result)
    first = record_completed_node(
        output_dir=output,
        protocol=protocol,
        node_id=node["node_id"],
        result=result,
        evaluation_manifest=evaluation_manifest,
    )
    second = record_completed_node(
        output_dir=output,
        protocol=protocol,
        node_id=node["node_id"],
        result=result,
        evaluation_manifest=evaluation_manifest,
    )
    assert first["resumed"] is False
    assert second["resumed"] is True
    assert {key: value for key, value in first.items() if key != "resumed"} == {
        key: value for key, value in second.items() if key != "resumed"
    }

    changed = dict(result, generation_time_s=1.0)
    with pytest.raises(ProtocolValidationError, match="differs|different"):
        record_completed_node(
            output_dir=output,
            protocol=protocol,
            node_id=node["node_id"],
            result=changed,
            evaluation_manifest=evaluation_manifest,
        )
    copied = tmp_path / "runs" / "copied"
    copied.mkdir()
    with pytest.raises(ProtocolValidationError, match="declared destination"):
        record_completed_node(
            output_dir=copied,
            protocol=protocol,
            node_id=node["node_id"],
            result=result,
            evaluation_manifest=evaluation_manifest,
        )

    evaluator_path = output / "bootstrap" / "manifest.json"
    run_manifest_path = output / "run_manifest.json"
    forged_evaluator = json.loads(evaluator_path.read_text())
    forged_evaluator["iv_alignment"]["future_max_abs_error"] = 2e-6
    evaluator_path.write_text(
        json.dumps(forged_evaluator, indent=2, sort_keys=True) + "\n"
    )
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["evaluation_manifest"] = forged_evaluator
    run_manifest["evaluation_manifest_sha256"] = hashlib.sha256(
        evaluator_path.read_bytes()
    ).hexdigest()
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ProtocolValidationError, match="alignment exceeds"):
        reproduction.collect_completed_runs(protocol["output_root"])

    evaluator_path.write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n"
    )
    run_manifest["evaluation_manifest"] = evaluation_manifest
    run_manifest["evaluation_manifest_sha256"] = hashlib.sha256(
        evaluator_path.read_bytes()
    ).hexdigest()
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )

    np.savez(output / "bootstrap" / "window_scores.npz", **_window_scores(2.0))
    with pytest.raises(ProtocolValidationError, match="window scores checksum"):
        reproduction.collect_completed_runs(protocol["output_root"])


def test_node_completion_requires_evaluation_manifest(tmp_path: Path) -> None:
    source, artifacts = _write_authority_manifests(tmp_path)
    for name in ("data", "checkpoints", "artifact-root"):
        (tmp_path / name).mkdir()
    protocol = build_protocol(
        output_root=tmp_path / "runs",
        source_manifest=source,
        artifact_manifest=artifacts,
        data_dir=tmp_path / "data",
        checkpoint_root=tmp_path / "checkpoints",
        artifact_root=tmp_path / "artifact-root",
    )
    node = next(
        item
        for item in protocol["nodes"]
        if item["action"] == "evaluate"
        and item["candidate"] == "bootstrap"
        and item["rollout_seed"] == 2001
    )
    with pytest.raises(ProtocolValidationError, match="evaluation manifest"):
        record_completed_node(
            output_dir=node["destination"],
            protocol=protocol,
            node_id=node["node_id"],
            result=_result("bootstrap", 1.0),
        )


def test_one_command_smoke_trains_scenarioflow_dag_evaluates_and_resumes(
    tmp_path: Path,
) -> None:
    source, artifacts = _write_authority_manifests(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    artifact_root = tmp_path / "artifact-root"
    checkpoint_root.mkdir()
    artifact_root.mkdir()
    output_root = tmp_path / "runs"
    summary = tmp_path / "summary.json"
    command = [
        "reproduce",
        "--smoke",
        "--candidate",
        "scenarioflow",
        "--candidate",
        "scenarioflow_graph",
        "--train-seed",
        "1001",
        "--output-root",
        str(output_root),
        "--source-manifest",
        str(source),
        "--artifact-manifest",
        str(artifacts),
        "--data-dir",
        "data",
        "--checkpoint-root",
        str(checkpoint_root),
        "--artifact-root",
        str(artifact_root),
        "--max-windows",
        "1",
        "--n-samples",
        "2",
        "--rollout-seed",
        "2001",
        "--output",
        str(summary),
    ]
    reproduction.main(command)
    first = json.loads(summary.read_text())
    assert first["schema_version"] == "scenarioflow.multiseed-smoke-aggregate.v1"
    assert first["candidates"]["scenarioflow"]["n_runs"] == 1
    assert first["candidates"]["scenarioflow_graph"]["n_runs"] == 1
    training_manifests = list(output_root.rglob("*.node_manifest.json"))
    assert len(training_manifests) == 3
    assert len(list(output_root.rglob("run_manifest.json"))) == 2
    for manifest_path in output_root.rglob("run_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        assert manifest["evaluation_manifest_sha256"]
        assert manifest["evaluation_manifest"]["provenance"]["data_artifacts"]
        assert manifest["benchmark_split"] == "published_validation_prefix_smoke"
    reproduction.main(command)
    assert json.loads(summary.read_text()) == first
