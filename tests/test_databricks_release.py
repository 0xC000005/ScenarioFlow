"""Contract tests for the optional Databricks job wrapper."""

from __future__ import annotations

from pathlib import Path

import yaml

from scenarioflow.workflows.multiseed import build_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_reproduction_manifests_build_the_native_protocol(
    tmp_path: Path,
) -> None:
    """Cloud execution must use the same frozen authorities as native execution."""
    checkpoint_root = tmp_path / "checkpoints"
    artifact_root = tmp_path / "artifacts"
    checkpoint_root.mkdir()
    artifact_root.mkdir()
    protocol = build_protocol(
        output_root=tmp_path / "runs",
        source_manifest=ROOT / "reproduction" / "FROZEN_SOURCE.tsv",
        artifact_manifest=ROOT / "reproduction" / "FROZEN_DATA.tsv",
        data_dir=ROOT / "data",
        checkpoint_root=checkpoint_root,
        artifact_root=artifact_root,
    )
    assert protocol["candidate_ids"] == [
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
    ]
    assert protocol["train_seeds"] == [1001, 1002, 1003]
    assert protocol["rollout_seeds"] == [2001, 2002, 2003, 2004, 2005]


def test_databricks_job_invokes_the_native_wheel_entrypoint() -> None:
    """The bundle may wrap `scenarioflow reproduce`, but must not fork it."""
    root = yaml.safe_load((ROOT / "databricks.yml").read_text())
    job_document = yaml.safe_load(
        (ROOT / "deploy" / "databricks" / "resources.yml").read_text()
    )
    assert root["include"] == ["deploy/databricks/*.yml"]
    assert root["artifacts"]["default"]["type"] == "whl"
    assert root["artifacts"]["default"]["path"] == "."

    task = job_document["resources"]["jobs"]["reproduce"]["tasks"][0]
    assert task["existing_cluster_id"] == "${var.cluster_id}"
    assert task["python_wheel_task"]["package_name"] == "scenarioflow"
    assert task["python_wheel_task"]["entry_point"] == "scenarioflow"
    assert task["python_wheel_task"]["parameters"] == [
        "reproduce",
        "--output-root",
        "${var.output_root}/runs",
        "--source-manifest",
        "/Workspace${workspace.file_path}/reproduction/FROZEN_SOURCE.tsv",
        "--artifact-manifest",
        "/Workspace${workspace.file_path}/reproduction/FROZEN_DATA.tsv",
        "--data-dir",
        "${var.data_dir}",
        "--checkpoint-root",
        "${var.checkpoint_root}",
        "--artifact-root",
        "${var.artifact_root}",
        "--output",
        "${var.output_root}/summary.json",
    ]
    assert task["libraries"] == [{"whl": "./dist/*.whl"}]
