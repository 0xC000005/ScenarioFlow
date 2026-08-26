"""Behavioral contract for the separate Databricks numerical image."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy/databricks/run_job.py"


def test_databricks_image_is_separate_and_numerical_only() -> None:
    dockerfile = (ROOT / "deploy/databricks/Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM databricksruntime/standard:18.3-LTS")
    assert "torch==2.4.1" in dockerfile
    assert "--torch-backend cu121" in dockerfile
    for required in (
        "COPY data ./data",
        "COPY reproduction ./reproduction",
        "COPY results/frozen ./results/frozen",
        "COPY scripts ./scripts",
    ):
        assert required in dockerfile
    for forbidden in ("codex", "node:", "nginx", "demo-openai", "web/"):
        assert forbidden not in dockerfile.lower()


def test_bundle_runs_the_launcher_in_the_custom_gpu_image() -> None:
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    resources = yaml.safe_load(
        (ROOT / "deploy/databricks/resources.yml").read_text(encoding="utf-8")
    )

    assert bundle["include"] == ["deploy/databricks/resources.yml"]
    variables = bundle["variables"]
    assert "image_url" in variables and "default" not in variables["image_url"]
    assert "gpu_node_type" in variables and "default" not in variables["gpu_node_type"]

    job = resources["resources"]["jobs"]["scenarioflow"]
    task = job["tasks"][0]
    cluster = job["job_clusters"][0]["new_cluster"]
    assert task["spark_python_task"]["python_file"] == "./run_job.py"
    assert task["spark_python_task"]["parameters"] == [
        "--operation",
        "{{job.parameters.operation}}",
        "--seed",
        "{{job.parameters.seed}}",
        "--artifact-root",
        "${var.artifact_root}",
        "--output-root",
        "${var.output_root}",
    ]
    assert cluster["docker_image"]["url"] == "${var.image_url}"
    assert cluster["node_type_id"] == "${var.gpu_node_type}"
    assert cluster["spark_version"] == "${var.spark_version}"
    assert cluster["data_security_mode"] == "DATA_SECURITY_MODE_DEDICATED"
    assert cluster["spark_conf"]["spark.databricks.unityCatalog.volumes.enabled"] == "true"


def _write_command(path: Path, label: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' '{label}:'\"$*\" >> \"$SCENARIOFLOW_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_launcher_forwards_seed_to_each_training_stage_and_redirects_runtime_state(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    scripts = release_root / "scripts"
    scripts.mkdir(parents=True)
    (release_root / "artifacts").mkdir()
    for filename, label in (
        ("train_next_step.sh", "stage1"),
        ("finetune_rollouts.sh", "stage2"),
        ("run_inference.sh", "inference"),
        ("reproduce.sh", "reproduce"),
    ):
        _write_command(scripts / filename, label)

    calls = tmp_path / "calls.txt"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--operation",
            "train",
            "--seed",
            "1002",
            "--artifact-root",
            str(artifact_root),
            "--output-root",
            str(output_root),
        ],
        env={
            **os.environ,
            "SCENARIOFLOW_ROOT": str(release_root),
            "SCENARIOFLOW_TEST_CALLS": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "stage1:--seed 1002",
        "stage2:--seed 1002",
    ]
    assert (release_root / "build").resolve() == output_root.resolve()
    assert (release_root / "artifacts/cache").resolve() == artifact_root.resolve()
    summary = json.loads(result.stdout)
    assert summary["operation"] == "train"
    assert summary["commands"] == [
        [str(scripts / "train_next_step.sh"), "--seed", "1002"],
        [str(scripts / "finetune_rollouts.sh"), "--seed", "1002"],
    ]


def test_launcher_forwards_seed_to_inference(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    scripts = release_root / "scripts"
    scripts.mkdir(parents=True)
    (release_root / "artifacts").mkdir()
    _write_command(scripts / "run_inference.sh", "inference")

    calls = tmp_path / "calls.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--operation",
            "inference",
            "--seed",
            "1003",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        env={
            **os.environ,
            "SCENARIOFLOW_ROOT": str(release_root),
            "SCENARIOFLOW_TEST_CALLS": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["inference:--seed 1003"]
    assert json.loads(result.stdout)["commands"] == [
        [str(scripts / "run_inference.sh"), "--seed", "1003"]
    ]


def test_launcher_runs_the_existing_root_reproduction_launcher(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    scripts = release_root / "scripts"
    scripts.mkdir(parents=True)
    (release_root / "artifacts").mkdir()
    _write_command(scripts / "reproduce.sh", "reproduce")

    calls = tmp_path / "calls.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--operation",
            "reproduce",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        env={
            **os.environ,
            "SCENARIOFLOW_ROOT": str(release_root),
            "SCENARIOFLOW_TEST_CALLS": str(calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["reproduce:"]
    assert json.loads(result.stdout)["commands"] == [
        [str(scripts / "reproduce.sh")]
    ]


def test_databricks_guide_has_copyable_build_deploy_and_run_commands() -> None:
    guide = (ROOT / "deploy/databricks/README.md").read_text(encoding="utf-8")

    for command in (
        "docker build -f deploy/databricks/Dockerfile",
        "docker push \"$IMAGE_URL\"",
        "databricks bundle validate",
        "databricks bundle deploy",
        "databricks bundle run scenarioflow --params operation=inference,seed=2001",
        "databricks bundle run scenarioflow --params operation=train,seed=1001",
        "databricks bundle run scenarioflow --params operation=reproduce",
    ):
        assert command in guide
    assert "separate numerical image" in guide
    assert "OpenAI" not in guide
    assert "Codex" not in guide
