"""Tests for the native ScenarioFlow command dispatcher."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scenarioflow import cli


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scenarioflow.cli", *arguments],
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "SCENARIOFLOW_ARTIFACT_BASE_URL"
        },
        text=True,
        capture_output=True,
        check=False,
    )


def _fixture_manifest(path: Path, payload: bytes = b"fixture") -> Path:
    relative = "models/core/fixture/best_model.pt"
    fields = (
        "schema_version",
        "artifact_id",
        "profile",
        "category",
        "relative_path",
        "bytes",
        "sha256",
        "frozen_source_path",
        "url_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": "scenarioflow.artifacts.v1",
                "artifact_id": "core.fixture",
                "profile": "core,all",
                "category": "checkpoint",
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "frozen_source_path": "frozen/fixture.bin",
                "url_path": relative,
            }
        )
    return path


@pytest.mark.parametrize(
    ("arguments", "needle"),
    [
        ((), "setup"),
        (("fetch", "--help"), "--bundle-dir"),
        (("artifacts", "verify", "--help"), "--artifact-root"),
        (("train", "--help"), "--stage"),
        (("infer", "--help"), "--checkpoint"),
        (("reproduce", "--help"), "--output-root"),
        (("demo", "doctor", "--help"), "--json"),
        (("demo", "launch", "--help"), "narrative-to-scenario"),
    ],
)
def test_native_help_surfaces_are_available(
    arguments: tuple[str, ...], needle: str
) -> None:
    """Removing a documented native command or forwarding parser must fail."""
    result = _run(*(arguments or ("--help",)))
    assert result.returncode == 0, result.stderr
    assert needle in result.stdout


def test_train_infer_and_reproduce_forward_argument_tails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The umbrella CLI must not reinterpret established numerical arguments."""
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "scenarioflow.training.main", lambda args: calls.append(("train", args))
    )
    monkeypatch.setattr(
        "scenarioflow.inference.main", lambda args: calls.append(("infer", args))
    )
    monkeypatch.setattr(
        "scenarioflow.workflows.reproduction.main",
        lambda args: calls.append(("reproduce", args)),
    )
    cli.main(["train", "--stage", "stage1", "--steps", "2"])
    cli.main(["infer", "--checkpoint", "model.pt", "--samples", "3"])
    cli.main(["reproduce", "--output-root", "out", "--smoke"])
    assert calls == [
        ("train", ["--stage", "stage1", "--steps", "2"]),
        ("infer", ["--checkpoint", "model.pt", "--samples", "3"]),
        ("reproduce", ["reproduce", "--output-root", "out", "--smoke"]),
    ]


def test_setup_check_validates_packaged_data_and_manifest() -> None:
    """A native installation with changed data or manifest must not report ready."""
    result = _run("setup", "--check", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["package"] == "ok"
    assert payload["processed_data"] == "ok"
    assert payload["artifact_manifest"] == "ok"
    assert payload["external_artifacts"] == {"present": 0, "total": 41}


def test_setup_check_reports_missing_data_without_a_traceback(tmp_path: Path) -> None:
    """A clean but incomplete checkout should receive one readable validation result."""
    result = _run(
        "setup",
        "--check",
        "--data-dir",
        str(tmp_path / "missing-data"),
        "--json",
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["processed_data"] == "invalid"
    assert "Traceback" not in result.stderr


def test_fetch_without_source_explains_release_url_or_bundle(tmp_path: Path) -> None:
    """A missing artifact must not trigger an invented URL or vague download failure."""
    manifest = _fixture_manifest(tmp_path / "manifest.tsv")
    result = _run(
        "fetch",
        "--manifest",
        str(manifest),
        "--artifact-root",
        str(tmp_path / "cache"),
        "--profile",
        "core",
    )
    assert result.returncode != 0
    assert "--base-url" in result.stderr
    assert "--bundle-dir" in result.stderr


def test_artifacts_verify_json_reports_safe_relative_status(tmp_path: Path) -> None:
    """Integrity output must use the manifest path and never leak the host root."""
    payload = b"fixture"
    manifest = _fixture_manifest(tmp_path / "manifest.tsv", payload)
    cache = tmp_path / "cache"
    final = cache / "models/core/fixture/best_model.pt"
    final.parent.mkdir(parents=True)
    final.write_bytes(payload)
    result = _run(
        "artifacts",
        "verify",
        "--manifest",
        str(manifest),
        "--artifact-root",
        str(cache),
        "--profile",
        "core",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["artifacts"] == [
        {
            "artifact_id": "core.fixture",
            "bytes": len(payload),
            "relative_path": "models/core/fixture/best_model.pt",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "state": "present",
        }
    ]
    assert str(tmp_path) not in result.stdout


def test_demo_failure_does_not_break_numerical_help() -> None:
    """Missing optional demo capabilities must leave core numerical commands usable."""
    doctor = _run("demo", "doctor", "--json")
    assert doctor.returncode == 0
    assert _run("train", "--help").returncode == 0
    assert _run("infer", "--help").returncode == 0
