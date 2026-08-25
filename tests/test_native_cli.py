"""Tests for the native ScenarioFlow command dispatcher."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

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
                "url_path": "core.fixture.pt",
            }
        )
    return path


@pytest.mark.parametrize(
    ("arguments", "needle"),
    [
        ((), "setup"),
        (("fetch", "--help"), "--bundle-dir"),
        (("artifacts", "verify", "--help"), "--artifact-root"),
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


def test_setup_check_validates_packaged_data_and_manifest() -> None:
    """A native installation with changed data or manifest must not report ready."""
    result = _run("setup", "--check", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["package"] == "ok"
    assert payload["processed_data"] == "ok"
    assert payload["artifact_manifest"] == "ok"
    external = payload["external_artifacts"]
    assert external["total"] == 43
    assert 0 <= external["present"] <= external["total"]


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


def test_demo_failure_does_not_break_setup_and_fetch_help() -> None:
    """Missing optional demo capabilities must leave release commands usable."""
    doctor = _run("demo", "doctor", "--json")
    assert doctor.returncode == 0
    assert _run("setup", "--help").returncode == 0
    assert _run("fetch", "--help").returncode == 0


def test_demo_launch_forwards_reader_facing_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The native command must launch the selected released backend."""
    calls: list[dict[str, object]] = []
    report = {
        "numerical_workflows": "available",
        "demos": {
            "narrative-to-scenario": {
                "available": True,
                "missing": [],
                "actions": [],
            },
            "scenario-to-narrative": {
                "available": True,
                "missing": [],
                "actions": [],
            },
        },
    }
    monkeypatch.setattr(cli, "inspect_demo_capabilities", lambda **_kwargs: report)
    module = types.ModuleType("scenarioflow_demo.launch")
    module.launch_demo = lambda **kwargs: calls.append(kwargs) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scenarioflow_demo.launch", module)
    manifest = _fixture_manifest(tmp_path / "manifest.tsv")

    result = cli.main(
        [
            "demo",
            "launch",
            "narrative-to-scenario",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(tmp_path / "cache"),
            "--data-dir",
            str(tmp_path / "data"),
            "--host",
            "127.0.0.1",
            "--port",
            "9010",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "demo": "narrative-to-scenario",
            "manifest": manifest,
            "artifact_root": tmp_path / "cache",
            "data_dir": tmp_path / "data",
            "host": "127.0.0.1",
            "port": 9010,
        }
    ]


def test_demo_launch_stops_with_exact_doctor_actions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unavailable demo should fail before importing its runtime."""
    report = {
        "numerical_workflows": "available",
        "demos": {
            "narrative-to-scenario": {
                "available": False,
                "missing": ["OPENAI_API_KEY"],
                "actions": ["Set OPENAI_API_KEY in your environment."],
            },
            "scenario-to-narrative": {
                "available": True,
                "missing": [],
                "actions": [],
            },
        },
    }
    monkeypatch.setattr(cli, "inspect_demo_capabilities", lambda **_kwargs: report)

    assert cli.main(["demo", "launch", "narrative-to-scenario"]) == 2
    captured = capsys.readouterr()
    assert captured.err == "Set OPENAI_API_KEY in your environment.\n"
    assert "Task 6" not in captured.err
