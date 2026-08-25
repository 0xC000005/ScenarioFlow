"""Behavioral contract for the Docker and Compose release."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMAGE = "scenarioflow-narrative-to-scenario"
FRONTEND_IMAGE = "scenarioflow-frontend"


def _compose_config() -> dict[str, object]:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    environment = {
        **os.environ,
        "OPENAI_API_KEY": "",
        "SCENARIOFLOW_CODEX_HOME": str(ROOT / ".codex-unavailable"),
    }
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _volume_by_target(service: dict[str, object], target: str) -> dict[str, object]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    return next(volume for volume in volumes if volume.get("target") == target)


def test_compose_uses_native_demo_commands_and_separate_build_targets() -> None:
    """A container wrapper must not fork the native CLI or merge the two backends."""
    document = _compose_config()
    services = document["services"]
    assert isinstance(services, dict)
    assert set(services) == {
        "frontend",
        "narrative-to-scenario",
        "scenario-to-narrative",
    }
    first = services["narrative-to-scenario"]
    second = services["scenario-to-narrative"]
    frontend = services["frontend"]
    assert first["build"]["target"] == "runtime"
    assert second["build"]["target"] == "runtime"
    assert frontend["build"]["target"] == "web"
    assert first["command"] == [
        "scenarioflow",
        "demo",
        "launch",
        "narrative-to-scenario",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
    ]
    assert second["command"] == [
        "scenarioflow",
        "demo",
        "launch",
        "scenario-to-narrative",
        "--host",
        "0.0.0.0",
        "--port",
        "7861",
    ]


def test_compose_mounts_runtime_inputs_without_writable_data_or_artifacts() -> None:
    """Processed data and fetched model files must stay outside image layers."""
    services = _compose_config()["services"]
    for name in ("narrative-to-scenario", "scenario-to-narrative"):
        service = services[name]
        data = _volume_by_target(service, "/app/data")
        artifacts = _volume_by_target(service, "/app/artifacts/cache")
        assert data["type"] == "bind" and data["read_only"] is True
        assert artifacts["type"] == "bind" and artifacts["read_only"] is True
    first_environment = services["narrative-to-scenario"]["environment"]
    assert first_environment["OPENAI_API_KEY"] == ""
    codex = _volume_by_target(
        services["scenario-to-narrative"], "/run/scenarioflow/codex"
    )
    assert codex["type"] == "bind" and codex["read_only"] is True


def _require_built_runtime() -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", RUNTIME_IMAGE],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("build the Docker runtime image to run container integration")


def test_built_runtime_executes_the_native_cli() -> None:
    """Relocating the virtual environment must not break its console entry point."""
    _require_built_runtime()
    result = subprocess.run(
        ["docker", "run", "--rm", RUNTIME_IMAGE, "scenarioflow", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "setup" in result.stdout
    assert "fetch" in result.stdout
    assert "demo" in result.stdout


def test_built_runtime_contains_the_codex_platform_binary() -> None:
    """The scenario-to-narrative image must include Codex, not only its JS shim."""
    _require_built_runtime()
    result = subprocess.run(
        ["docker", "run", "--rm", RUNTIME_IMAGE, "codex", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("codex-cli ")


def test_built_frontend_can_start_without_either_optional_backend() -> None:
    """The static landing page must not require both demo servers at startup."""
    result = subprocess.run(
        ["docker", "image", "inspect", FRONTEND_IMAGE],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("build the Docker frontend image to run container integration")
    check = subprocess.run(
        ["docker", "run", "--rm", FRONTEND_IMAGE, "nginx", "-t"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
