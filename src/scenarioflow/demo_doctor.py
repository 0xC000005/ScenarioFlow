"""Non-invasive capability checks for the two optional ScenarioFlow demos."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .artifacts import ArtifactResolver

NARRATIVE_TO_SCENARIO_ARTIFACTS = (
    "nl.bridge-adapter",
    "nl.full-start-bridge-arrays",
    "nl.full-start-bridge-report",
    "nl.support-bank",
    "nl.support-bank-report",
)


def _module_available(find_spec: Callable[[str], object | None], name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _artifacts_ready(
    resolver: ArtifactResolver,
    artifact_root: str | Path,
    artifact_ids: tuple[str, ...],
) -> bool:
    return all(
        status.state == "present"
        for status in resolver.verify(artifact_ids, artifact_root)
    )


def inspect_demo_capabilities(
    *,
    resolver: ArtifactResolver,
    artifact_root: str | Path,
    environ: Mapping[str, str] | None = None,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Report local demo readiness without making a model request or starting a server."""
    environment = os.environ if environ is None else environ
    nl_runtime_ids = tuple(
        record.artifact_id for record in resolver.for_profile("nl-runtime")
    )
    narrative_runtime_ready = _artifacts_ready(
        resolver,
        artifact_root,
        NARRATIVE_TO_SCENARIO_ARTIFACTS,
    )
    stage2_ready = _artifacts_ready(
        resolver,
        artifact_root,
        ("core.stage2",),
    )
    narrative_backend_ready = _module_available(
        find_spec, "scenarioflow_demo.narrative_to_scenario_app"
    )
    scenario_backend_ready = _module_available(
        find_spec, "scenarioflow_demo.scenario_to_narrative_app"
    )
    server_ready = _module_available(find_spec, "gradio")
    openai_ready = _module_available(find_spec, "openai")

    demo_one_missing: list[str] = []
    demo_one_actions: list[str] = []
    if not environment.get("OPENAI_API_KEY", "").strip():
        demo_one_missing.append("OPENAI_API_KEY")
        demo_one_actions.append("Set `OPENAI_API_KEY`.")
    if not openai_ready:
        demo_one_missing.append("demo-openai extra")
        demo_one_actions.append(
            "Run `uv sync --extra demo-openai --extra demo-server`."
        )
    if not server_ready:
        demo_one_missing.append("demo-server extra")
        if openai_ready:
            demo_one_actions.append("Run `uv sync --extra demo-server`.")
    if not narrative_runtime_ready:
        demo_one_missing.append("nl-runtime artifacts")
        demo_one_actions.append(
            "Run `./scripts/setup.sh --profile nl-runtime`."
        )
    if not stage2_ready:
        demo_one_missing.append("Stage 2 checkpoint")
        demo_one_actions.append(
            "Run `./scripts/setup.sh`."
        )
    if not narrative_backend_ready:
        demo_one_missing.append("demo backend")
        demo_one_actions.append("Run `uv sync --extra demo-server`.")

    demo_two_missing: list[str] = []
    demo_two_actions: list[str] = []
    if not server_ready:
        demo_two_missing.append("demo-server extra")
        demo_two_actions.append("Run `uv sync --extra demo-server`.")
    if not _artifacts_ready(resolver, artifact_root, nl_runtime_ids):
        demo_two_missing.append("nl-runtime artifacts")
        demo_two_actions.append(
            "Run `./scripts/setup.sh --profile nl-runtime`."
        )
    codex = which("codex")
    if codex is None:
        demo_two_missing.append("Codex CLI")
        demo_two_actions.append("Install the Codex CLI.")
    else:
        try:
            status = run(
                [codex, "login", "status"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            logged_in = status.returncode == 0
        except (OSError, subprocess.SubprocessError):
            logged_in = False
        if not logged_in:
            demo_two_missing.append("Codex CLI login")
            demo_two_actions.append("Run `codex login`.")
    if not scenario_backend_ready:
        demo_two_missing.append("demo backend")
        demo_two_actions.append("Run `uv sync --extra demo-server`.")

    return {
        "numerical_workflows": "available",
        "demos": {
            "narrative-to-scenario": {
                "available": not demo_one_missing,
                "missing": demo_one_missing,
                "actions": demo_one_actions,
            },
            "scenario-to-narrative": {
                "available": not demo_two_missing,
                "missing": demo_two_missing,
                "actions": demo_two_actions,
            },
        },
    }
