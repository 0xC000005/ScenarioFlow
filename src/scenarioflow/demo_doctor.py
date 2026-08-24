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
    runtime_ready = all(
        status.state == "present"
        for status in resolver.verify(
            [record.artifact_id for record in resolver.for_profile("nl-runtime")],
            artifact_root,
        )
    )
    backend_ready = find_spec("scenarioflow_demo") is not None

    demo_one_missing: list[str] = []
    demo_one_actions: list[str] = []
    if not environment.get("OPENAI_API_KEY", "").strip():
        demo_one_missing.append("OPENAI_API_KEY")
        demo_one_actions.append("Set `OPENAI_API_KEY`.")
    if find_spec("openai") is None:
        demo_one_missing.append("demo-openai extra")
        demo_one_actions.append("Install the `demo-openai` extra.")
    if not runtime_ready:
        demo_one_missing.append("nl-runtime artifacts")
        demo_one_actions.append("Fetch the `nl-runtime` artifact profile.")
    if not backend_ready:
        demo_one_missing.append("demo backend")
        demo_one_actions.append(
            "Install the optional demo package when Task 6 is available."
        )

    demo_two_missing: list[str] = []
    demo_two_actions: list[str] = []
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
    if not backend_ready:
        demo_two_missing.append("demo backend")
        demo_two_actions.append(
            "Install the optional demo package when Task 6 is available."
        )

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
