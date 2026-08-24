"""Lazy local launchers for the two optional ScenarioFlow demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioflow.artifacts import ArtifactResolver


def _build_narrative_to_scenario(
    *, manifest: Path, artifact_root: Path, data_dir: Path
) -> Any:
    del data_dir
    from .narrative_to_scenario_app import (
        NarrativeToScenarioRuntime,
        NarrativeToScenarioService,
        build_narrative_to_scenario_app,
    )

    resolver = ArtifactResolver.from_tsv(manifest)
    runtime = NarrativeToScenarioRuntime.from_artifacts(
        resolver=resolver, artifact_root=artifact_root
    )
    return build_narrative_to_scenario_app(NarrativeToScenarioService(runtime))


def _build_scenario_to_narrative(
    *, manifest: Path, artifact_root: Path, data_dir: Path
) -> Any:
    from .scenario_to_narrative_app import (
        ScenarioToNarrativeRuntime,
        ScenarioToNarrativeService,
        build_scenario_to_narrative_app,
    )

    resolver = ArtifactResolver.from_tsv(manifest)
    runtime = ScenarioToNarrativeRuntime.from_artifacts(
        resolver=resolver,
        artifact_root=artifact_root,
        data_path=data_dir / "multi_factor_data.npz",
    )
    return build_scenario_to_narrative_app(ScenarioToNarrativeService(runtime))


def launch_demo(
    *,
    demo: str,
    manifest: str | Path,
    artifact_root: str | Path,
    data_dir: str | Path,
    host: str,
    port: int,
) -> int:
    """Build one backend lazily and serve it without creating a public tunnel."""
    builders = {
        "narrative-to-scenario": _build_narrative_to_scenario,
        "scenario-to-narrative": _build_scenario_to_narrative,
    }
    try:
        builder = builders[demo]
    except KeyError as error:
        raise ValueError(f"unknown demo: {demo}") from error
    app = builder(
        manifest=Path(manifest),
        artifact_root=Path(artifact_root),
        data_dir=Path(data_dir),
    )
    app.launch(
        server_name=host,
        server_port=int(port),
        share=False,
        show_error=False,
    )
    return 0
