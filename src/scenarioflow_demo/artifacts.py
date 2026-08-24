"""Resolve demo runtime files through the public artifact manifest."""

from __future__ import annotations

from pathlib import Path

from scenarioflow.artifacts import ArtifactResolver


def resolve_nl_runtime_artifacts(
    resolver: ArtifactResolver,
    artifact_root: str | Path,
) -> dict[str, Path]:
    """Resolve every artifact in the manifest's ``nl-runtime`` profile."""
    return {
        record.artifact_id: resolver.resolve(record.artifact_id, artifact_root)
        for record in resolver.for_profile("nl-runtime")
    }
