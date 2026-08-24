"""Public, cached contracts shared by the two optional demo backends."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scenarioflow.artifacts import ArtifactError, ArtifactResolver

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "demo"


def _load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_demo_package_import_is_lightweight() -> None:
    """The optional package is inspectable without server or model dependencies."""
    script = """
import sys
import scenarioflow_demo
blocked = {'gradio', 'openai', 'plotly', 'torch'} & set(sys.modules)
assert not blocked, blocked
assert scenarioflow_demo.__version__ == '0.1.0'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cached_narrative_to_scenario_contract() -> None:
    from scenarioflow_demo.contracts import NarrativeScenarioPayload

    fixture = _load("narrative_to_scenario.json")
    payload = NarrativeScenarioPayload.from_report(
        window_index=fixture["window_index"],
        context=fixture["context"],
        report=fixture["report"],
    )
    assert payload.window_index == 120
    assert payload.context["SPX"][-1] == 2018.0
    assert payload.factors["SPX"].conditional.p50 == (2010.0, 2000.0)
    assert payload.factors["SPX"].start_only.mean == (2016.0, 2013.0)
    assert payload.precedents[0].calendar_date == "2005-09-09"
    assert payload.precedents[0].similarity == pytest.approx(0.84)


def test_cached_scenario_to_narrative_contract() -> None:
    from scenarioflow_demo.contracts import NarrativePacket, ScenarioSidecar

    fixture = _load("scenario_to_narrative.json")
    sidecar = ScenarioSidecar.from_mapping(fixture["sidecar"])
    packet = NarrativePacket.from_mapping(fixture["packet"])
    assert sidecar.horizon_days == 30
    assert sidecar.factor_rows[0].direction == "down"
    assert packet.scenario_sidecar == sidecar
    assert packet.positive_narratives[0]["view"] == "risk_manager"
    assert packet.validation == {"status": "passed", "error_count": 0}


def test_contract_rejects_invalid_values_near_the_field() -> None:
    from scenarioflow_demo.contracts import FanSeries, ScenarioSidecar

    with pytest.raises(ValueError, match="fan series p50"):
        FanSeries.from_mapping({"p10": [], "p50": ["not-a-number"]})
    with pytest.raises(ValueError, match="horizon_days"):
        ScenarioSidecar.from_mapping(
            {
                "scenario_id": "bad",
                "scenario_type": "historical_joint39",
                "horizon_days": 0,
                "mechanical_summary": "bad",
            }
        )


def test_runtime_artifacts_are_resolved_only_by_manifest_ids(tmp_path: Path) -> None:
    from scenarioflow_demo.artifacts import resolve_nl_runtime_artifacts

    resolver = ArtifactResolver.from_tsv(ROOT / "artifacts" / "manifest-v1.tsv")
    calls: list[tuple[str, Path]] = []

    def resolve(artifact_id: str, artifact_root: str | Path) -> Path:
        calls.append((artifact_id, Path(artifact_root)))
        return Path(artifact_root) / f"resolved-{len(calls)}"

    resolver.resolve = resolve  # type: ignore[method-assign]
    paths = resolve_nl_runtime_artifacts(resolver, tmp_path)
    expected = tuple(
        record.artifact_id for record in resolver.for_profile("nl-runtime")
    )
    assert tuple(paths) == expected
    assert tuple(artifact_id for artifact_id, _root in calls) == expected
    assert all(root == tmp_path for _artifact_id, root in calls)


def test_runtime_resolution_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    from scenarioflow_demo.artifacts import resolve_nl_runtime_artifacts

    resolver = ArtifactResolver.from_tsv(ROOT / "artifacts" / "manifest-v1.tsv")
    with pytest.raises(ArtifactError, match="nl.episode-cards"):
        resolve_nl_runtime_artifacts(resolver, tmp_path)


def test_demo_contract_module_has_no_legacy_runtime_imports() -> None:
    module = importlib.import_module("scenarioflow_demo.contracts")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "experiments.backfill" not in source
    assert "nl_risk_manager_story_gradio_app" not in source
