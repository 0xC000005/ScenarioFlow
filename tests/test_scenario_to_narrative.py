"""Deterministic and mocked tests for the scenario-to-narrative demo."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scenarioflow_demo.codex_author import CodexAuthorError, author_narrative_packet
from scenarioflow_demo.contracts import FactorRow, ScenarioSidecar
from scenarioflow_demo.scenario_to_narrative import (
    HISTORICAL_FACTORS,
    build_mechanical_summary,
    factor_row_from_path,
    historical_start_dates,
    load_historical_scenario,
    scenario_movement,
    select_negative_candidates,
)
from scenarioflow_demo.scenario_to_narrative_app import (
    ScenarioToNarrativeRuntime,
    ScenarioToNarrativeService,
    build_scenario_to_narrative_app,
    render_packet_markdown,
)


def _factor(name: str = "SPX", start: float = 100.0, end: float = 90.0) -> FactorRow:
    return factor_row_from_path(name, [start, end])


def _sidecar(*rows: FactorRow) -> ScenarioSidecar:
    factor_rows = tuple(rows or (_factor(),))
    return ScenarioSidecar(
        scenario_id="window-7",
        scenario_type="historical_joint39",
        scenario_title="Risk-off month",
        archetype="risk_off",
        horizon_days=30,
        factor_rows=factor_rows,
        mechanical_summary=build_mechanical_summary(factor_rows),
        normalization_warnings=(),
        summary_source="historical_episode_card",
        sample_count=None,
        source_artifacts={"cards_jsonl": "nl.episode-cards"},
    )


def _historical_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        json.dumps(
            {
                "window_id": "window-7",
                "scenario_title": "Risk-off month",
                "archetype": "risk_off",
                "support_metadata": {"window_index": 1},
                "caption_fields": {"evidence_used": ["SPX down; BBB_OAS wider"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "support-report.json"
    report.write_text(
        json.dumps(
            {
                "window_metadata": [
                    {
                        "window_id": "window-6",
                        "calendar_start_date": "2016-03-21",
                    },
                    {
                        "window_id": "window-7",
                        "calendar_start_date": "2016-03-22",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    history = np.zeros((2, 4, 39), dtype=np.float32)
    for offset, (_factor_name, column) in enumerate(HISTORICAL_FACTORS):
        history[1, :, column] = np.arange(4, dtype=np.float32) + 100 + offset
    arrays = tmp_path / "support-arrays.npz"
    np.savez(arrays, history_raw=history)
    return cards, report, arrays


def test_factor_normalization_preserves_spread_language_and_path() -> None:
    spread = factor_row_from_path("BBB_OAS", [4.1, 4.3, 4.5])
    equity = factor_row_from_path("SPX", [2000.0, 1980.0])
    assert spread.direction == "wider"
    assert spread.delta == pytest.approx(0.4)
    assert spread.path_values == (4.1, 4.3, 4.5)
    assert equity.direction == "down"
    assert build_mechanical_summary((spread, equity)) == (
        "Mechanical baseline: BBB_OAS wider small; SPX down large"
    )


def test_historical_date_selection_and_loading_are_deterministic(
    tmp_path: Path,
) -> None:
    cards, report, arrays = _historical_fixture(tmp_path)
    assert historical_start_dates(report, cards) == ("2016-03-22",)
    first = load_historical_scenario(
        "2016-03-22",
        support_report_path=report,
        cards_path=cards,
        support_arrays_path=arrays,
    )
    second = load_historical_scenario(
        "2016-03-22",
        support_report_path=report,
        cards_path=cards,
        support_arrays_path=arrays,
    )
    assert first == second
    assert first.scenario_id == "window-7"
    assert tuple(row.factor for row in first.factor_rows) == tuple(
        factor for factor, _column in HISTORICAL_FACTORS
    )
    assert first.source_artifacts == {
        "cards_jsonl": "nl.episode-cards",
        "support_arrays": "nl.support-bank",
        "support_report": "nl.support-bank-report",
    }


def test_negative_candidates_must_contradict_the_scenario() -> None:
    sidecar = _sidecar(_factor("SPX", 100, 90), _factor("BBB_OAS", 4.0, 4.5))
    cards = [
        {
            "window_id": "opposite",
            "scenario_title": "Relief rally",
            "mechanical_summary": "SPX up; BBB_OAS tighter",
            "caption_fields": {"evidence_used": ["SPX up", "BBB_OAS tighter"]},
        },
        {
            "window_id": "same",
            "scenario_title": "Risk off",
            "mechanical_summary": "SPX down; BBB_OAS wider",
        },
    ]
    selected = select_negative_candidates(sidecar, cards, count=1)
    assert selected[0]["window_id"] == "opposite"
    assert selected[0]["contradiction_channels"] == ["SPX", "BBB_OAS"]


def test_mocked_codex_call_returns_validated_packet(tmp_path: Path) -> None:
    sidecar = _sidecar()
    candidates = [
        {
            "window_id": "opposite",
            "scenario_title": "Relief rally",
            "mechanical_summary": "SPX up",
            "evidence_used": ["SPX up"],
            "contradiction_channels": ["SPX"],
        }
    ]
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["input"] = kwargs["input"]
        output_path = Path(command[command.index("-o") + 1])
        from scenarioflow_demo.codex_author import NARRATIVE_VIEWS

        output_path.write_text(
            json.dumps(
                {
                    "target_window_id": "window-7",
                    "target_title": "Risk-off month",
                    "pairs": [
                        {
                            "view_name": view,
                            "positive_text": f"Positive reading for {view}.",
                            "negative_window_id": "opposite",
                            "negative_text": f"Contrasting reading for {view}.",
                            "quality_notes": ["Grounded in supplied factor moves."],
                        }
                        for view in NARRATIVE_VIEWS
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "private event stream", "")

    packet = author_narrative_packet(
        sidecar,
        candidates,
        which=lambda name: "/usr/bin/codex" if name == "codex" else None,
        run=run,
        temporary_root=tmp_path,
    )
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:2] == ["/usr/bin/codex", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert "xhigh" in command[command.index("--config") + 1]
    assert "private event stream" not in str(packet)
    assert len(packet.positive_narratives) == 14
    assert packet.validation == {"status": "passed", "error_count": 0}


def test_codex_failure_does_not_expose_process_output(tmp_path: Path) -> None:
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, "private stdout", "private account and path"
        )

    with pytest.raises(CodexAuthorError) as caught:
        author_narrative_packet(
            _sidecar(),
            [{"window_id": "opposite"}],
            which=lambda _name: "/usr/bin/codex",
            run=run,
            temporary_root=tmp_path,
        )
    assert "private stdout" not in str(caught.value)
    assert "private account" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)


def test_scenario_movement_reads_the_selected_30_day_period(tmp_path: Path) -> None:
    dates = np.arange(np.datetime64("2016-03-22"), np.datetime64("2016-05-05"))
    levels = np.column_stack(
        [np.arange(len(dates), dtype=np.float32), np.arange(len(dates)) + 100]
    )
    data_path = tmp_path / "panel.npz"
    np.savez(
        data_path,
        dates=dates,
        levels=levels,
        level_columns=np.asarray(["SPX", "VIX"]),
    )
    movement = scenario_movement("2016-03-22", data_path)
    assert tuple(movement) == ("SPX", "VIX")
    assert len(movement["SPX"]) == 30
    assert movement["VIX"][-1] == 129.0


def test_service_uses_fixed_runtime_paths_and_renders_frontend_packet(
    tmp_path: Path,
) -> None:
    cards, report, arrays = _historical_fixture(tmp_path)
    data_path = tmp_path / "panel.npz"
    dates = np.arange(np.datetime64("2016-03-22"), np.datetime64("2016-05-05"))
    np.savez(
        data_path,
        dates=dates,
        levels=np.arange(len(dates), dtype=np.float32)[:, None],
        level_columns=np.asarray(["SPX"]),
    )
    observed: dict[str, object] = {}

    def author(sidecar: ScenarioSidecar, candidates: list[dict[str, object]]):
        observed["sidecar"] = sidecar
        observed["candidates"] = candidates
        pair = {
            "view_name": "risk_manager_memo",
            "positive_text": "Equities softened during the selected month.",
            "negative_window_id": "opposite",
            "negative_text": "Equities rallied during the contrasting month.",
        }
        from scenarioflow_demo.contracts import NarrativePacket

        return NarrativePacket(
            scenario_sidecar=sidecar,
            positive_narratives=(),
            hard_negative_narratives=(),
            paired_review=(pair,),
            validation={"status": "passed", "error_count": 0},
            artifact_paths={},
        )

    with cards.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "window_id": "opposite",
                    "scenario_title": "Equity decline",
                    "mechanical_summary": "SPX down",
                }
            )
            + "\n"
        )
    service = ScenarioToNarrativeService(
        ScenarioToNarrativeRuntime(data_path, cards, arrays, report),
        candidate_count=1,
        author=author,
    )
    sidecar = service.historical_scenario("2016-03-22")
    packet = service.generate(sidecar)
    markdown = render_packet_markdown(packet)
    assert "### 1. Risk Manager Memo" in markdown
    assert "**Positive**" in markdown
    assert "**Hard negative**" in markdown
    assert observed["sidecar"] == sidecar


def test_gradio_app_exposes_the_three_frozen_endpoint_names(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gradio")
    cards, report, arrays = _historical_fixture(tmp_path)
    data_path = tmp_path / "panel.npz"
    np.savez(
        data_path,
        dates=np.asarray([np.datetime64("2016-03-22"), np.datetime64("2016-03-23")]),
        levels=np.asarray([[1.0], [2.0]], dtype=np.float32),
        level_columns=np.asarray(["SPX"]),
    )
    app = build_scenario_to_narrative_app(
        ScenarioToNarrativeService(
            ScenarioToNarrativeRuntime(data_path, cards, arrays, report)
        )
    )
    names = {
        dependency["api_name"]
        for dependency in app.config["dependencies"]
        if dependency.get("api_name")
    }
    assert {
        "historical_scenario",
        "generate_narrative",
        "scenario_movement",
    }.issubset(names)
