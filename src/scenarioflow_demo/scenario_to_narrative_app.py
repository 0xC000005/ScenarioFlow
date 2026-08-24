"""Gradio API used by the reviewed scenario-to-narrative React page."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scenarioflow.artifacts import ArtifactResolver

from .artifacts import resolve_nl_runtime_artifacts
from .codex_author import CodexAuthorError, author_narrative_packet
from .contracts import FactorRow, NarrativePacket, ScenarioSidecar
from .scenario_to_narrative import (
    historical_start_dates,
    load_historical_scenario,
    load_negative_candidates,
    scenario_movement,
)

FACTOR_COLUMNS = (
    "Factor",
    "Start",
    "End",
    "Delta",
    "Direction",
    "Magnitude",
    "Confidence",
)
VIEW_TITLES = {
    "sparse_user_query": "Short user query",
    "weekly_risk_monitor": "Weekly risk monitor",
    "mechanism_first": "Mechanism-first read",
    "technical_factor_evidence": "Technical factor evidence",
    "factor_list_baseline": "Factor checklist",
    "institutional_risk_committee_note": "Risk committee note",
    "risk_manager_memo": "Risk Manager Memo",
    "full_professional": "Full professional brief",
    "sparse_variant_tape_read": "Tape read",
    "sparse_variant_portfolio_concern": "Portfolio concern",
    "sparse_variant_macro_channel": "Macro channel",
    "sparse_variant_credit_ambiguity": "Credit ambiguity",
    "sparse_variant_rates_commodities": "Rates and commodities",
    "sparse_variant_desk_note": "Desk note",
}


@dataclass(frozen=True, slots=True)
class ScenarioToNarrativeRuntime:
    data_path: Path
    cards_path: Path
    support_arrays_path: Path
    support_report_path: Path

    @classmethod
    def from_artifacts(
        cls,
        *,
        resolver: ArtifactResolver,
        artifact_root: str | Path,
        data_path: str | Path,
    ) -> "ScenarioToNarrativeRuntime":
        paths = resolve_nl_runtime_artifacts(resolver, artifact_root)
        return cls(
            data_path=Path(data_path),
            cards_path=paths["nl.episode-cards"],
            support_arrays_path=paths["nl.support-bank"],
            support_report_path=paths["nl.support-bank-report"],
        )


def _factor_is_rate_or_spread(factor: str) -> bool:
    name = factor.upper()
    return name in {"US2Y", "US10Y"} or name.endswith("_OAS")


def _factor_table(rows: Sequence[FactorRow]) -> list[list[str]]:
    table: list[list[str]] = []
    for row in rows:
        if _factor_is_rate_or_spread(row.factor):
            start = f"{row.start:.2f}%"
            end = f"{row.end:.2f}%"
            delta = f"{row.delta * 100:+,.0f} bp"
        else:
            decimals = 0 if max(abs(row.start), abs(row.end)) >= 1000 else 2
            start = f"{row.start:,.{decimals}f} pts"
            end = f"{row.end:,.{decimals}f} pts"
            delta = f"{row.delta:+,.{decimals}f} pts"
        table.append(
            [
                row.factor,
                start,
                end,
                delta,
                row.direction,
                row.magnitude,
                "Observed",
            ]
        )
    return table


def render_packet_markdown(packet: NarrativePacket) -> str:
    """Render the paired narratives in the structure consumed by the React page."""
    sections = [
        "# Generated Narratives",
        "Each market scenario below is described two ways. The **Positive** narrative "
        "is a faithful reading of the scenario. The **Hard negative** is a deliberately "
        "contradictory control, not another reading of the same scenario.",
    ]
    for index, pair in enumerate(packet.paired_review, start=1):
        view = str(pair.get("view_name", ""))
        title = VIEW_TITLES.get(view, view.replace("_", " ").title())
        sections.append(
            f"### {index}. {title}\n"
            f"**Positive**\n{str(pair.get('positive_text', '')).strip()}\n\n"
            f"**Hard negative**\n{str(pair.get('negative_text', '')).strip()}"
        )
    return "\n\n".join(sections)


class ScenarioToNarrativeService:
    """In-memory backend operations with no browser-supplied file paths."""

    def __init__(
        self,
        runtime: ScenarioToNarrativeRuntime,
        *,
        candidate_count: int = 40,
        author: Callable[..., NarrativePacket] = author_narrative_packet,
    ) -> None:
        self.runtime = runtime
        self.candidate_count = candidate_count
        self.author = author

    def dates(self) -> tuple[str, ...]:
        return historical_start_dates(
            self.runtime.support_report_path,
            self.runtime.cards_path,
        )

    def historical_scenario(self, start_date: str) -> ScenarioSidecar:
        return load_historical_scenario(
            start_date,
            support_report_path=self.runtime.support_report_path,
            cards_path=self.runtime.cards_path,
            support_arrays_path=self.runtime.support_arrays_path,
        )

    def movement(self, start_date: str) -> dict[str, tuple[float, ...]]:
        return scenario_movement(start_date, self.runtime.data_path)

    def generate(self, sidecar: ScenarioSidecar) -> NarrativePacket:
        candidates = load_negative_candidates(
            sidecar,
            self.runtime.cards_path,
            count=self.candidate_count,
        )
        return self.author(sidecar, candidates)


def build_scenario_to_narrative_app(service: ScenarioToNarrativeService):
    """Build the three Gradio endpoints consumed under the React ``/g2`` route."""
    import gradio as gr

    dates = service.dates()
    if not dates:
        raise ValueError("no historical demo dates are available")

    def historical_endpoint(start_date: str, _cards_id: str):
        sidecar = service.historical_scenario(start_date)
        sidecar_json = json.dumps(asdict(sidecar))
        return (
            "Scenario loaded.",
            _factor_table(sidecar.factor_rows),
            "[]",
            sidecar_json,
            sidecar_json,
        )

    def narrative_endpoint(
        sidecar_json: str,
        _cards_id: str,
        _output_directory: str,
    ) -> tuple[str, str]:
        try:
            sidecar = ScenarioSidecar.from_mapping(json.loads(sidecar_json))
            packet = service.generate(sidecar)
        except (CodexAuthorError, ValueError, json.JSONDecodeError):
            return "Narrative generation failed.", "Narrative generation failed."
        return "Narratives generated.", render_packet_markdown(packet)

    def movement_endpoint(start_date: str) -> str:
        return json.dumps(service.movement(start_date))

    with gr.Blocks(title="ScenarioFlow scenario to narrative") as app:
        start_date = gr.Dropdown(
            choices=[(date, date) for date in dates],
            value=dates[0],
            label="Select the historical period",
        )
        cards_id = gr.Textbox(
            value="nl.episode-cards", label="Reference cards", visible=False
        )
        output_directory = gr.Textbox(value="", label="Output directory", visible=False)
        sidecar_input = gr.Textbox(visible=False)
        status = gr.Textbox(visible=False)
        factor_table = gr.Dataframe(
            headers=list(FACTOR_COLUMNS), datatype="str", visible=False
        )
        warnings = gr.Textbox(visible=False)
        sidecar_json = gr.Textbox(visible=False)
        sidecar_state = gr.Textbox(visible=False)
        narrative = gr.Textbox(visible=False)
        movement = gr.Textbox(visible=False)
        historical_button = gr.Button(visible=False)
        narrative_button = gr.Button(visible=False)
        movement_button = gr.Button(visible=False)

        historical_button.click(
            historical_endpoint,
            inputs=[start_date, cards_id],
            outputs=[status, factor_table, warnings, sidecar_json, sidecar_state],
            api_name="historical_scenario",
        )
        narrative_button.click(
            narrative_endpoint,
            inputs=[sidecar_input, cards_id, output_directory],
            outputs=[status, narrative],
            api_name="generate_narrative",
        )
        movement_button.click(
            movement_endpoint,
            inputs=[start_date],
            outputs=[movement],
            api_name="scenario_movement",
        )
    return app
