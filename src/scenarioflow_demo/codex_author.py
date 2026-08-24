"""One isolated Codex CLI call for grounded scenario narratives."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .contracts import NarrativePacket, ScenarioSidecar

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "xhigh"
NARRATIVE_VIEWS = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "mechanism_first",
    "technical_factor_evidence",
    "factor_list_baseline",
    "institutional_risk_committee_note",
    "risk_manager_memo",
    "full_professional",
    "sparse_variant_tape_read",
    "sparse_variant_portfolio_concern",
    "sparse_variant_macro_channel",
    "sparse_variant_credit_ambiguity",
    "sparse_variant_rates_commodities",
    "sparse_variant_desk_note",
)


class CodexAuthorError(RuntimeError):
    """A safe, reader-facing failure from the optional narrative author."""


def narrative_schema() -> dict[str, Any]:
    pair = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "view_name": {"type": "string"},
            "positive_text": {"type": "string", "minLength": 8},
            "negative_window_id": {"type": "string", "minLength": 1},
            "negative_text": {"type": "string", "minLength": 8},
            "quality_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "view_name",
            "positive_text",
            "negative_window_id",
            "negative_text",
            "quality_notes",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_window_id": {"type": "string", "minLength": 1},
            "target_title": {"type": "string", "minLength": 1},
            "pairs": {
                "type": "array",
                "minItems": len(NARRATIVE_VIEWS),
                "maxItems": len(NARRATIVE_VIEWS),
                "items": pair,
            },
        },
        "required": ["target_window_id", "target_title", "pairs"],
    }


def _target(sidecar: ScenarioSidecar) -> dict[str, Any]:
    return {
        "window_id": sidecar.scenario_id,
        "scenario_title": sidecar.scenario_title or sidecar.scenario_id,
        "archetype": sidecar.archetype,
        "mechanical_summary": sidecar.mechanical_summary,
        "factor_rows": [
            {
                key: value
                for key, value in asdict(row).items()
                if key not in {"path_values"} and value is not None
            }
            for row in sidecar.factor_rows
        ],
    }


def build_prompt(
    sidecar: ScenarioSidecar,
    negative_candidates: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "target": _target(sidecar),
        "negative_candidates": [dict(candidate) for candidate in negative_candidates],
        "expected_view_names": list(NARRATIVE_VIEWS),
    }
    return (
        "Write exactly one positive and one contradictory control narrative for each "
        "expected view. The positive narrative must faithfully describe the target "
        "scenario. The control must faithfully describe one supplied contrasting "
        "historical episode and use its window_id. Keep each pair in the same writing "
        "style. Do not mention the model, embeddings, training, retrieval, positive or "
        "negative labels, or invent news events. Return only JSON matching the supplied "
        "schema.\n\n"
        f"Payload:\n{json.dumps(payload, sort_keys=True)}\n"
    )


def _validate_output(
    value: object,
    sidecar: ScenarioSidecar,
    negative_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise CodexAuthorError("Codex returned an invalid narrative packet.")
    if value.get("target_window_id") != sidecar.scenario_id:
        raise CodexAuthorError("Codex returned an invalid narrative packet.")
    expected_title = sidecar.scenario_title or sidecar.scenario_id
    if value.get("target_title") != expected_title:
        raise CodexAuthorError("Codex returned an invalid narrative packet.")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(NARRATIVE_VIEWS):
        raise CodexAuthorError("Codex returned an incomplete narrative packet.")
    valid_windows = {
        str(candidate.get("window_id", "")) for candidate in negative_candidates
    }
    by_view: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise CodexAuthorError("Codex returned an invalid narrative packet.")
        view = str(pair.get("view_name", ""))
        if view in by_view or view not in NARRATIVE_VIEWS:
            raise CodexAuthorError("Codex returned an invalid narrative packet.")
        if str(pair.get("negative_window_id", "")) not in valid_windows:
            raise CodexAuthorError("Codex returned an invalid narrative packet.")
        for field in ("positive_text", "negative_text"):
            if len(str(pair.get(field, "")).strip()) < 8:
                raise CodexAuthorError("Codex returned an invalid narrative packet.")
        by_view[view] = dict(pair)
    if tuple(by_view) != NARRATIVE_VIEWS:
        by_view = {view: by_view[view] for view in NARRATIVE_VIEWS}
    return list(by_view.values())


def author_narrative_packet(
    sidecar: ScenarioSidecar,
    negative_candidates: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: int = 1200,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    temporary_root: str | Path | None = None,
) -> NarrativePacket:
    """Run Codex without exposing its session, files, or process output to the UI."""
    codex = which("codex")
    if codex is None:
        raise CodexAuthorError("Codex CLI is not installed.")
    if not negative_candidates:
        raise ValueError("at least one contrasting historical episode is required")
    root = None if temporary_root is None else str(temporary_root)
    with tempfile.TemporaryDirectory(prefix="scenarioflow-author-", dir=root) as temp:
        workspace = Path(temp)
        schema_path = workspace / "schema.json"
        output_path = workspace / "output.json"
        schema_path.write_text(json.dumps(narrative_schema()), encoding="utf-8")
        command = [
            codex,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--disable",
            "apps",
            "--disable",
            "image_generation",
            "--model",
            model,
            "--config",
            f"model_reasoning_effort='{reasoning_effort}'",
            "--sandbox",
            "read-only",
            "--cd",
            str(workspace),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-",
        ]
        try:
            completed = run(
                command,
                cwd=workspace,
                text=True,
                input=build_prompt(sidecar, negative_candidates),
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise CodexAuthorError("Codex could not author the narratives.") from error
        if completed.returncode != 0:
            raise CodexAuthorError("Codex could not author the narratives.")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CodexAuthorError(
                "Codex returned an invalid narrative packet."
            ) from error
    pairs = _validate_output(raw, sidecar, negative_candidates)
    positives = tuple(
        {"view_name": pair["view_name"], "text": pair["positive_text"]}
        for pair in pairs
    )
    negatives = tuple(
        {
            "view_name": pair["view_name"],
            "negative_window_id": pair["negative_window_id"],
            "text": pair["negative_text"],
        }
        for pair in pairs
    )
    return NarrativePacket(
        scenario_sidecar=sidecar,
        positive_narratives=positives,
        hard_negative_narratives=negatives,
        paired_review=tuple(pairs),
        validation={"status": "passed", "error_count": 0},
        artifact_paths={},
    )
