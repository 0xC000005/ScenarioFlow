"""OpenAI language grounding and text-to-memory projection for ScenarioFlow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from .narrative_to_scenario import NarrativeAdapter

GROUNDING_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
PROFESSIONAL_STORY_SECTIONS = (
    "Scenario title:",
    "Mechanical summary:",
    "Dominant mechanism:",
    "Trigger and transmission:",
    "Cross-asset reaction:",
    "Portfolio/risk implication:",
    "Evidence and ambiguity:",
    "No-forecast caveat:",
    "Warning-only forward risk:",
)
CONDITIONING_STORY_SECTIONS = {
    "Mechanical summary:",
    "Dominant mechanism:",
    "Trigger and transmission:",
    "Cross-asset reaction:",
    "Evidence and ambiguity:",
}
AUDIT_ONLY_STORY_SECTIONS = {
    "Scenario title:",
    "Portfolio/risk implication:",
    "No-forecast caveat:",
}
WARNING_ONLY_STORY_SECTIONS = {"Warning-only forward risk:"}
FORWARD_LANGUAGE_MARKERS = {
    "could",
    "forecast",
    "forward",
    "future",
    "going to",
    "next",
    "outlook",
    "projection",
    "risk is",
    "will",
    "would",
}


class ConditionMarketImplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(min_length=1)
    direction: Literal["up", "down", "flat", "wider", "tighter", "mixed", "unclear"]
    magnitude: Literal["small", "medium", "large", "flat", "unclear"]
    confidence: Literal["low", "medium", "high"]
    horizon: Literal["current_state", "recent_regime"]
    target_use: Literal["support_prior"] = "support_prior"
    evidence: list[str] = Field(default_factory=list)
    inferred: bool = False
    rationale: str = Field(min_length=1)


class NonConditioningForwardLanguage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    handling: Literal["warning_only", "ignore_for_conditioning"] = "warning_only"
    severity: Literal["info", "warning"] = "warning"


class ConditionGroundingWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1)
    target_use: Literal["support_prior", "ignore"] = "ignore"


class ConditionOnlyGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: Literal["condition_only_grounding_v1"] = (
        "condition_only_grounding_v1"
    )
    narrative_frame: str = Field(min_length=1)
    current_market_state_summary: str = Field(min_length=1)
    recent_regime_summary: str = Field(min_length=1)
    cleaned_conditioning_text: str = Field(min_length=12)
    current_market_state_implications: list[ConditionMarketImplication] = Field(
        default_factory=list
    )
    recent_regime_implications: list[ConditionMarketImplication] = Field(
        default_factory=list
    )
    non_conditioning_forward_language: list[NonConditioningForwardLanguage] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    grounding_warnings: list[ConditionGroundingWarning] = Field(default_factory=list)
    critique: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ConditionedNarrative:
    grounding: ConditionOnlyGroundingResult
    query_text: str
    query_memory: np.ndarray


def _has_forward_language(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in FORWARD_LANGUAGE_MARKERS)


def _professional_story_sections(story: str) -> list[tuple[str, str]]:
    canonical = {section.lower(): section for section in PROFESSIONAL_STORY_SECTIONS}
    pattern = re.compile(
        "|".join(re.escape(section) for section in PROFESSIONAL_STORY_SECTIONS),
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(str(story)))
    if len(matches) < 2:
        return []
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(story)
        text = str(story)[match.start() : end].strip()
        if text:
            sections.append((canonical[match.group(0).lower()], text))
    return sections


def split_story_for_conditioning(story: str) -> dict[str, Any]:
    structured = _professional_story_sections(story)
    if structured:
        conditioning: list[str] = []
        forward: list[str] = []
        audit: list[str] = []
        for label, text in structured:
            if label in WARNING_ONLY_STORY_SECTIONS:
                forward.append(text)
            elif label in AUDIT_ONLY_STORY_SECTIONS:
                audit.append(text)
            elif label in CONDITIONING_STORY_SECTIONS:
                conditioning.append(text)
            elif _has_forward_language(text):
                forward.append(text)
            else:
                conditioning.append(text)
        return {
            "conditioning_sentences": conditioning,
            "non_conditioning_forward_sentences": forward,
            "audit_only_sentences": audit,
        }
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(story).strip())
        if sentence.strip()
    ]
    return {
        "conditioning_sentences": [
            sentence for sentence in sentences if not _has_forward_language(sentence)
        ],
        "non_conditioning_forward_sentences": [
            sentence for sentence in sentences if _has_forward_language(sentence)
        ],
        "audit_only_sentences": [],
    }


def grounding_messages(story: str) -> list[dict[str, str]]:
    split = split_story_for_conditioning(story)
    conditioning = (
        "\n".join(f"- {sentence}" for sentence in split["conditioning_sentences"])
        or "- none"
    )
    forward = (
        "\n".join(
            f"- {sentence}" for sentence in split["non_conditioning_forward_sentences"]
        )
        or "- none"
    )
    audit = (
        "\n".join(f"- {sentence}" for sentence in split["audit_only_sentences"])
        or "- none"
    )
    return [
        {
            "role": "system",
            "content": (
                "You convert risk-manager narratives into auditable conditioning facts "
                "for a financial scenario generator. The narrative is not a requested "
                "future path. Extract only current or recent-regime market conditions. "
                "Future-looking phrases, stress forecasts, or desired outcomes must be "
                "reported as warnings and ignored for conditioning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return a ConditionOnlyGroundingResult JSON object. The market field "
                "must be one of AAA_OAS, BBB_OAS, CRUDE_OIL, DXY, GOLD, IV_SKEW, "
                "IV_SURFACE, SPX, US10Y, US2Y, USDJPY, or VIX. For generic credit "
                "spreads, prefer BBB_OAS as a broad proxy and add a proxy warning.\n\n"
                "Current implications are only for stated present conditions and use "
                "target_use=support_prior with horizon=current_state. Recent-regime "
                "implications are only for stated recent paths and use horizon=recent_regime. "
                "Do not create future scenario implications. Put forecasts, desired outcomes, "
                "and phrases such as 'could', 'will', 'next month', or 'forward risk' in "
                "non_conditioning_forward_language. Build the summaries and cleaned text "
                "only from the conditioning candidates. If no separate recent regime is "
                "stated, say so. Do not invent market channels or causal facts.\n\n"
                "Conditioning candidate sentences. Extract implications only from this section:\n"
                f"{conditioning}\n\n"
                "Audit-only professional sections. Do not use these as conditioning facts:\n"
                f"{audit}\n\n"
                "Warning-only forward or forecast sentences. Report these as warnings and "
                "do not use them for conditioning:\n"
                f"{forward}"
            ),
        },
    ]


def query_text_from_grounding(
    _story: str, grounding: ConditionOnlyGroundingResult
) -> str:
    def implication(prefix: str, item: ConditionMarketImplication) -> str:
        evidence = "; ".join(item.evidence) if item.evidence else "no direct quote"
        inferred = " inferred" if item.inferred else ""
        return (
            f"{prefix}: {item.market} {item.direction} {item.magnitude} "
            f"confidence={item.confidence} horizon={item.horizon} "
            f"target={item.target_use}{inferred} evidence={evidence}"
        )

    lines = [
        f"CONDITIONING_FRAME: {grounding.narrative_frame}",
        f"CURRENT_MARKET_STATE_SUMMARY: {grounding.current_market_state_summary}",
        f"RECENT_REGIME_SUMMARY: {grounding.recent_regime_summary}",
        f"CLEANED_CONDITIONING_TEXT: {grounding.cleaned_conditioning_text}",
        "CURRENT_SUPPORT_IMPLICATIONS:",
    ]
    lines.extend(
        implication("current", item)
        for item in grounding.current_market_state_implications
    )
    lines.append("RECENT_REGIME_IMPLICATIONS:")
    lines.extend(
        implication("recent", item) for item in grounding.recent_regime_implications
    )
    return "\n".join(lines)


def load_bridge_adapter(path: str | Path) -> NarrativeAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    embedding_dim = int(payload.get("embedding_dim", state["net.0.weight"].shape[0]))
    condition_dim = int(payload.get("condition_dim", state["net.3.weight"].shape[0]))
    hidden_dim = int(state["net.1.weight"].shape[0])
    adapter = NarrativeAdapter(embedding_dim, condition_dim, hidden_dim=hidden_dim)
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


class OpenAINarrativeConditioner:
    """Ground one narrative, embed it, and project it into historical memory space."""

    def __init__(
        self,
        bridge_adapter: str | Path,
        *,
        grounding_model: str = GROUNDING_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        client: Any | None = None,
    ) -> None:
        self.adapter = load_bridge_adapter(bridge_adapter)
        self.grounding_model = str(grounding_model)
        self.embedding_model = str(embedding_model)
        self._client = client

    def _client_or_default(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=60.0, max_retries=2)
        return self._client

    def condition(self, story: str) -> ConditionedNarrative:
        narrative = str(story).strip()
        if not narrative:
            raise ValueError("narrative must not be blank")
        client = self._client_or_default()
        response = client.responses.parse(
            model=self.grounding_model,
            input=grounding_messages(narrative),
            text_format=ConditionOnlyGroundingResult,
            max_output_tokens=1200,
        )
        grounding = response.output_parsed
        if not isinstance(grounding, ConditionOnlyGroundingResult):
            raise TypeError("OpenAI did not return the expected grounding object")
        query_text = query_text_from_grounding(narrative, grounding)
        embedded = client.embeddings.create(
            model=self.embedding_model, input=[query_text]
        )
        ordered = sorted(embedded.data, key=lambda item: item.index)
        if len(ordered) != 1:
            raise ValueError("OpenAI did not return one text embedding")
        vector = np.asarray(ordered[0].embedding, dtype=np.float32).reshape(1, -1)
        norm = np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        with torch.no_grad():
            memory = self.adapter(torch.from_numpy(vector / norm))
        return ConditionedNarrative(
            grounding=grounding,
            query_text=query_text,
            query_memory=memory.detach().cpu().numpy()[0].astype(np.float32),
        )
