"""Numerical evidence preparation for the scenario-to-narrative demo."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import FactorRow, ScenarioSidecar

SPREAD_FACTORS = frozenset({"AAA_OAS", "BBB_OAS", "HY_OAS", "IG_OAS"})
HISTORICAL_FACTORS = (
    ("SPX", 25),
    ("NIKKEI", 36),
    ("VIX", 38),
    ("BBB_OAS", 35),
    ("AAA_OAS", 34),
    ("DXY", 28),
    ("USDJPY", 27),
    ("CRUDE_OIL", 31),
    ("US2Y", 32),
    ("US10Y", 33),
    ("GOLD", 37),
)


def _compact(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _direction(factor: str, delta: float) -> str:
    if delta == 0:
        return "flat"
    if factor in SPREAD_FACTORS or factor.endswith("_OAS"):
        return "wider" if delta > 0 else "tighter"
    return "up" if delta > 0 else "down"


def _magnitude(delta: float) -> str:
    absolute = abs(delta)
    if absolute == 0:
        return "flat"
    if absolute < 1:
        return "small"
    if absolute < 10:
        return "medium"
    return "large"


def factor_row_from_path(
    factor: str,
    values: Iterable[float],
    *,
    confidence: str = "observed",
) -> FactorRow:
    """Summarize one finite numerical path without discarding the path itself."""
    name = _compact(factor).upper()
    path = tuple(float(value) for value in values)
    if not name:
        raise ValueError("factor must not be empty")
    if len(path) < 2 or any(not math.isfinite(value) for value in path):
        raise ValueError("factor path must contain at least two finite values")
    start, end = path[0], path[-1]
    delta = end - start
    evidence = f"start={start:.12g}; end={end:.12g}; delta={delta:.12g}"
    return FactorRow(
        factor=name,
        start=start,
        end=end,
        delta=delta,
        p10_delta=None,
        p50_delta=None,
        p90_delta=None,
        direction=_direction(name, delta),
        magnitude=_magnitude(delta),
        confidence=_compact(confidence) or "observed",
        evidence=f"{evidence}; path_points={len(path)}",
        path_values=path,
    )


def build_mechanical_summary(rows: Iterable[FactorRow]) -> str:
    """Give the narrative author a short numerical reading of the scenario."""
    pieces = [f"{row.factor} {row.direction} {row.magnitude}" for row in rows]
    return "Mechanical baseline: " + "; ".join(pieces)


def _load_cards(path: str | Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"episode card line {line_number} is not an object"
                    )
                cards.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("episode cards are unreadable") from error
    return cards


def _window_metadata(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("support report is unreadable") from error
    rows = payload.get("window_metadata") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("support report has no window_metadata list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def historical_start_dates(
    support_report_path: str | Path,
    cards_path: str | Path,
) -> tuple[str, ...]:
    """Return only dates that have both support metadata and an episode card."""
    card_ids = {_compact(card.get("window_id")) for card in _load_cards(cards_path)}
    dates: list[str] = []
    for row in _window_metadata(support_report_path):
        date = _compact(row.get("calendar_start_date"))
        window_id = _compact(row.get("window_id"))
        if date and window_id in card_ids:
            dates.append(date)
    return tuple(dates)


def _card_window_index(card: Mapping[str, Any]) -> int:
    metadata = card.get("support_metadata")
    if isinstance(metadata, Mapping):
        try:
            return int(metadata.get("window_index"))
        except (TypeError, ValueError):
            pass
    match = re.search(r"(\d+)\s*$", _compact(card.get("window_id")))
    if match is None:
        raise ValueError("episode card has no support window index")
    return int(match.group(1))


def _raw_history_rows(path: str | Path, window_index: int) -> tuple[FactorRow, ...]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            history_raw = np.asarray(arrays["history_raw"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("support arrays are unreadable") from error
    if history_raw.ndim != 3 or not 0 <= window_index < history_raw.shape[0]:
        raise ValueError("support arrays do not contain the selected window")
    history = history_raw[window_index]
    rows: list[FactorRow] = []
    for factor, column in HISTORICAL_FACTORS:
        if column >= history.shape[1]:
            raise ValueError("support arrays do not contain the 39-variable panel")
        values = _interpolate_finite(history[:, column])
        rows.append(factor_row_from_path(factor, values))
    return tuple(rows)


def load_historical_scenario(
    start_date: str,
    *,
    support_report_path: str | Path,
    cards_path: str | Path,
    support_arrays_path: str | Path,
) -> ScenarioSidecar:
    """Load the selected 30-day historical episode and its numerical paths."""
    selected = _compact(start_date)
    metadata = next(
        (
            row
            for row in _window_metadata(support_report_path)
            if _compact(row.get("calendar_start_date")) == selected
        ),
        None,
    )
    if metadata is None:
        raise ValueError("selected historical date is unavailable")
    window_id = _compact(metadata.get("window_id"))
    card = next(
        (
            card
            for card in _load_cards(cards_path)
            if _compact(card.get("window_id")) == window_id
        ),
        None,
    )
    if card is None:
        raise ValueError("selected historical episode has no narrative card")
    rows = _raw_history_rows(support_arrays_path, _card_window_index(card))
    return ScenarioSidecar(
        scenario_id=window_id,
        scenario_type="historical_joint39",
        scenario_title=_compact(card.get("scenario_title")),
        archetype=_compact(card.get("archetype")) or "mixed_ambiguous",
        horizon_days=30,
        factor_rows=rows,
        mechanical_summary=build_mechanical_summary(rows),
        normalization_warnings=(),
        summary_source="historical_episode_card",
        sample_count=None,
        source_artifacts={
            "cards_jsonl": "nl.episode-cards",
            "support_arrays": "nl.support-bank",
            "support_report": "nl.support-bank-report",
        },
    )


def _card_text(card: Mapping[str, Any]) -> str:
    caption = card.get("caption_fields")
    caption = caption if isinstance(caption, Mapping) else {}
    evidence = caption.get("evidence_used")
    evidence = evidence if isinstance(evidence, list) else []
    values = (
        card.get("scenario_title"),
        card.get("archetype"),
        card.get("mechanical_summary"),
        caption.get("mechanical_summary"),
        *evidence,
    )
    return "; ".join(_compact(value).lower() for value in values if _compact(value))


def _candidate_sign(text: str, factor: str) -> int:
    clauses = re.split(r"[;,\n|.]+", text)
    positive = (
        ("wider", "higher", "up")
        if factor.endswith("_OAS")
        else (
            "higher",
            "up",
            "firmer",
        )
    )
    negative = (
        ("tighter", "lower", "down")
        if factor.endswith("_OAS")
        else (
            "lower",
            "down",
            "softer",
        )
    )
    factor_pattern = rf"(?<![a-z0-9_]){re.escape(factor.lower())}(?![a-z0-9_])"
    for clause in clauses:
        if re.search(factor_pattern, clause) is None:
            continue
        has_positive = any(re.search(rf"\b{word}\b", clause) for word in positive)
        has_negative = any(re.search(rf"\b{word}\b", clause) for word in negative)
        if has_positive != has_negative:
            return 1 if has_positive else -1
    return 0


def select_negative_candidates(
    sidecar: ScenarioSidecar,
    cards: Iterable[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Choose historical descriptions that contradict visible factor directions."""
    if count <= 0:
        return []
    target_signs = {
        row.factor: 1 if row.direction in {"up", "wider"} else -1
        for row in sidecar.factor_rows
        if row.direction not in {"flat"}
    }
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for source in cards:
        window_id = _compact(source.get("window_id"))
        if not window_id or window_id == sidecar.scenario_id:
            continue
        text = _card_text(source)
        contradictions: list[str] = []
        agreements = 0
        for factor, target_sign in target_signs.items():
            sign = _candidate_sign(text, factor)
            if sign * target_sign < 0:
                contradictions.append(factor)
            elif sign * target_sign > 0:
                agreements += 1
        if not contradictions:
            continue
        caption = source.get("caption_fields")
        caption = caption if isinstance(caption, Mapping) else {}
        evidence = caption.get("evidence_used")
        candidate = {
            "window_id": window_id,
            "scenario_title": _compact(source.get("scenario_title")),
            "archetype": _compact(source.get("archetype")) or "mixed_ambiguous",
            "mechanical_summary": _compact(
                source.get("mechanical_summary") or caption.get("mechanical_summary")
            ),
            "evidence_used": (
                [_compact(value) for value in evidence if _compact(value)]
                if isinstance(evidence, list)
                else []
            ),
            "contradiction_channels": contradictions,
            "contradiction_count": len(contradictions),
            "agreement_count": agreements,
        }
        scored.append((100 * len(contradictions) + agreements, window_id, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [candidate for _score, _window, candidate in scored[:count]]
    if len(selected) < count:
        raise ValueError("not enough contrasting historical episodes")
    return selected


def load_negative_candidates(
    sidecar: ScenarioSidecar,
    cards_path: str | Path,
    *,
    count: int,
) -> list[dict[str, Any]]:
    return select_negative_candidates(sidecar, _load_cards(cards_path), count=count)


def _interpolate_finite(values: np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("selected path has no finite values")
    if not finite.all():
        positions = np.arange(array.size)
        array = np.interp(positions, positions[finite], array[finite])
    return tuple(float(value) for value in array)


def scenario_movement(
    start_date: str, data_path: str | Path
) -> dict[str, tuple[float, ...]]:
    """Return the observed 30-day factor paths shown beside the narrative."""
    try:
        with np.load(data_path, allow_pickle=False) as arrays:
            dates = np.asarray(arrays["dates"]).astype("datetime64[D]")
            levels = np.asarray(arrays["levels"], dtype=float)
            columns = tuple(str(value).upper() for value in arrays["level_columns"])
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("processed market panel is unreadable") from error
    selected = np.datetime64(_compact(start_date), "D")
    matches = np.flatnonzero(dates == selected)
    if matches.size != 1:
        raise ValueError("selected historical date is unavailable")
    start = int(matches[0])
    stop = min(start + 30, len(dates))
    if stop - start < 2:
        raise ValueError("selected historical period is incomplete")
    return {
        column: _interpolate_finite(levels[start:stop, index])
        for index, column in enumerate(columns)
    }
