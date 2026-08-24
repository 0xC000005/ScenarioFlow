"""Typed, dependency-free payloads shared by both demo directions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _items(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list")
    return value


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    return text


def _number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain finite numbers") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must contain finite numbers")
    return number


def _number_tuple(value: object, field: str) -> tuple[float, ...]:
    return tuple(_number(item, field) for item in _items(value, field))


def _record_tuple(value: object, field: str) -> tuple[dict[str, Any], ...]:
    return tuple(dict(_mapping(item, field)) for item in _items(value, field))


@dataclass(frozen=True, slots=True)
class FanSeries:
    """Quantiles, mean, and sample paths for one market factor."""

    p10: tuple[float, ...]
    p50: tuple[float, ...]
    p90: tuple[float, ...]
    mean: tuple[float, ...]
    paths: tuple[tuple[float, ...], ...]

    @classmethod
    def from_mapping(cls, value: object) -> "FanSeries":
        row = _mapping(value, "fan series")
        raw_paths = _items(row.get("sample_paths", ()), "fan series sample_paths")
        paths: list[tuple[float, ...]] = []
        for index, raw_path in enumerate(raw_paths):
            if isinstance(raw_path, Mapping):
                raw_path = raw_path.get("values", ())
            paths.append(_number_tuple(raw_path, f"fan series sample_paths[{index}]"))
        return cls(
            p10=_number_tuple(row.get("p10", ()), "fan series p10"),
            p50=_number_tuple(row.get("p50", ()), "fan series p50"),
            p90=_number_tuple(row.get("p90", ()), "fan series p90"),
            mean=_number_tuple(row.get("mean", ()), "fan series mean"),
            paths=tuple(paths),
        )


@dataclass(frozen=True, slots=True)
class FactorFan:
    """Language-conditioned and start-only distributions for one factor."""

    market: str
    display_name: str
    value_kind: str
    start_level: float
    conditional: FanSeries
    start_only: FanSeries

    @classmethod
    def from_rows(cls, conditional: object, start_only: object) -> "FactorFan":
        row = _mapping(conditional, "conditional factor row")
        baseline = _mapping(start_only, "start-only factor row")
        market = _text(row.get("market"), "market")
        return cls(
            market=market,
            display_name=_text(
                row.get("display_name", market), "display_name", allow_empty=True
            )
            or market,
            value_kind=_text(row.get("value_kind"), "value_kind"),
            start_level=_number(row.get("start_level"), "start_level"),
            conditional=FanSeries.from_mapping(row),
            start_only=FanSeries.from_mapping(baseline),
        )


@dataclass(frozen=True, slots=True)
class HistoricalPrecedent:
    """One retrieved historical period used by the language condition."""

    calendar_date: str
    window_id: str
    weight: float
    similarity: float
    path_count: int

    @classmethod
    def from_mapping(cls, value: object) -> "HistoricalPrecedent":
        row = _mapping(value, "historical precedent")
        date = _text(
            row.get("history_start_date", row.get("window_id")),
            "historical precedent date",
        )[:10]
        path_count = int(_number(row.get("component_sample_count", 0), "path_count"))
        if path_count < 0:
            raise ValueError("path_count must be non-negative")
        return cls(
            calendar_date=date,
            window_id=_text(row.get("window_id", date), "window_id"),
            weight=_number(row.get("base_support_weight", 0), "weight"),
            similarity=_number(row.get("memory_support_cosine", 0), "similarity"),
            path_count=path_count,
        )


@dataclass(frozen=True, slots=True)
class NarrativeScenarioPayload:
    """The numerical payload returned by narrative-to-scenario generation."""

    window_index: int
    context: dict[str, tuple[float, ...]]
    factors: dict[str, FactorFan]
    precedents: tuple[HistoricalPrecedent, ...]

    @classmethod
    def from_report(
        cls,
        *,
        window_index: object,
        context: object,
        report: object,
    ) -> "NarrativeScenarioPayload":
        report_row = _mapping(report, "scenario report")
        generation = _mapping(report_row.get("generation"), "generation")
        conditional_rows = _items(
            generation.get("path_quantiles", ()), "generation path_quantiles"
        )
        baseline_block = _mapping(
            generation.get("start_only_baseline", {}), "start_only_baseline"
        )
        baseline_rows = _items(
            baseline_block.get("path_quantiles", ()),
            "start-only path_quantiles",
        )
        baselines = {
            _text(_mapping(row, "start-only factor row").get("market"), "market"): row
            for row in baseline_rows
        }
        factors: dict[str, FactorFan] = {}
        for raw_row in conditional_rows:
            row = _mapping(raw_row, "conditional factor row")
            market = _text(row.get("market"), "market")
            factors[market] = FactorFan.from_rows(row, baselines.get(market, row))
        if not factors:
            raise ValueError("generation path_quantiles must not be empty")

        posterior = _mapping(
            generation.get("posterior_ensemble", {}), "posterior_ensemble"
        )
        precedents = tuple(
            HistoricalPrecedent.from_mapping(item)
            for item in _items(
                posterior.get("selected_support", ()), "selected_support"
            )
        )
        context_row = _mapping(context, "context")
        parsed_context = {
            _text(market, "context market"): _number_tuple(values, f"context {market}")
            for market, values in context_row.items()
        }
        try:
            index = int(window_index)
        except (TypeError, ValueError) as error:
            raise ValueError("window_index must be an integer") from error
        return cls(
            window_index=index,
            context=parsed_context,
            factors=factors,
            precedents=precedents,
        )


@dataclass(frozen=True, slots=True)
class FactorRow:
    """One factor movement in a numerical scenario sidecar."""

    factor: str
    start: float
    end: float
    delta: float
    p10_delta: float | None
    p50_delta: float | None
    p90_delta: float | None
    direction: str
    magnitude: str
    confidence: str
    evidence: str
    path_values: tuple[float, ...] | None

    @classmethod
    def from_mapping(cls, value: object) -> "FactorRow":
        row = _mapping(value, "factor row")

        def optional_number(field: str) -> float | None:
            raw = row.get(field)
            return None if raw is None else _number(raw, field)

        raw_path = row.get("path_values")
        return cls(
            factor=_text(row.get("factor"), "factor"),
            start=_number(row.get("start"), "start"),
            end=_number(row.get("end"), "end"),
            delta=_number(row.get("delta"), "delta"),
            p10_delta=optional_number("p10_delta"),
            p50_delta=optional_number("p50_delta"),
            p90_delta=optional_number("p90_delta"),
            direction=_text(row.get("direction"), "direction"),
            magnitude=_text(row.get("magnitude"), "magnitude"),
            confidence=_text(row.get("confidence", "medium"), "confidence"),
            evidence=_text(row.get("evidence"), "evidence"),
            path_values=(
                None if raw_path is None else _number_tuple(raw_path, "path_values")
            ),
        )


@dataclass(frozen=True, slots=True)
class ScenarioSidecar:
    """Machine-readable numerical evidence supplied to the narrative author."""

    scenario_id: str
    scenario_type: str
    scenario_title: str
    archetype: str
    horizon_days: int
    factor_rows: tuple[FactorRow, ...]
    mechanical_summary: str
    normalization_warnings: tuple[str | dict[str, str], ...]
    summary_source: str
    sample_count: int | None
    source_artifacts: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> "ScenarioSidecar":
        row = _mapping(value, "scenario sidecar")
        try:
            horizon = int(row.get("horizon_days", 30))
        except (TypeError, ValueError) as error:
            raise ValueError("horizon_days must be a positive integer") from error
        if horizon <= 0:
            raise ValueError("horizon_days must be a positive integer")
        sample_value = row.get("sample_count")
        sample_count = None if sample_value is None else int(sample_value)
        warnings: list[str | dict[str, str]] = []
        for warning in _items(
            row.get("normalization_warnings", ()), "normalization_warnings"
        ):
            if isinstance(warning, Mapping):
                warnings.append({str(key): str(item) for key, item in warning.items()})
            else:
                warnings.append(str(warning))
        source_artifacts = {
            str(key): str(item)
            for key, item in _mapping(
                row.get("source_artifacts", {}), "source_artifacts"
            ).items()
        }
        return cls(
            scenario_id=_text(row.get("scenario_id"), "scenario_id"),
            scenario_type=_text(row.get("scenario_type"), "scenario_type"),
            scenario_title=_text(
                row.get("scenario_title", ""), "scenario_title", allow_empty=True
            ),
            archetype=_text(row.get("archetype", "mixed_ambiguous"), "archetype"),
            horizon_days=horizon,
            factor_rows=tuple(
                FactorRow.from_mapping(item)
                for item in _items(row.get("factor_rows", ()), "factor_rows")
            ),
            mechanical_summary=_text(
                row.get("mechanical_summary"), "mechanical_summary"
            ),
            normalization_warnings=tuple(warnings),
            summary_source=_text(
                row.get("summary_source", "unknown"), "summary_source"
            ),
            sample_count=sample_count,
            source_artifacts=source_artifacts,
        )


@dataclass(frozen=True, slots=True)
class NarrativePacket:
    """Narratives and validation results grounded in one scenario sidecar."""

    scenario_sidecar: ScenarioSidecar
    positive_narratives: tuple[dict[str, Any], ...]
    hard_negative_narratives: tuple[dict[str, Any], ...]
    paired_review: tuple[dict[str, Any], ...]
    validation: dict[str, Any]
    artifact_paths: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> "NarrativePacket":
        row = _mapping(value, "narrative packet")
        return cls(
            scenario_sidecar=ScenarioSidecar.from_mapping(row.get("scenario_sidecar")),
            positive_narratives=_record_tuple(
                row.get("positive_narratives", ()), "positive_narratives"
            ),
            hard_negative_narratives=_record_tuple(
                row.get("hard_negative_narratives", ()), "hard_negative_narratives"
            ),
            paired_review=_record_tuple(row.get("paired_review", ()), "paired_review"),
            validation=dict(_mapping(row.get("validation", {}), "validation")),
            artifact_paths={
                str(key): str(item)
                for key, item in _mapping(
                    row.get("artifact_paths", {}), "artifact_paths"
                ).items()
            },
        )
