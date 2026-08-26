"""Export the frozen dollar-liquidity case used in the papers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "scenarioflow.dollar-liquidity-table.v1"

REPORT_IDENTITIES = {
    "narrative": (
        605_667,
        "a2bee8502f6538e0a4649c1f10e24e3bec052f52ac1cc02ab8c26883c7155008",
    ),
    "start_only": (
        616_669,
        "46fe2b8fd4363ea37a4aaa0f2377455ff13510ab601bd29a3d767c86feec24e3",
    ),
    "comparison": (
        94_517,
        "102e92b16579b2b185ac17b8eb0818517047619aee07ec5416da26678d284e8d",
    ),
}
ARRAY_IDENTITIES = {
    "narrative": (
        9_768_358,
        "8073dfa02d7d22de66ef568c2637be7e844a27bc119ec69e94e520557163f580",
    ),
    "start_only": (
        9_765_999,
        "2d582d31a04785334e528218817c8a46fd95d659ffdc0de493319223225d0e36",
    ),
}
GOLDEN_IDENTITIES = {
    "fmts-dollar-liquidity-full.tex": (
        2_503,
        "6c92a8310dc69502bfc132c49c84dbf2bbf8b28fdc230bd43df058423b1696bb",
    ),
    "fmts-dollar-liquidity-short.tex": (
        2_360,
        "db6e6e6aff6ba2f37da608a050cea94449bbb17b391ee38498311a42b9cfa0dc",
    ),
    "icaif-dollar-liquidity.tex": (
        2_422,
        "a49f1a6a9c2e0363d0ef86e30198ed909166dfd8fbf0a80cde9b8c22941bc767",
    ),
}

JOINT39_SPEC_NAMES = tuple(
    [f"iv:{index:02d}" for index in range(25)]
    + [
        "factor:spx",
        "factor:usdcad",
        "factor:usdjpy",
        "factor:dxy",
        "factor:copper",
        "factor:wheat",
        "factor:crude_oil",
        "factor:us2y",
        "factor:us10y",
        "factor:aaa_oas",
        "factor:bbb_oas",
        "factor:nikkei",
        "factor:gold",
        "factor:vix",
    ]
)
MARKETS = (
    ("IV_SURFACE", "IV surface", None),
    ("SPX", "SPX", 25),
    ("US2Y", "US2Y", 32),
    ("US10Y", "US10Y", 33),
    ("BBB_OAS", "BBB OAS", 35),
    ("AAA_OAS", "AAA OAS", 34),
    ("USDJPY", "USDJPY", 27),
    ("DXY", "DXY", 28),
    ("GOLD", "Gold", 37),
    ("CRUDE_OIL", "Crude oil", 31),
    ("VIX", "VIX", 38),
)
REQUIRED_ARRAY_KEYS = frozenset(
    {
        "generated_states",
        "requested_raw",
        "rollout_component_variant_index",
        "rollout_component_window_index",
        "rollout_component_weight",
        "rollout_component_sample_count",
    }
)


class DollarLiquidityExportError(ValueError):
    """Raised when dollar-liquidity evidence or output is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: tuple[int, str], *, label: str) -> Path:
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise DollarLiquidityExportError(f"{label} authority is not a regular file")
        if source.stat().st_size != expected[0] or _sha256(source) != expected[1]:
            raise DollarLiquidityExportError(f"{label} authority identity changed")
    except OSError as error:
        raise DollarLiquidityExportError(f"{label} authority is unreadable") from error
    return source


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DollarLiquidityExportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DollarLiquidityExportError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DollarLiquidityExportError(
            "dollar-liquidity report is unreadable"
        ) from error
    if not isinstance(document, dict):
        raise DollarLiquidityExportError("dollar-liquidity report must be an object")
    return document


def select_sparse_support(
    components: Sequence[Mapping[str, Any]],
    *,
    max_components: int = 3,
    min_cumulative_weight: float = 0.90,
) -> list[dict[str, Any]]:
    """Select the largest support components and renormalize their weights."""
    if max_components < 1:
        raise DollarLiquidityExportError("max_components must be at least one")
    if not 0.0 < min_cumulative_weight <= 1.0:
        raise DollarLiquidityExportError(
            "min_cumulative_weight must be in the interval (0, 1]"
        )
    ranked = sorted(
        components,
        key=lambda row: (
            float(row.get("weight", 0.0)),
            -int(row.get("component_no", 0)),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    cumulative = 0.0
    for component in ranked:
        if len(selected) >= max_components:
            break
        weight = float(component.get("weight", 0.0))
        if not math.isfinite(weight):
            raise DollarLiquidityExportError("component weight must be finite")
        selected.append(dict(component))
        cumulative += max(weight, 0.0)
        if cumulative >= min_cumulative_weight:
            break

    total = sum(max(float(row.get("weight", 0.0)), 0.0) for row in selected)
    if selected and total <= 0.0:
        total = float(len(selected))
        for row in selected:
            row["sparse_weight"] = 1.0 / total
    elif total > 0.0:
        for row in selected:
            row["sparse_weight"] = max(float(row.get("weight", 0.0)), 0.0) / total
    return sorted(selected, key=lambda row: int(row.get("component_no", 0)))


def _operational_variant(report: Mapping[str, Any], variant_count: int) -> int:
    selected = report.get("selected_start_state")
    value = selected.get("variant_index") if isinstance(selected, Mapping) else None
    try:
        index = int(value) if value is not None else variant_count - 1
    except (TypeError, ValueError) as error:
        raise DollarLiquidityExportError("selected variant index is invalid") from error
    if not 0 <= index < variant_count:
        raise DollarLiquidityExportError("selected variant index is out of range")
    return index


def _start_date(report: Mapping[str, Any]) -> str:
    rows = report.get("variant_rows")
    if not isinstance(rows, list):
        raise DollarLiquidityExportError("report has no variant rows")
    operational = [
        row for row in rows if isinstance(row, dict) and row.get("is_operational")
    ]
    if len(operational) != 1:
        raise DollarLiquidityExportError("report must identify one operational start")
    calendar = operational[0].get("start_calendar")
    value = calendar.get("calendar_end_date") if isinstance(calendar, dict) else None
    if not isinstance(value, str) or not value:
        raise DollarLiquidityExportError("operational start date is missing")
    return value


def _component_rows(
    arrays: Mapping[str, np.ndarray], variant: int
) -> list[dict[str, Any]]:
    metadata = [
        np.asarray(arrays[key])
        for key in (
            "rollout_component_variant_index",
            "rollout_component_window_index",
            "rollout_component_weight",
            "rollout_component_sample_count",
        )
    ]
    if (
        any(array.ndim != 1 for array in metadata)
        or len({len(array) for array in metadata}) != 1
    ):
        raise DollarLiquidityExportError("component metadata shapes are invalid")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for metadata_row, (variant_index, window, weight, count) in enumerate(
        zip(*metadata, strict=True)
    ):
        if int(variant_index) != variant:
            continue
        sample_count = int(count)
        if sample_count <= 0:
            raise DollarLiquidityExportError("component sample count must be positive")
        rows.append(
            {
                "component_no": len(rows),
                "metadata_row": metadata_row,
                "window_index": int(window),
                "weight": float(weight),
                "sample_count": sample_count,
                "sample_slice": [cursor, cursor + sample_count],
            }
        )
        cursor += sample_count
    return rows


def _direction(probability_up: float, probability_down: float) -> str:
    if probability_up >= 0.60 and probability_up > probability_down:
        return "up"
    if probability_down >= 0.60 and probability_down > probability_up:
        return "down"
    return "flat"


def _path_share(
    probability_up: float, probability_down: float
) -> tuple[str, float, str]:
    if abs(probability_up - probability_down) <= 0.005:
        share = max(probability_up, probability_down)
        return f"{share:.0%} split", share, "split"
    if probability_up >= probability_down:
        return f"{probability_up:.0%} up", probability_up, "up"
    return f"{probability_down:.0%} down", probability_down, "down"


def _signed(value: float, digits: int) -> str:
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.{digits}f}"


def _raw_move(market_id: str, value: float) -> str:
    if market_id == "IV_SURFACE":
        return f"{_signed(value * 100.0, 1)} vol pts"
    if market_id in {"US2Y", "US10Y", "BBB_OAS", "AAA_OAS"}:
        return f"{_signed(value * 100.0, 1)} bp"
    if market_id in {"SPX", "GOLD"}:
        return f"{_signed(value, 0)} pts"
    return f"{_signed(value, 1)} pts"


def _market_summary(market_id: str, values: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        raise DollarLiquidityExportError(f"{market_id} has no finite terminal paths")
    mean = float(np.mean(valid))
    p10, p90 = (float(value) for value in np.percentile(valid, (10, 90)))
    probability_up = float(np.mean(valid > 0.0))
    probability_down = float(np.mean(valid < 0.0))
    probability_flat = float(np.mean(valid == 0.0))
    width = abs(p90 - p10)
    sigma = mean / (width / 2.5631031310892007) if width > 1.0e-12 else None
    share_label, majority_share, majority_side = _path_share(
        probability_up, probability_down
    )
    raw_move = _raw_move(market_id, mean)
    mean_move = raw_move if sigma is None else f"{raw_move} / {_signed(sigma, 1)}σ"
    return {
        "direction": _direction(probability_up, probability_down),
        "majority_share": majority_share,
        "majority_side": majority_side,
        "mean_move_label": mean_move,
        "mean_terminal_delta": mean,
        "p10": p10,
        "p90": p90,
        "path_share_label": share_label,
        "probability_down": probability_down,
        "probability_flat": probability_flat,
        "probability_up": probability_up,
        "sigma": sigma,
    }


def summarize_sparse_deck(
    report: Mapping[str, Any], arrays_path: Path
) -> dict[str, Any]:
    """Summarize the selected top-three/90% component view of one deck."""
    try:
        with np.load(Path(arrays_path), allow_pickle=False) as archive:
            if not REQUIRED_ARRAY_KEYS.issubset(archive.files):
                raise DollarLiquidityExportError(
                    "dollar-liquidity arrays are incomplete"
                )
            arrays = {key: np.asarray(archive[key]) for key in REQUIRED_ARRAY_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, DollarLiquidityExportError):
            raise
        raise DollarLiquidityExportError(
            "dollar-liquidity arrays are unreadable"
        ) from error

    states = np.asarray(arrays["generated_states"], dtype=np.float64)
    starts = np.asarray(arrays["requested_raw"], dtype=np.float64)
    if (
        states.ndim != 4
        or starts.ndim != 2
        or states.shape[0] != starts.shape[0]
        or states.shape[-1] != len(JOINT39_SPEC_NAMES)
        or starts.shape[-1] != len(JOINT39_SPEC_NAMES)
    ):
        raise DollarLiquidityExportError("generated deck shape is invalid")
    variant = _operational_variant(report, states.shape[0])
    components = _component_rows(arrays, variant)
    if sum(row["sample_count"] for row in components) != states.shape[1]:
        raise DollarLiquidityExportError(
            "component sample counts do not match the deck"
        )
    selected = select_sparse_support(components)
    sample_indices = np.concatenate(
        [np.arange(*row["sample_slice"], dtype=np.int64) for row in selected]
    )
    terminal_delta = states[variant, sample_indices, -1, :] - starts[variant][None, :]

    markets: list[dict[str, Any]] = []
    for market_id, label, index in MARKETS:
        values = (
            np.mean(terminal_delta[:, :25], axis=1)
            if index is None
            else terminal_delta[:, index]
        )
        markets.append(
            {
                "market_id": market_id,
                "market": label,
                **_market_summary(market_id, values),
            }
        )

    public_support = [
        {
            key: row[key]
            for key in (
                "component_no",
                "window_index",
                "weight",
                "sparse_weight",
                "sample_count",
            )
        }
        for row in selected
    ]
    return {
        "start_date": _start_date(report),
        "variant_index": variant,
        "selected_component_count": len(selected),
        "selected_path_count": int(sample_indices.size),
        "selected_weight_mass": float(sum(row["weight"] for row in selected)),
        "selected_support": public_support,
        "start_state": starts[variant].astype(float).tolist(),
        "markets": markets,
    }


def _mean_direction(row: Mapping[str, Any]) -> str:
    mean = float(row["mean_terminal_delta"])
    width = abs(float(row["p90"]) - float(row["p10"]))
    threshold = max(0.05 * (width or abs(mean) or 1.0), 1.0e-8)
    if abs(mean) <= threshold:
        return "flat"
    return "up" if mean > 0.0 else "down"


def _comparison(
    narrative: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, str]:
    narrative_mean = float(narrative["mean_terminal_delta"])
    baseline_mean = float(baseline["mean_terminal_delta"])
    narrative_width = abs(float(narrative["p90"]) - float(narrative["p10"]))
    baseline_width = abs(float(baseline["p90"]) - float(baseline["p10"]))
    threshold = max(0.10 * max(narrative_width, baseline_width), 1.0e-8)
    delta = narrative_mean - baseline_mean
    if abs(delta) <= threshold:
        return {"category": "flat", "label": "Similar"}
    view = _mean_direction(narrative)
    if delta > 0.0:
        if view == "down":
            return {
                "category": "moderate_down",
                "label": "Less down than baseline",
            }
        if view == "up":
            return {"category": "up", "label": "More up than baseline"}
        return {"category": "up", "label": "Higher than baseline"}
    if view == "down":
        return {"category": "down", "label": "More down than baseline"}
    if view == "up":
        return {"category": "moderate_up", "label": "Less up than baseline"}
    return {"category": "down", "label": "Lower than baseline"}


def _validate_report_start(report: Mapping[str, Any], deck: Mapping[str, Any]) -> None:
    selected = report.get("selected_start_state")
    values = selected.get("values_by_name") if isinstance(selected, Mapping) else None
    if not isinstance(values, Mapping) or set(values) != set(JOINT39_SPEC_NAMES):
        raise DollarLiquidityExportError("selected start state is incomplete")
    expected = np.asarray(
        [values[name] for name in JOINT39_SPEC_NAMES], dtype=np.float64
    )
    actual = np.asarray(deck["start_state"], dtype=np.float64)
    if not np.array_equal(expected.astype(np.float32), actual.astype(np.float32)):
        raise DollarLiquidityExportError("report and arrays use different start states")


def _validate_comparison(
    comparison: Mapping[str, Any],
    narrative: Mapping[str, Any],
    start_only: Mapping[str, Any],
) -> None:
    policies = comparison.get("policies")
    if not isinstance(policies, list):
        raise DollarLiquidityExportError("comparison report has no policies")
    expected = {
        "cohesive_support_gap30": narrative,
        "start_only_topk": start_only,
    }
    found: dict[str, Mapping[str, Any]] = {}
    for policy in policies:
        if not isinstance(policy, Mapping) or policy.get("policy") not in expected:
            continue
        cases = policy.get("cases")
        if not isinstance(cases, list):
            continue
        matches = [
            case
            for case in cases
            if isinstance(case, Mapping) and case.get("case") == "dollar_liquidity"
        ]
        if len(matches) == 1:
            found[str(policy["policy"])] = matches[0]
    if set(found) != set(expected):
        raise DollarLiquidityExportError(
            "comparison report lacks both dollar-liquidity policies"
        )
    for policy, deck in expected.items():
        case = found[policy]
        if case.get("generated_shape") != [384, 30, 39]:
            raise DollarLiquidityExportError("comparison deck shape changed")
        support = case.get("support")
        if not isinstance(support, list):
            raise DollarLiquidityExportError("comparison support is missing")
        support_by_window = {
            int(row["window_index"]): float(row["weight"])
            for row in support
            if isinstance(row, Mapping) and "window_index" in row and "weight" in row
        }
        for selected in deck["selected_support"]:
            window = int(selected["window_index"])
            if window not in support_by_window or not math.isclose(
                support_by_window[window],
                float(selected["weight"]),
                rel_tol=1.0e-7,
            ):
                raise DollarLiquidityExportError(
                    "comparison support disagrees with arrays"
                )


def build_dollar_liquidity_document(
    narrative_report: Path,
    narrative_arrays: Path,
    start_only_report: Path,
    start_only_arrays: Path,
    comparison_report: Path,
) -> dict[str, Any]:
    """Validate frozen inputs and construct the shared numerical table object."""
    narrative_report = _verify_file(
        narrative_report,
        REPORT_IDENTITIES["narrative"],
        label="narrative report",
    )
    start_only_report = _verify_file(
        start_only_report,
        REPORT_IDENTITIES["start_only"],
        label="start-only report",
    )
    comparison_report = _verify_file(
        comparison_report,
        REPORT_IDENTITIES["comparison"],
        label="comparison report",
    )
    narrative_arrays = _verify_file(
        narrative_arrays,
        ARRAY_IDENTITIES["narrative"],
        label="narrative arrays",
    )
    start_only_arrays = _verify_file(
        start_only_arrays,
        ARRAY_IDENTITIES["start_only"],
        label="start-only arrays",
    )

    narrative_source = _load_json(narrative_report)
    start_only_source = _load_json(start_only_report)
    comparison_source = _load_json(comparison_report)
    narrative = summarize_sparse_deck(narrative_source, narrative_arrays)
    start_only = summarize_sparse_deck(start_only_source, start_only_arrays)
    _validate_report_start(narrative_source, narrative)
    _validate_report_start(start_only_source, start_only)
    if narrative["start_date"] != start_only["start_date"] or not np.array_equal(
        np.asarray(narrative["start_state"], dtype=np.float32),
        np.asarray(start_only["start_state"], dtype=np.float32),
    ):
        raise DollarLiquidityExportError("the two decks do not use one common start")
    _validate_comparison(comparison_source, narrative, start_only)

    narrative_markets = {row["market_id"]: row for row in narrative.pop("markets")}
    start_only_markets = {row["market_id"]: row for row in start_only.pop("markets")}
    rows = []
    for market_id, label, _index in MARKETS:
        baseline_row = start_only_markets[market_id]
        narrative_row = narrative_markets[market_id]
        rows.append(
            {
                "market_id": market_id,
                "market": label,
                "start_only": baseline_row,
                "narrative": narrative_row,
                "comparison": _comparison(narrative_row, baseline_row),
            }
        )
    narrative.pop("start_state")
    start_only.pop("start_state")
    return {
        "schema_version": SCHEMA_VERSION,
        "start_date": narrative["start_date"],
        "support_rule": {
            "max_components": 3,
            "min_cumulative_weight": 0.90,
        },
        "market_order": [market_id for market_id, _label, _index in MARKETS],
        "decks": {"start_only": start_only, "narrative": narrative},
        "rows": rows,
    }


def _direction_macro(direction: str) -> str:
    return {
        "up": r"\DemoViewUp",
        "down": r"\DemoViewDown",
        "flat": r"\DemoViewFlat",
    }[direction]


def _comparison_macro(comparison: Mapping[str, str]) -> str:
    macro = {
        "up": r"\DemoChangeUp",
        "down": r"\DemoChangeDown",
        "moderate_up": r"\DemoChangeModerateUp",
        "moderate_down": r"\DemoChangeModerateDown",
        "flat": r"\DemoChangeFlat",
    }[comparison["category"]]
    return f"{macro}{{{comparison['label']}}}"


def _tex_move(value: str) -> str:
    raw, sigma = value.rsplit(" / ", 1)
    number, unit = raw.split(" ", 1)
    sigma_tex = sigma.replace("σ", r"\sigma")
    return f"${number}$ {unit} / ${sigma_tex}$"


def _table_rows(document: Mapping[str, Any]) -> str:
    lines = []
    for row in document["rows"]:
        baseline = row["start_only"]
        narrative = row["narrative"]
        lines.append(
            " & ".join(
                (
                    row["market"],
                    _direction_macro(baseline["direction"]),
                    baseline["path_share_label"].replace("%", r"\%"),
                    _tex_move(baseline["mean_move_label"]),
                    _direction_macro(narrative["direction"]),
                    narrative["path_share_label"].replace("%", r"\%"),
                    _tex_move(narrative["mean_move_label"]),
                    _comparison_macro(row["comparison"]),
                )
            )
            + r" \\"
        )
    return "\n".join(lines)


def render_dollar_liquidity_fragments(
    document: Mapping[str, Any],
) -> dict[str, str]:
    """Render the three venue-specific TeX views from one numerical object."""
    rows = _table_rows(document)
    fmts_full = rf"""\begin{{table}}[H]
\setlength{{\intextsep}}{{1pt}}
\setlength{{\abovecaptionskip}}{{4pt}}
\centering
\caption{{Displayed dominant-support dollar-liquidity case (24 March 2016). The narrative
describes tight dollar funding, strong dollar demand, and de-risking across
equities, credit, commodities, and leveraged carry.}}
\label{{tab:nl-fixed-start-full}}
\scriptsize
\setlength{{\tabcolsep}}{{2.1pt}}
\renewcommand{{\arraystretch}}{{1.00}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllllll@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{Start-only deck}} & \multicolumn{{3}}{{c}}{{Dollar-liquidity deck}} & \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Market & Direction & Majority share & Mean move & Direction & Majority share & Mean move & Narrative vs. start-only \\
\midrule
{rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    fmts_short = rf"""\begin{{table}}[H]
\setlength{{\intextsep}}{{1pt}}
\setlength{{\abovecaptionskip}}{{2pt}}
\centering
\caption{{Dominant-support dollar-liquidity scenario shift (24 March 2016).}}
\label{{tab:nl-fixed-start}}
\scriptsize
\setlength{{\tabcolsep}}{{2.1pt}}
\renewcommand{{\arraystretch}}{{0.54}}
\resizebox{{0.84\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllllll@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{Start-only deck}} & \multicolumn{{3}}{{c}}{{Dollar-liquidity deck}} & \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Market & Direction & Majority share & Mean move & Direction & Majority share & Mean move & Narrative vs. start-only \\
\midrule
{rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    icaif = rf"""\begin{{table*}}[t]
\centering
\caption{{Fixed-start dollar-liquidity case (24 March 2016). The narrative describes tight dollar funding, strong dollar demand, and de-risking across equities, credit, commodities, and leveraged carry.}}
\label{{tab:nl-fixed-start}}
\scriptsize
\setlength{{\tabcolsep}}{{2.1pt}}
\renewcommand{{\arraystretch}}{{0.70}}
\resizebox{{0.78\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllllll@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{Start-only support}} & \multicolumn{{3}}{{c}}{{Dollar-liquidity support}} & \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Market & View & Dominant paths & Mean move & View & Dominant paths & Mean move & Narrative vs. start-only \\
\midrule
{rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    return {
        "fmts-dollar-liquidity-full.tex": fmts_full,
        "fmts-dollar-liquidity-short.tex": fmts_short,
        "icaif-dollar-liquidity.tex": icaif,
    }


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DollarLiquidityExportError("dollar-liquidity JSON is invalid") from error


def _csv_bytes(document: Mapping[str, Any]) -> bytes:
    columns = (
        "market_id",
        "market",
        "start_date",
        "start_only_direction",
        "start_only_majority_share",
        "start_only_mean_terminal_delta",
        "start_only_sigma",
        "start_only_mean_move",
        "narrative_direction",
        "narrative_majority_share",
        "narrative_mean_terminal_delta",
        "narrative_sigma",
        "narrative_mean_move",
        "comparison",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in document["rows"]:
        baseline = row["start_only"]
        narrative = row["narrative"]
        writer.writerow(
            {
                "market_id": row["market_id"],
                "market": row["market"],
                "start_date": document["start_date"],
                "start_only_direction": baseline["direction"],
                "start_only_majority_share": baseline["majority_share"],
                "start_only_mean_terminal_delta": baseline["mean_terminal_delta"],
                "start_only_sigma": baseline["sigma"],
                "start_only_mean_move": baseline["mean_move_label"],
                "narrative_direction": narrative["direction"],
                "narrative_majority_share": narrative["majority_share"],
                "narrative_mean_terminal_delta": narrative["mean_terminal_delta"],
                "narrative_sigma": narrative["sigma"],
                "narrative_mean_move": narrative["mean_move_label"],
                "comparison": row["comparison"]["label"],
            }
        )
    return output.getvalue().encode("utf-8")


def _golden_payload(name: str) -> bytes:
    repository = Path(__file__).resolve().parents[3]
    path = repository / "paper_assets" / "tables" / "golden" / name
    return _verify_file(
        path, GOLDEN_IDENTITIES[name], label=f"{name} golden"
    ).read_bytes()


def _safe_output(output_dir: Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    for protected in (
        (repository / "paper_assets" / "tables" / "golden").resolve(),
        (repository / "results" / "frozen").resolve(),
    ):
        if output == protected or protected in output.parents:
            raise DollarLiquidityExportError("refusing to overwrite protected evidence")
    return output


def export_dollar_liquidity_tables(
    narrative_report: Path,
    narrative_arrays: Path,
    start_only_report: Path,
    start_only_arrays: Path,
    comparison_report: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Export JSON, CSV, and exact ICAIF/FMTS views from frozen evidence."""
    document = build_dollar_liquidity_document(
        narrative_report,
        narrative_arrays,
        start_only_report,
        start_only_arrays,
        comparison_report,
    )
    output = _safe_output(output_dir)
    paths = {
        "json": output / "dollar-liquidity.json",
        "csv": output / "dollar-liquidity.csv",
        "fmts_full_tex": output / "fmts-dollar-liquidity-full.tex",
        "fmts_short_tex": output / "fmts-dollar-liquidity-short.tex",
        "icaif_tex": output / "icaif-dollar-liquidity.tex",
    }
    fragments = render_dollar_liquidity_fragments(document)
    payloads = {
        paths["json"]: _json_bytes(document),
        paths["csv"]: _csv_bytes(document),
    }
    for key, name in (
        ("fmts_full_tex", "fmts-dollar-liquidity-full.tex"),
        ("fmts_short_tex", "fmts-dollar-liquidity-short.tex"),
        ("icaif_tex", "icaif-dollar-liquidity.tex"),
    ):
        payload = fragments[name].encode("utf-8")
        if payload != _golden_payload(name):
            raise DollarLiquidityExportError(f"rendered {name} drifted from its golden")
        payloads[paths[key]] = payload

    # This transaction is imported lazily so the pure calculations remain
    # independent of benchmark and multi-seed modules.
    try:
        from .tables import BenchmarkExportError, _write_output_batch

        _write_output_batch(payloads)
    except BenchmarkExportError as error:
        raise DollarLiquidityExportError(str(error)) from error
    return paths
