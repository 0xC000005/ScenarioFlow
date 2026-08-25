"""Read-only Phase-0 comparator for the FMTS reproduction gate.

This utility deliberately does not run training or evaluation.  Callers supply
an explicit canonical/replay mapping, including the non-result contract fields
that historical baseline manifests do not record (reconstruction, metric, and
threshold definitions).  It reads those artifacts and writes only the two
report files under an isolated output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExplicitInputMappingRequiredError(ValueError):
    """Raised when a report would otherwise guess its source artifacts."""


# Kept in sync with table_operational_red_thresholds.tex and the existing
# G3 sensitivity implementation.  These are reporting rules, not tuned gates.
TABLE_THRESHOLD_RULES: dict[str, dict[str, float | str]] = {
    "Cov90": {"kind": "band", "lo": 0.75, "hi": 0.98},
    "CalErr": {"kind": "upper", "hi": 0.20},
    "Level KS": {"kind": "upper", "hi": 0.20},
    "Width rho": {"kind": "lower", "lo": 0.20},
    "MR ratio": {"kind": "band", "lo": 0.25, "hi": 1.75},
    "Econ ratio": {"kind": "band", "lo": 0.75, "hi": 1.30},
    "Jump KS": {"kind": "upper", "hi": 0.50},
    "Kurtosis": {"kind": "band", "lo": 0.50, "hi": 1.50},
    "Corr score": {"kind": "lower", "lo": 0.30},
    "Factor KS": {"kind": "upper", "hi": 0.15},
    "Tail q99": {"kind": "band", "lo": 0.50, "hi": 1.50},
}


METRIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CRPS", ("panel_b_crps", "overall")),
    ("IV CRPS", ("panel_b_crps", "iv_crps")),
    ("Factor CRPS", ("panel_b_crps", "factor_crps")),
    ("Energy Score", ("panel_b_energy_score", "overall")),
    ("Variogram", ("panel_b_variogram_score", "variogram_score")),
    ("Corr score", ("panel_b_correlation", "corr_score")),
    ("Factor KS", ("panel_b_factor_ks", "mean_statistic")),
    ("Tail q99", ("joint_panel_quality", "factor_tail_q99_ratio_median")),
    ("Cov90", ("current_panel_iv_metrics", "cov90")),
    ("CalErr", ("current_panel_iv_metrics", "calibration_error")),
    ("Level KS", ("current_panel_iv_metrics", "level_ks_median")),
    ("Median dev", ("current_panel_iv_metrics", "median_abs_dev")),
    ("Width rho", ("current_panel_iv_metrics", "width_rho")),
    ("MR ratio", ("current_panel_iv_metrics", "mr_ratio")),
    ("Econ ratio", ("current_panel_iv_metrics", "economic_ratio")),
    ("Jump KS", ("current_panel_iv_metrics", "jump_ks")),
    ("Kurtosis", ("current_panel_iv_metrics", "kurtosis_ratio")),
    ("ACF corr", ("current_panel_iv_metrics", "acf_corr")),
)

_LAYER_B_RELATIVE = {
    "CRPS",
    "IV CRPS",
    "Factor CRPS",
    "Energy Score",
    "Variogram",
}
_LAYER_B_ABSOLUTE = {
    "Cov90",
    "CalErr",
    "Level KS",
    "Width rho",
    "MR ratio",
    "Econ ratio",
    "Jump KS",
    "Kurtosis",
    "ACF corr",
    "Corr score",
    "Factor KS",
    "Median dev",
    "Tail q99",
}
_MISSING = object()


def _read_json(path_value: object, description: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path_value, str) or not path_value:
        raise ExplicitInputMappingRequiredError(
            f"{description} must be an explicit JSON path"
        )
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return path, payload


def _side(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise ExplicitInputMappingRequiredError(
            f"mapping.{name} must be supplied explicitly"
        )
    for key in ("result", "manifest", "contract"):
        if key not in value:
            raise ExplicitInputMappingRequiredError(f"mapping.{name}.{key} is required")
    if not isinstance(value["contract"], Mapping) or not value["contract"]:
        raise ExplicitInputMappingRequiredError(
            f"mapping.{name}.contract must be a nonempty object"
        )
    return value


def _candidate_result(
    result_document: Mapping[str, Any], candidate: str
) -> Mapping[str, Any]:
    baselines = result_document.get("baselines")
    if isinstance(baselines, Mapping):
        payload = baselines.get(candidate)
        if not isinstance(payload, Mapping):
            raise KeyError(f"candidate {candidate!r} is absent from summary baselines")
        return payload
    return result_document


def _path_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _relative_difference(canonical: float, replay: float) -> float:
    difference = abs(replay - canonical)
    if canonical == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / abs(canonical)


def _threshold_status(
    value: float | None, rule: Mapping[str, Any] | None
) -> str | None:
    if value is None or rule is None:
        return None
    kind = rule["kind"]
    if kind == "upper":
        red = value > float(rule["hi"])
    elif kind == "lower":
        red = value < float(rule["lo"])
    elif kind == "band":
        red = value < float(rule["lo"]) or value > float(rule["hi"])
    else:
        raise ValueError(f"unsupported threshold kind: {kind!r}")
    return "fail" if red else "pass"


def _layer_tolerance(layer: str, device: str, metric: str) -> dict[str, Any]:
    if layer == "A":
        if device == "cpu":
            return {"kind": "isclose", "rtol": 1e-6, "atol": 1e-8}
        return {"kind": "isclose", "rtol": 1e-4, "atol": 1e-6}
    if metric in _LAYER_B_RELATIVE:
        return {"kind": "relative", "value": 0.02}
    if metric in _LAYER_B_ABSOLUTE:
        return {"kind": "absolute", "value": 0.03}
    raise ValueError(f"Layer-B tolerance is missing for table metric: {metric}")


def _metric_pass(
    canonical: float | None, replay: float | None, tolerance: Mapping[str, Any]
) -> bool:
    if canonical is None or replay is None:
        return False
    kind = tolerance["kind"]
    if kind == "isclose":
        return math.isclose(
            canonical,
            replay,
            rel_tol=float(tolerance["rtol"]),
            abs_tol=float(tolerance["atol"]),
        )
    if kind == "relative":
        return _relative_difference(canonical, replay) <= float(tolerance["value"])
    if kind == "absolute":
        return abs(replay - canonical) <= float(tolerance["value"])
    return False


def _invariant(
    name: str, canonical: object, replay: object, *, expected: object = _MISSING
) -> dict[str, Any]:
    values_present = canonical is not _MISSING and replay is not _MISSING
    matches = values_present and canonical == replay
    expected_matches = expected is _MISSING or (
        canonical == expected and replay == expected
    )
    return {
        "name": name,
        "canonical": None if canonical is _MISSING else canonical,
        "replay": None if replay is _MISSING else replay,
        "expected": None if expected is _MISSING else expected,
        "passed": bool(matches and expected_matches),
    }


def _split_indices(manifest: Mapping[str, Any]) -> object:
    metadata = manifest.get("metadata")
    keys = (
        "train_indices_start",
        "train_indices_end",
        "val_indices_start",
        "val_indices_end",
    )
    if not isinstance(metadata, Mapping) or any(key not in metadata for key in keys):
        return _MISSING
    return {key: metadata[key] for key in keys}


def _preprocessing(manifest: Mapping[str, Any]) -> object:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        return _MISSING
    keys = (
        "source_columns",
        "positive_level_policy",
        "iv_transform",
        "iv_lower_bound",
        "iv_upper_bound",
        "cleaning_report",
    )
    if any(key not in metadata for key in keys):
        return _MISSING
    return {key: metadata[key] for key in keys}


def _structural_invariants(
    canonical_result: Mapping[str, Any],
    replay_result: Mapping[str, Any],
    canonical_manifest: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    replay_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical_states = canonical_manifest.get("state_specs", _MISSING)
    replay_states = replay_manifest.get("state_specs", _MISSING)
    canonical_result_states = canonical_result.get("state_specs", _MISSING)
    replay_result_states = replay_result.get("state_specs", _MISSING)
    canonical_variables = (
        len(canonical_states) if isinstance(canonical_states, list) else _MISSING
    )
    replay_variables = (
        len(replay_states) if isinstance(replay_states, list) else _MISSING
    )
    canonical_windows = canonical_result.get("n_windows", _MISSING)
    replay_windows = replay_result.get("n_windows", _MISSING)
    canonical_samples = canonical_result.get("n_samples", _MISSING)
    replay_samples = replay_result.get("n_samples", _MISSING)
    canonical_joint = canonical_result.get("joint_panel_quality", {})
    replay_joint = replay_result.get("joint_panel_quality", {})
    if not isinstance(canonical_joint, Mapping):
        canonical_joint = {}
    if not isinstance(replay_joint, Mapping):
        replay_joint = {}

    invariants = [
        _invariant("n_variables", canonical_variables, replay_variables, expected=39),
        _invariant("n_windows", canonical_windows, replay_windows, expected=441),
        _invariant("n_samples", canonical_samples, replay_samples, expected=64),
        _invariant(
            "n_windows_in_joint_quality",
            canonical_joint.get("n_windows", _MISSING),
            replay_joint.get("n_windows", _MISSING),
            expected=441,
        ),
        _invariant(
            "n_samples_in_joint_quality",
            canonical_joint.get("n_samples", _MISSING),
            replay_joint.get("n_samples", _MISSING),
            expected=64,
        ),
        _invariant(
            "forecast_horizon",
            canonical_joint.get("future_len", _MISSING),
            replay_joint.get("future_len", _MISSING),
            expected=30,
        ),
        _invariant(
            "history_horizon",
            canonical_contract.get("history_horizon", _MISSING),
            replay_contract.get("history_horizon", _MISSING),
            expected=30,
        ),
        _invariant(
            "split_indices",
            _split_indices(canonical_manifest),
            _split_indices(replay_manifest),
        ),
        _invariant("state_specs", canonical_states, replay_states),
        _invariant("result_state_specs", canonical_result_states, replay_result_states),
        _invariant(
            "preprocessing",
            _preprocessing(canonical_manifest),
            _preprocessing(replay_manifest),
        ),
        _invariant(
            "reconstruction",
            canonical_contract.get("reconstruction", _MISSING),
            replay_contract.get("reconstruction", _MISSING),
        ),
        _invariant(
            "metric_definitions",
            canonical_contract.get("metric_definitions", _MISSING),
            replay_contract.get("metric_definitions", _MISSING),
        ),
        _invariant(
            "threshold_definitions",
            canonical_contract.get("threshold_definitions", _MISSING),
            replay_contract.get("threshold_definitions", _MISSING),
            expected=TABLE_THRESHOLD_RULES,
        ),
    ]
    return invariants


def _metric_rows(
    canonical_result: Mapping[str, Any],
    replay_result: Mapping[str, Any],
    layer: str,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in METRIC_PATHS:
        canonical = _numeric(_path_value(canonical_result, path))
        replay = _numeric(_path_value(replay_result, path))
        tolerance = _layer_tolerance(layer, device, name)
        rule = TABLE_THRESHOLD_RULES.get(name)
        canonical_classification = _threshold_status(canonical, rule)
        replay_classification = _threshold_status(replay, rule)
        classification_matches = canonical_classification == replay_classification
        rows.append(
            {
                "name": name,
                "source_path": ".".join(path),
                "canonical": canonical,
                "replay": replay,
                "absolute_difference": (
                    None
                    if canonical is None or replay is None
                    else abs(replay - canonical)
                ),
                "relative_difference": (
                    None
                    if canonical is None or replay is None
                    else _relative_difference(canonical, replay)
                ),
                "tolerance": tolerance,
                "numeric_pass": _metric_pass(canonical, replay, tolerance),
                "threshold_classification": {
                    "rule": rule,
                    "canonical": canonical_classification,
                    "replay": replay_classification,
                    "matches": classification_matches,
                },
                "passed": _metric_pass(canonical, replay, tolerance)
                and classification_matches,
            }
        )
    return rows


def _assert_output_isolated(output_dir: Path, source_paths: Sequence[Path]) -> None:
    resolved_output = output_dir.expanduser().resolve()
    for source_path in source_paths:
        source_parent = source_path.parent.resolve()
        if (
            resolved_output == source_parent
            or source_parent in resolved_output.parents
            or resolved_output in source_parent.parents
        ):
            raise ValueError(
                "output_dir must be isolated from every source artifact directory: "
                f"{resolved_output} overlaps {source_parent}"
            )


def compare_reproduction_runs(
    *, mappings: Sequence[Mapping[str, Any]], layer: str, device: str
) -> dict[str, Any]:
    """Compare explicit canonical/replay artifact mappings without writing files."""
    if not mappings:
        raise ExplicitInputMappingRequiredError(
            "at least one explicit input mapping is required"
        )
    layer = layer.upper()
    device = device.lower()
    if layer not in {"A", "B"}:
        raise ValueError("layer must be 'A' or 'B'")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")

    candidates: list[dict[str, Any]] = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            raise ExplicitInputMappingRequiredError(
                "each input mapping must be an object"
            )
        candidate = mapping.get("candidate")
        if not isinstance(candidate, str) or not candidate:
            raise ExplicitInputMappingRequiredError("mapping.candidate is required")
        canonical_side = _side(mapping, "canonical")
        replay_side = _side(mapping, "replay")
        canonical_result_path, canonical_result_document = _read_json(
            canonical_side["result"], f"{candidate} canonical result"
        )
        replay_result_path, replay_result_document = _read_json(
            replay_side["result"], f"{candidate} replay result"
        )
        canonical_manifest_path, canonical_manifest = _read_json(
            canonical_side["manifest"], f"{candidate} canonical manifest"
        )
        replay_manifest_path, replay_manifest = _read_json(
            replay_side["manifest"], f"{candidate} replay manifest"
        )
        canonical_result = _candidate_result(canonical_result_document, candidate)
        replay_result = _candidate_result(replay_result_document, candidate)
        invariants = _structural_invariants(
            canonical_result,
            replay_result,
            canonical_manifest,
            replay_manifest,
            canonical_side["contract"],
            replay_side["contract"],
        )
        metrics = _metric_rows(canonical_result, replay_result, layer, device)
        structural_invariants_pass = all(item["passed"] for item in invariants)
        metrics_pass = all(item["passed"] for item in metrics)
        candidates.append(
            {
                "candidate": candidate,
                "source_artifacts": {
                    "canonical_result": str(canonical_result_path),
                    "replay_result": str(replay_result_path),
                    "canonical_manifest": str(canonical_manifest_path),
                    "replay_manifest": str(replay_manifest_path),
                },
                "structural_invariants": invariants,
                "structural_invariants_pass": structural_invariants_pass,
                "metrics": metrics,
                "metrics_pass": metrics_pass,
                "passed": structural_invariants_pass and metrics_pass,
            }
        )
    return {
        "schema_version": 1,
        "layer": layer,
        "device": device,
        "threshold_contract": TABLE_THRESHOLD_RULES,
        "candidates": candidates,
        "overall_pass": all(candidate["passed"] for candidate in candidates),
    }


def _markdown_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, sort_keys=True)}`"
    return str(value)


def render_reproduction_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON result in a compact, reviewer-readable Markdown form."""
    lines = [
        "# FMTS Phase-0 reproduction report",
        "",
        f"- Layer: `{report['layer']}`",
        f"- Device: `{report['device']}`",
        f"- Overall gate: **{'PASS' if report['overall_pass'] else 'FAIL'}**",
        "",
    ]
    for candidate in report["candidates"]:
        lines.extend(
            [
                f"## {candidate['candidate']}",
                "",
                f"Structural gate: **{'PASS' if candidate['structural_invariants_pass'] else 'FAIL'}**  "
                f"Metric gate: **{'PASS' if candidate['metrics_pass'] else 'FAIL'}**",
                "",
                "### Structural invariants",
                "",
                "| Invariant | Canonical | Replay | Expected | Pass |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in candidate["structural_invariants"]:
            lines.append(
                "| {name} | {canonical} | {replay} | {expected} | {passed} |".format(
                    name=item["name"],
                    canonical=_markdown_value(item["canonical"]),
                    replay=_markdown_value(item["replay"]),
                    expected=_markdown_value(item["expected"]),
                    passed="PASS" if item["passed"] else "FAIL",
                )
            )
        lines.extend(
            [
                "",
                "### Metrics",
                "",
                "| Metric | Canonical | Replay | Abs. diff | Rel. diff | Tolerance | Classification | Pass |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for metric in candidate["metrics"]:
            classification = metric["threshold_classification"]
            threshold = (
                "n/a"
                if classification["rule"] is None
                else f"{classification['canonical']} → {classification['replay']}"
            )
            lines.append(
                "| {name} | {canonical} | {replay} | {absolute} | {relative} | "
                "`{tolerance}` | {classification} | {passed} |".format(
                    name=metric["name"],
                    canonical=_markdown_value(metric["canonical"]),
                    replay=_markdown_value(metric["replay"]),
                    absolute=_markdown_value(metric["absolute_difference"]),
                    relative=_markdown_value(metric["relative_difference"]),
                    tolerance=json.dumps(metric["tolerance"], sort_keys=True),
                    classification=threshold,
                    passed="PASS" if metric["passed"] else "FAIL",
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_reproduction_report(
    *,
    mappings: Sequence[Mapping[str, Any]],
    layer: str,
    device: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the report JSON and Markdown under ``output_dir`` only."""
    report = compare_reproduction_runs(mappings=mappings, layer=layer, device=device)
    source_paths = [
        Path(path)
        for candidate in report["candidates"]
        for path in candidate["source_artifacts"].values()
    ]
    output_path = Path(output_dir)
    _assert_output_isolated(output_path, source_paths)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "reproduction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "reproduction_report.md").write_text(
        render_reproduction_markdown(report), encoding="utf-8"
    )
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mappings", required=True, help="JSON object containing a 'mappings' list"
    )
    parser.add_argument("--layer", required=True, choices=("A", "B", "a", "b"))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    _, mapping_document = _read_json(args.mappings, "mappings")
    mappings = mapping_document.get("mappings")
    if not isinstance(mappings, list):
        raise ExplicitInputMappingRequiredError(
            "mappings JSON must contain a 'mappings' list"
        )
    report = write_reproduction_report(
        mappings=mappings,
        layer=args.layer,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"reproduction gate: {'PASS' if report['overall_pass'] else 'FAIL'}")


if __name__ == "__main__":
    _main()
