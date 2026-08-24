"""Checksum-bound validator for the immutable published Gaussian GARCH-CCC row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_SOURCE_REVISION = "da91172a5449aa91a7f5a44c52b06f94dc5d4bf0"
EXPECTED_SOURCE_BLOB = "bca996a09b40795db04df57d6eb726bb42c6dbc6"
SOURCE_TAG = "refs/tags/provenance/garch-ccc-gaussian-da91172a"
FROZEN_SUMMARY_SHA256 = (
    "d1431f0dc94974301f0bd8642666ba70c8d093bb06cad8a43311f8900b33a771"
)

EXPECTED_HEADLINES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("CRPS", ("panel_b_crps", "overall"), 3.332548859024846),
    ("Energy Score", ("panel_b_energy_score", "overall"), 3.9032458813508333),
    ("Variogram", ("panel_b_variogram_score", "variogram_score"), 0.20482109487056732),
)

_DISTRIBUTIONAL_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CRPS", ("panel_b_crps", "overall")),
    ("IV CRPS", ("panel_b_crps", "iv_crps")),
    ("Factor CRPS", ("panel_b_crps", "factor_crps")),
    ("Energy Score", ("panel_b_energy_score", "overall")),
    ("Variogram", ("panel_b_variogram_score", "variogram_score")),
    ("Corr score", ("panel_b_correlation", "corr_score")),
    ("Factor KS", ("panel_b_factor_ks", "mean_statistic")),
    ("Tail q99", ("joint_panel_quality", "factor_tail_q99_ratio_median")),
)
_RISK_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _result_from_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if document.get("baseline_name") == "garch_ccc":
        return document
    baselines = document.get("baselines")
    if isinstance(baselines, Mapping) and isinstance(
        baselines.get("garch_ccc"), Mapping
    ):
        return baselines["garch_ccc"]
    result = document.get("result")
    if isinstance(result, Mapping) and result.get("baseline_name") == "garch_ccc":
        return result
    raise ValueError("frozen result document has no garch_ccc row")


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError("missing published metric path: " + ".".join(path))
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError("published metric is not numeric: " + ".".join(path))
    value = float(current)
    if not math.isfinite(value):
        raise ValueError("published metric is not finite: " + ".".join(path))
    return value


def _strip_latex_number(value: str) -> str:
    cleaned = value.strip()
    while True:
        wrapped = re.fullmatch(r"\\(?:textcolor\{[^{}]+\}|textbf)\{([^{}]+)\}", cleaned)
        if wrapped is None:
            return cleaned
        cleaned = wrapped.group(1).strip()


def _published_rows(table_text: str) -> tuple[list[str], list[str]]:
    rows: list[list[str]] = []
    for line in table_text.splitlines():
        if not line.strip().startswith("GARCH-CCC &"):
            continue
        body = re.sub(r"\\\\\s*$", "", line.strip())
        rows.append([_strip_latex_number(cell) for cell in body.split("&")[1:]])
    if len(rows) != 2 or len(rows[0]) != 8 or len(rows[1]) != 10:
        raise ValueError(
            "published table must contain the two canonical GARCH-CCC rows"
        )
    return rows[0], rows[1]


def _assert_isolated(output_dir: Path, sources: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for source in sources:
        parent = source.resolve().parent
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError("output directory must be isolated from source artifacts")


def _require(label: str, actual: object, expected: object) -> dict[str, Any]:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")
    return {"name": label, "actual": actual, "expected": expected, "passed": True}


def validate_published_row(
    *,
    result_path: str | Path,
    replay_manifest_path: str | Path,
    table_path: str | Path,
    output_dir: str | Path,
    expected_sha256: Sequence[str],
) -> dict[str, Any]:
    """Validate pinned bytes, provenance, structure, and surviving metric precision."""
    sources = tuple(
        Path(value).expanduser().resolve()
        for value in (result_path, replay_manifest_path, table_path)
    )
    if len(expected_sha256) != 3:
        raise ValueError("expected_sha256 must pin result, manifest, and table")
    if any(not source.is_file() for source in sources):
        raise FileNotFoundError("all three frozen GARCH contract sources are required")
    output = Path(output_dir).expanduser().resolve()
    _assert_isolated(output, sources)
    actual_hashes = tuple(_sha256(path) for path in sources)
    if actual_hashes != tuple(expected_sha256):
        raise ValueError("frozen GARCH contract checksum mismatch")

    result = _result_from_document(_read_object(sources[0]))
    manifest = _read_object(sources[1])
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = manifest
    checks = [
        _require("baseline name", result.get("baseline_name"), "garch_ccc"),
        _require("result mode", result.get("mode"), "current_realvix_39_state_panel"),
        _require(
            "manifest mode", manifest.get("mode"), "current_realvix_39_state_panel"
        ),
        _require("evaluation seed", manifest.get("seed"), 772),
        _require("result windows", result.get("n_windows"), 441),
        _require("validation windows", manifest.get("n_val_windows"), 441),
        _require("training windows", manifest.get("n_train_windows"), 4010),
        _require("result deck size", result.get("n_samples"), 64),
        _require("manifest deck size", manifest.get("n_samples"), 64),
        _require("result state count", len(result.get("state_specs", [])), 39),
        _require("manifest state count", len(manifest.get("state_specs", [])), 39),
        _require(
            "matching state specs",
            result.get("state_specs"),
            manifest.get("state_specs"),
        ),
        _require(
            "source revision",
            provenance.get("source_revision"),
            EXPECTED_SOURCE_REVISION,
        ),
        _require("source blob", provenance.get("source_blob"), EXPECTED_SOURCE_BLOB),
        _require("source tag", provenance.get("source_tag"), SOURCE_TAG),
    ]
    metadata = (
        manifest.get("metadata")
        if isinstance(manifest.get("metadata"), Mapping)
        else {}
    )
    checks.extend(
        [
            _require("training index start", metadata.get("train_indices_start"), 0),
            _require("training index end", metadata.get("train_indices_end"), 4009),
            _require("validation index start", metadata.get("val_indices_start"), 4010),
            _require("validation index end", metadata.get("val_indices_end"), 4450),
            _require("IV transform", metadata.get("iv_transform"), "bounded_logit"),
            _require(
                "positive-level policy",
                metadata.get("positive_level_policy"),
                "reference_based",
            ),
        ]
    )
    headline_checks = []
    for name, path, expected in EXPECTED_HEADLINES:
        actual = _path_value(result, path)
        _require(name, actual, expected)
        headline_checks.append(
            {
                "name": name,
                "path": ".".join(path),
                "expected": expected,
                "actual": actual,
                "passed": True,
            }
        )

    distributional, risk = _published_rows(sources[2].read_text(encoding="utf-8"))
    display_checks = []
    for (name, path), published in zip(
        _DISTRIBUTIONAL_COLUMNS + _RISK_COLUMNS, distributional + risk, strict=True
    ):
        actual = _path_value(result, path)
        decimals = len(published.partition(".")[2]) if "." in published else 0
        rendered = f"{actual:.{decimals}f}"
        _require(f"{name} displayed precision", rendered, published)
        display_checks.append(
            {
                "name": name,
                "published": published,
                "actual": actual,
                "rendered": rendered,
                "passed": True,
            }
        )

    report = {
        "schema_version": "scenarioflow.garch-ccc-published-row.v1",
        "candidate": "garch_ccc",
        "immutable_authority": True,
        "source_artifacts": [
            {"path": str(path), "sha256": digest}
            for path, digest in zip(sources, actual_hashes, strict=True)
        ],
        "structural_checks": checks,
        "exact_headline_checks": headline_checks,
        "display_rounded_checks": display_checks,
        "overall_pass": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "garch_ccc_contract_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    validate_published_row(
        result_path=args.result,
        replay_manifest_path=args.manifest,
        table_path=args.table,
        output_dir=args.output_dir,
        expected_sha256=(args.result_sha256, args.manifest_sha256, args.table_sha256),
    )
    print("GARCH-CCC published-row contract: PASS")


if __name__ == "__main__":
    main()
