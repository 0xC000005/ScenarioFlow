"""Regression tests for the frozen dollar-liquidity paper table."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scenarioflow.artifacts import ArtifactResolver
from scenarioflow.paper.dollar_liquidity import (
    DollarLiquidityExportError,
    export_dollar_liquidity_tables,
    select_sparse_support,
    summarize_sparse_deck,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "results" / "frozen" / "dollar-liquidity"
GOLDEN = ROOT / "paper_assets" / "tables" / "golden"
ARTIFACT_MANIFEST = ROOT / "artifacts" / "manifest-v1.tsv"

NARRATIVE_ID = "evidence.nl.dollar-liquidity-cohesive"
START_ONLY_ID = "evidence.nl.dollar-liquidity-start-only"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_artifact_paths() -> tuple[Path, Path]:
    """Find verified evidence arrays without making the public test suite fetch them."""
    resolver = ArtifactResolver.from_tsv(ARTIFACT_MANIFEST)
    cache = ROOT / "artifacts" / "cache"
    paths = (
        cache / resolver.record(NARRATIVE_ID).relative_path,
        cache / resolver.record(START_ONLY_ID).relative_path,
    )
    if not all(path.is_file() for path in paths):
        pytest.skip(
            "fetch the evidence profile to run the exact dollar-table integration"
        )
    return paths


def _export(tmp_path: Path) -> tuple[dict[str, Path], dict[str, object]]:
    narrative_arrays, start_only_arrays = _local_artifact_paths()
    outputs = export_dollar_liquidity_tables(
        REPORT_ROOT / "narrative-report.json",
        narrative_arrays,
        REPORT_ROOT / "start-only-report.json",
        start_only_arrays,
        REPORT_ROOT / "fixed-start-comparison.json",
        tmp_path,
    )
    document = json.loads(outputs["json"].read_text(encoding="utf-8"))
    return outputs, document


def _artifact_root(tmp_path: Path) -> Path:
    resolver = ArtifactResolver.from_tsv(ARTIFACT_MANIFEST)
    sources = dict(zip((NARRATIVE_ID, START_ONLY_ID), _local_artifact_paths()))
    root = tmp_path / "artifacts"
    for artifact_id, source in sources.items():
        destination = root / resolver.record(artifact_id).relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    return root


def test_sparse_support_keeps_at_most_three_components_and_renormalizes() -> None:
    """The paper view uses the largest components up to its top-three limit."""
    selected = select_sparse_support(
        [
            {"component_no": 0, "weight": 0.14},
            {"component_no": 1, "weight": 0.13},
            {"component_no": 2, "weight": 0.12},
            {"component_no": 3, "weight": 0.61},
        ],
        max_components=3,
        min_cumulative_weight=0.90,
    )

    assert [row["component_no"] for row in selected] == [0, 1, 3]
    assert sum(float(row["sparse_weight"]) for row in selected) == pytest.approx(1.0)


def test_sparse_deck_uses_selected_paths_and_the_sixty_percent_direction_rule(
    tmp_path: Path,
) -> None:
    """Sign shares come from selected paths; a 50/50 split remains mixed."""
    arrays = tmp_path / "deck.npz"
    states = np.zeros((1, 4, 2, 39), dtype=np.float32)
    states[0, :, -1, 25] = np.array([2.0, -1.0, 4.0, 3.0], dtype=np.float32)
    np.savez(
        arrays,
        generated_states=states,
        requested_raw=np.zeros((1, 39), dtype=np.float32),
        rollout_component_variant_index=np.zeros(4, dtype=np.int64),
        rollout_component_window_index=np.arange(4, dtype=np.int64),
        rollout_component_weight=np.array([0.14, 0.13, 0.12, 0.61]),
        rollout_component_sample_count=np.ones(4, dtype=np.int64),
    )
    report = {
        "selected_start_state": {"variant_index": 0},
        "variant_rows": [
            {
                "is_operational": True,
                "start_calendar": {"calendar_end_date": "2016-03-24"},
            }
        ],
    }

    deck = summarize_sparse_deck(report, arrays)
    spx = next(row for row in deck["markets"] if row["market_id"] == "SPX")

    assert deck["selected_component_count"] == 3
    assert deck["selected_path_count"] == 3
    assert spx["probability_up"] == pytest.approx(2 / 3)
    assert spx["direction"] == "up"
    assert spx["path_share_label"] == "67% up"


def test_golden_dollar_fragments_have_frozen_source_identities() -> None:
    """The two FMTS copies and semantic ICAIF extraction must remain exact."""
    assert {path.name for path in GOLDEN.glob("*dollar-liquidity*.tex")} == set(
        GOLDEN_IDENTITIES
    )
    for name, (size, digest) in GOLDEN_IDENTITIES.items():
        path = GOLDEN / name
        assert path.stat().st_size == size
        assert _sha256(path) == digest


def test_exact_evidence_exports_all_eleven_markets_and_three_golden_views(
    tmp_path: Path,
) -> None:
    """The frozen arrays and reports must reconstruct every displayed paper value."""
    first, document = _export(tmp_path / "first")
    second, second_document = _export(tmp_path / "second")

    assert document == second_document
    assert document["schema_version"] == "scenarioflow.dollar-liquidity-table.v1"
    assert document["start_date"] == "2016-03-24"
    assert document["market_order"] == [
        "IV_SURFACE",
        "SPX",
        "US2Y",
        "US10Y",
        "BBB_OAS",
        "AAA_OAS",
        "USDJPY",
        "DXY",
        "GOLD",
        "CRUDE_OIL",
        "VIX",
    ]
    assert document["decks"]["start_only"]["selected_path_count"] == 318
    assert document["decks"]["narrative"]["selected_path_count"] == 153
    assert len(document["rows"]) == 11

    rows = {row["market_id"]: row for row in document["rows"]}
    assert rows["IV_SURFACE"]["narrative"]["path_share_label"] == "78% down"
    assert rows["SPX"]["narrative"]["mean_move_label"] == "+17 pts / +0.6σ"
    assert rows["BBB_OAS"]["comparison"]["label"] == "Less down than baseline"
    assert rows["DXY"]["comparison"]["label"] == "More up than baseline"

    with first["csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["market_id"] for row in csv_rows] == document["market_order"]

    for key, name in {
        "fmts_full_tex": "fmts-dollar-liquidity-full.tex",
        "fmts_short_tex": "fmts-dollar-liquidity-short.tex",
        "icaif_tex": "icaif-dollar-liquidity.tex",
    }.items():
        assert first[key].read_bytes() == (GOLDEN / name).read_bytes()
        assert second[key].read_bytes() == first[key].read_bytes()
    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["csv"].read_bytes() == second["csv"].read_bytes()


def test_exporter_import_has_no_demo_or_language_model_dependencies() -> None:
    """Paper-table reproduction must stay independent of both demo backends."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import scenarioflow.paper.dollar_liquidity; "
                "print(json.dumps(sorted(sys.modules)))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    modules = set(json.loads(completed.stdout))
    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.")
        for forbidden in ("gradio", "openai", "pandas")
        for name in modules
    )
    assert not any(name.startswith("experiments.") for name in modules)


def test_export_rejects_changed_report_authority(tmp_path: Path) -> None:
    """A locally edited report must not silently produce a paper table."""
    narrative_arrays, start_only_arrays = _local_artifact_paths()
    edited = tmp_path / "narrative.json"
    edited.write_bytes((REPORT_ROOT / "narrative-report.json").read_bytes() + b"\n")

    with pytest.raises(DollarLiquidityExportError, match="authority"):
        export_dollar_liquidity_tables(
            edited,
            narrative_arrays,
            REPORT_ROOT / "start-only-report.json",
            start_only_arrays,
            REPORT_ROOT / "fixed-start-comparison.json",
            tmp_path / "output",
        )


def test_native_cli_exports_the_dollar_liquidity_table(tmp_path: Path) -> None:
    """The reader-facing command resolves both arrays by semantic artifact ID."""
    artifact_root = _artifact_root(tmp_path)
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.cli",
            "paper",
            "export-dollar-liquidity",
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "dollar-liquidity.json").is_file()
    assert (output / "fmts-dollar-liquidity-short.tex").read_bytes() == (
        GOLDEN / "fmts-dollar-liquidity-short.tex"
    ).read_bytes()
