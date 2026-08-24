"""Regression tests for frozen benchmark table exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scenarioflow.paper.tables import (
    BenchmarkExportError,
    authority_paths_from_manifest,
    export_benchmark_tables,
    render_benchmark_fragments,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_MANIFEST = ROOT / "results" / "FROZEN_AUTHORITY.tsv"
GOLDEN = ROOT / "paper_assets" / "tables" / "golden"

GOLDEN_IDENTITIES = {
    "fmts-benchmark-full.tex": (
        5_638,
        "89a2614a6c292dd472ebf373cf751bc96d0a28484cc8aefdc7dc0270ea39d5f7",
    ),
    "fmts-benchmark-short.tex": (
        4_379,
        "ba7d806d76d74c67bb2ed7104d541f1b36f831ccdda2c0b0ab75b0ca2d7ff2ed",
    ),
    "icaif-benchmark.tex": (
        4_212,
        "915f6a3cafd464b9d0e911fdb77a5530044dc09a9f29f44b485ebdddafb54943",
    ),
}

FMTS_CANDIDATES = [
    "scenarioflow",
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
    "deepvar",
    "deepvar_sni",
    "timegrad",
    "path_diffusion",
    "moirai_1_1_finetuned",
    "toto_1_0_finetuned",
]
ICAIF_CANDIDATES = [
    "scenarioflow",
    "scenarioflow_graph",
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
    "deepvar",
    "timegrad",
    "path_diffusion",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(tmp_path: Path) -> tuple[dict[str, Path], dict[str, object]]:
    authorities = authority_paths_from_manifest(AUTHORITY_MANIFEST, root=ROOT)
    outputs = export_benchmark_tables(authorities, tmp_path)
    document = json.loads(outputs["benchmark_json"].read_text(encoding="utf-8"))
    return outputs, document


def test_golden_benchmark_fragments_have_the_verified_source_identities() -> None:
    """A wrong archive copy or non-semantic ICAIF extraction must fail."""
    assert set(path.name for path in GOLDEN.glob("*-benchmark*.tex")) == set(
        GOLDEN_IDENTITIES
    )
    for name, (size, digest) in GOLDEN_IDENTITIES.items():
        path = GOLDEN / name
        assert path.stat().st_size == size
        assert _sha256(path) == digest


def test_export_matches_all_three_benchmark_goldens_and_is_deterministic(
    tmp_path: Path,
) -> None:
    """Formatter drift in any venue-specific benchmark fragment must fail."""
    first, _document = _export(tmp_path / "first")
    second, _ = _export(tmp_path / "second")

    for key, name in {
        "fmts_full_tex": "fmts-benchmark-full.tex",
        "fmts_short_tex": "fmts-benchmark-short.tex",
        "icaif_tex": "icaif-benchmark.tex",
    }.items():
        assert first[key].read_bytes() == (GOLDEN / name).read_bytes()
        assert second[key].read_bytes() == first[key].read_bytes()
    assert first["benchmark_csv"].read_bytes() == second["benchmark_csv"].read_bytes()
    assert first["benchmark_json"].read_bytes() == second["benchmark_json"].read_bytes()


def test_benchmark_tex_is_rendered_from_validated_metrics_not_copied_unchanged(
    tmp_path: Path,
) -> None:
    """Replacing the formatter with an unconditional golden-file copy must fail."""
    _outputs, document = _export(tmp_path)
    scenarioflow = next(
        row for row in document["rows"] if row["candidate_id"] == "scenarioflow"
    )
    scenarioflow["metrics"]["CRPS"] = 9.9999
    scenarioflow["fmts_metrics"]["CRPS"] = 9.9999
    fragments = render_benchmark_fragments(document)
    assert "9.9999" in fragments["fmts-benchmark-full.tex"]
    assert "9.9999" in fragments["fmts-benchmark-short.tex"]
    assert "9.9999" in fragments["icaif-benchmark.tex"]
    assert "3.1772" not in fragments["icaif-benchmark.tex"]


def test_reader_exports_preserve_order_gaussian_authority_and_toto_invalidity(
    tmp_path: Path,
) -> None:
    """Reordering rows, using a live Gaussian replay, or hiding Toto invalidity fails."""
    outputs, document = _export(tmp_path)
    assert document["schema_version"] == "scenarioflow.paper-benchmark.v1"
    assert document["candidate_order"] == {
        "fmts": FMTS_CANDIDATES,
        "icaif": ICAIF_CANDIDATES,
    }
    rows = {row["candidate_id"]: row for row in document["rows"]}
    assert rows["garch_ccc"]["authority_ids"]["fmts"].endswith(
        "checkpoint-replay.gaussian-ccc.results.json"
    )
    assert rows["garch_ccc"]["metrics"]["CRPS"] == 3.332548859024846
    assert rows["toto_1_0_finetuned"]["complete_deck_valid"] is False
    assert rows["toto_1_0_finetuned"]["metrics"]["CRPS"] is None
    assert rows["toto_1_0_finetuned"]["metrics"]["IV CRPS"] == 0.013277928253369056

    with outputs["benchmark_csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["candidate_id"] for row in csv_rows] == FMTS_CANDIDATES + [
        "scenarioflow_graph"
    ]


def test_threshold_status_arrows_rounding_and_fmts_metric_lineage_are_explicit(
    tmp_path: Path,
) -> None:
    """Changing Task-3 screens or silently relabeling the FMTS legacy ratio fails."""
    outputs, document = _export(tmp_path)
    rows = {row["candidate_id"]: row for row in document["rows"]}
    assert rows["scenarioflow"]["thresholds"]["Factor KS"] == "pass"
    assert rows["random_walk"]["thresholds"]["Factor KS"] == "fail"
    assert rows["scenarioflow"]["threshold_failure_count"] == 0
    assert rows["random_walk"]["threshold_failure_count"] == 6
    assert document["venue_metric_overrides"] == {
        "fmts": {
            "Econ ratio": [
                "current_panel_iv",
                "iv_ewma_economic_link",
                "slope_ratio",
            ]
        }
    }
    assert document["metric_lineage"]["venues"] == {
        "fmts": {
            "Econ ratio": [
                "current_panel_iv",
                "iv_ewma_economic_link",
                "slope_ratio",
            ]
        },
        "icaif": {"Econ ratio": ["current_panel_iv_metrics", "economic_ratio"]},
        "multiseed": {"Econ ratio": ["current_panel_iv_metrics", "economic_ratio"]},
    }
    assert rows["scenarioflow"]["fmts_metrics"]["Econ ratio"] == (0.9404304619239114)
    assert rows["timegrad"]["fmts_metrics"]["Econ ratio"] == 1.0440387178287187
    full = outputs["fmts_full_tex"].read_text(encoding="utf-8")
    short = outputs["fmts_short_tex"].read_text(encoding="utf-8")
    icaif = outputs["icaif_tex"].read_text(encoding="utf-8")
    assert "CRPS$\\downarrow$" in full
    assert "Corr.$\\uparrow$" in full
    assert "Tail q99$\\to1$" in full
    assert r"\textcolor{red}{0.162}" in full
    assert r"\textbf{3.1772}" in full
    assert r"\PassCount{0}" in short
    assert "0.906" in icaif
    assert "0.940" in full


def test_export_rejects_noncanonical_authority_and_protected_output(
    tmp_path: Path,
) -> None:
    """Edited evidence and attempts to overwrite shipped goldens must fail closed."""
    authorities = authority_paths_from_manifest(AUTHORITY_MANIFEST, root=ROOT)
    scenarioflow_id = next(
        key
        for key in authorities
        if key.endswith("current-panel.scenarioflow.results.json")
    )
    edited = tmp_path / "edited.json"
    edited.write_bytes(authorities[scenarioflow_id].read_bytes() + b"\n")
    changed = dict(authorities)
    changed[scenarioflow_id] = edited
    with pytest.raises(BenchmarkExportError, match="authority"):
        export_benchmark_tables(changed, tmp_path / "edited-output")
    with pytest.raises(BenchmarkExportError, match="golden|protected"):
        export_benchmark_tables(authorities, GOLDEN)


@pytest.mark.parametrize("temporary_name", ["benchmark.json.tmp", "benchmark.csv.tmp"])
def test_benchmark_export_cannot_follow_a_preexisting_temporary_symlink(
    tmp_path: Path, temporary_name: str
) -> None:
    """A predictable temporary symlink must never redirect an export write."""
    authorities = authority_paths_from_manifest(AUTHORITY_MANIFEST, root=ROOT)
    output = tmp_path / "output"
    output.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"leave this file unchanged")
    (output / temporary_name).symlink_to(victim)

    paths = export_benchmark_tables(authorities, output)

    assert victim.read_bytes() == b"leave this file unchanged"
    assert paths["benchmark_json"].is_file()
    assert paths["benchmark_csv"].is_file()


def test_table_module_and_native_cli_export_only_to_requested_directory(
    tmp_path: Path,
) -> None:
    """The table CLI must not use donor paths or mutate the shipped goldens."""
    before = {name: _sha256(GOLDEN / name) for name in GOLDEN_IDENTITIES}
    module_output = tmp_path / "module"
    module = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.paper.tables",
            "--authority",
            str(AUTHORITY_MANIFEST),
            "--output",
            str(module_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert module.returncode == 0, module.stderr

    native_output = tmp_path / "native"
    native = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.cli",
            "paper",
            "export-tables",
            "--authority",
            str(AUTHORITY_MANIFEST),
            "--output",
            str(native_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    assert (native_output / "fmts-multiseed.tex").is_file()
    assert {name: _sha256(GOLDEN / name) for name in GOLDEN_IDENTITIES} == before


def test_table_cli_reports_invalid_authority_without_a_traceback(
    tmp_path: Path,
) -> None:
    """A bad authority path should produce a short CLI error, not an exception dump."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.paper.tables",
            "--authority",
            str(tmp_path / "missing.tsv"),
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "error:" in completed.stderr
