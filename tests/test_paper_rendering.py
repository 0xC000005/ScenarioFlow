"""Public contracts for evidence-only ICAIF and FMTS rendering."""

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

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest-v1.tsv"
ASSETS = ROOT / "paper_assets" / "ASSETS.tsv"

FIGURE_GOLDEN_IDENTITIES = {
    "iv-marginal-daily-changes.png": (
        233_707,
        "1a4efc0fe403246074a5d6fc8b2e610218119aaf8b71e146165ad3ba218b110b",
    ),
    "iv-cross-cell-sensitivity.png": (
        963_886,
        "a642aa971e5360559e968db603765ac3ddab700331f23aa73fdeff464e1fa14d",
    ),
    "joint-factor-fans.png": (
        277_531,
        "c4348af952343b905339915faf47eeb57ea7f4ac9124a7126e33a65a90db954c",
    ),
}

EXACT_PLOT_IDENTITIES = {
    "joint-factor-fans-selection.json": (
        1_536,
        "c30a1b42580d6e6686924ae3ef33e4dbe5436f95af11d6c89e331057fedb79f1",
    ),
}

TABLE_IDENTITIES = {
    "fmts-multiseed.tex": (
        7_398,
        "a04e9cdf9fc8c0cb36e1606e479d894b7255c91311ae678e257e690546878192",
    ),
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

PRESERVED_BENCHMARK_IDENTITIES = {
    "fmts.benchmark-full": (
        5_436,
        "521f596246bbd74b9ed2b2735977d6ba606b4bc91f36cdfba8b2d06baf9492fb",
    ),
    "fmts.benchmark-short": (
        4_231,
        "b505c03adbc507c01b8d0fffb1444ab4a0b85a06b32a73998d0a2dbae27c5e0d",
    ),
    "icaif.benchmark": (
        4_212,
        "915f6a3cafd464b9d0e911fdb77a5530044dc09a9f29f44b485ebdddafb54943",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_figure_content_matches_golden(rendered: Path, golden: Path) -> None:
    from PIL import Image

    with Image.open(rendered) as current_image, Image.open(golden) as golden_image:
        assert all(
            abs(current - expected) <= 1
            for current, expected in zip(
                current_image.size, golden_image.size, strict=True
            )
        )
        current_pixels = np.asarray(
            current_image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        golden_pixels = np.asarray(
            golden_image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    normalized_mae = float(np.mean(np.abs(current_pixels - golden_pixels)) / 255.0)
    assert normalized_mae <= 0.045


def _asset_rows() -> dict[str, dict[str, str]]:
    with ASSETS.open(encoding="utf-8", newline="") as handle:
        return {row["asset_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def _artifact_root() -> Path:
    configured = os.environ.get("SCENARIOFLOW_TEST_ARTIFACT_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else ROOT / "artifacts" / "cache",
        ROOT / "artifacts" / "cache",
    ]
    resolver = ArtifactResolver.from_tsv(MANIFEST)
    required = (
        "paper.icaif.iv-plot-arrays",
        "paper.icaif.joint-factor-fan-arrays",
        "evidence.nl.dollar-liquidity-cohesive",
        "evidence.nl.dollar-liquidity-start-only",
    )
    for candidate in candidates:
        if all(
            resolver.status(artifact_id, candidate).state == "present"
            for artifact_id in required
        ):
            return candidate
    pytest.skip("fetch the evidence profile to run exact paper-renderer integration")


def _run_paper(
    mode: str, artifact_root: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_HOME", None)
    return subprocess.run(
        [
            str(ROOT / "scripts" / "reproduce_paper.sh"),
            mode,
            "--artifact-root",
            str(artifact_root),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_launcher_validates_released_numerical_outputs(tmp_path: Path) -> None:
    """Wrong recipes, inputs, or formatting must fail published output identities."""
    completed = _run_paper("all", _artifact_root(), tmp_path / "paper")

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["mode"] == "all"
    assert set(report["preserved_only"]) == set(PRESERVED_BENCHMARK_IDENTITIES)
    for directory, identities in (
        (tmp_path / "paper" / "plots", EXACT_PLOT_IDENTITIES),
        (tmp_path / "paper" / "tables", TABLE_IDENTITIES),
    ):
        for name, (expected_size, expected_digest) in identities.items():
            path = directory / name
            assert path.stat().st_size == expected_size
            assert _sha256(path) == expected_digest
    for name in FIGURE_GOLDEN_IDENTITIES:
        _assert_figure_content_matches_golden(
            tmp_path / "paper" / "plots" / name,
            ROOT / "paper_assets" / "figures" / "generated" / name,
        )


def test_tables_and_plots_modes_are_deterministic_for_the_same_evidence(
    tmp_path: Path,
) -> None:
    """Mode selection must not change any generated bytes."""
    artifact_root = _artifact_root()
    all_output = tmp_path / "all"
    tables_output = tmp_path / "tables"
    plots_output = tmp_path / "plots"
    assert _run_paper("all", artifact_root, all_output).returncode == 0
    assert _run_paper("tables", artifact_root, tables_output).returncode == 0
    assert _run_paper("plots", artifact_root, plots_output).returncode == 0

    for name in TABLE_IDENTITIES:
        assert (all_output / "tables" / name).read_bytes() == (
            tables_output / "tables" / name
        ).read_bytes()
    for name in (*FIGURE_GOLDEN_IDENTITIES, *EXACT_PLOT_IDENTITIES):
        assert (all_output / "plots" / name).read_bytes() == (
            plots_output / "plots" / name
        ).read_bytes()
    assert not (tables_output / "plots").exists()
    assert not (plots_output / "tables").exists()


def test_plot_mode_reports_a_missing_released_array_without_a_traceback(
    tmp_path: Path,
) -> None:
    """A missing evidence payload must name the semantic artifact that is absent."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.cli",
            "paper",
            "plots",
            "--artifact-root",
            str(tmp_path / "empty-artifacts"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "paper.icaif.iv-plot-arrays" in completed.stderr
    assert "missing" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "output").exists()


def test_asset_ledger_covers_exact_bytes_and_honest_reproduction_modes() -> None:
    """Preserved outputs must never advertise a generator command."""
    rows = _asset_rows()
    released_paths = [ROOT / row["released_path"] for row in rows.values()]
    actual_assets = {
        path
        for directory in (
            ROOT / "paper_assets" / "tables" / "golden",
            ROOT / "paper_assets" / "figures" / "generated",
            ROOT / "paper_assets" / "figures" / "illustrations",
        )
        for path in directory.iterdir()
        if path.is_file()
    }

    assert set(released_paths) == actual_assets
    assert set(row["mode"] for row in rows.values()) <= {
        "generated",
        "verified-from-evidence",
        "preserved-only",
    }
    for row, path in zip(rows.values(), released_paths, strict=True):
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]
        if row["mode"] == "preserved-only":
            assert row["generator"] == ""
        else:
            assert row["generator"]

    assert rows["icaif.iv-marginal"]["released_path"].endswith(
        "iv-marginal-daily-changes.png"
    )
    assert (
        rows["icaif.iv-marginal"]["sha256"]
        == FIGURE_GOLDEN_IDENTITIES["iv-marginal-daily-changes.png"][1]
    )
    assert rows["readme.iv-marginal-all25"]["released_path"].endswith(
        "iv-marginal-daily-changes-all25.png"
    )
    assert rows["readme.iv-marginal-all25"]["mode"] == "preserved-only"

    for asset_id in (
        "icaif.iv-marginal",
        "icaif.iv-cross-cell",
        "icaif.joint-factor-fans",
    ):
        assert rows[asset_id]["mode"] == "verified-from-evidence"

    for asset_id, identity in PRESERVED_BENCHMARK_IDENTITIES.items():
        assert rows[asset_id]["mode"] == "preserved-only"
        assert rows[asset_id]["generator"] == ""
        assert (int(rows[asset_id]["bytes"]), rows[asset_id]["sha256"]) == identity
        assert rows[asset_id]["source_authority"]
    assert all(
        row["mode"] == "preserved-only" and not row["generator"]
        for row in rows.values()
        if row["kind"] == "illustration"
    )


def test_paper_renderer_import_has_no_demo_or_language_model_dependencies() -> None:
    """Numerical paper rendering must not import either interactive demo backend."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import scenarioflow.paper.render; "
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
        for forbidden in ("openai", "gradio")
        for name in modules
    )
    assert not any(name.startswith("experiments.") for name in modules)
    assert not any(
        name == "matplotlib" or name.startswith("matplotlib.") for name in modules
    )
