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
from scenarioflow.paper.render import PaperRenderError, _verify_figure_content

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

TABLE_IDENTITIES = {
    "fmts-benchmark-full.tex": PRESERVED_BENCHMARK_IDENTITIES["fmts.benchmark-full"],
    "fmts-benchmark-short.tex": PRESERVED_BENCHMARK_IDENTITIES["fmts.benchmark-short"],
    "icaif-benchmark.tex": PRESERVED_BENCHMARK_IDENTITIES["icaif.benchmark"],
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


def _mutate_panel_figure(source: Path, destination: Path, mutation: str) -> None:
    from PIL import Image, ImageChops, ImageDraw

    with Image.open(source) as loaded:
        image = loaded.convert("RGB")
    width, height = image.size
    columns = (0, width // 3, 2 * width // 3, width)
    rows = (0, height // 3, 2 * height // 3, height)
    center = (columns[1], rows[1], columns[2], rows[2])
    if mutation == "erase-center-panel":
        ImageDraw.Draw(image).rectangle(center, fill="white")
    elif mutation == "swap-panels":
        left = (columns[0], rows[1], columns[1], rows[2])
        right = (columns[2], rows[1], columns[3], rows[2])
        left_panel = image.crop(left)
        right_panel = image.crop(right).resize(left_panel.size)
        image.paste(right_panel, left)
        image.paste(
            left_panel.resize((right[2] - right[0], right[3] - right[1])), right
        )
    elif mutation == "alter-one-panel":
        panel = image.crop(center)
        image.paste(ImageChops.invert(panel), center)
    else:  # pragma: no cover - test helper contract
        raise AssertionError(mutation)
    image.save(destination)


def test_public_launcher_validates_released_numerical_outputs(tmp_path: Path) -> None:
    """Wrong recipes, inputs, or formatting must fail published output identities."""
    completed = _run_paper("all", _artifact_root(), tmp_path / "paper")

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["mode"] == "all"
    assert set(report["preserved_only"]) == set(PRESERVED_BENCHMARK_IDENTITIES)
    assert set(report["materialized_preserved"]) == set(
        PRESERVED_BENCHMARK_IDENTITIES
    )
    assert {entry["verification"] for entry in report["preserved_only"].values()} == {
        "exact identity; declared paper-source provenance; "
        "frozen benchmark row checked"
    }
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


@pytest.mark.parametrize(
    "mutation",
    ("erase-center-panel", "swap-panels", "alter-one-panel"),
)
def test_visual_gate_rejects_localized_panel_corruption(
    tmp_path: Path, mutation: str
) -> None:
    """A local or structural panel failure must not hide in a global average."""
    golden = (
        ROOT
        / "paper_assets"
        / "figures"
        / "generated"
        / "iv-marginal-daily-changes.png"
    )
    changed = tmp_path / f"{mutation}.png"
    _mutate_panel_figure(golden, changed, mutation)

    with pytest.raises(PaperRenderError, match="figure (tile|edge) agreement changed"):
        _verify_figure_content("icaif.iv-marginal", changed, golden)


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


def test_paper_command_requires_a_caller_selected_output_directory(
    tmp_path: Path,
) -> None:
    """The public command must not silently choose a generated destination."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.cli",
            "paper",
            "plots",
            "--artifact-root",
            str(_artifact_root()),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--output-dir" in completed.stderr
    assert "required" in completed.stderr
    assert not (tmp_path / "build" / "paper-reproduction").exists()


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

    dollar_inputs = {
        "artifact:evidence.nl.dollar-liquidity-cohesive",
        "artifact:evidence.nl.dollar-liquidity-start-only",
        (
            "frozen:release.results.frozen.dollar-liquidity."
            "narrative-report.json@"
            "a2bee8502f6538e0a4649c1f10e24e3bec052f52ac1cc02ab8c26883c7155008"
        ),
        (
            "frozen:release.results.frozen.dollar-liquidity."
            "start-only-report.json@"
            "46fe2b8fd4363ea37a4aaa0f2377455ff13510ab601bd29a3d767c86feec24e3"
        ),
        (
            "frozen:release.results.frozen.dollar-liquidity."
            "fixed-start-comparison.json@"
            "102e92b16579b2b185ac17b8eb0818517047619aee07ec5416da26678d284e8d"
        ),
    }
    for asset_id in (
        "fmts.dollar-liquidity-full",
        "fmts.dollar-liquidity-short",
        "icaif.dollar-liquidity",
    ):
        assert set(rows[asset_id]["source_authority"].split(";")) == dollar_inputs

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
