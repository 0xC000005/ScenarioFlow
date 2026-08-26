"""Reader-facing boundary for the released ICAIF and FMTS paper assets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from scenarioflow.artifacts import ArtifactResolver
from scenarioflow.paper.tables import (
    BenchmarkExportError,
    verify_dollar_liquidity_authorities,
    verify_preserved_table_views,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER_ASSETS = ROOT / "paper_assets"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_contract_root(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    for source in (
        PAPER_ASSETS / "tables" / "golden" / "fmts-benchmark-full.tex",
        PAPER_ASSETS / "tables" / "golden" / "fmts-benchmark-short.tex",
        PAPER_ASSETS / "tables" / "golden" / "icaif-benchmark.tex",
        ROOT
        / "results"
        / "frozen"
        / "benchmark"
        / "current-panel"
        / "scenarioflow"
        / "results.json",
        ROOT
        / "results"
        / "frozen"
        / "benchmark"
        / "current-panel"
        / "scenarioflow-graph"
        / "results.json",
        ROOT / "results" / "frozen" / "dollar-liquidity" / "narrative-report.json",
        ROOT / "results" / "frozen" / "dollar-liquidity" / "start-only-report.json",
        ROOT
        / "results"
        / "frozen"
        / "dollar-liquidity"
        / "fixed-start-comparison.json",
        ROOT / "paper_assets" / "ASSETS.tsv",
        ROOT / "results" / "FROZEN_AUTHORITY.tsv",
    ):
        destination = root / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root


def _update_asset_identity(root: Path, asset_id: str, table: Path) -> None:
    manifest = root / "paper_assets" / "ASSETS.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    row = next(item for item in rows if item["asset_id"] == asset_id)
    row["bytes"] = str(table.stat().st_size)
    row["sha256"] = _sha256(table)
    _write_tsv(manifest, rows, fieldnames)


def _update_authority_identity(root: Path, authority_id: str, payload: Path) -> str:
    manifest = root / "results" / "FROZEN_AUTHORITY.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    row = next(item for item in rows if item["authority_id"] == authority_id)
    size = str(payload.stat().st_size)
    digest = _sha256(payload)
    row["released_bytes"] = size
    row["released_sha256"] = digest
    row["frozen_bytes"] = size
    row["frozen_sha256"] = digest
    _write_tsv(manifest, rows, fieldnames)
    return digest


def _replace_row_authority_digest(root: Path, authority_id: str, digest: str) -> None:
    manifest = root / "paper_assets" / "ASSETS.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    marker = f"frozen-row:{authority_id}@"
    for row in rows:
        tokens = row["source_authority"].split(";")
        row["source_authority"] = ";".join(
            f"{marker}{digest}" if token.startswith(marker) else token
            for token in tokens
        )
    _write_tsv(manifest, rows, fieldnames)


def _release_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if tracked.returncode == 0:
        return [
            ROOT / Path(filename)
            for filename in tracked.stdout.decode().split("\0")
            if filename and (ROOT / filename).is_file()
        ]

    ignored = {
        ".git",
        ".pytest_cache",
        ".superpowers",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in ignored for part in path.relative_to(ROOT).parts)
    ]


def test_release_excludes_latex_workspaces_and_authoring_toolchains() -> None:
    """The release includes final assets, not either paper's build workspace."""
    files = _release_files()
    forbidden_suffixes = {
        ".bib",
        ".aux",
        ".log",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
    }
    forbidden_directories = {
        "camera_ready",
        "narrative_grounded_scenarios",
        "plotneuralnet",
    }

    assert not [path for path in files if path.suffix.lower() in forbidden_suffixes]
    assert not [
        path for path in files if path.name.startswith("main") and path.suffix == ".tex"
    ]
    assert not [
        path
        for path in files
        if forbidden_directories.intersection(path.relative_to(ROOT).parts)
    ]
    assert not [
        path
        for path in files
        if path.suffix == ".py"
        and ("tikz" in path.name.lower() or "illustration" in path.name.lower())
    ]

    tex_files = {path.relative_to(ROOT) for path in files if path.suffix == ".tex"}
    assert tex_files
    assert all(
        path.parts[:3] == ("paper_assets", "tables", "golden") for path in tex_files
    )


def test_paper_asset_guide_gives_copyable_reader_workflows() -> None:
    """A reader can distinguish verification, regeneration, and preservation."""
    guide = (PAPER_ASSETS / "README.md").read_text(encoding="utf-8")
    required_text = (
        "ASSETS.tsv",
        "Frozen paper-facing views",
        "historical experiment tree",
        "Illustrations",
    )
    for text in required_text:
        assert text in guide


def test_paper_table_manifest_distinguishes_frozen_views_from_reruns() -> None:
    """Fresh training must not be presented as a byte-exact table rebuild."""
    with (PAPER_ASSETS / "ASSETS.tsv").open(encoding="utf-8", newline="") as handle:
        rows = {row["asset_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    for asset_id in (
        "fmts.benchmark-full",
        "fmts.benchmark-short",
        "icaif.benchmark",
    ):
        assert rows[asset_id]["mode"] == "preserved-only"
        assert rows[asset_id]["generator"] == ""
    assert rows["fmts.multiseed"]["mode"] == "generated"
    assert rows["fmts.multiseed"]["generator"] == "./scripts/reproduce_paper.sh tables"

    guide = (PAPER_ASSETS / "README.md").read_text(encoding="utf-8")
    assert "historical experiment tree" in guide


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (r"\scenarioflow{} & \textbf{3.1772}", r"Wrong model & \textbf{3.1772}"),
        (r"\scenarioflow{} & \textbf{3.1772}", r"\scenarioflow{} & \textbf{9.9999}"),
    ),
)
def test_preserved_benchmark_rejects_a_corrupt_model_row_even_with_new_identity(
    tmp_path: Path, old: str, new: str
) -> None:
    """A rehashed final table must still agree with its surviving result row."""
    root = _benchmark_contract_root(tmp_path)
    table = root / "paper_assets" / "tables" / "golden" / "icaif-benchmark.tex"
    contents = table.read_text(encoding="utf-8")
    assert old in contents
    table.write_text(contents.replace(old, new, 1), encoding="utf-8")
    _update_asset_identity(root, "icaif.benchmark", table)

    with pytest.raises(BenchmarkExportError, match="row contract"):
        verify_preserved_table_views(root, root / "paper_assets" / "ASSETS.tsv")


def test_preserved_benchmark_rejects_a_corrupt_surviving_authority_value(
    tmp_path: Path,
) -> None:
    """The checked paper row must be derived from the authenticated result JSON."""
    root = _benchmark_contract_root(tmp_path)
    authority_id = (
        "release.results.frozen.benchmark.current-panel.scenarioflow.results.json"
    )
    result = (
        root
        / "results"
        / "frozen"
        / "benchmark"
        / "current-panel"
        / "scenarioflow"
        / "results.json"
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["panel_b_crps"]["overall"] = 9.9999
    result.write_text(json.dumps(payload), encoding="utf-8")
    digest = _update_authority_identity(root, authority_id, result)
    _replace_row_authority_digest(root, authority_id, digest)

    with pytest.raises(BenchmarkExportError, match="row contract"):
        verify_preserved_table_views(root, root / "paper_assets" / "ASSETS.tsv")


def test_preserved_benchmark_rejects_a_changed_declared_provenance(
    tmp_path: Path,
) -> None:
    """The non-mechanical paper-source claim must remain explicitly declared."""
    root = _benchmark_contract_root(tmp_path)
    manifest = root / "paper_assets" / "ASSETS.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    row = next(item for item in rows if item["asset_id"] == "icaif.benchmark")
    row["source_authority"] = "paper-authority:wrong-source"
    _write_tsv(manifest, rows, fieldnames)

    with pytest.raises(BenchmarkExportError, match="declared provenance"):
        verify_preserved_table_views(root, manifest)


def test_preserved_benchmark_rejects_a_changed_frozen_row_reference(
    tmp_path: Path,
) -> None:
    """The mechanically checked row authority must remain explicit and pinned."""
    root = _benchmark_contract_root(tmp_path)
    manifest = root / "paper_assets" / "ASSETS.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    row = next(item for item in rows if item["asset_id"] == "icaif.benchmark")
    row["source_authority"] = row["source_authority"].replace(
        "frozen-row:release.results.frozen.benchmark.current-panel."
        "scenarioflow.results.json@",
        "frozen-row:release.results.frozen.benchmark.current-panel."
        "wrong.results.json@",
    )
    _write_tsv(manifest, rows, fieldnames)

    with pytest.raises(BenchmarkExportError, match="row authority"):
        verify_preserved_table_views(root, manifest)


def test_dollar_liquidity_rejects_an_incomplete_composite_authority(
    tmp_path: Path,
) -> None:
    """All array and frozen-report inputs must remain declared together."""
    root = _benchmark_contract_root(tmp_path)
    manifest = root / "paper_assets" / "ASSETS.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    row = next(item for item in rows if item["asset_id"] == "icaif.dollar-liquidity")
    tokens = row["source_authority"].split(";")
    row["source_authority"] = ";".join(
        token for token in tokens if "fixed-start-comparison" not in token
    )
    _write_tsv(manifest, rows, fieldnames)
    resolver = ArtifactResolver.from_tsv(ROOT / "artifacts" / "manifest-v1.tsv")

    with pytest.raises(BenchmarkExportError, match="authority contract"):
        verify_dollar_liquidity_authorities(root, manifest, resolver)


def test_root_readme_links_paper_assets() -> None:
    """The root README must point readers to the released paper assets."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Paper assets](paper_assets/README.md)" in readme
    headings = (
        "## How ScenarioFlow works",
        "## Quick start",
        "## Example outputs",
        "## Model architecture",
        "## Run training using our data",
        "## Run inference",
    )
    assert all(heading in readme for heading in headings)
    assert [readme.index(heading) for heading in headings] == sorted(
        readme.index(heading) for heading in headings
    )
    for text in (
        "The released weights are used directly; this does not retrain the model.",
        "./scripts/train_next_step.sh --seed 1001",
        "./scripts/finetune_rollouts.sh --seed 1001",
        "### Run ScenarioFlow inference",
        "./scripts/run_inference.sh",
        "### Run multimodal conditional inference",
        "./scripts/run_demo.sh",
        "paper_assets/figures/illustrations/flow-transport.png",
        "paper_assets/figures/generated/joint-factor-fans.png",
        "paper_assets/figures/generated/iv-marginal-daily-changes-all25.png",
        "paper_assets/figures/illustrations/scenarioflow-architecture.png",
        "Core model and temporal prefix encoder",
        "Conditional velocity network",
        "State representation and reconstruction",
        "Stage 1 next-step training",
        "Stage 2 rollout fine-tuning",
        "Numerical inference",
        "lightweight alignment modules",
    ):
        assert text in readme
    assert "ICAIF numerical plots" not in readme
    assert "## Choose a route" not in readme
    assert "./scripts/check_installation.sh" in readme
    assert "./scripts/reproduce.sh" in readme
    for command in (
        "reproduction/source/",
        "./scripts/reproduce_paper.sh",
    ):
        assert command not in readme
    assert "OpenAI and Codex are used only by" in readme
    assert "the interactive demos" in readme


def test_reader_guides_have_no_broken_local_links() -> None:
    """Every committed relative Markdown link in the reader guides must resolve."""
    for document in (ROOT / "README.md", PAPER_ASSETS / "README.md"):
        contents = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", contents):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            assert (document.parent / relative).exists(), f"broken link: {target}"
