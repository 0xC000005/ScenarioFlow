"""Reader-facing boundary for the released ICAIF and FMTS paper assets."""

from __future__ import annotations

import csv
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER_ASSETS = ROOT / "paper_assets"


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
