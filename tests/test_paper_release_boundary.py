"""Reader-facing boundary for the released ICAIF and FMTS paper assets."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER_ASSETS = ROOT / "paper_assets"


def _release_files() -> list[Path]:
    ignored = {".git", ".pytest_cache", ".superpowers", ".venv", "__pycache__"}
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
        "results/FROZEN_AUTHORITY.tsv",
        "paper_assets/ASSETS.tsv",
        "scenarioflow paper export-tables",
        "scenarioflow paper export-dollar-liquidity",
        "scenarioflow paper plot iv-marginal",
        "scenarioflow paper plot iv-cross-cell",
        "scenarioflow paper plot joint-factor-fans",
        "Preserved illustrations",
    )
    for text in required_text:
        assert text in guide


def test_root_readme_links_assets_and_reserves_verified_paper_links() -> None:
    """The root points to assets while publication URLs remain explicit release gates."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Paper assets](paper_assets/README.md)" in readme
    assert "ICAIF paper:" in readme
    assert "FMTS workshop paper:" in readme
    for line in readme.splitlines():
        if line.startswith(("- ICAIF paper:", "- FMTS workshop paper:")):
            assert "http://" not in line and "https://" not in line


def test_reader_guides_have_no_broken_local_links() -> None:
    """Every committed relative Markdown link in the reader guides must resolve."""
    for document in (ROOT / "README.md", PAPER_ASSETS / "README.md"):
        contents = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", contents):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            assert (document.parent / relative).exists(), f"broken link: {target}"
