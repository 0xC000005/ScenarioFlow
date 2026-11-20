"""Frozen-source and public-boundary checks for the reviewed React site."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
AUTHORITY = WEB / "SOURCE_AUTHORITY.tsv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reviewed_web_snapshot_is_complete_and_byte_exact() -> None:
    """All 70 released frontend files must match the frozen source authority."""
    with AUTHORITY.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert tuple(reader.fieldnames or ()) == (
            "path",
            "source_bytes",
            "source_sha256",
            "released_bytes",
            "released_sha256",
            "transformation",
        )
        rows = list(reader)

    assert len(rows) == 70
    assert len({row["path"] for row in rows}) == 70
    for row in rows:
        assert row["path"].startswith("web/")
        path = ROOT / row["path"]
        assert path.is_file() and not path.is_symlink()
        assert int(row["source_bytes"]) > 0
        assert len(row["source_sha256"]) == 64
        assert path.stat().st_size == int(row["released_bytes"])
        assert _sha256(path) == row["released_sha256"]
        assert row["transformation"] in {
            "identity",
            "public-release-v1",
            "camera-ready-paper-v1",
        }

    ignored = {"node_modules", "dist"}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in WEB.rglob("*")
        if path.is_file()
        and path != AUTHORITY
        and not ignored.intersection(path.relative_to(WEB).parts)
    }
    assert actual == {row["path"] for row in rows}
    changed = [row for row in rows if row["transformation"] != "identity"]
    assert [row["path"] for row in changed] == [
        "web/.gitignore",
        "web/eslint.config.js",
        "web/package-lock.json",
        "web/package.json",
        "web/public/data/memory_map_story.json",
        "web/public/data/narrative_shift_2016-01-20.json",
        "web/public/data/vg/scenario_to_narrative_2008-10-31.json",
        "web/public/paper/ScenarioFlow_FMTS2026_paper.pdf",
        "web/public/story/engine.js",
        "web/src/components/ScenarioComparisonTable.tsx",
        "web/src/components/ScenarioFan.tsx",
        "web/src/components/Sparkline.tsx",
        "web/src/components/vg/ExplainerVideo.tsx",
        "web/src/components/vg/Hatch.tsx",
        "web/src/components/vg/ScenarioToNarrative.tsx",
        "web/src/components/vg/anim/chart.ts",
        "web/src/components/vg/landing/BeforeAfterGrid.tsx",
        "web/src/components/vg/landing/EmbeddingMap.tsx",
        "web/src/components/vg/landing/HeroCrude.tsx",
        "web/src/components/vg/story/ScrollStory.tsx",
        "web/src/data/narratives.ts",
        "web/src/index.css",
        "web/src/lib/palette.ts",
    ]


def test_web_source_has_no_private_host_or_credential_material() -> None:
    """The browser release must not carry the live host's private configuration."""
    forbidden = (
        "/home/",
        "/tmp/",
        "vol-surface-vae-pub",
        "OPENAI_API_KEY=",
        "BEGIN PRIVATE KEY",
        "scenarioflow-public-autostart",
    )
    for path in WEB.rglob("*"):
        if not path.is_file() or {"node_modules", "dist"}.intersection(
            path.relative_to(WEB).parts
        ):
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not [value for value in forbidden if value in contents], path
