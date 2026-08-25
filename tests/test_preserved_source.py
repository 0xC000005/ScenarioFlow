"""Integrity checks for the source-preserving numerical release."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "scripts/run_reproduction.py"
    spec = importlib.util.spec_from_file_location("scenarioflow_release_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_preserved_source_matches_its_manifest() -> None:
    runner = _runner()
    with (ROOT / "reproduction/SOURCE_MANIFEST.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        expected = sum(1 for _row in csv.DictReader(handle, delimiter="\t"))
    assert runner.verify_sources() == expected


def test_reimplemented_scientific_pipeline_is_absent() -> None:
    package = ROOT / "src/scenarioflow"
    absent = (
        "training.py",
        "inference.py",
        "losses.py",
        "baselines",
        "evaluation",
        "workflows",
    )
    assert not [name for name in absent if (package / name).exists()]
    assert not [
        path
        for path in (package / "paper").glob("*.py")
        if path.name != "__init__.py" and path.name != "authority.py"
    ]


def test_preserved_snapshots_share_the_packaged_data() -> None:
    for snapshot in ("scenarioflow", "baselines", "garch_ccc", "deepvar_sni"):
        data = ROOT / "reproduction/source" / snapshot / "data"
        assert data.is_symlink()
        assert (data / "vol_surface_with_ret.npz").is_file()

