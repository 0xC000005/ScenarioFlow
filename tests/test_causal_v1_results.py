"""Checks for the frozen target-disjoint (causal-v1) rerun results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "causal-v1"


def test_causal_v1_files_match_their_checksums() -> None:
    """Every released causal-v1 file must match its recorded size and digest."""
    with (RESULTS / "CHECKSUMS.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 4
    listed = {row["path"] for row in rows}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in RESULTS.iterdir()
        if path.name not in {"README.md", "CHECKSUMS.tsv"}
    }
    assert listed == actual
    for row in rows:
        payload = (ROOT / row["path"]).read_bytes()
        assert len(payload) == int(row["bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_causal_v1_table_agrees_with_the_aggregate() -> None:
    """The printed ScenarioFlow row must come from the released aggregate."""
    aggregate = json.loads((RESULTS / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["independent_test"] is False
    assert aggregate["all_evaluations_complete"] is False

    candidates = aggregate["candidates"]
    assert len(candidates) == 12
    assert candidates["scenarioflow"]["passes_all_screens"] == 1
    assert candidates["toto_1_0_finetuned"]["completed_runs"] == 0

    runs = {run["candidate"]: run for run in aggregate["runs"]}
    metrics = runs["scenarioflow"]["metrics"]
    table = (
        ROOT / "paper_assets" / "tables" / "golden" / "fmts-causal-v1.tex"
    ).read_text(encoding="utf-8")
    assert f"{metrics['CRPS']:.4f}" in table
    assert f"{metrics['Cov90']:.3f}" in table


def test_causal_v1_split_is_target_disjoint() -> None:
    """The recorded split must leave a full horizon between train and validation."""
    results = json.loads(
        (RESULTS / "scenarioflow-results.json").read_text(encoding="utf-8")
    )
    metadata = results["metadata"]
    assert metadata["data_contract"] == "causal-v1"
    gap = metadata["val_indices_start"] - metadata["train_indices_end"]
    assert gap >= metadata["future_len"]
    assert metadata["last_train_target_date"] < metadata["first_val_target_date"]
