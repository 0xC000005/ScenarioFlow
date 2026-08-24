"""Regression tests for the frozen FMTS multi-seed table."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t

from scenarioflow.evaluation.current_panel import METRIC_PATHS
from scenarioflow.paper.multiseed import export_multiseed_table
from scenarioflow.workflows.multiseed import ROLLOUT_SEEDS, TRAIN_SEEDS


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "frozen" / "multiseed" / "summary.json"
GOLDEN = ROOT / "paper_assets" / "tables" / "golden" / "fmts-multiseed.tex"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiseed_golden_has_verified_archive_identity() -> None:
    """Copying the wrong paper fragment must fail independently of the exporter."""
    assert GOLDEN.stat().st_size == 7_398
    assert _sha256(GOLDEN) == (
        "a04e9cdf9fc8c0cb36e1606e479d894b7255c91311ae678e257e690546878192"
    )


def test_multiseed_export_is_exact_and_deterministic(tmp_path: Path) -> None:
    """Any change in table formatting or aggregate selection must fail."""
    first = export_multiseed_table(SUMMARY, tmp_path / "first.tex")
    second = export_multiseed_table(SUMMARY, tmp_path / "second.tex")
    assert first.read_bytes() == GOLDEN.read_bytes()
    assert second.read_bytes() == first.read_bytes()


def test_multiseed_table_contains_all_task3_metrics_in_frozen_order(
    tmp_path: Path,
) -> None:
    """Dropping or reordering one of the 18 benchmark metrics must fail."""
    text = export_multiseed_table(SUMMARY, tmp_path / "table.tex").read_text()
    metric_names = [name for name, _path in METRIC_PATHS]
    assert len(metric_names) == 18
    expected_headers = [
        r"CRPS$\downarrow$",
        r"IV CRPS$\downarrow$",
        r"Factor CRPS$\downarrow$",
        r"Energy$\downarrow$",
        r"Variogram$\downarrow$",
        r"Corr.$\uparrow$",
        r"Factor KS$\downarrow$",
        r"Tail q99$\to1$",
        r"Cov90$\to0.90$",
        r"CalErr$\downarrow$",
        r"Level KS$_{\rm med}\downarrow$",
        r"Med. dev.$\downarrow$",
        r"Width $\rho\uparrow$",
        r"MR$\to1$",
        r"Econ.$\to1$",
        r"Jump KS$\downarrow$",
        r"Kurt.$\to1$",
        r"ACF$\uparrow$",
    ]
    positions = [text.index(header) for header in expected_headers]
    assert positions == sorted(positions)


def test_learned_and_classical_intervals_use_three_and_five_observations(
    tmp_path: Path,
) -> None:
    """Pooling 15 learned runs or treating classical rows as three seeds must fail."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    runs = summary["per_runs"]

    learned_values = []
    for train_seed in TRAIN_SEEDS:
        rollout_values = [
            run["metrics"]["CRPS"]
            for run in runs
            if run["candidate"] == "scenarioflow"
            and run["training_seed"] == train_seed
            and run["rollout_seed"] in ROLLOUT_SEEDS
        ]
        assert len(rollout_values) == 5
        learned_values.append(statistics.fmean(rollout_values))
    learned_mean = statistics.fmean(learned_values)
    learned_half = (
        student_t.ppf(0.975, 2) * statistics.stdev(learned_values) / math.sqrt(3)
    )

    classical_values = [
        run["metrics"]["CRPS"]
        for run in runs
        if run["candidate"] == "random_walk"
        and run["training_seed"] is None
        and run["rollout_seed"] in ROLLOUT_SEEDS
    ]
    assert len(classical_values) == 5
    classical_mean = statistics.fmean(classical_values)
    classical_half = (
        student_t.ppf(0.975, 4) * statistics.stdev(classical_values) / math.sqrt(5)
    )

    text = export_multiseed_table(SUMMARY, tmp_path / "table.tex").read_text()
    assert (
        rf"\textbf{{{learned_mean:.4f}}}\,[{learned_mean-learned_half:.4f}, "
        rf"{learned_mean+learned_half:.4f}]"
    ) in text
    assert (
        rf"{classical_mean:.4f}\,[{classical_mean-classical_half:.4f}, "
        rf"{classical_mean+classical_half:.4f}]"
    ) in text


def test_multiseed_export_does_not_recompute_or_rewrite_summary(tmp_path: Path) -> None:
    """The formatter must leave the sole frozen aggregate authority untouched."""
    before = _sha256(SUMMARY)
    export_multiseed_table(SUMMARY, tmp_path / "table.tex")
    assert _sha256(SUMMARY) == before


def test_multiseed_export_cannot_follow_a_preexisting_temporary_symlink(
    tmp_path: Path,
) -> None:
    """A predictable temporary symlink must never redirect an export write."""
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"leave this file unchanged")
    output = tmp_path / "table.tex"
    output.with_name(output.name + ".tmp").symlink_to(victim)

    exported = export_multiseed_table(SUMMARY, output)

    assert victim.read_bytes() == b"leave this file unchanged"
    assert exported.read_bytes() == GOLDEN.read_bytes()
