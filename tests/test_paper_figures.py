"""Tests for numerical paper plots and preserved illustrations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scenarioflow.evaluation.current_panel import prepare_validation_blocks
from scenarioflow.paper.plots import (
    _encode_history_levels,
    _prepare_iv_validation_block,
    PaperPlotError,
    build_factor_fan_selection,
    plot_iv_cross_cell_sensitivity,
    plot_iv_marginal_daily_changes,
    plot_joint_factor_fans,
    select_common_histories,
)
from scenarioflow.state import UnifiedVariableSpec, normalize_increment_windows

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = ROOT / "paper_assets" / "ASSETS.tsv"
GENERATED = ROOT / "paper_assets" / "figures" / "generated"
ILLUSTRATIONS = ROOT / "paper_assets" / "figures" / "illustrations"

FIGURE_IDENTITIES = {
    "generated/iv-marginal-daily-changes.png": (
        233_707,
        "1a4efc0fe403246074a5d6fc8b2e610218119aaf8b71e146165ad3ba218b110b",
    ),
    "generated/iv-cross-cell-sensitivity.png": (
        963_886,
        "a642aa971e5360559e968db603765ac3ddab700331f23aa73fdeff464e1fa14d",
    ),
    "generated/joint-factor-fans.png": (
        277_531,
        "c4348af952343b905339915faf47eeb57ea7f4ac9124a7126e33a65a90db954c",
    ),
    "generated/joint-factor-fans-selection.json": (
        1_536,
        "c30a1b42580d6e6686924ae3ef33e4dbe5436f95af11d6c89e331057fedb79f1",
    ),
    "illustrations/flow-transport.png": (
        1_122_275,
        "253d7e8ec98949b17b718c8e272666b88f3fa6f1a0e39a8899a93719a419c9e9",
    ),
    "illustrations/adaptive-graph-decoder.png": (
        1_215_456,
        "b86d56db1a5bec35a3ee65662dfffb08f869e61de68b2a7be9b694086ee80e45",
    ),
    "illustrations/scenarioflow-architecture.pdf": (
        161_485,
        "d6b0f02c20fbf19a858bd737796db5a9a4fb75aaff0c9c71554bfe5b473dd3c4",
    ),
    "illustrations/scenarioflow-nl-bridge.pdf": (
        161_127,
        "a60503b000365e5fd7067a940471cb715ef543381ee695c834e531ce16a98545",
    ),
    "illustrations/scenarioflow-nl-pipeline.pdf": (
        81_026,
        "5a54a7ff9f4d50596ab75c38fdcecaad5d50a85cef6fcf904e2669ac0fbd9849",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_iv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(734)
    history = 0.20 + rng.normal(0.0, 0.005, size=(12, 30, 5, 5)).cumsum(axis=1)
    future = history[:, -1:, :, :] + rng.normal(0.0, 0.006, size=(12, 30, 5, 5)).cumsum(
        axis=1
    )
    samples = history[:, None, -1:, :, :] + rng.normal(
        0.0, 0.006, size=(12, 8, 30, 5, 5)
    ).cumsum(axis=2)
    return history, future, samples


def test_all_nine_frozen_figure_assets_have_exact_identities() -> None:
    """Published outputs must remain byte-identical to the frozen paper assets."""
    for relative, (size, digest) in FIGURE_IDENTITIES.items():
        path = ROOT / "paper_assets" / "figures" / relative
        assert path.stat().st_size == size
        assert _sha256(path) == digest


def test_asset_manifest_separates_generated_outputs_from_illustrations() -> None:
    """Only numerical plots and tables may advertise a regeneration command."""
    with ASSET_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 16
    assert {row["mode"] for row in rows} == {
        "generated-table",
        "generated-plot",
        "preserved-illustration",
    }
    preserved = [row for row in rows if row["mode"] == "preserved-illustration"]
    assert len(preserved) == 5
    assert all(row["generator"] == "" for row in preserved)
    assert all(row["generator"] for row in rows if row not in preserved)


def test_asset_manifest_covers_every_released_asset_with_exact_identity() -> None:
    """The ledger must be complete and must describe the bytes in the release."""
    with ASSET_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    released_paths = [ROOT / row["released_path"] for row in rows]
    assert len(released_paths) == len(set(released_paths))
    for row, path in zip(rows, released_paths, strict=True):
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]

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


def test_plot_module_keeps_matplotlib_optional_until_a_plot_is_requested() -> None:
    """Authority and table commands must not require the paper plotting extra."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import scenarioflow.paper.plots; "
                "print(json.dumps(sorted(sys.modules)))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not any(
        name == "matplotlib" or name.startswith("matplotlib.")
        for name in json.loads(completed.stdout)
    )


def test_iv_plot_recipes_accept_portable_arrays(tmp_path: Path) -> None:
    """Both IV recipes run from arrays without importing a donor experiment."""
    pytest.importorskip("matplotlib")
    history, future, samples = _synthetic_iv()

    marginal = plot_iv_marginal_daily_changes(
        history, future, samples, tmp_path / "marginal.png"
    )
    sensitivity = plot_iv_cross_cell_sensitivity(
        history, future, samples, tmp_path / "sensitivity.png"
    )

    assert marginal.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sensitivity.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_recipe_refuses_a_redirected_output_path(tmp_path: Path) -> None:
    """A caller-controlled symlink must not redirect a generated figure write."""
    pytest.importorskip("matplotlib")
    history, future, samples = _synthetic_iv()
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"do not replace")
    output = tmp_path / "figure.png"
    output.symlink_to(victim)

    with pytest.raises(PaperPlotError, match="symlink"):
        plot_iv_marginal_daily_changes(history, future, samples, output)

    assert victim.read_bytes() == b"do not replace"


def test_iv_checkpoint_recipe_uses_log_coordinates_for_model_levels() -> None:
    """The frozen IV-only checkpoint was trained on log-IV level coordinates."""
    block = _prepare_iv_validation_block(ROOT / "data")

    assert block.history_state.shape == (441, 30, 25)
    assert block.future_state.shape == (441, 30, 25)
    assert {spec.transform for spec in block.specs} == {"log_level"}

    spec = UnifiedVariableSpec(
        name="iv:test",
        source_column="iv:test",
        source_index=0,
        transform="log_level",
    )
    raw = np.asarray([[[0.20], [0.25]]], dtype=np.float32)
    encoded = _encode_history_levels(raw, [spec])
    np.testing.assert_allclose(encoded[..., 0], np.log(raw[..., 0]), rtol=1e-6)


def test_common_history_selection_reconstructs_the_frozen_selection_json() -> None:
    """The 15th/85th-percentile history rule must select indices 326 and 226."""
    _panel, _columns, _train, validation = prepare_validation_blocks(
        data_dir=ROOT / "data"
    )
    _history, _future, _center, scale = normalize_increment_windows(
        validation.history_increment,
        validation.future_increment,
        half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
    )
    selected, activity = select_common_histories(
        scale[:, 25:], calm_quantile=0.15, turbulent_quantile=0.85
    )
    document = build_factor_fan_selection(
        selected=selected,
        activity_score=activity,
        raw_indices=validation.indices,
        data_dir=ROOT / "data",
    )

    assert selected.tolist() == [326, 226]
    expected = json.loads(
        (GENERATED / "joint-factor-fans-selection.json").read_text(encoding="utf-8")
    )
    assert document == expected


def test_joint_factor_fan_recipe_accepts_portable_arrays(tmp_path: Path) -> None:
    """The fan plot needs only histories, futures, samples, labels, and dates."""
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(7345)
    histories = rng.normal(size=(2, 30, 39)).cumsum(axis=1)
    futures = histories[:, -1:, :] + rng.normal(size=(2, 30, 39)).cumsum(axis=1)
    samples = histories[:, None, -1:, :] + rng.normal(size=(2, 64, 30, 39)).cumsum(
        axis=2
    )
    names = [f"iv:{index:02d}" for index in range(25)] + [
        "factor:spx",
        "factor:usdcad",
        "factor:usdjpy",
        "factor:dxy",
        "factor:copper",
        "factor:wheat",
        "factor:crude_oil",
        "factor:us2y",
        "factor:us10y",
        "factor:aaa_oas",
        "factor:bbb_oas",
        "factor:nikkei",
        "factor:gold",
        "factor:vix",
    ]

    output = plot_joint_factor_fans(
        histories,
        futures,
        samples,
        names,
        ["2017-05-12", "2016-12-16"],
        tmp_path / "fans.png",
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_native_plot_cli_exposes_only_the_three_numerical_recipes() -> None:
    """Preserved illustrations must never appear as runnable plot choices."""
    help_result = subprocess.run(
        [sys.executable, "-m", "scenarioflow.cli", "paper", "plot", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    for recipe in ("iv-marginal", "iv-cross-cell", "joint-factor-fans"):
        assert recipe in help_result.stdout
    assert "flow-transport" not in help_result.stdout
    assert "scenarioflow-architecture" not in help_result.stdout

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "scenarioflow.cli",
            "paper",
            "plot",
            "flow-transport",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "Traceback" not in rejected.stderr
