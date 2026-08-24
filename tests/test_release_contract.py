"""Regression contract for the five processed datasets shipped with ScenarioFlow."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPECTED_DATA = {
    "multi_factor_data.npz": (420_993, "fcb34d7d0fdc9a874e06c95649f0ba93f7e203afb0e9690901cf8f67ebba5991"),
    "multi_factor_levels.parquet": (347_987, "3ac3aeb04fd227fb471bc026f5f4191f5655c4cf3e09443b102e4d6d0019c75f"),
    "multi_factor_returns.parquet": (463_257, "77970f0b506c1ea6bd0ab45b539c621e2cc4199b10b223f2a07c6d4a2003f21d"),
    "spx_vol_surface_history_full_data_fixed.parquet": (1_761_788, "d657a9cf50ba513fa1806aaf8b1a30a7646438bda09ebeabc5f6804f6e0f704a"),
    "vol_surface_with_ret.npz": (1_398_758, "3e5d05a4cc72cfbd60646aab6f491aac763d4667144409dd72f7969e40b4d50f"),
}
LEVEL_COLUMNS = [
    "spx", "usdcad", "usdjpy", "dxy", "copper", "wheat", "crude_oil",
    "us2y", "us10y", "aaa_oas", "bbb_oas", "nikkei", "gold", "vix",
]
RETURN_COLUMNS = [
    "spx_logret", "usdcad_logret", "usdjpy_logret", "dxy_logret",
    "copper_logret", "wheat_logret", "crude_oil_logret", "us2y_diff",
    "us10y_diff", "aaa_oas_diff", "bbb_oas_diff", "nikkei_logret",
    "gold_logret", "vix_logret",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_processed_data_filenames_checksums_and_total_size() -> None:
    """A renamed, changed, or omitted frozen data byte must fail this release gate."""
    assert {path.name for path in DATA_DIR.iterdir() if path.is_file()} >= set(EXPECTED_DATA)
    assert sum(path.stat().st_size for path in DATA_DIR.iterdir() if path.name in EXPECTED_DATA) == 4_392_783
    for filename, (expected_size, expected_hash) in EXPECTED_DATA.items():
        path = DATA_DIR / filename
        assert path.stat().st_size == expected_size
        assert _sha256(path) == expected_hash


def test_checksum_manifest_matches_frozen_data_contract() -> None:
    """A stale or incomplete release manifest must not authenticate the data bundle."""
    with (DATA_DIR / "CHECKSUMS.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {(row["path"], int(row["bytes"]), row["sha256"]) for row in rows} == {
        (f"data/{filename}", size, digest)
        for filename, (size, digest) in EXPECTED_DATA.items()
    }


def test_data_schema_and_date_coverage() -> None:
    """A changed array, column order, or observed-date range must fail the contract."""
    with np.load(DATA_DIR / "multi_factor_data.npz", allow_pickle=False) as factor_data:
        assert factor_data.files == ["dates", "levels", "level_columns", "returns", "return_columns"]
        assert factor_data["dates"].shape == (5_825,)
        assert factor_data["dates"].dtype == np.dtype("datetime64[D]")
        assert str(factor_data["dates"][0]) == "2000-01-03"
        assert str(factor_data["dates"][-1]) == "2023-02-27"
        assert factor_data["levels"].shape == (5_825, 14)
        assert factor_data["returns"].shape == (5_825, 14)
        assert factor_data["levels"].dtype == np.dtype("float32")
        assert factor_data["returns"].dtype == np.dtype("float32")
        assert factor_data["level_columns"].tolist() == LEVEL_COLUMNS
        assert factor_data["return_columns"].tolist() == RETURN_COLUMNS

    levels = pd.read_parquet(DATA_DIR / "multi_factor_levels.parquet")
    returns = pd.read_parquet(DATA_DIR / "multi_factor_returns.parquet")
    surface_frame = pd.read_parquet(DATA_DIR / "spx_vol_surface_history_full_data_fixed.parquet")
    assert levels.shape == (5_825, 14)
    assert returns.shape == (5_825, 14)
    assert levels.columns.tolist() == LEVEL_COLUMNS
    assert returns.columns.tolist() == RETURN_COLUMNS
    assert [str(dtype) for dtype in levels.dtypes] == ["float32"] * 11 + ["float64"] * 3
    assert [str(dtype) for dtype in returns.dtypes] == ["float32"] * 11 + ["float64"] * 3
    assert surface_frame.shape == (5_822, 32)
    assert surface_frame.columns[0] == "date"
    assert str(surface_frame["date"].dtype) == "datetime64[ns]"
    assert all(str(dtype) == "float64" for dtype in surface_frame.dtypes.iloc[1:])
    assert surface_frame["date"].min() == pd.Timestamp("2000-01-03")
    assert surface_frame["date"].max() == pd.Timestamp("2023-02-24")

    with np.load(DATA_DIR / "vol_surface_with_ret.npz", allow_pickle=False) as iv_data:
        assert iv_data.files == ["surface", "ret", "price", "slopes", "skews", "levels"]
        assert iv_data["surface"].shape == (5_822, 5, 5)
        for key in ("ret", "price", "slopes", "skews", "levels"):
            assert iv_data[key].shape == (5_822,)


def test_date_aligned_panel_has_39_variables() -> None:
    """Index-based alignment or a missing IV/factor state must fail this joint-panel gate."""
    surface_frame = pd.read_parquet(DATA_DIR / "spx_vol_surface_history_full_data_fixed.parquet")
    iv_dates = pd.DatetimeIndex(surface_frame["date"])
    with np.load(DATA_DIR / "vol_surface_with_ret.npz", allow_pickle=False) as iv_data, np.load(
        DATA_DIR / "multi_factor_data.npz", allow_pickle=False
    ) as factor_data:
        factor_dates = pd.DatetimeIndex(factor_data["dates"])
        index_by_date = pd.Series(np.arange(factor_dates.size), index=factor_dates)
        alignment = index_by_date.reindex(iv_dates)
        assert alignment.notna().all()
        indices = alignment.to_numpy(dtype=np.int64)
        assert np.array_equal(factor_dates[indices].to_numpy(), iv_dates.to_numpy())
        panel = np.concatenate(
            [iv_data["surface"].reshape(5_822, 25), factor_data["levels"][indices]], axis=1
        )

    assert panel.shape == (5_822, 39)


def test_release_scanners_reject_secrets_and_host_paths(tmp_path: Path) -> None:
    """A public-release scanner must reject credential-shaped values and host paths."""
    (tmp_path / "credential.txt").write_text("ghp_" + "a" * 36)
    (tmp_path / "host-path.txt").write_text("/" + "home" + "/" + "example" + "/data")
    command = [
        "python",
        str(ROOT / "scripts" / "check_release.py"),
        "--root",
        str(tmp_path),
        "--all",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "credential-shaped value" in result.stdout
    assert "absolute host path" in result.stdout
