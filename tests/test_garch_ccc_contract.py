from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scenarioflow.evaluation.garch_ccc_contract import (
    EXPECTED_SOURCE_BLOB,
    EXPECTED_SOURCE_REVISION,
    SOURCE_TAG,
    validate_published_row,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path, *, source_revision: str = EXPECTED_SOURCE_REVISION):
    state_specs = [{"name": f"s{index}"} for index in range(39)]
    result = {
        "baseline_name": "garch_ccc",
        "mode": "current_realvix_39_state_panel",
        "n_windows": 441,
        "n_samples": 64,
        "state_specs": state_specs,
        "panel_b_crps": {
            "overall": 3.332548859024846,
            "iv_crps": 0.0137,
            "factor_crps": 9.2590,
        },
        "panel_b_energy_score": {"overall": 3.9032458813508333},
        "panel_b_variogram_score": {"variogram_score": 0.20482109487056732},
        "panel_b_correlation": {"corr_score": 0.649},
        "panel_b_factor_ks": {"mean_statistic": 0.123},
        "joint_panel_quality": {
            "finite_rate": 1.0,
            "n_windows": 441,
            "n_samples": 64,
            "future_len": 30,
            "n_factors": 14,
            "factor_tail_q99_ratio_median": 1.062,
        },
        "current_panel_iv_metrics": {
            "cov90": 0.973,
            "calibration_error": 0.207,
            "level_ks_median": 0.177,
            "median_abs_dev": 0.061,
            "width_rho": 0.291,
            "mr_ratio": 0.144,
            "economic_ratio": 1.011,
            "jump_ks": 0.506,
            "kurtosis_ratio": 0.921,
            "acf_corr": 0.977,
        },
    }
    manifest = {
        "mode": "current_realvix_39_state_panel",
        "seed": 772,
        "n_train_windows": 4010,
        "n_val_windows": 441,
        "n_samples": 64,
        "columns": [f"c{index}" for index in range(53)],
        "state_specs": state_specs,
        "metadata": {
            "train_indices_start": 0,
            "train_indices_end": 4009,
            "val_indices_start": 4010,
            "val_indices_end": 4450,
            "iv_transform": "bounded_logit",
            "positive_level_policy": "reference_based",
            "iv_lower_bound": 0.0001,
            "iv_upper_bound": 1.0,
        },
        "provenance": {
            "source_revision": source_revision,
            "source_blob": EXPECTED_SOURCE_BLOB,
            "source_tag": SOURCE_TAG,
        },
    }
    table = """GARCH-CCC & 3.3325 & 0.0137 & 9.2590 & 3.903 & 0.2048 & 0.649 & 0.123 & 1.062 \\\\
GARCH-CCC & 0.973 & 0.207 & 0.177 & 0.061 & 0.291 & 0.144 & 1.011 & 0.506 & 0.921 & 0.977 \\\\
"""
    result_path = root / "result.json"
    manifest_path = root / "manifest.json"
    table_path = root / "table.tex"
    result_path.write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))
    table_path.write_text(table)
    return result_path, manifest_path, table_path


def test_frozen_gaussian_row_validates_precision_identity_and_isolated_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paths = _write_fixture(source)
    report = validate_published_row(
        result_path=paths[0],
        replay_manifest_path=paths[1],
        table_path=paths[2],
        output_dir=tmp_path / "report",
        expected_sha256=tuple(_sha(path) for path in paths),
    )
    assert report["overall_pass"] is True
    assert report["candidate"] == "garch_ccc"
    assert (tmp_path / "report" / "garch_ccc_contract_report.json").is_file()


def test_frozen_gaussian_row_rejects_a_substituted_live_implementation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paths = _write_fixture(source, source_revision="current-live-revision")
    with pytest.raises(ValueError, match="source revision"):
        validate_published_row(
            result_path=paths[0],
            replay_manifest_path=paths[1],
            table_path=paths[2],
            output_dir=tmp_path / "report",
            expected_sha256=tuple(_sha(path) for path in paths),
        )


def test_frozen_gaussian_row_rejects_tamper_and_source_output_overlap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paths = _write_fixture(source)
    hashes = tuple(_sha(path) for path in paths)
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_published_row(
            result_path=paths[0],
            replay_manifest_path=paths[1],
            table_path=paths[2],
            output_dir=tmp_path / "report",
            expected_sha256=hashes,
        )
    with pytest.raises(ValueError, match="isolated"):
        validate_published_row(
            result_path=paths[0],
            replay_manifest_path=paths[1],
            table_path=paths[2],
            output_dir=source / "report",
            expected_sha256=tuple(_sha(path) for path in paths),
        )
