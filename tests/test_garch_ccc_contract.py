from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scenarioflow.evaluation.garch_ccc_contract import (
    EXPECTED_SOURCE_BLOB,
    EXPECTED_SOURCE_REVISION,
    FROZEN_SUMMARY_SHA256,
    FROZEN_TABLE_SHA256,
    SOURCE_TAG,
    build_garch_ccc_contract_report,
    validate_published_row,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _documents(*, source_revision: str = EXPECTED_SOURCE_REVISION):
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
    return result, manifest, table


def test_frozen_gaussian_document_contract_validates_precision_and_identity() -> None:
    result, manifest, table = _documents()
    report = build_garch_ccc_contract_report(
        result=result, manifest=manifest, table_text=table
    )
    assert report["overall_pass"] is True
    assert report["candidate"] == "garch_ccc"
    assert report["authority"]["summary_sha256"] == FROZEN_SUMMARY_SHA256
    assert report["authority"]["table_sha256"] == FROZEN_TABLE_SHA256


def test_frozen_gaussian_row_rejects_a_substituted_live_implementation(
    tmp_path: Path,
) -> None:
    result, manifest, table = _documents(source_revision="current-live-revision")
    with pytest.raises(ValueError, match="source revision"):
        build_garch_ccc_contract_report(
            result=result, manifest=manifest, table_text=table
        )


def test_public_validator_rejects_arbitrary_self_hashed_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    result, manifest, table = _documents()
    summary = source / "summary.json"
    table_path = source / "table.tex"
    summary.write_text(json.dumps({"result": result, "manifest": manifest}))
    table_path.write_text(table)
    with pytest.raises(ValueError, match="frozen summary checksum"):
        validate_published_row(
            summary_path=summary,
            table_path=table_path,
            output_dir=tmp_path / "report",
        )
