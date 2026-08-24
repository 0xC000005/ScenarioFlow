from __future__ import annotations

import numpy as np

from scenarioflow.state import (
    UnifiedVariableSpec,
    build_increment_coordinate_block,
    build_unified_variable_specs,
    encode_state,
    normalize_increment_windows,
    reconstruct_state_from_normalized_increments,
)


def test_state_coordinate_and_history_only_normalization_roundtrip() -> None:
    panel = np.array(
        [
            [0.20, 100.0, 5.0, 0.00, 0.00],
            [0.22, 110.0, 5.5, 0.10, 0.50],
            [0.21, 121.0, 5.3, 0.10, -0.20],
            [0.24, 133.1, 5.8, 0.10, 0.50],
            [0.25, 146.41, 6.0, 0.10, 0.20],
        ],
        dtype=np.float32,
    )
    columns = [
        "iv:00",
        "factor:spx",
        "factor:rate",
        "factor:spx_logret",
        "factor:rate_diff",
    ]
    block = build_increment_coordinate_block(
        panel, columns, indices=[0], history_len=2, future_len=3, iv_count=1
    )
    history_norm, future_norm, center, scale = normalize_increment_windows(
        block.history_increment, block.future_increment, center_mode="ewma_mean"
    )
    reconstructed = reconstruct_state_from_normalized_increments(
        block.history_state[:, -1], future_norm, center, scale, block.specs
    )

    assert [spec.name for spec in block.specs] == [
        "iv:00",
        "factor:spx",
        "factor:rate",
    ]
    assert history_norm.shape == (1, 2, 3)
    assert np.allclose(reconstructed, block.future_state, rtol=1e-5, atol=1e-5)


def test_bounded_logit_transform_uses_frozen_bounds() -> None:
    specs = [
        UnifiedVariableSpec(
            name="iv:00",
            source_column="iv:00",
            source_index=0,
            transform="bounded_logit",
            lower_bound=0.0001,
            upper_bound=1.0,
        )
    ]
    encoded = encode_state(np.array([[0.50005]], dtype=np.float64), specs)
    assert np.allclose(encoded, 0.0, atol=1e-12)


def test_observed_positive_policy_keeps_negative_factor_in_difference_space() -> None:
    columns = ["iv:00", "factor:spread", "factor:spread_diff"]
    panel = np.array([[0.2, 1.0, 0.0], [0.2, -0.1, -1.1]], dtype=np.float32)
    specs = build_unified_variable_specs(
        columns, panel=panel, iv_count=1, positive_level_policy="observed_positive"
    )
    assert [spec.transform for spec in specs] == ["log_level", "diff_level"]
