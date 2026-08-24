from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from scenarioflow.baselines.classical import (
    JointBootstrap,
    JointPCAVAR,
    JointRandomWalk,
)
from scenarioflow.baselines.deep import CurrentPanelDeepBaseline, build_model
from scenarioflow.baselines.g2 import DeepVARSNIInterfaceBaseline
from scenarioflow.data import official_train_val_indices
from scenarioflow.evaluation.current_panel import (
    BASELINES,
    CURRENT_PANEL_RESULT_KEYS,
    METRIC_PATHS,
    PUBLISHED_VALIDATION_PROTOCOL,
    TABLE_THRESHOLD_RULES,
    WindowHistoricalSim,
    build_run_manifest,
    main as evaluator_main,
    validate_result_schema,
)
from scenarioflow.evaluation.window_scores import build_window_score_artifact


def _tiny_changes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 4)).astype(np.float32)
    history = rng.normal(size=(2, 5, 4)).astype(np.float32)
    return train, history


def test_published_candidate_order_and_validation_split_are_frozen() -> None:
    assert BASELINES == (
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "garch_ccc",
        "garch_dcc",
        "filtered_hs",
        "deepvar",
        "timegrad",
        "path_diffusion",
        "cond_wgan",
        "moirai_1_1_finetuned",
        "toto_1_0_finetuned",
        "sni_joint",
    )
    assert asdict(PUBLISHED_VALIDATION_PROTOCOL) == {
        "mode": "current_realvix_39_state_panel",
        "state_count": 39,
        "iv_count": 25,
        "history_len": 30,
        "future_len": 30,
        "test_start": 4511,
        "val_size": 441,
        "n_samples": 64,
        "evaluation_seed": 772,
    }
    train, val = official_train_val_indices(
        test_start=4511, val_size=441, history_len=30, future_len=30
    )
    assert (len(train), train[0], train[-1]) == (4010, 0, 4009)
    assert (len(val), val[0], val[-1]) == (441, 4010, 4450)


@pytest.mark.parametrize("factory", [JointRandomWalk, JointBootstrap, JointPCAVAR])
def test_tiny_classical_adapters_are_seeded_and_shape_stable(factory: type) -> None:
    train, history = _tiny_changes()
    left = factory(train, future_len=3, seed=11).sample_joint(history, n_samples=5)
    right = factory(train, future_len=3, seed=11).sample_joint(history, n_samples=5)
    assert left.shape == (2, 5, 3, 4)
    np.testing.assert_array_equal(left, right)
    assert np.isfinite(left).all()


def test_tiny_deep_and_g2_adapters_are_seeded_and_shape_stable() -> None:
    _, history = _tiny_changes()
    torch.manual_seed(5)
    model = build_model(
        "deepvar", input_dim=4, future_len=3, hidden_dim=8, num_layers=1, rank=2
    )
    baseline = CurrentPanelDeepBaseline(
        model,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        model_type="deepvar",
        future_len=3,
        seed=17,
    )
    first = baseline.sample_joint(history[:1], n_samples=2)
    second = baseline.sample_joint(history[1:], n_samples=2)
    assert first.shape == second.shape == (1, 2, 3, 4)
    assert not np.array_equal(first, second)

    torch.manual_seed(5)
    full_model = build_model(
        "deepvar", input_dim=4, future_len=3, hidden_dim=8, num_layers=1, rank=2
    )
    full = CurrentPanelDeepBaseline(
        full_model,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        model_type="deepvar",
        future_len=3,
        seed=17,
    ).sample_joint(history, n_samples=2)
    np.testing.assert_allclose(np.concatenate([first, second]), full, rtol=0, atol=0)

    g2 = DeepVARSNIInterfaceBaseline(
        model,
        scale_half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
        future_len=3,
        seed=17,
    )
    g2_first = g2.sample_joint(history[:1], n_samples=2)
    g2_second = g2.sample_joint(history[1:], n_samples=2)
    assert g2_first.shape == g2_second.shape == (1, 2, 3, 4)
    assert not np.array_equal(g2_first, g2_second)

    torch.manual_seed(5)
    g2_full_model = build_model(
        "deepvar", input_dim=4, future_len=3, hidden_dim=8, num_layers=1, rank=2
    )
    g2_full = DeepVARSNIInterfaceBaseline(
        g2_full_model,
        scale_half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
        future_len=3,
        seed=17,
    ).sample_joint(history, n_samples=2)
    np.testing.assert_allclose(
        np.concatenate([g2_first, g2_second]), g2_full, rtol=0, atol=0
    )


def test_tiny_state_matched_historical_adapter_is_seeded_and_shape_stable() -> None:
    rng = np.random.default_rng(4)
    history_increment = rng.normal(size=(6, 5, 39)).astype(np.float32)
    future_increment = rng.normal(size=(6, 3, 39)).astype(np.float32)
    history_state = rng.uniform(0.1, 0.5, size=(6, 5, 39)).astype(np.float32)
    model = WindowHistoricalSim(
        history_increment,
        future_increment,
        history_state,
        iv_count=25,
        seed=9,
    )
    first = model.sample_joint(
        history_increment[:2], n_samples=4, history_state=history_state[:2]
    )
    second_model = WindowHistoricalSim(
        history_increment,
        future_increment,
        history_state,
        iv_count=25,
        seed=9,
    )
    second = second_model.sample_joint(
        history_increment[:2], n_samples=4, history_state=history_state[:2]
    )
    assert first.shape == (2, 4, 3, 39)
    np.testing.assert_array_equal(first, second)


def test_result_manifest_metric_and_screen_contracts_are_exact() -> None:
    assert tuple(TABLE_THRESHOLD_RULES) == (
        "Cov90",
        "CalErr",
        "Level KS",
        "Width rho",
        "MR ratio",
        "Econ ratio",
        "Jump KS",
        "Kurtosis",
        "Corr score",
        "Factor KS",
        "Tail q99",
    )
    assert METRIC_PATHS[0] == ("CRPS", ("panel_b_crps", "overall"))
    assert METRIC_PATHS[-1] == ("ACF corr", ("current_panel_iv_metrics", "acf_corr"))
    assert "schema_version" not in CURRENT_PANEL_RESULT_KEYS

    leaf = {
        "cov90": 0.9,
        "calibration_error": 0.1,
        "level_ks_median": 0.1,
        "median_abs_dev": 0.1,
        "width_rho": 0.3,
        "mr_ratio": 1.0,
        "economic_ratio": 1.0,
        "jump_ks": 0.2,
        "kurtosis_ratio": 1.0,
        "acf_corr": 0.9,
    }
    result = {key: {} for key in CURRENT_PANEL_RESULT_KEYS}
    result.update(
        baseline_name="bootstrap",
        mode="current_realvix_39_state_panel",
        generation_time_s=0.1,
        n_windows=441,
        n_samples=64,
        state_specs=[{}] * 39,
        current_panel_iv_metrics=leaf,
        panel_b_crps={"overall": 1.0, "iv_crps": 1.0, "factor_crps": 1.0},
        panel_b_energy_score={"overall": 1.0},
        panel_b_variogram_score={"variogram_score": 1.0},
        panel_b_correlation={"corr_score": 0.5},
        panel_b_factor_ks={"mean_statistic": 0.1},
        joint_panel_quality={"factor_tail_q99_ratio_median": 1.0},
    )
    validate_result_schema(result)
    manifest = build_run_manifest(
        columns=[f"column-{index}" for index in range(53)],
        metadata={},
        state_specs=[{}] * 39,
        n_train_windows=4010,
        n_val_windows=441,
        n_samples=64,
        seed=772,
        history_alignment_error=0.0,
        future_alignment_error=0.0,
    )
    assert manifest["benchmark_split"] == "published_validation"
    assert tuple(manifest) == (
        "mode",
        "benchmark_split",
        "columns",
        "metadata",
        "state_specs",
        "n_train_windows",
        "n_val_windows",
        "n_samples",
        "seed",
        "iv_alignment",
    )


def test_portable_evaluator_cli_executes_a_bounded_committed_panel_run(
    tmp_path,
) -> None:
    evaluator_main(
        [
            "--baseline",
            "bootstrap",
            "--data-dir",
            "data",
            "--output-dir",
            str(tmp_path),
            "--max-windows",
            "2",
            "--n-samples",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "772",
        ]
    )
    result = __import__("json").loads(
        (tmp_path / "bootstrap" / "results_current_panel.json").read_text()
    )
    assert "schema_version" not in result
    assert result["n_windows"] == 2
    assert result["n_samples"] == 2
    assert set(result) == set(CURRENT_PANEL_RESULT_KEYS)
    validate_result_schema(result, canonical_counts=False)


def test_window_score_deck_reconciles_to_published_aggregates() -> None:
    samples = np.zeros((1, 2, 30, 39), dtype=np.float32)
    truth = np.zeros((1, 30, 39), dtype=np.float32)
    result = {
        "panel_b_crps": {
            "per_horizon": {str(h): [0.0] * 39 for h in (1, 7, 14, 30)},
            "per_dimension": [0.0] * 39,
            "overall": 0.0,
            "iv_crps": 0.0,
            "factor_crps": 0.0,
        },
        "panel_b_energy_score": {
            "per_horizon": {str(h): 0.0 for h in (1, 7, 14, 30)},
            "overall": 0.0,
        },
        "panel_b_variogram_score": {"variogram_score": 0.0},
        "current_panel_iv": {
            "coverage": {
                "overall": {str(level): 1.0 for level in (0.5, 0.8, 0.9, 0.95)}
            },
            "risk_state_allocation": {"history_width_spearman": 0.0},
        },
        "joint_panel_quality": {
            "conditional_panel": {"history_activity_width_spearman": float("nan")}
        },
    }
    artifact = build_window_score_artifact(
        samples_delta=samples,
        ground_truth_delta=truth,
        train_delta=np.zeros((3, 39), dtype=np.float32),
        conditional_iv_samples=np.zeros((1, 2, 30, 5, 5), dtype=np.float32),
        ground_truth_iv=np.zeros((1, 30, 5, 5), dtype=np.float32),
        history_iv=np.zeros((1, 30, 5, 5), dtype=np.float32),
        raw_history=np.zeros((1, 30, 39), dtype=np.float32),
        sample_raw=samples,
        window_indices=np.array([4010]),
        variable_names=[f"state-{index}" for index in range(39)],
        existing_result=result,
    )
    assert artifact["schema_version"].item() == "fmts.window_scores.v1"
    np.testing.assert_array_equal(artifact["horizons"], [1, 7, 14, 30])
    np.testing.assert_array_equal(artifact["coverage_levels"], [0.5, 0.8, 0.9, 0.95])
