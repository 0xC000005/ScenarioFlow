"""Regression contracts for the extracted ScenarioFlow numerical routes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scenarioflow.reproduction import finetune_rollouts, train_next_step
from scenarioflow.reproduction.inference import (
    generate_deck,
    load_checkpoint_strict,
    main as inference_main,
)
from scenarioflow.reproduction.panel import (
    prepare_panel_blocks,
    select_normalized_innovation_scope,
)
from scenarioflow.state import reconstruct_state_from_increments

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def two_window_inference_inputs():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _columns, _metadata, _train, validation = prepare_panel_blocks(
        data_dir=ROOT / "data"
    )
    selected = select_normalized_innovation_scope(
        validation,
        "joint38",
        25,
        scale_half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )
    arrays = tuple(
        value[:2] if isinstance(value, np.ndarray) else value for value in selected
    )
    model, _payload = load_checkpoint_strict(
        ROOT / "artifacts/cache/models/core/stage2/best_model.pt",
        torch.device("cpu"),
    )
    return model, arrays


def test_generate_deck_batch_size_one_matches_frozen_window_batching(
    two_window_inference_inputs,
) -> None:
    """All-at-once sampling must not replace the donor's window batches."""
    model, selected = two_window_inference_inputs
    (
        history_level,
        history_norm,
        _future_level,
        _future_norm,
        center,
        scale,
        drift,
        history_raw,
        specs,
    ) = selected
    actual = generate_deck(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift,
        history_raw,
        specs,
        samples=2,
        n_steps=30,
        batch_size=1,
        chunk_size=2,
        temperature=1.0,
        seed=2001,
        device=torch.device("cpu"),
    )

    torch.manual_seed(2001)
    donor_chunks = []
    with torch.no_grad():
        for start in range(2):
            end = start + 1
            increment = model.sample_batched(
                torch.from_numpy(history_level[start:end]),
                torch.from_numpy(history_norm[start:end]),
                torch.from_numpy(center[start:end]),
                torch.from_numpy(scale[start:end]),
                drift_feature=torch.from_numpy(drift[start:end]),
                n_samples=2,
                n_steps=30,
                chunk_size=2,
                temperature=1.0,
            ).numpy()
            donor_chunks.append(
                reconstruct_state_from_increments(
                    history_raw[start:end, -1], increment, specs
                ).astype(np.float32)
            )
    donor_batched = np.concatenate(donor_chunks, axis=0)

    torch.manual_seed(2001)
    with torch.no_grad():
        all_at_once_increment = model.sample_batched(
            torch.from_numpy(history_level),
            torch.from_numpy(history_norm),
            torch.from_numpy(center),
            torch.from_numpy(scale),
            drift_feature=torch.from_numpy(drift),
            n_samples=2,
            n_steps=30,
            chunk_size=2,
            temperature=1.0,
        ).numpy()
    all_at_once = reconstruct_state_from_increments(
        history_raw[:, -1], all_at_once_increment, specs
    ).astype(np.float32)

    np.testing.assert_array_equal(actual, donor_batched)
    assert not np.array_equal(actual, all_at_once)


def test_inference_cli_batch_size_one_uses_frozen_batch_assignment(
    tmp_path: Path,
) -> None:
    """The public --batch-size option must reach the donor batching loop."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inference_main(
        [
            "--checkpoint",
            str(ROOT / "artifacts/cache/models/core/stage2/best_model.pt"),
            "--data-dir",
            str(ROOT / "data"),
            "--output-dir",
            str(tmp_path),
            "--max-windows",
            "2",
            "--samples",
            "2",
            "--n-steps",
            "30",
            "--batch-size",
            "1",
            "--chunk-size",
            "2",
            "--seed",
            "2001",
            "--device",
            "cpu",
        ]
    )

    with np.load(tmp_path / "scenarios.npz", allow_pickle=False) as archive:
        deck = archive["scenarios"]
    assert deck.shape == (2, 2, 30, 39)
    assert (
        hashlib.sha256(deck.tobytes(order="C")).hexdigest()
        == "9afc78afa81703da5b2755f1ca7ed551dd9b57a6ece87929e92e7b5bbf49f137"
    )


def test_stage1_checkpoint_metadata_preserves_frozen_payload() -> None:
    """Reducing the donor metadata must fail this complete payload contract."""
    args = SimpleNamespace(
        state_scope="joint38",
        center_mode="zero",
        drift_feature_mode="none",
        scale_floor=1e-4,
        iv_transform="bounded_logit",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        innovation_coordinate="normalized",
        sticky_observation_loss="none",
        sticky_observation_zero_weight=0.0,
        mixed_support_observation="none",
        mixed_support_zero_eps=1e-10,
        mixed_support_zero_rate_gate=0.25,
        mixed_support_loss_weight=1.0,
        condition_contrast_weight=0.0,
        condition_contrast_margin=0.0,
        risk_state_dim=0,
        risk_state_weight=0.0,
        risk_state_rank_weight=0.0,
        iv_count=25,
        positive_level_policy="reference_based",
    )
    cfg = SimpleNamespace(
        base_noise_rho=0.0,
        conditional_base_noise_scale=False,
        base_noise_scale_min=0.5,
        base_noise_scale_max=2.0,
    )
    sticky = {"policy": "disabled", "selected_names": []}
    tail = {"policy": "disabled", "selected_names": []}
    mixed = {"policy": "disabled", "selected_names": []}
    state_specs = [{"name": "iv_1"}]
    panel_metadata = {"val_indices_start": 4010}

    actual = train_next_step._checkpoint_metadata(
        args,
        cfg,
        scale_half_life=None,
        sticky_score_report=sticky,
        tail_asinh_report=tail,
        mixed_support_report=mixed,
        train_future_weight=np.ones((2, 30, 39), dtype=np.float32),
        val_future_weight=np.ones((2, 30, 39), dtype=np.float32),
        train_no_update_target=np.zeros((2, 30, 39), dtype=np.float32),
        val_no_update_target=np.zeros((2, 30, 39), dtype=np.float32),
        state_specs=state_specs,
        panel_metadata=panel_metadata,
    )

    normalization = {
        "coordinate": "encoded_increment",
        "center_mode": "zero",
        "drift_feature_mode": "none",
        "scale_method": "ewma_rms",
        "scale_half_life": None,
        "scale_floor": 1e-4,
        "iv_transform": "bounded_logit",
        "iv_lower_bound": 1e-4,
        "iv_upper_bound": 1.0,
        "innovation_coordinate": "normalized",
        "sticky_score": sticky,
        "tail_asinh": tail,
        "sticky_observation_loss": "none",
        "sticky_observation_zero_weight": 0.0,
        "mixed_support": mixed,
        "mixed_support_observation": "none",
        "mixed_support_zero_eps": 1e-10,
        "mixed_support_zero_rate_gate": 0.25,
        "mixed_support_loss_weight": 1.0,
        "train_future_weight_mean": 1.0,
        "val_future_weight_mean": 1.0,
        "train_no_update_target_mean": 0.0,
        "val_no_update_target_mean": 0.0,
    }
    assert actual == {
        "state_scope": "joint38",
        "model_coordinate": "state_aware_normalized_innovation",
        "normalization": normalization,
        "training_objective": {
            "base": "flow_matching_mse",
            "condition_contrast_weight": 0.0,
            "condition_contrast_margin": 0.0,
            "risk_state_dim": 0,
            "risk_state_weight": 0.0,
            "risk_state_rank_weight": 0.0,
            "base_noise_rho": 0.0,
            "conditional_base_noise_scale": False,
            "base_noise_scale_min": 0.5,
            "base_noise_scale_max": 2.0,
            "flow_coordinate": "normalized",
            "sticky_score": sticky,
            "tail_asinh": tail,
            "sticky_observation_loss": "none",
            "sticky_observation_zero_weight": 0.0,
            "mixed_support": mixed,
            "mixed_support_observation": "none",
            "mixed_support_loss_weight": 1.0,
        },
        "iv_transform": "bounded_logit",
        "iv_lower_bound": 1e-4,
        "iv_upper_bound": 1.0,
        "iv_count": 25,
        "state_specs": state_specs,
        "panel_metadata": panel_metadata,
        "positive_level_policy": "reference_based",
        "sticky_score": sticky,
        "tail_asinh": tail,
        "mixed_support": mixed,
    }


def test_stage2_objective_metadata_preserves_graph_and_loss_provenance() -> None:
    """Dropping active loss, noise, sampler, or graph fields must fail."""
    args = SimpleNamespace(
        train_sample_count=2,
        rollout_flow_steps=4,
        energy_weight=0.2,
        marginal_crps_weight=0.03,
        interval_score_weight=0.04,
        interval_alpha=0.1,
        variogram_weight=0.05,
        variogram_power=0.5,
        level_marginal_crps_weight=0.06,
        level_marginal_crps_coordinate="scaled_delta",
        level_energy_weight=0.07,
        channel_level_energy_weight=0.08,
        channel_level_energy_coordinate="level",
        condition_rollout_contrast_weight=0.09,
        condition_rollout_contrast_margin=0.1,
        condition_rollout_negative_mode="roll",
        risk_state_weight=0.11,
        risk_state_rank_weight=0.12,
        mixed_support_loss_weight=0.13,
        dispersion_calibration_weight=0.14,
        dispersion_calibration_mode="window",
        free_running_fm_weight=0.15,
        free_running_fm_prefix_steps=5,
        state_tail_sampler_weight=0.16,
        state_tail_sampler_quantile=0.8,
        velocity_readout_mode="shared",
        adaptive_graph_k=7,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(
            mixed_support_observation="none",
            base_noise_rho=0.17,
            conditional_base_noise_scale=True,
            base_noise_scale_min=0.5,
            base_noise_scale_max=2.0,
            velocity_mixer_mode="adaptive_graph_residual",
            adaptive_graph_k=7,
        )
    )

    actual = finetune_rollouts._finetune_objective(
        args,
        model,
        np.asarray([1.0, 1.5], dtype=np.float32),
    )

    assert actual == {
        "base": "flow_matching_mse",
        "rollout_energy_coordinate": "normalized_innovation",
        "train_sample_count": 2,
        "rollout_flow_steps": 4,
        "energy_weight": 0.2,
        "marginal_crps_weight": 0.03,
        "interval_score_weight": 0.04,
        "interval_alpha": 0.1,
        "variogram_weight": 0.05,
        "variogram_power": 0.5,
        "level_marginal_crps_weight": 0.06,
        "level_marginal_crps_coordinate": "scaled_delta",
        "level_energy_weight": 0.07,
        "channel_level_energy_weight": 0.08,
        "channel_level_energy_coordinate": "level",
        "condition_rollout_contrast_weight": 0.09,
        "condition_rollout_contrast_margin": 0.1,
        "condition_rollout_negative_mode": "roll",
        "risk_state_weight": 0.11,
        "risk_state_rank_weight": 0.12,
        "mixed_support_loss_weight": 0.13,
        "mixed_support_observation": "none",
        "dispersion_calibration_weight": 0.14,
        "dispersion_calibration_mode": "window",
        "free_running_fm_weight": 0.15,
        "free_running_fm_prefix_steps": 5,
        "state_tail_sampler_weight": 0.16,
        "state_tail_sampler_quantile": 0.8,
        "state_tail_sampler_mean": 1.25,
        "state_tail_sampler_max": 1.5,
        "base_noise_rho": 0.17,
        "conditional_base_noise_scale": True,
        "base_noise_scale_min": 0.5,
        "base_noise_scale_max": 2.0,
        "velocity_readout_mode": "shared",
        "velocity_mixer_mode": "adaptive_graph_residual",
        "adaptive_graph_k": 7,
        "fm_anchor_weight": 1.0,
        "horizon_end_weight": 1.2,
    }


def test_finetune_command_applies_nondefault_settings_before_recording(
    tmp_path: Path,
) -> None:
    """A real command must not record tail/noise/readout settings it ignored."""
    output_dir = tmp_path / "finetune"
    finetune_rollouts.main(
        [
            "--checkpoint",
            str(ROOT / "artifacts/cache/models/core/stage2/best_model.pt"),
            "--data-dir",
            str(ROOT / "data"),
            "--output_dir",
            str(output_dir),
            "--max_train_windows",
            "8",
            "--epochs",
            "0",
            "--batch_size",
            "4",
            "--state_tail_sampler_weight",
            "2.0",
            "--state_tail_sampler_quantile",
            "0.5",
            "--base_noise_rho",
            "0.3",
            "--conditional_base_noise_scale",
            "--base_noise_scale_min",
            "0.7",
            "--base_noise_scale_max",
            "1.8",
            "--velocity_readout_mode",
            "group_residual",
            "--device",
            "cpu",
        ]
    )

    payload = torch.load(
        output_dir / "final_model.pt", map_location="cpu", weights_only=False
    )
    config = payload["config"]
    objective = payload["finetune_objective"]
    state_keys = set(payload["model_state_dict"])

    assert config["base_noise_rho"] == 0.3
    assert config["conditional_base_noise_scale"] is True
    assert config["base_noise_scale_min"] == 0.7
    assert config["base_noise_scale_max"] == 1.8
    assert config["velocity_readout_mode"] == "group_residual"
    assert config["readout_iv_count"] == 25
    assert objective["base_noise_rho"] == config["base_noise_rho"]
    assert objective["conditional_base_noise_scale"] is True
    assert objective["base_noise_scale_min"] == config["base_noise_scale_min"]
    assert objective["base_noise_scale_max"] == config["base_noise_scale_max"]
    assert objective["velocity_readout_mode"] == config["velocity_readout_mode"]
    assert objective["state_tail_sampler_weight"] == 2.0
    assert objective["state_tail_sampler_quantile"] == 0.5
    assert objective["state_tail_sampler_mean"] == pytest.approx(1.0)
    assert objective["state_tail_sampler_max"] > 1.0
    assert any(name.startswith("base_noise_log_scale.") for name in state_keys)
    assert any(name.startswith("velocity.iv_head.") for name in state_keys)
    assert any(name.startswith("velocity.anchor_head.") for name in state_keys)
