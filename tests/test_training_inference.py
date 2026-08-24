from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from scenarioflow.inference import sample_normalized_innovations
from scenarioflow.model import save_checkpoint
from scenarioflow.training import (
    initialize_level_quantiles,
    stage1_training_step,
    stage2_training_step,
)

from test_model_checkpoint_graph import tiny_model


def synthetic_batch(model):
    generator = torch.Generator().manual_seed(17)
    bsz = 2
    history_norm = torch.randn(bsz, 2, 3, generator=generator) * 0.1
    future_norm = torch.randn(bsz, 2, 3, generator=generator) * 0.1
    history_level = torch.randn(bsz, 2, 3, generator=generator) * 0.1
    future_level = history_level[:, -1:, :] + torch.cumsum(future_norm, dim=1)
    center = torch.zeros(bsz, 3)
    scale = torch.ones(bsz, 3)
    return history_level, history_norm, future_level, future_norm, center, scale


def test_reusable_stage1_and_stage2_training_steps_are_finite() -> None:
    model = tiny_model()
    batch = synthetic_batch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    stage1 = stage1_training_step(model, optimizer, *batch)
    stage2 = stage2_training_step(
        model,
        optimizer,
        *batch,
        train_sample_count=2,
        rollout_flow_steps=1,
        energy_weight=0.1,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=0.5,
    )
    assert np.isfinite(stage1["total"])
    assert np.isfinite(stage2["total"])


def test_new_stage1_model_uses_frozen_empirical_quantile_grid() -> None:
    model = tiny_model()
    model.cfg.cdf_eps = 0.1
    model.cfg.n_quantiles = 5
    model.quantile_levels = torch.zeros(5)
    model.level_quantiles = torch.zeros(model.cfg.n_cells, 5)
    model._quantiles_ready.fill_(False)
    history_level = torch.arange(0.0, 6.0).view(1, 6, 1).repeat(1, 1, 3)
    future_level = torch.arange(6.0, 11.0).view(1, 5, 1).repeat(1, 1, 3)
    initialize_level_quantiles(model, history_level, future_level)
    assert bool(model._quantiles_ready.item())
    expected_grid = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    expected_quantiles = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]).repeat(3, 1)
    assert torch.allclose(model.quantile_levels, expected_grid)
    assert torch.allclose(model.level_quantiles, expected_quantiles)


def test_synthetic_checkpoint_inference_is_deterministic(tmp_path) -> None:
    model = tiny_model().eval()
    batch = synthetic_batch(model)
    path = tmp_path / "synthetic.pt"
    save_checkpoint(path, model, model.cfg, epoch=0, best_val=0.0)
    kwargs = dict(
        checkpoint=path,
        history_level_values=batch[0],
        history_normalized_innovation=batch[1],
        center=batch[4],
        scale=batch[5],
        n_samples=2,
        n_steps=2,
        flow_steps=2,
        temperature=0.5,
        seed=23,
        device="cpu",
    )
    first = sample_normalized_innovations(**kwargs)
    second = sample_normalized_innovations(**kwargs)
    assert first.shape == (2, 2, 2, 3)
    assert torch.equal(first, second)


def test_training_and_inference_module_help() -> None:
    for module in ("scenarioflow.training", "scenarioflow.inference"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
