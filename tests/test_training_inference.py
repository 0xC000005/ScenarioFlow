from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scenarioflow.inference import sample_normalized_innovations
from scenarioflow.model import save_checkpoint
from scenarioflow.training import (
    _panel_batches,
    initialize_level_quantiles,
    stage1_training_step,
    stage2_training_step,
)
import scenarioflow.workflows.reproduction as reproduction

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


def test_deep_trainer_saves_strict_best_validation_epoch_not_last(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(41)
    block = SimpleNamespace(
        history_increment=rng.normal(size=(2, 30, 3)).astype(np.float32),
        future_increment=rng.normal(size=(2, 30, 3)).astype(np.float32),
    )
    monkeypatch.setattr(
        reproduction,
        "prepare_validation_blocks",
        lambda **_kwargs: (None, None, block, block),
    )

    class ValidationSequenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.validation_losses = iter((1.0, 0.5, 0.8))

        def forward(self, history, future):
            if self.training:
                return (self.weight - 1.0).square() + history.mean() * 0.0
            return torch.tensor(next(self.validation_losses))

    monkeypatch.setattr(
        reproduction, "build_model", lambda *_args, **_kwargs: ValidationSequenceModel()
    )
    saved = []

    def save(path, _model, **kwargs):
        saved.append(kwargs["best_val_loss"])
        torch.save({"best_val_loss": kwargs["best_val_loss"]}, path)

    monkeypatch.setattr(reproduction, "save_checkpoint", save)
    checkpoint = tmp_path / "best.pt"
    reproduction.train_baseline(
        candidate="deepvar",
        data_dir=tmp_path,
        checkpoint_out=checkpoint,
        train_seed=1001,
        epochs=3,
        batch_size=2,
    )
    assert saved == [1.0, 0.5]
    assert torch.load(checkpoint, weights_only=False)["best_val_loss"] == 0.5


def test_scenarioflow_panel_batches_preserve_all_441_validation_windows(
    monkeypatch,
) -> None:
    train = SimpleNamespace(
        history_increment=np.zeros((3, 2, 3), dtype=np.float32),
        future_increment=np.zeros((3, 2, 3), dtype=np.float32),
        history_state=np.ones((3, 2, 3), dtype=np.float32),
        future_state=np.ones((3, 2, 3), dtype=np.float32),
    )
    validation = SimpleNamespace(
        history_increment=np.zeros((441, 2, 3), dtype=np.float32),
        future_increment=np.zeros((441, 2, 3), dtype=np.float32),
        history_state=np.ones((441, 2, 3), dtype=np.float32),
        future_state=np.ones((441, 2, 3), dtype=np.float32),
    )
    import scenarioflow.evaluation.current_panel as current_panel

    monkeypatch.setattr(
        current_panel,
        "prepare_validation_blocks",
        lambda **_kwargs: (None, None, train, validation),
    )
    _train_batch, validation_batch = _panel_batches(
        __import__("pathlib").Path("unused"), torch.device("cpu")
    )
    assert validation_batch[0].shape[0] == 441
