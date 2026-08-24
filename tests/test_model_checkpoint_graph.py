from __future__ import annotations

from dataclasses import asdict
import inspect

import pytest
import torch

import scenarioflow._causal as causal_primitives
from scenarioflow.model import (
    AdaptiveGraphResidualTokenTransitionVelocity,
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_adaptive_graph_velocity_mixer,
    load_model,
    save_checkpoint,
)
from scenarioflow.training import load_training_model


def tiny_model() -> GenericStateAwareNormalizedInnovationFlowMatching:
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=2,
        future_len=2,
        n_cells=3,
        token_dim=8,
        token_layers=1,
        token_heads=1,
        token_ff=16,
        memory_dim=8,
        memory_layers=1,
        memory_heads=1,
        memory_ff=16,
        model_dropout=0.0,
        time_dim=4,
        flow_steps=2,
        n_quantiles=9,
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    quantiles = torch.linspace(-2.0, 2.0, cfg.n_quantiles).repeat(cfg.n_cells, 1)
    model.set_level_quantiles(quantiles)
    return model


def test_checkpoint_roundtrip_preserves_exact_config_and_state_dict_keys(
    tmp_path,
) -> None:
    model = tiny_model()
    path = tmp_path / "synthetic.pt"
    save_checkpoint(
        path, model, model.cfg, epoch=4, best_val=0.25, extra={"state_scope": "joint38"}
    )
    loaded, payload = load_model(path, torch.device("cpu"))
    assert payload["config"] == asdict(model.cfg)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_adaptive_graph_upgrade_is_output_preserving_and_checkpoint_loadable(
    tmp_path,
) -> None:
    model = tiny_model().eval()
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(3))
    current = torch.randn(4, 3, generator=torch.Generator().manual_seed(4))
    memory = torch.randn(4, 8, generator=torch.Generator().manual_seed(5))
    time = torch.linspace(0.1, 0.9, 4)
    before = model.velocity(x, current, memory, time)
    enable_adaptive_graph_velocity_mixer(model, k_neighbors=2)
    after = model.velocity(x, current, memory, time)
    assert isinstance(model.velocity, AdaptiveGraphResidualTokenTransitionVelocity)
    assert torch.equal(before, after)

    path = tmp_path / "graph.pt"
    save_checkpoint(path, model, model.cfg, epoch=1, best_val=1.0)
    loaded, _ = load_model(path, torch.device("cpu"))
    assert isinstance(loaded.velocity, AdaptiveGraphResidualTokenTransitionVelocity)
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_graph_training_loads_ordinary_stage2_then_applies_output_preserving_upgrade(
    tmp_path,
) -> None:
    ordinary = tiny_model().eval()
    checkpoint = tmp_path / "stage2.pt"
    save_checkpoint(checkpoint, ordinary, ordinary.cfg, epoch=3, best_val=0.25)
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(31))
    current = torch.randn(4, 3, generator=torch.Generator().manual_seed(32))
    memory = torch.randn(4, 8, generator=torch.Generator().manual_seed(33))
    time = torch.linspace(0.1, 0.9, 4)
    expected = ordinary.velocity(x, current, memory, time)
    migrated, payload = load_training_model(
        checkpoint, stage="graph", device=torch.device("cpu"), graph_k=2
    )
    actual = migrated.velocity(x, current, memory, time)
    assert torch.equal(expected, actual)
    assert isinstance(migrated.velocity, AdaptiveGraphResidualTokenTransitionVelocity)
    assert payload["graph_migration"] == {
        "source": "ordinary_stage2_checkpoint",
        "method": "output_preserving_zero_residual",
        "k_neighbors": 2,
    }


def test_loader_accepts_false_narrative_metadata_but_rejects_enabled_metadata(
    tmp_path,
) -> None:
    model = tiny_model()
    base_path = tmp_path / "base.pt"
    save_checkpoint(base_path, model, model.cfg, epoch=0, best_val=0.0)
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    payload["config"].update(
        {
            "narrative_conditioning": False,
            "narrative_dim": 1536,
            "narrative_hidden": 512,
        }
    )
    false_path = tmp_path / "false_metadata.pt"
    torch.save(payload, false_path)
    loaded, _ = load_model(false_path, torch.device("cpu"))
    assert set(loaded.state_dict()) == set(model.state_dict())

    payload["config"]["narrative_conditioning"] = True
    enabled_path = tmp_path / "enabled_metadata.pt"
    torch.save(payload, enabled_path)
    with pytest.raises(ValueError, match="numerical package does not support"):
        load_model(enabled_path, torch.device("cpu"))
    with pytest.raises(ValueError, match="numerical package does not support"):
        load_model(
            enabled_path,
            torch.device("cpu"),
            cfg_overrides={"narrative_conditioning": False},
        )


def test_numerical_model_has_no_narrative_execution_surface() -> None:
    model = tiny_model()
    assert not hasattr(causal_primitives, "NarrativeConditioningAdapter")
    assert not hasattr(model, "narrative_adapter")
    for method in (model.training_loss, model.sample_batched):
        parameters = inspect.signature(method).parameters
        assert "narrative_emb" not in parameters
        assert "narrative_present" not in parameters
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
