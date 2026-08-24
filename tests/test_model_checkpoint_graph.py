from __future__ import annotations

from dataclasses import asdict

import torch

from scenarioflow.model import (
    AdaptiveGraphResidualTokenTransitionVelocity,
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_adaptive_graph_velocity_mixer,
    load_model,
    save_checkpoint,
)


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
