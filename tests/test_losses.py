from __future__ import annotations

import torch

from scenarioflow.losses import (
    full_path_energy_score,
    horizon_path_weights,
    marginal_crps_path_score,
    sliced_projection_wasserstein_score,
)


def test_full_path_energy_prefers_target_aligned_samples() -> None:
    target = torch.zeros(3, 4, 2)
    generator = torch.Generator().manual_seed(11)
    good = 0.01 * torch.randn(3, 5, 4, 2, generator=generator)
    bad = good + 2.0
    weights = horizon_path_weights(
        4, end_weight=1.2, device=target.device, dtype=target.dtype
    )
    good_score, _, _ = full_path_energy_score(good, target, horizon_weights=weights)
    bad_score, _, _ = full_path_energy_score(bad, target, horizon_weights=weights)
    assert good_score < bad_score


def test_two_member_marginal_crps_matches_hand_computation() -> None:
    samples = torch.tensor([[[[0.0]], [[2.0]]]])
    target = torch.tensor([[[1.0]]])
    score, target_dist, pair_dist = marginal_crps_path_score(samples, target)
    assert torch.allclose(target_dist, torch.tensor(1.0))
    assert torch.allclose(pair_dist, torch.tensor(1.0))
    assert torch.allclose(score, torch.tensor(0.5))


def test_sliced_wasserstein_is_differentiable_and_shift_sensitive() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(4, 3, 2, generator=generator)
    samples = (
        target[:, None] + 0.05 * torch.randn(4, 6, 3, 2, generator=generator)
    ).requires_grad_(True)
    projection_generator = torch.Generator().manual_seed(13)
    aligned = sliced_projection_wasserstein_score(
        samples, target, n_projections=8, n_quantiles=8, generator=projection_generator
    )
    projection_generator = torch.Generator().manual_seed(13)
    shifted = sliced_projection_wasserstein_score(
        samples + 1.0,
        target,
        n_projections=8,
        n_quantiles=8,
        generator=projection_generator,
    )
    assert aligned < shifted
    aligned.backward()
    assert samples.grad is not None and torch.isfinite(samples.grad).all()
