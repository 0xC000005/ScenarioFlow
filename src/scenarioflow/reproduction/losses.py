from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..model import GenericStateAwareNormalizedInnovationFlowMatching


def horizon_path_weights(
    horizon: int,
    *,
    end_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic normalized horizon weights for final-path losses."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    end = max(1e-6, float(end_weight))
    if horizon == 1:
        weights = torch.ones(1, device=device, dtype=dtype)
    else:
        weights = torch.linspace(1.0, end, horizon, device=device, dtype=dtype)
    return weights / weights.mean().clamp_min(1e-12)


def _weighted_flatten(
    paths: torch.Tensor,
    horizon_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if paths.ndim not in {3, 4}:
        raise ValueError("paths must have shape [B,T,C] or [B,K,T,C]")
    if horizon_weights is None:
        weighted = paths
        weights = torch.ones(paths.shape[-2], device=paths.device, dtype=paths.dtype)
    else:
        if horizon_weights.shape != (paths.shape[-2],):
            raise ValueError(
                f"horizon_weights must have shape ({paths.shape[-2]},), "
                f"got {tuple(horizon_weights.shape)}"
            )
        shape = (1,) * (paths.ndim - 2) + (paths.shape[-2], 1)
        weights = horizon_weights.to(device=paths.device, dtype=paths.dtype)
        weighted = paths * weights.view(shape)
    return (
        weighted.reshape(*paths.shape[:-2], paths.shape[-2] * paths.shape[-1]),
        weights,
    )


def full_path_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score over the complete future path."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    samples_flat, weights = _weighted_flatten(samples, horizon_weights)
    target_flat, _ = _weighted_flatten(target, horizon_weights)
    scale = torch.sqrt(weights.square().sum() * float(samples.shape[-1])).clamp_min(
        1e-12
    )
    target_dist = (
        torch.sqrt(
            (samples_flat - target_flat[:, None, :]).pow(2).sum(dim=-1) + float(eps)
        ).mean(dim=1)
        / scale
    )
    pair_dist = torch.cdist(samples_flat, samples_flat, p=2).mean(dim=(1, 2)) / scale
    score = target_dist - 0.5 * pair_dist
    return score.mean(), target_dist.mean(), pair_dist.mean()


def sliced_projection_wasserstein_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    n_projections: int,
    n_quantiles: int,
    horizon_weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced Wasserstein discrepancy between generated and realized paths."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    if n_projections < 1:
        raise ValueError("n_projections must be positive")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    samples_flat, _ = _weighted_flatten(samples, horizon_weights)
    target_flat, _ = _weighted_flatten(target, horizon_weights)
    generated = samples_flat.reshape(samples_flat.shape[0] * samples_flat.shape[1], -1)
    realized = target_flat.reshape(target_flat.shape[0], -1)
    projections = torch.randn(
        int(n_projections),
        generated.shape[-1],
        device=generated.device,
        dtype=generated.dtype,
        generator=generator,
    )
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    generated_projection = generated @ projections.T
    realized_projection = realized @ projections.T
    quantiles = torch.linspace(
        0.0, 1.0, int(n_quantiles), device=generated.device, dtype=generated.dtype
    )
    generated_quantiles = torch.quantile(generated_projection, quantiles, dim=0)
    realized_quantiles = torch.quantile(realized_projection, quantiles, dim=0)
    return torch.sqrt((generated_quantiles - realized_quantiles).pow(2).mean() + 1e-12)


def effective_readout_iv_count(state_scope: str, *, n_cells: int, iv_count: int) -> int:
    """Map panel scope to the IV/anchor split used by group readout heads."""
    if state_scope == "iv_only":
        return int(n_cells)
    if state_scope == "anchor_only":
        return 0
    if state_scope == "joint38":
        return min(max(int(iv_count), 0), int(n_cells))
    raise ValueError(f"unknown state_scope {state_scope!r}")


def state_tail_sampling_weights(
    history_level_values: np.ndarray,
    *,
    tail_quantile: float,
    strength: float,
) -> np.ndarray:
    """Symmetric train-derived weights for rare current-level regions."""
    current = np.asarray(history_level_values, dtype=np.float64)[:, -1, :]
    n_windows, n_cells = current.shape
    if n_windows < 2 or float(strength) <= 0.0:
        return np.ones(n_windows, dtype=np.float32)
    q = min(max(float(tail_quantile), 0.0), 0.999)
    ranks = np.empty_like(current, dtype=np.float64)
    for cell in range(n_cells):
        values = current[:, cell]
        if float(np.max(values) - np.min(values)) <= 1e-12:
            ranks[:, cell] = 0.5
            continue
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        left = np.searchsorted(sorted_vals, values, side="left")
        right = np.searchsorted(sorted_vals, values, side="right")
        ranks[:, cell] = (left + right + 1.0) / (2.0 * float(n_windows))
    tail_depth = 2.0 * np.abs(ranks - 0.5)
    rarity = np.maximum((tail_depth - q) / max(1.0 - q, 1e-6), 0.0).mean(axis=1)
    weights = 1.0 + float(strength) * rarity
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def channelwise_path_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score averaged per channel so weak coordinates are not washed out."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    if horizon_weights is None:
        weights = torch.ones(
            samples.shape[-2], device=samples.device, dtype=samples.dtype
        )
    else:
        if horizon_weights.shape != (samples.shape[-2],):
            raise ValueError(
                f"horizon_weights must have shape ({samples.shape[-2]},), "
                f"got {tuple(horizon_weights.shape)}"
            )
        weights = horizon_weights.to(device=samples.device, dtype=samples.dtype)
    weighted_samples = samples * weights.view(1, 1, samples.shape[-2], 1)
    weighted_target = target * weights.view(1, target.shape[-2], 1)
    per_channel_samples = weighted_samples.permute(0, 3, 1, 2)
    per_channel_target = weighted_target.permute(0, 2, 1)
    scale = torch.sqrt(weights.square().sum()).clamp_min(1e-12)
    target_dist = (
        torch.sqrt(
            (per_channel_samples - per_channel_target[:, :, None, :]).pow(2).sum(dim=-1)
            + float(eps)
        ).mean(dim=2)
        / scale
    )
    bsz, n_cells, n_samples, horizon = per_channel_samples.shape
    flat_samples = per_channel_samples.reshape(bsz * n_cells, n_samples, horizon)
    pair_dist = (
        torch.cdist(flat_samples, flat_samples, p=2).mean(dim=(1, 2)).view(bsz, n_cells)
        / scale
    )
    score = target_dist - 0.5 * pair_dist
    return score.mean(), target_dist.mean(), pair_dist.mean()


def marginal_crps_path_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate-wise ensemble CRPS averaged across horizon and channels."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    target_dist_grid = (samples - target[:, None, :, :]).abs().mean(dim=1)
    pair_dist_grid = (
        (samples[:, :, None, :, :] - samples[:, None, :, :, :]).abs().mean(dim=(1, 2))
    )
    score_grid = target_dist_grid - 0.5 * pair_dist_grid
    if horizon_weights is not None:
        if horizon_weights.shape != (samples.shape[-2],):
            raise ValueError(
                f"horizon_weights must have shape ({samples.shape[-2]},), "
                f"got {tuple(horizon_weights.shape)}"
            )
        weights = horizon_weights.to(device=samples.device, dtype=samples.dtype)
        weights = weights / weights.mean().clamp_min(1e-12)
        score_grid = score_grid * weights.view(1, samples.shape[-2], 1)
        target_dist_grid = target_dist_grid * weights.view(1, samples.shape[-2], 1)
        pair_dist_grid = pair_dist_grid * weights.view(1, samples.shape[-2], 1)
    return score_grid.mean(), target_dist_grid.mean(), pair_dist_grid.mean()


def interval_score_path_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.1,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Central interval score for ensemble scenario paths.

    The score is proper for a central (1-alpha) prediction interval and
    penalizes both over-wide intervals and misses outside the interval.
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    lo = torch.quantile(samples, alpha_value / 2.0, dim=1)
    hi = torch.quantile(samples, 1.0 - alpha_value / 2.0, dim=1)
    width = (hi - lo).clamp_min(0.0)
    miss_low = (lo - target).clamp_min(0.0)
    miss_high = (target - hi).clamp_min(0.0)
    penalty = (2.0 / alpha_value) * (miss_low + miss_high)
    score_grid = width + penalty
    if horizon_weights is not None:
        if horizon_weights.shape != (samples.shape[-2],):
            raise ValueError(
                f"horizon_weights must have shape ({samples.shape[-2]}), "
                f"got {tuple(horizon_weights.shape)}"
            )
        weights = horizon_weights.to(device=samples.device, dtype=samples.dtype)
        weights = weights / weights.mean().clamp_min(1e-12)
        score_grid = score_grid * weights.view(1, samples.shape[-2], 1)
        width = width * weights.view(1, samples.shape[-2], 1)
        penalty = penalty * weights.view(1, samples.shape[-2], 1)
    return score_grid.mean(), width.mean(), penalty.mean()


def structured_variogram_path_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    power: float = 0.5,
) -> torch.Tensor:
    """Variogram score over adjacent time pairs and same-horizon channel pairs."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    bsz, n_samples, horizon, n_cells = samples.shape
    left_parts: list[torch.Tensor] = []
    right_parts: list[torch.Tensor] = []
    if horizon > 1:
        temporal_left = (
            torch.arange(horizon - 1, device=samples.device)[:, None] * n_cells
        )
        temporal_channels = torch.arange(n_cells, device=samples.device)[None, :]
        left_parts.append((temporal_left + temporal_channels).reshape(-1))
        right_parts.append((temporal_left + n_cells + temporal_channels).reshape(-1))
    if n_cells > 1:
        channel_pairs = torch.triu_indices(
            n_cells, n_cells, offset=1, device=samples.device
        )
        horizon_offsets = (
            torch.arange(horizon, device=samples.device)[:, None] * n_cells
        )
        left_parts.append((horizon_offsets + channel_pairs[0][None, :]).reshape(-1))
        right_parts.append((horizon_offsets + channel_pairs[1][None, :]).reshape(-1))
    if not left_parts:
        return torch.zeros((), device=samples.device, dtype=samples.dtype)
    left = torch.cat(left_parts)
    right = torch.cat(right_parts)
    sample_flat = samples.reshape(bsz, n_samples, horizon * n_cells)
    target_flat = target.reshape(bsz, horizon * n_cells)
    sample_diff = (
        (sample_flat[:, :, left] - sample_flat[:, :, right])
        .abs()
        .clamp_min(1e-12)
        .pow(float(power))
    )
    target_diff = (
        (target_flat[:, left] - target_flat[:, right])
        .abs()
        .clamp_min(1e-12)
        .pow(float(power))
    )
    return (sample_diff.mean(dim=1) - target_diff).square().mean()


def dispersion_calibration_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    mode: str = "window",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-wise calibration between ensemble spread and realized path activity.

    The rank component asks high-realized-activity histories to receive wider
    ensembles. The global component prevents the trivial low-spread solution.
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    if mode not in {"window", "channel", "window_channel"}:
        raise ValueError("mode must be 'window', 'channel', or 'window_channel'")

    def _loss_for_activity(
        spread_activity: torch.Tensor,
        target_activity: torch.Tensor,
        *,
        normalize_dim: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalize_dim is None:
            spread_z = (spread_activity - spread_activity.mean()) / spread_activity.std(
                unbiased=False
            ).clamp_min(eps)
            target_z = (target_activity - target_activity.mean()) / target_activity.std(
                unbiased=False
            ).clamp_min(eps)
            global_loss = (
                torch.log(spread_activity.mean().clamp_min(eps))
                - torch.log(target_activity.mean().detach().clamp_min(eps))
            ).square()
        else:
            spread_mean = spread_activity.mean(dim=normalize_dim, keepdim=True)
            target_mean = target_activity.mean(dim=normalize_dim, keepdim=True)
            spread_std = spread_activity.std(
                dim=normalize_dim, unbiased=False, keepdim=True
            ).clamp_min(eps)
            target_std = target_activity.std(
                dim=normalize_dim, unbiased=False, keepdim=True
            ).clamp_min(eps)
            spread_z = (spread_activity - spread_mean) / spread_std
            target_z = (target_activity - target_mean) / target_std
            global_loss = (
                (
                    torch.log(spread_mean.squeeze(normalize_dim).clamp_min(eps))
                    - torch.log(
                        target_mean.detach().squeeze(normalize_dim).clamp_min(eps)
                    )
                )
                .square()
                .mean()
            )
        rank_loss = (spread_z - target_z.detach()).square().mean()
        spread_flat = spread_activity.reshape(-1)
        target_flat = target_activity.reshape(-1)
        spread_centered = spread_flat - spread_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr_value = (spread_centered * target_centered).mean() / (
            spread_centered.std(unbiased=False).clamp_min(eps)
            * target_centered.std(unbiased=False).clamp_min(eps)
        )
        return rank_loss, global_loss, spread_activity.mean(), corr_value

    pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if mode in {"window", "window_channel"}:
        pieces.append(
            _loss_for_activity(
                samples.var(dim=1, unbiased=False).mean(dim=(1, 2)),
                target.square().mean(dim=(1, 2)),
                normalize_dim=None,
            )
        )
    if mode in {"channel", "window_channel"}:
        pieces.append(
            _loss_for_activity(
                samples.var(dim=1, unbiased=False).mean(dim=1),
                target.square().mean(dim=1),
                normalize_dim=0,
            )
        )

    rank_mse = torch.stack([piece[0] for piece in pieces]).mean()
    global_log_mse = torch.stack([piece[1] for piece in pieces]).mean()
    spread_activity_mean = torch.stack([piece[2] for piece in pieces]).mean()
    corr = torch.stack([piece[3] for piece in pieces]).mean()
    return (
        rank_mse + global_log_mse,
        rank_mse,
        global_log_mse,
        spread_activity_mean,
        corr.detach(),
    )


def standardized_level_delta_paths(
    sampled_level: torch.Tensor,
    target_level: torch.Tensor,
    history_level_values: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert level paths to unit-free deltas from the last conditioned level."""
    if sampled_level.ndim != 4 or target_level.ndim != 3:
        raise ValueError("Expected sampled_level [B,K,T,C] and target_level [B,T,C]")
    if history_level_values.ndim != 3 or scale.ndim != 2:
        raise ValueError("Expected history_level_values [B,H,C] and scale [B,C]")
    if (
        sampled_level.shape[0] != target_level.shape[0]
        or sampled_level.shape[2:] != target_level.shape[1:]
    ):
        raise ValueError("sampled_level and target_level path dimensions do not match")
    if (
        history_level_values.shape[0] != target_level.shape[0]
        or history_level_values.shape[-1] != target_level.shape[-1]
    ):
        raise ValueError("history_level_values dimensions do not match target_level")
    if scale.shape != (target_level.shape[0], target_level.shape[-1]):
        raise ValueError(
            f"scale must have shape {(target_level.shape[0], target_level.shape[-1])}"
        )
    base = history_level_values[:, -1, :]
    safe_scale = scale.clamp_min(1e-8)
    sampled_delta = (sampled_level - base[:, None, None, :]) / safe_scale[
        :, None, None, :
    ]
    target_delta = (target_level - base[:, None, :]) / safe_scale[:, None, :]
    return sampled_delta, target_delta


def condition_negative_permutation(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Select generic in-batch negative histories for conditional contrast."""
    bsz = int(history_level_values.shape[0])
    if bsz < 2:
        return torch.arange(bsz, device=history_level_values.device)
    if mode == "roll":
        return torch.roll(
            torch.arange(bsz, device=history_level_values.device), shifts=1
        )
    if mode == "nearest_history":
        with torch.no_grad():
            level_scores = model.level_values_to_scores(history_level_values).detach()
            flow_coordinate = model._to_flow_coordinate(
                history_normalized_innovation
            ).detach()
            features = torch.cat(
                [
                    level_scores.reshape(bsz, -1),
                    flow_coordinate.reshape(bsz, -1),
                ],
                dim=1,
            )
            features = features - features.mean(dim=0, keepdim=True)
            features = features / features.std(
                dim=0, unbiased=False, keepdim=True
            ).clamp_min(1e-6)
            distances = torch.cdist(features, features, p=2)
            distances.fill_diagonal_(float("inf"))
            return torch.argmin(distances, dim=1)
    raise ValueError(
        "condition_rollout_negative_mode must be 'roll' or 'nearest_history'"
    )


def differentiable_rollout_paths(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
    future_no_update_target: torch.Tensor | None = None,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable free-running sampler returning normalized innovations and level paths."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(
            f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}"
        )
    level_scores = model.level_values_to_scores(history_level_values)
    history_flow_coordinate = model._to_flow_coordinate(history_normalized_innovation)
    bsz = int(level_scores.shape[0])
    k = int(n_samples)
    prefix_level_values = (
        history_level_values.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_level_scores = (
        level_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_norm = (
        history_flow_coordinate.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    center_rep = (
        center.unsqueeze(1)
        .expand(bsz, k, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.n_cells)
    )
    scale_rep = (
        scale.unsqueeze(1)
        .expand(bsz, k, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.n_cells)
    )
    if drift_feature is None:
        drift_rep = None
    else:
        drift_rep = (
            drift_feature.unsqueeze(1)
            .expand(bsz, k, model.cfg.n_cells)
            .reshape(bsz * k, model.cfg.n_cells)
        )
    _risk_prediction, risk_context = model._risk_state_from_history(
        prefix_level_scores,
        prefix_norm,
        center_rep,
        scale_rep,
        drift_rep,
    )
    base_noise = float(temperature) * model._base_noise_like(
        torch.empty(
            bsz * k,
            int(n_steps),
            model.cfg.n_cells,
            device=history_level_values.device,
            dtype=history_level_values.dtype,
        )
    )
    dt = 1.0 / float(max(1, int(flow_steps)))
    norm_frames: list[torch.Tensor] = []
    level_frames: list[torch.Tensor] = []
    for _step in range(int(n_steps)):
        memory_state = model._encode_prefix(
            prefix_level_scores, prefix_norm, center_rep, scale_rep, drift_rep
        )[:, -1]
        if risk_context is not None:
            memory_state = memory_state + risk_context
        current_level_score = prefix_level_scores[:, -1]
        x = base_noise[:, _step]
        base_noise_scale = model._conditional_base_noise_scale(memory_state)
        if base_noise_scale is not None:
            x = x * base_noise_scale
        for flow_step in range(max(1, int(flow_steps))):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=history_level_values.device,
                dtype=history_level_values.dtype,
            )
            x = x + dt * model.velocity(x, current_level_score, memory_state, t)
        next_flow_coordinate = x
        next_norm = model._from_flow_coordinate(next_flow_coordinate)
        next_increment = next_norm * scale_rep + center_rep
        next_level_value = prefix_level_values[:, -1] + next_increment
        next_level_score = model.level_values_to_scores(next_level_value)
        norm_frames.append(next_norm.view(bsz, k, model.cfg.n_cells))
        level_frames.append(next_level_value.view(bsz, k, model.cfg.n_cells))
        prefix_level_values = torch.cat(
            [prefix_level_values, next_level_value[:, None, :]], dim=1
        )
        prefix_level_scores = torch.cat(
            [prefix_level_scores, next_level_score[:, None, :]], dim=1
        )
        prefix_norm = torch.cat([prefix_norm, next_flow_coordinate[:, None, :]], dim=1)
    return torch.stack(norm_frames, dim=2), torch.stack(level_frames, dim=2)


def differentiable_normalized_rollout_samples(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable free-running sampler returning normalized innovations."""
    sampled_norm, _sampled_level = differentiable_rollout_paths(
        model,
        history_level_values,
        history_normalized_innovation,
        center,
        scale,
        drift_feature,
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        flow_steps=int(flow_steps),
        temperature=float(temperature),
    )
    return sampled_norm


def prefix_conditioned_flow_matching_loss(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    prefix_future_level_values: torch.Tensor,
    prefix_future_normalized_innovation: torch.Tensor,
    future_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
) -> torch.Tensor:
    """FM loss under an off-policy future prefix, for scheduled-sampling-style robustness."""
    if prefix_future_level_values.shape != future_normalized_innovation.shape:
        raise ValueError(
            "prefix_future_level_values and future_normalized_innovation must match"
        )
    if prefix_future_normalized_innovation.shape != future_normalized_innovation.shape:
        raise ValueError(
            "prefix_future_normalized_innovation and future_normalized_innovation must match"
        )
    history_level_scores = model.level_values_to_scores(history_level_values)
    prefix_future_level_scores = model.level_values_to_scores(
        prefix_future_level_values
    )
    history_flow_coordinate = model._to_flow_coordinate(history_normalized_innovation)
    prefix_future_flow_coordinate = model._to_flow_coordinate(
        prefix_future_normalized_innovation
    )
    target_flow_coordinate = model._to_flow_coordinate(future_normalized_innovation)
    prefix_level_scores = torch.cat(
        [history_level_scores, prefix_future_level_scores[:, :-1]], dim=1
    )
    prefix_flow_coordinate = torch.cat(
        [history_flow_coordinate, prefix_future_flow_coordinate[:, :-1]], dim=1
    )
    hidden = model._encode_prefix(
        prefix_level_scores, prefix_flow_coordinate, center, scale, drift_feature
    )
    start = model.cfg.history_len - 1
    memory_states = hidden[:, start : start + model.cfg.future_len]
    _risk_prediction, risk_context = model._risk_state_from_history(
        history_level_scores,
        history_flow_coordinate,
        center,
        scale,
        drift_feature,
    )
    if risk_context is not None:
        memory_states = memory_states + risk_context[:, None, :]
    current_level_scores = prefix_level_scores[:, start : start + model.cfg.future_len]
    x1 = target_flow_coordinate
    x0 = model._base_noise_like(x1)
    base_noise_scale = model._conditional_base_noise_scale(memory_states)
    if base_noise_scale is not None:
        x0 = x0 * base_noise_scale
    bsz, horizon, n_vars = x1.shape
    t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
    x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
    target_velocity = x1 - x0
    pred_velocity = model.velocity(
        x_t.reshape(bsz * horizon, n_vars),
        current_level_scores.reshape(bsz * horizon, n_vars),
        memory_states.reshape(bsz * horizon, model.cfg.memory_dim),
        t.reshape(bsz * horizon),
    ).view_as(x1)
    return F.mse_loss(pred_velocity, target_velocity)


def normalized_rollout_energy_loss(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    future_level_values: torch.Tensor,
    future_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
    future_no_update_target: torch.Tensor | None = None,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
    marginal_crps_weight: float = 0.0,
    interval_score_weight: float = 0.0,
    interval_alpha: float = 0.1,
    variogram_weight: float = 0.0,
    variogram_power: float = 0.5,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    energy_eps: float,
    temperature: float,
    level_marginal_crps_weight: float = 0.0,
    level_marginal_crps_coordinate: str = "scaled_delta",
    level_energy_weight: float = 0.0,
    channel_level_energy_weight: float = 0.0,
    channel_level_energy_coordinate: str = "level",
    condition_rollout_contrast_weight: float = 0.0,
    condition_rollout_contrast_margin: float = 0.0,
    condition_rollout_negative_mode: str = "roll",
    risk_state_weight: float = 0.0,
    risk_state_rank_weight: float = 0.0,
    mixed_support_weight: float = 0.0,
    vov_supervision_weight: float = 0.0,
    dispersion_calibration_weight: float = 0.0,
    dispersion_calibration_mode: str = "window",
    free_running_fm_weight: float = 0.0,
    free_running_fm_prefix_steps: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(
        history_level_values,
        history_normalized_innovation,
        future_level_values,
        future_normalized_innovation,
        center,
        scale,
        drift_feature=drift_feature,
        future_no_update_target=future_no_update_target,
        risk_state_weight=float(risk_state_weight),
        risk_state_rank_weight=float(risk_state_rank_weight),
        mixed_support_weight=float(mixed_support_weight),
        vov_supervision_weight=float(vov_supervision_weight),
    )
    sampled_norm, sampled_level = differentiable_rollout_paths(
        model,
        history_level_values,
        history_normalized_innovation,
        center,
        scale,
        drift_feature,
        n_samples=int(train_sample_count),
        n_steps=int(future_normalized_innovation.shape[1]),
        flow_steps=int(rollout_flow_steps),
        temperature=float(temperature),
    )
    if float(free_running_fm_weight) > 0.0:
        prefix_level = sampled_level[:, 0].detach()
        prefix_norm = sampled_norm[:, 0].detach()
        prefix_steps = int(free_running_fm_prefix_steps)
        if prefix_steps > 0:
            prefix_steps = min(prefix_steps, int(future_normalized_innovation.shape[1]))
            if prefix_steps < int(future_normalized_innovation.shape[1]):
                prefix_level = torch.cat(
                    [
                        prefix_level[:, :prefix_steps],
                        future_level_values[:, prefix_steps:].detach(),
                    ],
                    dim=1,
                )
                prefix_norm = torch.cat(
                    [
                        prefix_norm[:, :prefix_steps],
                        future_normalized_innovation[:, prefix_steps:].detach(),
                    ],
                    dim=1,
                )
        free_running_fm_loss = prefix_conditioned_flow_matching_loss(
            model,
            history_level_values,
            history_normalized_innovation,
            prefix_level,
            prefix_norm,
            future_normalized_innovation,
            center,
            scale,
            drift_feature=drift_feature,
        )
    else:
        free_running_fm_loss = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    weights = horizon_path_weights(
        int(future_normalized_innovation.shape[1]),
        end_weight=float(horizon_end_weight),
        device=future_normalized_innovation.device,
        dtype=future_normalized_innovation.dtype,
    )
    energy, target_dist, pair_dist = full_path_energy_score(
        sampled_norm,
        future_normalized_innovation,
        eps=float(energy_eps),
        horizon_weights=weights,
    )
    marginal_crps, marginal_crps_target_dist, marginal_crps_pair_dist = (
        marginal_crps_path_score(
            sampled_norm,
            future_normalized_innovation,
            horizon_weights=weights,
        )
    )
    if float(interval_score_weight) > 0.0:
        interval_score, interval_width, interval_miss_penalty = (
            interval_score_path_score(
                sampled_norm,
                future_normalized_innovation,
                alpha=float(interval_alpha),
                horizon_weights=weights,
            )
        )
    else:
        interval_score = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        interval_width = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        interval_miss_penalty = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    variogram = (
        structured_variogram_path_score(
            sampled_norm, future_normalized_innovation, power=float(variogram_power)
        )
        if float(variogram_weight) > 0.0
        else torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
    )
    if float(dispersion_calibration_weight) > 0.0:
        (
            dispersion_calibration,
            dispersion_rank_mse,
            dispersion_global_log_mse,
            spread_activity_mean,
            spread_future_activity_corr,
        ) = dispersion_calibration_loss(
            sampled_norm,
            future_normalized_innovation,
            eps=float(energy_eps),
            mode=dispersion_calibration_mode,
        )
    else:
        dispersion_calibration = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        dispersion_rank_mse = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        dispersion_global_log_mse = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        spread_activity_mean = sampled_norm.var(dim=1, unbiased=False).mean()
        spread_future_activity_corr = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    if float(level_energy_weight) > 0.0:
        level_energy, level_target_dist, level_pair_dist = full_path_energy_score(
            sampled_level,
            future_level_values,
            eps=float(energy_eps),
            horizon_weights=weights,
        )
    else:
        level_energy = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        level_target_dist = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        level_pair_dist = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
    if float(level_marginal_crps_weight) > 0.0:
        if level_marginal_crps_coordinate == "level":
            level_crps_samples = sampled_level
            level_crps_target = future_level_values
        elif level_marginal_crps_coordinate == "scaled_delta":
            level_crps_samples, level_crps_target = standardized_level_delta_paths(
                sampled_level,
                future_level_values,
                history_level_values,
                scale,
            )
        else:
            raise ValueError(
                "level_marginal_crps_coordinate must be 'level' or 'scaled_delta'"
            )
        (
            level_marginal_crps,
            level_marginal_crps_target_dist,
            level_marginal_crps_pair_dist,
        ) = marginal_crps_path_score(
            level_crps_samples,
            level_crps_target,
            horizon_weights=weights,
        )
    else:
        level_marginal_crps = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        level_marginal_crps_target_dist = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        level_marginal_crps_pair_dist = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    if float(channel_level_energy_weight) > 0.0:
        if channel_level_energy_coordinate == "level":
            channel_samples = sampled_level
            channel_target = future_level_values
        elif channel_level_energy_coordinate == "scaled_delta":
            channel_samples, channel_target = standardized_level_delta_paths(
                sampled_level,
                future_level_values,
                history_level_values,
                scale,
            )
        else:
            raise ValueError(
                "channel_level_energy_coordinate must be 'level' or 'scaled_delta'"
            )
        channel_level_energy, channel_level_target_dist, channel_level_pair_dist = (
            channelwise_path_energy_score(
                channel_samples,
                channel_target,
                eps=float(energy_eps),
                horizon_weights=weights,
            )
        )
    else:
        channel_level_energy = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        channel_level_target_dist = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        channel_level_pair_dist = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    if (
        float(condition_rollout_contrast_weight) > 0.0
        and int(history_level_values.shape[0]) > 1
    ):
        perm = condition_negative_permutation(
            model,
            history_level_values,
            history_normalized_innovation,
            mode=condition_rollout_negative_mode,
        )
        neg_drift = None if drift_feature is None else drift_feature[perm]
        neg_sampled_norm, _neg_sampled_level = differentiable_rollout_paths(
            model,
            history_level_values[perm],
            history_normalized_innovation[perm],
            center[perm],
            scale[perm],
            neg_drift,
            n_samples=int(train_sample_count),
            n_steps=int(future_normalized_innovation.shape[1]),
            flow_steps=int(rollout_flow_steps),
            temperature=float(temperature),
        )
        neg_energy, _neg_target_dist, _neg_pair_dist = full_path_energy_score(
            neg_sampled_norm,
            future_normalized_innovation,
            eps=float(energy_eps),
            horizon_weights=weights,
        )
        condition_rollout_contrast = F.softplus(
            energy - neg_energy + float(condition_rollout_contrast_margin)
        )
        condition_rollout_neg_energy = neg_energy
    else:
        condition_rollout_contrast = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
        condition_rollout_neg_energy = torch.zeros(
            (), device=fm_loss.device, dtype=fm_loss.dtype
        )
    total = (
        float(fm_anchor_weight) * fm_loss
        + float(energy_weight) * energy
        + float(marginal_crps_weight) * marginal_crps
        + float(interval_score_weight) * interval_score
        + float(variogram_weight) * variogram
        + float(dispersion_calibration_weight) * dispersion_calibration
        + float(level_marginal_crps_weight) * level_marginal_crps
        + float(level_energy_weight) * level_energy
        + float(channel_level_energy_weight) * channel_level_energy
        + float(condition_rollout_contrast_weight) * condition_rollout_contrast
        + float(free_running_fm_weight) * free_running_fm_loss
    )
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "marginal_crps": marginal_crps.detach(),
        "marginal_crps_target_dist": marginal_crps_target_dist.detach(),
        "marginal_crps_pair_dist": marginal_crps_pair_dist.detach(),
        "interval_score": interval_score.detach(),
        "interval_width": interval_width.detach(),
        "interval_miss_penalty": interval_miss_penalty.detach(),
        "variogram": variogram.detach(),
        "dispersion_calibration": dispersion_calibration.detach(),
        "dispersion_rank_mse": dispersion_rank_mse.detach(),
        "dispersion_global_log_mse": dispersion_global_log_mse.detach(),
        "dispersion_spread_activity_mean": spread_activity_mean.detach(),
        "dispersion_target_activity_mean": future_normalized_innovation.square()
        .mean()
        .detach(),
        "dispersion_spread_target_ratio": (
            spread_activity_mean
            / future_normalized_innovation.square().mean().detach().clamp_min(1e-8)
        ).detach(),
        "dispersion_spread_future_activity_corr": spread_future_activity_corr.detach(),
        "level_marginal_crps": level_marginal_crps.detach(),
        "level_marginal_crps_target_dist": level_marginal_crps_target_dist.detach(),
        "level_marginal_crps_pair_dist": level_marginal_crps_pair_dist.detach(),
        "level_energy": level_energy.detach(),
        "level_energy_target_dist": level_target_dist.detach(),
        "level_energy_pair_dist": level_pair_dist.detach(),
        "channel_level_energy": channel_level_energy.detach(),
        "channel_level_energy_target_dist": channel_level_target_dist.detach(),
        "channel_level_energy_pair_dist": channel_level_pair_dist.detach(),
        "condition_rollout_contrast": condition_rollout_contrast.detach(),
        "condition_rollout_pos_energy": energy.detach(),
        "condition_rollout_neg_energy": condition_rollout_neg_energy.detach(),
        "free_running_fm_loss": free_running_fm_loss.detach(),
        "free_running_fm_weight": torch.as_tensor(
            float(free_running_fm_weight),
            device=fm_loss.device,
            dtype=fm_loss.dtype,
        ),
        "free_running_fm_prefix_steps": torch.as_tensor(
            int(free_running_fm_prefix_steps),
            device=fm_loss.device,
            dtype=fm_loss.dtype,
        ),
        "base_noise_scale_enabled": fm_metrics["base_noise_scale_enabled"].detach(),
        "base_noise_scale_mean": fm_metrics["base_noise_scale_mean"].detach(),
        "base_noise_scale_std": fm_metrics["base_noise_scale_std"].detach(),
        "base_noise_scale_min": fm_metrics["base_noise_scale_min"].detach(),
        "base_noise_scale_max": fm_metrics["base_noise_scale_max"].detach(),
        "base_noise_scale_hist_corr": fm_metrics["base_noise_scale_hist_corr"].detach(),
        "base_noise_scale_realized_corr": fm_metrics[
            "base_noise_scale_realized_corr"
        ].detach(),
        "vov_supervision_loss": fm_metrics["vov_supervision_loss"].detach(),
        "vov_supervision_weight": fm_metrics["vov_supervision_weight"].detach(),
        "risk_state_enabled": fm_metrics["risk_state_enabled"].detach(),
        "risk_state_loss": fm_metrics["risk_state_loss"].detach(),
        "risk_state_rank_loss": fm_metrics["risk_state_rank_loss"].detach(),
        "risk_state_rank_rho": fm_metrics["risk_state_rank_rho"].detach(),
        "mixed_support_enabled": fm_metrics["mixed_support_enabled"].detach(),
        "mixed_support_bce_loss": fm_metrics["mixed_support_bce_loss"].detach(),
        "mixed_support_selected_count": fm_metrics[
            "mixed_support_selected_count"
        ].detach(),
        "mixed_support_target_rate": fm_metrics["mixed_support_target_rate"].detach(),
        "target_norm_std": future_normalized_innovation.std(unbiased=False).detach(),
        "sample_norm_std": sampled_norm.std(unbiased=False).detach(),
        "target_level_std": future_level_values.std(unbiased=False).detach(),
        "sample_level_std": sampled_level.std(unbiased=False).detach(),
        "sample_h1_std": sampled_norm[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_norm[:, :, -1].std(unbiased=False).detach(),
        "memory_abs": fm_metrics["memory_abs"].detach(),
    }
    return total, metrics


joint_energy_loss = full_path_energy_score
channel_energy_loss = channelwise_path_energy_score
