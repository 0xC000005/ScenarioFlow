#!/usr/bin/env python
"""666a: sampled-rollout energy fine-tune for normalized-innovation AR flows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_adaptive_graph_velocity_mixer,
    enable_conditional_base_noise_scale,
    enable_group_head_velocity_readout,
    enable_group_residual_velocity_readout,
    enable_prefix_feature_mode,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.train_596a_final_path_joint_objective import (  # noqa: E402
    full_path_energy_score,
    horizon_path_weights,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    no_update_target_from_block,
    sample_smoke,
    select_normalized_innovation_scope,
)


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


def run_epoch(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
    marginal_crps_weight: float,
    interval_score_weight: float,
    interval_alpha: float,
    variogram_weight: float,
    variogram_power: float,
    level_marginal_crps_weight: float,
    level_marginal_crps_coordinate: str,
    level_energy_weight: float,
    channel_level_energy_weight: float,
    channel_level_energy_coordinate: str,
    condition_rollout_contrast_weight: float,
    condition_rollout_contrast_margin: float,
    condition_rollout_negative_mode: str,
    risk_state_weight: float,
    risk_state_rank_weight: float,
    mixed_support_weight: float,
    dispersion_calibration_weight: float,
    dispersion_calibration_mode: str,
    free_running_fm_weight: float,
    free_running_fm_prefix_steps: int,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    energy_eps: float,
    temperature: float,
    clip_grad: float,
    max_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for (
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift_feature,
        no_update_target,
    ) in loader:
        if int(max_batches) > 0 and n_batches >= int(max_batches):
            break
        with torch.set_grad_enabled(train_mode):
            loss, metrics = normalized_rollout_energy_loss(
                model,
                history_level.to(device),
                history_norm.to(device),
                future_level.to(device),
                future_norm.to(device),
                center.to(device),
                scale.to(device),
                drift_feature=drift_feature.to(device),
                future_no_update_target=no_update_target.to(device),
                train_sample_count=int(train_sample_count),
                rollout_flow_steps=int(rollout_flow_steps),
                energy_weight=float(energy_weight),
                marginal_crps_weight=float(marginal_crps_weight),
                interval_score_weight=float(interval_score_weight),
                interval_alpha=float(interval_alpha),
                variogram_weight=float(variogram_weight),
                variogram_power=float(variogram_power),
                level_marginal_crps_weight=float(level_marginal_crps_weight),
                level_marginal_crps_coordinate=level_marginal_crps_coordinate,
                level_energy_weight=float(level_energy_weight),
                channel_level_energy_weight=float(channel_level_energy_weight),
                channel_level_energy_coordinate=channel_level_energy_coordinate,
                condition_rollout_contrast_weight=float(
                    condition_rollout_contrast_weight
                ),
                condition_rollout_contrast_margin=float(
                    condition_rollout_contrast_margin
                ),
                condition_rollout_negative_mode=condition_rollout_negative_mode,
                risk_state_weight=float(risk_state_weight),
                risk_state_rank_weight=float(risk_state_rank_weight),
                mixed_support_weight=float(mixed_support_weight),
                dispersion_calibration_weight=float(dispersion_calibration_weight),
                dispersion_calibration_mode=dispersion_calibration_mode,
                free_running_fm_weight=float(free_running_fm_weight),
                free_running_fm_prefix_steps=int(free_running_fm_prefix_steps),
                fm_anchor_weight=float(fm_anchor_weight),
                horizon_end_weight=float(horizon_end_weight),
                energy_eps=float(energy_eps),
                temperature=float(temperature),
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(clip_grad) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                optimizer.step()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--state_scope",
        choices=["iv_only", "anchor_only", "joint38"],
        default="joint38",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform", choices=["log_level", "bounded_logit"], default="log_level"
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument(
        "--drift_feature_mode", choices=["none", "ewma_mean"], default="none"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_weight", type=float, default=0.2)
    parser.add_argument("--marginal_crps_weight", type=float, default=0.0)
    parser.add_argument("--interval_score_weight", type=float, default=0.0)
    parser.add_argument("--interval_alpha", type=float, default=0.1)
    parser.add_argument("--variogram_weight", type=float, default=0.0)
    parser.add_argument("--variogram_power", type=float, default=0.5)
    parser.add_argument("--level_marginal_crps_weight", type=float, default=0.0)
    parser.add_argument(
        "--level_marginal_crps_coordinate",
        choices=["level", "scaled_delta"],
        default="scaled_delta",
    )
    parser.add_argument("--level_energy_weight", type=float, default=0.0)
    parser.add_argument("--channel_level_energy_weight", type=float, default=0.0)
    parser.add_argument(
        "--channel_level_energy_coordinate",
        choices=["level", "scaled_delta"],
        default="level",
    )
    parser.add_argument("--condition_rollout_contrast_weight", type=float, default=0.0)
    parser.add_argument("--condition_rollout_contrast_margin", type=float, default=0.0)
    parser.add_argument(
        "--condition_rollout_negative_mode",
        choices=["roll", "nearest_history"],
        default="roll",
    )
    parser.add_argument("--risk_state_weight", type=float, default=0.0)
    parser.add_argument("--risk_state_rank_weight", type=float, default=0.0)
    parser.add_argument("--risk_state_dim", type=int, default=0)  # >0 instantiates the inert risk_context slot (probe)
    parser.add_argument("--mixed_support_loss_weight", type=float, default=None)
    parser.add_argument("--dispersion_calibration_weight", type=float, default=0.0)
    parser.add_argument(
        "--dispersion_calibration_mode",
        choices=["window", "channel", "window_channel"],
        default="window",
    )
    parser.add_argument("--free_running_fm_weight", type=float, default=0.0)
    parser.add_argument(
        "--free_running_fm_prefix_steps",
        type=int,
        default=0,
        help=(
            "If >0, use generated prefixes only for the first K future steps in the "
            "free-running FM auxiliary and use teacher-forced prefixes afterward. "
            "The default 0 preserves full generated-prefix behavior."
        ),
    )
    parser.add_argument("--state_tail_sampler_weight", type=float, default=0.0)
    parser.add_argument("--state_tail_sampler_quantile", type=float, default=0.8)
    parser.add_argument(
        "--velocity_readout_mode",
        choices=["shared", "group_residual", "group_head"],
        default="shared",
    )
    parser.add_argument(
        "--velocity_mixer_mode",
        choices=["checkpoint", "transformer", "adaptive_graph_residual"],
        default="checkpoint",
    )
    parser.add_argument("--adaptive_graph_k", type=int, default=8)
    parser.add_argument(
        "--prefix_feature_mode",
        choices=["checkpoint", "basic", "scale", "scale_local", "scale_drift"],
        default="checkpoint",
    )
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=1.2)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--base_noise_rho", type=float, default=None)
    parser.add_argument("--conditional_base_noise_scale", action="store_true")
    parser.add_argument("--base_noise_scale_min", type=float, default=0.5)
    parser.add_argument("--base_noise_scale_max", type=float, default=2.0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(
        args.checkpoint,
        device,
        cfg_overrides=(
            {"risk_state_dim": int(args.risk_state_dim)}
            if int(args.risk_state_dim) > 0
            else None
        ),
    )
    if args.prefix_feature_mode != "checkpoint":
        enable_prefix_feature_mode(model, prefix_feature_mode=args.prefix_feature_mode)
    if args.velocity_mixer_mode != "checkpoint":
        if args.velocity_mixer_mode == "adaptive_graph_residual":
            enable_adaptive_graph_velocity_mixer(
                model, k_neighbors=int(args.adaptive_graph_k)
            )
        elif args.velocity_mixer_mode == "transformer":
            if (
                getattr(model.cfg, "velocity_mixer_mode", "transformer")
                != "transformer"
            ):
                raise ValueError(
                    "cannot downgrade an adaptive-graph checkpoint to transformer"
                )
            model.cfg.velocity_mixer_mode = "transformer"
        else:
            raise ValueError(
                f"unsupported velocity_mixer_mode {args.velocity_mixer_mode!r}"
            )
    if args.base_noise_rho is not None:
        model.cfg.base_noise_rho = float(args.base_noise_rho)
    if bool(args.conditional_base_noise_scale):
        enable_conditional_base_noise_scale(
            model,
            scale_min=float(args.base_noise_scale_min),
            scale_max=float(args.base_noise_scale_max),
        )
    if args.velocity_readout_mode in {"group_residual", "group_head"}:
        readout_iv_count = effective_readout_iv_count(
            payload.get("state_scope", args.state_scope),
            n_cells=int(model.cfg.n_cells),
            iv_count=int(args.iv_count),
        )
        if args.velocity_readout_mode == "group_residual":
            enable_group_residual_velocity_readout(model, iv_count=readout_iv_count)
        else:
            enable_group_head_velocity_readout(model, iv_count=readout_iv_count)
    args.history_len = int(model.cfg.history_len)
    args.future_len = int(model.cfg.future_len)
    args.state_scope = payload.get("state_scope", args.state_scope)
    norm_cfg = payload.get("normalization", {})
    args.iv_transform = norm_cfg.get(
        "iv_transform", payload.get("iv_transform", args.iv_transform)
    )
    args.iv_lower_bound = float(
        norm_cfg.get(
            "iv_lower_bound", payload.get("iv_lower_bound", args.iv_lower_bound)
        )
    )
    args.iv_upper_bound = float(
        norm_cfg.get(
            "iv_upper_bound", payload.get("iv_upper_bound", args.iv_upper_bound)
        )
    )
    args.scale_floor = float(norm_cfg.get("scale_floor", args.scale_floor))
    args.center_mode = norm_cfg.get("center_mode", args.center_mode)
    args.drift_feature_mode = norm_cfg.get(
        "drift_feature_mode", args.drift_feature_mode
    )
    if args.mixed_support_loss_weight is None:
        args.mixed_support_loss_weight = float(
            norm_cfg.get(
                "mixed_support_loss_weight",
                payload.get("training_objective", {}).get(
                    "mixed_support_loss_weight", 0.0
                ),
            )
        )
    half_life = norm_cfg.get("scale_half_life", args.scale_half_life)
    scale_half_life = (
        None if half_life is None or float(half_life) <= 0.0 else float(half_life)
    )

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    (
        train_level,
        train_norm,
        train_future_level,
        train_future_norm,
        train_center,
        train_scale,
        train_drift,
        _train_raw,
        train_specs,
    ) = select_normalized_innovation_scope(
        train_block,
        args.state_scope,
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    (
        val_level,
        val_norm,
        val_future_level,
        val_future_norm,
        val_center,
        val_scale,
        val_drift,
        val_raw,
        val_specs,
    ) = select_normalized_innovation_scope(
        val_block,
        args.state_scope,
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    if expected and expected != [spec.name for spec in train_specs]:
        raise RuntimeError("checkpoint state specs do not match rebuilt specs")
    mixed_support_enabled = (
        getattr(model.cfg, "mixed_support_observation", "none") == "bernoulli_no_update"
    )
    zero_eps = float(norm_cfg.get("mixed_support_zero_eps", 1e-10))
    if mixed_support_enabled:
        train_no_update_target = no_update_target_from_block(
            train_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=zero_eps,
        )
        val_no_update_target = no_update_target_from_block(
            val_block,
            args.state_scope,
            int(args.iv_count),
            zero_eps=zero_eps,
        )
    else:
        train_no_update_target = np.zeros_like(train_future_norm, dtype=np.float32)
        val_no_update_target = np.zeros_like(val_future_norm, dtype=np.float32)

    train_sampler = None
    train_shuffle = True
    train_state_tail_weights = np.ones(int(train_level.shape[0]), dtype=np.float32)
    if float(args.state_tail_sampler_weight) > 0.0:
        train_state_tail_weights = state_tail_sampling_weights(
            train_level,
            tail_quantile=float(args.state_tail_sampler_quantile),
            strength=float(args.state_tail_sampler_weight),
        )
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(train_state_tail_weights, dtype=torch.double),
            num_samples=int(train_state_tail_weights.shape[0]),
            replacement=True,
        )
        train_shuffle = False
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_level),
            torch.from_numpy(train_norm),
            torch.from_numpy(train_future_level),
            torch.from_numpy(train_future_norm),
            torch.from_numpy(train_center),
            torch.from_numpy(train_scale),
            torch.from_numpy(train_drift),
            torch.from_numpy(train_no_update_target),
        ),
        batch_size=int(args.batch_size),
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_level),
            torch.from_numpy(val_norm),
            torch.from_numpy(val_future_level),
            torch.from_numpy(val_future_norm),
            torch.from_numpy(val_center),
            torch.from_numpy(val_scale),
            torch.from_numpy(val_drift),
            torch.from_numpy(val_no_update_target),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, Any]] = []
    best_path = output_dir / "best_model.pt"
    objective = {
        "base": "flow_matching_mse",
        "rollout_energy_coordinate": "normalized_innovation",
        "train_sample_count": int(args.train_sample_count),
        "rollout_flow_steps": int(args.rollout_flow_steps),
        "energy_weight": float(args.energy_weight),
        "marginal_crps_weight": float(args.marginal_crps_weight),
        "interval_score_weight": float(args.interval_score_weight),
        "interval_alpha": float(args.interval_alpha),
        "variogram_weight": float(args.variogram_weight),
        "variogram_power": float(args.variogram_power),
        "level_marginal_crps_weight": float(args.level_marginal_crps_weight),
        "level_marginal_crps_coordinate": args.level_marginal_crps_coordinate,
        "level_energy_weight": float(args.level_energy_weight),
        "channel_level_energy_weight": float(args.channel_level_energy_weight),
        "channel_level_energy_coordinate": args.channel_level_energy_coordinate,
        "condition_rollout_contrast_weight": float(
            args.condition_rollout_contrast_weight
        ),
        "condition_rollout_contrast_margin": float(
            args.condition_rollout_contrast_margin
        ),
        "condition_rollout_negative_mode": args.condition_rollout_negative_mode,
        "risk_state_weight": float(args.risk_state_weight),
        "risk_state_rank_weight": float(args.risk_state_rank_weight),
        "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        "mixed_support_observation": getattr(
            model.cfg, "mixed_support_observation", "none"
        ),
        "dispersion_calibration_weight": float(args.dispersion_calibration_weight),
        "dispersion_calibration_mode": args.dispersion_calibration_mode,
        "free_running_fm_weight": float(args.free_running_fm_weight),
        "free_running_fm_prefix_steps": int(args.free_running_fm_prefix_steps),
        "state_tail_sampler_weight": float(args.state_tail_sampler_weight),
        "state_tail_sampler_quantile": float(args.state_tail_sampler_quantile),
        "state_tail_sampler_mean": float(train_state_tail_weights.mean()),
        "state_tail_sampler_max": float(train_state_tail_weights.max()),
        "base_noise_rho": float(model.cfg.base_noise_rho),
        "conditional_base_noise_scale": bool(model.cfg.conditional_base_noise_scale),
        "base_noise_scale_min": float(model.cfg.base_noise_scale_min),
        "base_noise_scale_max": float(model.cfg.base_noise_scale_max),
        "velocity_readout_mode": args.velocity_readout_mode,
        "velocity_mixer_mode": getattr(model.cfg, "velocity_mixer_mode", "transformer"),
        "adaptive_graph_k": int(
            getattr(model.cfg, "adaptive_graph_k", args.adaptive_graph_k)
        ),
        "fm_anchor_weight": float(args.fm_anchor_weight),
        "horizon_end_weight": float(args.horizon_end_weight),
    }
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": payload.get(
            "model_coordinate", "state_aware_normalized_innovation"
        ),
        "normalization": norm_cfg,
        "finetune_objective": objective,
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "iv_count": int(args.iv_count),
        "state_specs": payload.get("state_specs", []),
        "panel_metadata": panel_metadata,
        "source_checkpoint": args.checkpoint,
    }
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            energy_weight=float(args.energy_weight),
            marginal_crps_weight=float(args.marginal_crps_weight),
            interval_score_weight=float(args.interval_score_weight),
            interval_alpha=float(args.interval_alpha),
            variogram_weight=float(args.variogram_weight),
            variogram_power=float(args.variogram_power),
            level_marginal_crps_weight=float(args.level_marginal_crps_weight),
            level_marginal_crps_coordinate=args.level_marginal_crps_coordinate,
            level_energy_weight=float(args.level_energy_weight),
            channel_level_energy_weight=float(args.channel_level_energy_weight),
            channel_level_energy_coordinate=args.channel_level_energy_coordinate,
            condition_rollout_contrast_weight=float(
                args.condition_rollout_contrast_weight
            ),
            condition_rollout_contrast_margin=float(
                args.condition_rollout_contrast_margin
            ),
            condition_rollout_negative_mode=args.condition_rollout_negative_mode,
            risk_state_weight=float(args.risk_state_weight),
            risk_state_rank_weight=float(args.risk_state_rank_weight),
            mixed_support_weight=float(args.mixed_support_loss_weight),
            dispersion_calibration_weight=float(args.dispersion_calibration_weight),
            dispersion_calibration_mode=args.dispersion_calibration_mode,
            free_running_fm_weight=float(args.free_running_fm_weight),
            free_running_fm_prefix_steps=int(args.free_running_fm_prefix_steps),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            energy_eps=float(args.energy_eps),
            temperature=float(args.temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_train_batches),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device=device,
                train_sample_count=int(args.train_sample_count),
                rollout_flow_steps=int(args.rollout_flow_steps),
                energy_weight=float(args.energy_weight),
                marginal_crps_weight=float(args.marginal_crps_weight),
                interval_score_weight=float(args.interval_score_weight),
                interval_alpha=float(args.interval_alpha),
                variogram_weight=float(args.variogram_weight),
                variogram_power=float(args.variogram_power),
                level_marginal_crps_weight=float(args.level_marginal_crps_weight),
                level_marginal_crps_coordinate=args.level_marginal_crps_coordinate,
                level_energy_weight=float(args.level_energy_weight),
                channel_level_energy_weight=float(args.channel_level_energy_weight),
                channel_level_energy_coordinate=args.channel_level_energy_coordinate,
                condition_rollout_contrast_weight=float(
                    args.condition_rollout_contrast_weight
                ),
                condition_rollout_contrast_margin=float(
                    args.condition_rollout_contrast_margin
                ),
                condition_rollout_negative_mode=args.condition_rollout_negative_mode,
                risk_state_weight=float(args.risk_state_weight),
                risk_state_rank_weight=float(args.risk_state_rank_weight),
                mixed_support_weight=float(args.mixed_support_loss_weight),
                dispersion_calibration_weight=float(args.dispersion_calibration_weight),
                dispersion_calibration_mode=args.dispersion_calibration_mode,
                free_running_fm_weight=float(args.free_running_fm_weight),
                free_running_fm_prefix_steps=int(args.free_running_fm_prefix_steps),
                fm_anchor_weight=float(args.fm_anchor_weight),
                horizon_end_weight=float(args.horizon_end_weight),
                energy_eps=float(args.energy_eps),
                temperature=float(args.temperature),
                clip_grad=0.0,
                max_batches=int(args.max_val_batches),
            )
        record = {
            "epoch": int(epoch),
            "elapsed_s": float(time.time() - t0),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        records.append(record)
        val_total = float(val_metrics.get("total", float("inf")))
        if val_total < best_val:
            best_val = val_total
            best_epoch = int(epoch)
            save_checkpoint(
                str(best_path), model, model.cfg, epoch, best_val, extra=extra
            )
        print(
            f"epoch {epoch:03d} train_total={train_metrics['total']:.6f} "
            f"val_total={val_total:.6f}{' best' if best_epoch == epoch else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(
        str(final_path), model, model.cfg, int(args.epochs), best_val, extra=extra
    )
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_norm[: int(args.sample_windows)],
        val_center[: int(args.sample_windows)],
        val_scale[: int(args.sample_windows)],
        val_drift[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
    )
    summary = {
        "args": vars(args),
        "config": asdict(model.cfg),
        "state_scope": args.state_scope,
        "normalization": norm_cfg,
        "finetune_objective": objective,
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "sample_smoke": smoke,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(make_serializable(records), indent=2), encoding="utf-8"
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(make_serializable(summary), indent=2), encoding="utf-8"
    )
    (output_dir / "args.json").write_text(
        json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8"
    )
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
