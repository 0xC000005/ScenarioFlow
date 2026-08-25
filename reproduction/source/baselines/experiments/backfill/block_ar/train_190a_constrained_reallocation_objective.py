#!/usr/bin/env python
"""
190a_v0: constrained reallocation objective on top of the 183c one-shot pathwise residual-law backbone.

Principle:
  - keep the 183c mean / covariance / pathwise residual-law backbone fixed
  - do not add another diffuse positive-only width head
  - learn a state-dependent temporal budget and sparse source/sink reallocations
  - make broad widening expensive and local reallocation cheap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import checkpoint_key, evaluate_frontier_subset
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _scaled_logit(value: float, low: float, high: float) -> float:
    return _logit((value - low) / max(high - low, 1e-8))


def soft_zone_masses(
    x_abs: torch.Tensor,
    q_quiet: torch.Tensor,
    q_extreme: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quiet_raw = torch.sigmoid((q_quiet - x_abs) / tau)
    extreme_raw = torch.sigmoid((x_abs - q_extreme) / tau)
    shoulder_raw = (1.0 - quiet_raw) * (1.0 - extreme_raw)
    denom = (quiet_raw + shoulder_raw + extreme_raw).clamp_min(1e-6)
    quiet = (quiet_raw / denom).mean(dim=-1)
    shoulder = (shoulder_raw / denom).mean(dim=-1)
    extreme = (extreme_raw / denom).mean(dim=-1)
    return quiet, shoulder, extreme


def instantiate_anchor_from_warm_start(warm_start_path: str, device: str) -> tuple[StateMetricTransportModel, dict]:
    payload = torch.load(warm_start_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = StateMetricTransportModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


class ConstrainedReallocationModel(StateMetricTransportModel):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        path_config: dict,
        prior_config: dict,
        integrated_config: dict,
        state_config: dict,
        metric_config: dict,
        reallocation_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
        mix_chunk_size: int = 27,
    ):
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            flow_config=flow_config,
            path_config=path_config,
            prior_config=prior_config,
            integrated_config=integrated_config,
            state_config=state_config,
            metric_config=metric_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        self.reallocation_config = dict(reallocation_config)
        self.n_time_blocks = int(reallocation_config["n_time_blocks"])
        self.block_topk = int(reallocation_config["support_topk"])
        self.temporal_topk = int(reallocation_config["temporal_topk"])
        assert self.decoder.n_frames % self.n_time_blocks == 0, "n_frames must be divisible by n_time_blocks"
        self.block_frames = self.decoder.n_frames // self.n_time_blocks
        hidden = int(reallocation_config["hidden_dim"])
        ctx_dim = path_config["context_dim"]

        budget_in = ctx_dim + 2
        self.total_budget_head = nn.Sequential(
            nn.Linear(budget_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.time_budget_head = nn.Sequential(
            nn.Linear(budget_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_time_blocks),
        )
        nn.init.zeros_(self.total_budget_head[-1].weight)
        nn.init.constant_(
            self.total_budget_head[-1].bias,
            _scaled_logit(
                reallocation_config["init_total_budget"],
                reallocation_config["total_budget_min"],
                reallocation_config["total_budget_max"],
            ),
        )
        nn.init.zeros_(self.time_budget_head[-1].bias)

        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        pos_dim = hidden + 5
        self.plus_head = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.minus_head = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        frame_idx = torch.arange(self.decoder.n_frames, dtype=torch.float32)
        cell_idx = torch.arange(self.decoder.n_cells, dtype=torch.float32)
        row = torch.div(cell_idx.long(), 5, rounding_mode="floor").float()
        col = torch.remainder(cell_idx.long(), 5).float()
        self.register_buffer("frame_norm", frame_idx / max(self.decoder.n_frames - 1, 1), persistent=False)
        self.register_buffer("row_norm", row / 4.0, persistent=False)
        self.register_buffer("col_norm", col / 4.0, persistent=False)
        block_index = torch.div(torch.arange(self.decoder.n_frames), self.block_frames, rounding_mode="floor").long()
        self.register_buffer("frame_block_index", block_index, persistent=False)

        local_coords = []
        for lf in range(self.block_frames):
            for c in range(self.decoder.n_cells):
                local_coords.append([lf / max(self.block_frames - 1, 1), float(c // 5) / 4.0, float(c % 5) / 4.0])
        local_coords = torch.tensor(local_coords, dtype=torch.float32)
        dist = (
            reallocation_config["time_leak_weight"] * (local_coords[:, None, 0] - local_coords[None, :, 0]).abs()
            + reallocation_config["space_leak_weight"]
            * (
                (local_coords[:, None, 1] - local_coords[None, :, 1]).abs()
                + (local_coords[:, None, 2] - local_coords[None, :, 2]).abs()
            )
        )
        dist = dist / dist.max().clamp_min(1e-6)
        self.register_buffer("block_distance", dist, persistent=False)

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Warm start loaded from {ckpt_path}")
        print(
            f"  Warm start missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | "
            f"shape-skipped: {len(skipped)}"
        )

    def total_budget(self, raw: torch.Tensor) -> torch.Tensor:
        low = self.reallocation_config["total_budget_min"]
        high = self.reallocation_config["total_budget_max"]
        return low + (high - low) * torch.sigmoid(raw)

    def _sparse_topk_distribution(self, logits: torch.Tensor, topk: int) -> torch.Tensor:
        n = logits.shape[-1]
        k = min(max(topk, 1), n)
        top_vals, top_idx = logits.topk(k, dim=-1)
        top_prob = torch.softmax(top_vals, dim=-1)
        out = torch.zeros_like(logits)
        out.scatter_(-1, top_idx, top_prob)
        return out

    def build_reallocation(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        metric_local_base: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = z_t_flat.shape[0]
        state_local, _state_band = self.build_state_features(z_t_flat)
        global_state = state_local.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        budget_in = torch.cat([path_context, t.unsqueeze(-1), global_state], dim=-1)
        total_budget = self.total_budget(self.total_budget_head(budget_in)).squeeze(-1)
        time_logits = self.time_budget_head(budget_in)
        temporal_split = self._sparse_topk_distribution(time_logits, self.temporal_topk)
        block_budget = total_budget.unsqueeze(-1) * temporal_split

        ctx = self.ctx_proj(path_context).view(batch, 1, 1, -1).expand(batch, self.decoder.n_frames, self.decoder.n_cells, -1)
        frame_feat = self.frame_norm.view(1, self.decoder.n_frames, 1, 1).expand(batch, self.decoder.n_frames, self.decoder.n_cells, 1)
        row_feat = self.row_norm.view(1, 1, self.decoder.n_cells, 1).expand(batch, self.decoder.n_frames, self.decoder.n_cells, 1)
        col_feat = self.col_norm.view(1, 1, self.decoder.n_cells, 1).expand(batch, self.decoder.n_frames, self.decoder.n_cells, 1)
        pos_feat = torch.cat(
            [ctx, metric_local_base.unsqueeze(-1), state_local.unsqueeze(-1), frame_feat, row_feat, col_feat],
            dim=-1,
        )
        plus_logits = self.plus_head(pos_feat).squeeze(-1)
        minus_logits = self.minus_head(pos_feat).squeeze(-1)

        plus_dist = torch.zeros_like(metric_local_base)
        minus_dist = torch.zeros_like(metric_local_base)
        plus_top = []
        minus_top = []
        for block in range(self.n_time_blocks):
            start = block * self.block_frames
            end = start + self.block_frames
            plus_block = plus_logits[:, start:end].reshape(batch, -1)
            minus_block = minus_logits[:, start:end].reshape(batch, -1)
            plus_block_dist = self._sparse_topk_distribution(plus_block, self.block_topk)
            minus_block_dist = self._sparse_topk_distribution(minus_block, self.block_topk)
            plus_top.append(plus_block_dist.max(dim=-1).values)
            minus_top.append(minus_block_dist.max(dim=-1).values)
            plus_dist[:, start:end] = plus_block_dist.view(batch, self.block_frames, self.decoder.n_cells)
            minus_dist[:, start:end] = minus_block_dist.view(batch, self.block_frames, self.decoder.n_cells)

        realloc = torch.zeros_like(metric_local_base)
        for block in range(self.n_time_blocks):
            start = block * self.block_frames
            end = start + self.block_frames
            realloc[:, start:end] = block_budget[:, block].view(batch, 1, 1) * (
                plus_dist[:, start:end] - minus_dist[:, start:end]
            )

        metrics = {
            "total_realloc_budget_mean": total_budget.mean(),
            "temporal_split_top1": temporal_split.max(dim=-1).values.mean(),
            "plus_top1": torch.stack(plus_top, dim=1).mean(),
            "minus_top1": torch.stack(minus_top, dim=1).mean(),
            "plus_support_rate": (plus_dist > 0).float().mean(),
            "minus_support_rate": (minus_dist > 0).float().mean(),
            "realloc_pos_mean": F.relu(realloc).mean(),
            "realloc_neg_mean": F.relu(-realloc).mean(),
        }
        pack = {
            "temporal_split": temporal_split,
            "block_budget": block_budget,
            "plus_dist": plus_dist,
            "minus_dist": minus_dist,
            "realloc": realloc,
            **metrics,
        }
        return realloc, pack

    def build_state_metric_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raw_local, raw_band, metric_local_base, metric_band, local_gate, band_gate = super().build_state_metric_controls(
            z_t_flat, t, path_context
        )
        realloc, realloc_pack = self.build_reallocation(z_t_flat, t, path_context, metric_local_base)
        metric_local = (metric_local_base + realloc).clamp(
            min=-self.metric_config["local_metric_clip"],
            max=self.metric_config["local_metric_clip"],
        )
        return raw_local, raw_band, metric_local, metric_band, local_gate, band_gate, realloc_pack

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_local, raw_band, metric_local, metric_band, local_gate, band_gate, realloc_pack = self.build_state_metric_controls(
            z_t_flat, t, path_context
        )
        raw_v = self.path_transport(z_t_flat, t, path_context)
        raw_v_basis = raw_v.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        mod_v_basis = self.modulate_velocity_basis(raw_v_basis, metric_local, metric_band)
        metrics = {
            "pred_local_abs_mean": raw_local.abs().mean(),
            "metric_local_abs_mean": metric_local.abs().mean(),
            "pred_band_high_mean": raw_band[:, 2].mean(),
            "pred_band_mid_mean": raw_band[:, 1].mean(),
            "pred_band_low_mean": raw_band[:, 0].mean(),
            "metric_band_high_mean": metric_band[:, 2].mean(),
            "metric_band_mid_mean": metric_band[:, 1].mean(),
            "metric_band_low_mean": metric_band[:, 0].mean(),
            "local_metric_budget": self.local_metric_budget(),
            "band_metric_budget": self.band_metric_budget(),
            "local_state_gate_mean": local_gate.mean(),
            "band_state_gate_mean": band_gate.mean(),
            "band_state_gate_high_mean": band_gate[:, 2].mean(),
            "pred_velocity_norm_mean": mod_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            **{k: v for k, v in realloc_pack.items() if isinstance(v, torch.Tensor) and v.ndim == 0},
        }
        return mod_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def _teacher_topk_distribution(x: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n = x.shape
    dist = torch.zeros_like(x)
    soft = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i in range(batch):
        xi = x[i]
        total = xi.sum()
        if total <= 1e-8:
            continue
        soft[i] = xi / total
        k = min(max(topk, 1), n)
        vals, idx = xi.topk(k)
        vals = vals.clamp_min(0.0)
        if vals.sum() <= 1e-8:
            continue
        dist[i, idx] = vals / vals.sum()
        mask[i, idx] = True
    return dist, soft, mask


def _distance_weight_from_mask(mask: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
    batch, n = mask.shape
    out = torch.ones(batch, n, device=mask.device, dtype=dist_matrix.dtype)
    for i in range(batch):
        idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        out[i] = dist_matrix[:, idx].min(dim=1).values
    return out


def constrained_reallocation_loss(
    model: ConstrainedReallocationModel,
    anchor_model: StateMetricTransportModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    with torch.no_grad():
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(history_01)
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)

        (
            a_mu,
            a_time_factor,
            a_time_diag,
            a_cell_factor,
            a_cell_diag,
            a_scale,
            a_flow_context,
            a_base_local_delta,
            a_block_logits,
        ) = anchor_model.forward_from_history(history_01)
        anchor_path_context = anchor_model.build_path_context(a_flow_context, a_block_logits, a_base_local_delta, a_scale)

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0

    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    raw_local, raw_band, metric_local, metric_band, _local_gate, _band_gate, realloc_pack = model.build_state_metric_controls(
        z_t, t, path_context
    )
    with torch.no_grad():
        _a_raw_local, _a_raw_band, anchor_metric_local, _a_metric_band, *_unused = anchor_model.build_state_metric_controls(
            z_t, t, anchor_path_context
        )

    delta_local = target_local_log - anchor_metric_local
    underfit = F.relu(delta_local)
    overwide = F.relu(-delta_local)
    block_budget_target = []
    plus_target = torch.zeros_like(metric_local)
    minus_target = torch.zeros_like(metric_local)
    plus_soft = torch.zeros_like(metric_local)
    minus_soft = torch.zeros_like(metric_local)
    plus_leak_dist = torch.zeros_like(metric_local)
    minus_leak_dist = torch.zeros_like(metric_local)
    plus_mass = []
    minus_mass = []
    for block in range(model.n_time_blocks):
        start = block * model.block_frames
        end = start + model.block_frames
        u_block = underfit[:, start:end].reshape(underfit.shape[0], -1)
        o_block = overwide[:, start:end].reshape(overwide.shape[0], -1)
        plus_dist_block, plus_soft_block, plus_mask_block = _teacher_topk_distribution(u_block, objective_config["teacher_topk"])
        minus_dist_block, minus_soft_block, minus_mask_block = _teacher_topk_distribution(o_block, objective_config["teacher_topk"])
        plus_target[:, start:end] = plus_dist_block.view_as(plus_target[:, start:end])
        minus_target[:, start:end] = minus_dist_block.view_as(minus_target[:, start:end])
        plus_soft[:, start:end] = plus_soft_block.view_as(plus_soft[:, start:end])
        minus_soft[:, start:end] = minus_soft_block.view_as(minus_soft[:, start:end])
        plus_leak_dist[:, start:end] = _distance_weight_from_mask(plus_mask_block, model.block_distance).view_as(
            plus_leak_dist[:, start:end]
        )
        minus_leak_dist[:, start:end] = _distance_weight_from_mask(minus_mask_block, model.block_distance).view_as(
            minus_leak_dist[:, start:end]
        )
        u_mean = u_block.sum(dim=-1) / float(u_block.shape[-1])
        o_mean = o_block.sum(dim=-1) / float(o_block.shape[-1])
        block_budget_target.append(torch.minimum(u_mean, o_mean))
        plus_mass.append((u_block.sum(dim=-1) > 1e-8).float())
        minus_mass.append((o_block.sum(dim=-1) > 1e-8).float())
    block_budget_target = torch.stack(block_budget_target, dim=1)
    plus_mass = torch.stack(plus_mass, dim=1)
    minus_mass = torch.stack(minus_mass, dim=1)

    fm_loss = F.mse_loss(pred_v, target_v)
    local_fit_loss = F.smooth_l1_loss(metric_local, target_local_log)
    band_fit_loss = F.smooth_l1_loss(metric_band, target_band_log)

    plus_dist = realloc_pack["plus_dist"]
    minus_dist = realloc_pack["minus_dist"]
    plus_align = -(plus_target * plus_dist.clamp_min(1e-8).log()).sum(dim=(1, 2))
    minus_align = -(minus_target * minus_dist.clamp_min(1e-8).log()).sum(dim=(1, 2))
    plus_weight = plus_target.sum(dim=(1, 2)) > 1e-8
    minus_weight = minus_target.sum(dim=(1, 2)) > 1e-8
    plus_align = plus_align[plus_weight].mean() if plus_weight.any() else metric_local.new_tensor(0.0)
    minus_align = minus_align[minus_weight].mean() if minus_weight.any() else metric_local.new_tensor(0.0)

    block_budget = realloc_pack["block_budget"]
    budget_loss = F.mse_loss(block_budget, block_budget_target)
    budget_excess = F.relu(block_budget - block_budget_target).mean()

    plus_leak = (plus_dist * plus_leak_dist * (1.0 - plus_soft)).sum(dim=(1, 2)).mean()
    minus_leak = (minus_dist * minus_leak_dist * (1.0 - minus_soft)).sum(dim=(1, 2)).mean()

    pos_realloc = F.relu(realloc_pack["realloc"])
    neg_realloc = F.relu(-realloc_pack["realloc"])
    overwide_soft = overwide / overwide.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    underfit_soft = underfit / underfit.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    overcover_penalty = (pos_realloc * overwide_soft).mean()
    undersource_penalty = (neg_realloc * underfit_soft).mean()

    target_abs = target_v.abs()
    pred_abs = pred_v.abs()
    q_quiet = torch.quantile(target_abs.detach(), objective_config["quiet_quantile"], dim=-1, keepdim=True)
    q_extreme = torch.quantile(target_abs.detach(), objective_config["extreme_quantile"], dim=-1, keepdim=True)
    target_quiet, target_shoulder, target_extreme = soft_zone_masses(
        target_abs, q_quiet, q_extreme, objective_config["spectrum_tau"]
    )
    pred_quiet, pred_shoulder, pred_extreme = soft_zone_masses(
        pred_abs, q_quiet, q_extreme, objective_config["spectrum_tau"]
    )
    spectrum_loss = (
        (pred_quiet - target_quiet).abs()
        + (pred_shoulder - target_shoulder).abs()
        + (pred_extreme - target_extreme).abs()
    ).mean()

    underfit_gap = (
        F.relu(target_local_log - metric_local) * (underfit > 0).float()
    ).sum() / (underfit > 0).float().sum().clamp_min(1.0)
    overwide_gap = (
        F.relu(metric_local - target_local_log) * (overwide > 0).float()
    ).sum() / (overwide > 0).float().sum().clamp_min(1.0)

    total = (
        fm_loss
        + objective_config["local_fit_weight"] * local_fit_loss
        + objective_config["band_fit_weight"] * band_fit_loss
        + objective_config["plus_align_weight"] * plus_align
        + objective_config["minus_align_weight"] * minus_align
        + objective_config["budget_loss_weight"] * budget_loss
        + objective_config["budget_excess_weight"] * budget_excess
        + objective_config["plus_leak_weight"] * plus_leak
        + objective_config["minus_leak_weight"] * minus_leak
        + objective_config["overcover_weight"] * overcover_penalty
        + objective_config["undersource_weight"] * undersource_penalty
        + objective_config["underfit_gap_weight"] * underfit_gap
        + objective_config["overwide_gap_weight"] * overwide_gap
        + objective_config["spectrum_loss_weight"] * spectrum_loss
    )

    high_mask = model.path_geometry.high_band_mask().to(target_basis.device, dtype=target_basis.dtype)
    target_jump_like = (target_basis_flat.abs() > 2.5).float().mean(dim=-1)
    metrics = {
        "flow_match_loss": fm_loss,
        "target_basis_std_mean": target_basis_flat.std(dim=-1).mean(),
        "prior_std_mean": prior_stats["prior_std_mean"],
        "prior_jump_prob_mean": prior_stats["prior_jump_prob_mean"],
        "prior_jump_scale_mean": prior_stats["prior_jump_scale_mean"],
        "prior_active_rate": prior_stats["prior_active_rate"],
        "target_jump_like_rate": target_jump_like.mean(),
        "target_high_band_abs_mean": (
            (target_basis_flat.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)
        ).mean(),
        "realloc_total_loss": total,
        "local_fit_loss": local_fit_loss,
        "band_fit_loss": band_fit_loss,
        "plus_align_loss": plus_align,
        "minus_align_loss": minus_align,
        "budget_loss": budget_loss,
        "budget_excess": budget_excess,
        "plus_leak": plus_leak,
        "minus_leak": minus_leak,
        "overcover_penalty": overcover_penalty,
        "undersource_penalty": undersource_penalty,
        "underfit_gap": underfit_gap,
        "overwide_gap": overwide_gap,
        "spectrum_loss": spectrum_loss,
        "budget_target_mean": block_budget_target.mean(),
        "pred_quiet_mass": pred_quiet.mean(),
        "pred_shoulder_mass": pred_shoulder.mean(),
        "pred_extreme_mass": pred_extreme.mean(),
        "target_quiet_mass": target_quiet.mean(),
        "target_shoulder_mass": target_shoulder.mean(),
        "target_extreme_mass": target_extreme.mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: ConstrainedReallocationModel,
    anchor_model: StateMetricTransportModel,
    val_loader: DataLoader,
    objective_config: dict,
) -> dict[str, float]:
    model.eval()
    keys = [
        "flow_match_loss",
        "target_basis_std_mean",
        "prior_std_mean",
        "prior_jump_prob_mean",
        "prior_jump_scale_mean",
        "prior_active_rate",
        "pred_velocity_norm_mean",
        "target_jump_like_rate",
        "target_high_band_abs_mean",
        "realloc_total_loss",
        "local_fit_loss",
        "band_fit_loss",
        "plus_align_loss",
        "minus_align_loss",
        "budget_loss",
        "budget_excess",
        "plus_leak",
        "minus_leak",
        "overcover_penalty",
        "undersource_penalty",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "budget_target_mean",
        "pred_quiet_mass",
        "pred_shoulder_mass",
        "pred_extreme_mass",
        "target_quiet_mass",
        "target_shoulder_mass",
        "target_extreme_mass",
        "pred_local_abs_mean",
        "metric_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "metric_band_high_mean",
        "metric_band_mid_mean",
        "metric_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "total_realloc_budget_mean",
        "temporal_split_top1",
        "plus_top1",
        "minus_top1",
        "plus_support_rate",
        "minus_support_rate",
        "realloc_pos_mean",
        "realloc_neg_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = constrained_reallocation_loss(model, anchor_model, history_01, future_01, objective_config)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_realloc_total_loss"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="190a_v0: constrained reallocation objective on top of 183c")
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_stage1", type=float, default=5e-4)
    parser.add_argument("--lr_stage2", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--n_time_blocks", type=int, default=5)
    parser.add_argument("--temporal_topk", type=int, default=2)
    parser.add_argument("--support_topk", type=int, default=4)
    parser.add_argument("--teacher_topk", type=int, default=4)
    parser.add_argument("--realloc_hidden_dim", type=int, default=128)
    parser.add_argument("--total_budget_min", type=float, default=0.02)
    parser.add_argument("--total_budget_max", type=float, default=0.25)
    parser.add_argument("--init_total_budget", type=float, default=0.08)
    parser.add_argument("--time_leak_weight", type=float, default=0.45)
    parser.add_argument("--space_leak_weight", type=float, default=0.55)
    parser.add_argument("--local_fit_weight", type=float, default=0.45)
    parser.add_argument("--band_fit_weight", type=float, default=0.18)
    parser.add_argument("--plus_align_weight", type=float, default=0.70)
    parser.add_argument("--minus_align_weight", type=float, default=0.55)
    parser.add_argument("--budget_loss_weight", type=float, default=0.70)
    parser.add_argument("--budget_excess_weight", type=float, default=1.10)
    parser.add_argument("--plus_leak_weight", type=float, default=1.10)
    parser.add_argument("--minus_leak_weight", type=float, default=0.75)
    parser.add_argument("--overcover_weight", type=float, default=1.15)
    parser.add_argument("--undersource_weight", type=float, default=0.55)
    parser.add_argument("--underfit_gap_weight", type=float, default=0.35)
    parser.add_argument("--overwide_gap_weight", type=float, default=0.35)
    parser.add_argument("--quiet_quantile", type=float, default=0.50)
    parser.add_argument("--extreme_quantile", type=float, default=0.99)
    parser.add_argument("--spectrum_tau", type=float, default=0.15)
    parser.add_argument("--spectrum_loss_weight", type=float, default=0.22)
    parser.add_argument(
        "--warm_start_path",
        type=str,
        default=(
            "models/backfill/"
            "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_183c/best_model.pt"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    surf_tensor = torch.from_numpy(surfaces).to(args.device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    anchor_model, warm_cfg = instantiate_anchor_from_warm_start(args.warm_start_path, args.device)
    reallocation_config = {
        "n_time_blocks": args.n_time_blocks,
        "temporal_topk": args.temporal_topk,
        "support_topk": args.support_topk,
        "hidden_dim": args.realloc_hidden_dim,
        "total_budget_min": args.total_budget_min,
        "total_budget_max": args.total_budget_max,
        "init_total_budget": args.init_total_budget,
        "time_leak_weight": args.time_leak_weight,
        "space_leak_weight": args.space_leak_weight,
    }
    objective_config = {
        "teacher_topk": args.teacher_topk,
        "local_fit_weight": args.local_fit_weight,
        "band_fit_weight": args.band_fit_weight,
        "plus_align_weight": args.plus_align_weight,
        "minus_align_weight": args.minus_align_weight,
        "budget_loss_weight": args.budget_loss_weight,
        "budget_excess_weight": args.budget_excess_weight,
        "plus_leak_weight": args.plus_leak_weight,
        "minus_leak_weight": args.minus_leak_weight,
        "overcover_weight": args.overcover_weight,
        "undersource_weight": args.undersource_weight,
        "underfit_gap_weight": args.underfit_gap_weight,
        "overwide_gap_weight": args.overwide_gap_weight,
        "quiet_quantile": args.quiet_quantile,
        "extreme_quantile": args.extreme_quantile,
        "spectrum_tau": args.spectrum_tau,
        "spectrum_loss_weight": args.spectrum_loss_weight,
    }

    model = ConstrainedReallocationModel(
        encoder_config=EncoderConfig(**warm_cfg["encoder"]),
        decoder_config=warm_cfg["decoder"],
        flow_config=warm_cfg["flow"],
        path_config=warm_cfg["path"],
        prior_config=warm_cfg["prior"],
        integrated_config=warm_cfg["integrated"],
        state_config=warm_cfg["state"],
        metric_config=warm_cfg["metric"],
        reallocation_config=reallocation_config,
        support_lo=warm_cfg.get("support_lo", 0.01),
        support_hi=warm_cfg.get("support_hi", 1.0),
        support_eps=warm_cfg.get("support_eps", 1e-5),
        base_nu=warm_cfg.get("base_nu", 8.0),
        mix_chunk_size=warm_cfg.get("mix_chunk_size", 27),
    ).to(args.device)
    model.maybe_load_warm_start(args.warm_start_path, args.device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)
    model.path_transport.requires_grad_(False)
    model.path_context_adapter.requires_grad_(False)
    model.width_allocator.requires_grad_(False)
    model.band_tail.requires_grad_(False)
    model.local_state_gate.requires_grad_(False)
    model.band_state_gate.requires_grad_(False)
    model.local_metric_budget_logit.requires_grad_(False)
    model.band_metric_budget_logit.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    trainable_params = (
        list(model.total_budget_head.parameters())
        + list(model.time_budget_head.parameters())
        + list(model.ctx_proj.parameters())
        + list(model.plus_head.parameters())
        + list(model.minus_head.parameters())
    )

    def set_stage(stage: int):
        lr = args.lr_stage1 if stage == 1 else args.lr_stage2
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs_stage1 if stage == 1 else args.epochs_stage2, 1)
        )
        return optimizer, scheduler, f"realloc-{'warm' if stage == 1 else 'refine'}"

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n190a_v0: constrained reallocation objective\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start: {args.warm_start_path}")
    print(
        f"  nBlocks={args.n_time_blocks}  tTopK={args.temporal_topk}  sTopK={args.support_topk}  "
        f"budget=[{args.total_budget_min:.2f},{args.total_budget_max:.2f}]"
    )

    metric_keys = [
        "flow_match_loss",
        "target_basis_std_mean",
        "prior_std_mean",
        "prior_jump_prob_mean",
        "prior_jump_scale_mean",
        "prior_active_rate",
        "pred_velocity_norm_mean",
        "target_jump_like_rate",
        "target_high_band_abs_mean",
        "realloc_total_loss",
        "local_fit_loss",
        "band_fit_loss",
        "plus_align_loss",
        "minus_align_loss",
        "budget_loss",
        "budget_excess",
        "plus_leak",
        "minus_leak",
        "overcover_penalty",
        "undersource_penalty",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "budget_target_mean",
        "pred_quiet_mass",
        "pred_shoulder_mass",
        "pred_extreme_mass",
        "target_quiet_mass",
        "target_shoulder_mass",
        "target_extreme_mass",
        "pred_local_abs_mean",
        "metric_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "metric_band_high_mean",
        "metric_band_mid_mean",
        "metric_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "total_realloc_budget_mean",
        "temporal_split_top1",
        "plus_top1",
        "minus_top1",
        "plus_support_rate",
        "minus_support_rate",
        "realloc_pos_mean",
        "realloc_neg_mean",
    ]

    for epoch in range(1, total_epochs + 1):
        if epoch == args.epochs_stage1 + 1:
            optimizer, scheduler, stage_name = set_stage(2)
        t0 = time.time()
        model.train()
        totals = {f"train_{k}": 0.0 for k in metric_keys}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            history_01 = history_01.to(args.device)
            future_01 = future_01.to(args.device)
            loss, metrics = constrained_reallocation_loss(model, anchor_model, history_01, future_01, objective_config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += metrics[key.replace("train_", "")].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(model, anchor_model, val_loader, objective_config)
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        frontier_metrics = evaluate_frontier_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        current_key = checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics, **frontier_metrics}
            cfg_out = dict(warm_cfg)
            cfg_out.update(
                {
                    "type": "constrained_reallocation_objective_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_190a",
                    "reallocation": reallocation_config,
                    "objective": objective_config,
                    "history_len": args.history_len,
                    "future_len": args.future_len,
                    "train_windows": len(train_indices),
                    "val_windows": len(val_indices),
                    "frozen_backbone": True,
                    "anchor_path": args.warm_start_path,
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": cfg_out,
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        elapsed = time.time() - t0
        row = {"epoch": epoch, "stage": stage_name, **train_metrics, **val_metrics, **joint_metrics, **frontier_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))
        print(
            f"Ep {epoch:3d} [{stage_name}]  "
            f"val_total={val_metrics['val_total_loss']:.4f}  "
            f"worstLate={frontier_metrics['frontier_turb_late_worst_cov']:.3f}  "
            f"bestLate={frontier_metrics['frontier_turb_late_best_cov']:.3f}  "
            f"kurt={frontier_metrics['frontier_pooled_kurt_ratio']:.3f}  "
            f"highE={frontier_metrics['high_energy_ratio_p50']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"uGap={val_metrics['val_underfit_gap']:.3f}  oGap={val_metrics['val_overwide_gap']:.3f}  "
            f"bgt={val_metrics['val_total_realloc_budget_mean']:.3f}  tTop={val_metrics['val_temporal_split_top1']:.3f}  "
            f"pTop={val_metrics['val_plus_top1']:.3f}  pSupp={val_metrics['val_plus_support_rate']:.3f}  "
            f"pLeak={val_metrics['val_plus_leak']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    cfg_out = dict(warm_cfg)
    cfg_out.update(
        {
            "type": "constrained_reallocation_objective_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_190a",
            "reallocation": reallocation_config,
            "objective": objective_config,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "frozen_backbone": True,
            "anchor_path": args.warm_start_path,
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": total_epochs,
            "config": cfg_out,
            "best_key": best_key,
            "best_metrics": best_metrics,
        },
        f"{args.output_dir}/final_model.pt",
    )
    history_path.write_text(json.dumps(make_serializable(history), indent=2))


if __name__ == "__main__":
    main()
