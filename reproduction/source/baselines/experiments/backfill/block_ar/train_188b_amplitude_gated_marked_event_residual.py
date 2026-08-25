#!/usr/bin/env python
"""
188b_v0: Amplitude-gated marked-event residual model.

Mechanism fix over 188a:
  - keep the marked-event object abstraction
  - force the event branch to act only through decoded event amplitude/support
  - lower prior activity defaults and penalize always-on event slots
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import evaluate_frontier_subset
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel
from experiments.backfill.block_ar.train_188a_graph_group_marked_event_residual import (
    MarkedEventResidualModel as BaseMarkedEventResidualModel,
    bernoulli_kl,
)


def default_encoder_config() -> EncoderConfig:
    return EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)


def default_decoder_config(future_len: int) -> dict:
    return dict(
        n_frames=future_len,
        n_cells=25,
        d_model=128,
        n_heads=4,
        n_layers=4,
        cond_dim=128,
        time_rank=6,
        cell_rank=5,
        diag_floor=1e-3,
        scale_floor=1e-4,
        init_diag=0.05,
        init_scale=0.10,
        flow_context_dim=256,
        local_delta_clip=0.35,
        n_blocks=5,
        n_templates=3,
        template_diag_clip=0.30,
        template_offdiag_clip=0.18,
        drift_strength_max=0.75,
        equilibrium_offset_clip=0.20,
        init_drift_strength=0.20,
    )


def default_flow_config(future_len: int) -> dict:
    return dict(
        dim=future_len * 25,
        context_dim=256,
        n_frames=future_len,
        grid_h=5,
        grid_w=5,
        hidden_dim=256,
        n_layers=4,
        low_scale_clip=1.2,
        mid_scale_clip=0.7,
        high_scale_clip=0.35,
    )


def default_path_config() -> dict:
    return dict(
        context_dim=256,
        context_hidden_dim=256,
        d_model=192,
        n_heads=4,
        n_layers=4,
        ff_mult=4,
        time_embed_dim=64,
        n_ode_steps=8,
    )


def default_prior_config() -> dict:
    return dict(
        low_std=1.00,
        mid_std=0.75,
        high_std=0.35,
        low_jump_prob=0.005,
        mid_jump_prob=0.025,
        high_jump_prob=0.070,
        low_jump_scale=0.10,
        mid_jump_scale=0.30,
        high_jump_scale=0.75,
    )


def default_integrated_config() -> dict:
    return dict(
        width_rank=4,
        width_hidden_dim=256,
        width_clip=0.80,
        band_hidden_dim=128,
        band_clip=0.45,
        local_loss_weight=0.75,
        band_loss_weight=0.50,
        smooth_reg_weight=0.04,
        init_local_strength=0.05,
        init_band_strength=0.05,
    )


def default_state_config() -> dict:
    return dict(
        gate_hidden_dim=128,
        init_local_state_gate=0.30,
        init_band_state_gate=0.25,
    )


def default_metric_config() -> dict:
    return dict(
        local_metric_min=0.12,
        local_metric_max=0.55,
        band_metric_min=0.08,
        band_metric_max=0.35,
        init_local_metric_budget=0.24,
        init_band_metric_budget=0.14,
        local_metric_clip=0.60,
        band_metric_clip=0.35,
        budget_reg_weight=0.02,
    )


def default_slot_config() -> dict:
    return dict(
        n_slots=2,
        n_event_blocks=5,
        slot_embed_dim=16,
        slot_hidden_dim=128,
        teacher_feat_dim=8,
        init_prior_gate=0.03,
        init_post_gate=0.08,
        teacher_peak_threshold=0.18,
        teacher_suppress_time=3,
        teacher_suppress_radius=1.5,
        radius_min=0.50,
        radius_max=2.50,
        duration_min=1.0,
        duration_max=8.0,
        post_active_bce_weight=0.35,
        prior_active_bce_weight=0.10,
        post_time_ce_weight=0.25,
        prior_time_ce_weight=0.08,
        post_node_ce_weight=0.25,
        prior_node_ce_weight=0.08,
        amp_loss_weight=0.40,
        radius_loss_weight=0.12,
        duration_loss_weight=0.12,
        active_kl_weight=0.05,
        time_kl_weight=0.02,
        node_kl_weight=0.02,
        slot_overlap_weight=0.02,
        quiet_dominance_weight=0.04,
        activity_target=0.10,
        activity_excess_weight=0.12,
        prior_amp_min=0.05,
        prior_node_top1_min=0.30,
    )


def default_event_config() -> dict:
    return dict(
        amp_max=0.90,
        band_max=0.25,
        event_local_clip=0.70,
        event_band_clip=0.25,
        event_scale_max=1.20,
        init_event_scale=0.20,
        quiet_local_loss_weight=0.15,
        quiet_band_loss_weight=0.10,
        event_local_loss_weight=0.60,
        event_band_loss_weight=0.25,
    )


class AmplitudeGatedMarkedEventResidualModel(BaseMarkedEventResidualModel):
    def event_white_gate(
        self,
        event_local: torch.Tensor,
        event_band: torch.Tensor,
        event_raw_basis: torch.Tensor,
    ) -> torch.Tensor:
        band_map = torch.einsum(
            "bk,ktc->btc",
            event_band,
            self.band_masks.to(device=event_raw_basis.device, dtype=event_raw_basis.dtype),
        )
        band_scaled_basis = event_raw_basis * torch.exp(band_map)
        event_white = self.path_geometry.from_basis(band_scaled_basis)
        # Crucial 188b fix: zero decoded local field => zero event contribution.
        gated_white = event_white * event_local
        return self.path_geometry.to_basis(gated_white)

    def transport_velocity_with_pack(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        slot_pack: dict[str, dict[str, torch.Tensor]] | None = None,
        use_posterior: bool = False,
    ):
        (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            slot_dec,
        ) = self.build_marked_event_controls(z_t_flat, t, path_context, slot_pack=slot_pack, use_posterior=use_posterior)

        quiet_raw = self.path_transport(z_t_flat, t, path_context)
        quiet_raw_basis = quiet_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        quiet_v_basis = self.modulate_velocity_basis(quiet_raw_basis, quiet_metric_local, quiet_metric_band)

        event_raw = self.event_transport(z_t_flat, t, path_context)
        event_raw_basis = event_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        gated_event_v_basis = self.event_white_gate(event_local, event_band, event_raw_basis)
        total_v_basis = quiet_v_basis + self.event_scale() * gated_event_v_basis

        metrics = {
            "pred_local_abs_mean": raw_local.abs().mean(),
            "quiet_metric_local_abs_mean": quiet_metric_local.abs().mean(),
            "event_local_abs_mean": event_local.abs().mean(),
            "pred_band_high_mean": raw_band[:, 2].mean(),
            "pred_band_mid_mean": raw_band[:, 1].mean(),
            "pred_band_low_mean": raw_band[:, 0].mean(),
            "quiet_band_high_mean": quiet_metric_band[:, 2].mean(),
            "quiet_band_mid_mean": quiet_metric_band[:, 1].mean(),
            "quiet_band_low_mean": quiet_metric_band[:, 0].mean(),
            "event_band_high_mean": event_band[:, 2].mean(),
            "event_band_mid_mean": event_band[:, 1].mean(),
            "event_band_low_mean": event_band[:, 0].mean(),
            "local_metric_budget": self.local_metric_budget(),
            "band_metric_budget": self.band_metric_budget(),
            "event_scale": self.event_scale(),
            "local_state_gate_mean": local_gate.mean(),
            "band_state_gate_mean": band_gate.mean(),
            "band_state_gate_high_mean": band_gate[:, 2].mean(),
            "slot_active_mean": slot_dec["active_prob"].mean(),
            "slot_time_top1_mean": slot_dec["time_probs"].max(dim=-1).values.mean(),
            "slot_node_top1_mean": slot_dec["node_probs"].max(dim=-1).values.mean(),
            "slot_amp_abs_mean": slot_dec["amp"].abs().mean(),
            "slot_radius_mean": slot_dec["radius"].mean(),
            "slot_duration_mean": slot_dec["duration"].mean(),
            "quiet_velocity_norm_mean": quiet_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "event_velocity_norm_mean": gated_event_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "raw_event_velocity_norm_mean": event_raw_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "pred_velocity_norm_mean": total_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return total_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def amplitude_gated_marked_event_flow_matching_loss(
    model: AmplitudeGatedMarkedEventResidualModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
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

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0

    _, _, quiet_metric_local, quiet_metric_band, _, _ = StateMetricTransportModel.build_state_metric_controls(
        model, z_t, t, path_context
    )
    target_event_local = (target_local_log - quiet_metric_local).clamp(
        min=-model.event_config["event_local_clip"],
        max=model.event_config["event_local_clip"],
    )
    target_event_band = (target_band_log - quiet_metric_band).clamp(
        min=-model.event_config["event_band_clip"],
        max=model.event_config["event_band_clip"],
    )

    teacher_slots = model.extract_teacher_slots(target_event_local.detach())
    slot_pack = model.infer_slots(path_context, teacher_slots=teacher_slots)
    pred_v, control_metrics = model.transport_velocity_with_pack(
        z_t, t, path_context, slot_pack=slot_pack, use_posterior=True
    )
    fm_loss = F.mse_loss(pred_v, target_v)

    (
        _raw_local,
        _raw_band,
        quiet_metric_local2,
        quiet_metric_band2,
        event_local,
        event_band,
        _local_gate,
        _band_gate,
        post_dec,
    ) = model.build_marked_event_controls(z_t, t, path_context, slot_pack=slot_pack, use_posterior=True)
    prior_dec = model._decode_slots(slot_pack["prior"])

    teacher_dec = {
        "active_prob": teacher_slots["active"],
        "time_probs": F.one_hot(teacher_slots["time_idx"], num_classes=model.n_event_blocks).to(target_basis.dtype),
        "node_probs": F.one_hot(teacher_slots["node_idx"], num_classes=model.n_nodes).to(target_basis.dtype),
        "amp": teacher_slots["amp"],
        "radius": teacher_slots["radius"],
        "duration": teacher_slots["duration"],
        "band": torch.zeros_like(post_dec["band"]),
    }
    teacher_event_local, _ = model.decode_event_field(teacher_dec)

    quiet_local_loss = F.smooth_l1_loss(quiet_metric_local2, target_local_log)
    quiet_band_loss = F.smooth_l1_loss(quiet_metric_band2, target_band_log)
    event_local_loss = F.smooth_l1_loss(event_local, teacher_event_local)
    event_band_loss = F.smooth_l1_loss(event_band, target_event_band)

    active_mask = teacher_slots["active"]
    active_norm = active_mask.sum().clamp_min(1.0)
    post_active_bce = F.binary_cross_entropy(post_dec["active_prob"], active_mask)
    prior_active_bce = F.binary_cross_entropy(prior_dec["active_prob"], active_mask)

    time_targets = teacher_slots["time_idx"].reshape(-1)
    node_targets = teacher_slots["node_idx"].reshape(-1)
    flat_mask = active_mask.reshape(-1)
    post_time_ce = (
        F.cross_entropy(slot_pack["post"]["time_logits"].reshape(-1, model.n_event_blocks), time_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    prior_time_ce = (
        F.cross_entropy(slot_pack["prior"]["time_logits"].reshape(-1, model.n_event_blocks), time_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    post_node_ce = (
        F.cross_entropy(slot_pack["post"]["node_logits"].reshape(-1, model.n_nodes), node_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    prior_node_ce = (
        F.cross_entropy(slot_pack["prior"]["node_logits"].reshape(-1, model.n_nodes), node_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm

    amp_loss = (F.smooth_l1_loss(post_dec["amp"], teacher_slots["amp"], reduction="none") * active_mask).sum() / active_norm
    radius_loss = (
        F.smooth_l1_loss(post_dec["radius"], teacher_slots["radius"], reduction="none") * active_mask
    ).sum() / active_norm
    duration_loss = (
        F.smooth_l1_loss(post_dec["duration"], teacher_slots["duration"], reduction="none") * active_mask
    ).sum() / active_norm

    active_kl = bernoulli_kl(post_dec["active_prob"], prior_dec["active_prob"]).mean()
    time_kl = (
        post_dec["time_probs"] * (post_dec["time_probs"].clamp_min(1e-6).log() - prior_dec["time_probs"].clamp_min(1e-6).log())
    ).sum(dim=-1)
    time_kl = (time_kl * post_dec["active_prob"]).mean() / post_dec["active_prob"].mean().clamp_min(0.05)
    node_kl = (
        post_dec["node_probs"] * (post_dec["node_probs"].clamp_min(1e-6).log() - prior_dec["node_probs"].clamp_min(1e-6).log())
    ).sum(dim=-1)
    node_kl = (node_kl * post_dec["active_prob"]).mean() / post_dec["active_prob"].mean().clamp_min(0.05)

    slot_overlap = torch.zeros((), device=target_basis.device, dtype=target_basis.dtype)
    if model.n_slots > 1:
        kernels = []
        for k in range(model.n_slots):
            single = {
                "active_prob": post_dec["active_prob"][:, k : k + 1],
                "time_probs": post_dec["time_probs"][:, k : k + 1],
                "node_probs": post_dec["node_probs"][:, k : k + 1],
                "amp": post_dec["amp"][:, k : k + 1].abs(),
                "radius": post_dec["radius"][:, k : k + 1],
                "duration": post_dec["duration"][:, k : k + 1],
                "band": post_dec["band"][:, k : k + 1],
            }
            kern, _ = model.decode_event_field(single)
            kernels.append(kern.reshape(kern.shape[0], -1))
        overlap_terms = []
        for i in range(model.n_slots):
            for j in range(i + 1, model.n_slots):
                ki = F.normalize(kernels[i], dim=-1)
                kj = F.normalize(kernels[j], dim=-1)
                overlap_terms.append((ki * kj).sum(dim=-1).mean())
        if overlap_terms:
            slot_overlap = torch.stack(overlap_terms).mean()

    quiet_dom = post_dec["active_prob"].mean()
    activity_target = float(model.slot_config.get("activity_target", 0.10))
    activity_excess = F.relu(prior_dec["active_prob"].mean() - activity_target)

    total = (
        fm_loss
        + model.event_config["quiet_local_loss_weight"] * quiet_local_loss
        + model.event_config["quiet_band_loss_weight"] * quiet_band_loss
        + model.event_config["event_local_loss_weight"] * event_local_loss
        + model.event_config["event_band_loss_weight"] * event_band_loss
        + model.slot_config["post_active_bce_weight"] * post_active_bce
        + model.slot_config["prior_active_bce_weight"] * prior_active_bce
        + model.slot_config["post_time_ce_weight"] * post_time_ce
        + model.slot_config["prior_time_ce_weight"] * prior_time_ce
        + model.slot_config["post_node_ce_weight"] * post_node_ce
        + model.slot_config["prior_node_ce_weight"] * prior_node_ce
        + model.slot_config["amp_loss_weight"] * amp_loss
        + model.slot_config["radius_loss_weight"] * radius_loss
        + model.slot_config["duration_loss_weight"] * duration_loss
        + model.slot_config["active_kl_weight"] * active_kl
        + model.slot_config["time_kl_weight"] * time_kl
        + model.slot_config["node_kl_weight"] * node_kl
        + model.slot_config["slot_overlap_weight"] * slot_overlap
        + model.slot_config["quiet_dominance_weight"] * quiet_dom
        + model.slot_config["activity_excess_weight"] * activity_excess
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
        "pred_velocity_norm_mean": control_metrics["pred_velocity_norm_mean"],
        "target_jump_like_rate": target_jump_like.mean(),
        "target_high_band_abs_mean": (
            (target_basis_flat.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)
        ).mean(),
        "marked_total_loss": total,
        "quiet_local_loss": quiet_local_loss,
        "quiet_band_loss": quiet_band_loss,
        "event_local_loss": event_local_loss,
        "event_band_loss": event_band_loss,
        "post_active_bce": post_active_bce,
        "prior_active_bce": prior_active_bce,
        "post_time_ce": post_time_ce,
        "prior_time_ce": prior_time_ce,
        "post_node_ce": post_node_ce,
        "prior_node_ce": prior_node_ce,
        "amp_loss": amp_loss,
        "radius_loss": radius_loss,
        "duration_loss": duration_loss,
        "active_kl": active_kl,
        "time_kl": time_kl,
        "node_kl": node_kl,
        "slot_overlap": slot_overlap,
        "quiet_dominance": quiet_dom,
        "activity_excess": activity_excess,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "target_event_local_abs_mean": target_event_local.abs().mean(),
        "target_event_band_high_mean": target_event_band[:, 2].mean(),
        "teacher_slot_active_rate": teacher_slots["active"].mean(),
        "teacher_slot_amp_abs_mean": teacher_slots["amp"].abs().mean(),
        "teacher_slot_radius_mean": teacher_slots["radius"].mean(),
        "teacher_slot_duration_mean": teacher_slots["duration"].mean(),
        "post_slot_active_mean": post_dec["active_prob"].mean(),
        "prior_slot_active_mean": prior_dec["active_prob"].mean(),
        "post_slot_amp_abs_mean": post_dec["amp"].abs().mean(),
        "prior_slot_amp_abs_mean": prior_dec["amp"].abs().mean(),
        "post_slot_time_top1_mean": post_dec["time_probs"].max(dim=-1).values.mean(),
        "prior_slot_time_top1_mean": prior_dec["time_probs"].max(dim=-1).values.mean(),
        "post_slot_node_top1_mean": post_dec["node_probs"].max(dim=-1).values.mean(),
        "prior_slot_node_top1_mean": prior_dec["node_probs"].max(dim=-1).values.mean(),
        "activity_target": torch.tensor(activity_target, device=target_basis.device, dtype=target_basis.dtype),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(model: AmplitudeGatedMarkedEventResidualModel, val_loader: DataLoader) -> dict[str, float]:
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
        "marked_total_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_active_bce",
        "prior_active_bce",
        "post_time_ce",
        "prior_time_ce",
        "post_node_ce",
        "prior_node_ce",
        "amp_loss",
        "radius_loss",
        "duration_loss",
        "active_kl",
        "time_kl",
        "node_kl",
        "slot_overlap",
        "quiet_dominance",
        "activity_excess",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "teacher_slot_active_rate",
        "teacher_slot_amp_abs_mean",
        "teacher_slot_radius_mean",
        "teacher_slot_duration_mean",
        "post_slot_active_mean",
        "prior_slot_active_mean",
        "post_slot_amp_abs_mean",
        "prior_slot_amp_abs_mean",
        "post_slot_time_top1_mean",
        "prior_slot_time_top1_mean",
        "post_slot_node_top1_mean",
        "prior_slot_node_top1_mean",
        "activity_target",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "event_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "quiet_band_high_mean",
        "quiet_band_mid_mean",
        "quiet_band_low_mean",
        "event_band_high_mean",
        "event_band_mid_mean",
        "event_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "event_scale",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "slot_active_mean",
        "slot_time_top1_mean",
        "slot_node_top1_mean",
        "slot_amp_abs_mean",
        "slot_radius_mean",
        "slot_duration_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
        "raw_event_velocity_norm_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = amplitude_gated_marked_event_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_marked_total_loss"]
    return out


def strict_checkpoint_key_188b(val_metrics: dict, frontier_metrics: dict, joint_metrics: dict) -> tuple[float, ...]:
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    worst_gap = max(0.70 - float(frontier_metrics.get("frontier_turb_late_worst_cov", float("nan"))), 0.0)
    best_gap = max(float(frontier_metrics.get("frontier_turb_late_best_cov", float("nan"))) - 0.95, 0.0)
    kurt = float(frontier_metrics.get("frontier_pooled_kurt_ratio", float("nan")))
    kurt_gap = 1e6 if not np.isfinite(kurt) else max(0.8 - kurt, 0.0) + max(kurt - 1.25, 0.0)
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("nan")))
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    target = float(val_metrics.get("val_activity_target", 0.10))
    slot_excess = max(float(val_metrics.get("val_prior_slot_active_mean", 0.0)) - target, 0.0)
    amp_gap = max(float(default_slot_config()["prior_amp_min"]) - float(val_metrics.get("val_prior_slot_amp_abs_mean", 0.0)), 0.0)
    node_gap = max(float(default_slot_config()["prior_node_top1_min"]) - float(val_metrics.get("val_prior_slot_node_top1_mean", 0.0)), 0.0)
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    return (tc_gap, worst_gap, best_gap, kurt_gap, slot_excess, amp_gap, node_gap, jump_gap, mr_gap, val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="188b_v0: amplitude-gated marked-event residual model")
    parser.add_argument("--epochs_stage1", type=int, default=3)
    parser.add_argument("--epochs_stage2", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_event_stage1", type=float, default=7e-4)
    parser.add_argument("--lr_event_stage2", type=float, default=2.5e-4)
    parser.add_argument("--lr_adapter_stage2", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument(
        "--warm_start_path",
        type=str,
        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    encoder_config = default_encoder_config()
    decoder_config = default_decoder_config(args.future_len)
    flow_config = default_flow_config(args.future_len)
    path_config = default_path_config()
    prior_config = default_prior_config()
    integrated_config = default_integrated_config()
    state_config = default_state_config()
    metric_config = default_metric_config()
    slot_config = default_slot_config()
    event_config = default_event_config()

    support_lo = 0.01
    support_hi = 1.0
    support_eps = 1e-5
    base_nu = 8.0
    cov_jitter = 1e-5
    mix_chunk_size = 27

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

    model = AmplitudeGatedMarkedEventResidualModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        slot_config=slot_config,
        event_config=event_config,
        support_lo=support_lo,
        support_hi=support_hi,
        support_eps=support_eps,
        cov_jitter=cov_jitter,
        base_nu=base_nu,
        mix_chunk_size=mix_chunk_size,
    ).to(args.device)
    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, args.device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)
    model.local_metric_budget_logit.requires_grad_(False)
    model.band_metric_budget_logit.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        event_params = (
            list(model.prior_head.parameters())
            + list(model.post_head.parameters())
            + [model.slot_embed]
            + list(model.event_transport.parameters())
            + [model.event_scale_logit]
        )
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            optimizer = torch.optim.AdamW(event_params, lr=args.lr_event_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "amp-gated-event-head"
        else:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            optimizer = torch.optim.AdamW(
                [
                    {"params": event_params, "lr": args.lr_event_stage2},
                    {"params": list(model.path_context_adapter.parameters()), "lr": args.lr_adapter_stage2},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "amp-gated-event-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n188b_v0: amplitude-gated marked-event residual model\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start: {args.warm_start_path}")

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
        "marked_total_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_active_bce",
        "prior_active_bce",
        "post_time_ce",
        "prior_time_ce",
        "post_node_ce",
        "prior_node_ce",
        "amp_loss",
        "radius_loss",
        "duration_loss",
        "active_kl",
        "time_kl",
        "node_kl",
        "slot_overlap",
        "quiet_dominance",
        "activity_excess",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "teacher_slot_active_rate",
        "teacher_slot_amp_abs_mean",
        "teacher_slot_radius_mean",
        "teacher_slot_duration_mean",
        "post_slot_active_mean",
        "prior_slot_active_mean",
        "post_slot_amp_abs_mean",
        "prior_slot_amp_abs_mean",
        "post_slot_time_top1_mean",
        "prior_slot_time_top1_mean",
        "post_slot_node_top1_mean",
        "prior_slot_node_top1_mean",
        "activity_target",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "event_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "quiet_band_high_mean",
        "quiet_band_mid_mean",
        "quiet_band_low_mean",
        "event_band_high_mean",
        "event_band_mid_mean",
        "event_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "event_scale",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "slot_active_mean",
        "slot_time_top1_mean",
        "slot_node_top1_mean",
        "slot_amp_abs_mean",
        "slot_radius_mean",
        "slot_duration_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
        "raw_event_velocity_norm_mean",
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
            loss, metrics = amplitude_gated_marked_event_flow_matching_loss(model, history_01, future_01)
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
        val_metrics = evaluate_teacher_forced(model, val_loader)
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
        current_key = strict_checkpoint_key_188b(val_metrics, frontier_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics, **frontier_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "amplitude_gated_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_188b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "event_slots": slot_config,
                        "event": event_config,
                        "support_lo": support_lo,
                        "support_hi": support_hi,
                        "support_eps": support_eps,
                        "base_nu": base_nu,
                        "history_len": args.history_len,
                        "future_len": args.future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": mix_chunk_size,
                        "frozen_backbone": True,
                    },
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
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"worstLate={frontier_metrics['frontier_turb_late_worst_cov']:.3f}  "
            f"bestLate={frontier_metrics['frontier_turb_late_best_cov']:.3f}  "
            f"kurt={frontier_metrics['frontier_pooled_kurt_ratio']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"pSlot={val_metrics['val_prior_slot_active_mean']:.3f}  "
            f"pAmp={val_metrics['val_prior_slot_amp_abs_mean']:.3f}  "
            f"pNode={val_metrics['val_prior_slot_node_top1_mean']:.3f}  "
            f"qAmp={val_metrics['val_post_slot_amp_abs_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "amplitude_gated_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_188b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "event_slots": slot_config,
            "event": event_config,
            "support_lo": support_lo,
            "support_hi": support_hi,
            "support_eps": support_eps,
            "base_nu": base_nu,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": mix_chunk_size,
            "frozen_backbone": True,
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))


if __name__ == "__main__":
    main()
