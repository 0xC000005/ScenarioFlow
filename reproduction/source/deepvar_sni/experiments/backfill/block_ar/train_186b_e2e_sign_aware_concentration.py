#!/usr/bin/env python
"""
186b_v0: Controlled 183c end-to-end retrain with sign-aware concentration objective.

Principle:
  - keep the 183c model class fixed
  - unfreeze the encoder and path-context stack to test whether the remaining
    concentration frontier is a frozen-training bottleneck
  - replace magnitude-driven weighting with a sign-aware objective:
      * upweight positive local concentration targets (underwide hard slices)
      * explicitly penalize widening on negative local targets (already overwide cells)
      * keep a mild spectrum loss for quiet / shoulder / extreme tail shape
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
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


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


def robust_window_boost(window_score: torch.Tensor, focus_weight: float, clip: float) -> torch.Tensor:
    med = window_score.median()
    mad = (window_score - med).abs().median().clamp_min(1e-4)
    z = ((window_score - med) / mad).clamp(min=0.0, max=clip)
    return 1.0 + focus_weight * z


def build_sign_aware_weights(
    target_local_log: torch.Tensor,
    target_band_log: torch.Tensor,
    objective_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_frames, _n_cells = target_local_log.shape
    device = target_local_log.device
    dtype = target_local_log.dtype

    horizon_ramp = torch.linspace(
        1.0,
        objective_config["late_horizon_max_weight"],
        steps=n_frames,
        device=device,
        dtype=dtype,
    ).view(1, n_frames, 1)

    pos_local = F.relu(target_local_log)
    neg_local = F.relu(-target_local_log)

    pos_local_norm = pos_local / pos_local.mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    neg_local_norm = neg_local / neg_local.mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)

    pos_boost = (pos_local_norm - 1.0).clamp(min=0.0, max=objective_config["local_boost_clip"])
    neg_boost = (neg_local_norm - 1.0).clamp(min=0.0, max=objective_config["local_boost_clip"])

    late_span = min(max(n_frames // 3, 1), n_frames)
    late_pos = pos_local[:, -late_span:].mean(dim=(1, 2), keepdim=True)
    high_band_pos = F.relu(target_band_log[:, 2]).view(batch, 1, 1)
    window_score = late_pos + objective_config["window_high_band_mix"] * high_band_pos
    window_boost = robust_window_boost(
        window_score,
        focus_weight=objective_config["window_focus_weight"],
        clip=objective_config["window_focus_clip"],
    )

    pos_weight = 1.0 + window_boost * objective_config["positive_focus_weight"] * horizon_ramp * pos_boost
    neg_weight = 1.0 + window_boost * objective_config["negative_focus_weight"] * horizon_ramp * neg_boost
    pos_weight = pos_weight.clamp(max=objective_config["local_weight_max"])
    neg_weight = neg_weight.clamp(max=objective_config["local_weight_max"])

    pos_band = F.relu(target_band_log)
    neg_band = F.relu(-target_band_log)
    pos_band_norm = pos_band / pos_band.mean(dim=1, keepdim=True).clamp_min(1e-4)
    neg_band_norm = neg_band / neg_band.mean(dim=1, keepdim=True).clamp_min(1e-4)
    pos_band_boost = (pos_band_norm - 1.0).clamp(min=0.0, max=objective_config["band_boost_clip"])
    neg_band_boost = (neg_band_norm - 1.0).clamp(min=0.0, max=objective_config["band_boost_clip"])
    pos_band_weight = 1.0 + window_boost.squeeze(-1) * objective_config["positive_band_focus_weight"] * pos_band_boost
    neg_band_weight = 1.0 + window_boost.squeeze(-1) * objective_config["negative_band_focus_weight"] * neg_band_boost
    pos_band_weight = pos_band_weight.clamp(max=objective_config["band_weight_max"])
    neg_band_weight = neg_band_weight.clamp(max=objective_config["band_weight_max"])

    return pos_weight, neg_weight, pos_band_weight, neg_band_weight, window_boost.squeeze(-1).squeeze(-1)


def sign_aware_local_band_loss(
    metric_local: torch.Tensor,
    target_local_log: torch.Tensor,
    metric_band: torch.Tensor,
    target_band_log: torch.Tensor,
    pos_weight: torch.Tensor,
    neg_weight: torch.Tensor,
    pos_band_weight: torch.Tensor,
    neg_band_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_mask_local = (target_local_log > 0).to(metric_local.dtype)
    neg_mask_local = (target_local_log < 0).to(metric_local.dtype)
    pos_mask_band = (target_band_log > 0).to(metric_band.dtype)
    neg_mask_band = (target_band_log < 0).to(metric_band.dtype)

    local_pos_loss = weighted_smooth_l1(
        metric_local,
        target_local_log,
        (1e-3 + pos_weight) * pos_mask_local + 1e-3,
    )
    local_neg_loss = weighted_smooth_l1(
        metric_local,
        target_local_log,
        (1e-3 + neg_weight) * neg_mask_local + 1e-3,
    )
    band_pos_loss = weighted_smooth_l1(
        metric_band,
        target_band_log,
        (1e-3 + pos_band_weight) * pos_mask_band + 1e-3,
    )
    band_neg_loss = weighted_smooth_l1(
        metric_band,
        target_band_log,
        (1e-3 + neg_band_weight) * neg_mask_band + 1e-3,
    )

    underfit_gap = (
        (F.relu(target_local_log - metric_local) * pos_weight * pos_mask_local).sum()
        / (pos_weight * pos_mask_local).sum().clamp_min(1e-6)
    )
    overwide_gap = (
        (F.relu(metric_local - target_local_log) * neg_weight * neg_mask_local).sum()
        / (neg_weight * neg_mask_local).sum().clamp_min(1e-6)
    )

    return local_pos_loss, local_neg_loss, band_pos_loss, band_neg_loss, underfit_gap, overwide_gap


def sign_aware_e2e_loss(
    model: StateMetricTransportModel,
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

    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    raw_local, raw_band, metric_local, metric_band, _local_gate, _band_gate = model.build_state_metric_controls(
        z_t, t, path_context
    )

    pos_weight, neg_weight, pos_band_weight, neg_band_weight, window_boost = build_sign_aware_weights(
        target_local_log=target_local_log,
        target_band_log=target_band_log,
        objective_config=objective_config,
    )
    local_pos_loss, local_neg_loss, band_pos_loss, band_neg_loss, underfit_gap, overwide_gap = sign_aware_local_band_loss(
        metric_local=metric_local,
        target_local_log=target_local_log,
        metric_band=metric_band,
        target_band_log=target_band_log,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        pos_band_weight=pos_band_weight,
        neg_band_weight=neg_band_weight,
    )

    fm_loss = F.mse_loss(pred_v, target_v)
    surf = metric_local.view(metric_local.shape[0], metric_local.shape[1], 5, 5)
    time_smooth = (metric_local[:, 1:] - metric_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth
    budget_reg = (
        (model.local_metric_budget() - model.metric_config["init_local_metric_budget"]).pow(2)
        + (model.band_metric_budget() - model.metric_config["init_band_metric_budget"]).pow(2)
    )

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
        objective_config["quiet_weight"] * (pred_quiet - target_quiet).abs()
        + objective_config["shoulder_weight"] * (pred_shoulder - target_shoulder).abs()
        + objective_config["extreme_weight"] * (pred_extreme - target_extreme).abs()
    ).mean()

    total = (
        fm_loss
        + objective_config["local_pos_loss_weight"] * local_pos_loss
        + objective_config["local_neg_loss_weight"] * local_neg_loss
        + objective_config["band_pos_loss_weight"] * band_pos_loss
        + objective_config["band_neg_loss_weight"] * band_neg_loss
        + objective_config["underfit_gap_weight"] * underfit_gap
        + objective_config["overwide_gap_weight"] * overwide_gap
        + objective_config["spectrum_loss_weight"] * spectrum_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
        + model.metric_config["budget_reg_weight"] * budget_reg
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
        "sign_total_loss": total,
        "local_pos_loss": local_pos_loss,
        "local_neg_loss": local_neg_loss,
        "band_pos_loss": band_pos_loss,
        "band_neg_loss": band_neg_loss,
        "underfit_gap": underfit_gap,
        "overwide_gap": overwide_gap,
        "spectrum_loss": spectrum_loss,
        "smooth_reg": smooth_reg,
        "budget_reg": budget_reg,
        "window_boost_mean": window_boost.mean(),
        "target_local_pos_mean": F.relu(target_local_log).mean(),
        "target_local_neg_mean": F.relu(-target_local_log).mean(),
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
    model: StateMetricTransportModel,
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
        "sign_total_loss",
        "local_pos_loss",
        "local_neg_loss",
        "band_pos_loss",
        "band_neg_loss",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "smooth_reg",
        "budget_reg",
        "window_boost_mean",
        "target_local_pos_mean",
        "target_local_neg_mean",
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
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = sign_aware_e2e_loss(model, history_01, future_01, objective_config)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_sign_total_loss"]
    return out


def instantiate_from_warm_start(warm_start_path: str, device: str) -> tuple[StateMetricTransportModel, dict]:
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
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="186b_v0: controlled 183c end-to-end sign-aware concentration retrain")
    parser.add_argument("--epochs_stage1", type=int, default=3)
    parser.add_argument("--epochs_stage2", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_ctrl_stage1", type=float, default=4e-4)
    parser.add_argument("--lr_ctrl_stage2", type=float, default=2e-4)
    parser.add_argument("--lr_path_stage2", type=float, default=1e-4)
    parser.add_argument("--lr_encoder_stage1", type=float, default=5e-5)
    parser.add_argument("--lr_encoder_stage2", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--late_horizon_max_weight", type=float, default=2.0)
    parser.add_argument("--positive_focus_weight", type=float, default=0.90)
    parser.add_argument("--negative_focus_weight", type=float, default=0.55)
    parser.add_argument("--positive_band_focus_weight", type=float, default=0.30)
    parser.add_argument("--negative_band_focus_weight", type=float, default=0.18)
    parser.add_argument("--window_focus_weight", type=float, default=0.80)
    parser.add_argument("--window_high_band_mix", type=float, default=0.50)
    parser.add_argument("--window_focus_clip", type=float, default=3.0)
    parser.add_argument("--local_boost_clip", type=float, default=4.0)
    parser.add_argument("--band_boost_clip", type=float, default=2.5)
    parser.add_argument("--local_weight_max", type=float, default=6.0)
    parser.add_argument("--band_weight_max", type=float, default=3.0)
    parser.add_argument("--local_pos_loss_weight", type=float, default=0.75)
    parser.add_argument("--local_neg_loss_weight", type=float, default=0.45)
    parser.add_argument("--band_pos_loss_weight", type=float, default=0.30)
    parser.add_argument("--band_neg_loss_weight", type=float, default=0.15)
    parser.add_argument("--underfit_gap_weight", type=float, default=0.65)
    parser.add_argument("--overwide_gap_weight", type=float, default=0.60)
    parser.add_argument("--quiet_quantile", type=float, default=0.50)
    parser.add_argument("--extreme_quantile", type=float, default=0.99)
    parser.add_argument("--quiet_weight", type=float, default=1.25)
    parser.add_argument("--shoulder_weight", type=float, default=1.00)
    parser.add_argument("--extreme_weight", type=float, default=2.25)
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

    objective_config = {
        "late_horizon_max_weight": args.late_horizon_max_weight,
        "positive_focus_weight": args.positive_focus_weight,
        "negative_focus_weight": args.negative_focus_weight,
        "positive_band_focus_weight": args.positive_band_focus_weight,
        "negative_band_focus_weight": args.negative_band_focus_weight,
        "window_focus_weight": args.window_focus_weight,
        "window_high_band_mix": args.window_high_band_mix,
        "window_focus_clip": args.window_focus_clip,
        "local_boost_clip": args.local_boost_clip,
        "band_boost_clip": args.band_boost_clip,
        "local_weight_max": args.local_weight_max,
        "band_weight_max": args.band_weight_max,
        "local_pos_loss_weight": args.local_pos_loss_weight,
        "local_neg_loss_weight": args.local_neg_loss_weight,
        "band_pos_loss_weight": args.band_pos_loss_weight,
        "band_neg_loss_weight": args.band_neg_loss_weight,
        "underfit_gap_weight": args.underfit_gap_weight,
        "overwide_gap_weight": args.overwide_gap_weight,
        "quiet_quantile": args.quiet_quantile,
        "extreme_quantile": args.extreme_quantile,
        "quiet_weight": args.quiet_weight,
        "shoulder_weight": args.shoulder_weight,
        "extreme_weight": args.extreme_weight,
        "spectrum_tau": args.spectrum_tau,
        "spectrum_loss_weight": args.spectrum_loss_weight,
    }

    model, warm_cfg = instantiate_from_warm_start(args.warm_start_path, args.device)
    print(f"  Warm start loaded from {args.warm_start_path}")

    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        model.encoder.requires_grad_(True)
        model.width_allocator.requires_grad_(True)
        model.band_tail.requires_grad_(True)
        model.local_state_gate.requires_grad_(True)
        model.band_state_gate.requires_grad_(True)
        model.local_metric_budget_logit.requires_grad_(True)
        model.band_metric_budget_logit.requires_grad_(True)
        model.path_context_adapter.requires_grad_(True)
        if stage == 1:
            model.path_transport.requires_grad_(False)
            params = [
                {
                    "params": list(model.encoder.parameters()),
                    "lr": args.lr_encoder_stage1,
                },
                {
                    "params": (
                        list(model.width_allocator.parameters())
                        + list(model.band_tail.parameters())
                        + list(model.local_state_gate.parameters())
                        + list(model.band_state_gate.parameters())
                        + [model.local_metric_budget_logit, model.band_metric_budget_logit]
                        + list(model.path_context_adapter.parameters())
                    ),
                    "lr": args.lr_ctrl_stage1,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "sign-aware-e2e-ctrl"
        else:
            model.path_transport.requires_grad_(True)
            params = [
                {
                    "params": list(model.encoder.parameters()),
                    "lr": args.lr_encoder_stage2,
                },
                {
                    "params": (
                        list(model.width_allocator.parameters())
                        + list(model.band_tail.parameters())
                        + list(model.local_state_gate.parameters())
                        + list(model.band_state_gate.parameters())
                        + [model.local_metric_budget_logit, model.band_metric_budget_logit]
                        + list(model.path_context_adapter.parameters())
                    ),
                    "lr": args.lr_ctrl_stage2,
                },
                {
                    "params": list(model.path_transport.parameters()),
                    "lr": args.lr_path_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "sign-aware-e2e-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n186b_v0: controlled 183c end-to-end sign-aware concentration retrain\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start: {args.warm_start_path}")
    print(
        f"  Pos/neg focus: {args.positive_focus_weight:.2f}/{args.negative_focus_weight:.2f} | "
        f"underfit/overwide: {args.underfit_gap_weight:.2f}/{args.overwide_gap_weight:.2f}"
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
        "sign_total_loss",
        "local_pos_loss",
        "local_neg_loss",
        "band_pos_loss",
        "band_neg_loss",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "smooth_reg",
        "budget_reg",
        "window_boost_mean",
        "target_local_pos_mean",
        "target_local_neg_mean",
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
            loss, metrics = sign_aware_e2e_loss(model, history_01, future_01, objective_config)
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
        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config)
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
                    "type": "state_metric_transport_e2e_sign_aware_concentration_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_186b",
                    "objective": objective_config,
                    "history_len": args.history_len,
                    "future_len": args.future_len,
                    "train_windows": len(train_indices),
                    "val_windows": len(val_indices),
                    "frozen_backbone": False,
                    "encoder_unfrozen": True,
                    "path_context_unfrozen": True,
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
            f"wBoost={val_metrics['val_window_boost_mean']:.3f}  "
            f"encLR={optimizer.param_groups[0]['lr']:.1e}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    cfg_out = dict(warm_cfg)
    cfg_out.update(
        {
            "type": "state_metric_transport_e2e_sign_aware_concentration_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_186b",
            "objective": objective_config,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "frozen_backbone": False,
            "encoder_unfrozen": True,
            "path_context_unfrozen": True,
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
