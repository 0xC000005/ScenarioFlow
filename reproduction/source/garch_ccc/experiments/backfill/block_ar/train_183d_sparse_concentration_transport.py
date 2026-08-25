#!/usr/bin/env python
"""
183d_v0: Sparse concentration transport on top of the 183c pathwise residual-law backbone.

Principle:
  - keep the 183c mean/covariance/pathwise residual backbone
  - keep state-conditioned local and band metric control
  - add a quiet-vs-event decomposition over the local transport metric
  - train a sparse event allocator directly against teacher support on the
    horizon x cell grid, rather than only increasing diffuse metric strength
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
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
    TimeConditionedGate,
    _scaled_logit,
)


class ScaledSigmoidHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        low: float,
        high: float,
        init_value: float,
    ):
        super().__init__()
        self.low = low
        self.high = high
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _scaled_logit(init_value, low, high))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.net(x))
        return self.low + (self.high - self.low) * raw


def topk_soft_allocation(scores: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    flat = scores.reshape(scores.shape[0], -1)
    n = flat.shape[1]
    k = max(1, min(k, n))
    topv, topi = flat.topk(k, dim=1)
    masked = flat.new_full(flat.shape, -1e9)
    masked.scatter_(1, topi, topv / max(temperature, 1e-6))
    alloc = torch.softmax(masked, dim=1)
    active = (flat.sum(dim=1, keepdim=True) > 1e-8).to(flat.dtype)
    alloc = alloc * active
    alloc = alloc.view_as(scores)
    return alloc


class SparseConcentrationTransportModel(StateMetricTransportModel):
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
        concentration_config: dict,
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
        self.concentration_config = dict(concentration_config)
        event_in_dim = path_config["context_dim"] + 1 + 7
        self.event_gate = TimeConditionedGate(
            input_dim=event_in_dim,
            output_dim=1,
            hidden_dim=concentration_config["event_hidden_dim"],
            init_prob=concentration_config["init_event_gate"],
        )
        self.event_budget = ScaledSigmoidHead(
            input_dim=event_in_dim,
            hidden_dim=concentration_config["event_hidden_dim"],
            low=concentration_config["event_budget_min"],
            high=concentration_config["event_budget_max"],
            init_value=concentration_config["init_event_budget"],
        )

    def build_event_summary(
        self,
        state_local: torch.Tensor,
        state_band: torch.Tensor,
    ) -> torch.Tensor:
        pos_mean = F.relu(state_local).mean(dim=(1, 2))
        neg_mean = F.relu(-state_local).mean(dim=(1, 2))
        abs_mean = state_local.abs().mean(dim=(1, 2))
        abs_max = state_local.abs().amax(dim=(1, 2))
        return torch.cat(
            [
                pos_mean.unsqueeze(-1),
                neg_mean.unsqueeze(-1),
                abs_mean.unsqueeze(-1),
                abs_max.unsqueeze(-1),
                state_band,
            ],
            dim=-1,
        )

    def build_sparse_metric_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            raw_local,
            raw_band,
            base_metric_local,
            metric_band,
            local_gate,
            band_gate,
        ) = super().build_state_metric_controls(z_t_flat, t, path_context)
        state_local, state_band = self.build_state_features(z_t_flat)
        event_summary = self.build_event_summary(state_local, state_band)
        event_in = torch.cat([path_context, t.unsqueeze(-1), event_summary], dim=-1)
        event_gate = self.event_gate(event_in)
        event_budget = self.event_budget(event_in)

        pos_scores = F.relu(base_metric_local)
        neg_scores = F.relu(-base_metric_local)
        alloc_pos = topk_soft_allocation(
            pos_scores,
            self.concentration_config["event_topk"],
            self.concentration_config["event_temperature"],
        )
        alloc_neg = topk_soft_allocation(
            neg_scores,
            self.concentration_config["event_topk"],
            self.concentration_config["event_temperature"],
        )
        event_component = (alloc_pos - alloc_neg) * (event_gate * event_budget).unsqueeze(-1)
        background_scale = (
            self.concentration_config["background_floor"]
            + (1.0 - self.concentration_config["background_floor"]) * (1.0 - event_gate)
        ).unsqueeze(-1)
        metric_local = background_scale * base_metric_local + event_component
        metric_local = metric_local - metric_local.mean(dim=(1, 2), keepdim=True)
        metric_local = metric_local.clamp(
            min=-self.metric_config["local_metric_clip"],
            max=self.metric_config["local_metric_clip"],
        )
        return (
            raw_local,
            raw_band,
            metric_local,
            metric_band,
            local_gate,
            band_gate,
            event_gate,
            event_budget,
            alloc_pos,
            alloc_neg,
        )

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            raw_local,
            raw_band,
            metric_local,
            metric_band,
            local_gate,
            band_gate,
            event_gate,
            event_budget,
            alloc_pos,
            alloc_neg,
        ) = self.build_sparse_metric_controls(z_t_flat, t, path_context)
        raw_v = self.path_transport(z_t_flat, t, path_context)
        raw_v_basis = raw_v.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        mod_v_basis = self.modulate_velocity_basis(raw_v_basis, metric_local, metric_band)
        flat_pos = alloc_pos.reshape(alloc_pos.shape[0], -1)
        flat_neg = alloc_neg.reshape(alloc_neg.shape[0], -1)
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
            "event_gate_mean": event_gate.mean(),
            "event_budget_mean": event_budget.mean(),
            "event_pos_top1_mean": flat_pos.max(dim=1).values.mean(),
            "event_neg_top1_mean": flat_neg.max(dim=1).values.mean(),
            "pred_velocity_norm_mean": mod_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return mod_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def sparse_concentration_flow_matching_loss(
    model: SparseConcentrationTransportModel,
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
    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    fm_loss = F.mse_loss(pred_v, target_v)

    (
        raw_local,
        raw_band,
        metric_local,
        metric_band,
        _local_gate,
        _band_gate,
        event_gate,
        event_budget,
        alloc_pos,
        alloc_neg,
    ) = model.build_sparse_metric_controls(z_t, t, path_context)
    local_loss = F.smooth_l1_loss(metric_local, target_local_log)
    band_loss = F.smooth_l1_loss(metric_band, target_band_log)

    surf = metric_local.view(metric_local.shape[0], metric_local.shape[1], 5, 5)
    time_smooth = (metric_local[:, 1:] - metric_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth

    target_pos = topk_soft_allocation(
        F.relu(target_local_log),
        model.concentration_config["target_topk"],
        model.concentration_config["target_temperature"],
    )
    target_neg = topk_soft_allocation(
        F.relu(-target_local_log),
        model.concentration_config["target_topk"],
        model.concentration_config["target_temperature"],
    )
    support_loss = 0.5 * (
        F.smooth_l1_loss(alloc_pos, target_pos) + F.smooth_l1_loss(alloc_neg, target_neg)
    )

    budget_reg = (
        (model.local_metric_budget() - model.metric_config["init_local_metric_budget"]).pow(2)
        + (model.band_metric_budget() - model.metric_config["init_band_metric_budget"]).pow(2)
    )
    target_event_strength = (F.relu(target_local_log).amax(dim=(1, 2)) + F.relu(-target_local_log).amax(dim=(1, 2))).clamp(
        min=0.0,
        max=model.concentration_config["event_budget_max"],
    ) / max(model.concentration_config["event_budget_max"], 1e-6)
    event_gate_loss = F.smooth_l1_loss(event_gate.squeeze(-1), target_event_strength)

    total = (
        fm_loss
        + model.integrated_config["local_loss_weight"] * local_loss
        + model.integrated_config["band_loss_weight"] * band_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
        + model.metric_config["budget_reg_weight"] * budget_reg
        + model.concentration_config["support_loss_weight"] * support_loss
        + model.concentration_config["event_gate_loss_weight"] * event_gate_loss
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
        "sct_total_loss": total,
        "sct_local_loss": local_loss,
        "sct_band_loss": band_loss,
        "sct_smooth_reg": smooth_reg,
        "sct_budget_reg": budget_reg,
        "sct_support_loss": support_loss,
        "sct_event_gate_loss": event_gate_loss,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: SparseConcentrationTransportModel,
    val_loader: DataLoader,
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
        "sct_total_loss",
        "sct_local_loss",
        "sct_band_loss",
        "sct_smooth_reg",
        "sct_budget_reg",
        "sct_support_loss",
        "sct_event_gate_loss",
        "target_local_abs_mean",
        "target_band_high_mean",
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
        "event_gate_mean",
        "event_budget_mean",
        "event_pos_top1_mean",
        "event_neg_top1_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = sparse_concentration_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_sct_total_loss"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="183d_v0: sparse concentration transport")
    parser.add_argument("--epochs_stage1", type=int, default=4)
    parser.add_argument("--epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_ctrl_stage1", type=float, default=6e-4)
    parser.add_argument("--lr_ctrl_stage2", type=float, default=2.5e-4)
    parser.add_argument("--lr_path_stage2", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_low_scale_clip", type=float, default=1.2)
    parser.add_argument("--flow_mid_scale_clip", type=float, default=0.7)
    parser.add_argument("--flow_high_scale_clip", type=float, default=0.35)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--mix_chunk_size", type=int, default=27)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--path_context_dim", type=int, default=256)
    parser.add_argument("--path_context_hidden_dim", type=int, default=256)
    parser.add_argument("--path_d_model", type=int, default=192)
    parser.add_argument("--path_heads", type=int, default=4)
    parser.add_argument("--path_layers", type=int, default=4)
    parser.add_argument("--path_ff_mult", type=int, default=4)
    parser.add_argument("--path_time_embed_dim", type=int, default=64)
    parser.add_argument("--path_ode_steps", type=int, default=8)
    parser.add_argument("--prior_low_std", type=float, default=1.00)
    parser.add_argument("--prior_mid_std", type=float, default=0.75)
    parser.add_argument("--prior_high_std", type=float, default=0.35)
    parser.add_argument("--prior_low_jump_prob", type=float, default=0.005)
    parser.add_argument("--prior_mid_jump_prob", type=float, default=0.025)
    parser.add_argument("--prior_high_jump_prob", type=float, default=0.070)
    parser.add_argument("--prior_low_jump_scale", type=float, default=0.10)
    parser.add_argument("--prior_mid_jump_scale", type=float, default=0.30)
    parser.add_argument("--prior_high_jump_scale", type=float, default=0.75)
    parser.add_argument("--width_rank", type=int, default=4)
    parser.add_argument("--width_hidden_dim", type=int, default=256)
    parser.add_argument("--width_clip", type=float, default=0.80)
    parser.add_argument("--band_hidden_dim", type=int, default=128)
    parser.add_argument("--band_clip", type=float, default=0.45)
    parser.add_argument("--local_loss_weight", type=float, default=0.75)
    parser.add_argument("--band_loss_weight", type=float, default=0.50)
    parser.add_argument("--smooth_reg_weight", type=float, default=0.04)
    parser.add_argument("--state_gate_hidden_dim", type=int, default=128)
    parser.add_argument("--init_local_state_gate", type=float, default=0.30)
    parser.add_argument("--init_band_state_gate", type=float, default=0.25)
    parser.add_argument("--local_metric_min", type=float, default=0.12)
    parser.add_argument("--local_metric_max", type=float, default=0.55)
    parser.add_argument("--band_metric_min", type=float, default=0.08)
    parser.add_argument("--band_metric_max", type=float, default=0.35)
    parser.add_argument("--init_local_metric_budget", type=float, default=0.24)
    parser.add_argument("--init_band_metric_budget", type=float, default=0.14)
    parser.add_argument("--local_metric_clip", type=float, default=0.60)
    parser.add_argument("--band_metric_clip", type=float, default=0.35)
    parser.add_argument("--budget_reg_weight", type=float, default=0.02)
    parser.add_argument("--event_hidden_dim", type=int, default=128)
    parser.add_argument("--init_event_gate", type=float, default=0.18)
    parser.add_argument("--event_budget_min", type=float, default=0.08)
    parser.add_argument("--event_budget_max", type=float, default=0.45)
    parser.add_argument("--init_event_budget", type=float, default=0.22)
    parser.add_argument("--event_topk", type=int, default=5)
    parser.add_argument("--event_temperature", type=float, default=0.20)
    parser.add_argument("--target_topk", type=int, default=5)
    parser.add_argument("--target_temperature", type=float, default=0.20)
    parser.add_argument("--background_floor", type=float, default=0.35)
    parser.add_argument("--support_loss_weight", type=float, default=0.55)
    parser.add_argument("--event_gate_loss_weight", type=float, default=0.10)
    parser.add_argument("--warm_start_path", type=str, default=None)
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

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=args.future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
        n_blocks=args.n_blocks,
        n_templates=args.n_templates,
        template_diag_clip=args.template_diag_clip,
        template_offdiag_clip=args.template_offdiag_clip,
        drift_strength_max=args.drift_strength_max,
        equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    flow_config = dict(
        dim=args.future_len * 25,
        context_dim=args.flow_context_dim,
        n_frames=args.future_len,
        grid_h=5,
        grid_w=5,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        low_scale_clip=args.flow_low_scale_clip,
        mid_scale_clip=args.flow_mid_scale_clip,
        high_scale_clip=args.flow_high_scale_clip,
    )
    path_config = dict(
        context_dim=args.path_context_dim,
        context_hidden_dim=args.path_context_hidden_dim,
        d_model=args.path_d_model,
        n_heads=args.path_heads,
        n_layers=args.path_layers,
        ff_mult=args.path_ff_mult,
        time_embed_dim=args.path_time_embed_dim,
        n_ode_steps=args.path_ode_steps,
    )
    prior_config = dict(
        low_std=args.prior_low_std,
        mid_std=args.prior_mid_std,
        high_std=args.prior_high_std,
        low_jump_prob=args.prior_low_jump_prob,
        mid_jump_prob=args.prior_mid_jump_prob,
        high_jump_prob=args.prior_high_jump_prob,
        low_jump_scale=args.prior_low_jump_scale,
        mid_jump_scale=args.prior_mid_jump_scale,
        high_jump_scale=args.prior_high_jump_scale,
    )
    integrated_config = dict(
        width_rank=args.width_rank,
        width_hidden_dim=args.width_hidden_dim,
        width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim,
        band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight,
        band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
        init_local_strength=0.05,
        init_band_strength=0.05,
    )
    state_config = dict(
        gate_hidden_dim=args.state_gate_hidden_dim,
        init_local_state_gate=args.init_local_state_gate,
        init_band_state_gate=args.init_band_state_gate,
    )
    metric_config = dict(
        local_metric_min=args.local_metric_min,
        local_metric_max=args.local_metric_max,
        band_metric_min=args.band_metric_min,
        band_metric_max=args.band_metric_max,
        init_local_metric_budget=args.init_local_metric_budget,
        init_band_metric_budget=args.init_band_metric_budget,
        local_metric_clip=args.local_metric_clip,
        band_metric_clip=args.band_metric_clip,
        budget_reg_weight=args.budget_reg_weight,
    )
    concentration_config = dict(
        event_hidden_dim=args.event_hidden_dim,
        init_event_gate=args.init_event_gate,
        event_budget_min=args.event_budget_min,
        event_budget_max=args.event_budget_max,
        init_event_budget=args.init_event_budget,
        event_topk=args.event_topk,
        event_temperature=args.event_temperature,
        target_topk=args.target_topk,
        target_temperature=args.target_temperature,
        background_floor=args.background_floor,
        support_loss_weight=args.support_loss_weight,
        event_gate_loss_weight=args.event_gate_loss_weight,
    )

    model = SparseConcentrationTransportModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        concentration_config=concentration_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)
    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, args.device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        base_params = (
            list(model.width_allocator.parameters())
            + list(model.band_tail.parameters())
            + list(model.local_state_gate.parameters())
            + list(model.band_state_gate.parameters())
            + list(model.event_gate.parameters())
            + list(model.event_budget.parameters())
            + [model.local_metric_budget_logit, model.band_metric_budget_logit]
        )
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            model.local_state_gate.requires_grad_(True)
            model.band_state_gate.requires_grad_(True)
            model.event_gate.requires_grad_(True)
            model.event_budget.requires_grad_(True)
            optimizer = torch.optim.AdamW(base_params, lr=args.lr_ctrl_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "sparse-concentration-ctrl"
        else:
            model.path_transport.requires_grad_(True)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            model.local_state_gate.requires_grad_(True)
            model.band_state_gate.requires_grad_(True)
            model.event_gate.requires_grad_(True)
            model.event_budget.requires_grad_(True)
            params = [
                {"params": base_params, "lr": args.lr_ctrl_stage2},
                {
                    "params": list(model.path_transport.parameters()) + list(model.path_context_adapter.parameters()),
                    "lr": args.lr_path_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "sparse-concentration-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n183d_v0: sparse concentration transport\n{'=' * 72}")
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
        "sct_total_loss",
        "sct_local_loss",
        "sct_band_loss",
        "sct_smooth_reg",
        "sct_budget_reg",
        "sct_support_loss",
        "sct_event_gate_loss",
        "target_local_abs_mean",
        "target_band_high_mean",
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
        "event_gate_mean",
        "event_budget_mean",
        "event_pos_top1_mean",
        "event_neg_top1_mean",
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
            loss, metrics = sparse_concentration_flow_matching_loss(model, history_01, future_01)
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
        current_key = checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
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
                        "type": "sparse_concentration_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183d",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "concentration": concentration_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": args.history_len,
                        "future_len": args.future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": args.mix_chunk_size,
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
            f"val_total={val_metrics['val_total_loss']:.4f}  worstLate={frontier_metrics['frontier_turb_late_worst_cov']:.3f}  "
            f"bestLate={frontier_metrics['frontier_turb_late_best_cov']:.3f}  "
            f"kurt={frontier_metrics['frontier_pooled_kurt_ratio']:.3f}  "
            f"highE={frontier_metrics['high_energy_ratio_p50']:.3f}  midE={frontier_metrics['mid_energy_ratio_p50']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"lBud={val_metrics['val_local_metric_budget']:.3f}  bBud={val_metrics['val_band_metric_budget']:.3f}  "
            f"eGate={val_metrics['val_event_gate_mean']:.3f}  eBud={val_metrics['val_event_budget_mean']:.3f}  "
            f"eTop={val_metrics['val_event_pos_top1_mean']:.3f}/{val_metrics['val_event_neg_top1_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "sparse_concentration_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183d",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "concentration": concentration_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": args.mix_chunk_size,
            "frozen_backbone": True,
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))


if __name__ == "__main__":
    main()
