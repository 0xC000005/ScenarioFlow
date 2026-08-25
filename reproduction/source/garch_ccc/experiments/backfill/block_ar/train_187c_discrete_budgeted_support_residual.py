#!/usr/bin/env python
"""
187c_v0: Discrete budgeted-support residual model on top of the 183c backbone.

Principle:
  - keep the explicit mean-reverting mean branch and structured covariance branch
  - keep the validated 183c quiet residual backbone for ordinary residual behavior
  - keep the 187b underfit-based support target
  - replace soft diffuse support with exact blockwise top-k support
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import evaluate_frontier_subset
from experiments.backfill.block_ar.train_187b_underfit_support_residual import (
    SparseSupportResidualModel,
    evaluate_teacher_forced,
    sparse_support_flow_matching_loss,
    strict_checkpoint_key,
)


class DiscreteBudgetedSupportResidualModel(SparseSupportResidualModel):
    def _teacher_block_budget(self, teacher_abs: torch.Tensor) -> torch.Tensor:
        batch = teacher_abs.shape[0]
        budgets = []
        min_budget = int(self.support_config.get("support_budget_min", 0))
        max_budget = int(self.support_config.get("support_budget_max", 3))
        eps = float(self.support_config.get("support_budget_eps", 1e-8))
        for block_idx in range(self.n_support_blocks):
            frame_mask = self.support_block_ids == block_idx
            block = teacher_abs[:, frame_mask].reshape(batch, -1)
            count = (block > eps).sum(dim=1)
            budgets.append(count.clamp(min=min_budget, max=min(max_budget, block.shape[1])).long())
        return torch.stack(budgets, dim=1)

    def _prior_block_budget(self, support_probs: torch.Tensor) -> torch.Tensor:
        batch = support_probs.shape[0]
        budgets = []
        min_budget = int(self.support_config.get("support_budget_min", 0))
        max_budget = int(self.support_config.get("support_budget_max", 3))
        round_bias = float(self.support_config.get("support_budget_round_bias", 0.0))
        for block_idx in range(self.n_support_blocks):
            frame_mask = self.support_block_ids == block_idx
            block = support_probs[:, frame_mask].reshape(batch, -1)
            budget = torch.floor(block.sum(dim=1) + round_bias + 0.5)
            budgets.append(budget.clamp(min=min_budget, max=min(max_budget, block.shape[1])).long())
        return torch.stack(budgets, dim=1)

    def _budgeted_support_from_logits(
        self,
        support_logits: torch.Tensor,
        support_probs: torch.Tensor,
        budgets: torch.Tensor,
        straight_through: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = support_logits.shape[0]
        support_hard = torch.zeros_like(support_probs)
        for block_idx in range(self.n_support_blocks):
            frame_mask = self.support_block_ids == block_idx
            block_logits = support_logits[:, frame_mask].reshape(batch, -1)
            flat_hard = torch.zeros_like(block_logits)
            block_budget = budgets[:, block_idx].clamp(min=0, max=block_logits.shape[1])
            kmax = int(block_budget.max().item())
            if kmax > 0:
                top_idx = block_logits.topk(kmax, dim=1).indices
                active_mask = (
                    torch.arange(kmax, device=block_logits.device)[None, :] < block_budget[:, None]
                ).to(block_logits.dtype)
                flat_hard.scatter_(1, top_idx, active_mask)
            support_hard[:, frame_mask] = flat_hard.view(batch, int(frame_mask.sum().item()), self.decoder.n_cells)
        if straight_through:
            support_sample = support_hard + support_probs - support_probs.detach()
        else:
            support_sample = support_hard
        return support_sample, support_hard

    def infer_support_process(
        self,
        path_context: torch.Tensor,
        state_local: torch.Tensor,
        quiet_metric_local: torch.Tensor,
        t: torch.Tensor,
        teacher_abs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = super().infer_support_process(
            path_context=path_context,
            state_local=state_local,
            quiet_metric_local=quiet_metric_local,
            t=t,
            teacher_abs=teacher_abs,
        )
        if teacher_abs is not None:
            out["teacher_budget"] = self._teacher_block_budget(teacher_abs)
        return out

    def build_sparse_support_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        support_pack: dict[str, torch.Tensor] | None = None,
        use_posterior: bool = False,
    ):
        raw_local, raw_band, quiet_metric_local, quiet_metric_band, local_gate, band_gate = self.build_quiet_controls(
            z_t_flat,
            t,
            path_context,
        )
        state_local, state_band = self.build_state_features(z_t_flat)
        if support_pack is None:
            support_pack = self.infer_support_process(path_context, state_local, quiet_metric_local, t)
        logits_key = "post_logits" if use_posterior and "post_logits" in support_pack else "prior_logits"
        probs_key = "post_probs" if use_posterior and "post_probs" in support_pack else "prior_probs"
        support_logits = support_pack[logits_key]
        support_probs = support_pack[probs_key]

        if "teacher_budget" in support_pack:
            budgets = support_pack["teacher_budget"]
        else:
            budgets = self._prior_block_budget(support_probs)
        support_sample, support_hard = self._budgeted_support_from_logits(
            support_logits=support_logits,
            support_probs=support_probs,
            budgets=budgets,
            straight_through=self.training,
        )

        node_features = self.build_node_features(state_local, quiet_metric_local, t, path_context)
        event_amp = self.event_amplitude_head(node_features).squeeze(-1)
        event_local = (support_sample * event_amp).clamp(
            min=-self.event_config["event_local_clip"],
            max=self.event_config["event_local_clip"],
        )

        band_in = torch.cat([path_context, state_band, t.unsqueeze(-1)], dim=-1)
        event_band = self.event_band_head(band_in)
        support_rate = support_hard.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        event_band = event_band * support_rate
        band_weights = (self.band_masks.reshape(3, -1).sum(dim=1) / self.band_masks.numel()).to(
            event_band.device,
            dtype=event_band.dtype,
        )
        event_band = event_band - (event_band * band_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        event_band = event_band.clamp(
            min=-self.event_config["event_band_clip"],
            max=self.event_config["event_band_clip"],
        )

        return (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            support_logits,
            support_probs,
            support_hard,
            event_amp,
        )

    def transport_velocity_with_pack(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        support_pack: dict[str, torch.Tensor] | None = None,
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
            support_logits,
            support_probs,
            support_hard,
            event_amp,
        ) = self.build_sparse_support_controls(
            z_t_flat,
            t,
            path_context,
            support_pack=support_pack,
            use_posterior=use_posterior,
        )
        quiet_raw = self.path_transport(z_t_flat, t, path_context)
        quiet_raw_basis = quiet_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        quiet_v_basis = self.modulate_velocity_basis(quiet_raw_basis, quiet_metric_local, quiet_metric_band)

        event_raw = self.event_transport(z_t_flat, t, path_context)
        event_raw_basis = event_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        event_v_basis = self.modulate_velocity_basis(event_raw_basis, event_local, event_band)
        total_v_basis = quiet_v_basis + self.event_scale() * event_v_basis
        flat_support = support_hard.reshape(support_hard.shape[0], -1)
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
            "prior_support_rate": support_hard.mean(),
            "prior_support_top1_mean": flat_support.max(dim=1).values.mean(),
            "support_logit_abs_mean": support_logits.abs().mean(),
            "event_amp_abs_mean": event_amp.abs().mean(),
            "quiet_velocity_norm_mean": quiet_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "event_velocity_norm_mean": event_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "pred_velocity_norm_mean": total_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return total_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="187c_v0: discrete budgeted-support residual model")
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
    parser.add_argument("--support_hidden_dim", type=int, default=128)
    parser.add_argument("--context_feat_dim", type=int, default=2)
    parser.add_argument("--n_support_blocks", type=int, default=5)
    parser.add_argument("--init_support_rate", type=float, default=0.05)
    parser.add_argument("--teacher_topk", type=int, default=3)
    parser.add_argument("--teacher_threshold", type=float, default=0.08)
    parser.add_argument("--relax_temperature", type=float, default=0.33)
    parser.add_argument("--support_budget_min", type=int, default=0)
    parser.add_argument("--support_budget_max", type=int, default=3)
    parser.add_argument("--support_budget_eps", type=float, default=1e-8)
    parser.add_argument("--support_budget_round_bias", type=float, default=0.0)
    parser.add_argument("--quiet_floor", type=float, default=0.15)
    parser.add_argument("--event_floor", type=float, default=0.05)
    parser.add_argument("--quiet_local_loss_weight", type=float, default=0.25)
    parser.add_argument("--quiet_band_loss_weight", type=float, default=0.15)
    parser.add_argument("--event_local_loss_weight", type=float, default=0.70)
    parser.add_argument("--event_band_loss_weight", type=float, default=0.25)
    parser.add_argument("--post_support_loss_weight", type=float, default=0.50)
    parser.add_argument("--prior_support_loss_weight", type=float, default=0.20)
    parser.add_argument("--support_kl_weight", type=float, default=0.10)
    parser.add_argument("--support_rate_reg_weight", type=float, default=0.05)
    parser.add_argument("--support_tv_reg_weight", type=float, default=0.0)
    parser.add_argument("--event_quiet_penalty_weight", type=float, default=0.05)
    parser.add_argument("--event_hidden_dim", type=int, default=96)
    parser.add_argument("--event_amp_max", type=float, default=0.85)
    parser.add_argument("--event_band_max", type=float, default=0.22)
    parser.add_argument("--event_local_clip", type=float, default=0.70)
    parser.add_argument("--event_band_clip", type=float, default=0.25)
    parser.add_argument("--event_scale_max", type=float, default=1.10)
    parser.add_argument("--init_event_scale", type=float, default=0.55)
    parser.add_argument("--event_residual_fm_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_prior_std", type=float, default=0.20)
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
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)
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
    support_config = dict(
        support_hidden_dim=args.support_hidden_dim,
        context_feat_dim=args.context_feat_dim,
        n_support_blocks=args.n_support_blocks,
        init_support_rate=args.init_support_rate,
        teacher_topk=args.teacher_topk,
        teacher_threshold=args.teacher_threshold,
        relax_temperature=args.relax_temperature,
        support_budget_min=args.support_budget_min,
        support_budget_max=args.support_budget_max,
        support_budget_eps=args.support_budget_eps,
        support_budget_round_bias=args.support_budget_round_bias,
        quiet_floor=args.quiet_floor,
        event_floor=args.event_floor,
        quiet_local_loss_weight=args.quiet_local_loss_weight,
        quiet_band_loss_weight=args.quiet_band_loss_weight,
        event_local_loss_weight=args.event_local_loss_weight,
        event_band_loss_weight=args.event_band_loss_weight,
        post_support_loss_weight=args.post_support_loss_weight,
        prior_support_loss_weight=args.prior_support_loss_weight,
        support_kl_weight=args.support_kl_weight,
        support_rate_reg_weight=args.support_rate_reg_weight,
        support_tv_reg_weight=args.support_tv_reg_weight,
        event_quiet_penalty_weight=args.event_quiet_penalty_weight,
    )
    event_config = dict(
        event_hidden_dim=args.event_hidden_dim,
        event_amp_max=args.event_amp_max,
        event_band_max=args.event_band_max,
        event_local_clip=args.event_local_clip,
        event_band_clip=args.event_band_clip,
        event_scale_max=args.event_scale_max,
        init_event_scale=args.init_event_scale,
        event_residual_fm_loss_weight=args.event_residual_fm_loss_weight,
        event_prior_std=args.event_prior_std,
    )

    model = DiscreteBudgetedSupportResidualModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        support_config=support_config,
        event_config=event_config,
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
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            params = (
                list(model.support_context.parameters())
                + list(model.support_prior_head.parameters())
                + list(model.support_post_head.parameters())
                + list(model.event_amplitude_head.parameters())
                + list(model.event_band_head.parameters())
                + list(model.event_transport.parameters())
                + [model.event_scale_logit]
            )
            optimizer = torch.optim.AdamW(params, lr=args.lr_event_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "budget-support-head"
        else:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            params = [
                {
                    "params": (
                        list(model.support_context.parameters())
                        + list(model.support_prior_head.parameters())
                        + list(model.support_post_head.parameters())
                        + list(model.event_amplitude_head.parameters())
                        + list(model.event_band_head.parameters())
                        + list(model.event_transport.parameters())
                        + [model.event_scale_logit]
                    ),
                    "lr": args.lr_event_stage2,
                },
                {
                    "params": list(model.path_context_adapter.parameters()),
                    "lr": args.lr_adapter_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "budget-support-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n187c_v0: discrete budgeted-support residual model\n{'=' * 72}")
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
        "ssr_total_loss",
        "event_residual_fm_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_support_loss",
        "prior_support_loss",
        "support_kl",
        "support_rate_reg",
        "support_tv_reg",
        "event_quiet_penalty",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_support_rate",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "target_event_basis_abs_mean",
        "pred_event_basis_abs_mean",
        "post_support_rate",
        "post_support_top1_mean",
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
        "prior_support_rate",
        "prior_support_top1_mean",
        "support_logit_abs_mean",
        "event_amp_abs_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
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
            loss, metrics = sparse_support_flow_matching_loss(model, history_01, future_01)
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
        current_key = strict_checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
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
                        "type": "discrete_budgeted_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187c",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "support_model": support_config,
                        "event": event_config,
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
            f"val_total={val_metrics['val_total_loss']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"worstLate={frontier_metrics['frontier_turb_late_worst_cov']:.3f}  "
            f"bestLate={frontier_metrics['frontier_turb_late_best_cov']:.3f}  "
            f"kurt={frontier_metrics['frontier_pooled_kurt_ratio']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"pSup={val_metrics['val_prior_support_rate']:.3f}  "
            f"qSup={val_metrics['val_post_support_rate']:.3f}  "
            f"eAmp={val_metrics['val_event_amp_abs_mean']:.3f}  "
            f"eBasis={val_metrics['val_pred_event_basis_abs_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "discrete_budgeted_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187c",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "support_model": support_config,
            "event": event_config,
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
