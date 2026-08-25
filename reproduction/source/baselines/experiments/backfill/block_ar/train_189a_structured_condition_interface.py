#!/usr/bin/env python
"""
189a_v0: Structured condition interface on top of the 183c backbone.

Principle:
  - keep the explicit mean/covariance/pathwise residual-law backbone
  - replace the single detached global path context with trainable history tokens
    and dynamic context attention
  - explicitly force context sensitivity so shuffled context cannot be ignored
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
    normalize_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel
from experiments.backfill.block_ar.train_186b_e2e_sign_aware_concentration import (
    build_sign_aware_weights,
    sign_aware_local_band_loss,
    soft_zone_masses,
)


class StructuredConditionInterfaceModel(StateMetricTransportModel):
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
        condition_interface_config: dict,
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
        self.condition_interface_config = dict(condition_interface_config)
        ctx_dim = path_config["context_dim"]
        max_history_len = int(condition_interface_config["max_history_len"])
        self.history_token_proj = nn.Linear(encoder_config.gru_hidden_dim, ctx_dim)
        self.history_pos_emb = nn.Parameter(torch.randn(1, max_history_len, ctx_dim) * 0.02)
        self.context_query = nn.Sequential(
            nn.Linear(ctx_dim + 5, condition_interface_config["query_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(condition_interface_config["query_hidden_dim"], ctx_dim),
        )
        self.history_attn = nn.MultiheadAttention(
            embed_dim=ctx_dim,
            num_heads=condition_interface_config["attn_heads"],
            batch_first=True,
        )
        self.context_blend = nn.Sequential(
            nn.Linear(ctx_dim * 2, condition_interface_config["blend_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(condition_interface_config["blend_hidden_dim"], ctx_dim),
        )
        self.context_norm = nn.LayerNorm(ctx_dim)

    def build_history_tokens(self, history_01: torch.Tensor) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        x = history_norm.reshape(history_norm.shape[0], history_norm.shape[1], -1)
        seq_out, _ = self.encoder.gru(x)
        tokens = self.history_token_proj(seq_out)
        pos = self.history_pos_emb[:, : history_norm.shape[1]]
        return tokens + pos

    def build_dynamic_context(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        base_path_context: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local_state, band_state = self.build_state_features(z_t_flat)
        local_abs_mean = local_state.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        query_in = torch.cat([base_path_context, t.unsqueeze(-1), local_abs_mean, band_state], dim=-1)
        query = self.context_query(query_in).unsqueeze(1)
        attn_out, attn_weights = self.history_attn(query, history_tokens, history_tokens, need_weights=True)
        attn_out = attn_out.squeeze(1)
        dyn = self.context_blend(torch.cat([base_path_context, attn_out], dim=-1))
        dynamic_context = self.context_norm(base_path_context + dyn)
        weights = attn_weights.squeeze(1)
        top1 = weights.max(dim=-1).values.mean()
        entropy = (-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        metrics = {
            "ctx_delta_norm": (dynamic_context - base_path_context).norm(dim=-1).mean(),
            "ctx_attn_top1": top1,
            "ctx_attn_entropy": entropy,
        }
        return dynamic_context, metrics

    def build_state_metric_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        base_path_context: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dynamic_context, ctx_metrics = self.build_dynamic_context(z_t_flat, t, base_path_context, history_tokens)
        static_local = self.width_allocator(dynamic_context)
        static_band = self.band_tail(dynamic_context)
        state_local, state_band = self.build_state_features(z_t_flat)
        gate_in = torch.cat([dynamic_context, t.unsqueeze(-1)], dim=-1)
        local_gate = self.local_state_gate(gate_in)
        band_gate = self.band_state_gate(gate_in)

        raw_local = static_local + local_gate.unsqueeze(-1) * state_local
        raw_local = raw_local - raw_local.mean(dim=(1, 2), keepdim=True)
        raw_local = raw_local.clamp(
            min=-self.integrated_config["width_clip"],
            max=self.integrated_config["width_clip"],
        )

        raw_band = static_band + band_gate * state_band
        raw_band = raw_band.clamp(
            min=-self.integrated_config["band_clip"],
            max=self.integrated_config["band_clip"],
        )

        metric_local = self.normalize_local_metric(raw_local)
        metric_band = self.normalize_band_metric(raw_band)
        return raw_local, raw_band, metric_local, metric_band, local_gate, band_gate, ctx_metrics

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        base_path_context: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_local, raw_band, metric_local, metric_band, local_gate, band_gate, ctx_metrics = self.build_state_metric_controls(
            z_t_flat, t, base_path_context, history_tokens
        )
        dynamic_context, _ = self.build_dynamic_context(z_t_flat, t, base_path_context, history_tokens)
        raw_v = self.path_transport(z_t_flat, t, dynamic_context)
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
            **ctx_metrics,
        }
        return mod_v_basis.reshape(z_t_flat.shape[0], -1), metrics

    def sample_basis_paths_conditioned(
        self,
        base_path_context: torch.Tensor,
        history_tokens: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = base_path_context.shape[0]
        z, prior_stats = self.prior.sample(batch * n_samples, device=base_path_context.device, dtype=base_path_context.dtype)
        ctx = base_path_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        tokens = history_tokens.unsqueeze(1).expand(batch, n_samples, history_tokens.shape[1], history_tokens.shape[2])
        tokens = tokens.reshape(batch * n_samples, history_tokens.shape[1], history_tokens.shape[2])
        dt = 1.0 / float(self.n_ode_steps)
        ctx_top1 = []
        for step in range(self.n_ode_steps):
            t = torch.full(
                (batch * n_samples,),
                fill_value=step * dt,
                device=base_path_context.device,
                dtype=base_path_context.dtype,
            )
            v, metrics = self.transport_velocity(z, t, ctx, tokens)
            z = z + dt * v
            ctx_top1.append(metrics["ctx_attn_top1"])
        stats = dict(prior_stats)
        stats["sample_basis_std_mean"] = z.std(dim=-1).mean()
        stats["sample_ctx_attn_top1"] = torch.stack(ctx_top1).mean()
        return z, stats

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
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
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        base_path_context = self.build_path_context(flow_context, block_logits, base_local_delta, scale)
        history_tokens = self.build_history_tokens(history_01)
        basis_paths, _stats = self.sample_basis_paths_conditioned(base_path_context, history_tokens, n_samples=n_samples)
        base_white_flat = self.path_geometry.from_basis(
            basis_paths.view(batch * n_samples, n_frames, n_cells)
        ).reshape(batch * n_samples, n_frames * n_cells)
        base_white = base_white_flat.view(batch * n_samples, self.decoder.n_blocks, self.decoder.block_len, n_cells)

        block_probs = F.softmax(block_logits, dim=-1)
        sampled_blocks = []
        for b in range(self.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)
        assign_flat = sampled_assign.reshape(batch * n_samples, self.decoder.n_blocks)
        sampled_factors, _logdet_cov, _offdiag_rms = self.decoder.build_template_factors(assign_flat)
        lhs = base_white.permute(0, 1, 3, 2).reshape(
            batch * n_samples * self.decoder.n_blocks,
            n_cells,
            self.decoder.block_len,
        )
        factor_flat = sampled_factors.reshape(
            batch * n_samples * self.decoder.n_blocks,
            n_cells,
            n_cells,
        )
        routed_white = torch.matmul(factor_flat, lhs)
        routed_white = routed_white.reshape(
            batch * n_samples,
            self.decoder.n_blocks,
            n_cells,
            self.decoder.block_len,
        ).permute(0, 1, 3, 2)
        routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
        return mu.unsqueeze(1) + noise * local_scale


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def structured_condition_loss(
    model: StructuredConditionInterfaceModel,
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
        det_outputs = model.forward_from_history(history_01)
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=det_outputs[0],
            time_factor=det_outputs[1],
            time_diag=det_outputs[2],
            cell_factor=det_outputs[3],
            cell_diag=det_outputs[4],
            scale=det_outputs[5],
            base_local_delta=det_outputs[7],
            block_logits=det_outputs[8],
        ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)

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
    base_path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
    history_tokens = model.build_history_tokens(history_01)

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0

    pred_v, control_metrics = model.transport_velocity(z_t, t, base_path_context, history_tokens)
    raw_local, raw_band, metric_local, metric_band, _local_gate, _band_gate, ctx_metrics = model.build_state_metric_controls(
        z_t, t, base_path_context, history_tokens
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

    perm = torch.randperm(history_tokens.shape[0], device=history_tokens.device)
    with torch.no_grad():
        _shuf_raw_local, _shuf_raw_band, shuf_metric_local, shuf_metric_band, *_rest = model.build_state_metric_controls(
            z_t, t, base_path_context, history_tokens[perm]
        )
    local_shift = (metric_local - shuf_metric_local).abs().mean(dim=(1, 2))
    band_shift = (metric_band - shuf_metric_band).abs().mean(dim=1)
    sensitivity_local = F.relu(objective_config["context_margin_local"] - local_shift)
    sensitivity_band = F.relu(objective_config["context_margin_band"] - band_shift)
    context_sensitivity = (window_boost * (sensitivity_local + sensitivity_band)).mean()

    total = (
        fm_loss
        + objective_config["local_pos_loss_weight"] * local_pos_loss
        + objective_config["local_neg_loss_weight"] * local_neg_loss
        + objective_config["band_pos_loss_weight"] * band_pos_loss
        + objective_config["band_neg_loss_weight"] * band_neg_loss
        + objective_config["underfit_gap_weight"] * underfit_gap
        + objective_config["overwide_gap_weight"] * overwide_gap
        + objective_config["spectrum_loss_weight"] * spectrum_loss
        + objective_config["context_sensitivity_weight"] * context_sensitivity
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
        "sci_total_loss": total,
        "local_pos_loss": local_pos_loss,
        "local_neg_loss": local_neg_loss,
        "band_pos_loss": band_pos_loss,
        "band_neg_loss": band_neg_loss,
        "underfit_gap": underfit_gap,
        "overwide_gap": overwide_gap,
        "spectrum_loss": spectrum_loss,
        "context_sensitivity_loss": context_sensitivity,
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
        **ctx_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: StructuredConditionInterfaceModel,
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
        "sci_total_loss",
        "local_pos_loss",
        "local_neg_loss",
        "band_pos_loss",
        "band_neg_loss",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "context_sensitivity_loss",
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
        "ctx_delta_norm",
        "ctx_attn_top1",
        "ctx_attn_entropy",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = structured_condition_loss(model, history_01, future_01, objective_config)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_sci_total_loss"]
    return out


def instantiate_from_warm_start(
    warm_start_path: str,
    device: str,
    condition_interface_config: dict,
) -> tuple[StructuredConditionInterfaceModel, dict]:
    payload = torch.load(warm_start_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = StructuredConditionInterfaceModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        condition_interface_config=condition_interface_config,
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="189a_v0: structured condition interface retrain")
    parser.add_argument("--epochs_stage1", type=int, default=3)
    parser.add_argument("--epochs_stage2", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_interface_stage1", type=float, default=4e-4)
    parser.add_argument("--lr_interface_stage2", type=float, default=2e-4)
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
    parser.add_argument("--context_sensitivity_weight", type=float, default=0.18)
    parser.add_argument("--context_margin_local", type=float, default=0.030)
    parser.add_argument("--context_margin_band", type=float, default=0.018)
    parser.add_argument("--cond_attn_heads", type=int, default=4)
    parser.add_argument("--cond_query_hidden_dim", type=int, default=256)
    parser.add_argument("--cond_blend_hidden_dim", type=int, default=256)
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

    condition_interface_config = dict(
        max_history_len=args.history_len,
        attn_heads=args.cond_attn_heads,
        query_hidden_dim=args.cond_query_hidden_dim,
        blend_hidden_dim=args.cond_blend_hidden_dim,
    )
    objective_config = dict(
        late_horizon_max_weight=args.late_horizon_max_weight,
        positive_focus_weight=args.positive_focus_weight,
        negative_focus_weight=args.negative_focus_weight,
        positive_band_focus_weight=args.positive_band_focus_weight,
        negative_band_focus_weight=args.negative_band_focus_weight,
        window_focus_weight=args.window_focus_weight,
        window_high_band_mix=args.window_high_band_mix,
        window_focus_clip=args.window_focus_clip,
        local_boost_clip=args.local_boost_clip,
        band_boost_clip=args.band_boost_clip,
        local_weight_max=args.local_weight_max,
        band_weight_max=args.band_weight_max,
        local_pos_loss_weight=args.local_pos_loss_weight,
        local_neg_loss_weight=args.local_neg_loss_weight,
        band_pos_loss_weight=args.band_pos_loss_weight,
        band_neg_loss_weight=args.band_neg_loss_weight,
        underfit_gap_weight=args.underfit_gap_weight,
        overwide_gap_weight=args.overwide_gap_weight,
        quiet_quantile=args.quiet_quantile,
        extreme_quantile=args.extreme_quantile,
        quiet_weight=args.quiet_weight,
        shoulder_weight=args.shoulder_weight,
        extreme_weight=args.extreme_weight,
        spectrum_tau=args.spectrum_tau,
        spectrum_loss_weight=args.spectrum_loss_weight,
        context_sensitivity_weight=args.context_sensitivity_weight,
        context_margin_local=args.context_margin_local,
        context_margin_band=args.context_margin_band,
    )

    model, warm_cfg = instantiate_from_warm_start(args.warm_start_path, args.device, condition_interface_config)

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

    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def interface_params():
        return (
            list(model.path_context_adapter.parameters())
            + list(model.width_allocator.parameters())
            + list(model.band_tail.parameters())
            + list(model.local_state_gate.parameters())
            + list(model.band_state_gate.parameters())
            + list(model.history_token_proj.parameters())
            + list(model.context_query.parameters())
            + list(model.history_attn.parameters())
            + list(model.context_blend.parameters())
            + list(model.context_norm.parameters())
            + [model.local_metric_budget_logit, model.band_metric_budget_logit, model.history_pos_emb]
        )

    def set_stage(stage: int):
        model.encoder.requires_grad_(True)
        if stage == 1:
            model.path_transport.requires_grad_(False)
            params = [
                {"params": interface_params(), "lr": args.lr_interface_stage1},
                {"params": list(model.encoder.parameters()), "lr": args.lr_encoder_stage1},
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "structured-cond-interface"
        else:
            model.path_transport.requires_grad_(True)
            params = [
                {"params": interface_params(), "lr": args.lr_interface_stage2},
                {"params": list(model.path_transport.parameters()), "lr": args.lr_path_stage2},
                {"params": list(model.encoder.parameters()), "lr": args.lr_encoder_stage2},
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "structured-cond-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n189a_v0: structured condition interface\n{'=' * 72}")
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
        "sci_total_loss",
        "local_pos_loss",
        "local_neg_loss",
        "band_pos_loss",
        "band_neg_loss",
        "underfit_gap",
        "overwide_gap",
        "spectrum_loss",
        "context_sensitivity_loss",
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
        "ctx_delta_norm",
        "ctx_attn_top1",
        "ctx_attn_entropy",
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
            loss, metrics = structured_condition_loss(model, history_01, future_01, objective_config)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "structured_condition_interface_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_189a",
                        "encoder": vars(model.encoder_config),
                        "decoder": warm_cfg["decoder"],
                        "flow": warm_cfg["flow"],
                        "path": warm_cfg["path"],
                        "prior": warm_cfg["prior"],
                        "integrated": warm_cfg["integrated"],
                        "state": warm_cfg["state"],
                        "metric": warm_cfg["metric"],
                        "cond_interface": condition_interface_config,
                        "support_lo": warm_cfg.get("support_lo", 0.01),
                        "support_hi": warm_cfg.get("support_hi", 1.0),
                        "support_eps": warm_cfg.get("support_eps", 1e-5),
                        "base_nu": warm_cfg.get("base_nu", 8.0),
                        "history_len": args.history_len,
                        "future_len": args.future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": warm_cfg.get("mix_chunk_size", 27),
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
            f"ctxTop={val_metrics['val_ctx_attn_top1']:.3f}  ctxH={val_metrics['val_ctx_attn_entropy']:.3f}  "
            f"ctxSens={val_metrics['val_context_sensitivity_loss']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "structured_condition_interface_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_189a",
            "encoder": vars(model.encoder_config),
            "decoder": warm_cfg["decoder"],
            "flow": warm_cfg["flow"],
            "path": warm_cfg["path"],
            "prior": warm_cfg["prior"],
            "integrated": warm_cfg["integrated"],
            "state": warm_cfg["state"],
            "metric": warm_cfg["metric"],
            "cond_interface": condition_interface_config,
            "support_lo": warm_cfg.get("support_lo", 0.01),
            "support_hi": warm_cfg.get("support_hi", 1.0),
            "support_eps": warm_cfg.get("support_eps", 1e-5),
            "base_nu": warm_cfg.get("base_nu", 8.0),
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": warm_cfg.get("mix_chunk_size", 27),
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))


if __name__ == "__main__":
    main()
