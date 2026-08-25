#!/usr/bin/env python
"""
181a_v0: Unified residual-state conditional law on top of the 179b backbone.

Keep:
  - explicit mean-reverting mean dynamics
  - structured covariance backbone
  - geometry-aware centered residual transport

Change:
  - replace separate width/jump patches with one latent residual-state context
  - use a blockwise prior from history and posterior from teacher residuals
  - let the same residual-state context drive both smooth residual shape and sparse jumps
  - freeze the validated backbone in v0 and train only the missing residual-state modules
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_178d_exact_block_covariance_mixture_mean_reverting_residual_flow import (
    aggregate_slope_ratio,
)
from experiments.backfill.block_ar.train_179b_basis_centered_residual_transport import (
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
)
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import (
    SparseGroupedJumpModule,
    build_block_band_group_masks,
    evaluate_joint_subset,
)


def build_block_feature_masks(geometry, n_blocks: int) -> tuple[torch.Tensor, list[dict[str, int | str]], torch.Tensor]:
    group_masks, metadata = build_block_band_group_masks(geometry, n_blocks)
    band_to_idx = {"low": 0, "mid": 1, "high": 2}
    feature_masks = torch.zeros(n_blocks, 4, geometry.n_frames * geometry.n_cells, dtype=group_masks.dtype)
    for gi, meta in enumerate(metadata):
        b = int(meta["block"])
        feature_masks[b, band_to_idx[str(meta["band"])]] = group_masks[gi]
        feature_masks[b, 3] = torch.maximum(feature_masks[b, 3], group_masks[gi])
    return feature_masks, metadata, group_masks


class ResidualStateModule(nn.Module):
    def __init__(
        self,
        context_dim: int,
        n_blocks: int,
        n_states: int = 4,
        post_feat_dim: int = 4,
        hidden_dim: int = 128,
        init_scale: float = 0.05,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.n_blocks = n_blocks
        self.n_states = n_states
        self.prior_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_blocks * n_states),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(post_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_states),
        )
        self.state_table = nn.Parameter(torch.zeros(n_blocks, n_states, context_dim))
        nn.init.normal_(self.state_table, mean=0.0, std=init_scale)

    def prior_logits(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.prior_net(context)
        return logits.view(context.shape[0], self.n_blocks, self.n_states)

    def posterior_logits(self, block_feats: torch.Tensor) -> torch.Tensor:
        batch, n_blocks, feat_dim = block_feats.shape
        logits = self.posterior_net(block_feats.reshape(batch * n_blocks, feat_dim))
        return logits.view(batch, n_blocks, self.n_states)

    def context_shift(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: [B, n_blocks, n_states]
        block_ctx = torch.einsum("bks,ksd->bkd", probs, self.state_table)
        return block_ctx.mean(dim=1)


class UnifiedResidualStateMeanRevertingCovarianceMixtureModel(
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        residual_state_config: dict,
        jump_config: dict,
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
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        feature_masks, feature_metadata, jump_group_masks = build_block_feature_masks(self.flow.geometry, self.decoder.n_blocks)
        self.register_buffer("residual_feature_masks", feature_masks.float())
        self.residual_feature_metadata = feature_metadata
        self.residual_state = ResidualStateModule(
            context_dim=flow_config["context_dim"],
            n_blocks=self.decoder.n_blocks,
            **residual_state_config,
        )
        self.jump = SparseGroupedJumpModule(
            context_dim=flow_config["context_dim"],
            group_masks=jump_group_masks,
            **jump_config,
        )
        self.jump_group_metadata = [
            {"block": int(m["block"]), "band": str(m["band"])} for m in build_block_band_group_masks(self.flow.geometry, self.decoder.n_blocks)[1]
        ]
        self.residual_state_config = residual_state_config
        self.jump_config = jump_config

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("residual_state.") or key.startswith("jump."):
                skipped.append(key)
                continue
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

    def posterior_block_features(self, basis_flat: torch.Tensor) -> torch.Tensor:
        sq = basis_flat.pow(2)
        numer = torch.einsum("bd,kfd->bkf", sq, self.residual_feature_masks)
        denom = self.residual_feature_masks.sum(dim=-1).unsqueeze(0).clamp_min(1.0)
        return torch.sqrt(numer / denom)

    def infer_residual_contexts(
        self,
        flow_context: torch.Tensor,
        basis_flat: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        prior_logits = self.residual_state.prior_logits(flow_context)
        prior_probs = F.softmax(prior_logits, dim=-1)
        prior_shift = self.residual_state.context_shift(prior_probs)
        prior_context = flow_context + prior_shift

        out = {
            "prior_logits": prior_logits,
            "prior_probs": prior_probs,
            "prior_shift": prior_shift,
            "prior_context": prior_context,
        }
        if basis_flat is not None:
            block_feats = self.posterior_block_features(basis_flat)
            posterior_logits = self.residual_state.posterior_logits(block_feats)
            posterior_probs = F.softmax(posterior_logits, dim=-1)
            posterior_shift = self.residual_state.context_shift(posterior_probs)
            posterior_context = flow_context + posterior_shift
            kl = (
                posterior_probs
                * (torch.log(posterior_probs.clamp_min(1e-8)) - torch.log(prior_probs.clamp_min(1e-8)))
            ).sum(dim=-1)
            out.update(
                {
                    "posterior_logits": posterior_logits,
                    "posterior_probs": posterior_probs,
                    "posterior_shift": posterior_shift,
                    "posterior_context": posterior_context,
                    "posterior_block_feats": block_feats,
                    "kl": kl,
                }
            )
        return out

    @torch.no_grad()
    def teacher_basis_flat(self, history_01: torch.Tensor, future_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_u = iv_to_unconstrained(future_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
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
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        shared_local_scale = torch.exp(0.5 * shared_local_delta)
        diff = (target_u - mu) / shared_local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white_shared = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_blocks = white_shared.view(history_01.shape[0], self.decoder.n_blocks, self.decoder.block_len, self.flow.n_cells)
        map_assign = block_logits.argmax(dim=-1)
        map_factors, _map_logdet, _map_offdiag_rms = self.decoder.build_template_factors(map_assign)
        rhs_map = white_blocks.permute(0, 1, 3, 2).reshape(history_01.shape[0] * self.decoder.n_blocks, self.flow.n_cells, self.decoder.block_len)
        factor_map = map_factors.reshape(history_01.shape[0] * self.decoder.n_blocks, self.flow.n_cells, self.flow.n_cells)
        base_blocks_map = torch.linalg.solve_triangular(factor_map, rhs_map, upper=False)
        base_blocks_map = base_blocks_map.reshape(history_01.shape[0], self.decoder.n_blocks, self.flow.n_cells, self.decoder.block_len).permute(0, 1, 3, 2)
        white_flat_map = base_blocks_map.reshape(history_01.shape[0], self.flow.n_frames * self.flow.n_cells)
        basis_flat = self.flow.to_basis_flat(white_flat_map)
        return basis_flat, flow_context.detach()

    def log_prob_future(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        time_factor: torch.Tensor,
        time_diag: torch.Tensor,
        cell_factor: torch.Tensor,
        cell_diag: torch.Tensor,
        scale: torch.Tensor,
        flow_context: torch.Tensor,
        base_local_delta: torch.Tensor,
        block_logits: torch.Tensor,
    ):
        batch, n_frames, n_cells = target_u.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        assignments = self.decoder.assignment_index.to(target_u.device)
        log_prior_blocks = F.log_softmax(block_logits, dim=-1)
        block_probs = F.softmax(block_logits, dim=-1)

        log_prior = target_u.new_zeros(batch, assignments.shape[0])
        for b in range(self.decoder.n_blocks):
            idx = assignments[:, b].unsqueeze(0).expand(batch, -1)
            log_prior = log_prior + log_prior_blocks[:, b].gather(1, idx)

        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        shared_local_scale = torch.exp(0.5 * shared_local_delta)
        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c
        logdet_local = 2.0 * torch.log(shared_local_scale).sum(dim=(1, 2))
        diff = (target_u - mu) / shared_local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white_shared = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_blocks = white_shared.view(batch, self.decoder.n_blocks, self.decoder.block_len, n_cells)

        map_assign = block_logits.argmax(dim=-1)
        local_scale_map = shared_local_scale
        map_factors, _map_logdet, map_offdiag_rms = self.decoder.build_template_factors(map_assign)
        rhs_map = white_blocks.permute(0, 1, 3, 2).reshape(batch * self.decoder.n_blocks, n_cells, self.decoder.block_len)
        factor_map = map_factors.reshape(batch * self.decoder.n_blocks, n_cells, n_cells)
        base_blocks_map = torch.linalg.solve_triangular(factor_map, rhs_map, upper=False)
        base_blocks_map = base_blocks_map.reshape(batch, self.decoder.n_blocks, n_cells, self.decoder.block_len).permute(0, 1, 3, 2)
        white_flat_map = base_blocks_map.reshape(batch, n_frames * n_cells)
        basis_coeff_map = self.flow.to_basis_flat(white_flat_map)

        state_aux = self.infer_residual_contexts(flow_context, basis_coeff_map)
        train_context = state_aux["posterior_context"]

        mix_logprob = None
        for start in range(0, assignments.shape[0], self.mix_chunk_size):
            end = min(start + self.mix_chunk_size, assignments.shape[0])
            chunk_assign = assignments[start:end]
            factors, logdet_template_cov, _offdiag_rms = self.decoder.build_template_factors(chunk_assign)
            chunk = end - start
            obs = white_blocks.unsqueeze(1).expand(batch, chunk, -1, -1, -1)
            rhs = obs.permute(0, 1, 2, 4, 3).reshape(batch * chunk * self.decoder.n_blocks, n_cells, self.decoder.block_len)
            factor_batch = factors.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(
                batch * chunk * self.decoder.n_blocks, n_cells, n_cells
            )
            base_blocks = torch.linalg.solve_triangular(factor_batch, rhs, upper=False)
            base_blocks = base_blocks.reshape(batch, chunk, self.decoder.n_blocks, n_cells, self.decoder.block_len).permute(0, 1, 2, 4, 3)
            white_flat = base_blocks.reshape(batch * chunk, n_frames * n_cells)
            ctx = train_context.unsqueeze(1).expand(batch, chunk, -1).reshape(batch * chunk, -1)
            z, flow_logdet = self.flow(white_flat, ctx)
            base_logprob = self._base_logprob(z).view(batch, chunk)
            flow_logdet = flow_logdet.view(batch, chunk)
            comp_logprob = base_logprob + flow_logdet - 0.5 * (
                logdet_cov.unsqueeze(1) + logdet_local.unsqueeze(1) + logdet_template_cov.unsqueeze(0)
            )
            chunk_lse = torch.logsumexp(log_prior[:, start:end] + comp_logprob, dim=1)
            mix_logprob = chunk_lse if mix_logprob is None else torch.logaddexp(mix_logprob, chunk_lse)

        z_map, flow_logdet_map, flow_stats = self.flow.forward_with_stats(white_flat_map, train_context)
        high_mask = self.flow.geometry.high_band_mask().to(white_flat_map.device)
        high_denom = high_mask.sum().clamp_min(1.0)
        high_coeff_std = ((basis_coeff_map.pow(2) * high_mask.unsqueeze(0)).sum(dim=-1) / high_denom).sqrt()
        basis_coeff_std = basis_coeff_map.std(dim=-1)

        jump_logits, jump_probs, jump_scales = self.jump.params(train_context)
        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "flow_logdet": flow_logdet_map,
            "white_std": white_flat_map.std(dim=-1),
            "z_std": z_map.std(dim=-1),
            "basis_coeff_std": basis_coeff_std,
            "high_band_coeff_std": high_coeff_std,
            "basis_flat_map": basis_coeff_map,
            "flow_mean_logscale": flow_stats["mean_logscale"],
            "flow_abs_logscale": flow_stats["abs_logscale"],
            "high_band_neg_logscale": flow_stats["high_band_neg_logscale"],
            "high_band_abs_logscale": flow_stats["high_band_abs_logscale"],
            "local_delta_rms": shared_local_delta.pow(2).mean(dim=(1, 2)).sqrt(),
            "local_scale_min": local_scale_map.amin(dim=(1, 2)),
            "local_scale_max": local_scale_map.amax(dim=(1, 2)),
            "template_diag_mean": torch.diagonal(map_factors, dim1=-2, dim2=-1).mean(dim=(1, 2)),
            "template_offdiag_rms": map_offdiag_rms,
            "block_gate_entropy": (-(block_probs * torch.log(block_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "block_gate_max": block_probs.max(dim=-1).values.mean(dim=-1),
            "block_usage": block_probs.mean(dim=(0, 1)),
            "resid_prior_probs": state_aux["prior_probs"],
            "resid_post_probs": state_aux["posterior_probs"],
            "resid_prior_entropy": (-(state_aux["prior_probs"] * torch.log(state_aux["prior_probs"].clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "resid_post_entropy": (-(state_aux["posterior_probs"] * torch.log(state_aux["posterior_probs"].clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "resid_kl": state_aux["kl"],
            "resid_shift_norm": state_aux["posterior_shift"].norm(dim=-1),
            "resid_usage_prior": state_aux["prior_probs"].mean(dim=(0, 1)),
            "resid_usage_post": state_aux["posterior_probs"].mean(dim=(0, 1)),
            "jump_logits": jump_logits,
            "jump_group_probs": jump_probs,
            "jump_group_scales": jump_scales,
        }
        return mix_logprob, aux

    @torch.no_grad()
    def sample_future_u(self, history_01: torch.Tensor, n_samples: int):
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
        state_aux = self.infer_residual_contexts(flow_context, basis_flat=None)
        sample_context = state_aux["prior_context"]

        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        ctx = sample_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        smooth_white_flat, _ = self.flow.inverse(z, ctx)
        smooth_basis_flat = self.flow.to_basis_flat(smooth_white_flat)
        jump_shift, _jump_stats = self.jump.sample_shifts(sample_context, n_samples=n_samples)
        basis_with_jump = smooth_basis_flat.view(batch, n_samples, -1) + jump_shift
        base_white_flat = self.flow.from_basis_flat(basis_with_jump.reshape(batch * n_samples, -1))
        base_white = base_white_flat.view(batch * n_samples, self.decoder.n_blocks, self.decoder.block_len, n_cells)

        block_probs = F.softmax(block_logits, dim=-1)
        sampled_blocks = []
        for b in range(self.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)
        assign_flat = sampled_assign.reshape(batch * n_samples, self.decoder.n_blocks)
        sampled_factors, _logdet_cov, _offdiag_rms = self.decoder.build_template_factors(assign_flat)
        lhs = base_white.permute(0, 1, 3, 2).reshape(batch * n_samples * self.decoder.n_blocks, n_cells, self.decoder.block_len)
        factor_flat = sampled_factors.reshape(batch * n_samples * self.decoder.n_blocks, n_cells, n_cells)
        routed_white = torch.matmul(factor_flat, lhs)
        routed_white = routed_white.reshape(batch * n_samples, self.decoder.n_blocks, n_cells, self.decoder.block_len).permute(0, 1, 3, 2)
        routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        shared_local_delta = self.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
        return mu.unsqueeze(1) + noise * local_scale


def checkpoint_key(val_metrics: dict, joint_metrics: dict) -> tuple[float, float, float, float, float]:
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("nan")))
    kurt_ratio = float(joint_metrics.get("joint_kurtosis_ratio", float("nan")))
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    cov90 = float(joint_metrics.get("joint_cov90", float("nan")))
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    kurt_gap = 1e6 if not np.isfinite(kurt_ratio) else max(0.5 - kurt_ratio, 0.0) + max(kurt_ratio - 2.0, 0.0)
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    cov_gap = 1e6 if not np.isfinite(cov90) else abs(cov90 - 0.90)
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    return (jump_gap, kurt_gap, mr_gap, cov_gap, tc_gap)


def estimate_jump_thresholds(
    model: UnifiedResidualStateMeanRevertingCovarianceMixtureModel,
    train_loader: DataLoader,
    max_batches: int,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_rms = []
    for batch_idx, (history_01, future_01) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        basis_flat, _ = model.teacher_basis_flat(history_01, future_01)
        all_rms.append(model.jump.group_rms(basis_flat))
    rms = torch.cat(all_rms, dim=0)
    thresholds = torch.quantile(rms, quantile, dim=0)
    target_rates = (rms > thresholds.unsqueeze(0)).float().mean(dim=0)
    return thresholds.detach(), target_rates.detach()


def unified_residual_state_loss(
    model: UnifiedResidualStateMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    thresholds: torch.Tensor,
    target_rates: torch.Tensor,
    kl_weight: float,
    jump_bce_weight: float,
    jump_scale_weight: float,
    jump_rate_weight: float,
    state_balance_weight: float,
):
    target_u = iv_to_unconstrained(future_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
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
    logprob, aux = model.log_prob_future(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        flow_context=flow_context,
        base_local_delta=base_local_delta,
        block_logits=block_logits,
    )

    group_rms = model.jump.group_rms(aux["basis_flat_map"])
    target_active = (group_rms > thresholds.unsqueeze(0)).float()
    target_excess = (group_rms - thresholds.unsqueeze(0)).clamp_min(0.0)
    jump_bce = F.binary_cross_entropy_with_logits(aux["jump_logits"], target_active)
    active_mask = target_active > 0.5
    if active_mask.any():
        jump_scale = F.smooth_l1_loss(aux["jump_group_scales"][active_mask], target_excess[active_mask], beta=0.10)
        target_excess_active_mean = target_excess[active_mask].mean()
    else:
        jump_scale = group_rms.new_tensor(0.0)
        target_excess_active_mean = group_rms.new_tensor(0.0)
    jump_rate = (aux["jump_group_probs"].mean(dim=0) - target_rates).abs().mean()

    uniform_state = torch.full_like(aux["resid_usage_prior"], 1.0 / aux["resid_usage_prior"].numel())
    state_balance = (aux["resid_usage_prior"] - uniform_state).pow(2).mean() + (aux["resid_usage_post"] - uniform_state).pow(2).mean()
    nll = (-logprob).mean()
    kl = aux["resid_kl"].mean()
    loss = (
        nll
        + kl_weight * kl
        + jump_bce_weight * jump_bce
        + jump_scale_weight * jump_scale
        + jump_rate_weight * jump_rate
        + state_balance_weight * state_balance
    )

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    det_mr_ratio = aggregate_slope_ratio(
        history_01[:, -1].reshape(history_01.shape[0], -1),
        future_01[:, 0].reshape(future_01.shape[0], -1),
        pred_01[:, 0],
    )
    metrics = {
        "total_loss": loss,
        "joint_nll": nll,
        "resid_kl": kl,
        "jump_bce": jump_bce,
        "jump_scale": jump_scale,
        "jump_rate": jump_rate,
        "state_balance": state_balance,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "joint_det_mr_ratio": pred_01.new_tensor(det_mr_ratio),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
        "basis_coeff_std_mean": aux["basis_coeff_std"].mean(),
        "high_band_coeff_std_mean": aux["high_band_coeff_std"].mean(),
        "flow_mean_logscale_mean": aux["flow_mean_logscale"].mean(),
        "flow_abs_logscale_mean": aux["flow_abs_logscale"].mean(),
        "high_band_neg_logscale_mean": aux["high_band_neg_logscale"].mean(),
        "high_band_abs_logscale_mean": aux["high_band_abs_logscale"].mean(),
        "resid_prior_entropy": aux["resid_prior_entropy"].mean(),
        "resid_post_entropy": aux["resid_post_entropy"].mean(),
        "resid_shift_norm": aux["resid_shift_norm"].mean(),
        "jump_pred_active_rate": aux["jump_group_probs"].mean(),
        "jump_pred_scale_mean": aux["jump_group_scales"].mean(),
        "target_active_rate": target_active.mean(),
        "target_excess_active_mean": target_excess_active_mean,
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: UnifiedResidualStateMeanRevertingCovarianceMixtureModel,
    val_loader: DataLoader,
    thresholds: torch.Tensor,
    target_rates: torch.Tensor,
    kl_weight: float,
    jump_bce_weight: float,
    jump_scale_weight: float,
    jump_rate_weight: float,
    state_balance_weight: float,
):
    model.eval()
    keys = [
        "total_loss","joint_nll","resid_kl","jump_bce","jump_scale","jump_rate","state_balance",
        "joint_mae","joint_det_mr_ratio","white_std_mean","z_std_mean","basis_coeff_std_mean",
        "high_band_coeff_std_mean","flow_mean_logscale_mean","flow_abs_logscale_mean",
        "high_band_neg_logscale_mean","high_band_abs_logscale_mean","resid_prior_entropy",
        "resid_post_entropy","resid_shift_norm","jump_pred_active_rate","jump_pred_scale_mean",
        "target_active_rate","target_excess_active_mean"
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = unified_residual_state_loss(
            model,
            history_01,
            future_01,
            thresholds=thresholds,
            target_rates=target_rates,
            kl_weight=kl_weight,
            jump_bce_weight=jump_bce_weight,
            jump_scale_weight=jump_scale_weight,
            jump_rate_weight=jump_rate_weight,
            state_balance_weight=state_balance_weight,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="181a_v0: unified residual-state conditional law")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_state", type=float, default=7e-4)
    parser.add_argument("--lr_jump", type=float, default=7e-4)
    parser.add_argument("--weight_decay_state", type=float, default=0.0)
    parser.add_argument("--weight_decay_jump", type=float, default=0.0)
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
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--mix_chunk_size", type=int, default=27)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--resid_n_states", type=int, default=4)
    parser.add_argument("--resid_hidden_dim", type=int, default=128)
    parser.add_argument("--resid_init_scale", type=float, default=0.05)
    parser.add_argument("--jump_hidden_dim", type=int, default=128)
    parser.add_argument("--jump_max_scale", type=float, default=0.60)
    parser.add_argument("--jump_init_logit_bias", type=float, default=-1.25)
    parser.add_argument("--jump_init_scale_bias", type=float, default=-2.0)
    parser.add_argument("--jump_amplitude_df", type=float, default=5.0)
    parser.add_argument("--threshold_quantile", type=float, default=0.97)
    parser.add_argument("--threshold_batches", type=int, default=120)
    parser.add_argument("--kl_weight", type=float, default=0.10)
    parser.add_argument("--jump_bce_weight", type=float, default=0.50)
    parser.add_argument("--jump_scale_weight", type=float, default=0.25)
    parser.add_argument("--jump_rate_weight", type=float, default=0.25)
    parser.add_argument("--state_balance_weight", type=float, default=0.10)
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    hist_len = args.history_len
    future_len = args.future_len

    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)
    decoder_config = dict(
        n_frames=future_len,
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
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        n_frames=future_len,
        grid_h=5,
        grid_w=5,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        low_scale_clip=args.flow_low_scale_clip,
        mid_scale_clip=args.flow_mid_scale_clip,
        high_scale_clip=args.flow_high_scale_clip,
    )
    residual_state_config = dict(
        n_states=args.resid_n_states,
        hidden_dim=args.resid_hidden_dim,
        init_scale=args.resid_init_scale,
    )
    jump_config = dict(
        hidden_dim=args.jump_hidden_dim,
        max_scale=args.jump_max_scale,
        init_logit_bias=args.jump_init_logit_bias,
        init_scale_bias=args.jump_init_scale_bias,
        amplitude_df=args.jump_amplitude_df,
    )
    model = UnifiedResidualStateMeanRevertingCovarianceMixtureModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        residual_state_config=residual_state_config,
        jump_config=jump_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(device)

    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.residual_state.requires_grad_(True)
    model.jump.requires_grad_(True)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    n_state = sum(p.numel() for p in model.residual_state.parameters())
    n_jump = sum(p.numel() for p in model.jump.parameters())
    print(f"\n{'=' * 64}")
    print("181a_v0: unified residual-state conditional law")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Basis transport params: {n_flow:,}")
    print(f"  Residual-state params: {n_state:,}")
    print(f"  Jump params: {n_jump:,}")
    print(f"  Total params: {n_enc + n_dec + n_flow + n_state + n_jump:,}")
    if args.warm_start_path:
        print(f"  Warm start: {args.warm_start_path}")

    thresholds, target_rates = estimate_jump_thresholds(
        model,
        train_loader,
        max_batches=args.threshold_batches,
        quantile=args.threshold_quantile,
    )
    print(f"  Threshold quantile: {args.threshold_quantile}")
    print(f"  Mean target active rate: {target_rates.mean().item():.4f}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.residual_state.parameters(), "lr": args.lr_state, "weight_decay": args.weight_decay_state},
            {"params": model.jump.parameters(), "lr": args.lr_jump, "weight_decay": args.weight_decay_jump},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metric_names = [
        "total_loss","joint_nll","resid_kl","jump_bce","jump_scale","jump_rate","state_balance","joint_mae",
        "joint_det_mr_ratio","white_std_mean","z_std_mean","basis_coeff_std_mean","high_band_coeff_std_mean",
        "flow_mean_logscale_mean","flow_abs_logscale_mean","high_band_neg_logscale_mean","high_band_abs_logscale_mean",
        "resid_prior_entropy","resid_post_entropy","resid_shift_norm","jump_pred_active_rate","jump_pred_scale_mean",
        "target_active_rate","target_excess_active_mean"
    ]

    best_key = None
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {f"train_{k}": 0.0 for k in metric_names}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = unified_residual_state_loss(
                model,
                history_01,
                future_01,
                thresholds=thresholds,
                target_rates=target_rates,
                kl_weight=args.kl_weight,
                jump_bce_weight=args.jump_bce_weight,
                jump_scale_weight=args.jump_scale_weight,
                jump_rate_weight=args.jump_rate_weight,
                state_balance_weight=args.state_balance_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.residual_state.parameters()) + list(model.jump.parameters()), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += metrics[key.replace("train_", "")].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            thresholds=thresholds,
            target_rates=target_rates,
            kl_weight=args.kl_weight,
            jump_bce_weight=args.jump_bce_weight,
            jump_scale_weight=args.jump_scale_weight,
            jump_rate_weight=args.jump_rate_weight,
            state_balance_weight=args.state_balance_weight,
        )
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        elapsed = time.time() - t0
        current_key = checkpoint_key(val_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "unified_residual_state_mean_reverting_covariance_mixture_structured_joint_student_t_181a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "residual_state": residual_state_config,
                        "jump": jump_config,
                        "jump_group_metadata": model.jump_group_metadata,
                        "jump_thresholds": thresholds.detach().cpu().tolist(),
                        "jump_target_rates": target_rates.detach().cpu().tolist(),
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": args.mix_chunk_size,
                        "kl_weight": args.kl_weight,
                        "jump_bce_weight": args.jump_bce_weight,
                        "jump_scale_weight": args.jump_scale_weight,
                        "jump_rate_weight": args.jump_rate_weight,
                        "state_balance_weight": args.state_balance_weight,
                        "threshold_quantile": args.threshold_quantile,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))
        print(
            f"Ep {epoch:3d}  train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  mr_samp={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  kurt={joint_metrics['joint_kurtosis_ratio']:.3f}  "
            f"stateKL={val_metrics['val_resid_kl']:.4f}  pAct={val_metrics['val_jump_pred_active_rate']:.4f}  "
            f"pEnt={val_metrics['val_resid_prior_entropy']:.3f}  ({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "config": {
            "type": "unified_residual_state_mean_reverting_covariance_mixture_structured_joint_student_t_181a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "residual_state": residual_state_config,
            "jump": jump_config,
            "jump_group_metadata": model.jump_group_metadata,
            "jump_thresholds": thresholds.detach().cpu().tolist(),
            "jump_target_rates": target_rates.detach().cpu().tolist(),
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": args.mix_chunk_size,
            "kl_weight": args.kl_weight,
            "jump_bce_weight": args.jump_bce_weight,
            "jump_scale_weight": args.jump_scale_weight,
            "jump_rate_weight": args.jump_rate_weight,
            "state_balance_weight": args.state_balance_weight,
            "threshold_quantile": args.threshold_quantile,
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    if best_metrics is not None:
        print(
            "\nBest diagnostics: "
            f"joint_jump_ks={best_metrics['joint_pathwise_jump_ks']:.4f}, "
            f"joint_kurtosis_ratio={best_metrics['joint_kurtosis_ratio']:.3f}, "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
