#!/usr/bin/env python
"""
193b: Reparameterized-rollout training for the 193a latent-factor AR model.

Keep the same latent-factor AR model class as 193a, but change training:
  - teacher-forced exact mixture likelihood
  - teacher-forced sharpness control
  - high-variance mixture occupancy regularization
  - differentiable closed-loop rollout loss via reparameterized innovation sampling
  - rollout variance amplification penalty
  - rollout correlation drift penalty

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_193b_reparameterized_rollout_latent_factor.py \
        --output_dir models/backfill/graph_ar_latent_factor_rollout_193b --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    make_serializable,
    normalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_193a_graph_ar_latent_factor_innovation import (
    LatentFactorInnovationARModel,
    compute_cond_from_outputs,
    evaluate_rollout_subset,
    maybe_load_partial_warm_start,
    parse_component_nus,
)


def cov_to_corr(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps))
    denom = std.unsqueeze(-1) * std.unsqueeze(-2)
    corr = cov / denom.clamp_min(eps)
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye) + eye
    return corr.clamp(-1.0, 1.0)


def offdiag_corr_mse(corr_a: torch.Tensor, corr_b: torch.Tensor) -> torch.Tensor:
    n = corr_a.shape[-1]
    mask = 1.0 - torch.eye(n, device=corr_a.device, dtype=corr_a.dtype)
    diff = (corr_a - corr_b) * mask.unsqueeze(0)
    denom = mask.sum().clamp_min(1.0)
    return diff.pow(2).sum(dim=(-1, -2)).mean() / denom


def safe_mixture_nll(
    model: LatentFactorInnovationARModel,
    target_u: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    mix_logits: torch.Tensor,
    component_nus: torch.Tensor,
) -> torch.Tensor:
    d = target_u.shape[-1]
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0).unsqueeze(0)
    cov_sym = 0.5 * (cov + cov.transpose(-1, -2))
    jitter = model.cov_jitter
    chol = None
    cov_safe = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        try:
            cov_safe = cov_sym + (jitter * mult) * eye
            chol = torch.linalg.cholesky(cov_safe)
            break
        except RuntimeError:
            continue
    if chol is None or cov_safe is None:
        raise RuntimeError("Failed to stabilize mixture covariance for Cholesky factorization")

    diff = (target_u - mu).unsqueeze(1).unsqueeze(-1)
    solved = torch.cholesky_solve(diff, chol).squeeze(-1)
    mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

    nu = component_nus.clamp_min(2.1 + 1e-6)
    pi = target_u.new_tensor(math.pi)
    log_norm = (
        torch.lgamma((nu + d) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * (d * torch.log(nu * pi) + logdet)
    )
    log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
    component_logprob = log_norm + log_kernel
    mix_logprob = F.log_softmax(mix_logits, dim=-1)
    return -torch.logsumexp(mix_logprob + component_logprob, dim=-1)


def reparameterized_next_u(
    model: LatentFactorInnovationARModel,
    mu: torch.Tensor,
    base_factor: torch.Tensor,
    diag: torch.Tensor,
    factor_scales: torch.Tensor,
    diag_component_mod: torch.Tensor,
    mix_logits: torch.Tensor,
    component_nus: torch.Tensor,
    gumbel_tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mix_soft = F.gumbel_softmax(mix_logits, tau=gumbel_tau, hard=False, dim=-1)
    chosen_factor_scales = torch.einsum("bc,bcf->bf", mix_soft, factor_scales)
    chosen_diag_mod = torch.einsum("bc,bc->b", mix_soft, diag_component_mod)
    chosen_nu = torch.einsum("bc,bc->b", mix_soft, component_nus).clamp_min(2.1 + 1e-6)

    eps_factor = torch.randn(
        mu.shape[0], base_factor.shape[-1], device=mu.device, dtype=mu.dtype
    )
    eps_diag = torch.randn(
        mu.shape[0], mu.shape[-1], device=mu.device, dtype=mu.dtype
    )
    lowrank_noise = torch.einsum("bnk,bk->bn", base_factor, chosen_factor_scales * eps_factor)
    diag_noise = diag * chosen_diag_mod.unsqueeze(-1) * eps_diag

    gamma = torch.distributions.Gamma(chosen_nu / 2.0, chosen_nu / 2.0)
    gamma_sample = gamma.rsample().clamp_min(1e-6)
    t_scale = torch.rsqrt(gamma_sample).unsqueeze(-1)
    sample_u = mu + (lowrank_noise + diag_noise) * t_scale
    return sample_u, {
        "mix_soft": mix_soft,
        "chosen_nu": chosen_nu,
    }


def teacher_forced_objective(
    model: LatentFactorInnovationARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, hist_len = history_01.shape[:2]
    future_len = future_01.shape[1]
    n_cells = future_01.shape[-1]

    hist_norm = normalize_iv(history_01).reshape(batch_size, hist_len, n_cells)
    gru_outputs, gru_state = model.encoder.gru(hist_norm)
    prev_01 = history_01[:, -1].reshape(batch_size, n_cells)

    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_idio_share = 0.0
    total_factor_scale = 0.0
    total_mix_entropy = 0.0
    total_high_var_prob = 0.0
    total_total_var = 0.0
    total_attention_top1 = 0.0
    total_sharpness_pen = 0.0

    pred_mean_path = []
    target_mean_path = []

    for step in range(future_len):
        cond, attn_weights = compute_cond_from_outputs(model, gru_outputs)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus = model.decoder(cond, prev_u)
        cov, _loadings, diag_per_component = model.mixture_covariances(
            base_factor=base_factor,
            diag=diag,
            factor_scales=factor_scales,
            diag_component_mod=diag_component_mod,
        )
        expected_cov = model.expected_covariance(cov, mix_logits)
        total_var = torch.diagonal(expected_cov, dim1=-2, dim2=-1).mean(dim=-1)

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        nll_t = safe_mixture_nll(model, target_u, mu, cov, mix_logits, component_nus)
        idio_share_t = model.idio_share(cov, diag_per_component, mix_logits)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
        mix_probs = F.softmax(mix_logits, dim=-1)
        mix_entropy = -(mix_probs * mix_probs.clamp_min(1e-8).log()).sum(dim=-1)
        comp_total_var = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
        high_var_idx = comp_total_var.argmax(dim=-1, keepdim=True)
        high_var_prob = mix_probs.gather(1, high_var_idx).squeeze(-1)

        pred_var_cells = torch.diagonal(expected_cov, dim1=-2, dim2=-1)
        resid2 = (target_u - mu).detach().pow(2)
        sharpness_pen = F.relu(
            pred_var_cells - (objective_config["sharpness_ratio"] * resid2 + objective_config["sharpness_floor"])
        ).mean()

        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(expected_cov).mean()
        total_idio_share = total_idio_share + idio_share_t.mean()
        total_factor_scale = total_factor_scale + factor_scales.mean()
        total_mix_entropy = total_mix_entropy + mix_entropy.mean()
        total_high_var_prob = total_high_var_prob + high_var_prob.mean()
        total_total_var = total_total_var + total_var.mean()
        total_attention_top1 = total_attention_top1 + attn_weights.max(dim=1).values.mean()
        total_sharpness_pen = total_sharpness_pen + sharpness_pen

        pred_mean_path.append(mu_iv.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        next_norm = normalize_iv(target_t).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = target_t

    pred_mean_path_t = torch.stack(pred_mean_path, dim=1)
    target_mean_path_t = torch.stack(target_mean_path, dim=1)
    path_mean_loss = F.smooth_l1_loss(pred_mean_path_t, target_mean_path_t)
    idio_share_mean = total_idio_share / future_len
    idio_share_penalty = F.relu(idio_share_mean - objective_config["idio_share_cap"]).pow(2)
    high_var_prob_mean = total_high_var_prob / future_len
    high_var_penalty = F.relu(high_var_prob_mean - objective_config["high_var_prob_cap"]).pow(2)
    sharpness_penalty = total_sharpness_pen / future_len

    total_loss = (
        total_nll / future_len
        + objective_config["path_mean_weight"] * path_mean_loss
        + objective_config["idio_share_weight"] * idio_share_penalty
        + objective_config["sharpness_weight"] * sharpness_penalty
        + objective_config["high_var_weight"] * high_var_penalty
    )

    metrics = {
        "total_loss": total_loss,
        "multistep_nll": total_nll / future_len,
        "multistep_mae": total_mae / future_len,
        "pred_eff_rank": total_rank / future_len,
        "idio_share": idio_share_mean,
        "factor_scale_mean": total_factor_scale / future_len,
        "mix_entropy": total_mix_entropy / future_len,
        "high_var_component_prob": high_var_prob_mean,
        "total_var_per_cell": total_total_var / future_len,
        "path_mean_loss": path_mean_loss,
        "idio_share_penalty": idio_share_penalty,
        "sharpness_penalty": sharpness_penalty,
        "high_var_penalty": high_var_penalty,
        "attention_top1": total_attention_top1 / future_len,
    }
    return total_loss, metrics


def rollout_augmented_objective(
    model: LatentFactorInnovationARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
    gumbel_tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, hist_len = history_01.shape[:2]
    future_len = future_01.shape[1]
    n_cells = future_01.shape[-1]

    hist_norm = normalize_iv(history_01).reshape(batch_size, hist_len, n_cells)
    tf_outputs, tf_state = model.encoder.gru(hist_norm)
    ro_outputs, ro_state = model.encoder.gru(hist_norm)
    prev_tf_01 = history_01[:, -1].reshape(batch_size, n_cells)
    prev_ro_01 = prev_tf_01.clone()

    total_tf_nll = 0.0
    total_tf_mae = 0.0
    total_tf_rank = 0.0
    total_tf_idio_share = 0.0
    total_tf_factor_scale = 0.0
    total_tf_mix_entropy = 0.0
    total_tf_high_var_prob = 0.0
    total_tf_total_var = 0.0
    total_tf_attention_top1 = 0.0
    total_tf_sharpness_pen = 0.0

    total_ro_nll = 0.0
    total_ro_total_var = 0.0
    total_ro_high_var_prob = 0.0
    total_ro_mix_entropy = 0.0
    total_ro_var_amp_pen = 0.0
    total_ro_corr_pen = 0.0

    pred_mean_path_tf = []
    pred_mean_path_ro = []
    target_mean_path = []

    for step in range(future_len):
        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )

        cond_tf, attn_weights_tf = compute_cond_from_outputs(model, tf_outputs)
        prev_tf_u = iv_to_unconstrained(
            prev_tf_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu_tf, base_factor_tf, diag_tf, factor_scales_tf, diag_component_mod_tf, mix_logits_tf, component_nus_tf = model.decoder(cond_tf, prev_tf_u)
        cov_tf, _load_tf, diag_per_component_tf = model.mixture_covariances(
            base_factor_tf, diag_tf, factor_scales_tf, diag_component_mod_tf
        )
        expected_cov_tf = model.expected_covariance(cov_tf, mix_logits_tf)
        total_var_tf = torch.diagonal(expected_cov_tf, dim1=-2, dim2=-1).mean(dim=-1)
        mix_probs_tf = F.softmax(mix_logits_tf, dim=-1)
        mix_entropy_tf = -(mix_probs_tf * mix_probs_tf.clamp_min(1e-8).log()).sum(dim=-1)
        comp_total_var_tf = torch.diagonal(cov_tf, dim1=-2, dim2=-1).sum(dim=-1)
        high_var_idx_tf = comp_total_var_tf.argmax(dim=-1, keepdim=True)
        high_var_prob_tf = mix_probs_tf.gather(1, high_var_idx_tf).squeeze(-1)
        idio_share_tf = model.idio_share(cov_tf, diag_per_component_tf, mix_logits_tf)
        mu_iv_tf = unconstrained_to_iv(mu_tf, lo=model.support_lo, hi=model.support_hi)
        pred_var_cells_tf = torch.diagonal(expected_cov_tf, dim1=-2, dim2=-1)
        resid2 = (target_u - mu_tf).detach().pow(2)
        sharpness_pen_tf = F.relu(
            pred_var_cells_tf - (objective_config["sharpness_ratio"] * resid2 + objective_config["sharpness_floor"])
        ).mean()

        nll_tf = safe_mixture_nll(model, target_u, mu_tf, cov_tf, mix_logits_tf, component_nus_tf)

        cond_ro, _attn_weights_ro = compute_cond_from_outputs(model, ro_outputs)
        prev_ro_u = iv_to_unconstrained(
            prev_ro_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu_ro, base_factor_ro, diag_ro, factor_scales_ro, diag_component_mod_ro, mix_logits_ro, component_nus_ro = model.decoder(cond_ro, prev_ro_u)
        cov_ro, _load_ro, _diag_per_component_ro = model.mixture_covariances(
            base_factor_ro, diag_ro, factor_scales_ro, diag_component_mod_ro
        )
        expected_cov_ro = model.expected_covariance(cov_ro, mix_logits_ro)
        total_var_ro = torch.diagonal(expected_cov_ro, dim1=-2, dim2=-1).mean(dim=-1)
        mix_probs_ro = F.softmax(mix_logits_ro, dim=-1)
        mix_entropy_ro = -(mix_probs_ro * mix_probs_ro.clamp_min(1e-8).log()).sum(dim=-1)
        comp_total_var_ro = torch.diagonal(cov_ro, dim1=-2, dim2=-1).sum(dim=-1)
        high_var_idx_ro = comp_total_var_ro.argmax(dim=-1, keepdim=True)
        high_var_prob_ro = mix_probs_ro.gather(1, high_var_idx_ro).squeeze(-1)
        mu_iv_ro = unconstrained_to_iv(mu_ro, lo=model.support_lo, hi=model.support_hi)
        nll_ro = safe_mixture_nll(model, target_u, mu_ro, cov_ro, mix_logits_ro, component_nus_ro)

        corr_tf = cov_to_corr(expected_cov_tf.detach())
        corr_ro = cov_to_corr(expected_cov_ro)
        corr_pen = offdiag_corr_mse(corr_ro, corr_tf)
        var_amp_pen = F.relu(
            total_var_ro - objective_config["rollout_var_allowance"] * total_var_tf.detach()
        ).pow(2).mean()

        total_tf_nll = total_tf_nll + nll_tf.mean()
        total_tf_mae = total_tf_mae + (mu_iv_tf - target_t).abs().mean()
        total_tf_rank = total_tf_rank + effective_rank(expected_cov_tf).mean()
        total_tf_idio_share = total_tf_idio_share + idio_share_tf.mean()
        total_tf_factor_scale = total_tf_factor_scale + factor_scales_tf.mean()
        total_tf_mix_entropy = total_tf_mix_entropy + mix_entropy_tf.mean()
        total_tf_high_var_prob = total_tf_high_var_prob + high_var_prob_tf.mean()
        total_tf_total_var = total_tf_total_var + total_var_tf.mean()
        total_tf_attention_top1 = total_tf_attention_top1 + attn_weights_tf.max(dim=1).values.mean()
        total_tf_sharpness_pen = total_tf_sharpness_pen + sharpness_pen_tf

        total_ro_nll = total_ro_nll + nll_ro.mean()
        total_ro_total_var = total_ro_total_var + total_var_ro.mean()
        total_ro_high_var_prob = total_ro_high_var_prob + high_var_prob_ro.mean()
        total_ro_mix_entropy = total_ro_mix_entropy + mix_entropy_ro.mean()
        total_ro_var_amp_pen = total_ro_var_amp_pen + var_amp_pen
        total_ro_corr_pen = total_ro_corr_pen + corr_pen

        pred_mean_path_tf.append(mu_iv_tf.mean(dim=-1))
        pred_mean_path_ro.append(mu_iv_ro.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        next_tf_norm = normalize_iv(target_t).unsqueeze(1)
        next_tf_out, tf_state = model.encoder.gru(next_tf_norm, tf_state)
        tf_outputs = torch.cat([tf_outputs, next_tf_out], dim=1)
        prev_tf_01 = target_t

        sample_ro_u, _sample_stats = reparameterized_next_u(
            model,
            mu_ro,
            base_factor_ro,
            diag_ro,
            factor_scales_ro,
            diag_component_mod_ro,
            mix_logits_ro,
            component_nus_ro,
            gumbel_tau=gumbel_tau,
        )
        next_ro = unconstrained_to_iv(sample_ro_u, lo=model.support_lo, hi=model.support_hi)
        next_ro_norm = normalize_iv(next_ro).unsqueeze(1)
        next_ro_out, ro_state = model.encoder.gru(next_ro_norm, ro_state)
        ro_outputs = torch.cat([ro_outputs, next_ro_out], dim=1)
        prev_ro_01 = next_ro

    pred_mean_path_tf = torch.stack(pred_mean_path_tf, dim=1)
    pred_mean_path_ro = torch.stack(pred_mean_path_ro, dim=1)
    target_mean_path = torch.stack(target_mean_path, dim=1)

    path_mean_loss_tf = F.smooth_l1_loss(pred_mean_path_tf, target_mean_path)
    path_mean_loss_ro = F.smooth_l1_loss(pred_mean_path_ro, target_mean_path)
    idio_share_mean = total_tf_idio_share / future_len
    idio_share_penalty = F.relu(idio_share_mean - objective_config["idio_share_cap"]).pow(2)
    high_var_prob_mean_tf = total_tf_high_var_prob / future_len
    high_var_prob_mean_ro = total_ro_high_var_prob / future_len
    high_var_penalty_tf = F.relu(high_var_prob_mean_tf - objective_config["high_var_prob_cap"]).pow(2)
    high_var_penalty_ro = F.relu(high_var_prob_mean_ro - objective_config["high_var_prob_cap"]).pow(2)
    sharpness_penalty = total_tf_sharpness_pen / future_len

    total_loss = (
        total_tf_nll / future_len
        + objective_config["path_mean_weight"] * path_mean_loss_tf
        + objective_config["idio_share_weight"] * idio_share_penalty
        + objective_config["sharpness_weight"] * sharpness_penalty
        + objective_config["high_var_weight"] * high_var_penalty_tf
        + objective_config["rollout_nll_weight"] * (total_ro_nll / future_len)
        + objective_config["rollout_path_mean_weight"] * path_mean_loss_ro
        + objective_config["rollout_var_amp_weight"] * (total_ro_var_amp_pen / future_len)
        + objective_config["rollout_corr_weight"] * (total_ro_corr_pen / future_len)
        + objective_config["high_var_weight"] * high_var_penalty_ro
    )

    metrics = {
        "total_loss": total_loss,
        "multistep_nll": total_tf_nll / future_len,
        "multistep_mae": total_tf_mae / future_len,
        "pred_eff_rank": total_tf_rank / future_len,
        "idio_share": idio_share_mean,
        "factor_scale_mean": total_tf_factor_scale / future_len,
        "mix_entropy": total_tf_mix_entropy / future_len,
        "high_var_component_prob": high_var_prob_mean_tf,
        "total_var_per_cell": total_tf_total_var / future_len,
        "path_mean_loss": path_mean_loss_tf,
        "idio_share_penalty": idio_share_penalty,
        "sharpness_penalty": sharpness_penalty,
        "high_var_penalty": high_var_penalty_tf,
        "attention_top1": total_tf_attention_top1 / future_len,
        "rollout_nll": total_ro_nll / future_len,
        "rollout_total_var_per_cell": total_ro_total_var / future_len,
        "rollout_high_var_component_prob": high_var_prob_mean_ro,
        "rollout_mix_entropy": total_ro_mix_entropy / future_len,
        "rollout_path_mean_loss": path_mean_loss_ro,
        "rollout_var_amp_penalty": total_ro_var_amp_pen / future_len,
        "rollout_corr_penalty": total_ro_corr_pen / future_len,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: LatentFactorInnovationARModel,
    val_loader: DataLoader,
    objective_config: dict,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = teacher_forced_objective(
            model,
            history_01,
            future_01,
            objective_config=objective_config,
        )
        batch_size = history_01.shape[0]
        total_count += batch_size
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
    if total_count == 0:
        return {}
    return {f"val_{k}": v / total_count for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="193b: rollout-trained latent-factor AR")
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_stage1", type=float, default=5e-4)
    parser.add_argument("--lr_stage2", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_factors", type=int, default=5)
    parser.add_argument("--num_components", type=int, default=3)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--factor_scale_floor", type=float, default=1e-4)
    parser.add_argument("--component_diag_mod_max", type=float, default=0.10)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_factor_scale", type=float, default=0.18)
    parser.add_argument("--component_nus", type=str, default="24.0,8.0,3.5")
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--path_mean_weight", type=float, default=0.20)
    parser.add_argument("--idio_share_weight", type=float, default=3.0)
    parser.add_argument("--idio_share_cap", type=float, default=0.42)
    parser.add_argument("--sharpness_weight", type=float, default=0.80)
    parser.add_argument("--sharpness_ratio", type=float, default=1.15)
    parser.add_argument("--sharpness_floor", type=float, default=2e-3)
    parser.add_argument("--high_var_weight", type=float, default=1.50)
    parser.add_argument("--high_var_prob_cap", type=float, default=0.55)
    parser.add_argument("--rollout_nll_weight", type=float, default=0.60)
    parser.add_argument("--rollout_path_mean_weight", type=float, default=0.10)
    parser.add_argument("--rollout_var_amp_weight", type=float, default=0.40)
    parser.add_argument("--rollout_var_allowance", type=float, default=1.10)
    parser.add_argument("--rollout_corr_weight", type=float, default=0.12)
    parser.add_argument("--gumbel_tau", type=float, default=0.70)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--warm_start", type=str, default="models/backfill/graph_ar_latent_factor_innovation_193a/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    component_nus = parse_component_nus(args.component_nus)
    if len(component_nus) != args.num_components:
        raise ValueError("num_components must match number of component_nus")

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
        train_indices = train_indices[:args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[:args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        num_factors=args.num_factors,
        num_components=args.num_components,
        diag_floor=args.diag_floor,
        factor_scale_floor=args.factor_scale_floor,
        component_diag_mod_max=args.component_diag_mod_max,
        init_diag=args.init_diag,
        init_factor_scale=args.init_factor_scale,
        component_nus=component_nus,
    )
    objective_config = dict(
        path_mean_weight=args.path_mean_weight,
        idio_share_weight=args.idio_share_weight,
        idio_share_cap=args.idio_share_cap,
        sharpness_weight=args.sharpness_weight,
        sharpness_ratio=args.sharpness_ratio,
        sharpness_floor=args.sharpness_floor,
        high_var_weight=args.high_var_weight,
        high_var_prob_cap=args.high_var_prob_cap,
        rollout_nll_weight=args.rollout_nll_weight,
        rollout_path_mean_weight=args.rollout_path_mean_weight,
        rollout_var_amp_weight=args.rollout_var_amp_weight,
        rollout_var_allowance=args.rollout_var_allowance,
        rollout_corr_weight=args.rollout_corr_weight,
    )

    model = LatentFactorInnovationARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_partial_warm_start(model, args.warm_start)

    print("=" * 72)
    print("193b: Reparameterized-rollout latent-factor AR")
    print("=" * 72)
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"Stage1 epochs: {args.stage1_epochs} | Stage2 epochs: {args.stage2_epochs}")
    print(f"Factors: {args.num_factors} | Components: {args.num_components}")
    print(f"Warm start: {args.warm_start}")

    best_score = float("inf")
    best_metrics = None
    history = []
    total_epochs = args.stage1_epochs + args.stage2_epochs

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        if epoch <= args.stage1_epochs:
            stage = 1
            lr = args.lr_stage1
        else:
            stage = 2
            lr = args.lr_stage2

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        train_sums: dict[str, float] = {}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            if stage == 1:
                loss, metrics = teacher_forced_objective(
                    model,
                    history_01,
                    future_01,
                    objective_config=objective_config,
                )
            else:
                loss, metrics = rollout_augmented_objective(
                    model,
                    history_01,
                    future_01,
                    objective_config=objective_config,
                    gumbel_tau=args.gumbel_tau,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in metrics.items():
                train_sums[k] = train_sums.get(k, 0.0) + float(v.item())
            nb += 1

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in train_sums.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )

        score = (
            val_metrics["val_total_loss"]
            + 0.40 * max(0.0, rollout_metrics["rollout_cov90"] - 0.90)
            + 0.10 * max(0.0, 0.90 - rollout_metrics["rollout_cov90"])
            + 0.14 * max(0.0, 1.10 - rollout_metrics["rollout_turb_calm_ratio"])
            + 0.12 * max(0.0, 0.72 - rollout_metrics["rollout_corr_ratio_h30"])
            + 0.10 * max(0.0, rollout_metrics["rollout_rank_ratio_h30"] - 1.80)
            + 0.12 * max(0.0, rollout_metrics["rollout_width90"] - 0.18)
            + 0.30 * max(0.0, val_metrics["val_high_var_component_prob"] - args.high_var_prob_cap)
            + 0.18 * max(0.0, val_metrics["val_sharpness_penalty"] - 0.02)
        )
        is_best = score < best_score
        if is_best:
            best_score = score
            best_metrics = {**val_metrics, **rollout_metrics, "score": score}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": val_metrics["val_total_loss"],
                    "score": score,
                    "config": {
                        "type": "graph_ar_latent_factor_rollout_193b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {
            "epoch": epoch,
            "stage": stage,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
            "selection_score": score,
        }
        history.append(row)
        elapsed = time.time() - t0
        print(
            f"Ep {epoch:3d} stg={stage} "
            f"train={train_metrics['train_total_loss']:.4f} "
            f"val={val_metrics['val_total_loss']:.4f} "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f} "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f} "
            f"roll_corr={rollout_metrics['rollout_corr_ratio_h30']:.3f} "
            f"roll_rank={rollout_metrics['rollout_rank_ratio_h30']:.3f} "
            f"highvar={val_metrics['val_high_var_component_prob']:.3f} "
            f"sharp={val_metrics['val_sharpness_penalty']:.4f} "
            f"({elapsed:.1f}s)"
            + (" *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "score": history[-1]["selection_score"] if history else float("nan"),
        "config": {
            "type": "graph_ar_latent_factor_rollout_193b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "cov_jitter": args.cov_jitter,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        },
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    if best_metrics is not None:
        print(
            f"Best score: {best_score:.4f} | "
            f"roll_cov90={best_metrics['rollout_cov90']:.4f} | "
            f"roll_corr_h30={best_metrics['rollout_corr_ratio_h30']:.3f} | "
            f"roll_rank_h30={best_metrics['rollout_rank_ratio_h30']:.3f}"
        )


if __name__ == "__main__":
    main()
