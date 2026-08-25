#!/usr/bin/env python
"""
193a: Graph-aware AR latent-factor innovation mixture.

Fresh AR model class:
  - true AR recursion over future time
  - explicit conditional mean head
  - graph-aware shared loading matrix
  - exact conditional mixture of latent-factor Student-t innovations
  - rollout-aware selection to guard against correlation collapse / width blow-up

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_193a_graph_ar_latent_factor_innovation.py \
        --output_dir models/backfill/graph_ar_latent_factor_innovation_193a --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    inverse_softplus,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[mask].mean())


def corr_from_cov(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


class SpatialLatentFactorMixtureDecoder(nn.Module):
    """Spatial transformer trunk with latent-factor innovation mixture heads."""

    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        num_factors: int = 5,
        num_components: int = 3,
        diag_floor: float = 1e-3,
        factor_scale_floor: float = 1e-4,
        component_diag_mod_max: float = 0.10,
        init_diag: float = 0.10,
        init_factor_scale: float = 0.18,
        component_nus: tuple[float, ...] = (24.0, 8.0, 3.5),
    ):
        super().__init__()
        self.n_cells = n_cells
        self.num_factors = num_factors
        self.num_components = num_components
        self.diag_floor = diag_floor
        self.factor_scale_floor = factor_scale_floor
        self.component_diag_mod_max = component_diag_mod_max
        if len(component_nus) != num_components:
            raise ValueError("component_nus must match num_components")
        self.register_buffer("component_nus", torch.tensor(component_nus, dtype=torch.float32))

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)

        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ff_norm": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            }))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, num_factors)
        self.diag_head = nn.Linear(d_model, 1)

        context_dim = d_model + cond_dim
        self.mix_logits_head = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_components),
        )
        self.factor_scale_head = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_components * num_factors),
        )
        self.diag_component_mod_head = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_components),
        )

        self._init_parameters(init_diag=init_diag, init_factor_scale=init_factor_scale)

    def _init_parameters(self, init_diag: float, init_factor_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in (
                        "mean_head",
                        "factor_head",
                        "diag_head",
                        "mix_logits_head",
                        "factor_scale_head",
                        "diag_component_mod_head",
                    )
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.factor_head.bias, mean=0.0, std=1e-3)

        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(
            self.diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )

        for module in self.mix_logits_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mix_logits_head[-1].weight)
        nn.init.zeros_(self.mix_logits_head[-1].bias)

        for module in self.factor_scale_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.factor_scale_head[-1].weight)
        init_scale_bias = inverse_softplus(max(init_factor_scale - self.factor_scale_floor, 1e-6))
        nn.init.constant_(self.factor_scale_head[-1].bias, init_scale_bias)

        for module in self.diag_component_mod_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.diag_component_mod_head[-1].weight)
        nn.init.zeros_(self.diag_component_mod_head[-1].bias)

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(prev_u.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[2 * li]
            ls_f = self.ls_params[2 * li + 1]
            h_norm = layer["attn_norm"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer["ff"](layer["ff_norm"](h))

        h = self.out_norm(h)
        pooled = h.mean(dim=1)
        context = torch.cat([pooled, cond], dim=-1)

        mu = prev_u + self.mean_head(h).squeeze(-1)

        base_factor = self.factor_head(h)
        col_norm = torch.sqrt(base_factor.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-6))
        base_factor = base_factor / col_norm

        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor

        factor_scale_steps = F.softplus(
            self.factor_scale_head(context).view(-1, self.num_components, self.num_factors)
        ) + self.factor_scale_floor
        factor_scales = torch.cumsum(factor_scale_steps, dim=1)

        diag_mod_raw = self.diag_component_mod_head(context)
        diag_component_mod = 1.0 + self.component_diag_mod_max * torch.tanh(diag_mod_raw)

        mix_logits = self.mix_logits_head(context)
        component_nus = self.component_nus.unsqueeze(0).expand(context.shape[0], -1).to(context.dtype)
        return mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus


class LatentFactorInnovationARModel(nn.Module):
    """GRU history encoder + exact latent-factor innovation mixture."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = SpatialLatentFactorMixtureDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_norm = normalize_iv(history_01)
        output, _ = self.encoder.gru(history_norm.reshape(history_norm.shape[0], history_norm.shape[1], -1))
        attn_logits = self.encoder.attn_proj(output).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * output).sum(dim=1)
        cond = self.encoder.bottleneck(pooled)
        cond = self.encoder.dropout(cond)
        return cond, attn_weights, output

    def forward_from_history(
        self, history_01: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond, attn_weights, _ = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        out = self.decoder(cond, prev_u)
        return (*out, attn_weights.max(dim=1).values)

    def mixture_covariances(
        self,
        base_factor: torch.Tensor,
        diag: torch.Tensor,
        factor_scales: torch.Tensor,
        diag_component_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loadings = base_factor.unsqueeze(1) * factor_scales.unsqueeze(2)
        diag_per_component = diag.unsqueeze(1) * diag_component_mod.unsqueeze(-1)
        cov = torch.einsum("bcik,bcjk->bcij", loadings, loadings)
        cov = cov + torch.diag_embed(diag_per_component.pow(2) + self.cov_jitter)
        return cov, loadings, diag_per_component

    def expected_covariance(
        self,
        cov: torch.Tensor,
        mix_logits: torch.Tensor,
    ) -> torch.Tensor:
        mix_probs = F.softmax(mix_logits, dim=-1)
        return torch.einsum("bc,bcij->bij", mix_probs, cov)

    def idio_share(
        self,
        cov: torch.Tensor,
        diag_per_component: torch.Tensor,
        mix_logits: torch.Tensor,
    ) -> torch.Tensor:
        diag_var = diag_per_component.pow(2).sum(dim=-1)
        total_var = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-8)
        comp_share = diag_var / total_var
        mix_probs = F.softmax(mix_logits, dim=-1)
        return (mix_probs * comp_share).sum(dim=-1)

    def mixture_nll(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        cov: torch.Tensor,
        mix_logits: torch.Tensor,
        component_nus: torch.Tensor,
    ) -> torch.Tensor:
        d = target_u.shape[-1]
        chol = torch.linalg.cholesky(cov)

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

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus, _ = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        num_factors = base_factor.shape[-1]
        num_components = factor_scales.shape[1]

        mix_probs = F.softmax(mix_logits, dim=-1)
        comp_idx = torch.multinomial(mix_probs, n_samples, replacement=True)

        eps_factor = torch.randn(batch_size, n_samples, num_factors, device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)

        factor_scales_sel = torch.gather(
            factor_scales.unsqueeze(1).expand(-1, n_samples, -1, -1),
            2,
            comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_factors),
        ).squeeze(2)
        diag_mod_sel = torch.gather(
            diag_component_mod.unsqueeze(1).expand(-1, n_samples, -1),
            2,
            comp_idx.unsqueeze(-1),
        ).squeeze(-1)
        nu_sel = torch.gather(
            component_nus.unsqueeze(1).expand(-1, n_samples, -1),
            2,
            comp_idx.unsqueeze(-1),
        ).squeeze(-1)

        lowrank_noise = torch.einsum("bnk,bsk->bsn", base_factor, factor_scales_sel * eps_factor)
        diag_noise = diag.unsqueeze(1) * diag_mod_sel.unsqueeze(-1) * eps_diag

        gamma = torch.distributions.Gamma(nu_sel / 2.0, nu_sel / 2.0)
        mix = gamma.sample().to(mu.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)

        return mu.unsqueeze(1) + (lowrank_noise + diag_noise) * t_scale

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        samples_u = self.sample_next_u(history_01, n_samples=n_samples)
        return unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        history_01 = denormalize_iv(history)
        batch_size, hist_len = history_01.shape[:2]

        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist_k = history_01.unsqueeze(1).expand(batch_size, k, -1, -1, -1)
            hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()

            frames = []
            for _ in range(n_steps):
                next_iv = self.sample_next_iv(hist_k, n_samples=1).squeeze(1)
                frames.append(next_iv.reshape(batch_size, k, 5, 5))
                hist_k = torch.cat([hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)], dim=1)

            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def compute_cond_from_outputs(
    model: LatentFactorInnovationARModel,
    gru_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(pooled)
    cond = model.encoder.dropout(cond)
    return cond, attn_weights


def latent_factor_multistep_loss(
    model: LatentFactorInnovationARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
    self_feed_prob: float = 0.0,
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
    total_attn_top1 = 0.0

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

        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        nll_t = model.mixture_nll(target_u, mu, cov, mix_logits, component_nus)
        cov_proxy = model.expected_covariance(cov, mix_logits)
        idio_share_t = model.idio_share(cov, diag_per_component, mix_logits)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
        mix_probs = F.softmax(mix_logits, dim=-1)
        mix_entropy = -(mix_probs * mix_probs.clamp_min(1e-8).log()).sum(dim=-1)

        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(cov_proxy).mean()
        total_idio_share = total_idio_share + idio_share_t.mean()
        total_factor_scale = total_factor_scale + factor_scales.mean()
        total_mix_entropy = total_mix_entropy + mix_entropy.mean()
        total_attn_top1 = total_attn_top1 + attn_weights.max(dim=1).values.mean()

        pred_mean_path.append(mu_iv.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        use_pred = (
            self_feed_prob > 0.0
            and step < future_len - 1
            and torch.rand((), device=history_01.device).item() < self_feed_prob
        )
        next_frame = mu_iv.detach() if use_pred else target_t
        next_norm = normalize_iv(next_frame).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = next_frame

    pred_mean_path_t = torch.stack(pred_mean_path, dim=1)
    target_mean_path_t = torch.stack(target_mean_path, dim=1)
    path_mean_loss = F.smooth_l1_loss(pred_mean_path_t, target_mean_path_t)
    idio_share_mean = total_idio_share / future_len
    idio_share_penalty = F.relu(idio_share_mean - objective_config["idio_share_cap"]).pow(2)

    total_loss = (
        total_nll / future_len
        + objective_config["path_mean_weight"] * path_mean_loss
        + objective_config["idio_share_weight"] * idio_share_penalty
    )

    metrics = {
        "total_loss": total_loss,
        "multistep_nll": total_nll / future_len,
        "multistep_mae": total_mae / future_len,
        "pred_eff_rank": total_rank / future_len,
        "idio_share": idio_share_mean,
        "factor_scale_mean": total_factor_scale / future_len,
        "mix_entropy": total_mix_entropy / future_len,
        "path_mean_loss": path_mean_loss,
        "idio_share_penalty": idio_share_penalty,
        "attention_top1": total_attn_top1 / future_len,
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
        _loss, metrics = latent_factor_multistep_loss(
            model,
            history_01,
            future_01,
            objective_config=objective_config,
            self_feed_prob=0.0,
        )
        batch_size = history_01.shape[0]
        total_count += batch_size
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
    if total_count == 0:
        return {}
    return {f"val_{k}": v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_rollout_subset(
    model: LatentFactorInnovationARModel,
    val_loader: DataLoader,
    rollout_val_samples: int,
    rollout_eval_limit: int,
) -> dict[str, float]:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []

    gt_delta_sum = np.zeros(25, dtype=np.float64)
    gt_delta_outer = np.zeros((25, 25), dtype=np.float64)
    gt_delta_count = 0
    ro_delta_sum = np.zeros(25, dtype=np.float64)
    ro_delta_outer = np.zeros((25, 25), dtype=np.float64)
    ro_delta_count = 0

    for history_01, future_01 in val_loader:
        if total_count >= rollout_eval_limit:
            break
        if total_count + history_01.shape[0] > rollout_eval_limit:
            keep = rollout_eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(
            history_norm,
            n_samples=rollout_val_samples,
            n_steps=future_01.shape[1],
        )

        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = (
            (samples < model.support_lo) | (samples > model.support_hi)
        ).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        future_flat = future_01
        if future_flat.dim() == 4:
            future_flat = future_flat.reshape(future_flat.shape[0], future_flat.shape[1], -1)
        samples_flat = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)
        gt_delta = future_flat[:, 29, :] - future_flat[:, 28, :]
        gen_delta = samples_flat[:, :, 29, :] - samples_flat[:, :, 28, :]
        gen_flat = gen_delta.reshape(-1, 25)

        gt_np = gt_delta.detach().cpu().numpy()
        gen_np = gen_flat.detach().cpu().numpy()
        gt_delta_sum += gt_np.sum(axis=0)
        gt_delta_outer += gt_np.T @ gt_np
        gt_delta_count += gt_np.shape[0]
        ro_delta_sum += gen_np.sum(axis=0)
        ro_delta_outer += gen_np.T @ gen_np
        ro_delta_count += gen_np.shape[0]

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "rollout_cov90": float("nan"),
            "rollout_width90": float("nan"),
            "rollout_mae": float("nan"),
            "rollout_support_violation_rate": float("nan"),
            "rollout_turb_calm_ratio": float("nan"),
            "rollout_corr_ratio_h30": float("nan"),
            "rollout_rank_ratio_h30": float("nan"),
        }

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    if calm_mask.any() and turb_mask.any():
        turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item()
    else:
        turb_calm_ratio = float("nan")

    gt_mean = gt_delta_sum / max(gt_delta_count, 1)
    gt_cov = gt_delta_outer / max(gt_delta_count, 1) - np.outer(gt_mean, gt_mean)
    ro_mean = ro_delta_sum / max(ro_delta_count, 1)
    ro_cov = ro_delta_outer / max(ro_delta_count, 1) - np.outer(ro_mean, ro_mean)
    gt_corr = mean_offdiag_corr(gt_cov)
    ro_corr = mean_offdiag_corr(ro_cov)
    gt_rank = eff_rank_from_matrix(corr_from_cov(gt_cov))
    ro_rank = eff_rank_from_matrix(corr_from_cov(ro_cov))

    return {
        "rollout_cov90": total_cov / total_count,
        "rollout_width90": total_width / total_count,
        "rollout_mae": total_mae / total_count,
        "rollout_support_violation_rate": total_support_viol / total_count,
        "rollout_turb_calm_ratio": turb_calm_ratio,
        "rollout_corr_ratio_h30": float(ro_corr / gt_corr) if abs(gt_corr) > 1e-8 else float("nan"),
        "rollout_rank_ratio_h30": float(ro_rank / gt_rank) if gt_rank > 1e-8 else float("nan"),
    }


def maybe_load_partial_warm_start(model: LatentFactorInnovationARModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    skipped_prefixes = (
        "decoder.diag_head.",
    )
    loadable = {
        k: v
        for k, v in state.items()
        if k in current
        and current[k].shape == v.shape
        and not any(k.startswith(prefix) for prefix in skipped_prefixes)
    }
    model.load_state_dict(loadable, strict=False)
    print(f"Partial warm start loaded from {checkpoint_path}")
    print(f"  matched keys: {len(loadable)}")


def parse_component_nus(text: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if len(vals) == 0:
        raise ValueError("component_nus must be non-empty")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="193a: graph-aware AR latent-factor innovation mixture")
    parser.add_argument("--stage1_epochs", type=int, default=4)
    parser.add_argument("--stage2_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=5e-4)
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
    parser.add_argument("--self_feed_max", type=float, default=0.25)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
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
    print("193a: Graph-aware AR latent-factor innovation mixture")
    print("=" * 72)
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"Stage1 epochs: {args.stage1_epochs} | Stage2 epochs: {args.stage2_epochs}")
    print(f"Factors: {args.num_factors} | Components: {args.num_components}")
    print(f"Component nus: {component_nus}")

    best_score = float("inf")
    best_metrics = None
    history = []
    total_epochs = args.stage1_epochs + args.stage2_epochs

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        if epoch <= args.stage1_epochs:
            stage = 1
            self_feed_prob = 0.0
            lr = args.lr_stage1
        else:
            stage = 2
            stage2_idx = epoch - args.stage1_epochs
            stage2_den = max(args.stage2_epochs - 1, 1)
            self_feed_prob = args.self_feed_max * (stage2_idx - 1) / stage2_den
            lr = args.lr_stage2

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        train_sums: dict[str, float] = {}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = latent_factor_multistep_loss(
                model,
                history_01,
                future_01,
                objective_config=objective_config,
                self_feed_prob=self_feed_prob,
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
            + 0.25 * abs(rollout_metrics["rollout_cov90"] - 0.90)
            + 0.12 * max(0.0, 1.10 - rollout_metrics["rollout_turb_calm_ratio"])
            + 0.12 * max(0.0, 0.70 - rollout_metrics["rollout_corr_ratio_h30"])
            + 0.08 * max(0.0, rollout_metrics["rollout_rank_ratio_h30"] - 1.80)
            + 0.08 * max(0.0, rollout_metrics["rollout_width90"] - 0.22)
            + 0.35 * max(0.0, val_metrics["val_idio_share"] - args.idio_share_cap)
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
                        "type": "graph_ar_latent_factor_innovation_193a",
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
            "self_feed_prob": self_feed_prob,
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
            f"idio={val_metrics['val_idio_share']:.3f} "
            f"sf={self_feed_prob:.2f} "
            f"({elapsed:.1f}s)"
            + (" *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "score": history[-1]["selection_score"] if history else float("nan"),
        "config": {
            "type": "graph_ar_latent_factor_innovation_193a",
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
