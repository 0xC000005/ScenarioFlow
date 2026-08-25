#!/usr/bin/env python
"""
194a: Regime-switching AR latent-factor Student-t model.

Fresh AR model class after closing 193x:
  - true AR recursion over future time
  - explicit mean head
  - explicit latent-factor covariance head
  - sticky discrete latent regime path with exact teacher-forced marginalization
  - modest self-fed stage 2 only after local law is stable

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_194a_regime_switching_ar_latent_factor.py \
        --output_dir models/backfill/regime_switching_ar_latent_factor_194a --device cuda
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
from experiments.backfill.block_ar.train_193a_graph_ar_latent_factor_innovation import (
    corr_from_cov,
    eff_rank_from_matrix,
    mean_offdiag_corr,
)


def compute_cond_from_outputs(
    model: "RegimeSwitchingLatentFactorARModel",
    gru_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(pooled)
    cond = model.encoder.dropout(cond)
    return cond, attn_weights


class StickyRegimeShapeScaleDecoder(nn.Module):
    """State-conditioned version of the 169c shape/scale Student-t decoder."""

    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        rank: int = 5,
        num_states: int = 3,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.10,
        init_scale: float = 0.10,
        init_nu: float = 8.0,
        fixed_nu: float | None = None,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.rank = rank
        self.num_states = num_states
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.nu_floor = nu_floor
        self.nu_max = nu_max
        self.fixed_nu = fixed_nu

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
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn_norm": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "ff_norm": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
            )
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, rank)
        self.diag_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, 1)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        # New regime-specific parameters; base trunk stays warm-start compatible with 169c.
        self.state_cond_offsets = nn.Parameter(torch.zeros(num_states, cond_dim))
        self.state_factor_log_scale = nn.Parameter(torch.tensor([-0.20, 0.00, 0.20], dtype=torch.float32)[:num_states])
        self.state_diag_log_scale = nn.Parameter(torch.tensor([-0.10, 0.00, 0.15], dtype=torch.float32)[:num_states])
        self.state_scale_log_scale = nn.Parameter(torch.tensor([-0.45, 0.00, 0.40], dtype=torch.float32)[:num_states])
        if self.fixed_nu is None:
            self.state_nu_log_scale = nn.Parameter(torch.tensor([0.15, 0.00, -0.25], dtype=torch.float32)[:num_states])

        self._init_parameters(init_diag=init_diag, init_scale=init_scale, init_nu=init_nu)

    def _init_parameters(self, init_diag: float, init_scale: float, init_nu: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in ("mean_head", "factor_head", "diag_head", "scale_head", "nu_head")
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

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            inverse_softplus(max(init_scale - self.scale_floor, 1e-6)),
        )

        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(
                self.nu_head.bias,
                inverse_softplus(max(init_nu - self.nu_floor, 1e-6)),
            )

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = cond.shape[0]
        num_states = self.num_states

        state_cond = cond.unsqueeze(1) + self.state_cond_offsets.unsqueeze(0)
        cond_flat = state_cond.reshape(batch_size * num_states, -1)
        prev_flat = prev_u.unsqueeze(1).expand(-1, num_states, -1).reshape(batch_size * num_states, -1)

        h = self.input_proj(prev_flat.unsqueeze(-1))
        h = h + self.cond_proj(cond_flat).unsqueeze(1)
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

        mu = prev_flat + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)

        mu = mu.view(batch_size, num_states, self.n_cells)
        factor = factor.view(batch_size, num_states, self.n_cells, self.rank)
        diag = diag.view(batch_size, num_states, self.n_cells)
        scale = scale.view(batch_size, num_states)
        nu = nu.view(batch_size, num_states)

        factor = factor * self.state_factor_log_scale.exp().view(1, num_states, 1, 1)
        diag = diag * self.state_diag_log_scale.exp().view(1, num_states, 1)
        scale = scale * self.state_scale_log_scale.exp().view(1, num_states)
        if self.fixed_nu is None:
            nu = torch.clamp(
                nu * self.state_nu_log_scale.exp().view(1, num_states),
                min=self.nu_floor + 1e-6,
                max=self.nu_max,
            )

        return mu, factor, diag, scale, nu


class RegimeSwitchingLatentFactorARModel(nn.Module):
    """GRU history encoder + sticky discrete-state latent-factor Student-t emissions."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        regime_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = StickyRegimeShapeScaleDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.regime_config = regime_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter
        self.num_states = int(regime_config["num_states"])

        cond_dim = encoder_config.bottleneck_dim
        self.init_head = nn.Linear(cond_dim, self.num_states)
        self.transition_context_head = nn.Linear(cond_dim, self.num_states)
        self.transition_base = nn.Parameter(torch.zeros(self.num_states, self.num_states))
        sticky_init = regime_config.get("sticky_init", 1.75)
        self.sticky_bias = nn.Parameter(torch.full((self.num_states,), float(sticky_init)))

        nn.init.zeros_(self.init_head.weight)
        nn.init.zeros_(self.init_head.bias)
        nn.init.zeros_(self.transition_context_head.weight)
        nn.init.zeros_(self.transition_context_head.bias)

    def encode(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_norm = normalize_iv(history_01)
        output, _ = self.encoder.gru(history_norm.reshape(history_norm.shape[0], history_norm.shape[1], -1))
        attn_logits = self.encoder.attn_proj(output).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * output).sum(dim=1)
        cond = self.encoder.bottleneck(pooled)
        cond = self.encoder.dropout(cond)
        return cond, attn_weights, output

    def initial_logits(self, cond: torch.Tensor) -> torch.Tensor:
        return self.init_head(cond)

    def transition_logits(self, cond: torch.Tensor) -> torch.Tensor:
        logits = self.transition_base.unsqueeze(0) + self.transition_context_head(cond).unsqueeze(1)
        eye = torch.eye(self.num_states, device=cond.device, dtype=cond.dtype).unsqueeze(0)
        return logits + eye * self.sticky_bias.view(1, self.num_states)

    def normalized_components(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm, avg_var

    def covariance(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        cov = torch.einsum("bknr,bkmr->bknm", factor_norm, factor_norm)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def predictive_covariance(
        self,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        cov = self.covariance(factor, diag, scale)
        var_scale = torch.clamp(nu / (nu - 2.0), min=1.0, max=10.0)
        return cov * var_scale.unsqueeze(-1).unsqueeze(-1)

    def student_t_logprob_states(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)
        diff = (target_u.unsqueeze(1) - mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        d = target_u.shape[-1]
        pi = target_u.new_tensor(math.pi)
        nu = nu.clamp_min(self.decoder.nu_floor + 1e-6)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return log_norm + log_kernel

    def state_params_from_history(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond, attn_weights, _ = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        mu, factor, diag, scale, nu = self.decoder(cond, prev_u)
        init_probs = F.softmax(self.initial_logits(cond), dim=-1)
        return mu, factor, diag, scale, nu, init_probs

    def forward_from_history(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, factor, diag, scale, nu, init_probs = self.state_params_from_history(history_01)
        mu_m = torch.einsum("bk,bkn->bn", init_probs, mu)
        factor_m = torch.einsum("bk,bknr->bnr", init_probs, factor)
        diag_m = torch.einsum("bk,bkn->bn", init_probs, diag)
        scale_m = torch.einsum("bk,bk->b", init_probs, scale)
        nu_m = torch.einsum("bk,bk->b", init_probs, nu)
        return mu_m, factor_m, diag_m, scale_m, nu_m

    def _gather_state_params(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_draws = state_idx.shape
        cell_dim = mu.shape[-1]
        rank = factor.shape[-1]
        gather_idx = state_idx.unsqueeze(-1)
        mu_sel = mu.unsqueeze(1).expand(-1, n_draws, -1, -1).gather(
            2, gather_idx.unsqueeze(-1).expand(-1, -1, 1, cell_dim)
        ).squeeze(2)
        diag_sel = diag.unsqueeze(1).expand(-1, n_draws, -1, -1).gather(
            2, gather_idx.unsqueeze(-1).expand(-1, -1, 1, cell_dim)
        ).squeeze(2)
        factor_sel = factor.unsqueeze(1).expand(-1, n_draws, -1, -1, -1).gather(
            2, gather_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, cell_dim, rank)
        ).squeeze(2)
        scale_sel = scale.unsqueeze(1).expand(-1, n_draws, -1).gather(2, gather_idx).squeeze(-1)
        nu_sel = nu.unsqueeze(1).expand(-1, n_draws, -1).gather(2, gather_idx).squeeze(-1)
        return mu_sel, factor_sel, diag_sel, scale_sel, nu_sel

    def sample_next_u(
        self,
        history_01: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        mu, factor, diag, scale, nu, init_probs = self.state_params_from_history(history_01)
        state_idx = torch.multinomial(init_probs, n_samples, replacement=True)
        mu_sel, factor_sel, diag_sel, scale_sel, nu_sel = self._gather_state_params(
            mu, factor, diag, scale, nu, state_idx
        )

        factor_norm, diag_norm, _ = self.normalized_components(factor_sel, diag_sel)
        eps_factor = torch.randn(
            history_01.shape[0], n_samples, factor_sel.shape[-1], device=history_01.device, dtype=history_01.dtype
        )
        eps_diag = torch.randn(
            history_01.shape[0], n_samples, mu.shape[-1], device=history_01.device, dtype=history_01.dtype
        )
        lowrank_noise = torch.einsum("bsnr,bsr->bsn", factor_norm, eps_factor)
        diag_noise = diag_norm * eps_diag
        gamma = torch.distributions.Gamma(nu_sel / 2.0, nu_sel / 2.0)
        mix = gamma.sample().to(mu_sel.dtype).clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        total_noise = (lowrank_noise + diag_noise) * scale_sel.unsqueeze(-1)
        return mu_sel + total_noise * t_scale

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        return unconstrained_to_iv(
            self.sample_next_u(history_01, n_samples=n_samples),
            lo=self.support_lo,
            hi=self.support_hi,
        )

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
            prev_state = None

            frames = []
            for _ in range(n_steps):
                cond, _, _ = self.encode(hist_k)
                prev_01 = hist_k[:, -1].reshape(hist_k.shape[0], -1)
                prev_u = iv_to_unconstrained(
                    prev_01,
                    lo=self.support_lo,
                    hi=self.support_hi,
                    eps=self.support_eps,
                )
                mu, factor, diag, scale, nu = self.decoder(cond, prev_u)

                if prev_state is None:
                    probs = F.softmax(self.initial_logits(cond), dim=-1)
                else:
                    trans_probs = F.softmax(self.transition_logits(cond), dim=-1)
                    probs = trans_probs[torch.arange(hist_k.shape[0], device=hist_k.device), prev_state]

                state_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
                gather = state_idx.view(-1, 1, 1)
                mu_sel = mu.gather(1, gather.expand(-1, 1, mu.shape[-1])).squeeze(1)
                diag_sel = diag.gather(1, gather.expand(-1, 1, diag.shape[-1])).squeeze(1)
                factor_sel = factor.gather(
                    1, gather.unsqueeze(-1).expand(-1, 1, factor.shape[-2], factor.shape[-1])
                ).squeeze(1)
                scale_sel = scale.gather(1, state_idx.view(-1, 1)).squeeze(1)
                nu_sel = nu.gather(1, state_idx.view(-1, 1)).squeeze(1)

                factor_norm, diag_norm, _ = self.normalized_components(
                    factor_sel.unsqueeze(1), diag_sel.unsqueeze(1)
                )
                factor_norm = factor_norm.squeeze(1)
                diag_norm = diag_norm.squeeze(1)

                eps_factor = torch.randn(
                    hist_k.shape[0], factor_sel.shape[-1], device=hist_k.device, dtype=hist_k.dtype
                )
                eps_diag = torch.randn(
                    hist_k.shape[0], mu_sel.shape[-1], device=hist_k.device, dtype=hist_k.dtype
                )
                lowrank_noise = torch.einsum("bnr,br->bn", factor_norm, eps_factor)
                diag_noise = diag_norm * eps_diag
                gamma = torch.distributions.Gamma(nu_sel / 2.0, nu_sel / 2.0)
                mix = gamma.sample().to(mu_sel.dtype).clamp_min(1e-6)
                t_scale = torch.rsqrt(mix).unsqueeze(-1)
                total_noise = (lowrank_noise + diag_noise) * scale_sel.unsqueeze(-1)
                next_u = mu_sel + total_noise * t_scale
                next_iv = unconstrained_to_iv(next_u, lo=self.support_lo, hi=self.support_hi)

                frames.append(next_iv.reshape(batch_size, k, 5, 5))
                hist_k = torch.cat([hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)], dim=1)
                prev_state = state_idx

            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def forward_backward_posteriors(
    log_init: torch.Tensor,
    log_trans: list[torch.Tensor],
    log_emit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, future_len, num_states = log_emit.shape

    log_alpha = []
    alpha_t = log_init + log_emit[:, 0]
    log_alpha.append(alpha_t)
    for step in range(1, future_len):
        alpha_t = log_emit[:, step] + torch.logsumexp(
            log_alpha[-1].unsqueeze(-1) + log_trans[step - 1], dim=1
        )
        log_alpha.append(alpha_t)
    log_alpha_t = torch.stack(log_alpha, dim=1)
    loglik = torch.logsumexp(log_alpha_t[:, -1], dim=-1)

    log_beta: list[torch.Tensor] = [torch.zeros(batch_size, num_states, device=log_emit.device, dtype=log_emit.dtype)]
    for step in range(future_len - 2, -1, -1):
        beta_t = torch.logsumexp(
            log_trans[step]
            + log_emit[:, step + 1].unsqueeze(1)
            + log_beta[0].unsqueeze(1),
            dim=2,
        )
        log_beta.insert(0, beta_t)
    log_beta_t = torch.stack(log_beta, dim=1)

    gamma = torch.exp(log_alpha_t + log_beta_t - loglik.view(-1, 1, 1))
    xi_list = []
    for step in range(future_len - 1):
        log_xi = (
            log_alpha_t[:, step].unsqueeze(-1)
            + log_trans[step]
            + log_emit[:, step + 1].unsqueeze(1)
            + log_beta_t[:, step + 1].unsqueeze(1)
            - loglik.view(-1, 1, 1)
        )
        xi_list.append(torch.exp(log_xi))
    if xi_list:
        xi = torch.stack(xi_list, dim=1)
    else:
        xi = torch.empty(batch_size, 0, num_states, num_states, device=log_emit.device, dtype=log_emit.dtype)
    return gamma, xi, loglik


def regime_sequence_objective(
    model: RegimeSwitchingLatentFactorARModel,
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

    log_emit_steps = []
    log_trans_steps: list[torch.Tensor] = []
    mu_states_u = []
    mu_states_iv = []
    pred_cov_states = []
    scale_states = []
    nu_states = []
    attn_top1_sum = 0.0
    filtered_probs = None
    transition_entropy_sum = 0.0

    for step in range(future_len):
        cond, attn_weights = compute_cond_from_outputs(model, gru_outputs)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mu, factor, diag, scale, nu = model.decoder(cond, prev_u)
        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )

        log_emit_t = model.student_t_logprob_states(target_u, mu, factor, diag, scale, nu)
        log_emit_steps.append(log_emit_t)
        mu_states_u.append(mu)
        mu_states_iv.append(unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi))
        pred_cov_states.append(model.predictive_covariance(factor, diag, scale, nu))
        scale_states.append(scale)
        nu_states.append(nu)
        attn_top1_sum = attn_top1_sum + attn_weights.max(dim=1).values.mean()

        if step == 0:
            log_alpha_t = F.log_softmax(model.initial_logits(cond), dim=-1) + log_emit_t
        else:
            trans_logits_t = model.transition_logits(cond)
            log_trans_t = F.log_softmax(trans_logits_t, dim=-1)
            log_trans_steps.append(log_trans_t)
            log_alpha_t = log_emit_t + torch.logsumexp(log_alpha_t.unsqueeze(-1) + log_trans_t, dim=1)
            trans_probs_t = F.softmax(trans_logits_t, dim=-1)
            if filtered_probs is not None:
                filtered_self = (
                    filtered_probs.unsqueeze(-1) * trans_probs_t * torch.eye(
                        model.num_states, device=trans_probs_t.device, dtype=trans_probs_t.dtype
                    ).unsqueeze(0)
                ).sum(dim=(1, 2))
                trans_entropy = -(trans_probs_t * trans_probs_t.clamp_min(1e-8).log()).sum(dim=-1)
                transition_entropy_sum = transition_entropy_sum + (
                    filtered_probs * trans_entropy
                ).sum(dim=-1).mean()

        filtered_probs = F.softmax(log_alpha_t, dim=-1)
        if step == 0:
            mu_filtered_iv = torch.einsum("bk,bkn->bn", filtered_probs, mu_states_iv[-1])
        else:
            mu_filtered_iv = torch.einsum("bk,bkn->bn", filtered_probs, mu_states_iv[-1])

        use_pred = (
            self_feed_prob > 0.0
            and step < future_len - 1
            and torch.rand((), device=history_01.device).item() < self_feed_prob
        )
        next_frame = mu_filtered_iv.detach() if use_pred else target_t
        next_norm = normalize_iv(next_frame).unsqueeze(1)
        next_out, gru_state = model.encoder.gru(next_norm, gru_state)
        gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
        prev_01 = next_frame

    # Recompute initial logits from the original history for exact sequence marginalization.
    init_cond, _, _ = model.encode(history_01)
    log_init = F.log_softmax(model.initial_logits(init_cond), dim=-1)

    log_emit = torch.stack(log_emit_steps, dim=1)
    gamma, xi, loglik = forward_backward_posteriors(log_init, log_trans_steps, log_emit)

    mu_states_u_t = torch.stack(mu_states_u, dim=1)
    mu_states_iv_t = torch.stack(mu_states_iv, dim=1)
    pred_cov_states_t = torch.stack(pred_cov_states, dim=1)
    scale_states_t = torch.stack(scale_states, dim=1)
    nu_states_t = torch.stack(nu_states, dim=1)

    gamma_u = gamma.unsqueeze(-1)
    mu_bar_u = (gamma_u * mu_states_u_t).sum(dim=2)
    mu_bar_iv = (gamma_u * mu_states_iv_t).sum(dim=2)

    pred_var_cells = torch.diagonal(pred_cov_states_t, dim1=-2, dim2=-1)
    state_mean_dev = (mu_states_u_t - mu_bar_u.unsqueeze(2)).pow(2)
    mixture_var_cells = (gamma_u * (pred_var_cells + state_mean_dev)).sum(dim=2)

    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    resid2 = (target_u - mu_bar_u).detach().pow(2)
    sharpness_penalty = F.relu(
        mixture_var_cells
        - (objective_config["sharpness_ratio"] * resid2 + objective_config["sharpness_floor"])
    ).mean()

    occ = gamma.mean(dim=(0, 1))
    dominant_occ = occ.max()
    occupancy_penalty = F.relu(dominant_occ - objective_config["occupancy_cap"]).pow(2)

    if xi.shape[1] > 0:
        expected_self = torch.diagonal(xi, dim1=-2, dim2=-1).sum(dim=-1).sum(dim=-1)
        expected_self = expected_self / xi.sum(dim=(1, 2, 3)).clamp_min(1e-8)
        self_transition_mean = expected_self.mean()
    else:
        self_transition_mean = history_01.new_tensor(1.0)
    self_transition_penalty = F.relu(
        objective_config["self_transition_min"] - self_transition_mean
    ).pow(2)

    if future_len > 1:
        transition_entropy_mean = transition_entropy_sum / (future_len - 1)
    else:
        transition_entropy_mean = history_01.new_tensor(0.0)
    transition_entropy_penalty = F.relu(
        objective_config["transition_entropy_min"] - transition_entropy_mean
    ).pow(2)

    path_mean_loss = F.smooth_l1_loss(
        mu_bar_iv.mean(dim=-1),
        future_01.mean(dim=-1),
    )

    nll = -loglik.mean()
    total_loss = (
        nll
        + objective_config["path_mean_weight"] * path_mean_loss
        + objective_config["sharpness_weight"] * sharpness_penalty
        + objective_config["occupancy_weight"] * occupancy_penalty
        + objective_config["self_transition_weight"] * self_transition_penalty
        + objective_config["transition_entropy_weight"] * transition_entropy_penalty
    )

    expected_cov = (
        gamma.unsqueeze(-1).unsqueeze(-1) * pred_cov_states_t
    ).sum(dim=2)
    metrics = {
        "total_loss": total_loss,
        "multistep_nll": nll,
        "multistep_mae": (mu_bar_iv - future_01).abs().mean(),
        "pred_eff_rank": effective_rank(expected_cov.reshape(-1, n_cells, n_cells)).mean(),
        "scale_mean": scale_states_t.mean(),
        "nu_mean": nu_states_t.mean(),
        "path_mean_loss": path_mean_loss,
        "sharpness_penalty": sharpness_penalty,
        "state_dominance": dominant_occ,
        "self_transition_mean": self_transition_mean,
        "transition_entropy": transition_entropy_mean,
        "occupancy_penalty": occupancy_penalty,
        "self_transition_penalty": self_transition_penalty,
        "transition_entropy_penalty": transition_entropy_penalty,
        "attention_top1": attn_top1_sum / future_len,
        "pred_total_var_per_cell": mixture_var_cells.mean(),
        "state0_occ": occ[0],
        "state1_occ": occ[1] if model.num_states > 1 else occ[0].new_tensor(0.0),
        "state2_occ": occ[2] if model.num_states > 2 else occ[0].new_tensor(0.0),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: RegimeSwitchingLatentFactorARModel,
    val_loader: DataLoader,
    objective_config: dict,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = regime_sequence_objective(
            model,
            history_01,
            future_01,
            objective_config=objective_config,
            self_feed_prob=0.0,
        )
        batch_size = history_01.shape[0]
        total_count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {f"val_{k}": v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_rollout_subset(
    model: RegimeSwitchingLatentFactorARModel,
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
        support_viol = ((samples < model.support_lo) | (samples > model.support_hi)).float().mean()

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


def maybe_load_partial_warm_start(
    model: RegimeSwitchingLatentFactorARModel,
    checkpoint_path: str | None,
) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    loadable = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(loadable, strict=False)
    print(f"Partial warm start loaded from {checkpoint_path}")
    print(f"  matched keys: {len(loadable)}")


def build_objective_config(args: argparse.Namespace) -> dict[str, float]:
    return dict(
        path_mean_weight=args.path_mean_weight,
        sharpness_weight=args.sharpness_weight,
        sharpness_ratio=args.sharpness_ratio,
        sharpness_floor=args.sharpness_floor,
        occupancy_weight=args.occupancy_weight,
        occupancy_cap=args.occupancy_cap,
        self_transition_weight=args.self_transition_weight,
        self_transition_min=args.self_transition_min,
        transition_entropy_weight=args.transition_entropy_weight,
        transition_entropy_min=args.transition_entropy_min,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="194a: regime-switching AR latent-factor Student-t")
    parser.add_argument("--stage1_epochs", type=int, default=4)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_stage1", type=float, default=8e-4)
    parser.add_argument("--lr_stage2", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--num_states", type=int, default=3)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=0.0)
    parser.add_argument("--sticky_init", type=float, default=1.75)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--path_mean_weight", type=float, default=0.20)
    parser.add_argument("--sharpness_weight", type=float, default=0.40)
    parser.add_argument("--sharpness_ratio", type=float, default=1.10)
    parser.add_argument("--sharpness_floor", type=float, default=0.015)
    parser.add_argument("--occupancy_weight", type=float, default=0.20)
    parser.add_argument("--occupancy_cap", type=float, default=0.88)
    parser.add_argument("--self_transition_weight", type=float, default=0.10)
    parser.add_argument("--self_transition_min", type=float, default=0.72)
    parser.add_argument("--transition_entropy_weight", type=float, default=0.05)
    parser.add_argument("--transition_entropy_min", type=float, default=0.55)
    parser.add_argument("--stage2_self_feed_max", type=float, default=0.25)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    fixed_nu = None if args.fixed_nu <= 0.0 else args.fixed_nu
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
        rank=args.rank,
        num_states=args.num_states,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.diag_floor if args.diag_floor > 0.10 else 0.10,
        init_scale=0.10,
        init_nu=args.nu_init,
        fixed_nu=fixed_nu,
    )
    regime_config = dict(
        num_states=args.num_states,
        sticky_init=args.sticky_init,
    )
    model = RegimeSwitchingLatentFactorARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        regime_config=regime_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_partial_warm_start(model, args.warm_start)

    objective_config = build_objective_config(args)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_reg = sum(p.numel() for n, p in model.named_parameters() if n.startswith("init_head") or n.startswith("transition_") or n.startswith("sticky_bias"))
    print(f"\n{'=' * 64}")
    print("194a: Regime-Switching AR Latent-Factor Student-t")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Regime params:  {n_reg:,}")
    print(f"  Total params:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  States={args.num_states} | rank={args.rank} | d_model={args.d_model}")
    print("  Objective: exact teacher-forced discrete-state marginal likelihood")
    print("  Stage 2: weak self-fed continuation only")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay)
    total_epochs = args.stage1_epochs + args.stage2_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))

    best_score = float("inf")
    best_metrics: dict[str, float] | None = None
    history = []

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        ep_totals: dict[str, float] = {}
        nb = 0

        if epoch == args.stage1_epochs + 1:
            for group in optimizer.param_groups:
                group["lr"] = args.lr_stage2

        if epoch <= args.stage1_epochs:
            self_feed_prob = 0.0
        else:
            if args.stage2_epochs <= 0:
                self_feed_prob = 0.0
            else:
                frac = (epoch - args.stage1_epochs) / max(args.stage2_epochs, 1)
                self_feed_prob = args.stage2_self_feed_max * frac

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = regime_sequence_objective(
                model,
                history_01,
                future_01,
                objective_config=objective_config,
                self_feed_prob=self_feed_prob,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key, value in metrics.items():
                ep_totals[key] = ep_totals.get(key, 0.0) + float(value.item())
            nb += 1

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep_totals.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )

        selection_score = (
            val_metrics["val_multistep_nll"]
            + 0.18 * max(0.0, rollout_metrics["rollout_cov90"] - 0.84)
            + 0.12 * max(0.0, rollout_metrics["rollout_width90"] - 0.13)
            + 0.12 * max(0.0, 1.15 - rollout_metrics["rollout_turb_calm_ratio"])
            + 0.06 * max(0.0, val_metrics["val_sharpness_penalty"] - 0.02)
            + 0.04 * max(0.0, val_metrics["val_state_dominance"] - 0.88)
            + 0.04 * max(0.0, 0.72 - val_metrics["val_self_transition_mean"])
            + 0.06 * max(0.0, 0.80 - rollout_metrics["rollout_corr_ratio_h30"])
            + 0.05 * max(0.0, rollout_metrics["rollout_rank_ratio_h30"] - 1.90)
        )

        elapsed = time.time() - t0
        epoch_metrics = {
            "epoch": epoch,
            "stage": 1 if epoch <= args.stage1_epochs else 2,
            "self_feed_prob": self_feed_prob,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
            "selection_score": selection_score,
            "epoch_time_sec": elapsed,
        }
        history.append(make_serializable(epoch_metrics))

        if selection_score < best_score:
            best_score = selection_score
            best_metrics = {**val_metrics, **rollout_metrics, "selection_score": best_score}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": {
                        "type": "regime_switching_ar_latent_factor_194a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "regime": regime_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                    },
                    "metrics": make_serializable(best_metrics),
                },
                output_dir / "best_model.pt",
            )

        print(
            f"Epoch {epoch:02d}/{total_epochs} | stage={'1' if epoch <= args.stage1_epochs else '2'} "
            f"| train_nll={train_metrics['train_multistep_nll']:.4f} "
            f"| val_nll={val_metrics['val_multistep_nll']:.4f} "
            f"| cov90={rollout_metrics['rollout_cov90']:.3f} "
            f"| width90={rollout_metrics['rollout_width90']:.3f} "
            f"| turb/calm={rollout_metrics['rollout_turb_calm_ratio']:.3f} "
            f"| occ={val_metrics['val_state0_occ']:.3f}/{val_metrics['val_state1_occ']:.3f}/{val_metrics['val_state2_occ']:.3f} "
            f"| dom={val_metrics['val_state_dominance']:.3f} "
            f"| self={val_metrics['val_self_transition_mean']:.3f} "
            f"| sharp={val_metrics['val_sharpness_penalty']:.4f} "
            f"| roll_corr={rollout_metrics['rollout_corr_ratio_h30']:.3f} "
            f"| roll_rank={rollout_metrics['rollout_rank_ratio_h30']:.3f} "
            f"| score={selection_score:.4f} "
            f"| {elapsed:.1f}s"
        )

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": total_epochs,
            "config": {
                "type": "regime_switching_ar_latent_factor_194a",
                "encoder": vars(encoder_config),
                "decoder": decoder_config,
                "regime": regime_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
            },
            "metrics": make_serializable(best_metrics or {}),
        },
        output_dir / "final_model.pt",
    )

    print("\nTraining complete.")
    if best_metrics is not None:
        print(
            f"Best selection score={best_metrics['selection_score']:.4f} | "
            f"roll_cov90={best_metrics['rollout_cov90']:.3f} | "
            f"roll_turb_calm={best_metrics['rollout_turb_calm_ratio']:.3f} | "
            f"roll_corr_h30={best_metrics['rollout_corr_ratio_h30']:.3f} | "
            f"roll_rank_h30={best_metrics['rollout_rank_ratio_h30']:.3f}"
        )


if __name__ == "__main__":
    main()
