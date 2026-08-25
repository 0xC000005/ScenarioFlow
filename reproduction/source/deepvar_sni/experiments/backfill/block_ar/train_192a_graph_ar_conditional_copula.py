#!/usr/bin/env python
"""
192a: Graph-aware AR conditional copula model.

Fresh AR model class:
  - true AR recursion over future time
  - explicit conditional mean head
  - explicit conditional covariance head
  - conditional autoregressive flow over whitened residual innovations
  - richer history-token conditioning via cross-attention

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_192a_graph_ar_conditional_copula.py \
        --output_dir models/backfill/graph_ar_conditional_copula_192a --device cuda
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

from diffusion.block_ar.gru_encoder import EncoderConfig
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
    evaluate_rollout_subset,
)


class HistoryTokenEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, token_dim: int, decoder_state_dim: int):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_dim + config.extra_features,
            hidden_size=config.gru_hidden_dim,
            batch_first=True,
        )
        self.token_proj = nn.Linear(config.gru_hidden_dim, token_dim)
        self.attn_proj = nn.Linear(token_dim, 1)
        self.global_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.init_state_proj = nn.Sequential(
            nn.Linear(token_dim * 2, decoder_state_dim),
            nn.SiLU(),
            nn.Linear(decoder_state_dim, decoder_state_dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.cond_aug_sigma = config.cond_aug_sigma

    def encode_norm(
        self,
        history_norm_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, _ = self.gru(history_norm_flat)
        tokens = self.token_proj(output)
        tokens = self.dropout(tokens)

        if self.training and self.cond_aug_sigma > 0.0:
            tokens = tokens + self.cond_aug_sigma * torch.randn_like(tokens)

        attn_logits = self.attn_proj(tokens).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * tokens).sum(dim=1)
        pooled = self.global_proj(pooled)
        last = tokens[:, -1]
        decoder_state = torch.tanh(self.init_state_proj(torch.cat([pooled, last], dim=-1)))
        return tokens, pooled, decoder_state


class NodeTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ls_attn = nn.Parameter(torch.ones(d_model) * 0.1)
        self.ls_ff = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.ls_attn * attn_out
        x = x + self.ls_ff * self.ff(self.ff_norm(x))
        return x


class GraphARStepDecoder(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        cond_dim: int = 128,
        token_dim: int = 128,
        decoder_state_dim: int = 128,
        rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        init_diag: float = 0.10,
        init_scale: float = 0.10,
        flow_context_dim: int = 160,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.rank = rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor

        self.query_proj = nn.Linear(decoder_state_dim, token_dim)
        self.history_attn = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.cond_merge = nn.Sequential(
            nn.Linear(decoder_state_dim + token_dim + cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.node_embed = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.layers = nn.ModuleList([NodeTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, rank)
        self.diag_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.flow_context_head = nn.Sequential(
            nn.Linear(d_model + cond_dim + n_cells, flow_context_dim),
            nn.SiLU(),
            nn.Linear(flow_context_dim, flow_context_dim),
        )

        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(head_name in name for head_name in ("mean_head", "factor_head", "diag_head", "scale_head")):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.factor_head.bias, mean=0.0, std=1e-3)
        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(self.diag_head.bias, inverse_softplus(max(init_diag - self.diag_floor, 1e-6)))

        last = self.scale_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, inverse_softplus(max(init_scale - self.scale_floor, 1e-6)))

    def forward(
        self,
        prev_u: torch.Tensor,
        decoder_state: torch.Tensor,
        history_tokens: torch.Tensor,
        pooled_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_proj(decoder_state).unsqueeze(1)
        attn_ctx, attn_weights = self.history_attn(query, history_tokens, history_tokens)
        attn_ctx = attn_ctx.squeeze(1)
        cond = self.cond_merge(torch.cat([decoder_state, attn_ctx, pooled_history], dim=-1))

        h = self.input_proj(prev_u.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.node_embed
        for layer in self.layers:
            h = layer(h)
        h = self.out_norm(h)
        pooled_nodes = h.mean(dim=1)

        mu = prev_u + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(torch.cat([pooled_nodes, cond], dim=-1)).squeeze(-1)) + self.scale_floor
        flow_context = self.flow_context_head(torch.cat([pooled_nodes, cond, prev_u], dim=-1))
        attn_top1 = attn_weights.squeeze(1).max(dim=-1).values
        return mu, factor, diag, scale, flow_context, attn_top1


class ConditionalAutoregressiveLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        permutation: torch.Tensor,
        scale_clip: float = 1.5,
        dim_embed_dim: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.scale_clip = scale_clip
        self.register_buffer("permutation", permutation.long())
        self.register_buffer("inv_permutation", torch.argsort(permutation).long())
        prefix_masks = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer("prefix_masks", prefix_masks[:, permutation])

        self.dim_embed = nn.Embedding(dim, dim_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim + dim_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        for module in self.net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _shift_logscale(
        self,
        prev_full: torch.Tensor,
        context: torch.Tensor,
        dim_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim_token = self.dim_embed(self.permutation[dim_idx]).unsqueeze(0).expand(prev_full.shape[0], -1)
        h = torch.cat([prev_full, context, dim_token], dim=-1)
        shift, log_scale = self.net(h).chunk(2, dim=-1)
        shift = shift.squeeze(-1)
        log_scale = torch.tanh(log_scale.squeeze(-1)) * self.scale_clip
        return shift, log_scale

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_perm = x[:, self.permutation]
        z_perm = torch.zeros_like(x_perm)
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for idx in range(self.dim):
            prev_full = x_perm * self.prefix_masks[idx].unsqueeze(0)
            shift, log_scale = self._shift_logscale(prev_full, context, idx)
            z_perm[:, idx] = (x_perm[:, idx] - shift) * torch.exp(-log_scale)
            total_logdet = total_logdet - log_scale
        z = z_perm[:, self.inv_permutation]
        return z, total_logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_perm = z[:, self.permutation]
        x_perm = torch.zeros_like(z_perm)
        total_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for idx in range(self.dim):
            prev_full = x_perm * self.prefix_masks[idx].unsqueeze(0)
            shift, log_scale = self._shift_logscale(prev_full, context, idx)
            x_perm[:, idx] = shift + torch.exp(log_scale) * z_perm[:, idx]
            total_logdet = total_logdet + log_scale
        x = x_perm[:, self.inv_permutation]
        return x, total_logdet


class ConditionalAutoregressiveFlow(nn.Module):
    def __init__(
        self,
        dim: int = 25,
        context_dim: int = 160,
        hidden_dim: int = 96,
        n_layers: int = 1,
        scale_clip: float = 1.5,
        dim_embed_dim: int = 16,
        base_nu_init: float = 8.0,
        base_nu_floor: float = 2.1,
        base_nu_max: float = 30.0,
    ):
        super().__init__()
        self.dim = dim
        self.base_nu_floor = base_nu_floor
        self.base_nu_max = base_nu_max
        perms = [torch.arange(dim)]
        if n_layers > 1:
            perms.append(torch.arange(dim - 1, -1, -1))
        while len(perms) < n_layers:
            perms.append(torch.randperm(dim))
        self.layers = nn.ModuleList([
            ConditionalAutoregressiveLayer(
                dim=dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                permutation=perm,
                scale_clip=scale_clip,
                dim_embed_dim=dim_embed_dim,
            )
            for perm in perms[:n_layers]
        ])
        self.base_nu_unconstrained = nn.Parameter(
            torch.tensor(inverse_softplus(max(base_nu_init - base_nu_floor, 1e-6)), dtype=torch.float32)
        )

    def base_nu(self) -> torch.Tensor:
        nu = F.softplus(self.base_nu_unconstrained) + self.base_nu_floor
        return torch.clamp(nu, max=self.base_nu_max)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, logdet = layer.forward(z, context)
            total_logdet = total_logdet + logdet
        return z, total_logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, context)
            total_logdet = total_logdet + logdet
        return x, total_logdet

    def base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        nu = self.base_nu().to(z.dtype)
        pi = z.new_tensor(math.pi)
        log_norm = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0) - 0.5 * torch.log(nu * pi)
        log_kernel = -0.5 * (nu + 1.0) * torch.log1p(z.pow(2) / nu)
        return (log_norm + log_kernel).sum(dim=-1)

    def sample_base(self, batch_size: int, n_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        nu = self.base_nu().to(dtype=dtype, device=device)
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample((batch_size * n_samples, self.dim)).to(dtype).clamp_min(1e-6)
        eps = torch.randn(batch_size * n_samples, self.dim, device=device, dtype=dtype)
        return eps * torch.rsqrt(mix)


class GraphARConditionalCopulaModel(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        token_dim = decoder_config["token_dim"]
        decoder_state_dim = decoder_config["decoder_state_dim"]
        self.history_encoder = HistoryTokenEncoder(
            config=encoder_config,
            token_dim=token_dim,
            decoder_state_dim=decoder_state_dim,
        )
        self.step_state = nn.GRUCell(encoder_config.input_dim, decoder_state_dim)
        self.decoder = GraphARStepDecoder(**decoder_config)
        self.flow = ConditionalAutoregressiveFlow(**flow_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.flow_config = flow_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode_history(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_norm = normalize_iv(history_01).reshape(history_01.shape[0], history_01.shape[1], -1)
        return self.history_encoder.encode_norm(history_norm)

    def update_state(self, prev_frame_01: torch.Tensor, decoder_state: torch.Tensor) -> torch.Tensor:
        prev_norm = normalize_iv(prev_frame_01).reshape(prev_frame_01.shape[0], -1)
        return self.step_state(prev_norm, decoder_state)

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
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def decode_step(
        self,
        prev_01: torch.Tensor,
        decoder_state: torch.Tensor,
        history_tokens: torch.Tensor,
        pooled_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_u = iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        return self.decoder(prev_u, decoder_state, history_tokens, pooled_history)

    def forward_from_history(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tokens, pooled_history, decoder_state = self.encode_history(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        decoder_state = self.update_state(prev_01, decoder_state)
        mu, factor, diag, scale, flow_context, _attn_top1 = self.decode_step(
            prev_01, decoder_state, history_tokens, pooled_history
        )
        return mu, factor, diag, scale, flow_context

    def log_prob_step(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        flow_context: torch.Tensor,
        flow_active: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)

        diff = (target_u - mu).unsqueeze(-1)
        white = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
        if flow_active:
            z, flow_logdet = self.flow(white, flow_context)
        else:
            z = white
            flow_logdet = torch.zeros(target_u.shape[0], device=target_u.device, dtype=target_u.dtype)

        base_logprob = self.flow.base_log_prob(z)
        logdet_cov = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        logprob = base_logprob + flow_logdet - logdet_cov

        _, _, avg_var = self.normalized_components(factor, diag)
        white_energy = white.pow(2).mean(dim=-1)
        aux = {
            "cov": cov,
            "avg_var": avg_var,
            "flow_logdet": flow_logdet,
            "white_norm": white.norm(dim=-1),
            "white_energy": white_energy,
            "base_nu": self.flow.base_nu().expand_as(white_energy),
        }
        return logprob, aux

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, flow_context = self.forward_from_history(history_01)
        batch_size, n_cells = mu.shape
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)

        z = self.flow.sample_base(batch_size, n_samples, device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch_size, n_samples, -1).reshape(batch_size * n_samples, -1)
        white, _ = self.flow.inverse(z, ctx)
        white = white.view(batch_size, n_samples, n_cells)
        samples = mu.unsqueeze(1) + torch.einsum("bij,bnj->bni", chol, white)
        return samples

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


def conditional_copula_multistep_loss(
    model: GraphARConditionalCopulaModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict,
    self_feed_prob: float = 0.0,
    flow_active: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = history_01.shape[0]
    future_len = future_01.shape[1]

    history_tokens, pooled_history, decoder_state = model.encode_history(history_01)
    prev_01 = history_01[:, -1].reshape(batch_size, -1)
    decoder_state = model.update_state(prev_01, decoder_state)

    total_nll = 0.0
    total_mae = 0.0
    total_rank = 0.0
    total_shape_var = 0.0
    total_scale = 0.0
    total_flow_logdet = 0.0
    total_white_norm = 0.0
    total_white_energy = 0.0
    total_attn_top1 = 0.0

    pred_mean_path = []
    target_mean_path = []

    for step in range(future_len):
        mu, factor, diag, scale, flow_context, attn_top1 = model.decode_step(
            prev_01, decoder_state, history_tokens, pooled_history
        )
        target_t = future_01[:, step, :]
        target_u = iv_to_unconstrained(
            target_t,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        logprob_t, aux = model.log_prob_step(
            target_u=target_u,
            mu=mu,
            factor=factor,
            diag=diag,
            scale=scale,
            flow_context=flow_context,
            flow_active=flow_active,
        )
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        total_nll = total_nll - logprob_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_rank = total_rank + effective_rank(aux["cov"]).mean()
        total_shape_var = total_shape_var + aux["avg_var"].mean()
        total_scale = total_scale + scale.mean()
        total_flow_logdet = total_flow_logdet + aux["flow_logdet"].mean()
        total_white_norm = total_white_norm + aux["white_norm"].mean()
        total_white_energy = total_white_energy + aux["white_energy"].mean()
        total_attn_top1 = total_attn_top1 + attn_top1.mean()

        pred_mean_path.append(mu_iv.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        use_pred = (
            self_feed_prob > 0.0
            and step < future_len - 1
            and torch.rand((), device=history_01.device).item() < self_feed_prob
        )
        next_frame = mu_iv.detach() if use_pred else target_t
        prev_01 = next_frame
        decoder_state = model.update_state(next_frame, decoder_state)

    pred_mean_path_t = torch.stack(pred_mean_path, dim=1)
    target_mean_path_t = torch.stack(target_mean_path, dim=1)
    path_mean_loss = F.smooth_l1_loss(pred_mean_path_t, target_mean_path_t)
    white_energy_loss = F.smooth_l1_loss(
        (total_white_energy / future_len).reshape(1),
        torch.ones(1, device=history_01.device, dtype=history_01.dtype),
    )
    flow_logdet_penalty = (total_flow_logdet / future_len).pow(2)

    total_loss = (
        total_nll / future_len
        + objective_config["path_mean_weight"] * path_mean_loss
        + objective_config["white_energy_weight"] * white_energy_loss
        + objective_config["flow_logdet_weight"] * flow_logdet_penalty
    )

    metrics = {
        "total_loss": total_loss,
        "multistep_nll": total_nll / future_len,
        "multistep_mae": total_mae / future_len,
        "pred_eff_rank": total_rank / future_len,
        "shape_avg_var": total_shape_var / future_len,
        "scale_mean": total_scale / future_len,
        "flow_logdet_mean": total_flow_logdet / future_len,
        "white_norm_mean": total_white_norm / future_len,
        "white_energy_mean": total_white_energy / future_len,
        "path_mean_loss": path_mean_loss,
        "white_energy_loss": white_energy_loss,
        "flow_logdet_penalty": flow_logdet_penalty,
        "attention_top1": total_attn_top1 / future_len,
        "base_nu": model.flow.base_nu(),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: GraphARConditionalCopulaModel,
    val_loader: DataLoader,
    objective_config: dict,
    flow_active: bool = True,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = conditional_copula_multistep_loss(
            model,
            history_01,
            future_01,
            objective_config=objective_config,
            self_feed_prob=0.0,
            flow_active=flow_active,
        )
        batch_size = history_01.shape[0]
        total_count += batch_size
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
    if total_count == 0:
        return {}
    return {f"val_{k}": v / total_count for k, v in totals.items()}


def maybe_load_partial_warm_start(model: GraphARConditionalCopulaModel, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    current = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    print(f"Partial warm start loaded from {checkpoint_path}")
    print(f"  matched keys: {len(loadable)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="192a: graph-aware AR conditional copula model")
    parser.add_argument("--stage1_epochs", type=int, default=4)
    parser.add_argument("--stage2_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_stage1", type=float, default=2e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--decoder_state_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=160)
    parser.add_argument("--flow_hidden_dim", type=int, default=96)
    parser.add_argument("--flow_layers", type=int, default=1)
    parser.add_argument("--flow_scale_clip", type=float, default=1.5)
    parser.add_argument("--flow_dim_embed_dim", type=int, default=16)
    parser.add_argument("--flow_base_nu_init", type=float, default=8.0)
    parser.add_argument("--flow_base_nu_floor", type=float, default=2.1)
    parser.add_argument("--flow_base_nu_max", type=float, default=30.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--path_mean_weight", type=float, default=0.20)
    parser.add_argument("--white_energy_weight", type=float, default=0.06)
    parser.add_argument("--flow_logdet_weight", type=float, default=0.01)
    parser.add_argument("--self_feed_max", type=float, default=0.35)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--warm_start", type=str, default=None)
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
        bottleneck_dim=args.token_dim,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=args.token_dim,
        token_dim=args.token_dim,
        decoder_state_dim=args.decoder_state_dim,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
    )
    flow_config = dict(
        dim=25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
        dim_embed_dim=args.flow_dim_embed_dim,
        base_nu_init=args.flow_base_nu_init,
        base_nu_floor=args.flow_base_nu_floor,
        base_nu_max=args.flow_base_nu_max,
    )
    objective_config = dict(
        path_mean_weight=args.path_mean_weight,
        white_energy_weight=args.white_energy_weight,
        flow_logdet_weight=args.flow_logdet_weight,
    )

    model = GraphARConditionalCopulaModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_partial_warm_start(model, args.warm_start)

    print("=" * 72)
    print("192a: Graph-aware AR conditional copula model")
    print("=" * 72)
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"Stage1 epochs: {args.stage1_epochs} | Stage2 epochs: {args.stage2_epochs}")
    print(f"Flow layers: {args.flow_layers} | flow hidden: {args.flow_hidden_dim}")
    print(f"Token dim: {args.token_dim} | decoder state: {args.decoder_state_dim}")

    stage1_params = list(model.history_encoder.parameters()) + list(model.step_state.parameters()) + list(model.decoder.parameters())

    best_score = float("inf")
    best_metrics = None
    history = []
    total_epochs = args.stage1_epochs + args.stage2_epochs

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        if epoch <= args.stage1_epochs:
            stage = 1
            model.history_encoder.requires_grad_(True)
            model.step_state.requires_grad_(True)
            model.decoder.requires_grad_(True)
            model.flow.requires_grad_(False)
            optimizer = torch.optim.AdamW(stage1_params, lr=args.lr_stage1, weight_decay=args.weight_decay)
            self_feed_prob = 0.0
            flow_active = False
        else:
            stage = 2
            model.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
            stage2_idx = epoch - args.stage1_epochs
            stage2_den = max(args.stage2_epochs - 1, 1)
            self_feed_prob = args.self_feed_max * (stage2_idx - 1) / stage2_den
            flow_active = True

        model.train()
        train_sums: dict[str, float] = {}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = conditional_copula_multistep_loss(
                model,
                history_01,
                future_01,
                objective_config=objective_config,
                self_feed_prob=self_feed_prob,
                flow_active=flow_active,
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
        val_metrics = evaluate_teacher_forced(model, val_loader, objective_config, flow_active=flow_active)
        rollout_metrics = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )
        elapsed = time.time() - t0

        score = (
            val_metrics["val_total_loss"]
            + 0.25 * abs(rollout_metrics["rollout_cov90"] - 0.90)
            + 0.12 * max(0.0, 1.15 - rollout_metrics["rollout_turb_calm_ratio"])
            + 0.10 * rollout_metrics["rollout_support_violation_rate"]
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
                        "type": "graph_ar_conditional_copula_192a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
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
            "flow_active": flow_active,
            "self_feed_prob": self_feed_prob,
            **train_metrics,
            **val_metrics,
            **rollout_metrics,
            "selection_score": score,
        }
        history.append(row)
        print(
            f"Ep {epoch:3d} stg={stage} "
            f"train={train_metrics['train_total_loss']:.4f} "
            f"val={val_metrics['val_total_loss']:.4f} "
            f"roll_cov90={rollout_metrics['rollout_cov90']:.4f} "
            f"roll_tc={rollout_metrics['rollout_turb_calm_ratio']:.3f} "
            f"attn={val_metrics['val_attention_top1']:.3f} "
            f"scale={val_metrics['val_scale_mean']:.3f} "
            f"nu={val_metrics['val_base_nu']:.2f} "
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
            "type": "graph_ar_conditional_copula_192a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
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
            f"roll_tc={best_metrics['rollout_turb_calm_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
