#!/usr/bin/env python
"""
180d_v0: Sparse centered jump residual adapter on top of the 179b backbone.

Keep:
  - explicit mean-reverting mean dynamics
  - structured covariance with exact block covariance-mixture semantics
  - geometry-aware centered smooth residual transport

Change:
  - add a sparse centered jump residual process in whitened basis space
  - train the jump branch against backbone residual exceedances with the backbone frozen
  - preserve the existing backbone while enlarging residual expressiveness
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
from experiments.backfill.block_ar.basis_geometry import BasisGeometry
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
    ExactBlockCovarianceMixtureMeanRevertingResidualFlowStructuredJointStudentTModel,
    aggregate_slope_ratio,
    evaluate_joint_subset,
)


class ConditionalBandCenteredScaleCoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_clip_vec: torch.Tensor,
        high_band_mask: torch.Tensor,
    ):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float().view(1, dim))
        self.register_buffer("inv_mask", 1.0 - self.mask)
        self.register_buffer("scale_clip_vec", scale_clip_vec.float().view(1, dim))
        self.register_buffer("high_band_mask", high_band_mask.float().view(1, dim))
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
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

    def _log_s(self, x_masked: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        even_features = x_masked.pow(2)
        h = torch.cat([even_features, context], dim=-1)
        log_s = torch.tanh(self.net(h)) * self.scale_clip_vec
        return log_s * self.inv_mask

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_s = self._log_s(x_masked, context)
        y = x_masked + self.inv_mask * (x * torch.exp(log_s))
        logdet = log_s.sum(dim=-1)
        return y, logdet

    def forward_with_stats(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_masked = x * self.mask
        log_s = self._log_s(x_masked, context)
        y = x_masked + self.inv_mask * (x * torch.exp(log_s))
        logdet = log_s.sum(dim=-1)
        all_denom = self.inv_mask.sum().clamp_min(1.0)
        high_mask = self.high_band_mask * self.inv_mask
        high_denom = high_mask.sum().clamp_min(1.0)
        stats = {
            "mean_logscale": log_s.sum(dim=-1) / all_denom,
            "abs_logscale": log_s.abs().sum(dim=-1) / all_denom,
            # Negative forward log-scale means inverse-time expansion.
            "high_band_neg_logscale": (F.relu(-log_s) * high_mask).sum(dim=-1) / high_denom,
            "high_band_abs_logscale": (log_s.abs() * high_mask).sum(dim=-1) / high_denom,
        }
        return y, logdet, stats

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        log_s = self._log_s(y_masked, context)
        x = y_masked + self.inv_mask * (y * torch.exp(-log_s))
        logdet = -log_s.sum(dim=-1)
        return x, logdet


class BasisCenteredResidualTransport(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_frames: int,
        grid_h: int = 5,
        grid_w: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 4,
        low_scale_clip: float = 1.2,
        mid_scale_clip: float = 0.7,
        high_scale_clip: float = 0.35,
        seed: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.n_frames = n_frames
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_cells = grid_h * grid_w
        self.geometry = BasisGeometry(
            n_frames=n_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            low_scale_clip=low_scale_clip,
            mid_scale_clip=mid_scale_clip,
            high_scale_clip=high_scale_clip,
        )
        base_clip = self.geometry.clip_vector()
        high_mask = self.geometry.high_band_mask()

        g = torch.Generator()
        g.manual_seed(seed)
        layers = []
        perms = []
        inv_perms = []
        for li in range(n_layers):
            perm = torch.randperm(dim, generator=g)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(dim)
            perms.append(perm)
            inv_perms.append(inv_perm)
            mask = ((torch.arange(dim) + li) % 2 == 0).float()
            layers.append(
                ConditionalBandCenteredScaleCoupling(
                    dim=dim,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    scale_clip_vec=base_clip[perm],
                    high_band_mask=high_mask[perm],
                )
            )
        self.layers = nn.ModuleList(layers)
        self.register_buffer("perms", torch.stack(perms, dim=0))
        self.register_buffer("inv_perms", torch.stack(inv_perms, dim=0))

    def to_basis_flat(self, x: torch.Tensor) -> torch.Tensor:
        coeff = self.geometry.to_basis(x.view(x.shape[0], self.n_frames, self.n_cells))
        return coeff.reshape(x.shape[0], self.dim)

    def from_basis_flat(self, coeff_flat: torch.Tensor) -> torch.Tensor:
        x = self.geometry.from_basis(coeff_flat.view(coeff_flat.shape[0], self.n_frames, self.n_cells))
        return x.reshape(coeff_flat.shape[0], self.dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, logdet, _ = self.forward_with_stats(x, context)
        return z, logdet

    def forward_with_stats(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = self.to_basis_flat(x)
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        mean_logs = []
        abs_logs = []
        high_neg_logs = []
        high_abs_logs = []
        for li, layer in enumerate(self.layers):
            perm = self.perms[li]
            z = z[:, perm]
            z, logdet, stats = layer.forward_with_stats(z, context)
            total_logdet = total_logdet + logdet
            mean_logs.append(stats["mean_logscale"])
            abs_logs.append(stats["abs_logscale"])
            high_neg_logs.append(stats["high_band_neg_logscale"])
            high_abs_logs.append(stats["high_band_abs_logscale"])
        flow_stats = {
            "mean_logscale": torch.stack(mean_logs, dim=0).mean(dim=0),
            "abs_logscale": torch.stack(abs_logs, dim=0).mean(dim=0),
            "high_band_neg_logscale": torch.stack(high_neg_logs, dim=0).mean(dim=0),
            "high_band_abs_logscale": torch.stack(high_abs_logs, dim=0).mean(dim=0),
        }
        return z, total_logdet, flow_stats

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for li in reversed(range(len(self.layers))):
            x, logdet = self.layers[li].inverse(x, context)
            x = x[:, self.inv_perms[li]]
            total_logdet = total_logdet + logdet
        x = self.from_basis_flat(x)
        return x, total_logdet


def build_block_band_group_masks(geometry: BasisGeometry, n_blocks: int) -> tuple[torch.Tensor, list[dict[str, int | str]]]:
    if geometry.n_frames % n_blocks != 0:
        raise ValueError(f"n_frames={geometry.n_frames} must be divisible by n_blocks={n_blocks}")
    block_len = geometry.n_frames // n_blocks
    band_masks = {
        "low": geometry.low_band_mask().reshape(geometry.n_frames, geometry.n_cells),
        "mid": geometry.mid_band_mask().reshape(geometry.n_frames, geometry.n_cells),
        "high": geometry.high_band_mask().reshape(geometry.n_frames, geometry.n_cells),
    }
    masks = []
    metadata = []
    for block in range(n_blocks):
        time_mask = torch.zeros(geometry.n_frames, 1, dtype=geometry.basis_time.dtype)
        time_mask[block * block_len : (block + 1) * block_len] = 1.0
        for band in ("low", "mid", "high"):
            mask = (time_mask * band_masks[band]).reshape(-1)
            if mask.sum() <= 0:
                continue
            masks.append(mask)
            metadata.append({"block": int(block), "band": band})
    return torch.stack(masks, dim=0), metadata


class SparseGroupedJumpModule(nn.Module):
    def __init__(
        self,
        context_dim: int,
        group_masks: torch.Tensor,
        hidden_dim: int = 128,
        max_scale: float = 0.60,
        init_logit_bias: float = -2.5,
        init_scale_bias: float = -2.0,
        amplitude_df: float = 5.0,
    ):
        super().__init__()
        self.num_groups = int(group_masks.shape[0])
        self.dim = int(group_masks.shape[1])
        self.max_scale = float(max_scale)
        self.amplitude_df = float(amplitude_df)
        self.register_buffer("group_masks", group_masks.float())
        self.register_buffer("group_counts", group_masks.sum(dim=-1).clamp_min(1.0))
        self.logit_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_groups),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_groups),
        )
        for mod in self.logit_head:
            if isinstance(mod, nn.Linear):
                nn.init.zeros_(mod.bias)
        for mod in self.scale_head:
            if isinstance(mod, nn.Linear):
                nn.init.zeros_(mod.bias)
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.logit_head[-1].bias, init_logit_bias)
        nn.init.constant_(self.scale_head[-1].bias, init_scale_bias)

    def params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logit_head(context)
        probs = torch.sigmoid(logits)
        scales = torch.sigmoid(self.scale_head(context)) * self.max_scale
        return logits, probs, scales

    def group_rms(self, basis_flat: torch.Tensor) -> torch.Tensor:
        sq = basis_flat.pow(2)
        numer = torch.matmul(sq, self.group_masks.t())
        return torch.sqrt(numer / self.group_counts.unsqueeze(0))

    def sample_shifts(
        self,
        context: torch.Tensor,
        n_samples: int,
        force_nonzero: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, probs, scales = self.params(context)
        batch = context.shape[0]
        device = context.device
        dtype = context.dtype
        if force_nonzero:
            active = torch.ones(batch, n_samples, self.num_groups, device=device, dtype=dtype)
        else:
            active = torch.bernoulli(probs.unsqueeze(1).expand(batch, n_samples, -1))
        amp_dist = torch.distributions.StudentT(
            df=self.amplitude_df,
            loc=torch.zeros((), device=device, dtype=dtype),
            scale=torch.ones((), device=device, dtype=dtype),
        )
        amplitudes = amp_dist.sample((batch, n_samples, self.num_groups)) * scales.unsqueeze(1)
        shifts = torch.zeros(batch, n_samples, self.dim, device=device, dtype=dtype)
        for g in range(self.num_groups):
            mask = self.group_masks[g].view(1, 1, -1)
            eps = torch.randn(batch, n_samples, self.dim, device=device, dtype=dtype) * mask
            eps = eps / eps.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            shifts = shifts + active[:, :, g : g + 1] * amplitudes[:, :, g : g + 1] * eps
        stats = {
            "group_probs": probs,
            "group_scales": scales,
            "group_entropy": (
                -(probs * torch.log(probs.clamp_min(1e-8)) + (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-8)))
            ).mean(dim=-1),
            "active_rate": active.mean(dim=(1, 2)),
            "jump_l2": shifts.pow(2).sum(dim=-1).sqrt().mean(dim=1),
        }
        return shifts, stats


class BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel(
    ExactBlockCovarianceMixtureMeanRevertingResidualFlowStructuredJointStudentTModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
        mix_chunk_size: int = 27,
    ):
        base_flow_config = {
            "dim": flow_config["dim"],
            "context_dim": flow_config["context_dim"],
            "hidden_dim": flow_config["hidden_dim"],
            "n_layers": flow_config["n_layers"],
        }
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            flow_config=base_flow_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        self.flow = BasisCenteredResidualTransport(**flow_config)
        self.flow_config = flow_config

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("flow."):
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
            ctx = flow_context.unsqueeze(1).expand(batch, chunk, -1).reshape(batch * chunk, -1)
            z, flow_logdet = self.flow(white_flat, ctx)
            base_logprob = self._base_logprob(z).view(batch, chunk)
            flow_logdet = flow_logdet.view(batch, chunk)
            comp_logprob = base_logprob + flow_logdet - 0.5 * (
                logdet_cov.unsqueeze(1) + logdet_local.unsqueeze(1) + logdet_template_cov.unsqueeze(0)
            )
            chunk_lse = torch.logsumexp(log_prior[:, start:end] + comp_logprob, dim=1)
            mix_logprob = chunk_lse if mix_logprob is None else torch.logaddexp(mix_logprob, chunk_lse)

        map_assign = block_logits.argmax(dim=-1)
        local_scale_map = shared_local_scale
        map_factors, _map_logdet, map_offdiag_rms = self.decoder.build_template_factors(map_assign)
        rhs_map = white_blocks.permute(0, 1, 3, 2).reshape(batch * self.decoder.n_blocks, n_cells, self.decoder.block_len)
        factor_map = map_factors.reshape(batch * self.decoder.n_blocks, n_cells, n_cells)
        base_blocks_map = torch.linalg.solve_triangular(factor_map, rhs_map, upper=False)
        base_blocks_map = base_blocks_map.reshape(batch, self.decoder.n_blocks, n_cells, self.decoder.block_len).permute(0, 1, 3, 2)
        white_flat_map = base_blocks_map.reshape(batch, n_frames * n_cells)
        basis_coeff_map = self.flow.to_basis_flat(white_flat_map)
        z_map, flow_logdet_map, flow_stats = self.flow.forward_with_stats(white_flat_map, flow_context)
        high_mask = self.flow.geometry.high_band_mask().to(white_flat_map.device)
        high_denom = high_mask.sum().clamp_min(1.0)
        high_coeff_std = ((basis_coeff_map.pow(2) * high_mask.unsqueeze(0)).sum(dim=-1) / high_denom).sqrt()
        basis_coeff_std = basis_coeff_map.std(dim=-1)

        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "flow_logdet": flow_logdet_map,
            "white_std": white_flat_map.std(dim=-1),
            "z_std": z_map.std(dim=-1),
            "basis_coeff_std": basis_coeff_std,
            "high_band_coeff_std": high_coeff_std,
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
        }
        return mix_logprob, aux


class SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel(
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
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
        group_masks, group_metadata = build_block_band_group_masks(self.flow.geometry, decoder_config["n_blocks"])
        self.jump = SparseGroupedJumpModule(
            context_dim=flow_config["context_dim"],
            group_masks=group_masks,
            hidden_dim=jump_config.get("hidden_dim", 128),
            max_scale=jump_config.get("max_scale", 0.60),
            init_logit_bias=jump_config.get("init_logit_bias", -2.5),
            init_scale_bias=jump_config.get("init_scale_bias", -2.0),
            amplitude_df=jump_config.get("amplitude_df", 5.0),
        )
        self.jump_config = jump_config
        self.jump_group_metadata = group_metadata

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("jump."):
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
        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        smooth_white_flat, _ = self.flow.inverse(z, ctx)
        smooth_basis_flat = self.flow.to_basis_flat(smooth_white_flat)
        jump_shift, _jump_stats = self.jump.sample_shifts(flow_context, n_samples=n_samples)
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


def checkpoint_key(val_metrics: dict, joint_metrics: dict) -> tuple[float, float, float, float, float, float]:
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("nan")))
    kurt_ratio = float(joint_metrics.get("joint_kurtosis_ratio", float("nan")))
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    cov90 = float(joint_metrics.get("joint_cov90", float("nan")))
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    kurt_gap = 1e6 if not np.isfinite(kurt_ratio) else max(0.5 - kurt_ratio, 0.0) + max(kurt_ratio - 2.0, 0.0)
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    cov_gap = 1e6 if not np.isfinite(cov90) else abs(cov90 - 0.90)
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    return (jump_gap, kurt_gap, mr_gap, cov_gap, tc_gap, val_loss)


def pearson_kurtosis(x: torch.Tensor) -> float:
    arr = x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    xc = arr - arr.mean()
    var = np.mean(np.square(xc))
    if var <= 1e-12:
        return float("nan")
    return float(np.mean(np.power(xc, 4)) / (var * var))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def estimate_jump_thresholds(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
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
        basis_flat, _ctx = model.teacher_basis_flat(history_01, future_01)
        all_rms.append(model.jump.group_rms(basis_flat))
    rms = torch.cat(all_rms, dim=0)
    thresholds = torch.quantile(rms, quantile, dim=0)
    target_rates = (rms > thresholds.unsqueeze(0)).float().mean(dim=0)
    return thresholds.detach(), target_rates.detach()


def jump_stage_loss(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    thresholds: torch.Tensor,
    target_rates: torch.Tensor,
    rate_penalty: float,
    scale_penalty: float,
):
    history_01 = history_01.to(next(model.parameters()).device)
    future_01 = future_01.to(next(model.parameters()).device)
    with torch.no_grad():
        basis_flat, flow_context = model.teacher_basis_flat(history_01, future_01)
    logits, probs, scales = model.jump.params(flow_context)
    group_rms = model.jump.group_rms(basis_flat)
    target_active = (group_rms > thresholds.unsqueeze(0)).float()
    target_excess = (group_rms - thresholds.unsqueeze(0)).clamp_min(0.0)
    pos_weight = ((1.0 - target_rates).clamp_min(1e-4) / target_rates.clamp_min(1e-4)).detach()
    bce = F.binary_cross_entropy_with_logits(logits, target_active, pos_weight=pos_weight.unsqueeze(0))
    if target_active.any():
        scale_loss = F.smooth_l1_loss(scales[target_active > 0], target_excess[target_active > 0])
    else:
        scale_loss = scales.new_tensor(0.0)
    rate_loss = (probs.mean(dim=0) - target_rates).abs().mean()
    total = bce + scale_penalty * scale_loss + rate_penalty * rate_loss
    metrics = {
        "total_loss": total,
        "bce_loss": bce,
        "scale_loss": scale_loss,
        "rate_loss": rate_loss,
        "target_active_rate": target_active.mean(),
        "pred_active_rate": probs.mean(),
        "pred_scale_mean": scales.mean(),
        "pred_scale_max": scales.max(),
        "target_excess_mean": target_excess.mean(),
        "target_excess_active_mean": target_excess[target_active > 0].mean() if target_active.any() else scales.new_tensor(0.0),
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
    val_loader: DataLoader,
    thresholds: torch.Tensor,
    target_rates: torch.Tensor,
    rate_penalty: float,
    scale_penalty: float,
):
    model.eval()
    keys = [
        "total_loss", "bce_loss", "scale_loss", "rate_loss", "target_active_rate", "pred_active_rate",
        "pred_scale_mean", "pred_scale_max", "target_excess_mean", "target_excess_active_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = jump_stage_loss(
            model,
            history_01,
            future_01,
            thresholds=thresholds,
            target_rates=target_rates,
            rate_penalty=rate_penalty,
            scale_penalty=scale_penalty,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
):
    model.eval()
    total_cov = total_width = total_mae = total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    all_sample_eff_rank = []
    prev_chunks = []
    gt_next_chunks = []
    det_next_chunks = []
    sample_next_chunks = []
    gt_changes = []
    gen_changes = []
    gt_path_max = []
    gen_path_max = []

    for history_01, future_01 in val_loader:
        if total_count >= eval_limit:
            break
        if total_count + history_01.shape[0] > eval_limit:
            keep = eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples_u = model.sample_batched(history_norm, n_samples=joint_val_samples)
        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples_u.quantile(0.05, dim=1)
        hi = samples_u.quantile(0.95, dim=1)
        median = samples_u.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()
        support_viol = ((samples_u < model.support_lo) | (samples_u > model.support_hi)).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        first_sample = samples_u[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 1e-10)
        probs = eigvals / eigvals.sum()
        all_sample_eff_rank.append(float(np.exp(-(probs * np.log(probs)).sum())))

        det_outputs = model.forward_from_history(denormalize_iv(history_norm))
        det_next = unconstrained_to_iv(det_outputs[0][:, 0], lo=model.support_lo, hi=model.support_hi).reshape(history_01.shape[0], 5, 5)
        prev_chunks.append(history_01[:, -1].detach().cpu())
        gt_next_chunks.append(future_01[:, 0].detach().cpu())
        det_next_chunks.append(det_next.detach().cpu())
        sample_next_chunks.append(samples_u[:, :, 0].mean(dim=1).detach().cpu())

        future_path = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        gt_change = (future_path[:, 1:] - future_path[:, :-1]).detach().cpu()
        gen_change = (samples_u[:, 0, 1:] - samples_u[:, 0, :-1]).detach().cpu()
        gt_changes.append(gt_change)
        gen_changes.append(gen_change)

        gt_path = torch.cat([history_01[:, -1:].detach().cpu(), future_path.detach().cpu()], dim=1)
        gen_path = torch.cat([history_01[:, -1:].detach().cpu(), samples_u[:, 0].detach().cpu()], dim=1)
        gt_path_max.append((gt_path[:, 1:] - gt_path[:, :-1]).abs().amax(dim=(1, 2, 3)))
        gen_path_max.append((gen_path[:, 1:] - gen_path[:, :-1]).abs().amax(dim=(1, 2, 3)))

        total_cov += coverage.item() * history_01.shape[0]
        total_width += width.item() * history_01.shape[0]
        total_mae += mae.item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]
        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {k: float("nan") for k in [
            "joint_cov90","joint_width90","joint_mae","joint_support_violation_rate",
            "joint_turb_calm_ratio","joint_sample_eff_rank","joint_det_mr_ratio","joint_sample_mr_ratio",
            "joint_kurtosis_ratio","joint_pathwise_jump_ks"
        ]}

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    turb_calm_ratio = (widths[turb_mask].mean() / widths[calm_mask].mean()).item() if calm_mask.any() and turb_mask.any() else float("nan")

    prev = torch.cat(prev_chunks, dim=0)
    gt_next = torch.cat(gt_next_chunks, dim=0)
    det_next = torch.cat(det_next_chunks, dim=0)
    sample_next = torch.cat(sample_next_chunks, dim=0)
    det_mr_ratio = aggregate_slope_ratio(prev, gt_next, det_next)
    sample_mr_ratio = aggregate_slope_ratio(prev, gt_next, sample_next)

    gt_changes = torch.cat(gt_changes, dim=0)
    gen_changes = torch.cat(gen_changes, dim=0)
    gt_path_max = torch.cat(gt_path_max, dim=0).numpy()
    gen_path_max = torch.cat(gen_path_max, dim=0).numpy()
    gt_kurt = pearson_kurtosis(gt_changes)
    gen_kurt = pearson_kurtosis(gen_changes)
    kurt_ratio = gen_kurt / max(gt_kurt, 1e-12)
    jump_ks = ks_statistic(gt_path_max, gen_path_max)

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
        "joint_det_mr_ratio": det_mr_ratio,
        "joint_sample_mr_ratio": sample_mr_ratio,
        "joint_kurtosis_ratio": float(kurt_ratio),
        "joint_pathwise_jump_ks": float(jump_ks),
    }


def main():
    parser = argparse.ArgumentParser(description="180d_v0: sparse centered jump residual adapter")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_jump", type=float, default=5e-4)
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
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--template_penalty", type=float, default=0.10)
    parser.add_argument("--gate_balance_penalty", type=float, default=0.25)
    parser.add_argument("--gate_smooth_penalty", type=float, default=0.5)
    parser.add_argument("--spectral_neutrality_penalty", type=float, default=0.25)
    parser.add_argument("--high_band_penalty", type=float, default=0.5)
    parser.add_argument("--jump_hidden_dim", type=int, default=128)
    parser.add_argument("--jump_max_scale", type=float, default=0.60)
    parser.add_argument("--jump_init_logit_bias", type=float, default=-2.2)
    parser.add_argument("--jump_init_scale_bias", type=float, default=-2.0)
    parser.add_argument("--jump_amplitude_df", type=float, default=5.0)
    parser.add_argument("--jump_rate_penalty", type=float, default=0.25)
    parser.add_argument("--jump_scale_penalty", type=float, default=1.0)
    parser.add_argument("--threshold_quantile", type=float, default=0.97)
    parser.add_argument("--threshold_batches", type=int, default=120)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
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
    jump_config = dict(
        hidden_dim=args.jump_hidden_dim,
        max_scale=args.jump_max_scale,
        init_logit_bias=args.jump_init_logit_bias,
        init_scale_bias=args.jump_init_scale_bias,
        amplitude_df=args.jump_amplitude_df,
    )
    model = SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
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
    model.jump.requires_grad_(True)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    n_jump = sum(p.numel() for p in model.jump.parameters())
    print(f"\n{'=' * 64}")
    print("180d_v0: sparse centered jump residual adapter")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Basis transport params: {n_flow:,}")
    print(f"  Jump params: {n_jump:,}")
    print(f"  Total params: {n_enc + n_dec + n_flow + n_jump:,}")
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
        [{"params": model.jump.parameters(), "lr": args.lr_jump, "weight_decay": args.weight_decay_jump}]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metric_names = [
        "total_loss","bce_loss","scale_loss","rate_loss","target_active_rate","pred_active_rate",
        "pred_scale_mean","pred_scale_max","target_excess_mean","target_excess_active_mean"
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
            loss, metrics = jump_stage_loss(
                model,
                history_01,
                future_01,
                thresholds=thresholds,
                target_rates=target_rates,
                rate_penalty=args.jump_rate_penalty,
                scale_penalty=args.jump_scale_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            rate_penalty=args.jump_rate_penalty,
            scale_penalty=args.jump_scale_penalty,
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
                        "type": "sparse_centered_jump_residual_mean_reverting_covariance_mixture_structured_joint_student_t_180d",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
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
                        "jump_rate_penalty": args.jump_rate_penalty,
                        "jump_scale_penalty": args.jump_scale_penalty,
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
            f"pAct={val_metrics['val_pred_active_rate']:.4f}  sMean={val_metrics['val_pred_scale_mean']:.4f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "config": {
            "type": "sparse_centered_jump_residual_mean_reverting_covariance_mixture_structured_joint_student_t_180d",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
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
            "jump_rate_penalty": args.jump_rate_penalty,
            "jump_scale_penalty": args.jump_scale_penalty,
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
