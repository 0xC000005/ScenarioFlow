#!/usr/bin/env python
"""
179b_v0: Geometry-aware centered residual transport on top of the 178d backbone.

Keep:
  - explicit mean-reverting mean dynamics
  - structured covariance with exact block covariance-mixture semantics
  - centered residual transport (no translation)

Change:
  - move the residual transport into a smooth spatial-temporal basis
  - use tighter scale authority on higher-frequency coefficients
  - penalize high-frequency inverse expansion pressure
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
    effective_rank,
    iv_to_unconstrained,
    make_serializable,
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


def checkpoint_key(val_metrics: dict, joint_metrics: dict) -> tuple[float, float, float, float]:
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    cov90 = float(joint_metrics.get("joint_cov90", float("nan")))
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    cov_gap = 1e6 if not np.isfinite(cov90) else abs(cov90 - 0.90)
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    return (mr_gap, cov_gap, tc_gap, val_loss)


def joint_nll_loss(
    model: BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    gate_smooth_penalty: float,
    spectral_neutrality_penalty: float,
    high_band_penalty: float,
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

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    local_pen = base_local_delta.pow(2).mean()
    template_l2 = model.decoder.block_cov_diag_bank.pow(2).mean() + model.decoder.block_cov_lower_bank.pow(2).mean()
    static_l2 = model.decoder.static_local_logvar.pow(2).mean()
    block_probs = F.softmax(block_logits, dim=-1)
    usage = aux["block_usage"]
    target_usage = torch.full_like(usage, 1.0 / usage.numel())
    usage_penalty = (usage - target_usage).pow(2).mean()
    smooth_pen = (block_probs[:, 1:] - block_probs[:, :-1]).pow(2).mean() if block_probs.shape[1] > 1 else block_probs.new_tensor(0.0)
    spectral_pen = aux["flow_mean_logscale"].pow(2).mean()
    high_pen = aux["high_band_neg_logscale"].mean()
    nll = (-logprob).mean()
    loss = (
        nll
        + local_var_penalty * (local_pen + 0.25 * static_l2)
        + template_penalty * template_l2
        + gate_balance_penalty * usage_penalty
        + gate_smooth_penalty * smooth_pen
        + spectral_neutrality_penalty * spectral_pen
        + high_band_penalty * high_pen
    )

    det_mr_ratio = aggregate_slope_ratio(
        history_01[:, -1].reshape(history_01.shape[0], -1),
        future_01[:, 0].reshape(future_01.shape[0], -1),
        pred_01[:, 0],
    )

    metrics = {
        "total_loss": loss,
        "joint_nll": nll,
        "local_var_penalty": local_pen,
        "template_l2": template_l2,
        "gate_usage_penalty": usage_penalty,
        "smooth_penalty": smooth_pen,
        "spectral_neutrality_penalty": spectral_pen,
        "high_band_penalty": high_pen,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "joint_det_mr_ratio": pred_01.new_tensor(det_mr_ratio),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": aux["flow_logdet"].mean(),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
        "basis_coeff_std_mean": aux["basis_coeff_std"].mean(),
        "high_band_coeff_std_mean": aux["high_band_coeff_std"].mean(),
        "flow_mean_logscale_mean": aux["flow_mean_logscale"].mean(),
        "flow_abs_logscale_mean": aux["flow_abs_logscale"].mean(),
        "high_band_neg_logscale_mean": aux["high_band_neg_logscale"].mean(),
        "high_band_abs_logscale_mean": aux["high_band_abs_logscale"].mean(),
        "local_delta_rms": aux["local_delta_rms"].mean(),
        "local_scale_min": aux["local_scale_min"].mean(),
        "local_scale_max": aux["local_scale_max"].mean(),
        "template_diag_mean": aux["template_diag_mean"].mean(),
        "template_offdiag_rms": aux["template_offdiag_rms"].mean(),
        "block_gate_entropy": aux["block_gate_entropy"].mean(),
        "block_gate_max": aux["block_gate_max"].mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    gate_smooth_penalty: float,
    spectral_neutrality_penalty: float,
    high_band_penalty: float,
):
    model.eval()
    keys = [
        "total_loss","joint_nll","local_var_penalty","template_l2","gate_usage_penalty","smooth_penalty",
        "spectral_neutrality_penalty","high_band_penalty","joint_mae","joint_det_mr_ratio","time_eff_rank",
        "cell_eff_rank","scale_mean","flow_logdet_mean","white_std_mean","z_std_mean","basis_coeff_std_mean",
        "high_band_coeff_std_mean","flow_mean_logscale_mean","flow_abs_logscale_mean","high_band_neg_logscale_mean",
        "high_band_abs_logscale_mean","local_delta_rms","local_scale_min","local_scale_max","template_diag_mean",
        "template_offdiag_rms","block_gate_entropy","block_gate_max"
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            template_penalty=template_penalty,
            gate_balance_penalty=gate_balance_penalty,
            gate_smooth_penalty=gate_smooth_penalty,
            spectral_neutrality_penalty=spectral_neutrality_penalty,
            high_band_penalty=high_band_penalty,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="179b_v0: basis-centered residual transport on mean-reverting covariance mixture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=5e-4)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
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
    model = BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(device)

    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("179b_v0: basis-centered residual transport on mean-reverting covariance mixture")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Basis transport params: {n_flow:,}")
    print(f"  Total params: {n_enc + n_dec + n_flow:,}")
    if args.warm_start_path:
        print(f"  Warm start: {args.warm_start_path}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metric_names = [
        "total_loss","joint_nll","local_var_penalty","template_l2","gate_usage_penalty","smooth_penalty",
        "spectral_neutrality_penalty","high_band_penalty","joint_mae","joint_det_mr_ratio","time_eff_rank",
        "cell_eff_rank","scale_mean","flow_logdet_mean","white_std_mean","z_std_mean","basis_coeff_std_mean",
        "high_band_coeff_std_mean","flow_mean_logscale_mean","flow_abs_logscale_mean","high_band_neg_logscale_mean",
        "high_band_abs_logscale_mean","local_delta_rms","local_scale_min","local_scale_max","template_diag_mean",
        "template_offdiag_rms","block_gate_entropy","block_gate_max"
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
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
                template_penalty=args.template_penalty,
                gate_balance_penalty=args.gate_balance_penalty,
                gate_smooth_penalty=args.gate_smooth_penalty,
                spectral_neutrality_penalty=args.spectral_neutrality_penalty,
                high_band_penalty=args.high_band_penalty,
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
            local_var_penalty=args.local_var_penalty,
            template_penalty=args.template_penalty,
            gate_balance_penalty=args.gate_balance_penalty,
            gate_smooth_penalty=args.gate_smooth_penalty,
            spectral_neutrality_penalty=args.spectral_neutrality_penalty,
            high_band_penalty=args.high_band_penalty,
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
                        "type": "basis_centered_residual_transport_mean_reverting_covariance_mixture_structured_joint_student_t_179b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "mix_chunk_size": args.mix_chunk_size,
                        "local_var_penalty": args.local_var_penalty,
                        "template_penalty": args.template_penalty,
                        "gate_balance_penalty": args.gate_balance_penalty,
                        "gate_smooth_penalty": args.gate_smooth_penalty,
                        "spectral_neutrality_penalty": args.spectral_neutrality_penalty,
                        "high_band_penalty": args.high_band_penalty,
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
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  mr_det={joint_metrics['joint_det_mr_ratio']:.3f}  "
            f"mr_samp={joint_metrics['joint_sample_mr_ratio']:.3f}  highNeg={val_metrics['val_high_band_neg_logscale_mean']:.3f}  "
            f"gateH={val_metrics['val_block_gate_entropy']:.3f}  gateMax={val_metrics['val_block_gate_max']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "config": {
            "type": "basis_centered_residual_transport_mean_reverting_covariance_mixture_structured_joint_student_t_179b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "mix_chunk_size": args.mix_chunk_size,
            "local_var_penalty": args.local_var_penalty,
            "template_penalty": args.template_penalty,
            "gate_balance_penalty": args.gate_balance_penalty,
            "gate_smooth_penalty": args.gate_smooth_penalty,
            "spectral_neutrality_penalty": args.spectral_neutrality_penalty,
            "high_band_penalty": args.high_band_penalty,
        },
        "best_key": best_key,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    if best_metrics is not None:
        print(
            "\nBest diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_det_mr_ratio={best_metrics['joint_det_mr_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
