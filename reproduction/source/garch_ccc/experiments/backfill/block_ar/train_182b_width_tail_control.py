#!/usr/bin/env python
"""
182b_v0: Width-and-tail controlled pathwise residual law.

Keep:
  - support-aware transform
  - explicit mean-reverting mean path
  - structured covariance with exact block covariance-mixture semantics
  - 182a conditional path transport in whitened basis space

Add:
  - low-rank local width allocator over horizon x cell
  - bandwise radial tail controller over low / mid / high basis bands

Training is staged:
  - Stage 1: warm start from 182a_final, freeze backbone and path transport,
    train only amplitude controllers
  - Stage 2: unfreeze path transport lightly, keep backbone fixed
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
from experiments.backfill.block_ar.basis_geometry import BasisGeometry
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import (
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
    flow_matching_loss as base_flow_matching_loss,
)


def _pearson_kurtosis(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    xc = arr - arr.mean()
    var = np.mean(np.square(xc))
    if var <= 1e-12:
        return float("nan")
    return float(np.mean(np.power(xc, 4)) / (var * var))


class LowRankWidthAllocator(nn.Module):
    def __init__(
        self,
        context_dim: int,
        n_frames: int,
        n_cells: int,
        rank: int = 4,
        hidden_dim: int = 256,
        clip: float = 0.80,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.rank = rank
        self.clip = clip
        self.time_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_frames * rank),
        )
        self.cell_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_cells * rank),
        )
        nn.init.normal_(self.time_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.time_head[-1].bias)
        nn.init.normal_(self.cell_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.cell_head[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        t = self.time_head(context).view(batch, self.n_frames, self.rank)
        c = self.cell_head(context).view(batch, self.n_cells, self.rank)
        raw = torch.einsum("btr,bcr->btc", t, c) / math.sqrt(max(self.rank, 1))
        centered = raw - raw.mean(dim=(1, 2), keepdim=True)
        return torch.tanh(centered) * self.clip


class BandwiseRadialTailController(nn.Module):
    def __init__(
        self,
        context_dim: int,
        band_counts: torch.Tensor,
        hidden_dim: int = 128,
        clip: float = 0.45,
    ):
        super().__init__()
        self.clip = clip
        self.register_buffer("band_weights", band_counts / band_counts.sum().clamp_min(1.0), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.net(context)) * self.clip
        centered = raw - (raw * self.band_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        return centered


class WidthTailControlledPathwiseResidualLawModel(
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        path_config: dict,
        prior_config: dict,
        amplitude_config: dict,
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
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        self.amplitude_config = dict(amplitude_config)
        band_id = self.path_geometry.band_id.reshape(-1)
        band_counts = torch.stack([(band_id == i).sum() for i in range(3)]).float()
        self.width_allocator = LowRankWidthAllocator(
            context_dim=path_config["context_dim"],
            n_frames=decoder_config["n_frames"],
            n_cells=decoder_config["n_cells"],
            rank=amplitude_config["width_rank"],
            hidden_dim=amplitude_config["width_hidden_dim"],
            clip=amplitude_config["width_clip"],
        )
        self.band_tail = BandwiseRadialTailController(
            context_dim=path_config["context_dim"],
            band_counts=band_counts,
            hidden_dim=amplitude_config["band_hidden_dim"],
            clip=amplitude_config["band_clip"],
        )
        self.register_buffer(
            "band_masks",
            torch.stack(
                [
                    self.path_geometry.low_band_mask().reshape(decoder_config["n_frames"], decoder_config["n_cells"]),
                    self.path_geometry.mid_band_mask().reshape(decoder_config["n_frames"], decoder_config["n_cells"]),
                    self.path_geometry.high_band_mask().reshape(decoder_config["n_frames"], decoder_config["n_cells"]),
                ],
                dim=0,
            ),
            persistent=False,
        )

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("width_allocator.") or key.startswith("band_tail."):
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

    def build_amplitude_controls(self, path_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_log = self.width_allocator(path_context)
        band_log = self.band_tail(path_context)
        return local_log, band_log

    def apply_amplitude_controls(
        self,
        basis_paths: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, n_samples, n_frames, n_cells = basis_paths.shape
        local_log, band_log = self.build_amplitude_controls(path_context)
        band_map = torch.einsum(
            "bk,ktc->btc",
            band_log,
            self.band_masks.to(device=basis_paths.device, dtype=basis_paths.dtype),
        )
        band_scale = torch.exp(band_map).unsqueeze(1)
        basis_scaled = basis_paths * band_scale
        white = self.path_geometry.from_basis(
            basis_scaled.reshape(batch * n_samples, n_frames, n_cells)
        ).reshape(batch, n_samples, n_frames, n_cells)
        local_scale = torch.exp(local_log).unsqueeze(1)
        white = white * local_scale
        stats = {
            "local_log_abs_mean": local_log.abs().mean(),
            "band_log_abs_mean": band_log.abs().mean(),
            "band_log_high_mean": band_log[:, 2].mean(),
            "band_log_mid_mean": band_log[:, 1].mean(),
            "band_log_low_mean": band_log[:, 0].mean(),
        }
        return white, stats

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
        path_context = self.build_path_context(flow_context, block_logits, base_local_delta, scale)
        basis_paths, _stats = self.sample_basis_paths(path_context, n_samples=n_samples)
        basis_paths = basis_paths.view(batch, n_samples, n_frames, n_cells)
        base_white, _amp_stats = self.apply_amplitude_controls(basis_paths, path_context)
        base_white = base_white.view(batch * n_samples, self.decoder.n_blocks, self.decoder.block_len, n_cells)

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


def amplitude_control_loss(
    model: WidthTailControlledPathwiseResidualLawModel,
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
        target_basis_flat = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        )
        target_basis = target_basis_flat.view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        target_white = model.path_geometry.from_basis(target_basis)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)

        target_local_log = torch.log(target_white.abs().clamp_min(1e-4))
        target_local_log = target_local_log - target_local_log.mean(dim=(1, 2), keepdim=True)
        target_local_log = target_local_log.clamp(
            min=-model.amplitude_config["width_clip"],
            max=model.amplitude_config["width_clip"],
        )
        masks = model.band_masks.to(device=target_basis.device, dtype=target_basis.dtype)
        total_energy = target_basis.pow(2).sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        band_energy = torch.stack(
            [((target_basis.pow(2) * masks[i]).sum(dim=(1, 2)) / total_energy.squeeze(-1).squeeze(-1)) for i in range(3)],
            dim=1,
        )
        target_band_log = torch.log(band_energy.clamp_min(1e-8))
        band_weights = (model.band_masks.reshape(3, -1).sum(dim=1) / model.band_masks.numel()).to(target_basis.device, dtype=target_basis.dtype)
        target_band_log = target_band_log - (target_band_log * band_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        target_band_log = target_band_log.clamp(
            min=-model.amplitude_config["band_clip"],
            max=model.amplitude_config["band_clip"],
        )

    pred_local_log, pred_band_log = model.build_amplitude_controls(path_context)
    local_loss = F.smooth_l1_loss(pred_local_log, target_local_log)
    band_loss = F.smooth_l1_loss(pred_band_log, target_band_log)

    surf = pred_local_log.view(pred_local_log.shape[0], pred_local_log.shape[1], 5, 5)
    time_smooth = (pred_local_log[:, 1:] - pred_local_log[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth

    total = (
        model.amplitude_config["local_loss_weight"] * local_loss
        + model.amplitude_config["band_loss_weight"] * band_loss
        + model.amplitude_config["smooth_reg_weight"] * smooth_reg
    )
    metrics = {
        "amp_total_loss": total,
        "amp_local_loss": local_loss,
        "amp_band_loss": band_loss,
        "amp_smooth_reg": smooth_reg,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "pred_local_abs_mean": pred_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "pred_band_high_mean": pred_band_log[:, 2].mean(),
        "pred_band_mid_mean": pred_band_log[:, 1].mean(),
        "pred_band_low_mean": pred_band_log[:, 0].mean(),
    }
    return total, metrics


@torch.no_grad()
def evaluate_frontier_subset(
    model: WidthTailControlledPathwiseResidualLawModel,
    val_loader: DataLoader,
    joint_val_samples: int,
    eval_limit: int,
) -> dict[str, float]:
    model.eval()
    sample_batches = []
    gt_batches = []
    hist_batches = []
    count = 0
    for history_01, future_01 in val_loader:
        if count >= eval_limit:
            break
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        batch = min(history_01.shape[0], eval_limit - count)
        history_01 = history_01[:batch]
        future_01 = future_01[:batch]
        samples_u = model.sample_future_u(history_01, n_samples=joint_val_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).view(
            batch, joint_val_samples, model.decoder.n_frames, 5, 5
        )
        sample_batches.append(samples_01.detach().cpu())
        gt_batches.append(future_01.view(batch, model.decoder.n_frames, 5, 5).detach().cpu())
        hist_batches.append(history_01.detach().cpu())
        count += batch
    cond_samples = torch.cat(sample_batches, dim=0).numpy()  # (N,S,T,5,5)
    ground_truth = torch.cat(gt_batches, dim=0).numpy()  # (N,T,5,5)
    history = torch.cat(hist_batches, dim=0).numpy()  # (N,H,5,5)

    mean_iv = history.mean(axis=(2, 3))
    vov = np.diff(mean_iv, axis=1).std(axis=1)
    q80 = float(np.quantile(vov, 0.8))
    turb_mask = vov >= q80

    lo90 = np.quantile(cond_samples, 0.05, axis=1)
    hi90 = np.quantile(cond_samples, 0.95, axis=1)
    inside90 = (ground_truth >= lo90) & (ground_truth <= hi90)
    turb_late_worst = 1.0
    turb_late_best = 0.0
    if turb_mask.any():
        for h in [13, 29]:
            cov = inside90[turb_mask, h].mean(axis=0)
            turb_late_worst = min(turb_late_worst, float(cov.min()))
            turb_late_best = max(turb_late_best, float(cov.max()))

    gt_path = np.concatenate([history[:, -1:, :, :], ground_truth], axis=1)
    gen_path = np.concatenate(
        [np.repeat(history[:, None, -1:, :, :], cond_samples.shape[1], axis=1), cond_samples],
        axis=2,
    )
    gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
    gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]
    pooled_gt = np.abs(gt_diff).reshape(-1)
    pooled_gen = np.abs(gen_diff).reshape(-1)
    pooled_kurt_ratio = _pearson_kurtosis(pooled_gen) / max(_pearson_kurtosis(pooled_gt), 1e-12)

    geometry = model.path_geometry
    gt_coeff = geometry.to_basis(torch.from_numpy(gt_diff.reshape(gt_diff.shape[0], gt_diff.shape[1], 25)).to(next(model.parameters()).device))
    gen_coeff = geometry.to_basis(torch.from_numpy(gen_diff.reshape(gen_diff.shape[0] * gen_diff.shape[1], gen_diff.shape[2], 25)).to(next(model.parameters()).device))
    masks = {
        "low": geometry.low_band_mask().reshape(1, geometry.n_frames, geometry.n_cells).to(gt_coeff.device, dtype=gt_coeff.dtype),
        "mid": geometry.mid_band_mask().reshape(1, geometry.n_frames, geometry.n_cells).to(gt_coeff.device, dtype=gt_coeff.dtype),
        "high": geometry.high_band_mask().reshape(1, geometry.n_frames, geometry.n_cells).to(gt_coeff.device, dtype=gt_coeff.dtype),
    }
    gt_total = gt_coeff.reshape(gt_coeff.shape[0], -1).pow(2).sum(dim=1).clamp_min(1e-12)
    gen_total = gen_coeff.reshape(gen_coeff.shape[0], -1).pow(2).sum(dim=1).clamp_min(1e-12)
    band_energy_ratios = {}
    for band, mask in masks.items():
        gt_band = (gt_coeff * mask).reshape(gt_coeff.shape[0], -1).pow(2).sum(dim=1) / gt_total
        gen_band = (gen_coeff * mask).reshape(gen_coeff.shape[0], -1).pow(2).sum(dim=1) / gen_total
        band_energy_ratios[f"{band}_energy_ratio_p50"] = float(torch.quantile(gen_band, 0.5) / torch.clamp(torch.quantile(gt_band, 0.5), min=1e-12))

    return {
        "frontier_turb_late_worst_cov": turb_late_worst,
        "frontier_turb_late_best_cov": turb_late_best,
        "frontier_pooled_kurt_ratio": float(pooled_kurt_ratio),
        **band_energy_ratios,
    }


def checkpoint_key(val_metrics: dict, frontier_metrics: dict, joint_metrics: dict) -> tuple[float, ...]:
    worst_gap = max(0.70 - float(frontier_metrics.get("frontier_turb_late_worst_cov", float("nan"))), 0.0)
    best_gap = max(float(frontier_metrics.get("frontier_turb_late_best_cov", float("nan"))) - 0.95, 0.0)
    kurt = float(frontier_metrics.get("frontier_pooled_kurt_ratio", float("nan")))
    kurt_gap = 1e6 if not np.isfinite(kurt) else max(0.5 - kurt, 0.0) + max(kurt - 2.0, 0.0)
    high_gap = abs(float(frontier_metrics.get("high_energy_ratio_p50", float("nan"))) - 1.0)
    mid_gap = abs(float(frontier_metrics.get("mid_energy_ratio_p50", float("nan"))) - 1.0)
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("nan")))
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    return (worst_gap, best_gap, kurt_gap, high_gap + mid_gap, mr_gap, jump_gap, tc_gap, val_loss)


@torch.no_grad()
def evaluate_teacher_forced(
    model: WidthTailControlledPathwiseResidualLawModel,
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
        "amp_total_loss",
        "amp_local_loss",
        "amp_band_loss",
        "amp_smooth_reg",
        "target_local_abs_mean",
        "pred_local_abs_mean",
        "target_band_high_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss_fm, metrics_fm = base_flow_matching_loss(model, history_01, future_01)
        _loss_amp, metrics_amp = amplitude_control_loss(model, history_01, future_01)
        metrics = {**metrics_fm, **metrics_amp}
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = (
        out["val_flow_match_loss"]
        + out["val_amp_total_loss"]
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="182b_v0: width-and-tail controlled pathwise residual law")
    parser.add_argument("--epochs_stage1", type=int, default=4)
    parser.add_argument("--epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_amp_stage1", type=float, default=8e-4)
    parser.add_argument("--lr_amp_stage2", type=float, default=3e-4)
    parser.add_argument("--lr_path_stage2", type=float, default=1.5e-4)
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
    parser.add_argument("--local_loss_weight", type=float, default=1.0)
    parser.add_argument("--band_loss_weight", type=float, default=0.75)
    parser.add_argument("--smooth_reg_weight", type=float, default=0.05)
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
    amplitude_config = dict(
        width_rank=args.width_rank,
        width_hidden_dim=args.width_hidden_dim,
        width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim,
        band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight,
        band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
    )

    model = WidthTailControlledPathwiseResidualLawModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        amplitude_config=amplitude_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)
    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, args.device)

    # Backbone always frozen in 182b_v0.
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)
    model.path_context_adapter.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            params = list(model.width_allocator.parameters()) + list(model.band_tail.parameters())
            optimizer = torch.optim.AdamW(params, lr=args.lr_amp_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "amp-only"
        else:
            model.path_transport.requires_grad_(True)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            params = [
                {"params": list(model.width_allocator.parameters()) + list(model.band_tail.parameters()), "lr": args.lr_amp_stage2},
                {"params": list(model.path_transport.parameters()), "lr": args.lr_path_stage2},
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "amp+path"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)

    total_epochs = args.epochs_stage1 + args.epochs_stage2
    print(f"\n{'=' * 72}\n182b_v0: width-and-tail controlled residual law\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start: {args.warm_start_path}")

    for epoch in range(1, total_epochs + 1):
        if epoch == args.epochs_stage1 + 1:
            optimizer, scheduler, stage_name = set_stage(2)

        t0 = time.time()
        model.train()
        metric_keys = [
            "flow_match_loss", "target_basis_std_mean", "prior_std_mean", "prior_jump_prob_mean",
            "prior_jump_scale_mean", "prior_active_rate", "pred_velocity_norm_mean",
            "target_jump_like_rate", "target_high_band_abs_mean", "amp_total_loss", "amp_local_loss",
            "amp_band_loss", "amp_smooth_reg", "target_local_abs_mean", "pred_local_abs_mean",
            "target_band_high_mean", "pred_band_high_mean", "pred_band_mid_mean", "pred_band_low_mean",
        ]
        totals = {f"train_{k}": 0.0 for k in metric_keys}
        nb = 0
        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            history_01 = history_01.to(args.device)
            future_01 = future_01.to(args.device)
            if stage_name == "amp-only":
                flow_loss = None
                metrics_fm = {
                    "flow_match_loss": torch.tensor(0.0, device=args.device),
                    "target_basis_std_mean": torch.tensor(0.0, device=args.device),
                    "prior_std_mean": torch.tensor(0.0, device=args.device),
                    "prior_jump_prob_mean": torch.tensor(0.0, device=args.device),
                    "prior_jump_scale_mean": torch.tensor(0.0, device=args.device),
                    "prior_active_rate": torch.tensor(0.0, device=args.device),
                    "pred_velocity_norm_mean": torch.tensor(0.0, device=args.device),
                    "target_jump_like_rate": torch.tensor(0.0, device=args.device),
                    "target_high_band_abs_mean": torch.tensor(0.0, device=args.device),
                }
            else:
                flow_loss, metrics_fm = base_flow_matching_loss(model, history_01, future_01)
            amp_loss, metrics_amp = amplitude_control_loss(model, history_01, future_01)
            loss = amp_loss if flow_loss is None else amp_loss + flow_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            metrics = {**metrics_fm, **metrics_amp}
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
                        "type": "width_tail_controlled_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "amplitude": amplitude_config,
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
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "width_tail_controlled_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "amplitude": amplitude_config,
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
    if best_metrics is not None:
        print(
            "\nBest diagnostics: "
            f"worstLate={best_metrics['frontier_turb_late_worst_cov']:.3f}, "
            f"bestLate={best_metrics['frontier_turb_late_best_cov']:.3f}, "
            f"kurt={best_metrics['frontier_pooled_kurt_ratio']:.3f}, "
            f"highE={best_metrics['high_energy_ratio_p50']:.3f}, "
            f"midE={best_metrics['mid_energy_ratio_p50']:.3f}, "
            f"mr={best_metrics['joint_sample_mr_ratio']:.3f}, "
            f"jumpKS={best_metrics['joint_pathwise_jump_ks']:.3f}"
        )


if __name__ == "__main__":
    main()
