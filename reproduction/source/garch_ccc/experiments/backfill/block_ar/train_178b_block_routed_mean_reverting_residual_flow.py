#!/usr/bin/env python
"""
178b: Block-routed uncertainty on top of the 177a realism-side backbone.

Motivation from 178a mechanistic analysis:
  - the latent regime SSM collapsed entirely to one regime
  - the new drift law over-reverted badly
  - local uncertainty remained nearly static despite the new latent machinery
  - removing the residual flow likely contributed to the S4/S8 collapse

Keep 177a:
  - explicit mean-reverting drift anchored to history
  - support-aware transformed-space modeling
  - conditional residual flow
  - shared global covariance backbone

Change only the uncertainty branch:
  - replace one-window local-template routing with blockwise persistent routing
  - one routed template per horizon block, shared across the block
  - no latent posterior / KL; use deterministic condition-responsive routing
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
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_172a_residual_flow_structured_joint_student_t import (
    ConditionalResidualFlow,
    ResidualFlowStructuredJointDecoder,
    ResidualFlowStructuredJointStudentTModel,
)
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import (
    aggregate_slope_ratio,
)


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


class BlockRoutedMeanRevertingResidualFlowDecoder(ResidualFlowStructuredJointDecoder):
    """177a drift path plus blockwise persistent local-variance routing."""

    def __init__(
        self,
        *args,
        n_blocks: int = 5,
        n_templates: int = 3,
        local_delta_clip: float = 0.35,
        drift_strength_max: float = 0.75,
        equilibrium_offset_clip: float = 0.20,
        init_drift_strength: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_blocks = n_blocks
        self.block_len = self.n_frames // n_blocks
        self.n_templates = n_templates
        self.local_delta_clip = local_delta_clip
        self.drift_strength_max = drift_strength_max
        self.equilibrium_offset_clip = equilibrium_offset_clip

        d_model = self.mean_head.in_features
        self.equilibrium_offset_head = nn.Linear(d_model, 1)
        self.drift_strength_head = nn.Linear(d_model, 1)
        self.local_logvar_head = nn.Linear(d_model, 1)
        self.static_local_logvar = nn.Parameter(torch.zeros(self.n_frames, self.n_cells))
        self.block_template_bank = nn.Parameter(torch.zeros(n_templates, n_blocks, self.n_cells))
        nn.init.normal_(self.block_template_bank, mean=0.0, std=1e-3)
        self.block_gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_templates),
        )

        nn.init.zeros_(self.equilibrium_offset_head.weight)
        nn.init.zeros_(self.equilibrium_offset_head.bias)
        nn.init.zeros_(self.drift_strength_head.weight)
        init_frac = min(max(init_drift_strength / max(drift_strength_max, 1e-6), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.drift_strength_head.bias, logit(init_frac))
        nn.init.zeros_(self.local_logvar_head.weight)
        nn.init.zeros_(self.local_logvar_head.bias)
        for module in self.block_gate_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.block_gate_head[-1].weight)

    def forward(self, cond: torch.Tensor, prev_u: torch.Tensor, hist_mean_u: torch.Tensor):
        batch = prev_u.shape[0]
        n_frames, n_cells = self.n_frames, self.n_cells

        prev_rep = prev_u.unsqueeze(1).expand(batch, n_frames, n_cells)
        hist_mean_rep = hist_mean_u.unsqueeze(1).expand(batch, n_frames, n_cells)
        hidden = self.input_proj(prev_rep.unsqueeze(-1))
        hidden = hidden + self.cond_proj(cond).unsqueeze(1).unsqueeze(1)
        hidden = hidden + self.temporal_pos + self.spatial_pos

        for layer in self.layers:
            h_temp = hidden.permute(0, 2, 1, 3).reshape(batch * n_cells, n_frames, -1)
            h_norm = layer["temp_norm"](h_temp)
            attn_out, _ = layer["temp_attn"](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer["temp_ff"](layer["temp_ff_norm"](h_temp))
            hidden = h_temp.reshape(batch, n_cells, n_frames, -1).permute(0, 2, 1, 3)

            h_spat = hidden.reshape(batch * n_frames, n_cells, -1)
            h_norm = layer["spat_norm"](h_spat)
            attn_out, _ = layer["spat_attn"](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer["spat_ff"](layer["spat_ff_norm"](h_spat))
            hidden = h_spat.reshape(batch, n_frames, n_cells, -1)

        hidden = self.out_norm(hidden)
        pooled = hidden.mean(dim=(1, 2))
        time_summary = hidden.mean(dim=2)
        cell_summary = hidden.mean(dim=1)

        base_delta = self.mean_head(hidden).squeeze(-1)
        equilibrium_offset = torch.tanh(self.equilibrium_offset_head(hidden).squeeze(-1)) * self.equilibrium_offset_clip
        equilibrium = hist_mean_rep + equilibrium_offset
        drift_strength = torch.sigmoid(self.drift_strength_head(hidden).squeeze(-1)) * self.drift_strength_max
        drift_term = drift_strength * (equilibrium - prev_rep)
        mu = prev_rep + base_delta + drift_term

        time_factor = self.time_factor_head(time_summary)
        time_diag = F.softplus(self.time_diag_head(time_summary).squeeze(-1)) + self.diag_floor
        cell_factor = self.cell_factor_head(cell_summary)
        cell_diag = F.softplus(self.cell_diag_head(cell_summary).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        flow_context = self.flow_context_head(torch.cat([cond, prev_u], dim=-1))

        base_local_delta = torch.tanh(self.local_logvar_head(hidden).squeeze(-1)) * self.local_delta_clip
        block_summary = time_summary.view(batch, self.n_blocks, self.block_len, -1).mean(dim=2)
        block_logits = self.block_gate_head(block_summary)
        block_probs = F.softmax(block_logits, dim=-1)
        block_delta = torch.einsum("bkn,nkc->bkc", block_probs, self.block_template_bank)
        block_delta = block_delta.unsqueeze(2).expand(batch, self.n_blocks, self.block_len, n_cells).reshape(batch, n_frames, n_cells)

        local_delta_raw = base_local_delta + self.static_local_logvar.unsqueeze(0) + block_delta
        local_delta = torch.tanh(local_delta_raw) * self.local_delta_clip
        local_delta = local_delta - local_delta.mean(dim=(1, 2), keepdim=True)

        return (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            local_delta,
            block_probs,
        )


class BlockRoutedMeanRevertingResidualFlowStructuredJointStudentTModel(
    ResidualFlowStructuredJointStudentTModel
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
    ):
        base_decoder_config = dict(decoder_config)
        for key in (
            "local_delta_clip",
            "n_blocks",
            "n_templates",
            "drift_strength_max",
            "equilibrium_offset_clip",
            "init_drift_strength",
        ):
            base_decoder_config.pop(key, None)
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=base_decoder_config,
            flow_config=flow_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
        )
        self.decoder = BlockRoutedMeanRevertingResidualFlowDecoder(**decoder_config)
        self.decoder_config = decoder_config
        self.flow = ConditionalResidualFlow(**flow_config)
        self.flow_config = flow_config

    def forward_from_history(self, history_01: torch.Tensor):
        cond = self.encode(history_01)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        hist_mean_01 = history_01.mean(dim=1).reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        hist_mean_u = iv_to_unconstrained(hist_mean_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        return self.decoder(cond, prev_u, hist_mean_u)

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
        local_delta: torch.Tensor,
        block_probs: torch.Tensor,
    ):
        batch, n_frames, n_cells = target_u.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        local_scale = torch.exp(0.5 * local_delta)
        diff = (target_u - mu) / local_scale
        white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
        white = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
        white_flat = white.reshape(batch, n_frames * n_cells)

        z, flow_logdet = self.flow(white_flat, flow_context)
        logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
        logdet_cov = n_cells * logdet_t + n_frames * logdet_c
        logdet_local = 2.0 * torch.log(local_scale).sum(dim=(1, 2))

        base_logprob = self._base_logprob(z)
        logprob = base_logprob + flow_logdet - 0.5 * (logdet_cov + logdet_local)

        aux = {
            "cov_t": cov_t,
            "cov_c": cov_c,
            "flow_logdet": flow_logdet,
            "white_std": white_flat.std(dim=-1),
            "z_std": z.std(dim=-1),
            "local_delta_rms": local_delta.pow(2).mean(dim=(1, 2)).sqrt(),
            "local_scale_min": local_scale.amin(dim=(1, 2)),
            "local_scale_max": local_scale.amax(dim=(1, 2)),
            "block_gate_entropy": (-(block_probs * torch.log(block_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1),
            "block_gate_max": block_probs.max(dim=-1).values.mean(dim=-1),
        }
        return logprob, aux

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
            _base_local_delta,
            local_delta,
            _block_probs,
        ) = self.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = self.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        base = torch.distributions.StudentT(df=self.base_nu)
        z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
        ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        white_flat, _ = self.flow.inverse(z, ctx)
        white = white_flat.view(batch, n_samples, n_frames, n_cells)

        temp = torch.einsum("bij,bsjk->bsik", chol_t, white)
        noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
        local_scale = torch.exp(0.5 * local_delta).unsqueeze(1)
        return mu.unsqueeze(1) + noise * local_scale


def joint_nll_loss(
    model: BlockRoutedMeanRevertingResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    template_penalty: float,
    gate_smooth_penalty: float,
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
        local_delta,
        block_probs,
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
        local_delta=local_delta,
        block_probs=block_probs,
    )

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    local_pen = base_local_delta.pow(2).mean()
    template_l2 = model.decoder.block_template_bank.pow(2).mean()
    static_l2 = model.decoder.static_local_logvar.pow(2).mean()
    smooth_pen = (block_probs[:, 1:] - block_probs[:, :-1]).pow(2).mean() if block_probs.shape[1] > 1 else block_probs.new_tensor(0.0)
    nll = (-logprob).mean()
    loss = nll + local_var_penalty * (local_pen + 0.25 * static_l2) + template_penalty * template_l2 + gate_smooth_penalty * smooth_pen

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
        "smooth_penalty": smooth_pen,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "joint_det_mr_ratio": pred_01.new_tensor(det_mr_ratio),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": aux["flow_logdet"].mean(),
        "white_std_mean": aux["white_std"].mean(),
        "z_std_mean": aux["z_std"].mean(),
        "local_delta_rms": aux["local_delta_rms"].mean(),
        "local_scale_min": aux["local_scale_min"].mean(),
        "local_scale_max": aux["local_scale_max"].mean(),
        "block_gate_entropy": aux["block_gate_entropy"].mean(),
        "block_gate_max": aux["block_gate_max"].mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: BlockRoutedMeanRevertingResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    template_penalty: float,
    gate_smooth_penalty: float,
):
    model.eval()
    totals = {f"val_{k}": 0.0 for k in [
        "total_loss","joint_nll","local_var_penalty","template_l2","smooth_penalty",
        "joint_mae","joint_det_mr_ratio","time_eff_rank","cell_eff_rank","scale_mean",
        "flow_logdet_mean","white_std_mean","z_std_mean","local_delta_rms","local_scale_min",
        "local_scale_max","block_gate_entropy","block_gate_max"
    ]}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = joint_nll_loss(
            model, history_01, future_01,
            local_var_penalty=local_var_penalty,
            template_penalty=template_penalty,
            gate_smooth_penalty=gate_smooth_penalty,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    return {k: v / max(total_count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: BlockRoutedMeanRevertingResidualFlowStructuredJointStudentTModel,
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
            "joint_turb_calm_ratio","joint_sample_eff_rank","joint_det_mr_ratio","joint_sample_mr_ratio"
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

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
        "joint_det_mr_ratio": det_mr_ratio,
        "joint_sample_mr_ratio": sample_mr_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="178b: block-routed mean-reverting residual flow")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
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
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
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
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--template_penalty", type=float, default=0.25)
    parser.add_argument("--gate_smooth_penalty", type=float, default=0.5)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
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
        drift_strength_max=args.drift_strength_max,
        equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = BlockRoutedMeanRevertingResidualFlowStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("178b: block-routed mean-reverting residual flow")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {f"train_{k}": 0.0 for k in [
            "total_loss","joint_nll","local_var_penalty","template_l2","smooth_penalty",
            "joint_mae","joint_det_mr_ratio","time_eff_rank","cell_eff_rank","scale_mean",
            "flow_logdet_mean","white_std_mean","z_std_mean","local_delta_rms","local_scale_min",
            "local_scale_max","block_gate_entropy","block_gate_max"
        ]}
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model, history_01, future_01,
                local_var_penalty=args.local_var_penalty,
                template_penalty=args.template_penalty,
                gate_smooth_penalty=args.gate_smooth_penalty,
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
            model, val_loader,
            local_var_penalty=args.local_var_penalty,
            template_penalty=args.template_penalty,
            gate_smooth_penalty=args.gate_smooth_penalty,
        )
        joint_metrics = evaluate_joint_subset(
            model, val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        elapsed = time.time() - t0
        is_best = val_metrics["val_total_loss"] < best_val
        if is_best:
            best_val = val_metrics["val_total_loss"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": best_val,
                    "config": {
                        "type": "block_routed_mean_reverting_residual_flow_structured_joint_student_t_178b",
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
                        "local_var_penalty": args.local_var_penalty,
                        "template_penalty": args.template_penalty,
                        "gate_smooth_penalty": args.gate_smooth_penalty,
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
            f"val_loss={val_metrics['val_total_loss']:.4f}  val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"mr_det={joint_metrics['joint_det_mr_ratio']:.3f}  mr_samp={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"gateH={val_metrics['val_block_gate_entropy']:.3f}  gateMax={val_metrics['val_block_gate_max']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "block_routed_mean_reverting_residual_flow_structured_joint_student_t_178b",
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
            "local_var_penalty": args.local_var_penalty,
            "template_penalty": args.template_penalty,
            "gate_smooth_penalty": args.gate_smooth_penalty,
        },
        "best_val_total_loss": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    print(f"\nBest val total loss: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_det_mr_ratio={best_metrics['joint_det_mr_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
