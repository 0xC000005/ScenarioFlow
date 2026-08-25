#!/usr/bin/env python
"""
183a_v0: Integrated width/tail transport on top of the 182a pathwise residual-law backbone.

Principle:
  - keep the 182a mean/covariance/pathwise residual backbone
  - keep 182b's width and band-control idea
  - integrate those controls into the transport vector field itself
  - avoid post-transport amplitude patching
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
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import (
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
)
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    BandwiseRadialTailController,
    LowRankWidthAllocator,
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


class IntegratedWidthTailPathwiseResidualLawModel(
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel
):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        path_config: dict,
        prior_config: dict,
        integrated_config: dict,
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
        self.integrated_config = dict(integrated_config)
        band_id = self.path_geometry.band_id.reshape(-1)
        band_counts = torch.stack([(band_id == i).sum() for i in range(3)]).float()
        self.width_allocator = LowRankWidthAllocator(
            context_dim=path_config["context_dim"],
            n_frames=decoder_config["n_frames"],
            n_cells=decoder_config["n_cells"],
            rank=integrated_config["width_rank"],
            hidden_dim=integrated_config["width_hidden_dim"],
            clip=integrated_config["width_clip"],
        )
        self.band_tail = BandwiseRadialTailController(
            context_dim=path_config["context_dim"],
            band_counts=band_counts,
            hidden_dim=integrated_config["band_hidden_dim"],
            clip=integrated_config["band_clip"],
        )
        self.local_strength_logit = nn.Parameter(torch.tensor(_logit(integrated_config["init_local_strength"])))
        self.band_strength_logit = nn.Parameter(torch.tensor(_logit(integrated_config["init_band_strength"])))
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
            if key.startswith("width_allocator.") or key.startswith("band_tail.") or key.startswith("local_strength_logit") or key.startswith("band_strength_logit"):
                skipped.append(key)
                continue
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Backbone warm start loaded from {ckpt_path}")
        print(
            f"  Warm start missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | "
            f"shape-skipped: {len(skipped)}"
        )

    def maybe_load_control_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        for key in ("width_allocator.", "band_tail."):
            for state_key, value in state.items():
                if state_key.startswith(key) and state_key in model_state and model_state[state_key].shape == value.shape:
                    filtered[state_key] = value
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Control warm start loaded from {ckpt_path}")
        print(f"  Control warm start missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    def build_integrated_controls(
        self,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_local = self.width_allocator(path_context)
        raw_band = self.band_tail(path_context)
        local_strength = torch.sigmoid(self.local_strength_logit)
        band_strength = torch.sigmoid(self.band_strength_logit)
        eff_local = raw_local * local_strength
        eff_band = raw_band * band_strength
        return raw_local, raw_band, eff_local, eff_band

    def modulate_velocity_basis(
        self,
        basis_velocity: torch.Tensor,
        eff_local: torch.Tensor,
        eff_band: torch.Tensor,
    ) -> torch.Tensor:
        band_map = torch.einsum(
            "bk,ktc->btc",
            eff_band,
            self.band_masks.to(device=basis_velocity.device, dtype=basis_velocity.dtype),
        )
        basis_scaled = basis_velocity * torch.exp(band_map)
        white = self.path_geometry.from_basis(basis_scaled)
        white = white * torch.exp(eff_local)
        return self.path_geometry.to_basis(white)

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_local, raw_band, eff_local, eff_band = self.build_integrated_controls(path_context)
        raw_v = self.path_transport(z_t_flat, t, path_context)
        raw_v_basis = raw_v.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        mod_v_basis = self.modulate_velocity_basis(raw_v_basis, eff_local, eff_band)
        metrics = {
            "pred_local_abs_mean": raw_local.abs().mean(),
            "pred_band_high_mean": raw_band[:, 2].mean(),
            "pred_band_mid_mean": raw_band[:, 1].mean(),
            "pred_band_low_mean": raw_band[:, 0].mean(),
            "local_strength": torch.sigmoid(self.local_strength_logit),
            "band_strength": torch.sigmoid(self.band_strength_logit),
            "pred_velocity_norm_mean": mod_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return mod_v_basis.reshape(z_t_flat.shape[0], -1), metrics

    def sample_basis_paths(self, path_context: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = path_context.shape[0]
        z, prior_stats = self.prior.sample(batch * n_samples, device=path_context.device, dtype=path_context.dtype)
        ctx = path_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        dt = 1.0 / float(self.n_ode_steps)
        local_strength = torch.sigmoid(self.local_strength_logit).detach()
        band_strength = torch.sigmoid(self.band_strength_logit).detach()
        for step in range(self.n_ode_steps):
            t = torch.full(
                (batch * n_samples,),
                fill_value=step * dt,
                device=path_context.device,
                dtype=path_context.dtype,
            )
            v, _metrics = self.transport_velocity(z, t, ctx)
            z = z + dt * v
        stats = dict(prior_stats)
        stats["sample_basis_std_mean"] = z.std(dim=-1).mean()
        stats["sample_local_strength"] = local_strength
        stats["sample_band_strength"] = band_strength
        return z, stats


def compute_control_targets(
    model: IntegratedWidthTailPathwiseResidualLawModel,
    target_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_white = model.path_geometry.from_basis(target_basis)
    target_local_log = torch.log(target_white.abs().clamp_min(1e-4))
    target_local_log = target_local_log - target_local_log.mean(dim=(1, 2), keepdim=True)
    target_local_log = target_local_log.clamp(
        min=-model.integrated_config["width_clip"],
        max=model.integrated_config["width_clip"],
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
        min=-model.integrated_config["band_clip"],
        max=model.integrated_config["band_clip"],
    )
    return target_local_log, target_band_log


def integrated_flow_matching_loss(
    model: IntegratedWidthTailPathwiseResidualLawModel,
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
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0
    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    fm_loss = F.mse_loss(pred_v, target_v)

    raw_local, raw_band, _eff_local, _eff_band = model.build_integrated_controls(path_context)
    local_loss = F.smooth_l1_loss(raw_local, target_local_log)
    band_loss = F.smooth_l1_loss(raw_band, target_band_log)
    surf = raw_local.view(raw_local.shape[0], raw_local.shape[1], 5, 5)
    time_smooth = (raw_local[:, 1:] - raw_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth

    total = (
        fm_loss
        + model.integrated_config["local_loss_weight"] * local_loss
        + model.integrated_config["band_loss_weight"] * band_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
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
        "target_high_band_abs_mean": ((target_basis_flat.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)).mean(),
        "int_total_loss": total,
        "int_local_loss": local_loss,
        "int_band_loss": band_loss,
        "int_smooth_reg": smooth_reg,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: IntegratedWidthTailPathwiseResidualLawModel,
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
        "int_total_loss",
        "int_local_loss",
        "int_band_loss",
        "int_smooth_reg",
        "target_local_abs_mean",
        "target_band_high_mean",
        "pred_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "local_strength",
        "band_strength",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = integrated_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_int_total_loss"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="183a_v0: integrated width/tail transport")
    parser.add_argument("--epochs_stage1", type=int, default=4)
    parser.add_argument("--epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_ctrl_stage1", type=float, default=6e-4)
    parser.add_argument("--lr_ctrl_stage2", type=float, default=2.5e-4)
    parser.add_argument("--lr_path_stage2", type=float, default=1.0e-4)
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
    parser.add_argument("--local_loss_weight", type=float, default=0.75)
    parser.add_argument("--band_loss_weight", type=float, default=0.50)
    parser.add_argument("--smooth_reg_weight", type=float, default=0.04)
    parser.add_argument("--init_local_strength", type=float, default=0.05)
    parser.add_argument("--init_band_strength", type=float, default=0.05)
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--control_warm_start_path", type=str, default=None)
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
    integrated_config = dict(
        width_rank=args.width_rank,
        width_hidden_dim=args.width_hidden_dim,
        width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim,
        band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight,
        band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
        init_local_strength=args.init_local_strength,
        init_band_strength=args.init_band_strength,
    )

    model = IntegratedWidthTailPathwiseResidualLawModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)
    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, args.device)
    if args.control_warm_start_path:
        model.maybe_load_control_warm_start(args.control_warm_start_path, args.device)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.flow.requires_grad_(False)
    model.prior.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            params = list(model.width_allocator.parameters()) + list(model.band_tail.parameters()) + [model.local_strength_logit, model.band_strength_logit]
            optimizer = torch.optim.AdamW(params, lr=args.lr_ctrl_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "integrated-ctrl"
        else:
            model.path_transport.requires_grad_(True)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(True)
            model.band_tail.requires_grad_(True)
            params = [
                {
                    "params": list(model.width_allocator.parameters()) + list(model.band_tail.parameters()) + [model.local_strength_logit, model.band_strength_logit],
                    "lr": args.lr_ctrl_stage2,
                },
                {
                    "params": list(model.path_transport.parameters()) + list(model.path_context_adapter.parameters()),
                    "lr": args.lr_path_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "integrated-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n183a_v0: integrated width/tail transport\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start backbone: {args.warm_start_path}")
    print(f"  Warm start controls: {args.control_warm_start_path}")

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
        "int_total_loss",
        "int_local_loss",
        "int_band_loss",
        "int_smooth_reg",
        "target_local_abs_mean",
        "target_band_high_mean",
        "pred_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "local_strength",
        "band_strength",
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
            loss, metrics = integrated_flow_matching_loss(model, history_01, future_01)
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
                        "type": "integrated_width_tail_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
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
            f"lStr={val_metrics['val_local_strength']:.3f}  bStr={val_metrics['val_band_strength']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "integrated_width_tail_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
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
