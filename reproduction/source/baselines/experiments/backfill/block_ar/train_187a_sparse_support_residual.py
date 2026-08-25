#!/usr/bin/env python
"""
187a_v0: Latent sparse-support residual model on top of the 183c backbone.

Principle:
  - keep the explicit mean-reverting mean branch and structured covariance branch
  - keep the validated 183c quiet residual backbone for ordinary residual behavior
  - replace smooth concentration fixes with a latent sparse support process over
    time x node and a separate event amplitude law on active support
  - make selective concentration part of the residual law itself
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
from experiments.backfill.block_ar.activity_geometry import StructuredResidualActivityGeometry
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import ConditionalPathFlowTransformer
from experiments.backfill.block_ar.train_182b_width_tail_control import evaluate_frontier_subset
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
    _logit,
    _scaled_logit,
)


def bernoulli_kl(post_p: torch.Tensor, prior_p: torch.Tensor) -> torch.Tensor:
    post_p = post_p.clamp(1e-5, 1.0 - 1e-5)
    prior_p = prior_p.clamp(1e-5, 1.0 - 1e-5)
    return post_p * (post_p.log() - prior_p.log()) + (1.0 - post_p) * (
        (1.0 - post_p).log() - (1.0 - prior_p).log()
    )


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def relaxed_bernoulli_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp_(1e-5, 1.0 - 1e-5)
    logistic = torch.log(u) - torch.log1p(-u)
    return torch.sigmoid((logits + logistic) / max(temperature, 1e-6))


class SupportLogitHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, init_prob: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _logit(init_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CenteredTanhHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(x))


class SparseSupportResidualModel(StateMetricTransportModel):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        path_config: dict,
        prior_config: dict,
        integrated_config: dict,
        state_config: dict,
        metric_config: dict,
        support_config: dict,
        event_config: dict,
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
            integrated_config=integrated_config,
            state_config=state_config,
            metric_config=metric_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
            mix_chunk_size=mix_chunk_size,
        )
        self.support_config = dict(support_config)
        self.event_config = dict(event_config)
        group_ids = self.support_config.get("group_ids")
        self.activity_geometry = StructuredResidualActivityGeometry(
            n_frames=self.decoder.n_frames,
            n_cells=self.decoder.n_cells,
            group_ids=group_ids,
        )
        self.n_support_blocks = int(self.support_config["n_support_blocks"])
        block_ids = torch.arange(self.decoder.n_frames, dtype=torch.long) * self.n_support_blocks // self.decoder.n_frames
        self.register_buffer("support_block_ids", block_ids, persistent=False)

        self.support_context = nn.Linear(path_config["context_dim"], self.support_config["context_feat_dim"])
        nn.init.zeros_(self.support_context.weight)
        nn.init.zeros_(self.support_context.bias)

        prior_in = 7 + self.support_config["context_feat_dim"]
        post_in = prior_in + 1
        self.support_prior_head = SupportLogitHead(
            input_dim=prior_in,
            hidden_dim=self.support_config["support_hidden_dim"],
            init_prob=self.support_config["init_support_rate"],
        )
        self.support_post_head = SupportLogitHead(
            input_dim=post_in,
            hidden_dim=self.support_config["support_hidden_dim"],
            init_prob=self.support_config["init_support_rate"],
        )
        self.event_amplitude_head = CenteredTanhHead(
            input_dim=prior_in,
            hidden_dim=self.event_config["event_hidden_dim"],
            output_dim=1,
            scale=self.event_config["event_amp_max"],
        )
        self.event_band_head = CenteredTanhHead(
            input_dim=path_config["context_dim"] + 4,
            hidden_dim=self.event_config["event_hidden_dim"],
            output_dim=3,
            scale=self.event_config["event_band_max"],
        )
        self.event_transport = ConditionalPathFlowTransformer(
            n_frames=decoder_config["n_frames"],
            n_cells=decoder_config["n_cells"],
            context_dim=path_config["context_dim"],
            d_model=path_config["d_model"],
            n_heads=path_config["n_heads"],
            n_layers=max(2, path_config["n_layers"] // 2),
            ff_mult=path_config["ff_mult"],
            time_embed_dim=path_config["time_embed_dim"],
        )
        self.event_scale_logit = nn.Parameter(
            torch.tensor(
                _scaled_logit(
                    self.event_config["init_event_scale"],
                    0.0,
                    self.event_config["event_scale_max"],
                )
            )
        )

        adjacency = self.activity_geometry.local_adjacency(include_self=True)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("normalized_adjacency", adjacency / degree, persistent=False)

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith(
                (
                    "support_context.",
                    "support_prior_head.",
                    "support_post_head.",
                    "event_amplitude_head.",
                    "event_band_head.",
                    "event_transport.",
                    "event_scale_logit",
                )
            ):
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

    def event_scale(self) -> torch.Tensor:
        return self.event_config["event_scale_max"] * torch.sigmoid(self.event_scale_logit)

    def build_quiet_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().build_state_metric_controls(z_t_flat, t, path_context)

    def build_node_features(
        self,
        state_local: torch.Tensor,
        quiet_metric_local: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> torch.Tensor:
        abs_state = state_local.abs()
        abs_quiet = quiet_metric_local.abs()
        neighbor_abs = torch.einsum(
            "ij,btj->bti",
            self.normalized_adjacency.to(device=state_local.device, dtype=state_local.dtype),
            abs_state,
        )
        group_abs = self.activity_geometry.broadcast_group(self.activity_geometry.pool_group(abs_state)).to(
            device=state_local.device,
            dtype=state_local.dtype,
        )
        ctx = self.support_context(path_context)
        ctx = ctx[:, None, None, :].expand(-1, self.decoder.n_frames, self.decoder.n_cells, -1)
        t_feat = t[:, None, None, None].expand(-1, self.decoder.n_frames, self.decoder.n_cells, 1)
        base = torch.cat(
            [
                state_local.unsqueeze(-1),
                quiet_metric_local.unsqueeze(-1),
                abs_state.unsqueeze(-1),
                abs_quiet.unsqueeze(-1),
                neighbor_abs.unsqueeze(-1),
                group_abs.unsqueeze(-1),
                t_feat,
                ctx,
            ],
            dim=-1,
        )
        return base

    def infer_support_process(
        self,
        path_context: torch.Tensor,
        state_local: torch.Tensor,
        quiet_metric_local: torch.Tensor,
        t: torch.Tensor,
        teacher_abs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.build_node_features(state_local, quiet_metric_local, t, path_context)
        prior_logits = self.support_prior_head(base)
        prior_probs = torch.sigmoid(prior_logits)
        out = {
            "prior_logits": prior_logits,
            "prior_probs": prior_probs,
        }
        if teacher_abs is not None:
            post_in = torch.cat([base, teacher_abs.unsqueeze(-1)], dim=-1)
            post_logits = self.support_post_head(post_in)
            post_probs = torch.sigmoid(post_logits)
            out["post_logits"] = post_logits
            out["post_probs"] = post_probs
        else:
            out["post_logits"] = prior_logits
            out["post_probs"] = prior_probs
        return out

    def build_teacher_support(
        self,
        target_event_local: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_frames, n_cells = target_event_local.shape
        score = target_event_local.abs()
        support = torch.zeros_like(score)
        topk = int(self.support_config["teacher_topk"])
        threshold = float(self.support_config["teacher_threshold"])
        for block_idx in range(self.n_support_blocks):
            frame_mask = self.support_block_ids == block_idx
            block_score = score[:, frame_mask]
            flat = block_score.reshape(batch, -1)
            valid = flat.shape[1]
            k = max(1, min(topk, valid))
            topi = flat.topk(k, dim=1).indices
            flat_support = torch.zeros_like(flat)
            flat_support.scatter_(1, topi, 1.0)
            if threshold > 0.0:
                flat_support = flat_support * (flat >= threshold).to(flat.dtype)
            empty = flat_support.sum(dim=1) <= 0
            if empty.any():
                fallback = flat.topk(1, dim=1).indices
                flat_support[empty] = 0.0
                flat_support.scatter_(1, fallback, 1.0)
            support[:, frame_mask] = flat_support.view(batch, int(frame_mask.sum().item()), n_cells)
        return support

    def build_teacher_event_components(
        self,
        target_basis: torch.Tensor,
        quiet_metric_local: torch.Tensor,
        quiet_metric_band: torch.Tensor,
        target_local_log: torch.Tensor,
        target_band_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_event_local_full = (target_local_log - quiet_metric_local).clamp(
            min=-self.event_config["event_local_clip"],
            max=self.event_config["event_local_clip"],
        )
        support = self.build_teacher_support(target_event_local_full)
        target_event_local = target_event_local_full * support
        target_white = self.path_geometry.from_basis(target_basis)
        target_event_white = target_white * support
        target_event_basis = self.path_geometry.to_basis(target_event_white)
        target_event_band = (target_band_log - quiet_metric_band).clamp(
            min=-self.event_config["event_band_clip"],
            max=self.event_config["event_band_clip"],
        )
        return support, target_event_local, target_event_basis, target_event_band

    def build_sparse_support_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        support_pack: dict[str, torch.Tensor] | None = None,
        use_posterior: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raw_local, raw_band, quiet_metric_local, quiet_metric_band, local_gate, band_gate = self.build_quiet_controls(
            z_t_flat,
            t,
            path_context,
        )
        state_local, state_band = self.build_state_features(z_t_flat)
        if support_pack is None:
            support_pack = self.infer_support_process(path_context, state_local, quiet_metric_local, t)
        logits_key = "post_logits" if use_posterior and "post_logits" in support_pack else "prior_logits"
        probs_key = "post_probs" if use_posterior and "post_probs" in support_pack else "prior_probs"
        support_logits = support_pack[logits_key]
        support_probs = support_pack[probs_key]
        if self.training:
            support_sample = relaxed_bernoulli_sample(support_logits, self.support_config["relax_temperature"])
        else:
            support_sample = support_probs

        node_features = self.build_node_features(state_local, quiet_metric_local, t, path_context)
        event_amp = self.event_amplitude_head(node_features).squeeze(-1)
        event_local = (support_sample * event_amp).clamp(
            min=-self.event_config["event_local_clip"],
            max=self.event_config["event_local_clip"],
        )

        band_in = torch.cat([path_context, state_band, t.unsqueeze(-1)], dim=-1)
        event_band = self.event_band_head(band_in)
        support_rate = support_probs.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        event_band = event_band * support_rate
        band_weights = (self.band_masks.reshape(3, -1).sum(dim=1) / self.band_masks.numel()).to(
            event_band.device,
            dtype=event_band.dtype,
        )
        event_band = event_band - (event_band * band_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        event_band = event_band.clamp(
            min=-self.event_config["event_band_clip"],
            max=self.event_config["event_band_clip"],
        )

        return (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            support_logits,
            support_probs,
            support_sample,
            event_amp,
        )

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.transport_velocity_with_pack(z_t_flat, t, path_context, support_pack=None, use_posterior=False)

    def transport_velocity_with_pack(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        support_pack: dict[str, torch.Tensor] | None = None,
        use_posterior: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            support_logits,
            support_probs,
            _support_sample,
            event_amp,
        ) = self.build_sparse_support_controls(
            z_t_flat,
            t,
            path_context,
            support_pack=support_pack,
            use_posterior=use_posterior,
        )
        quiet_raw = self.path_transport(z_t_flat, t, path_context)
        quiet_raw_basis = quiet_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        quiet_v_basis = self.modulate_velocity_basis(quiet_raw_basis, quiet_metric_local, quiet_metric_band)

        event_raw = self.event_transport(z_t_flat, t, path_context)
        event_raw_basis = event_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        event_v_basis = self.modulate_velocity_basis(event_raw_basis, event_local, event_band)
        total_v_basis = quiet_v_basis + self.event_scale() * event_v_basis
        flat_support = support_probs.reshape(support_probs.shape[0], -1)
        metrics = {
            "pred_local_abs_mean": raw_local.abs().mean(),
            "quiet_metric_local_abs_mean": quiet_metric_local.abs().mean(),
            "event_local_abs_mean": event_local.abs().mean(),
            "pred_band_high_mean": raw_band[:, 2].mean(),
            "pred_band_mid_mean": raw_band[:, 1].mean(),
            "pred_band_low_mean": raw_band[:, 0].mean(),
            "quiet_band_high_mean": quiet_metric_band[:, 2].mean(),
            "quiet_band_mid_mean": quiet_metric_band[:, 1].mean(),
            "quiet_band_low_mean": quiet_metric_band[:, 0].mean(),
            "event_band_high_mean": event_band[:, 2].mean(),
            "event_band_mid_mean": event_band[:, 1].mean(),
            "event_band_low_mean": event_band[:, 0].mean(),
            "local_metric_budget": self.local_metric_budget(),
            "band_metric_budget": self.band_metric_budget(),
            "event_scale": self.event_scale(),
            "local_state_gate_mean": local_gate.mean(),
            "band_state_gate_mean": band_gate.mean(),
            "band_state_gate_high_mean": band_gate[:, 2].mean(),
            "prior_support_rate": support_probs.mean(),
            "prior_support_top1_mean": flat_support.max(dim=1).values.mean(),
            "support_logit_abs_mean": support_logits.abs().mean(),
            "event_amp_abs_mean": event_amp.abs().mean(),
            "quiet_velocity_norm_mean": quiet_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "event_velocity_norm_mean": event_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "pred_velocity_norm_mean": total_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return total_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def strict_checkpoint_key(val_metrics: dict, frontier_metrics: dict, joint_metrics: dict) -> tuple[float, ...]:
    tc = float(joint_metrics.get("joint_turb_calm_ratio", float("nan")))
    tc_gap = 1e6 if not np.isfinite(tc) else max(1.15 - tc, 0.0)
    worst_gap = max(0.70 - float(frontier_metrics.get("frontier_turb_late_worst_cov", float("nan"))), 0.0)
    best_gap = max(float(frontier_metrics.get("frontier_turb_late_best_cov", float("nan"))) - 0.95, 0.0)
    kurt = float(frontier_metrics.get("frontier_pooled_kurt_ratio", float("nan")))
    kurt_gap = 1e6 if not np.isfinite(kurt) else max(0.8 - kurt, 0.0) + max(kurt - 1.25, 0.0)
    mr_ratio = float(joint_metrics.get("joint_sample_mr_ratio", float("nan")))
    mr_gap = 1e6 if not np.isfinite(mr_ratio) else abs(mr_ratio - 1.0)
    jump_ks = float(joint_metrics.get("joint_pathwise_jump_ks", float("nan")))
    jump_gap = 1e6 if not np.isfinite(jump_ks) else jump_ks
    rate_gap = abs(float(val_metrics.get("val_prior_support_rate", 0.0)) - float(val_metrics.get("val_target_support_rate", 0.0)))
    amp_gap = max(0.02 - float(val_metrics.get("val_pred_event_basis_abs_mean", 0.0)), 0.0)
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    return (tc_gap, worst_gap, best_gap, kurt_gap, rate_gap, amp_gap, jump_gap, mr_gap, val_loss)


def sparse_support_flow_matching_loss(
    model: SparseSupportResidualModel,
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

    state_local, _state_band = model.build_state_features(z_t)
    (
        _raw_local,
        _raw_band,
        quiet_metric_local,
        quiet_metric_band,
        _local_gate,
        _band_gate,
    ) = model.build_quiet_controls(z_t, t, path_context)
    (
        teacher_support,
        target_event_local,
        target_event_basis,
        target_event_band,
    ) = model.build_teacher_event_components(
        target_basis=target_basis,
        quiet_metric_local=quiet_metric_local,
        quiet_metric_band=quiet_metric_band,
        target_local_log=target_local_log,
        target_band_log=target_band_log,
    )
    teacher_abs = target_event_local.abs()
    support_pack = model.infer_support_process(
        path_context=path_context,
        state_local=state_local,
        quiet_metric_local=quiet_metric_local,
        t=t,
        teacher_abs=teacher_abs,
    )
    pred_v, control_metrics = model.transport_velocity_with_pack(
        z_t,
        t,
        path_context,
        support_pack=support_pack,
        use_posterior=True,
    )
    fm_loss = F.mse_loss(pred_v, target_v)

    event_z0 = torch.randn_like(target_basis_flat) * model.event_config["event_prior_std"]
    target_event_basis_flat = target_event_basis.reshape(target_basis.shape[0], -1)
    event_z_t = (1.0 - t.unsqueeze(-1)) * event_z0 + t.unsqueeze(-1) * target_event_basis_flat
    event_target_v = target_event_basis_flat - event_z0
    (
        _raw_local2,
        _raw_band2,
        quiet_metric_local2,
        quiet_metric_band2,
        event_local,
        event_band,
        _local_gate2,
        _band_gate2,
        post_logits,
        post_probs,
        _post_sample,
        event_amp,
    ) = model.build_sparse_support_controls(
        z_t,
        t,
        path_context,
        support_pack=support_pack,
        use_posterior=True,
    )
    event_raw = model.event_transport(event_z_t, t, path_context)
    event_raw_basis = event_raw.view(target_basis.shape[0], model.decoder.n_frames, model.decoder.n_cells)
    event_pred_v_basis = model.modulate_velocity_basis(event_raw_basis, event_local, event_band)
    scaled_event_pred_v = model.event_scale() * event_pred_v_basis.reshape(target_basis.shape[0], -1)
    event_residual_fm_loss = F.mse_loss(scaled_event_pred_v, event_target_v)

    quiet_weight = (1.0 - 0.85 * teacher_support).clamp_min(model.support_config["quiet_floor"])
    event_weight = teacher_support.clamp_min(model.support_config["event_floor"])
    quiet_local_loss = weighted_smooth_l1(quiet_metric_local2, target_local_log, quiet_weight)
    quiet_band_loss = F.smooth_l1_loss(quiet_metric_band2, target_band_log)
    event_local_loss = weighted_smooth_l1(event_local, target_event_local, event_weight)
    event_band_loss = F.smooth_l1_loss(event_band, target_event_band)

    prior_logits = support_pack["prior_logits"]
    prior_probs = support_pack["prior_probs"]
    target_rate = teacher_support.mean()
    pos_weight = ((1.0 - target_rate) / target_rate.clamp_min(1e-3)).clamp(1.0, 15.0)
    pos_weight_t = torch.tensor(float(pos_weight), device=teacher_support.device, dtype=teacher_support.dtype)
    post_support_loss = F.binary_cross_entropy_with_logits(post_logits, teacher_support, pos_weight=pos_weight_t)
    prior_support_loss = F.binary_cross_entropy_with_logits(prior_logits, teacher_support, pos_weight=pos_weight_t)
    support_kl = bernoulli_kl(post_probs, prior_probs).mean()
    support_rate_reg = (prior_probs.mean() - target_rate).pow(2) + (post_probs.mean() - target_rate).pow(2)
    support_surf = prior_probs.view(prior_probs.shape[0], prior_probs.shape[1], 5, 5)
    support_time_tv = (prior_probs[:, 1:] - prior_probs[:, :-1]).abs().mean()
    support_row_tv = (support_surf[:, :, 1:] - support_surf[:, :, :-1]).abs().mean()
    support_col_tv = (support_surf[:, :, :, 1:] - support_surf[:, :, :, :-1]).abs().mean()
    support_tv_reg = support_time_tv + support_row_tv + support_col_tv
    event_quiet_penalty = ((1.0 - teacher_support) * event_local.abs()).mean()

    total = (
        fm_loss
        + model.event_config["event_residual_fm_loss_weight"] * event_residual_fm_loss
        + model.support_config["quiet_local_loss_weight"] * quiet_local_loss
        + model.support_config["quiet_band_loss_weight"] * quiet_band_loss
        + model.support_config["event_local_loss_weight"] * event_local_loss
        + model.support_config["event_band_loss_weight"] * event_band_loss
        + model.support_config["post_support_loss_weight"] * post_support_loss
        + model.support_config["prior_support_loss_weight"] * prior_support_loss
        + model.support_config["support_kl_weight"] * support_kl
        + model.support_config["support_rate_reg_weight"] * support_rate_reg
        + model.support_config["support_tv_reg_weight"] * support_tv_reg
        + model.support_config["event_quiet_penalty_weight"] * event_quiet_penalty
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
        "pred_velocity_norm_mean": control_metrics["pred_velocity_norm_mean"],
        "target_jump_like_rate": target_jump_like.mean(),
        "target_high_band_abs_mean": (
            (target_basis_flat.abs() * high_mask.unsqueeze(0)).sum(dim=-1) / high_mask.sum().clamp_min(1.0)
        ).mean(),
        "ssr_total_loss": total,
        "event_residual_fm_loss": event_residual_fm_loss,
        "quiet_local_loss": quiet_local_loss,
        "quiet_band_loss": quiet_band_loss,
        "event_local_loss": event_local_loss,
        "event_band_loss": event_band_loss,
        "post_support_loss": post_support_loss,
        "prior_support_loss": prior_support_loss,
        "support_kl": support_kl,
        "support_rate_reg": support_rate_reg,
        "support_tv_reg": support_tv_reg,
        "event_quiet_penalty": event_quiet_penalty,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "target_support_rate": teacher_support.mean(),
        "target_event_local_abs_mean": target_event_local.abs().mean(),
        "target_event_band_high_mean": target_event_band[:, 2].mean(),
        "target_event_basis_abs_mean": target_event_basis_flat.abs().mean(),
        "pred_event_basis_abs_mean": scaled_event_pred_v.abs().mean(),
        "post_support_rate": post_probs.mean(),
        "post_support_top1_mean": post_probs.reshape(post_probs.shape[0], -1).max(dim=1).values.mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: SparseSupportResidualModel,
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
        "ssr_total_loss",
        "event_residual_fm_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_support_loss",
        "prior_support_loss",
        "support_kl",
        "support_rate_reg",
        "support_tv_reg",
        "event_quiet_penalty",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_support_rate",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "target_event_basis_abs_mean",
        "pred_event_basis_abs_mean",
        "post_support_rate",
        "post_support_top1_mean",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "event_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "quiet_band_high_mean",
        "quiet_band_mid_mean",
        "quiet_band_low_mean",
        "event_band_high_mean",
        "event_band_mid_mean",
        "event_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "event_scale",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "prior_support_rate",
        "prior_support_top1_mean",
        "support_logit_abs_mean",
        "event_amp_abs_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = sparse_support_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_ssr_total_loss"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="187a_v0: latent sparse-support residual model")
    parser.add_argument("--epochs_stage1", type=int, default=3)
    parser.add_argument("--epochs_stage2", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_event_stage1", type=float, default=7e-4)
    parser.add_argument("--lr_event_stage2", type=float, default=2.5e-4)
    parser.add_argument("--lr_adapter_stage2", type=float, default=8e-5)
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
    parser.add_argument("--state_gate_hidden_dim", type=int, default=128)
    parser.add_argument("--init_local_state_gate", type=float, default=0.30)
    parser.add_argument("--init_band_state_gate", type=float, default=0.25)
    parser.add_argument("--local_metric_min", type=float, default=0.12)
    parser.add_argument("--local_metric_max", type=float, default=0.55)
    parser.add_argument("--band_metric_min", type=float, default=0.08)
    parser.add_argument("--band_metric_max", type=float, default=0.35)
    parser.add_argument("--init_local_metric_budget", type=float, default=0.24)
    parser.add_argument("--init_band_metric_budget", type=float, default=0.14)
    parser.add_argument("--local_metric_clip", type=float, default=0.60)
    parser.add_argument("--band_metric_clip", type=float, default=0.35)
    parser.add_argument("--budget_reg_weight", type=float, default=0.02)
    parser.add_argument("--support_hidden_dim", type=int, default=128)
    parser.add_argument("--context_feat_dim", type=int, default=2)
    parser.add_argument("--n_support_blocks", type=int, default=5)
    parser.add_argument("--init_support_rate", type=float, default=0.08)
    parser.add_argument("--teacher_topk", type=int, default=5)
    parser.add_argument("--teacher_threshold", type=float, default=0.08)
    parser.add_argument("--relax_temperature", type=float, default=0.33)
    parser.add_argument("--quiet_floor", type=float, default=0.15)
    parser.add_argument("--event_floor", type=float, default=0.05)
    parser.add_argument("--quiet_local_loss_weight", type=float, default=0.25)
    parser.add_argument("--quiet_band_loss_weight", type=float, default=0.15)
    parser.add_argument("--event_local_loss_weight", type=float, default=0.70)
    parser.add_argument("--event_band_loss_weight", type=float, default=0.25)
    parser.add_argument("--post_support_loss_weight", type=float, default=0.50)
    parser.add_argument("--prior_support_loss_weight", type=float, default=0.20)
    parser.add_argument("--support_kl_weight", type=float, default=0.10)
    parser.add_argument("--support_rate_reg_weight", type=float, default=0.15)
    parser.add_argument("--support_tv_reg_weight", type=float, default=0.02)
    parser.add_argument("--event_quiet_penalty_weight", type=float, default=0.05)
    parser.add_argument("--event_hidden_dim", type=int, default=96)
    parser.add_argument("--event_amp_max", type=float, default=0.85)
    parser.add_argument("--event_band_max", type=float, default=0.22)
    parser.add_argument("--event_local_clip", type=float, default=0.70)
    parser.add_argument("--event_band_clip", type=float, default=0.25)
    parser.add_argument("--event_scale_max", type=float, default=1.10)
    parser.add_argument("--init_event_scale", type=float, default=0.55)
    parser.add_argument("--event_residual_fm_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_prior_std", type=float, default=0.20)
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
    integrated_config = dict(
        width_rank=args.width_rank,
        width_hidden_dim=args.width_hidden_dim,
        width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim,
        band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight,
        band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
        init_local_strength=0.05,
        init_band_strength=0.05,
    )
    state_config = dict(
        gate_hidden_dim=args.state_gate_hidden_dim,
        init_local_state_gate=args.init_local_state_gate,
        init_band_state_gate=args.init_band_state_gate,
    )
    metric_config = dict(
        local_metric_min=args.local_metric_min,
        local_metric_max=args.local_metric_max,
        band_metric_min=args.band_metric_min,
        band_metric_max=args.band_metric_max,
        init_local_metric_budget=args.init_local_metric_budget,
        init_band_metric_budget=args.init_band_metric_budget,
        local_metric_clip=args.local_metric_clip,
        band_metric_clip=args.band_metric_clip,
        budget_reg_weight=args.budget_reg_weight,
    )
    support_config = dict(
        support_hidden_dim=args.support_hidden_dim,
        context_feat_dim=args.context_feat_dim,
        n_support_blocks=args.n_support_blocks,
        init_support_rate=args.init_support_rate,
        teacher_topk=args.teacher_topk,
        teacher_threshold=args.teacher_threshold,
        relax_temperature=args.relax_temperature,
        quiet_floor=args.quiet_floor,
        event_floor=args.event_floor,
        quiet_local_loss_weight=args.quiet_local_loss_weight,
        quiet_band_loss_weight=args.quiet_band_loss_weight,
        event_local_loss_weight=args.event_local_loss_weight,
        event_band_loss_weight=args.event_band_loss_weight,
        post_support_loss_weight=args.post_support_loss_weight,
        prior_support_loss_weight=args.prior_support_loss_weight,
        support_kl_weight=args.support_kl_weight,
        support_rate_reg_weight=args.support_rate_reg_weight,
        support_tv_reg_weight=args.support_tv_reg_weight,
        event_quiet_penalty_weight=args.event_quiet_penalty_weight,
    )
    event_config = dict(
        event_hidden_dim=args.event_hidden_dim,
        event_amp_max=args.event_amp_max,
        event_band_max=args.event_band_max,
        event_local_clip=args.event_local_clip,
        event_band_clip=args.event_band_clip,
        event_scale_max=args.event_scale_max,
        init_event_scale=args.init_event_scale,
        event_residual_fm_loss_weight=args.event_residual_fm_loss_weight,
        event_prior_std=args.event_prior_std,
    )

    model = SparseSupportResidualModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        support_config=support_config,
        event_config=event_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
        mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)
    if args.warm_start_path:
        model.maybe_load_warm_start(args.warm_start_path, args.device)

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
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            params = (
                list(model.support_context.parameters())
                + list(model.support_prior_head.parameters())
                + list(model.support_post_head.parameters())
                + list(model.event_amplitude_head.parameters())
                + list(model.event_band_head.parameters())
                + list(model.event_transport.parameters())
                + [model.event_scale_logit]
            )
            optimizer = torch.optim.AdamW(params, lr=args.lr_event_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "sparse-support-head"
        else:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            params = [
                {
                    "params": (
                        list(model.support_context.parameters())
                        + list(model.support_prior_head.parameters())
                        + list(model.support_post_head.parameters())
                        + list(model.event_amplitude_head.parameters())
                        + list(model.event_band_head.parameters())
                        + list(model.event_transport.parameters())
                        + [model.event_scale_logit]
                    ),
                    "lr": args.lr_event_stage2,
                },
                {
                    "params": list(model.path_context_adapter.parameters()),
                    "lr": args.lr_adapter_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "sparse-support-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n187a_v0: latent sparse-support residual model\n{'=' * 72}")
    print(f"  Stage1 epochs: {args.epochs_stage1} | Stage2 epochs: {args.epochs_stage2}")
    print(f"  Warm start: {args.warm_start_path}")

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
        "ssr_total_loss",
        "event_residual_fm_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_support_loss",
        "prior_support_loss",
        "support_kl",
        "support_rate_reg",
        "support_tv_reg",
        "event_quiet_penalty",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_support_rate",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "target_event_basis_abs_mean",
        "pred_event_basis_abs_mean",
        "post_support_rate",
        "post_support_top1_mean",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "event_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "quiet_band_high_mean",
        "quiet_band_mid_mean",
        "quiet_band_low_mean",
        "event_band_high_mean",
        "event_band_mid_mean",
        "event_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "event_scale",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "prior_support_rate",
        "prior_support_top1_mean",
        "support_logit_abs_mean",
        "event_amp_abs_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
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
            loss, metrics = sparse_support_flow_matching_loss(model, history_01, future_01)
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
        current_key = strict_checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
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
                        "type": "latent_sparse_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "support_model": support_config,
                        "event": event_config,
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
            f"val_total={val_metrics['val_total_loss']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"worstLate={frontier_metrics['frontier_turb_late_worst_cov']:.3f}  "
            f"bestLate={frontier_metrics['frontier_turb_late_best_cov']:.3f}  "
            f"kurt={frontier_metrics['frontier_pooled_kurt_ratio']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"jumpKS={joint_metrics['joint_pathwise_jump_ks']:.3f}  "
            f"pSup={val_metrics['val_prior_support_rate']:.3f}  "
            f"qSup={val_metrics['val_post_support_rate']:.3f}  "
            f"eAmp={val_metrics['val_event_amp_abs_mean']:.3f}  "
            f"eBasis={val_metrics['val_pred_event_basis_abs_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "latent_sparse_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "support_model": support_config,
            "event": event_config,
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


if __name__ == "__main__":
    main()
