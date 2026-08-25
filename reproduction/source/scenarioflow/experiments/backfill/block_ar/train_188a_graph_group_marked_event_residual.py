#!/usr/bin/env python
"""
188a_v0: Graph/group-aware latent marked-event residual model.

Principle:
  - keep the explicit mean-reverting mean branch and structured covariance branch
  - keep the validated 183c quiet residual backbone for ordinary residual behavior
  - replace support masks with a small set of latent event objects
  - each event object carries time/block anchor, node anchor, amplitude,
    spatial spread, temporal duration, and band profile
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import ConditionalPathFlowTransformer
from experiments.backfill.block_ar.train_182b_width_tail_control import evaluate_frontier_subset
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel, _logit


def bernoulli_kl(post_p: torch.Tensor, prior_p: torch.Tensor) -> torch.Tensor:
    post_p = post_p.clamp(1e-5, 1.0 - 1e-5)
    prior_p = prior_p.clamp(1e-5, 1.0 - 1e-5)
    return post_p * (post_p.log() - prior_p.log()) + (1.0 - post_p) * (
        (1.0 - post_p).log() - (1.0 - prior_p).log()
    )


class SlotInferenceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_slots: int, n_blocks: int, n_nodes: int, init_gate: float):
        super().__init__()
        self.n_slots = n_slots
        self.n_blocks = n_blocks
        self.n_nodes = n_nodes
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.active = nn.Linear(hidden_dim, 1)
        self.time_logits = nn.Linear(hidden_dim, n_blocks)
        self.node_logits = nn.Linear(hidden_dim, n_nodes)
        self.amp = nn.Linear(hidden_dim, 1)
        self.radius = nn.Linear(hidden_dim, 1)
        self.duration = nn.Linear(hidden_dim, 1)
        self.band = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.active.weight)
        nn.init.constant_(self.active.bias, _logit(init_gate))
        for layer in (self.time_logits, self.node_logits, self.amp, self.radius, self.duration, self.band):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, K, D]
        h = self.hidden(x)
        return {
            "active_logit": self.active(h).squeeze(-1),
            "time_logits": self.time_logits(h),
            "node_logits": self.node_logits(h),
            "amp_raw": self.amp(h).squeeze(-1),
            "radius_raw": self.radius(h).squeeze(-1),
            "duration_raw": self.duration(h).squeeze(-1),
            "band_raw": self.band(h),
        }


class MarkedEventResidualModel(StateMetricTransportModel):
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
        slot_config: dict,
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
        self.slot_config = dict(slot_config)
        self.event_config = dict(event_config)
        self.n_slots = int(slot_config["n_slots"])
        self.n_event_blocks = int(slot_config["n_event_blocks"])
        self.n_nodes = self.decoder.n_cells

        block_ids = torch.arange(self.decoder.n_frames, dtype=torch.long) * self.n_event_blocks // self.decoder.n_frames
        self.register_buffer("event_block_ids", block_ids, persistent=False)
        centers = []
        for b in range(self.n_event_blocks):
            idx = torch.nonzero(block_ids == b, as_tuple=False).squeeze(-1).float()
            centers.append(idx.mean() if idx.numel() > 0 else torch.tensor(float(b), dtype=torch.float32))
        self.register_buffer("block_centers", torch.stack(centers), persistent=False)

        coords = []
        for i in range(5):
            for j in range(5):
                coords.append([float(i), float(j)])
        cell_coords = torch.tensor(coords, dtype=torch.float32)
        self.register_buffer("cell_coords", cell_coords, persistent=False)
        dist2 = ((cell_coords[:, None, :] - cell_coords[None, :, :]) ** 2).sum(dim=-1)
        self.register_buffer("cell_dist2", dist2, persistent=False)
        time_index = torch.arange(self.decoder.n_frames, dtype=torch.float32)
        self.register_buffer("time_index", time_index, persistent=False)

        self.slot_embed = nn.Parameter(torch.randn(self.n_slots, slot_config["slot_embed_dim"]) * 0.02)
        prior_in = path_config["context_dim"] + slot_config["slot_embed_dim"]
        post_in = prior_in + slot_config["teacher_feat_dim"]
        self.prior_head = SlotInferenceHead(
            input_dim=prior_in,
            hidden_dim=slot_config["slot_hidden_dim"],
            n_slots=self.n_slots,
            n_blocks=self.n_event_blocks,
            n_nodes=self.n_nodes,
            init_gate=slot_config["init_prior_gate"],
        )
        self.post_head = SlotInferenceHead(
            input_dim=post_in,
            hidden_dim=slot_config["slot_hidden_dim"],
            n_slots=self.n_slots,
            n_blocks=self.n_event_blocks,
            n_nodes=self.n_nodes,
            init_gate=slot_config["init_post_gate"],
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
        self.event_scale_logit = nn.Parameter(torch.tensor(_logit(event_config["init_event_scale"])))

        self.slot_suppress_time = int(slot_config["teacher_suppress_time"])
        self.slot_suppress_radius2 = float(slot_config["teacher_suppress_radius"] ** 2)

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        skip_prefixes = (
            "slot_embed",
            "prior_head.",
            "post_head.",
            "event_transport.",
            "event_scale_logit",
        )
        for key, value in state.items():
            if key.startswith(skip_prefixes):
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

    def _decode_scalar(self, raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    def _slot_context(self, path_context: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                path_context.unsqueeze(1).expand(-1, self.n_slots, -1),
                self.slot_embed.unsqueeze(0).expand(path_context.shape[0], -1, -1),
            ],
            dim=-1,
        )

    def _teacher_block_index(self, t_idx: int) -> int:
        return int(self.event_block_ids[t_idx].item())

    def extract_teacher_slots(self, target_event_local: torch.Tensor) -> dict[str, torch.Tensor]:
        # target_event_local: [B, T, N]
        B, T, N = target_event_local.shape
        device = target_event_local.device
        active = torch.zeros(B, self.n_slots, device=device)
        time_idx = torch.zeros(B, self.n_slots, dtype=torch.long, device=device)
        node_idx = torch.zeros(B, self.n_slots, dtype=torch.long, device=device)
        amp = torch.zeros(B, self.n_slots, device=device)
        radius = torch.full((B, self.n_slots), float(self.slot_config["radius_min"]), device=device)
        duration = torch.full((B, self.n_slots), float(self.slot_config["duration_min"]), device=device)
        feat = torch.zeros(B, self.n_slots, self.slot_config["teacher_feat_dim"], device=device)

        abs_gap = target_event_local.abs()
        thresh = float(self.slot_config["teacher_peak_threshold"])
        for b in range(B):
            avail = torch.ones(T, N, dtype=torch.bool, device=device)
            sample_gap = target_event_local[b]
            sample_abs = abs_gap[b]
            for k in range(self.n_slots):
                masked = sample_abs.masked_fill(~avail, -1.0)
                flat_idx = int(masked.reshape(-1).argmax().item())
                maxv = float(masked.reshape(-1)[flat_idx].item())
                if maxv < thresh:
                    break
                t_idx = flat_idx // N
                n_idx = flat_idx % N
                a = sample_gap[t_idx, n_idx]
                active[b, k] = 1.0
                time_idx[b, k] = self._teacher_block_index(t_idx)
                node_idx[b, k] = n_idx
                amp[b, k] = a

                time_w = sample_abs[:, n_idx]
                dt = (self.time_index.to(device) - float(t_idx)).abs()
                dur_val = (time_w * dt).sum() / time_w.sum().clamp_min(1e-6) + 1.0
                dur_val = dur_val.clamp(
                    min=self.slot_config["duration_min"],
                    max=self.slot_config["duration_max"],
                )
                duration[b, k] = dur_val

                node_w = sample_abs[t_idx]
                d2 = self.cell_dist2[n_idx].to(device=device, dtype=node_w.dtype)
                rad_val = torch.sqrt((node_w * d2).sum() / node_w.sum().clamp_min(1e-6) + 1e-6)
                rad_val = rad_val.clamp(
                    min=self.slot_config["radius_min"],
                    max=self.slot_config["radius_max"],
                )
                radius[b, k] = rad_val

                feat[b, k] = torch.tensor(
                    [
                        1.0,
                        float(t_idx) / max(T - 1, 1),
                        float(self.cell_coords[n_idx, 0].item()) / 4.0,
                        float(self.cell_coords[n_idx, 1].item()) / 4.0,
                        float(a.item()),
                        float(rad_val.item()) / float(self.slot_config["radius_max"]),
                        float(dur_val.item()) / float(self.slot_config["duration_max"]),
                        float(np.sign(float(a.item()))),
                    ],
                    device=device,
                )

                t_lo = max(0, t_idx - self.slot_suppress_time)
                t_hi = min(T, t_idx + self.slot_suppress_time + 1)
                suppress_nodes = self.cell_dist2[n_idx].to(device) <= self.slot_suppress_radius2
                avail[t_lo:t_hi, suppress_nodes] = False

        return {
            "active": active,
            "time_idx": time_idx,
            "node_idx": node_idx,
            "amp": amp,
            "radius": radius,
            "duration": duration,
            "feat": feat,
        }

    def infer_slots(self, path_context: torch.Tensor, teacher_slots: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        prior_in = self._slot_context(path_context)
        prior_raw = self.prior_head(prior_in)
        out = {"prior": prior_raw}
        if teacher_slots is not None:
            post_in = torch.cat([prior_in, teacher_slots["feat"]], dim=-1)
            out["post"] = self.post_head(post_in)
        else:
            out["post"] = prior_raw
        return out

    def _decode_slots(self, slot_raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        active_prob = torch.sigmoid(slot_raw["active_logit"])
        time_probs = torch.softmax(slot_raw["time_logits"], dim=-1)
        node_probs = torch.softmax(slot_raw["node_logits"], dim=-1)
        amp = self.event_config["amp_max"] * torch.tanh(slot_raw["amp_raw"])
        radius = self._decode_scalar(
            slot_raw["radius_raw"],
            self.slot_config["radius_min"],
            self.slot_config["radius_max"],
        )
        duration = self._decode_scalar(
            slot_raw["duration_raw"],
            self.slot_config["duration_min"],
            self.slot_config["duration_max"],
        )
        band = self.event_config["band_max"] * torch.tanh(slot_raw["band_raw"])
        return {
            "active_prob": active_prob,
            "time_probs": time_probs,
            "node_probs": node_probs,
            "amp": amp,
            "radius": radius,
            "duration": duration,
            "band": band,
        }

    def decode_event_field(self, slot_dec: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        B = slot_dec["active_prob"].shape[0]
        device = slot_dec["active_prob"].device
        dtype = slot_dec["active_prob"].dtype

        # Temporal kernel from block anchors.
        frame = self.time_index.to(device=device, dtype=dtype).view(1, 1, self.decoder.n_frames)
        centers = self.block_centers.to(device=device, dtype=dtype).view(1, 1, self.n_event_blocks)
        dur = slot_dec["duration"].unsqueeze(-1).unsqueeze(-1).clamp_min(1e-3)
        time_basis = torch.exp(-0.5 * ((frame.unsqueeze(-2) - centers.unsqueeze(-1)) / dur) ** 2)
        time_kernel = (slot_dec["time_probs"].unsqueeze(-1) * time_basis).sum(dim=2)
        time_kernel = time_kernel / time_kernel.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        # Spatial kernel from node anchors.
        d2 = self.cell_dist2.to(device=device, dtype=dtype).view(1, 1, self.n_nodes, self.n_nodes)
        rad = slot_dec["radius"].unsqueeze(-1).unsqueeze(-1).clamp_min(1e-3)
        node_basis = torch.exp(-0.5 * d2 / (rad**2))
        spatial_kernel = (slot_dec["node_probs"].unsqueeze(-1) * node_basis).sum(dim=2)
        spatial_kernel = spatial_kernel / spatial_kernel.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        active_amp = slot_dec["active_prob"] * slot_dec["amp"]
        event_local = (
            active_amp.unsqueeze(-1).unsqueeze(-1)
            * time_kernel.unsqueeze(-1)
            * spatial_kernel.unsqueeze(-2)
        ).sum(dim=1)
        event_local = event_local.clamp(
            min=-self.event_config["event_local_clip"],
            max=self.event_config["event_local_clip"],
        )

        weight = slot_dec["active_prob"].abs().unsqueeze(-1).clamp_min(1e-6)
        event_band = (slot_dec["band"] * weight).sum(dim=1) / weight.sum(dim=1)
        band_weights = (self.band_masks.reshape(3, -1).sum(dim=1) / self.band_masks.numel()).to(
            event_band.device, dtype=event_band.dtype
        )
        event_band = event_band - (event_band * band_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        event_band = event_band.clamp(
            min=-self.event_config["event_band_clip"],
            max=self.event_config["event_band_clip"],
        )
        return event_local, event_band

    def build_marked_event_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        slot_pack: dict[str, dict[str, torch.Tensor]] | None = None,
        use_posterior: bool = False,
    ):
        raw_local, raw_band, quiet_metric_local, quiet_metric_band, local_gate, band_gate = super().build_state_metric_controls(
            z_t_flat, t, path_context
        )
        if slot_pack is None:
            slot_pack = self.infer_slots(path_context, teacher_slots=None)
        key = "post" if use_posterior else "prior"
        slot_dec = self._decode_slots(slot_pack[key])
        event_local, event_band = self.decode_event_field(slot_dec)
        return (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            slot_dec,
        )

    def transport_velocity(self, z_t_flat: torch.Tensor, t: torch.Tensor, path_context: torch.Tensor):
        return self.transport_velocity_with_pack(z_t_flat, t, path_context, slot_pack=None, use_posterior=False)

    def transport_velocity_with_pack(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        slot_pack: dict[str, dict[str, torch.Tensor]] | None = None,
        use_posterior: bool = False,
    ):
        (
            raw_local,
            raw_band,
            quiet_metric_local,
            quiet_metric_band,
            event_local,
            event_band,
            local_gate,
            band_gate,
            slot_dec,
        ) = self.build_marked_event_controls(z_t_flat, t, path_context, slot_pack=slot_pack, use_posterior=use_posterior)
        quiet_raw = self.path_transport(z_t_flat, t, path_context)
        quiet_raw_basis = quiet_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        quiet_v_basis = self.modulate_velocity_basis(quiet_raw_basis, quiet_metric_local, quiet_metric_band)
        event_raw = self.event_transport(z_t_flat, t, path_context)
        event_raw_basis = event_raw.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        event_v_basis = self.modulate_velocity_basis(event_raw_basis, event_local, event_band)
        total_v_basis = quiet_v_basis + self.event_scale() * event_v_basis
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
            "slot_active_mean": slot_dec["active_prob"].mean(),
            "slot_time_top1_mean": slot_dec["time_probs"].max(dim=-1).values.mean(),
            "slot_node_top1_mean": slot_dec["node_probs"].max(dim=-1).values.mean(),
            "slot_amp_abs_mean": slot_dec["amp"].abs().mean(),
            "slot_radius_mean": slot_dec["radius"].mean(),
            "slot_duration_mean": slot_dec["duration"].mean(),
            "quiet_velocity_norm_mean": quiet_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "event_velocity_norm_mean": event_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
            "pred_velocity_norm_mean": total_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return total_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def marked_event_flow_matching_loss(
    model: MarkedEventResidualModel,
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

    _, _, quiet_metric_local, quiet_metric_band, _, _ = super(
        MarkedEventResidualModel, model
    ).build_state_metric_controls(z_t, t, path_context)
    target_event_local = (target_local_log - quiet_metric_local).clamp(
        min=-model.event_config["event_local_clip"],
        max=model.event_config["event_local_clip"],
    )
    target_event_band = (target_band_log - quiet_metric_band).clamp(
        min=-model.event_config["event_band_clip"],
        max=model.event_config["event_band_clip"],
    )
    teacher_slots = model.extract_teacher_slots(target_event_local.detach())
    slot_pack = model.infer_slots(path_context, teacher_slots=teacher_slots)
    pred_v, control_metrics = model.transport_velocity_with_pack(
        z_t, t, path_context, slot_pack=slot_pack, use_posterior=True
    )
    fm_loss = F.mse_loss(pred_v, target_v)

    (
        _raw_local,
        _raw_band,
        quiet_metric_local2,
        quiet_metric_band2,
        event_local,
        event_band,
        _local_gate,
        _band_gate,
        post_dec,
    ) = model.build_marked_event_controls(z_t, t, path_context, slot_pack=slot_pack, use_posterior=True)
    prior_dec = model._decode_slots(slot_pack["prior"])
    teacher_dec = {
        "active_prob": teacher_slots["active"],
        "time_probs": F.one_hot(teacher_slots["time_idx"], num_classes=model.n_event_blocks).to(target_basis.dtype),
        "node_probs": F.one_hot(teacher_slots["node_idx"], num_classes=model.n_nodes).to(target_basis.dtype),
        "amp": teacher_slots["amp"],
        "radius": teacher_slots["radius"],
        "duration": teacher_slots["duration"],
        "band": torch.zeros_like(post_dec["band"]),
    }
    teacher_event_local, _teacher_event_band_from_slots = model.decode_event_field(teacher_dec)

    quiet_local_loss = F.smooth_l1_loss(quiet_metric_local2, target_local_log)
    quiet_band_loss = F.smooth_l1_loss(quiet_metric_band2, target_band_log)
    event_local_loss = F.smooth_l1_loss(event_local, teacher_event_local)
    event_band_loss = F.smooth_l1_loss(event_band, target_event_band)

    active_mask = teacher_slots["active"]
    active_norm = active_mask.sum().clamp_min(1.0)
    post_active_bce = F.binary_cross_entropy(post_dec["active_prob"], active_mask)
    prior_active_bce = F.binary_cross_entropy(prior_dec["active_prob"], active_mask)

    time_targets = teacher_slots["time_idx"].reshape(-1)
    node_targets = teacher_slots["node_idx"].reshape(-1)
    flat_mask = active_mask.reshape(-1)
    post_time_ce = (
        F.cross_entropy(slot_pack["post"]["time_logits"].reshape(-1, model.n_event_blocks), time_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    prior_time_ce = (
        F.cross_entropy(slot_pack["prior"]["time_logits"].reshape(-1, model.n_event_blocks), time_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    post_node_ce = (
        F.cross_entropy(slot_pack["post"]["node_logits"].reshape(-1, model.n_nodes), node_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm
    prior_node_ce = (
        F.cross_entropy(slot_pack["prior"]["node_logits"].reshape(-1, model.n_nodes), node_targets, reduction="none")
        * flat_mask
    ).sum() / active_norm

    amp_loss = (F.smooth_l1_loss(post_dec["amp"], teacher_slots["amp"], reduction="none") * active_mask).sum() / active_norm
    radius_loss = (
        F.smooth_l1_loss(post_dec["radius"], teacher_slots["radius"], reduction="none") * active_mask
    ).sum() / active_norm
    duration_loss = (
        F.smooth_l1_loss(post_dec["duration"], teacher_slots["duration"], reduction="none") * active_mask
    ).sum() / active_norm

    active_kl = bernoulli_kl(post_dec["active_prob"], prior_dec["active_prob"]).mean()
    time_kl = (
        post_dec["time_probs"] * (post_dec["time_probs"].clamp_min(1e-6).log() - prior_dec["time_probs"].clamp_min(1e-6).log())
    ).sum(dim=-1)
    time_kl = (time_kl * post_dec["active_prob"]).mean() / post_dec["active_prob"].mean().clamp_min(0.05)
    node_kl = (
        post_dec["node_probs"] * (post_dec["node_probs"].clamp_min(1e-6).log() - prior_dec["node_probs"].clamp_min(1e-6).log())
    ).sum(dim=-1)
    node_kl = (node_kl * post_dec["active_prob"]).mean() / post_dec["active_prob"].mean().clamp_min(0.05)
    slot_overlap = torch.zeros((), device=target_basis.device, dtype=target_basis.dtype)
    if model.n_slots > 1:
        kernels = []
        for k in range(model.n_slots):
            single = {
                "active_prob": post_dec["active_prob"][:, k : k + 1],
                "time_probs": post_dec["time_probs"][:, k : k + 1],
                "node_probs": post_dec["node_probs"][:, k : k + 1],
                "amp": post_dec["amp"][:, k : k + 1].abs(),
                "radius": post_dec["radius"][:, k : k + 1],
                "duration": post_dec["duration"][:, k : k + 1],
                "band": post_dec["band"][:, k : k + 1],
            }
            kern, _ = model.decode_event_field(single)
            kernels.append(kern.reshape(kern.shape[0], -1))
        overlap_terms = []
        for i in range(model.n_slots):
            for j in range(i + 1, model.n_slots):
                ki = F.normalize(kernels[i], dim=-1)
                kj = F.normalize(kernels[j], dim=-1)
                overlap_terms.append((ki * kj).sum(dim=-1).mean())
        if overlap_terms:
            slot_overlap = torch.stack(overlap_terms).mean()

    quiet_dom = post_dec["active_prob"].mean()

    total = (
        fm_loss
        + model.event_config["quiet_local_loss_weight"] * quiet_local_loss
        + model.event_config["quiet_band_loss_weight"] * quiet_band_loss
        + model.event_config["event_local_loss_weight"] * event_local_loss
        + model.event_config["event_band_loss_weight"] * event_band_loss
        + model.slot_config["post_active_bce_weight"] * post_active_bce
        + model.slot_config["prior_active_bce_weight"] * prior_active_bce
        + model.slot_config["post_time_ce_weight"] * post_time_ce
        + model.slot_config["prior_time_ce_weight"] * prior_time_ce
        + model.slot_config["post_node_ce_weight"] * post_node_ce
        + model.slot_config["prior_node_ce_weight"] * prior_node_ce
        + model.slot_config["amp_loss_weight"] * amp_loss
        + model.slot_config["radius_loss_weight"] * radius_loss
        + model.slot_config["duration_loss_weight"] * duration_loss
        + model.slot_config["active_kl_weight"] * active_kl
        + model.slot_config["time_kl_weight"] * time_kl
        + model.slot_config["node_kl_weight"] * node_kl
        + model.slot_config["slot_overlap_weight"] * slot_overlap
        + model.slot_config["quiet_dominance_weight"] * quiet_dom
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
        "marked_total_loss": total,
        "quiet_local_loss": quiet_local_loss,
        "quiet_band_loss": quiet_band_loss,
        "event_local_loss": event_local_loss,
        "event_band_loss": event_band_loss,
        "post_active_bce": post_active_bce,
        "prior_active_bce": prior_active_bce,
        "post_time_ce": post_time_ce,
        "prior_time_ce": prior_time_ce,
        "post_node_ce": post_node_ce,
        "prior_node_ce": prior_node_ce,
        "amp_loss": amp_loss,
        "radius_loss": radius_loss,
        "duration_loss": duration_loss,
        "active_kl": active_kl,
        "time_kl": time_kl,
        "node_kl": node_kl,
        "slot_overlap": slot_overlap,
        "quiet_dominance": quiet_dom,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "target_event_local_abs_mean": target_event_local.abs().mean(),
        "target_event_band_high_mean": target_event_band[:, 2].mean(),
        "teacher_slot_active_rate": teacher_slots["active"].mean(),
        "teacher_slot_amp_abs_mean": teacher_slots["amp"].abs().mean(),
        "teacher_slot_radius_mean": teacher_slots["radius"].mean(),
        "teacher_slot_duration_mean": teacher_slots["duration"].mean(),
        "post_slot_active_mean": post_dec["active_prob"].mean(),
        "prior_slot_active_mean": prior_dec["active_prob"].mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(model: MarkedEventResidualModel, val_loader: DataLoader) -> dict[str, float]:
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
        "marked_total_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_active_bce",
        "prior_active_bce",
        "post_time_ce",
        "prior_time_ce",
        "post_node_ce",
        "prior_node_ce",
        "amp_loss",
        "radius_loss",
        "duration_loss",
        "active_kl",
        "time_kl",
        "node_kl",
        "slot_overlap",
        "quiet_dominance",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "teacher_slot_active_rate",
        "teacher_slot_amp_abs_mean",
        "teacher_slot_radius_mean",
        "teacher_slot_duration_mean",
        "post_slot_active_mean",
        "prior_slot_active_mean",
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
        "slot_active_mean",
        "slot_time_top1_mean",
        "slot_node_top1_mean",
        "slot_amp_abs_mean",
        "slot_radius_mean",
        "slot_duration_mean",
        "quiet_velocity_norm_mean",
        "event_velocity_norm_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = marked_event_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_marked_total_loss"]
    return out


def strict_checkpoint_key_188a(val_metrics: dict, frontier_metrics: dict, joint_metrics: dict) -> tuple[float, ...]:
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
    slot_gate = float(val_metrics.get("val_prior_slot_active_mean", 0.0))
    slot_gap = abs(slot_gate - float(val_metrics.get("val_teacher_slot_active_rate", 0.0)))
    val_loss = float(val_metrics.get("val_total_loss", float("inf")))
    return (tc_gap, worst_gap, best_gap, kurt_gap, slot_gap, jump_gap, mr_gap, val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="188a_v0: graph/group latent marked-event residual model")
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
    parser.add_argument("--n_slots", type=int, default=2)
    parser.add_argument("--n_event_blocks", type=int, default=5)
    parser.add_argument("--slot_embed_dim", type=int, default=16)
    parser.add_argument("--slot_hidden_dim", type=int, default=128)
    parser.add_argument("--teacher_feat_dim", type=int, default=8)
    parser.add_argument("--init_prior_gate", type=float, default=0.10)
    parser.add_argument("--init_post_gate", type=float, default=0.20)
    parser.add_argument("--teacher_peak_threshold", type=float, default=0.10)
    parser.add_argument("--teacher_suppress_time", type=int, default=3)
    parser.add_argument("--teacher_suppress_radius", type=float, default=1.5)
    parser.add_argument("--radius_min", type=float, default=0.50)
    parser.add_argument("--radius_max", type=float, default=2.50)
    parser.add_argument("--duration_min", type=float, default=1.0)
    parser.add_argument("--duration_max", type=float, default=8.0)
    parser.add_argument("--post_active_bce_weight", type=float, default=0.40)
    parser.add_argument("--prior_active_bce_weight", type=float, default=0.12)
    parser.add_argument("--post_time_ce_weight", type=float, default=0.25)
    parser.add_argument("--prior_time_ce_weight", type=float, default=0.08)
    parser.add_argument("--post_node_ce_weight", type=float, default=0.25)
    parser.add_argument("--prior_node_ce_weight", type=float, default=0.08)
    parser.add_argument("--amp_loss_weight", type=float, default=0.40)
    parser.add_argument("--radius_loss_weight", type=float, default=0.12)
    parser.add_argument("--duration_loss_weight", type=float, default=0.12)
    parser.add_argument("--active_kl_weight", type=float, default=0.05)
    parser.add_argument("--time_kl_weight", type=float, default=0.02)
    parser.add_argument("--node_kl_weight", type=float, default=0.02)
    parser.add_argument("--slot_overlap_weight", type=float, default=0.02)
    parser.add_argument("--quiet_dominance_weight", type=float, default=0.02)
    parser.add_argument("--amp_max", type=float, default=0.90)
    parser.add_argument("--band_max", type=float, default=0.25)
    parser.add_argument("--event_local_clip", type=float, default=0.70)
    parser.add_argument("--event_band_clip", type=float, default=0.25)
    parser.add_argument("--event_scale_max", type=float, default=1.20)
    parser.add_argument("--init_event_scale", type=float, default=0.30)
    parser.add_argument("--quiet_local_loss_weight", type=float, default=0.15)
    parser.add_argument("--quiet_band_loss_weight", type=float, default=0.10)
    parser.add_argument("--event_local_loss_weight", type=float, default=0.60)
    parser.add_argument("--event_band_loss_weight", type=float, default=0.25)
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
    slot_config = dict(
        n_slots=args.n_slots,
        n_event_blocks=args.n_event_blocks,
        slot_embed_dim=args.slot_embed_dim,
        slot_hidden_dim=args.slot_hidden_dim,
        teacher_feat_dim=args.teacher_feat_dim,
        init_prior_gate=args.init_prior_gate,
        init_post_gate=args.init_post_gate,
        teacher_peak_threshold=args.teacher_peak_threshold,
        teacher_suppress_time=args.teacher_suppress_time,
        teacher_suppress_radius=args.teacher_suppress_radius,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        duration_min=args.duration_min,
        duration_max=args.duration_max,
        post_active_bce_weight=args.post_active_bce_weight,
        prior_active_bce_weight=args.prior_active_bce_weight,
        post_time_ce_weight=args.post_time_ce_weight,
        prior_time_ce_weight=args.prior_time_ce_weight,
        post_node_ce_weight=args.post_node_ce_weight,
        prior_node_ce_weight=args.prior_node_ce_weight,
        amp_loss_weight=args.amp_loss_weight,
        radius_loss_weight=args.radius_loss_weight,
        duration_loss_weight=args.duration_loss_weight,
        active_kl_weight=args.active_kl_weight,
        time_kl_weight=args.time_kl_weight,
        node_kl_weight=args.node_kl_weight,
        slot_overlap_weight=args.slot_overlap_weight,
        quiet_dominance_weight=args.quiet_dominance_weight,
    )
    event_config = dict(
        amp_max=args.amp_max,
        band_max=args.band_max,
        event_local_clip=args.event_local_clip,
        event_band_clip=args.event_band_clip,
        event_scale_max=args.event_scale_max,
        init_event_scale=args.init_event_scale,
        quiet_local_loss_weight=args.quiet_local_loss_weight,
        quiet_band_loss_weight=args.quiet_band_loss_weight,
        event_local_loss_weight=args.event_local_loss_weight,
        event_band_loss_weight=args.event_band_loss_weight,
    )

    model = MarkedEventResidualModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        slot_config=slot_config,
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
    model.local_metric_budget_logit.requires_grad_(False)
    model.band_metric_budget_logit.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        event_params = (
            list(model.prior_head.parameters())
            + list(model.post_head.parameters())
            + [model.slot_embed]
            + list(model.event_transport.parameters())
            + [model.event_scale_logit]
        )
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            optimizer = torch.optim.AdamW(event_params, lr=args.lr_event_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "marked-event-head"
        else:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(True)
            model.width_allocator.requires_grad_(False)
            model.band_tail.requires_grad_(False)
            model.local_state_gate.requires_grad_(False)
            model.band_state_gate.requires_grad_(False)
            optimizer = torch.optim.AdamW(
                [
                    {"params": event_params, "lr": args.lr_event_stage2},
                    {"params": list(model.path_context_adapter.parameters()), "lr": args.lr_adapter_stage2},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "marked-event-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n188a_v0: graph/group latent marked-event residual model\n{'=' * 72}")
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
        "marked_total_loss",
        "quiet_local_loss",
        "quiet_band_loss",
        "event_local_loss",
        "event_band_loss",
        "post_active_bce",
        "prior_active_bce",
        "post_time_ce",
        "prior_time_ce",
        "post_node_ce",
        "prior_node_ce",
        "amp_loss",
        "radius_loss",
        "duration_loss",
        "active_kl",
        "time_kl",
        "node_kl",
        "slot_overlap",
        "quiet_dominance",
        "target_local_abs_mean",
        "target_band_high_mean",
        "target_event_local_abs_mean",
        "target_event_band_high_mean",
        "teacher_slot_active_rate",
        "teacher_slot_amp_abs_mean",
        "teacher_slot_radius_mean",
        "teacher_slot_duration_mean",
        "post_slot_active_mean",
        "prior_slot_active_mean",
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
        "slot_active_mean",
        "slot_time_top1_mean",
        "slot_node_top1_mean",
        "slot_amp_abs_mean",
        "slot_radius_mean",
        "slot_duration_mean",
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
            loss, metrics = marked_event_flow_matching_loss(model, history_01, future_01)
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
        current_key = strict_checkpoint_key_188a(val_metrics, frontier_metrics, joint_metrics)
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
                        "type": "graph_group_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_188a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "event_slots": slot_config,
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
            f"pSlot={val_metrics['val_prior_slot_active_mean']:.3f}  "
            f"qSlot={val_metrics['val_post_slot_active_mean']:.3f}  "
            f"sTop={val_metrics['val_slot_time_top1_mean']:.3f}/{val_metrics['val_slot_node_top1_mean']:.3f}  "
            f"eAmp={val_metrics['val_slot_amp_abs_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "graph_group_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_188a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "event_slots": slot_config,
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
