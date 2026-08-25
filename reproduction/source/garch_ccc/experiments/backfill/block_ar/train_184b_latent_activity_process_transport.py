#!/usr/bin/env python
"""
184b_v0: Latent activity-process transport on top of the 183c pathwise residual-law backbone.

Principle:
  - keep the 183c mean/covariance/pathwise residual backbone
  - keep state-conditioned local and band metric control as the quiet branch
  - replace 184a's teacher-guided activity gate with a learned latent activity process
  - use a prior from history/context and a posterior from target residual structure during training
  - use the prior only at inference, keeping the mechanism reusable beyond IV surfaces
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
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel, _scaled_logit


class ScaledSigmoidHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        low: float,
        high: float,
        init_value: float,
    ):
        super().__init__()
        self.low = low
        self.high = high
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _scaled_logit(init_value, low, high))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.net(x))
        return self.low + (self.high - self.low) * raw


def topk_soft_allocation_masked(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    temperature: float,
) -> torch.Tensor:
    flat = scores.reshape(scores.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1).bool()
    valid = int(mask_flat[0].sum().item())
    k = max(1, min(k, valid))
    masked_scores = flat.masked_fill(~mask_flat, -1e9)
    topv, topi = masked_scores.topk(k, dim=1)
    logits = masked_scores.new_full(masked_scores.shape, -1e9)
    logits.scatter_(1, topi, topv / max(temperature, 1e-6))
    alloc = torch.softmax(logits, dim=1)
    alloc = alloc * mask_flat.to(flat.dtype)
    active = ((flat * mask_flat.to(flat.dtype)).sum(dim=1, keepdim=True) > 1e-8).to(flat.dtype)
    alloc = alloc / alloc.sum(dim=1, keepdim=True).clamp_min(1e-6)
    alloc = alloc * active
    return alloc.view_as(scores)


class LocationActivityAllocator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(features)
        pos_scores = F.softplus(logits[..., 0])
        neg_scores = F.softplus(logits[..., 1])
        return pos_scores, neg_scores


class BlockLatentActivityProcess(nn.Module):
    def __init__(
        self,
        context_dim: int,
        state_dim: int,
        teacher_dim: int,
        hidden_dim: int,
        n_groups: int,
        budget_min: float,
        budget_max: float,
        init_gate: float,
        init_budget: float,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.budget_min = budget_min
        self.budget_max = budget_max

        self.prior_init = nn.Linear(context_dim, hidden_dim)
        self.prior_cell = nn.GRUCell(state_dim, hidden_dim)
        self.prior_gate = nn.Linear(hidden_dim, 1)
        self.prior_budget = nn.Linear(hidden_dim, 1)
        self.prior_group = nn.Linear(hidden_dim, n_groups)

        self.posterior_init = nn.Linear(context_dim, hidden_dim)
        self.posterior_cell = nn.GRUCell(state_dim + teacher_dim, hidden_dim)
        self.posterior_gate = nn.Linear(hidden_dim, 1)
        self.posterior_budget = nn.Linear(hidden_dim, 1)
        self.posterior_group = nn.Linear(hidden_dim, n_groups)

        for head in (self.prior_gate, self.posterior_gate):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, _scaled_logit(init_gate, 0.0, 1.0))
        for head in (self.prior_budget, self.posterior_budget):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, _scaled_logit(init_budget, budget_min, budget_max))
        for head in (self.prior_group, self.posterior_group):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _decode_budget(self, raw: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(raw)
        return self.budget_min + (self.budget_max - self.budget_min) * prob

    def _run_prior(self, path_context: torch.Tensor, block_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.prior_init(path_context))
        gate_list = []
        budget_list = []
        group_list = []
        for b in range(block_inputs.shape[1]):
            h = self.prior_cell(block_inputs[:, b], h)
            gate_list.append(torch.sigmoid(self.prior_gate(h)))
            budget_list.append(self._decode_budget(self.prior_budget(h)))
            group_list.append(torch.softmax(self.prior_group(h), dim=-1))
        return (
            torch.stack(gate_list, dim=1),
            torch.stack(budget_list, dim=1),
            torch.stack(group_list, dim=1),
        )

    def _run_posterior(
        self,
        path_context: torch.Tensor,
        block_inputs: torch.Tensor,
        teacher_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.posterior_init(path_context))
        gate_list = []
        budget_list = []
        group_list = []
        for b in range(block_inputs.shape[1]):
            inp = torch.cat([block_inputs[:, b], teacher_inputs[:, b]], dim=-1)
            h = self.posterior_cell(inp, h)
            gate_list.append(torch.sigmoid(self.posterior_gate(h)))
            budget_list.append(self._decode_budget(self.posterior_budget(h)))
            group_list.append(torch.softmax(self.posterior_group(h), dim=-1))
        return (
            torch.stack(gate_list, dim=1),
            torch.stack(budget_list, dim=1),
            torch.stack(group_list, dim=1),
        )

    def forward(
        self,
        path_context: torch.Tensor,
        block_inputs: torch.Tensor,
        teacher_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prior_gate, prior_budget, prior_group = self._run_prior(path_context, block_inputs)
        out = {
            "prior_gate": prior_gate,
            "prior_budget": prior_budget,
            "prior_group": prior_group,
        }
        if teacher_inputs is not None:
            post_gate, post_budget, post_group = self._run_posterior(path_context, block_inputs, teacher_inputs)
            out.update(
                {
                    "post_gate": post_gate,
                    "post_budget": post_budget,
                    "post_group": post_group,
                }
            )
        return out


class LatentActivityProcessTransportModel(StateMetricTransportModel):
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
        activity_config: dict,
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
        self.activity_config = dict(activity_config)
        group_ids = self.activity_config.get("group_ids")
        self.activity_geometry = StructuredResidualActivityGeometry(
            n_frames=self.decoder.n_frames,
            n_cells=self.decoder.n_cells,
            group_ids=group_ids,
        )
        self.n_activity_blocks = int(self.activity_config["n_activity_blocks"])
        block_ids = torch.arange(self.decoder.n_frames, dtype=torch.long) * self.n_activity_blocks // self.decoder.n_frames
        self.register_buffer("activity_block_ids", block_ids, persistent=False)
        n_groups = self.activity_geometry.n_groups
        self.n_groups = n_groups
        block_state_dim = 9 + 2 * n_groups
        teacher_dim = 5 + n_groups
        self.activity_process = BlockLatentActivityProcess(
            context_dim=path_config["context_dim"],
            state_dim=block_state_dim,
            teacher_dim=teacher_dim,
            hidden_dim=self.activity_config["activity_hidden_dim"],
            n_groups=n_groups,
            budget_min=self.activity_config["activity_budget_min"],
            budget_max=self.activity_config["activity_budget_max"],
            init_gate=self.activity_config["init_activity_gate"],
            init_budget=self.activity_config["init_activity_budget"],
        )
        self.activity_allocator = LocationActivityAllocator(self.activity_config["allocator_hidden_dim"])

    def maybe_load_warm_start(self, ckpt_path: str | None, device: str | torch.device) -> None:
        if not ckpt_path:
            return
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = dict(payload["model_state_dict"])
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        skip_prefixes = (
            "activity_process.",
            "activity_allocator.",
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

    def _block_mask(self, block_idx: int, batch: int, device: torch.device) -> torch.Tensor:
        mask = (self.activity_block_ids == block_idx).to(device=device)
        return mask.view(1, self.decoder.n_frames, 1).expand(batch, -1, self.decoder.n_cells)

    def _expand_block_values(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, self.activity_block_ids]

    def _expand_block_group_values(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, self.activity_block_ids, :]

    def build_block_state_inputs(
        self,
        state_local: torch.Tensor,
        state_band: torch.Tensor,
        quiet_metric_local: torch.Tensor,
    ) -> torch.Tensor:
        batch = state_local.shape[0]
        abs_local = state_local.abs()
        quiet_abs = quiet_metric_local.abs()
        feats = []
        for b in range(self.n_activity_blocks):
            mask = self._block_mask(b, batch, state_local.device)
            denom = mask[:, :, :1].sum(dim=1).sum(dim=1).clamp_min(1.0)
            pos_mean = (F.relu(state_local) * mask).sum(dim=(1, 2)) / denom
            neg_mean = (F.relu(-state_local) * mask).sum(dim=(1, 2)) / denom
            abs_mean = (abs_local * mask).sum(dim=(1, 2)) / denom
            abs_max = (abs_local * mask).amax(dim=(1, 2))
            quiet_mean = (quiet_abs * mask).sum(dim=(1, 2)) / denom
            quiet_max = (quiet_abs * mask).amax(dim=(1, 2))
            band_part = state_band
            group_mean = self.activity_geometry.pool_group(abs_local * mask).mean(dim=1)
            group_max = self.activity_geometry.pool_group(abs_local * mask).amax(dim=1)
            feats.append(
                torch.cat(
                    [
                        pos_mean.unsqueeze(-1),
                        neg_mean.unsqueeze(-1),
                        abs_mean.unsqueeze(-1),
                        abs_max.unsqueeze(-1),
                        quiet_mean.unsqueeze(-1),
                        quiet_max.unsqueeze(-1),
                        band_part,
                        group_mean,
                        group_max,
                    ],
                    dim=-1,
                )
            )
        return torch.stack(feats, dim=1)

    def build_block_teacher_inputs(self, target_local_log: torch.Tensor) -> torch.Tensor:
        batch = target_local_log.shape[0]
        abs_target = target_local_log.abs()
        feats = []
        for b in range(self.n_activity_blocks):
            mask = self._block_mask(b, batch, target_local_log.device)
            denom = mask[:, :, :1].sum(dim=1).sum(dim=1).clamp_min(1.0)
            pos_mean = (F.relu(target_local_log) * mask).sum(dim=(1, 2)) / denom
            neg_mean = (F.relu(-target_local_log) * mask).sum(dim=(1, 2)) / denom
            abs_mean = (abs_target * mask).sum(dim=(1, 2)) / denom
            abs_max = (abs_target * mask).amax(dim=(1, 2))
            flat_mask = mask.reshape(batch, -1)
            flat_abs = abs_target.reshape(batch, -1)
            block_values = flat_abs * flat_mask.to(flat_abs.dtype)
            k = max(1, min(self.activity_config["teacher_topk"], int(flat_mask[0].sum().item())))
            topk_mass = block_values.topk(k, dim=1).values.sum(dim=1)
            total_mass = block_values.sum(dim=1).clamp_min(1e-6)
            concentration = topk_mass / total_mass
            group_mean = self.activity_geometry.pool_group(abs_target * mask).mean(dim=1)
            feats.append(
                torch.cat(
                    [
                        pos_mean.unsqueeze(-1),
                        neg_mean.unsqueeze(-1),
                        abs_mean.unsqueeze(-1),
                        abs_max.unsqueeze(-1),
                        concentration.unsqueeze(-1),
                        group_mean,
                    ],
                    dim=-1,
                )
            )
        return torch.stack(feats, dim=1)

    def infer_activity_process(
        self,
        path_context: torch.Tensor,
        state_local: torch.Tensor,
        state_band: torch.Tensor,
        quiet_metric_local: torch.Tensor,
        target_local_log: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        block_inputs = self.build_block_state_inputs(state_local, state_band, quiet_metric_local)
        teacher_inputs = None
        if target_local_log is not None:
            teacher_inputs = self.build_block_teacher_inputs(target_local_log)
        return self.activity_process(path_context, block_inputs, teacher_inputs)

    def build_activity_metric_controls(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        activity_pack: dict[str, torch.Tensor] | None = None,
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
        (
            raw_local,
            raw_band,
            quiet_metric_local,
            metric_band,
            local_gate,
            band_gate,
        ) = super().build_state_metric_controls(z_t_flat, t, path_context)
        state_local, state_band = self.build_state_features(z_t_flat)
        if activity_pack is None:
            activity_pack = self.infer_activity_process(path_context, state_local, state_band, quiet_metric_local)
        gate_key = "post_gate" if use_posterior and "post_gate" in activity_pack else "prior_gate"
        budget_key = "post_budget" if use_posterior and "post_budget" in activity_pack else "prior_budget"
        group_key = "post_group" if use_posterior and "post_group" in activity_pack else "prior_group"
        block_gate = activity_pack[gate_key].squeeze(-1)
        block_budget = activity_pack[budget_key].squeeze(-1)
        block_group = activity_pack[group_key]
        activity_gate = self._expand_block_values(block_gate)
        activity_budget = self._expand_block_values(block_budget)
        group_probs = self._expand_block_group_values(block_group)
        router_map = self.activity_geometry.broadcast_group(group_probs).to(
            device=quiet_metric_local.device,
            dtype=quiet_metric_local.dtype,
        )

        alloc_features = torch.stack(
            [
                state_local,
                quiet_metric_local,
                state_local.abs(),
                quiet_metric_local.abs(),
            ],
            dim=-1,
        )
        pos_scores, neg_scores = self.activity_allocator(alloc_features)
        alloc_pos = torch.zeros_like(pos_scores)
        alloc_neg = torch.zeros_like(neg_scores)
        for b in range(self.n_activity_blocks):
            mask = self._block_mask(b, pos_scores.shape[0], pos_scores.device)
            alloc_pos = alloc_pos + topk_soft_allocation_masked(
                pos_scores,
                mask,
                self.activity_config["activity_topk"],
                self.activity_config["activity_temperature"],
            )
            alloc_neg = alloc_neg + topk_soft_allocation_masked(
                neg_scores,
                mask,
                self.activity_config["activity_topk"],
                self.activity_config["activity_temperature"],
            )
        activity_component = (
            router_map
            * (alloc_pos - alloc_neg)
            * (activity_gate * activity_budget).unsqueeze(-1)
        )
        background_scale = (
            self.activity_config["background_floor"]
            + (1.0 - self.activity_config["background_floor"]) * (1.0 - activity_gate)
        ).unsqueeze(-1)
        metric_local = background_scale * quiet_metric_local + activity_component
        metric_local = metric_local - metric_local.mean(dim=(1, 2), keepdim=True)
        metric_local = metric_local.clamp(
            min=-self.metric_config["local_metric_clip"],
            max=self.metric_config["local_metric_clip"],
        )
        return (
            raw_local,
            raw_band,
            metric_local,
            metric_band,
            local_gate,
            band_gate,
            block_gate,
            block_budget,
            block_group,
            activity_gate,
            activity_budget,
            group_probs,
            alloc_pos,
            alloc_neg,
            quiet_metric_local,
            router_map,
        )

    def transport_velocity(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.transport_velocity_with_pack(z_t_flat, t, path_context, activity_pack=None, use_posterior=False)

    def transport_velocity_with_pack(
        self,
        z_t_flat: torch.Tensor,
        t: torch.Tensor,
        path_context: torch.Tensor,
        activity_pack: dict[str, torch.Tensor] | None = None,
        use_posterior: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            raw_local,
            raw_band,
            metric_local,
            metric_band,
            local_gate,
            band_gate,
            block_gate,
            block_budget,
            block_group,
            activity_gate,
            activity_budget,
            group_probs,
            alloc_pos,
            alloc_neg,
            quiet_metric_local,
            router_map,
        ) = self.build_activity_metric_controls(
            z_t_flat,
            t,
            path_context,
            activity_pack=activity_pack,
            use_posterior=use_posterior,
        )
        raw_v = self.path_transport(z_t_flat, t, path_context)
        raw_v_basis = raw_v.view(z_t_flat.shape[0], self.decoder.n_frames, self.decoder.n_cells)
        mod_v_basis = self.modulate_velocity_basis(raw_v_basis, metric_local, metric_band)
        flat_pos = alloc_pos.reshape(alloc_pos.shape[0], -1)
        flat_neg = alloc_neg.reshape(alloc_neg.shape[0], -1)
        flat_router = router_map.reshape(router_map.shape[0], -1)
        metrics = {
            "pred_local_abs_mean": raw_local.abs().mean(),
            "quiet_metric_local_abs_mean": quiet_metric_local.abs().mean(),
            "metric_local_abs_mean": metric_local.abs().mean(),
            "pred_band_high_mean": raw_band[:, 2].mean(),
            "pred_band_mid_mean": raw_band[:, 1].mean(),
            "pred_band_low_mean": raw_band[:, 0].mean(),
            "metric_band_high_mean": metric_band[:, 2].mean(),
            "metric_band_mid_mean": metric_band[:, 1].mean(),
            "metric_band_low_mean": metric_band[:, 0].mean(),
            "local_metric_budget": self.local_metric_budget(),
            "band_metric_budget": self.band_metric_budget(),
            "local_state_gate_mean": local_gate.mean(),
            "band_state_gate_mean": band_gate.mean(),
            "band_state_gate_high_mean": band_gate[:, 2].mean(),
            "block_activity_gate_mean": block_gate.mean(),
            "block_activity_budget_mean": block_budget.mean(),
            "block_activity_router_top1_mean": block_group.max(dim=-1).values.mean(),
            "activity_gate_mean": activity_gate.mean(),
            "activity_budget_mean": activity_budget.mean(),
            "activity_pos_top1_mean": flat_pos.max(dim=1).values.mean(),
            "activity_neg_top1_mean": flat_neg.max(dim=1).values.mean(),
            "activity_router_top1_mean": group_probs.max(dim=1).values.mean(),
            "activity_router_entropy_mean": (-(group_probs * group_probs.clamp_min(1e-8).log()).sum(dim=1)).mean(),
            "activity_router_map_top1_mean": flat_router.max(dim=1).values.mean(),
            "pred_velocity_norm_mean": mod_v_basis.reshape(z_t_flat.shape[0], -1).norm(dim=-1).mean(),
        }
        return mod_v_basis.reshape(z_t_flat.shape[0], -1), metrics


def bernoulli_kl(post_p: torch.Tensor, prior_p: torch.Tensor) -> torch.Tensor:
    post_p = post_p.clamp(1e-5, 1.0 - 1e-5)
    prior_p = prior_p.clamp(1e-5, 1.0 - 1e-5)
    return post_p * (post_p.log() - prior_p.log()) + (1.0 - post_p) * (
        (1.0 - post_p).log() - (1.0 - prior_p).log()
    )


def latent_activity_process_flow_matching_loss(
    model: LatentActivityProcessTransportModel,
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
    state_local, state_band = model.build_state_features(z_t)
    _, _, quiet_metric_local, _, _, _ = super(
        LatentActivityProcessTransportModel, model
    ).build_state_metric_controls(z_t, t, path_context)
    activity_pack = model.infer_activity_process(
        path_context=path_context,
        state_local=state_local,
        state_band=state_band,
        quiet_metric_local=quiet_metric_local,
        target_local_log=target_local_log,
    )
    pred_v, control_metrics = model.transport_velocity_with_pack(z_t, t, path_context, activity_pack, use_posterior=True)
    fm_loss = F.mse_loss(pred_v, target_v)

    (
        raw_local,
        raw_band,
        metric_local,
        metric_band,
        _local_gate,
        _band_gate,
        block_gate,
        block_budget,
        block_group,
        activity_gate,
        activity_budget,
        group_probs,
        alloc_pos,
        alloc_neg,
        quiet_metric_local,
        _router_map,
    ) = model.build_activity_metric_controls(z_t, t, path_context, activity_pack=activity_pack, use_posterior=True)
    local_loss = F.smooth_l1_loss(metric_local, target_local_log)
    band_loss = F.smooth_l1_loss(metric_band, target_band_log)

    surf = metric_local.view(metric_local.shape[0], metric_local.shape[1], 5, 5)
    time_smooth = (metric_local[:, 1:] - metric_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth

    full_mask = torch.ones_like(target_local_log, dtype=torch.bool)
    target_pos = topk_soft_allocation_masked(
        F.relu(target_local_log),
        full_mask,
        model.activity_config["teacher_topk"],
        model.activity_config["teacher_temperature"],
    )
    target_neg = topk_soft_allocation_masked(
        F.relu(-target_local_log),
        full_mask,
        model.activity_config["teacher_topk"],
        model.activity_config["teacher_temperature"],
    )
    support_weight = activity_gate.unsqueeze(-1)
    support_norm = support_weight.mean().clamp_min(0.05)
    support_loss = 0.5 * (
        (
            F.smooth_l1_loss(alloc_pos, target_pos, reduction="none") * support_weight
        ).mean()
        / support_norm
        + (
            F.smooth_l1_loss(alloc_neg, target_neg, reduction="none") * support_weight
        ).mean()
        / support_norm
    )

    budget_reg = (
        (model.local_metric_budget() - model.metric_config["init_local_metric_budget"]).pow(2)
        + (model.band_metric_budget() - model.metric_config["init_band_metric_budget"]).pow(2)
    )
    prior_gate = activity_pack["prior_gate"].squeeze(-1)
    prior_budget = activity_pack["prior_budget"].squeeze(-1)
    prior_group = activity_pack["prior_group"]
    post_gate = activity_pack["post_gate"].squeeze(-1)
    post_budget = activity_pack["post_budget"].squeeze(-1)
    post_group = activity_pack["post_group"]
    activity_gate_kl = bernoulli_kl(post_gate, prior_gate).mean()
    activity_budget_consistency = ((post_budget - prior_budget).pow(2) * post_gate).mean() / post_gate.mean().clamp_min(0.05)
    activity_group_kl = ((post_group.clamp_min(1e-6).log() - prior_group.clamp_min(1e-6).log()) * post_group).sum(dim=-1)
    activity_group_kl = (activity_group_kl * post_gate).mean() / post_gate.mean().clamp_min(0.05)

    total = (
        fm_loss
        + model.integrated_config["local_loss_weight"] * local_loss
        + model.integrated_config["band_loss_weight"] * band_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
        + model.metric_config["budget_reg_weight"] * budget_reg
        + model.activity_config["activity_support_loss_weight"] * support_loss
        + model.activity_config["activity_gate_kl_weight"] * activity_gate_kl
        + model.activity_config["activity_budget_consistency_weight"] * activity_budget_consistency
        + model.activity_config["activity_group_kl_weight"] * activity_group_kl
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
        "lat_total_loss": total,
        "lat_local_loss": local_loss,
        "lat_band_loss": band_loss,
        "lat_smooth_reg": smooth_reg,
        "lat_budget_reg": budget_reg,
        "lat_support_loss": support_loss,
        "lat_activity_gate_kl": activity_gate_kl,
        "lat_activity_budget_consistency": activity_budget_consistency,
        "lat_activity_group_kl": activity_group_kl,
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "post_activity_gate_mean": post_gate.mean(),
        "prior_activity_gate_mean": prior_gate.mean(),
        "post_group_top1_mean": post_group.max(dim=-1).values.mean(),
        "prior_group_top1_mean": prior_group.max(dim=-1).values.mean(),
        **control_metrics,
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: LatentActivityProcessTransportModel,
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
        "lat_total_loss",
        "lat_local_loss",
        "lat_band_loss",
        "lat_smooth_reg",
        "lat_budget_reg",
        "lat_support_loss",
        "lat_activity_gate_kl",
        "lat_activity_budget_consistency",
        "lat_activity_group_kl",
        "target_local_abs_mean",
        "target_band_high_mean",
        "post_activity_gate_mean",
        "prior_activity_gate_mean",
        "post_group_top1_mean",
        "prior_group_top1_mean",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "metric_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "metric_band_high_mean",
        "metric_band_mid_mean",
        "metric_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "block_activity_gate_mean",
        "block_activity_budget_mean",
        "block_activity_router_top1_mean",
        "activity_gate_mean",
        "activity_budget_mean",
        "activity_pos_top1_mean",
        "activity_neg_top1_mean",
        "activity_router_top1_mean",
        "activity_router_entropy_mean",
        "activity_router_map_top1_mean",
    ]
    totals = {f"val_{k}": 0.0 for k in keys}
    total_count = 0
    for history_01, future_01 in val_loader:
        history_01 = history_01.to(next(model.parameters()).device)
        future_01 = future_01.to(next(model.parameters()).device)
        _loss, metrics = latent_activity_process_flow_matching_loss(model, history_01, future_01)
        batch_size = history_01.shape[0]
        for key in totals:
            totals[key] += metrics[key.replace("val_", "")].item() * batch_size
        total_count += batch_size
    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_total_loss"] = out["val_lat_total_loss"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="184b_v0: latent activity-process transport")
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
    parser.add_argument("--activity_hidden_dim", type=int, default=128)
    parser.add_argument("--allocator_hidden_dim", type=int, default=96)
    parser.add_argument("--n_activity_blocks", type=int, default=5)
    parser.add_argument("--init_activity_gate", type=float, default=0.12)
    parser.add_argument("--activity_budget_min", type=float, default=0.08)
    parser.add_argument("--activity_budget_max", type=float, default=0.45)
    parser.add_argument("--init_activity_budget", type=float, default=0.24)
    parser.add_argument("--activity_topk", type=int, default=5)
    parser.add_argument("--activity_temperature", type=float, default=0.20)
    parser.add_argument("--target_topk", type=int, default=5)
    parser.add_argument("--target_temperature", type=float, default=0.20)
    parser.add_argument("--teacher_topk", type=int, default=5)
    parser.add_argument("--teacher_temperature", type=float, default=0.20)
    parser.add_argument("--background_floor", type=float, default=0.35)
    parser.add_argument("--activity_support_loss_weight", type=float, default=0.45)
    parser.add_argument("--activity_gate_kl_weight", type=float, default=0.20)
    parser.add_argument("--activity_budget_consistency_weight", type=float, default=0.05)
    parser.add_argument("--activity_group_kl_weight", type=float, default=0.02)
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
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
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
    activity_config = dict(
        group_ids=[0] * 25,
        activity_hidden_dim=args.activity_hidden_dim,
        allocator_hidden_dim=args.allocator_hidden_dim,
        n_activity_blocks=args.n_activity_blocks,
        init_activity_gate=args.init_activity_gate,
        activity_budget_min=args.activity_budget_min,
        activity_budget_max=args.activity_budget_max,
        init_activity_budget=args.init_activity_budget,
        activity_topk=args.activity_topk,
        activity_temperature=args.activity_temperature,
        teacher_topk=args.teacher_topk,
        teacher_temperature=args.teacher_temperature,
        background_floor=args.background_floor,
        activity_support_loss_weight=args.activity_support_loss_weight,
        activity_gate_kl_weight=args.activity_gate_kl_weight,
        activity_budget_consistency_weight=args.activity_budget_consistency_weight,
        activity_group_kl_weight=args.activity_group_kl_weight,
    )

    model = LatentActivityProcessTransportModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        path_config=path_config,
        prior_config=prior_config,
        integrated_config=integrated_config,
        state_config=state_config,
        metric_config=metric_config,
        activity_config=activity_config,
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
    model.width_allocator.requires_grad_(False)
    model.band_tail.requires_grad_(False)
    model.local_state_gate.requires_grad_(False)
    model.band_state_gate.requires_grad_(False)
    model.local_metric_budget_logit.requires_grad_(False)
    model.band_metric_budget_logit.requires_grad_(False)

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    def set_stage(stage: int):
        activity_params = (
            list(model.activity_process.parameters())
            + list(model.activity_allocator.parameters())
        )
        if stage == 1:
            model.path_transport.requires_grad_(False)
            model.path_context_adapter.requires_grad_(False)
            optimizer = torch.optim.AdamW(activity_params, lr=args.lr_ctrl_stage1, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))
            stage_name = "latent-activity-process-ctrl"
        else:
            model.path_transport.requires_grad_(True)
            model.path_context_adapter.requires_grad_(True)
            params = [
                {"params": activity_params, "lr": args.lr_ctrl_stage2},
                {
                    "params": list(model.path_transport.parameters()) + list(model.path_context_adapter.parameters()),
                    "lr": args.lr_path_stage2,
                },
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))
            stage_name = "latent-activity-process-joint"
        return optimizer, scheduler, stage_name

    optimizer, scheduler, stage_name = set_stage(1)
    total_epochs = args.epochs_stage1 + args.epochs_stage2

    print(f"\n{'=' * 72}\n184b_v0: latent activity-process transport\n{'=' * 72}")
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
        "lat_total_loss",
        "lat_local_loss",
        "lat_band_loss",
        "lat_smooth_reg",
        "lat_budget_reg",
        "lat_support_loss",
        "lat_activity_gate_kl",
        "lat_activity_budget_consistency",
        "lat_activity_group_kl",
        "target_local_abs_mean",
        "target_band_high_mean",
        "post_activity_gate_mean",
        "prior_activity_gate_mean",
        "post_group_top1_mean",
        "prior_group_top1_mean",
        "pred_local_abs_mean",
        "quiet_metric_local_abs_mean",
        "metric_local_abs_mean",
        "pred_band_high_mean",
        "pred_band_mid_mean",
        "pred_band_low_mean",
        "metric_band_high_mean",
        "metric_band_mid_mean",
        "metric_band_low_mean",
        "local_metric_budget",
        "band_metric_budget",
        "local_state_gate_mean",
        "band_state_gate_mean",
        "band_state_gate_high_mean",
        "block_activity_gate_mean",
        "block_activity_budget_mean",
        "block_activity_router_top1_mean",
        "activity_gate_mean",
        "activity_budget_mean",
        "activity_pos_top1_mean",
        "activity_neg_top1_mean",
        "activity_router_top1_mean",
        "activity_router_entropy_mean",
        "activity_router_map_top1_mean",
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
            loss, metrics = latent_activity_process_flow_matching_loss(model, history_01, future_01)
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
                        "type": "latent_activity_process_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_184b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "activity": activity_config,
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
            f"lBud={val_metrics['val_local_metric_budget']:.3f}  bBud={val_metrics['val_band_metric_budget']:.3f}  "
            f"pGate={val_metrics['val_prior_activity_gate_mean']:.3f}  qGate={val_metrics['val_post_activity_gate_mean']:.3f}  "
            f"aBud={val_metrics['val_activity_budget_mean']:.3f}  "
            f"aTop={val_metrics['val_activity_pos_top1_mean']:.3f}/{val_metrics['val_activity_neg_top1_mean']:.3f}  "
            f"gTop={val_metrics['val_activity_router_top1_mean']:.3f}  "
            f"({elapsed:.1f}s)" + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": total_epochs,
        "config": {
            "type": "latent_activity_process_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_184b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "path": path_config,
            "prior": prior_config,
            "integrated": integrated_config,
            "state": state_config,
            "metric": metric_config,
            "activity": activity_config,
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
