#!/usr/bin/env python
"""
221d: Temporal adapter for multi-day AR conditional flow.

Key insight from 221a/b/c: the flow's conditioning sensitivity means GRU changes
propagate to output quality. GRU LR >= 1e-3 enables h=30 MR but destroys h=1 coverage.
GRU LR = 5e-5 preserves h=1 but prevents h=30 learning.

Solution: decouple the paths. Keep GRU frozen (5e-5) for base conditioning.
Add a small MLP adapter that takes the recurrent cell state (learned at 1e-3)
and produces a residual added to the base GRU state. This injects temporal
information without disrupting the base distribution.

Pattern from 220g: state_top = state_stack[-1] + self.regime_to_cond(regime_state)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
    RecurrentFlowTransitionModel,
)


class AdapterRecurrentFlowTransitionModel(RecurrentFlowTransitionModel):
    """RecurrentFlowTransitionModel with a temporal adapter for decoupled conditioning.

    The adapter takes the evolving recurrent cell state and produces a residual
    that is added to the fixed GRU base state before conditioning the flow.
    """

    def __init__(self, *args, adapter_hidden: int = 64, adapter_ramp_steps: int = 0, **kwargs):
        kwargs.pop("adapter_ramp_steps", None)  # don't pass to parent
        super().__init__(*args, **kwargs)
        self.adapter_hidden = adapter_hidden
        self.adapter_ramp_steps = adapter_ramp_steps  # 0 = no ramp (full adapter at all steps)
        self.temporal_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, self.hidden_dim),
        )
        self._init_adapter()

    def _init_adapter(self):
        for m in self.temporal_adapter:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def encode_history_with_state(self, history_01):
        h_n, local_scale, prev = super().encode_history_with_state(history_01)
        gru_base = h_n[-1].clone()  # Fixed base state for adapter
        return h_n, local_scale, prev, gru_base

    def sample_next_from_state_with_adapter(
        self,
        state_stack: torch.Tensor,
        gru_base: torch.Tensor,
        prev_01: torch.Tensor,
        local_scale: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
        adapter_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_state = state_stack[-1]
        adapter_out = self.temporal_adapter(cell_state)
        augmented = gru_base + adapter_weight * adapter_out
        batch = augmented.shape[0]
        cond = self._expand_condition(augmented, n_samples)
        if noise is None:
            z = torch.randn(batch, n_samples, self.n_cells, device=augmented.device, dtype=augmented.dtype)
        else:
            z = noise.to(device=augmented.device, dtype=augmented.dtype)
        v, _logdet = self._flow_forward(z, cond)
        innovation = torch.sinh(v)
        delta = innovation * local_scale.unsqueeze(1)
        next_iv = (prev_01.unsqueeze(1) + delta).clamp(self.support_lo, self.support_hi)
        return v, next_iv

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        state_stack, local_scale, prev, gru_base = self.encode_history_with_state(history_01)
        _v, next_iv = self.sample_next_from_state_with_adapter(
            state_stack, gru_base, prev, local_scale, n_samples=n_samples, noise=noise
        )
        return next_iv

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
        batch_size = history_01.shape[0]
        base_state, base_scale, base_prev, gru_base = self.encode_history_with_state(history_01)

        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            state_stack = (
                base_state.unsqueeze(2)
                .expand(self.gru.num_layers, batch_size, k, self.hidden_dim)
                .reshape(self.gru.num_layers, batch_size * k, self.hidden_dim)
                .clone()
            )
            local_scale = base_scale.unsqueeze(1).expand(batch_size, k, -1).reshape(batch_size * k, -1).clone()
            prev = base_prev.unsqueeze(1).expand(batch_size, k, -1).reshape(batch_size * k, -1).clone()
            gru_base_k = gru_base.unsqueeze(1).expand(batch_size, k, -1).reshape(batch_size * k, -1).clone()

            frames = []
            for step_idx in range(n_steps):
                aw = min(step_idx / max(self.adapter_ramp_steps, 1), 1.0) if self.adapter_ramp_steps > 0 else 1.0
                _v, next_iv = self.sample_next_from_state_with_adapter(
                    state_stack, gru_base_k, prev, local_scale, n_samples=1,
                    adapter_weight=aw,
                )
                next_flat = next_iv.squeeze(1)
                frames.append(next_flat.view(batch_size, k, 5, 5))
                step_feat, local_scale = self._step_features(prev, next_flat, local_scale)
                state_stack = self.recurrent_step(step_feat, state_stack)
                prev = next_flat
            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def parse_curriculum(s: str) -> list[tuple[int, int]]:
    pairs = []
    for part in s.split(","):
        epoch_str, horizon_str = part.split(":")
        pairs.append((int(epoch_str), int(horizon_str)))
    return sorted(pairs, key=lambda x: x[0])


def get_curriculum_horizon(epoch: int, schedule: list[tuple[int, int]]) -> int:
    h = schedule[0][1]
    for ep, horizon in schedule:
        if epoch >= ep:
            h = horizon
    return h


def get_scheduled_sampling_prob(epoch: int) -> float:
    if epoch <= 20:
        return 0.0
    if epoch <= 35:
        return 0.3 * (epoch - 20) / 15
    return min(0.5, 0.3 + 0.2 * (epoch - 35) / 15)


def multistep_ar_flow_loss(
    model: AdapterRecurrentFlowTransitionModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    train_samples: int,
    future_len: int,
    scoring_horizons: list[int],
    horizon_weights: list[float],
    free_run_prob: float,
    bptt_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = history_01.shape[0]
    device = history_01.device
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(batch_size, future_01.shape[1], -1)

    state_stack, local_scale, prev, gru_base = model.encode_history_with_state(history_01)

    anchor_loss = torch.tensor(0.0, device=device)
    curriculum_loss = torch.tensor(0.0, device=device)
    n_curriculum = 0
    running_es_innov = 0.0
    running_es_level = 0.0
    running_mae = 0.0
    running_v_std = 0.0
    running_floor = 0.0
    running_adapter_norm = 0.0
    count_steps = 0

    for step in range(future_len):
        target_t = future_flat[:, step, :]
        target_v = torch.asinh(
            (target_t - prev) / local_scale.clamp_min(model.scale_floor)
        )

        aw = min(step / max(model.adapter_ramp_steps, 1), 1.0) if model.adapter_ramp_steps > 0 else 1.0
        v_samples, next_samples = model.sample_next_from_state_with_adapter(
            state_stack=state_stack,
            gru_base=gru_base,
            prev_01=prev,
            local_scale=local_scale,
            n_samples=train_samples,
            adapter_weight=aw,
        )

        h = step + 1

        es_innov = energy_score(v_samples, target_v)
        es_level = energy_score(next_samples, target_t)
        es_combined = 0.5 * es_innov + 0.5 * es_level

        if h == 1:
            anchor_loss = es_combined
            running_es_innov += float(es_innov.detach())
            running_es_level += float(es_level.detach())
        elif h in scoring_horizons:
            idx = scoring_horizons.index(h)
            w = horizon_weights[idx] if idx < len(horizon_weights) else 1.0
            curriculum_loss = curriculum_loss + w * es_combined
            n_curriculum += 1
            running_es_innov += float(es_innov.detach())
            running_es_level += float(es_level.detach())

        mean_next = next_samples.mean(dim=1)
        running_mae += float((mean_next - target_t).abs().mean().detach())
        running_v_std += float(v_samples.std(dim=1).mean().detach())
        running_floor += float((next_samples < 0.02).float().mean().detach())
        running_adapter_norm += float(model.temporal_adapter(state_stack[-1]).norm(dim=-1).mean().detach())
        count_steps += 1

        use_own = (
            free_run_prob > 0.0
            and step < future_len - 1
            and torch.rand(()).item() < free_run_prob
        )
        next_frame = mean_next.detach() if use_own else target_t

        step_feat, next_scale = model._step_features(prev, next_frame, local_scale)
        state_stack = model.recurrent_step(step_feat, state_stack)
        local_scale = next_scale
        prev = next_frame

        if bptt_steps > 0 and (step + 1) % bptt_steps == 0:
            state_stack = state_stack.detach()
            local_scale = local_scale.detach()
            prev = prev.detach()

    if n_curriculum > 0:
        curriculum_loss = curriculum_loss / n_curriculum
    total_loss = 0.5 * anchor_loss + 0.5 * curriculum_loss

    n_scored = n_curriculum + 1
    metrics = {
        "total_loss": total_loss.detach(),
        "anchor_es": float(anchor_loss.detach()),
        "curriculum_es": float(curriculum_loss.detach()),
        "mean_es_innov": running_es_innov / max(n_scored, 1),
        "mean_es_level": running_es_level / max(n_scored, 1),
        "mean_mae": running_mae / max(count_steps, 1),
        "mean_v_std": running_v_std / max(count_steps, 1),
        "mean_floor_rate": running_floor / max(count_steps, 1),
        "adapter_norm": running_adapter_norm / max(count_steps, 1),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_rollout(
    model: AdapterRecurrentFlowTransitionModel,
    val_loader: DataLoader,
    future_len: int,
    eval_samples: int,
    scoring_horizons: list[int],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0

    for history_01, future_01 in val_loader:
        batch_size = history_01.shape[0]
        future_flat = future_01.reshape(batch_size, future_01.shape[1], -1)
        state_stack, local_scale, prev, gru_base = model.encode_history_with_state(history_01)

        for step in range(future_len):
            target_t = future_flat[:, step, :]
            target_v = torch.asinh(
                (target_t - prev) / local_scale.clamp_min(model.scale_floor)
            )

            v_samples, next_samples = model.sample_next_from_state_with_adapter(
                state_stack, gru_base, prev, local_scale, n_samples=eval_samples
            )

            h = step + 1
            if h == 1 or h in scoring_horizons:
                es = float(energy_score(v_samples, target_v).item())
                totals[f"es_h{h}"] = totals.get(f"es_h{h}", 0.0) + es * batch_size

                mean_next = next_samples.mean(dim=1)
                mae = float((mean_next - target_t).abs().mean().item())
                totals[f"mae_h{h}"] = totals.get(f"mae_h{h}", 0.0) + mae * batch_size

                lower = next_samples.quantile(0.05, dim=1)
                upper = next_samples.quantile(0.95, dim=1)
                covered = ((target_t >= lower) & (target_t <= upper)).float().mean()
                totals[f"cov90_h{h}"] = totals.get(f"cov90_h{h}", 0.0) + float(covered) * batch_size

            step_feat, next_scale = model._step_features(prev, target_t, local_scale)
            state_stack = model.recurrent_step(step_feat, state_stack)
            local_scale = next_scale
            prev = target_t

        total_count += batch_size

    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_free_rollout(
    model: AdapterRecurrentFlowTransitionModel,
    val_loader: DataLoader,
    future_len: int,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0

    for history_01, future_01 in val_loader:
        batch_size = history_01.shape[0]
        future_flat = future_01.reshape(batch_size, future_01.shape[1], -1)
        state_stack, local_scale, prev, gru_base = model.encode_history_with_state(history_01)

        for step in range(future_len):
            target_t = future_flat[:, step, :]

            _v, next_samples = model.sample_next_from_state_with_adapter(
                state_stack, gru_base, prev, local_scale, n_samples=eval_samples
            )

            mean_next = next_samples.mean(dim=1)

            for h_check in [1, 5, 15, 30]:
                if step + 1 == h_check:
                    mae = float((mean_next - target_t).abs().mean().item())
                    totals[f"freerun_mae_h{h_check}"] = totals.get(f"freerun_mae_h{h_check}", 0.0) + mae * batch_size

                    lower = next_samples.quantile(0.05, dim=1)
                    upper = next_samples.quantile(0.95, dim=1)
                    covered = ((target_t >= lower) & (target_t <= upper)).float().mean()
                    totals[f"freerun_cov90_h{h_check}"] = (
                        totals.get(f"freerun_cov90_h{h_check}", 0.0) + float(covered) * batch_size
                    )

                    floor_rate = float((next_samples < 0.02).float().mean().item())
                    totals[f"freerun_floor_h{h_check}"] = (
                        totals.get(f"freerun_floor_h{h_check}", 0.0) + floor_rate * batch_size
                    )

            step_feat, next_scale = model._step_features(prev, mean_next, local_scale)
            state_stack = model.recurrent_step(step_feat, state_stack)
            local_scale = next_scale
            prev = mean_next

        total_count += batch_size

    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="221d adapter multi-day AR conditional flow")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--init_checkpoint", type=str,
                        default="models/backfill/minimal_h1_conditional_flow_212ai_full_h1_local_scale_asinh_staged_nll/best_model.pt")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_flow", type=float, default=1e-5)
    parser.add_argument("--lr_gru", type=float, default=5e-5)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_cells", type=float, default=1e-3)
    parser.add_argument("--adapter_hidden", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=32)
    parser.add_argument("--bptt_steps", type=int, default=10)
    parser.add_argument("--curriculum_schedule", type=str, default="0:5,15:15,30:30")
    parser.add_argument("--scoring_horizons", type=str, default="5,15,30")
    parser.add_argument("--horizon_weights", type=str, default="0.4,0.3,0.3")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--freerun_eval_interval", type=int, default=2)
    parser.add_argument("--freerun_eval_limit", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = parse_curriculum(args.curriculum_schedule)
    scoring_horizons = [int(x) for x in args.scoring_horizons.split(",")]
    horizon_weights = [float(x) for x in args.horizon_weights.split(",")]

    # --- Data ---
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.future_len
    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]

    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)

    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    freerun_hist = val_hist[: args.freerun_eval_limit]
    freerun_future = val_future[: args.freerun_eval_limit]
    freerun_loader = DataLoader(TensorDataset(freerun_hist, freerun_future), batch_size=args.batch_size, shuffle=False)

    # --- Model: warm-start from 212ai + adapter ---
    print(f"Loading init checkpoint: {args.init_checkpoint}")
    init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    cfg = init_payload["config"]
    model = AdapterRecurrentFlowTransitionModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        flow_hidden=cfg["flow_hidden"],
        n_coupling_layers=cfg["n_coupling_layers"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    missing, unexpected = model.load_state_dict(init_payload["model_state_dict"], strict=False)
    model.init_recurrent_cells_from_gru()
    print(f"  Loaded 212ai weights (missing={len(missing)}, unexpected={len(unexpected)})")

    # --- 4 param groups ---
    flow_params = list(model.layers.parameters())
    gru_params = list(model.gru.parameters())
    adapter_params = list(model.temporal_adapter.parameters())
    flow_ids = {id(p) for p in flow_params}
    gru_ids = {id(p) for p in gru_params}
    adapter_ids = {id(p) for p in adapter_params}
    cell_params = [p for p in model.parameters() if id(p) not in flow_ids | gru_ids | adapter_ids]

    param_groups = [
        {"params": flow_params, "lr": args.lr_flow},
        {"params": gru_params, "lr": args.lr_gru},
        {"params": adapter_params, "lr": args.lr_adapter},
        {"params": cell_params, "lr": args.lr_cells},
    ]
    total_assigned = sum(len(g["params"]) for g in param_groups)
    total_model = sum(1 for _ in model.parameters())
    assert total_assigned == total_model, f"Param group mismatch: {total_assigned} vs {total_model}"

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters())
    n_adapter = sum(p.numel() for p in adapter_params)
    print(f"221d adapter multi-day AR conditional flow")
    print(f"  params={n_params:,d} (adapter={n_adapter:,d})")
    print(f"  LR: flow={args.lr_flow} gru={args.lr_gru} adapter={args.lr_adapter} cells={args.lr_cells}")
    print(f"  curriculum={curriculum}")
    print(f"  train_windows={train_hist.shape[0]}  val_windows={val_hist.shape[0]}")

    history_log: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0

    config = {
        "type": "adapter_multiday_ar_221d",
        "n_cells": cfg["n_cells"],
        "history_feat_dim": cfg["history_feat_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "gru_layers": cfg["gru_layers"],
        "gru_dropout": cfg["gru_dropout"],
        "flow_hidden": cfg["flow_hidden"],
        "n_coupling_layers": cfg["n_coupling_layers"],
        "ewma_alpha": cfg["ewma_alpha"],
        "scale_floor": cfg["scale_floor"],
        "include_scale_feature": cfg["include_scale_feature"],
        "support_lo": args.support_lo,
        "support_hi": args.support_hi,
        "adapter_hidden": args.adapter_hidden,
        "history_len": args.history_len,
        "future_len": args.future_len,
        "init_checkpoint": args.init_checkpoint,
        "curriculum_schedule": args.curriculum_schedule,
        "scoring_horizons": scoring_horizons,
        "horizon_weights": horizon_weights,
        "bptt_steps": args.bptt_steps,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "lr_flow": args.lr_flow,
        "lr_gru": args.lr_gru,
        "lr_adapter": args.lr_adapter,
        "lr_cells": args.lr_cells,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_horizon = get_curriculum_horizon(epoch, curriculum)
        free_run_prob = get_scheduled_sampling_prob(epoch)
        active_horizons = [h for h in scoring_horizons if h <= curr_horizon]
        active_weights = [horizon_weights[scoring_horizons.index(h)] for h in active_horizons]

        model.train()
        running: dict[str, float] = {}
        count = 0

        for history_01, future_01 in train_loader:
            loss, metrics = multistep_ar_flow_loss(
                model=model,
                history_01=history_01,
                future_01=future_01,
                train_samples=args.train_samples,
                future_len=curr_horizon,
                scoring_horizons=active_horizons,
                horizon_weights=active_weights,
                free_run_prob=free_run_prob,
                bptt_steps=args.bptt_steps,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = history_01.shape[0]
            count += bs
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v) * bs

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        scheduler.step()

        val_scoring = [1] + active_horizons if 1 not in active_horizons else active_horizons
        val_metrics = evaluate_rollout(model, val_loader, future_len=min(curr_horizon, args.future_len),
                                       eval_samples=args.eval_samples, scoring_horizons=val_scoring)

        freerun_metrics: dict[str, float] = {}
        if epoch % args.freerun_eval_interval == 0 or epoch == args.epochs:
            freerun_metrics = evaluate_free_rollout(model, freerun_loader,
                                                    future_len=min(curr_horizon, args.future_len),
                                                    eval_samples=args.eval_samples)

        # Selection score
        score_parts: list[float] = []
        n_score = 0
        if "val_freerun_cov90_h1" in freerun_metrics:
            score_parts.append(2.0 * max(0.0, 0.80 - freerun_metrics["val_freerun_cov90_h1"]))
            n_score += 1
        if "val_freerun_cov90_h30" in freerun_metrics:
            score_parts.append(3.0 * max(0.0, 0.70 - freerun_metrics["val_freerun_cov90_h30"]))
            n_score += 1
        if "val_freerun_floor_h30" in freerun_metrics:
            score_parts.append(5.0 * freerun_metrics["val_freerun_floor_h30"])
            n_score += 1
        for h in [1] + active_horizons:
            es_key = f"val_es_h{h}"
            if es_key in val_metrics:
                score_parts.append(0.5 * val_metrics[es_key])
                n_score += 1
        score = sum(score_parts) / max(n_score, 1)

        adapter_norm = train_metrics.get("train_adapter_norm", 0.0)
        record = {
            "epoch": epoch, "elapsed_sec": time.time() - t0,
            "curriculum_horizon": curr_horizon, "free_run_prob": free_run_prob,
            "lr_flow": float(optimizer.param_groups[0]["lr"]),
            "lr_gru": float(optimizer.param_groups[1]["lr"]),
            "lr_adapter": float(optimizer.param_groups[2]["lr"]),
            "lr_cells": float(optimizer.param_groups[3]["lr"]),
            **train_metrics, **val_metrics, **freerun_metrics,
            "selection_score": float(score),
        }
        history_log.append(make_serializable(record))

        cov_h1 = val_metrics.get("val_cov90_h1", 0)
        es_str = " ".join(f"es_h{h}={val_metrics.get(f'val_es_h{h}', 0):.4f}" for h in [1] + active_horizons)
        fr_str = ""
        if freerun_metrics:
            fr_str = " | FR: " + " ".join(
                f"h{h}={freerun_metrics.get(f'val_freerun_cov90_h{h}', 0):.1%}"
                for h in [1, 15, 30] if f"val_freerun_cov90_h{h}" in freerun_metrics
            )
        print(
            f"  [{epoch:3d}/{args.epochs}] T={curr_horizon:2d} p_free={free_run_prob:.2f} "
            f"cov_h1={cov_h1:.1%} adapt={adapter_norm:.3f} | {es_str}{fr_str} | score={score:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": history_log[-1],
        }
        torch.save(payload, out_dir / "last_model.pt")

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, out_dir / "best_model.pt")
            print(f"    *** new best at epoch {epoch} (score={score:.4f})")

        if epoch == args.epochs:
            torch.save(payload, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nDone. Best epoch={best_epoch} score={best_score:.4f}")
    print(f"Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
