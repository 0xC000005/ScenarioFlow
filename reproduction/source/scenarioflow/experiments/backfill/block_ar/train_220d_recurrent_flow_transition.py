#!/usr/bin/env python
"""
220d: recurrent-state conditional flow transition model for multi-day rollout.

Goal:
  - keep the successful 212ai local-scale asinh innovation geometry
  - replace sliding-window re-encoding during rollout with a persistent hidden
    state and online EWMA scale update
  - train directly on multistep futures with optional scheduled self-feeding
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)
from experiments.backfill.block_ar.train_193a_graph_ar_latent_factor_innovation import (
    evaluate_rollout_subset,
)


class RecurrentFlowTransitionModel(ConditionalFlowLocalScaleAsinhNLLModel):
    def __init__(self, *args, support_lo: float = 0.0, support_hi: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_lo = float(support_lo)
        self.support_hi = float(support_hi)
        self.recurrent_cells = nn.ModuleList(
            [
                nn.GRUCell(self.gru.input_size if i == 0 else self.hidden_dim, self.hidden_dim)
                for i in range(self.gru.num_layers)
            ]
        )

    def init_recurrent_cells_from_gru(self) -> None:
        with torch.no_grad():
            for layer_idx, cell in enumerate(self.recurrent_cells):
                cell.weight_ih.copy_(getattr(self.gru, f"weight_ih_l{layer_idx}"))
                cell.weight_hh.copy_(getattr(self.gru, f"weight_hh_l{layer_idx}"))
                cell.bias_ih.copy_(getattr(self.gru, f"bias_ih_l{layer_idx}"))
                cell.bias_hh.copy_(getattr(self.gru, f"bias_hh_l{layer_idx}"))

    def encode_history_with_state(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, local_scale = build_local_scale_history_features(
            history_01=history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        return h_n, local_scale, prev

    def recurrent_step(self, step_feat: torch.Tensor, state_stack: torch.Tensor) -> torch.Tensor:
        next_states = []
        layer_input = step_feat
        for layer_idx, cell in enumerate(self.recurrent_cells):
            h_next = cell(layer_input, state_stack[layer_idx])
            next_states.append(h_next)
            layer_input = h_next
        return torch.stack(next_states, dim=0)

    def _step_features(
        self,
        prev_01: torch.Tensor,
        next_01: torch.Tensor,
        current_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = next_01 - prev_01
        new_scale = (
            self.ewma_alpha * delta.abs().clamp_min(self.scale_floor)
            + (1.0 - self.ewma_alpha) * current_scale
        ).clamp_min(self.scale_floor)
        feat_parts = [(next_01 * 2.0 - 1.0), delta / new_scale]
        if self.include_scale_feature:
            feat_parts.append(torch.log(new_scale))
        feat = torch.cat(feat_parts, dim=-1)
        return feat, new_scale

    def sample_next_from_state(
        self,
        state_stack: torch.Tensor,
        prev_01: torch.Tensor,
        local_scale: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_top = state_stack[-1]
        batch = state_top.shape[0]
        cond = self._expand_condition(state_top, n_samples)
        if noise is None:
            z = torch.randn(batch, n_samples, self.n_cells, device=state_top.device, dtype=state_top.dtype)
        else:
            z = noise.to(device=state_top.device, dtype=state_top.dtype)
        v, _logdet = self._flow_forward(z, cond)
        innovation = torch.sinh(v)
        delta = innovation * local_scale.unsqueeze(1)
        next_iv = (prev_01.unsqueeze(1) + delta).clamp(self.support_lo, self.support_hi)
        return v, next_iv

    def log_prob_v_given_state(self, state_stack: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        cond = self._expand_condition(state_stack[-1], 1)
        z, logdet_inv = self._flow_inverse(target_v.unsqueeze(1), cond)
        log_base = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=-1)
        return (log_base + logdet_inv).squeeze(1)

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        state_stack, local_scale, prev = self.encode_history_with_state(history_01)
        _v, next_iv = self.sample_next_from_state(state_stack, prev, local_scale, n_samples=n_samples, noise=noise)
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
        base_state, base_scale, base_prev = self.encode_history_with_state(history_01)

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

            frames = []
            for _ in range(n_steps):
                _v, next_iv = self.sample_next_from_state(state_stack, prev, local_scale, n_samples=1)
                next_flat = next_iv.squeeze(1)
                frames.append(next_flat.view(batch_size, k, 5, 5))
                step_feat, local_scale = self._step_features(prev, next_flat, local_scale)
                state_stack = self.recurrent_step(step_feat, state_stack)
                prev = next_flat
            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def multistep_flow_loss(
    model: RecurrentFlowTransitionModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    train_samples: int,
    nll_weight: float,
    self_feed_prob: float,
    state_consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = history_01.shape[0]
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(batch_size, future_01.shape[1], -1)
    future_len = future_flat.shape[1]

    state_stack, local_scale, prev = model.encode_history_with_state(history_01)
    student_state_stack = state_stack.clone() if state_consistency_weight > 0.0 else None
    student_local_scale = local_scale.clone() if state_consistency_weight > 0.0 else None
    student_prev = prev.clone() if state_consistency_weight > 0.0 else None
    total_es = torch.tensor(0.0, device=history_01.device)
    total_nll = torch.tensor(0.0, device=history_01.device)
    total_mae = torch.tensor(0.0, device=history_01.device)
    total_v_std = torch.tensor(0.0, device=history_01.device)
    total_delta_std = torch.tensor(0.0, device=history_01.device)
    total_state_consistency = torch.tensor(0.0, device=history_01.device)
    pred_mean_path = []
    target_mean_path = []

    for step in range(future_len):
        target_t = future_flat[:, step, :]
        target_delta = target_t - prev
        target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))

        v_samples, next_samples = model.sample_next_from_state(
            state_stack=state_stack,
            prev_01=prev,
            local_scale=local_scale,
            n_samples=train_samples,
        )
        es_t = energy_score(v_samples, target_v)
        nll_t = -model.log_prob_v_given_state(state_stack, target_v).mean() / model.n_cells
        mean_next = next_samples.mean(dim=1)

        total_es = total_es + es_t
        total_nll = total_nll + nll_t
        total_mae = total_mae + (mean_next - target_t).abs().mean()
        total_v_std = total_v_std + v_samples.std(dim=1).mean()
        total_delta_std = total_delta_std + (next_samples - prev.unsqueeze(1)).std(dim=1).mean()
        pred_mean_path.append(mean_next.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        use_pred = (
            self_feed_prob > 0.0
            and step < future_len - 1
            and torch.rand((), device=history_01.device).item() < self_feed_prob
        )
        next_frame = mean_next.detach() if use_pred else target_t
        step_feat, next_local_scale = model._step_features(prev, next_frame, local_scale)
        next_state_stack = model.recurrent_step(step_feat, state_stack)

        if state_consistency_weight > 0.0:
            assert student_state_stack is not None
            assert student_local_scale is not None
            assert student_prev is not None
            _student_v, student_next_samples = model.sample_next_from_state(
                state_stack=student_state_stack,
                prev_01=student_prev,
                local_scale=student_local_scale,
                n_samples=min(8, train_samples),
            )
            student_mean_next = student_next_samples.mean(dim=1)
            student_step_feat, next_student_scale = model._step_features(
                student_prev,
                student_mean_next,
                student_local_scale,
            )
            next_student_state = model.recurrent_step(student_step_feat, student_state_stack)
            total_state_consistency = total_state_consistency + F.mse_loss(
                next_student_state,
                next_state_stack.detach(),
            )
            student_state_stack = next_student_state
            student_local_scale = next_student_scale
            student_prev = student_mean_next

        local_scale = next_local_scale
        state_stack = next_state_stack
        prev = next_frame

    path_mean_loss = F.smooth_l1_loss(torch.stack(pred_mean_path, dim=1), torch.stack(target_mean_path, dim=1))
    total_loss = (
        (total_es / future_len)
        + nll_weight * (total_nll / future_len)
        + 0.05 * path_mean_loss
        + state_consistency_weight * (total_state_consistency / future_len)
    )
    metrics = {
        "total_loss": total_loss.detach(),
        "multistep_energy": (total_es / future_len).detach(),
        "multistep_nll": (total_nll / future_len).detach(),
        "multistep_mae": (total_mae / future_len).detach(),
        "sample_v_std": (total_v_std / future_len).detach(),
        "sample_delta_std": (total_delta_std / future_len).detach(),
        "path_mean_loss": path_mean_loss.detach(),
        "state_consistency": (total_state_consistency / future_len).detach(),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: RecurrentFlowTransitionModel,
    val_loader: DataLoader,
    train_samples: int,
    nll_weight: float,
    state_consistency_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for history_01, future_01 in val_loader:
        _loss, metrics = multistep_flow_loss(
            model,
            history_01,
            future_01,
            train_samples=train_samples,
            nll_weight=nll_weight,
            self_feed_prob=0.0,
            state_consistency_weight=state_consistency_weight,
        )
        bs = history_01.shape[0]
        total_count += bs
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * bs
    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


def load_model(checkpoint_path: str, device: torch.device) -> tuple[RecurrentFlowTransitionModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] not in {
        "recurrent_flow_local_scale_asinh_220d",
        "recurrent_flow_local_scale_asinh_220e",
        "recurrent_flow_local_scale_asinh_220f",
    }:
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = RecurrentFlowTransitionModel(
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
        support_lo=cfg.get("support_lo", 0.0),
        support_hi=cfg.get("support_hi", 1.0),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="220d recurrent-state flow transition model")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=24)
    parser.add_argument("--rollout_val_samples", type=int, default=20)
    parser.add_argument("--rollout_eval_limit", type=int, default=160)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--self_feed_prob_max", type=float, default=0.0)
    parser.add_argument("--state_consistency_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    cfg = init_payload["config"]
    model = RecurrentFlowTransitionModel(
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
    ).to(device)
    model.load_state_dict(init_payload["model_state_dict"], strict=False)
    model.init_recurrent_cells_from_gru()

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.future_len
    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]

    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("220d recurrent-state flow transition model")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  future_len={args.future_len} train_samples={args.train_samples} self_feed_prob_max={args.self_feed_prob_max}")

    best_score = float("inf")
    history: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    if args.state_consistency_weight > 0.0:
        rollout_objective_name = "220f"
    elif args.self_feed_prob_max == 0.0:
        rollout_objective_name = "220d"
    else:
        rollout_objective_name = "220e"
    model_type = f"recurrent_flow_local_scale_asinh_{rollout_objective_name}"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        self_feed_prob = args.self_feed_prob_max * (epoch / args.epochs)
        model.train()
        running: dict[str, float] = {}
        n_count = 0

        for history_01, future_01 in train_loader:
            loss, metrics = multistep_flow_loss(
                model,
                history_01,
                future_01,
                train_samples=args.train_samples,
                nll_weight=args.nll_weight,
                self_feed_prob=self_feed_prob,
                state_consistency_weight=args.state_consistency_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = history_01.shape[0]
            n_count += bs
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value.item()) * bs

        train_metrics = {f"train_{k}": v / max(n_count, 1) for k, v in running.items()}
        val_tf = evaluate_teacher_forced(
            model,
            val_loader,
            train_samples=min(args.train_samples, 16),
            nll_weight=args.nll_weight,
            state_consistency_weight=args.state_consistency_weight,
        )
        rollout_probe = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )
        score = (
            max(0.0, 0.80 - rollout_probe["rollout_cov90"])
            + max(0.0, 1.15 - rollout_probe["rollout_turb_calm_ratio"])
            + max(0.0, 0.80 - rollout_probe["rollout_corr_ratio_h30"])
            + max(0.0, rollout_probe["rollout_rank_ratio_h30"] - 2.5)
            + 5.0 * max(0.0, rollout_probe["rollout_support_violation_rate"] - 0.01)
            + 0.1 * val_tf.get("val_path_mean_loss", 0.0)
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "self_feed_prob": self_feed_prob,
            **train_metrics,
            **val_tf,
            **rollout_probe,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": model_type,
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": args.history_len,
                "future_len": args.future_len,
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "support_lo": 0.0,
                "support_hi": 1.0,
                "train_samples": args.train_samples,
                "nll_weight": args.nll_weight,
                "self_feed_prob_max": args.self_feed_prob_max,
                "state_consistency_weight": args.state_consistency_weight,
                "init_checkpoint": args.init_checkpoint,
            },
            "metrics": history[-1],
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"sf={self_feed_prob:.3f} "
            f"train_loss={train_metrics['train_total_loss']:.4f} "
            f"roll_cov={rollout_probe['rollout_cov90']:.3f} "
            f"roll_turb/calm={rollout_probe['rollout_turb_calm_ratio']:.3f} "
            f"roll_corr={rollout_probe['rollout_corr_ratio_h30']:.3f} "
            f"roll_rank={rollout_probe['rollout_rank_ratio_h30']:.3f} "
            f"score={score:.3f}"
        )

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
