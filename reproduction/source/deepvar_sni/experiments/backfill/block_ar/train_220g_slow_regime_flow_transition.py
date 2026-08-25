#!/usr/bin/env python
"""
220g: slow-regime latent conditional flow transition model for multi-day rollout.

Goal:
  - keep the successful 212ai local-scale asinh innovation geometry
  - keep the persistent recurrent transition from 220d
  - add a low-dimensional slow regime/factor state that persists across days
    and modulates the shared flow condition
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_193a_graph_ar_latent_factor_innovation import (
    evaluate_rollout_subset,
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


class SlowRegimeFlowTransitionModel(ConditionalFlowLocalScaleAsinhNLLModel):
    def __init__(
        self,
        *args,
        support_lo: float = 0.0,
        support_hi: float = 1.0,
        regime_dim: int = 16,
        regime_alpha: float = 0.08,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.support_lo = float(support_lo)
        self.support_hi = float(support_hi)
        self.regime_dim = int(regime_dim)
        self.regime_alpha = float(regime_alpha)
        self.recurrent_cells = nn.ModuleList(
            [
                nn.GRUCell(self.gru.input_size if i == 0 else self.hidden_dim, self.hidden_dim)
                for i in range(self.gru.num_layers)
            ]
        )
        self.regime_init = nn.Linear(self.hidden_dim, self.regime_dim)
        self.regime_update = nn.GRUCell(self.hidden_dim, self.regime_dim)
        self.regime_to_cond = nn.Linear(self.regime_dim, self.hidden_dim)
        self._init_regime_params()

    def _init_regime_params(self) -> None:
        nn.init.normal_(self.regime_init.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regime_init.bias)
        nn.init.normal_(self.regime_to_cond.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.regime_to_cond.bias)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, local_scale = build_local_scale_history_features(
            history_01=history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        regime_state = torch.tanh(self.regime_init(h_n[-1]))
        return h_n, local_scale, prev, regime_state

    def recurrent_step(self, step_feat: torch.Tensor, state_stack: torch.Tensor) -> torch.Tensor:
        next_states = []
        layer_input = step_feat
        for layer_idx, cell in enumerate(self.recurrent_cells):
            h_next = cell(layer_input, state_stack[layer_idx])
            next_states.append(h_next)
            layer_input = h_next
        return torch.stack(next_states, dim=0)

    def update_regime_state(
        self,
        regime_state: torch.Tensor,
        state_stack: torch.Tensor,
    ) -> torch.Tensor:
        candidate = torch.tanh(self.regime_update(state_stack[-1], regime_state))
        return (1.0 - self.regime_alpha) * regime_state + self.regime_alpha * candidate

    def _condition_state(
        self,
        state_stack: torch.Tensor,
        regime_state: torch.Tensor,
    ) -> torch.Tensor:
        return state_stack[-1] + self.regime_to_cond(regime_state)

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
        regime_state: torch.Tensor,
        prev_01: torch.Tensor,
        local_scale: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_top = self._condition_state(state_stack, regime_state)
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

    def log_prob_v_given_state(
        self,
        state_stack: torch.Tensor,
        regime_state: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._expand_condition(self._condition_state(state_stack, regime_state), 1)
        z, logdet_inv = self._flow_inverse(target_v.unsqueeze(1), cond)
        log_base = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=-1)
        return (log_base + logdet_inv).squeeze(1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_stack, local_scale, prev, regime_state = self.encode_history_with_state(history_01)
        _v, next_iv = self.sample_next_from_state(
            state_stack,
            regime_state,
            prev,
            local_scale,
            n_samples=n_samples,
            noise=noise,
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
        base_state, base_scale, base_prev, base_regime = self.encode_history_with_state(history_01)

        all_chunks = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            state_stack = (
                base_state.unsqueeze(2)
                .expand(self.gru.num_layers, batch_size, k, self.hidden_dim)
                .reshape(self.gru.num_layers, batch_size * k, self.hidden_dim)
                .clone()
            )
            regime_state = (
                base_regime.unsqueeze(1)
                .expand(batch_size, k, self.regime_dim)
                .reshape(batch_size * k, self.regime_dim)
                .clone()
            )
            local_scale = base_scale.unsqueeze(1).expand(batch_size, k, -1).reshape(batch_size * k, -1).clone()
            prev = base_prev.unsqueeze(1).expand(batch_size, k, -1).reshape(batch_size * k, -1).clone()

            frames = []
            for _ in range(n_steps):
                _v, next_iv = self.sample_next_from_state(
                    state_stack,
                    regime_state,
                    prev,
                    local_scale,
                    n_samples=1,
                )
                next_flat = next_iv.squeeze(1)
                frames.append(next_flat.view(batch_size, k, 5, 5))
                step_feat, local_scale = self._step_features(prev, next_flat, local_scale)
                state_stack = self.recurrent_step(step_feat, state_stack)
                regime_state = self.update_regime_state(regime_state, state_stack)
                prev = next_flat
            all_chunks.append(torch.stack(frames, dim=2))
        return torch.cat(all_chunks, dim=1)


def multistep_flow_loss(
    model: SlowRegimeFlowTransitionModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    train_samples: int,
    nll_weight: float,
    regime_slow_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = history_01.shape[0]
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(batch_size, future_01.shape[1], -1)
    future_len = future_flat.shape[1]

    state_stack, local_scale, prev, regime_state = model.encode_history_with_state(history_01)
    total_es = torch.tensor(0.0, device=history_01.device)
    total_nll = torch.tensor(0.0, device=history_01.device)
    total_mae = torch.tensor(0.0, device=history_01.device)
    total_v_std = torch.tensor(0.0, device=history_01.device)
    total_delta_std = torch.tensor(0.0, device=history_01.device)
    total_regime_drift = torch.tensor(0.0, device=history_01.device)
    pred_mean_path = []
    target_mean_path = []

    for step in range(future_len):
        target_t = future_flat[:, step, :]
        target_delta = target_t - prev
        target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))

        prev_regime_state = regime_state
        v_samples, next_samples = model.sample_next_from_state(
            state_stack=state_stack,
            regime_state=regime_state,
            prev_01=prev,
            local_scale=local_scale,
            n_samples=train_samples,
        )
        es_t = energy_score(v_samples, target_v)
        nll_t = -model.log_prob_v_given_state(state_stack, regime_state, target_v).mean() / model.n_cells
        mean_next = next_samples.mean(dim=1)

        total_es = total_es + es_t
        total_nll = total_nll + nll_t
        total_mae = total_mae + (mean_next - target_t).abs().mean()
        total_v_std = total_v_std + v_samples.std(dim=1).mean()
        total_delta_std = total_delta_std + (next_samples - prev.unsqueeze(1)).std(dim=1).mean()
        pred_mean_path.append(mean_next.mean(dim=-1))
        target_mean_path.append(target_t.mean(dim=-1))

        step_feat, next_local_scale = model._step_features(prev, target_t, local_scale)
        next_state_stack = model.recurrent_step(step_feat, state_stack)
        regime_state = model.update_regime_state(regime_state, next_state_stack)
        total_regime_drift = total_regime_drift + (regime_state - prev_regime_state).pow(2).mean()

        local_scale = next_local_scale
        state_stack = next_state_stack
        prev = target_t

    path_mean_loss = F.smooth_l1_loss(torch.stack(pred_mean_path, dim=1), torch.stack(target_mean_path, dim=1))
    total_loss = (
        (total_es / future_len)
        + nll_weight * (total_nll / future_len)
        + 0.05 * path_mean_loss
        + regime_slow_weight * (total_regime_drift / future_len)
    )
    metrics = {
        "total_loss": total_loss.detach(),
        "multistep_energy": (total_es / future_len).detach(),
        "multistep_nll": (total_nll / future_len).detach(),
        "multistep_mae": (total_mae / future_len).detach(),
        "sample_v_std": (total_v_std / future_len).detach(),
        "sample_delta_std": (total_delta_std / future_len).detach(),
        "path_mean_loss": path_mean_loss.detach(),
        "regime_drift": (total_regime_drift / future_len).detach(),
        "regime_norm": regime_state.norm(dim=-1).mean().detach(),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: SlowRegimeFlowTransitionModel,
    val_loader: DataLoader,
    train_samples: int,
    nll_weight: float,
    regime_slow_weight: float,
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
            regime_slow_weight=regime_slow_weight,
        )
        bs = history_01.shape[0]
        total_count += bs
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * bs
    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SlowRegimeFlowTransitionModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "slow_regime_flow_local_scale_asinh_220g":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = SlowRegimeFlowTransitionModel(
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
        regime_dim=cfg.get("regime_dim", 16),
        regime_alpha=cfg.get("regime_alpha", 0.08),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="220g slow-regime flow transition model")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=24)
    parser.add_argument("--rollout_val_samples", type=int, default=20)
    parser.add_argument("--rollout_eval_limit", type=int, default=160)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--regime_dim", type=int, default=16)
    parser.add_argument("--regime_alpha", type=float, default=0.08)
    parser.add_argument("--regime_slow_weight", type=float, default=0.02)
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
    model = SlowRegimeFlowTransitionModel(
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
        regime_dim=args.regime_dim,
        regime_alpha=args.regime_alpha,
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

    print("220g slow-regime flow transition model")
    print(f"  init={args.init_checkpoint}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(
        f"  future_len={args.future_len} train_samples={args.train_samples} "
        f"regime_dim={args.regime_dim} regime_alpha={args.regime_alpha}"
    )

    best_score = float("inf")
    history: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
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
                regime_slow_weight=args.regime_slow_weight,
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
            regime_slow_weight=args.regime_slow_weight,
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
                "type": "slow_regime_flow_local_scale_asinh_220g",
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
                "regime_dim": args.regime_dim,
                "regime_alpha": args.regime_alpha,
                "regime_slow_weight": args.regime_slow_weight,
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
