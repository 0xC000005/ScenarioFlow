#!/usr/bin/env python
"""
212ae: H=1 direct population-trained conditional flow decoder on local-scale
asinh innovation space.

This keeps the 212y local-scale transformed innovation setup but replaces the
one-shot decoder with a conditional affine coupling flow. Training remains on a
sample-population loss (energy score), so this tests whether the main issue was
the limited one-shot Gaussian->MLP law rather than the population objective.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.view(1, 1, dim))
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("2") or name.endswith("4") or name == "net.4":
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out, _logdet = self.forward_with_logdet(x, cond)
        return out

    def forward_with_logdet(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = x * self.mask
        h = torch.cat([masked, cond], dim=-1)
        log_s, t = self.net(h).chunk(2, dim=-1)
        scale = torch.exp(0.5 * torch.tanh(log_s))
        out = masked + (1.0 - self.mask) * (x * scale + t)
        logdet = ((1.0 - self.mask) * torch.log(scale)).sum(dim=-1)
        return out, logdet

    def inverse_with_logdet(self, y: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = y * self.mask
        h = torch.cat([masked, cond], dim=-1)
        log_s, t = self.net(h).chunk(2, dim=-1)
        scale = torch.exp(0.5 * torch.tanh(log_s))
        x = masked + (1.0 - self.mask) * ((y - t) / scale)
        logdet = -((1.0 - self.mask) * torch.log(scale)).sum(dim=-1)
        return x, logdet


class ConditionalFlowLocalScaleAsinhModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 75,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        flow_hidden: int = 256,
        n_coupling_layers: int = 6,
        ewma_alpha: float = 0.20,
        scale_floor: float = 1e-4,
        include_scale_feature: bool = True,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = n_cells
        self.hidden_dim = hidden_dim
        self.ewma_alpha = float(ewma_alpha)
        self.scale_floor = float(scale_floor)
        self.include_scale_feature = bool(include_scale_feature)
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        masks = []
        base = torch.tensor([(i % 2) for i in range(n_cells)], dtype=torch.float32)
        for i in range(n_coupling_layers):
            masks.append(base if i % 2 == 0 else 1.0 - base)
        self.layers = nn.ModuleList(
            [AffineCoupling(n_cells, hidden_dim, flow_hidden, mask) for mask in masks]
        )

    def _expand_condition(self, state: torch.Tensor, n_samples: int) -> torch.Tensor:
        return state.unsqueeze(1).expand(state.shape[0], n_samples, self.hidden_dim)

    def _flow_forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            x, layer_logdet = layer.forward_with_logdet(x, cond)
            logdet = logdet + layer_logdet
        return x, logdet

    def _flow_inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, layer_logdet = layer.inverse_with_logdet(z, cond)
            logdet = logdet + layer_logdet
        return z, logdet

    def encode_with_scale(self, history_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat, local_scale = build_local_scale_history_features(
            history_01=history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        return h_n[-1], local_scale

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        state, _ = self.encode_with_scale(history_01)
        return state

    def sample_transformed_innovation(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, local_scale = self.encode_with_scale(history_01)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)
        if noise is None:
            x = torch.randn(batch, n_samples, self.n_cells, device=history_01.device, dtype=history_01.dtype)
        else:
            if noise.shape[-1] != self.n_cells:
                raise ValueError(f"Expected noise last dim {self.n_cells}, got {noise.shape[-1]}")
            x = noise.to(device=history_01.device, dtype=history_01.dtype)
        cond = self._expand_condition(state, n_samples)
        x, _logdet = self._flow_forward(x, cond)
        return x, local_scale, prev

    def log_prob_transformed_innovation(
        self,
        history_01: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        state, _local_scale = self.encode_with_scale(history_01)
        cond = self._expand_condition(state, 1)
        y = target_v.unsqueeze(1)
        z, logdet_inv = self._flow_inverse(y, cond)
        log_base = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        return (log_base + logdet_inv).squeeze(1)

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v, local_scale, prev = self.sample_transformed_innovation(history_01, n_samples=n_samples, noise=noise)
        innovation = torch.sinh(v)
        delta = innovation * local_scale.unsqueeze(1)
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        return torch.maximum(torch.minimum(delta, upper), lower)

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        return (prev + self.sample_delta(history_01, n_samples=n_samples, noise=noise)).clamp(0.0, 1.0)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        v_samples, local_scale, _ = self.sample_transformed_innovation(history_01, n_samples=n_samples)
        target_v = torch.asinh(target_delta / local_scale.clamp_min(self.scale_floor))
        loss = energy_score(v_samples, target_v)
        sample_delta = torch.sinh(v_samples) * local_scale.unsqueeze(1)
        metrics = {
            "energy": loss.detach(),
            "sample_v_std": v_samples.std(dim=1).mean().detach(),
            "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ConditionalFlowLocalScaleAsinhModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_flow_212ae_local_scale_asinh":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = ConditionalFlowLocalScaleAsinhModel(
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
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212ae H=1 conditional flow on local-scale asinh innovation")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--flow_hidden", type=int, default=256)
    parser.add_argument("--n_coupling_layers", type=int, default=6)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--include_scale_feature", action="store_true", default=True)
    parser.add_argument("--no_scale_feature", action="store_false", dest="include_scale_feature")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices = val_indices[: args.max_val_windows]

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    history_feat_dim = 25 + 25 + (25 if args.include_scale_feature else 0)
    model = ConditionalFlowLocalScaleAsinhModel(
        history_feat_dim=history_feat_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        flow_hidden=args.flow_hidden,
        n_coupling_layers=args.n_coupling_layers,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=args.include_scale_feature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212ae H=1 conditional flow on local-scale asinh innovation")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  n_coupling_layers={args.n_coupling_layers} train_samples={args.train_samples}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "energy": 0.0, "sample_v_std": 0.0, "sample_delta_std": 0.0}
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(history_01, target_01, n_samples=args.train_samples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["sample_v_std"] += float(metrics["sample_v_std"].item()) * batch
            running["sample_delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_212ae_local_scale_asinh",
                "n_cells": 25,
                "history_feat_dim": history_feat_dim,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "flow_hidden": args.flow_hidden,
                "n_coupling_layers": args.n_coupling_layers,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": args.ewma_alpha,
                "scale_floor": args.scale_floor,
                "include_scale_feature": args.include_scale_feature,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"valES={val_metrics['val_energy']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
