#!/usr/bin/env python
"""
212u: H=1 minimal direct stochastic delta model with raw direct output.

Same as 212b except:
  - decoder outputs raw next-day delta directly
  - remove output tanh
  - remove output-side per-cell delta_scale multiplication
  - keep only final physical clipping to [-prev, 1 - prev]

Input normalization is unchanged.
Training loss is still energy score only.
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

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    compute_h1_shape_stats,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    build_history_features,
    energy_score,
    evaluate_h1,
)


class MinimalDirectStochasticDeltaRawOutputModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 50,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        noise_dim: int = 16,
        decoder_hidden: int = 256,
        delta_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, n_cells),
        )
        if delta_scale is None:
            delta_scale = torch.ones(n_cells, dtype=torch.float32) * 0.05
        self.register_buffer("delta_scale", delta_scale.float())
        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("2") or name.endswith("4") or name == "decoder.4":
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        feat = build_history_features(history_01, self.delta_scale)
        _out, h_n = self.gru(feat)
        return h_n[-1]

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.encode(history_01)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)

        state_rep = state.unsqueeze(1).expand(batch, n_samples, self.hidden_dim)
        if noise is None:
            noise = torch.randn(batch, n_samples, self.noise_dim, device=history_01.device, dtype=history_01.dtype)
        decoder_in = torch.cat([state_rep, noise], dim=-1).reshape(batch * n_samples, self.hidden_dim + self.noise_dim)
        raw_delta = self.decoder(decoder_in).reshape(batch, n_samples, self.n_cells)
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        return torch.maximum(torch.minimum(raw_delta, upper), lower)

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
        samples = self.sample_delta(history_01, n_samples=n_samples)
        loss = energy_score(samples, target_delta)
        mean_delta = samples.mean(dim=1)
        metrics = {
            "energy": loss.detach(),
            "mean_abs_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
            "sample_delta_std": samples.std(dim=1).mean().detach(),
        }
        return loss, metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MinimalDirectStochasticDeltaRawOutputModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_direct_stochastic_delta_212u_raw_output":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = MinimalDirectStochasticDeltaRawOutputModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        noise_dim=cfg["noise_dim"],
        decoder_hidden=cfg["decoder_hidden"],
        delta_scale=torch.tensor(cfg["delta_scale"], dtype=torch.float32),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212u H=1 minimal direct stochastic delta raw output")
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
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--decoder_hidden", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
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
    delta_scale = np.quantile(np.abs(train_delta_np), 0.99, axis=0).astype(np.float32)
    delta_scale = np.clip(delta_scale, 1e-3, None)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    model = MinimalDirectStochasticDeltaRawOutputModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        noise_dim=args.noise_dim,
        decoder_hidden=args.decoder_hidden,
        delta_scale=torch.from_numpy(delta_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212u H=1 minimal direct stochastic delta raw output")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  input delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "energy": 0.0, "mae": 0.0, "delta_std": 0.0}
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
            running["mae"] += float(metrics["mean_abs_delta_mae"].item()) * batch
            running["delta_std"] += float(metrics["sample_delta_std"].item()) * batch
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
                "type": "minimal_h1_direct_stochastic_delta_212u_raw_output",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "noise_dim": args.noise_dim,
                "decoder_hidden": args.decoder_hidden,
                "delta_scale": delta_scale.tolist(),
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
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
