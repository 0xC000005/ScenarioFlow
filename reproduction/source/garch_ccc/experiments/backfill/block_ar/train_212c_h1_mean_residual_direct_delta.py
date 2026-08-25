#!/usr/bin/env python
"""
212c: H=1 minimal mean-plus-residual direct stochastic delta model.

Smallest structured follow-up to 212b:
  - same history input = levels + day-over-day deltas
  - same target = next-day delta surface
  - encoder = small GRU
  - deterministic mean-delta head with normalized MSE
  - bounded stochastic residual head with energy score
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
)


class MinimalMeanResidualDirectDeltaModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 50,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        noise_dim: int = 16,
        mean_hidden: int = 128,
        residual_hidden: int = 256,
        delta_scale: torch.Tensor | None = None,
        residual_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.residual_scale_factor = residual_scale_factor
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, mean_hidden),
            nn.SiLU(),
            nn.Linear(mean_hidden, n_cells),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, residual_hidden),
            nn.SiLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.SiLU(),
            nn.Linear(residual_hidden, n_cells),
        )
        if delta_scale is None:
            delta_scale = torch.ones(n_cells, dtype=torch.float32) * 0.05
        self.register_buffer("delta_scale", delta_scale.float())
        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("mean_head.2") or name.endswith("residual_head.4"):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        feat = build_history_features(history_01, self.delta_scale)
        _out, h_n = self.gru(feat)
        return h_n[-1]

    def mean_delta(self, history_01: torch.Tensor) -> torch.Tensor:
        state = self.encode(history_01)
        return torch.tanh(self.mean_head(state)) * self.delta_scale.view(1, self.n_cells)

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode(history_01)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)

        mean_delta = torch.tanh(self.mean_head(state)) * self.delta_scale.view(1, self.n_cells)

        state_rep = state.unsqueeze(1).expand(batch, n_samples, self.hidden_dim)
        if noise is None:
            noise = torch.randn(batch, n_samples, self.noise_dim, device=history_01.device, dtype=history_01.dtype)
        decoder_in = torch.cat([state_rep, noise], dim=-1).reshape(batch * n_samples, self.hidden_dim + self.noise_dim)
        raw_resid = self.residual_head(decoder_in).reshape(batch, n_samples, self.n_cells)
        residual = torch.tanh(raw_resid) * (self.residual_scale_factor * self.delta_scale.view(1, 1, self.n_cells))

        delta = mean_delta.unsqueeze(1) + residual
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        delta = torch.maximum(torch.minimum(delta, upper), lower)
        return delta, mean_delta

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        delta, _mean_delta = self.sample_delta(history_01, n_samples=n_samples, noise=noise)
        return (prev + delta).clamp(0.0, 1.0)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
        mean_weight: float,
        residual_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        samples, mean_delta = self.sample_delta(history_01, n_samples=n_samples)

        target_norm = target_delta / self.delta_scale.view(1, self.n_cells).clamp_min(1e-6)
        mean_norm = mean_delta / self.delta_scale.view(1, self.n_cells).clamp_min(1e-6)
        mean_mse = ((mean_norm - target_norm) ** 2).mean()
        residual_es = energy_score(samples, target_delta)
        loss = mean_weight * mean_mse + residual_weight * residual_es

        metrics = {
            "mean_mse": mean_mse.detach(),
            "residual_energy": residual_es.detach(),
            "mean_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
            "sample_delta_std": samples.std(dim=1).mean().detach(),
        }
        return loss, metrics


@torch.no_grad()
def evaluate_h1(
    model: MinimalMeanResidualDirectDeltaModel,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    total = {
        "val_mean_mse": 0.0,
        "val_residual_energy": 0.0,
        "val_mean_mae": 0.0,
        "val_sample_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
        "val_sample_delta_std": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    total_count = 0
    gt_delta_all = []
    sample_delta_all = []

    for history_01, target_01 in loader:
        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_delta = target_01 - prev
        sample_delta, mean_delta = model.sample_delta(history_01, n_samples=eval_samples)
        sample_iv = (prev.unsqueeze(1) + sample_delta).clamp(0.0, 1.0)
        mean_iv = (prev + mean_delta).clamp(0.0, 1.0)
        q05 = sample_iv.quantile(0.05, dim=1)
        q95 = sample_iv.quantile(0.95, dim=1)
        sample_mean_pred = sample_iv.mean(dim=1)
        coverage = ((target_01 >= q05) & (target_01 <= q95)).float().mean()
        width = (q95 - q05).mean()

        target_abs = target_delta.abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold
        q95_cov = (
            ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean()
            if q95_mask.any()
            else target_01.new_tensor(0.0)
        )
        q99_cov = (
            ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean()
            if q99_mask.any()
            else target_01.new_tensor(0.0)
        )

        target_norm = target_delta / model.delta_scale.view(1, model.n_cells).clamp_min(1e-6)
        mean_norm = mean_delta / model.delta_scale.view(1, model.n_cells).clamp_min(1e-6)
        mean_mse = ((mean_norm - target_norm) ** 2).mean()

        batch_size = history_01.shape[0]
        total["val_mean_mse"] += float(mean_mse.item()) * batch_size
        total["val_residual_energy"] += float(energy_score(sample_delta, target_delta).item()) * batch_size
        total["val_mean_mae"] += float((mean_iv - target_01).abs().mean().item()) * batch_size
        total["val_sample_mae"] += float((sample_mean_pred - target_01).abs().mean().item()) * batch_size
        total["val_coverage_90"] += float(coverage.item()) * batch_size
        total["val_width_90"] += float(width.item()) * batch_size
        total["val_sample_delta_std"] += float(sample_delta.std(dim=1).mean().item()) * batch_size

        if q95_mask.any():
            q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())
        total_count += batch_size

        gt_delta_all.append(target_delta.detach().cpu().numpy())
        sample_delta_all.append(sample_delta.detach().cpu().numpy())

    out = {k: v / max(total_count, 1) for k, v in total.items()}
    out["val_realized_q95_coverage_90"] = q95_cover_sum / max(q95_count, 1)
    out["val_realized_q99_coverage_90"] = q99_cover_sum / max(q99_count, 1)
    out["val_q95_cell_count"] = q95_count
    out["val_q99_cell_count"] = q99_count

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    out.update({f"val_h1_{k}": v for k, v in shape.items()})
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MinimalMeanResidualDirectDeltaModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_mean_residual_direct_delta_212c":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = MinimalMeanResidualDirectDeltaModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        noise_dim=cfg["noise_dim"],
        mean_hidden=cfg["mean_hidden"],
        residual_hidden=cfg["residual_hidden"],
        delta_scale=torch.tensor(cfg["delta_scale"], dtype=torch.float32),
        residual_scale_factor=cfg["residual_scale_factor"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212c H=1 minimal mean-plus-residual direct delta model")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--mean_hidden", type=int, default=128)
    parser.add_argument("--residual_hidden", type=int, default=256)
    parser.add_argument("--residual_scale_factor", type=float, default=1.0)
    parser.add_argument("--train_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--mean_weight", type=float, default=1.0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
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

    model = MinimalMeanResidualDirectDeltaModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        noise_dim=args.noise_dim,
        mean_hidden=args.mean_hidden,
        residual_hidden=args.residual_hidden,
        delta_scale=torch.from_numpy(delta_scale),
        residual_scale_factor=args.residual_scale_factor,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212c H=1 minimal mean-plus-residual direct delta")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "mean_mse": 0.0, "residual_energy": 0.0, "mean_mae": 0.0, "delta_std": 0.0}
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(
                history_01,
                target_01,
                n_samples=args.train_samples,
                mean_weight=args.mean_weight,
                residual_weight=args.residual_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["mean_mse"] += float(metrics["mean_mse"].item()) * batch
            running["residual_energy"] += float(metrics["residual_energy"].item()) * batch
            running["mean_mae"] += float(metrics["mean_delta_mae"].item()) * batch
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
                "type": "minimal_h1_mean_residual_direct_delta_212c",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "noise_dim": args.noise_dim,
                "mean_hidden": args.mean_hidden,
                "residual_hidden": args.residual_hidden,
                "delta_scale": delta_scale.tolist(),
                "residual_scale_factor": args.residual_scale_factor,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "mean_weight": args.mean_weight,
                "residual_weight": args.residual_weight,
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
            f"mse={train_metrics['train_mean_mse']:.4f}  "
            f"es={train_metrics['train_residual_energy']:.4f}  "
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
