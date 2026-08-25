#!/usr/bin/env python
"""
212v: H=1 minimal direct stochastic delta with tail-incidence head and
imbalance-aware window sampling.

Base:
  - plain 212b direct stochastic delta decoder
  - pure energy score remains the main distribution loss

Add:
  - encoder q95 / q99 tail-incidence heads
  - predicted tail probabilities fed back into decoder input
  - rarity-aware window sampling over ordinary / stress / extreme windows
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
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_window_tail_labels(
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prev = history_01[:, -1].reshape(history_01.shape[0], -1)
    target_delta = target_01 - prev
    max_abs = target_delta.abs().amax(dim=1)
    y95 = (max_abs >= q95_threshold).float()
    y99 = (max_abs >= q99_threshold).float()
    return max_abs, y95, y99


def assign_tail_buckets(
    max_abs: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
) -> torch.Tensor:
    ordinary = max_abs < q95_threshold
    extreme = max_abs >= q99_threshold
    stress = (~ordinary) & (~extreme)
    buckets = torch.zeros_like(max_abs, dtype=torch.long)
    buckets[stress] = 1
    buckets[extreme] = 2
    return buckets


class MinimalDirectStochasticDeltaTailIncidenceModel(nn.Module):
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
        self.tail_head = nn.Linear(hidden_dim, 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 2 + noise_dim, decoder_hidden),
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
                if name == "decoder.4":
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

    def tail_logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.tail_head(state)

    def predict_tail_probs(self, history_01: torch.Tensor) -> torch.Tensor:
        state = self.encode(history_01)
        return torch.sigmoid(self.tail_logits_from_state(state))

    def sample_delta_from_state(
        self,
        history_01: torch.Tensor,
        state: torch.Tensor,
        tail_logits: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)
        tail_probs = torch.sigmoid(tail_logits)

        state_rep = state.unsqueeze(1).expand(batch, n_samples, self.hidden_dim)
        tail_rep = tail_probs.unsqueeze(1).expand(batch, n_samples, 2)
        if noise is None:
            noise = torch.randn(batch, n_samples, self.noise_dim, device=history_01.device, dtype=history_01.dtype)
        decoder_in = torch.cat([state_rep, tail_rep, noise], dim=-1).reshape(
            batch * n_samples, self.hidden_dim + 2 + self.noise_dim
        )
        raw = self.decoder(decoder_in).reshape(batch, n_samples, self.n_cells)
        delta = torch.tanh(raw) * self.delta_scale.view(1, 1, self.n_cells)
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        return torch.maximum(torch.minimum(delta, upper), lower)

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.encode(history_01)
        tail_logits = self.tail_logits_from_state(state)
        return self.sample_delta_from_state(history_01, state, tail_logits, n_samples=n_samples, noise=noise)

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
        y95: torch.Tensor,
        y99: torch.Tensor,
        n_samples: int,
        lambda_q95: float,
        lambda_q99: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        state = self.encode(history_01)
        tail_logits = self.tail_logits_from_state(state)
        samples = self.sample_delta_from_state(history_01, state, tail_logits, n_samples=n_samples)

        es = energy_score(samples, target_delta)
        bce95 = F.binary_cross_entropy_with_logits(tail_logits[:, 0], y95)
        bce99 = F.binary_cross_entropy_with_logits(tail_logits[:, 1], y99)
        loss = es + lambda_q95 * bce95 + lambda_q99 * bce99

        probs = torch.sigmoid(tail_logits)
        mean_delta = samples.mean(dim=1)
        metrics = {
            "energy": es.detach(),
            "bce95": bce95.detach(),
            "bce99": bce99.detach(),
            "mean_abs_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
            "sample_delta_std": samples.std(dim=1).mean().detach(),
            "avg_p95": probs[:, 0].mean().detach(),
            "avg_p99": probs[:, 1].mean().detach(),
        }
        return loss, metrics


@torch.no_grad()
def evaluate_h1(
    model: MinimalDirectStochasticDeltaTailIncidenceModel,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    total = {
        "val_energy": 0.0,
        "val_mae": 0.0,
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
    y95_all = []
    y99_all = []
    p95_all = []
    p99_all = []
    mean_width_all = []
    realized_max_all = []

    for history_01, target_01, y95, y99 in loader:
        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_delta = target_01 - prev
        sample_delta = model.sample_delta(history_01, n_samples=eval_samples)
        sample_iv = (prev.unsqueeze(1) + sample_delta).clamp(0.0, 1.0)
        q05 = sample_iv.quantile(0.05, dim=1)
        q95_iv = sample_iv.quantile(0.95, dim=1)
        mean_pred = sample_iv.mean(dim=1)
        coverage = ((target_01 >= q05) & (target_01 <= q95_iv)).float().mean()
        width = (q95_iv - q05).mean()

        target_abs = target_delta.abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold
        q95_cov = (
            ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95_iv[q95_mask])).float().mean()
            if q95_mask.any()
            else target_01.new_tensor(0.0)
        )
        q99_cov = (
            ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95_iv[q99_mask])).float().mean()
            if q99_mask.any()
            else target_01.new_tensor(0.0)
        )

        tail_probs = model.predict_tail_probs(history_01)
        window_width = (q95_iv - q05).mean(dim=1)
        realized_max_abs = target_delta.abs().amax(dim=1)

        batch_size = history_01.shape[0]
        total["val_energy"] += float(energy_score(sample_delta, target_delta).item()) * batch_size
        total["val_mae"] += float((mean_pred - target_01).abs().mean().item()) * batch_size
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
        y95_all.append(y95.detach().cpu().numpy())
        y99_all.append(y99.detach().cpu().numpy())
        p95_all.append(tail_probs[:, 0].detach().cpu().numpy())
        p99_all.append(tail_probs[:, 1].detach().cpu().numpy())
        mean_width_all.append(window_width.detach().cpu().numpy())
        realized_max_all.append(realized_max_abs.detach().cpu().numpy())

    out = {k: v / max(total_count, 1) for k, v in total.items()}
    out["val_realized_q95_coverage_90"] = q95_cover_sum / max(q95_count, 1)
    out["val_realized_q99_coverage_90"] = q99_cover_sum / max(q99_count, 1)
    out["val_q95_cell_count"] = q95_count
    out["val_q99_cell_count"] = q99_count

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    out.update({f"val_h1_{k}": v for k, v in shape.items()})

    y95_np = np.concatenate(y95_all, axis=0)
    y99_np = np.concatenate(y99_all, axis=0)
    p95_np = np.concatenate(p95_all, axis=0)
    p99_np = np.concatenate(p99_all, axis=0)
    width_np = np.concatenate(mean_width_all, axis=0)
    realized_max_np = np.concatenate(realized_max_all, axis=0)

    out["val_tail_q95_auc"] = binary_auc(y95_np, p95_np)
    out["val_tail_q99_auc"] = binary_auc(y99_np, p99_np)
    out["val_tail_q95_brier"] = float(np.mean((p95_np - y95_np) ** 2))
    out["val_tail_q99_brier"] = float(np.mean((p99_np - y99_np) ** 2))
    out["val_tail_q95_rate"] = float(y95_np.mean())
    out["val_tail_q99_rate"] = float(y99_np.mean())
    out["val_tail_width_p95_corr"] = safe_corr(width_np, p95_np)
    out["val_tail_width_p99_corr"] = safe_corr(width_np, p99_np)
    out["val_tail_realizedmax_p95_corr"] = safe_corr(realized_max_np, p95_np)
    out["val_tail_realizedmax_p99_corr"] = safe_corr(realized_max_np, p99_np)
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MinimalDirectStochasticDeltaTailIncidenceModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_direct_stochastic_delta_212v_tail_incidence_sampling":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = MinimalDirectStochasticDeltaTailIncidenceModel(
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
    parser = argparse.ArgumentParser(description="212v H=1 minimal direct stochastic delta with tail-incidence sampling")
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
    parser.add_argument("--lambda_q95", type=float, default=0.05)
    parser.add_argument("--lambda_q99", type=float, default=0.10)
    parser.add_argument("--ordinary_weight", type=float, default=1.0)
    parser.add_argument("--stress_weight", type=float, default=3.0)
    parser.add_argument("--extreme_weight", type=float, default=6.0)
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

    train_prev = train_hist[:, -1].reshape(train_hist.shape[0], -1)
    train_target_delta = train_target - train_prev
    train_target_delta_np = train_target_delta.detach().cpu().numpy()
    q95_threshold = float(np.quantile(np.abs(train_target_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_target_delta_np.reshape(-1)), 0.99))

    train_max_abs, train_y95, train_y99 = compute_window_tail_labels(train_hist, train_target, q95_threshold, q99_threshold)
    val_max_abs, val_y95, val_y99 = compute_window_tail_labels(val_hist, val_target, q95_threshold, q99_threshold)

    train_buckets = assign_tail_buckets(train_max_abs, q95_threshold, q99_threshold)
    bucket_weights = torch.tensor(
        [args.ordinary_weight, args.stress_weight, args.extreme_weight],
        dtype=torch.double,
    )
    train_weights = bucket_weights[train_buckets.cpu()]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(train_hist, train_target, train_y95, train_y99),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target, val_y95, val_y99),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MinimalDirectStochasticDeltaTailIncidenceModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        noise_dim=args.noise_dim,
        decoder_hidden=args.decoder_hidden,
        delta_scale=torch.from_numpy(delta_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ordinary_count = int((train_buckets == 0).sum().item())
    stress_count = int((train_buckets == 1).sum().item())
    extreme_count = int((train_buckets == 2).sum().item())

    print("212v H=1 minimal direct stochastic delta + tail-incidence sampling")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(
        f"  bucket_counts ordinary={ordinary_count} stress={stress_count} extreme={extreme_count}"
    )
    print(
        f"  sampler_weights ordinary={args.ordinary_weight:.1f} stress={args.stress_weight:.1f} extreme={args.extreme_weight:.1f}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "energy": 0.0,
            "bce95": 0.0,
            "bce99": 0.0,
            "mae": 0.0,
            "delta_std": 0.0,
            "avg_p95": 0.0,
            "avg_p99": 0.0,
        }
        count = 0

        for history_01, target_01, y95, y99 in train_loader:
            loss, metrics = model.training_loss(
                history_01,
                target_01,
                y95,
                y99,
                n_samples=args.train_samples,
                lambda_q95=args.lambda_q95,
                lambda_q99=args.lambda_q99,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["bce95"] += float(metrics["bce95"].item()) * batch
            running["bce99"] += float(metrics["bce99"].item()) * batch
            running["mae"] += float(metrics["mean_abs_delta_mae"].item()) * batch
            running["delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["avg_p95"] += float(metrics["avg_p95"].item()) * batch
            running["avg_p99"] += float(metrics["avg_p99"].item()) * batch
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
                "type": "minimal_h1_direct_stochastic_delta_212v_tail_incidence_sampling",
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
                "lambda_q95": args.lambda_q95,
                "lambda_q99": args.lambda_q99,
                "ordinary_weight": args.ordinary_weight,
                "stress_weight": args.stress_weight,
                "extreme_weight": args.extreme_weight,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
            "bucket_counts": {
                "ordinary": ordinary_count,
                "stress": stress_count,
                "extreme": extreme_count,
            },
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
            f"q99AUC={val_metrics['val_tail_q99_auc']:.3f}  "
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
