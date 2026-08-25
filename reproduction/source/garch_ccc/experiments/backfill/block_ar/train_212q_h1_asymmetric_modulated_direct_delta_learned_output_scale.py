#!/usr/bin/env python
"""
212q: H=1 asymmetric modulated direct stochastic delta with learned output scale.

Minimal follow-up to 212e:
  - keep the 212e encoder, modulation, tanh noise pattern, and physical clipping
  - keep delta_scale only for input normalization
  - remove fixed output-side multiplication by per-cell train q99 delta_scale
  - replace it with learned per-cell output amplitude parameters
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

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    build_history_features,
    energy_score,
)
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import (
    evaluate_h1,
)


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x))


class LearnedOutputScaleAsymmetricModulatedDirectDeltaModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 50,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        noise_dim: int = 16,
        noise_hidden: int = 128,
        state_hidden: int = 128,
        delta_scale: torch.Tensor | None = None,
        center_scale_factor: float = 0.5,
        init_scale_factor: float = 0.15,
        modulation_scale: float = 0.5,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.noise_hidden = noise_hidden
        self.center_scale_factor = center_scale_factor
        self.init_scale_factor = init_scale_factor
        self.modulation_scale = modulation_scale

        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )

        self.noise_proj = nn.Linear(noise_dim, noise_hidden)
        self.noise_norm = nn.LayerNorm(noise_hidden)
        self.noise_post = nn.Sequential(
            nn.Linear(noise_hidden, noise_hidden),
            nn.SiLU(),
            nn.Linear(noise_hidden, n_cells),
        )

        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_dim, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, noise_hidden),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, noise_hidden),
        )
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, n_cells),
        )
        self.pos_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, n_cells),
        )
        self.neg_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, n_cells),
        )

        if delta_scale is None:
            delta_scale = torch.ones(n_cells, dtype=torch.float32) * 0.05
        self.register_buffer("delta_scale", delta_scale.float())

        center_init = torch.clamp(self.center_scale_factor * self.delta_scale, min=1e-4)
        scale_init = torch.clamp(self.init_scale_factor * self.delta_scale, min=1e-4)
        self.center_amp_raw = nn.Parameter(_softplus_inv(center_init.clone()))
        self.pos_base_raw = nn.Parameter(_softplus_inv(scale_init.clone()))
        self.neg_base_raw = nn.Parameter(_softplus_inv(scale_init.clone()))

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("noise_post.2"):
                    continue
                if (
                    name.endswith("gamma_head.2")
                    or name.endswith("beta_head.2")
                    or name.endswith("center_head.2")
                    or name.endswith("pos_scale_head.2")
                    or name.endswith("neg_scale_head.2")
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.noise_post[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.noise_post[-1].bias)

        for head in (self.gamma_head, self.beta_head, self.center_head, self.pos_scale_head, self.neg_scale_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        feat = build_history_features(history_01, self.delta_scale)
        _out, h_n = self.gru(feat)
        return h_n[-1]

    def decode_params_from_state(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        center_amp = F.softplus(self.center_amp_raw).view(1, self.n_cells)
        pos_base = F.softplus(self.pos_base_raw).view(1, self.n_cells)
        neg_base = F.softplus(self.neg_base_raw).view(1, self.n_cells)

        center = torch.tanh(self.center_head(state)) * center_amp
        pos_scale = F.softplus(self.pos_scale_head(state) + self.pos_base_raw.view(1, self.n_cells))
        neg_scale = F.softplus(self.neg_scale_head(state) + self.neg_base_raw.view(1, self.n_cells))
        gamma = self.modulation_scale * torch.tanh(self.gamma_head(state))
        beta = self.modulation_scale * torch.tanh(self.beta_head(state))
        return center, pos_scale, neg_scale, gamma, beta

    def decode_params(
        self,
        history_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encode(history_01)
        return self.decode_params_from_state(state)

    def noise_pattern(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_samples, _ = noise.shape
        gamma = self.modulation_scale * torch.tanh(self.gamma_head(state))
        beta = self.modulation_scale * torch.tanh(self.beta_head(state))

        u = self.noise_proj(noise.reshape(batch * n_samples, self.noise_dim))
        u = self.noise_norm(u).reshape(batch, n_samples, self.noise_hidden)
        u_mod = (1.0 + gamma.unsqueeze(1)) * u + beta.unsqueeze(1)
        raw_eps = self.noise_post(u_mod.reshape(batch * n_samples, self.noise_hidden)).reshape(batch, n_samples, self.n_cells)
        eps = torch.tanh(raw_eps)
        return eps, u_mod

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.encode(history_01)
        batch = history_01.shape[0]
        prev = history_01[:, -1].reshape(batch, self.n_cells)
        center, pos_scale, neg_scale, gamma, beta = self.decode_params_from_state(state)

        if noise is None:
            noise = torch.randn(batch, n_samples, self.noise_dim, device=history_01.device, dtype=history_01.dtype)
        eps, u_mod = self.noise_pattern(state, noise)

        pos_part = torch.relu(eps)
        neg_part = torch.relu(-eps)
        delta = center.unsqueeze(1) + pos_scale.unsqueeze(1) * pos_part - neg_scale.unsqueeze(1) * neg_part

        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        delta = torch.maximum(torch.minimum(delta, upper), lower)

        aux = {
            "state": state,
            "center": center,
            "pos_scale": pos_scale,
            "neg_scale": neg_scale,
            "gamma": gamma,
            "beta": beta,
            "eps": eps,
            "u_mod": u_mod,
        }
        return delta, aux

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        delta, _aux = self.sample_delta(history_01, n_samples=n_samples, noise=noise)
        return (prev + delta).clamp(0.0, 1.0)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        samples, aux = self.sample_delta(history_01, n_samples=n_samples)
        loss = energy_score(samples, target_delta)
        mean_delta = samples.mean(dim=1)
        metrics = {
            "energy": loss.detach(),
            "mean_abs_delta_mae": (mean_delta - target_delta).abs().mean().detach(),
            "sample_delta_std": samples.std(dim=1).mean().detach(),
            "center_abs_mean": aux["center"].abs().mean().detach(),
            "pos_scale_mean": aux["pos_scale"].mean().detach(),
            "neg_scale_mean": aux["neg_scale"].mean().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LearnedOutputScaleAsymmetricModulatedDirectDeltaModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_asymmetric_modulated_direct_delta_212q":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = LearnedOutputScaleAsymmetricModulatedDirectDeltaModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        noise_dim=cfg["noise_dim"],
        noise_hidden=cfg["noise_hidden"],
        state_hidden=cfg["state_hidden"],
        delta_scale=torch.tensor(cfg["delta_scale"], dtype=torch.float32),
        center_scale_factor=cfg["center_scale_factor"],
        init_scale_factor=cfg["init_scale_factor"],
        modulation_scale=cfg["modulation_scale"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212q H=1 asymmetric modulated direct stochastic delta with learned output scale")
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
    parser.add_argument("--noise_hidden", type=int, default=128)
    parser.add_argument("--state_hidden", type=int, default=128)
    parser.add_argument("--center_scale_factor", type=float, default=0.5)
    parser.add_argument("--init_scale_factor", type=float, default=0.15)
    parser.add_argument("--modulation_scale", type=float, default=0.5)
    parser.add_argument("--train_samples", type=int, default=8)
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

    model = LearnedOutputScaleAsymmetricModulatedDirectDeltaModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        noise_dim=args.noise_dim,
        noise_hidden=args.noise_hidden,
        state_hidden=args.state_hidden,
        delta_scale=torch.from_numpy(delta_scale),
        center_scale_factor=args.center_scale_factor,
        init_scale_factor=args.init_scale_factor,
        modulation_scale=args.modulation_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212q H=1 asymmetric modulated direct delta with learned output scale")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "mae": 0.0,
            "delta_std": 0.0,
            "center_abs_mean": 0.0,
            "pos_scale_mean": 0.0,
            "neg_scale_mean": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(history_01, target_01, n_samples=args.train_samples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["mae"] += float(metrics["mean_abs_delta_mae"].item()) * batch
            running["delta_std"] += float(metrics["sample_delta_std"].item()) * batch
            running["center_abs_mean"] += float(metrics["center_abs_mean"].item()) * batch
            running["pos_scale_mean"] += float(metrics["pos_scale_mean"].item()) * batch
            running["neg_scale_mean"] += float(metrics["neg_scale_mean"].item()) * batch
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
                "type": "minimal_h1_asymmetric_modulated_direct_delta_212q",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "noise_dim": args.noise_dim,
                "noise_hidden": args.noise_hidden,
                "state_hidden": args.state_hidden,
                "delta_scale": delta_scale.tolist(),
                "center_scale_factor": args.center_scale_factor,
                "init_scale_factor": args.init_scale_factor,
                "modulation_scale": args.modulation_scale,
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
            f"mae={train_metrics['train_mae']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"center={val_metrics['val_center_abs_mean']:.4f}  "
            f"ps={val_metrics['val_pos_scale_mean']:.4f}  "
            f"ns={val_metrics['val_neg_scale_mean']:.4f}  "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")


if __name__ == "__main__":
    main()
