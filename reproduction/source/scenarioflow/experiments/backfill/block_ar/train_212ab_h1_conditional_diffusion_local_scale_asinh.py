#!/usr/bin/env python
"""
212ab: H=1 conditional diffusion model on local-scale asinh innovation space.

This keeps the proven 212y preprocessing:

    v_t(c) = asinh(delta_t(c) / s_t(c))

where `s_t(c)` is a causal EWMA absolute-delta scale computed from the history.

The change is the conditional law:
  - 212y/212z/212aa: one-shot MLP([h, z]) trained by energy score
  - 212ab: conditional DDPM / DDIM sampler over the 25D innovation vector `v`

This is the Bitter-Lesson move for the remaining hard cells: keep the same
encoder and scale geometry, but replace the hand-limited one-shot cloud with a
generic conditional density learner.
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
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        angles = t[:, None].float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ConditionalInnovationDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
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

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_proj(self.time_emb(t))
        c_emb = self.cond_proj(condition)
        h = torch.cat([x_noisy, t_emb, c_emb], dim=-1)
        return self.net(h)


class ConditionalDiffusionLocalScaleAsinhModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 75,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        denoiser_hidden: int = 256,
        time_emb_dim: int = 64,
        n_diffusion_steps: int = 64,
        ddim_steps: int = 16,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        ewma_alpha: float = 0.20,
        scale_floor: float = 1e-4,
        include_scale_feature: bool = True,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = n_cells  # used by paired-noise diagnostics as initial x_T
        self.hidden_dim = hidden_dim
        self.ewma_alpha = float(ewma_alpha)
        self.scale_floor = float(scale_floor)
        self.include_scale_feature = bool(include_scale_feature)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.ddim_steps = int(ddim_steps)

        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.denoiser = ConditionalInnovationDenoiser(
            input_dim=n_cells,
            cond_dim=hidden_dim,
            hidden_dim=denoiser_hidden,
            time_emb_dim=time_emb_dim,
        )

        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

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
        sample_steps: int | None = None,
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

        cond = state.unsqueeze(1).expand(batch, n_samples, self.hidden_dim).reshape(batch * n_samples, self.hidden_dim)
        x = x.reshape(batch * n_samples, self.n_cells)
        num_steps = int(sample_steps or self.ddim_steps)
        num_steps = min(max(num_steps, 1), self.n_diffusion_steps)
        step_indices = torch.linspace(0, self.n_diffusion_steps - 1, num_steps + 1, device=history_01.device).long()
        timesteps = step_indices.flip(0)[:-1]

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch * n_samples,), int(t_val.item()), device=history_01.device, dtype=torch.long)
            noise_pred = self.denoiser(x, t, cond)
            alpha_t = self.alphas_cumprod[t_val]
            if i < len(timesteps) - 1:
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=history_01.device, dtype=history_01.dtype)
            x0_pred = (x - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * noise_pred

        v = x.reshape(batch, n_samples, self.n_cells)
        return v, local_scale, prev

    def sample_delta(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        v, local_scale, prev = self.sample_transformed_innovation(
            history_01, n_samples=n_samples, noise=noise, sample_steps=sample_steps
        )
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
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        return (prev + self.sample_delta(history_01, n_samples=n_samples, noise=noise, sample_steps=sample_steps)).clamp(0.0, 1.0)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        state, local_scale = self.encode_with_scale(history_01)
        target_v = torch.asinh(target_delta / local_scale.clamp_min(self.scale_floor))

        batch = target_v.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch,), device=history_01.device)
        noise = torch.randn_like(target_v)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_noisy = sqrt_alpha * target_v + sqrt_one_minus * noise
        noise_pred = self.denoiser(x_noisy, t, state)
        loss = F.mse_loss(noise_pred, noise)

        metrics = {
            "denoise_mse": loss.detach(),
            "target_v_abs_mean": target_v.abs().mean().detach(),
            "local_scale_mean": local_scale.mean().detach(),
        }
        return loss, metrics


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ConditionalDiffusionLocalScaleAsinhModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_diffusion_212ab_local_scale_asinh":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = ConditionalDiffusionLocalScaleAsinhModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        denoiser_hidden=cfg["denoiser_hidden"],
        time_emb_dim=cfg["time_emb_dim"],
        n_diffusion_steps=cfg["n_diffusion_steps"],
        ddim_steps=cfg["ddim_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212ab H=1 conditional diffusion on local-scale asinh innovation")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--denoiser_hidden", type=int, default=256)
    parser.add_argument("--time_emb_dim", type=int, default=64)
    parser.add_argument("--n_diffusion_steps", type=int, default=64)
    parser.add_argument("--ddim_steps", type=int, default=16)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--eval_samples", type=int, default=64)
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
    model = ConditionalDiffusionLocalScaleAsinhModel(
        history_feat_dim=history_feat_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        denoiser_hidden=args.denoiser_hidden,
        time_emb_dim=args.time_emb_dim,
        n_diffusion_steps=args.n_diffusion_steps,
        ddim_steps=args.ddim_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=args.include_scale_feature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212ab H=1 conditional diffusion on local-scale asinh innovation")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  ewma_alpha={args.ewma_alpha:.3f} scale_floor={args.scale_floor:.1e}")
    print(f"  diffusion_steps={args.n_diffusion_steps} ddim_steps={args.ddim_steps}")
    print(f"  include_scale_feature={args.include_scale_feature} history_feat_dim={history_feat_dim}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "denoise_mse": 0.0, "target_v_abs_mean": 0.0, "local_scale_mean": 0.0}
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(history_01, target_01)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["denoise_mse"] += float(metrics["denoise_mse"].item()) * batch
            running["target_v_abs_mean"] += float(metrics["target_v_abs_mean"].item()) * batch
            running["local_scale_mean"] += float(metrics["local_scale_mean"].item()) * batch
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
                "type": "minimal_h1_conditional_diffusion_212ab_local_scale_asinh",
                "n_cells": 25,
                "history_feat_dim": history_feat_dim,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "denoiser_hidden": args.denoiser_hidden,
                "time_emb_dim": args.time_emb_dim,
                "history_len": args.history_len,
                "n_diffusion_steps": args.n_diffusion_steps,
                "ddim_steps": args.ddim_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
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
