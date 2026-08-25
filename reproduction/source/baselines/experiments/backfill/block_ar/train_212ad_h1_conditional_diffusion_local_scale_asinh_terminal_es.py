#!/usr/bin/env python
"""
212ad: fine-tune 212ab diffusion on terminal population loss.

Start from a pretrained 212ab conditional diffusion model and fine-tune it on
the final sampled innovation population:

    Loss = ES(v_samples, target_v) + lambda_denoise * denoise_MSE

This directly addresses the diagnosis from 212ab:
- the generator is expressive enough
- but pure denoising training is not aligned with the final scenario
  distribution quality we care about
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from experiments.backfill.block_ar.train_212ab_h1_conditional_diffusion_local_scale_asinh import (
    ConditionalDiffusionLocalScaleAsinhModel,
    load_model as load_pretrained_model,
)


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ConditionalDiffusionLocalScaleAsinhModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_conditional_diffusion_212ad_local_scale_asinh_terminal_es":
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


def terminal_training_loss(
    model: ConditionalDiffusionLocalScaleAsinhModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    n_samples: int,
    sample_steps: int,
    lambda_denoise: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
    target_delta = target_01 - prev
    state, local_scale = model.encode_with_scale(history_01)
    target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))

    base_noise = torch.randn(
        history_01.shape[0], n_samples, model.n_cells,
        device=history_01.device, dtype=history_01.dtype,
    )
    v_samples, _, _ = model.sample_transformed_innovation(
        history_01, n_samples=n_samples, noise=base_noise, sample_steps=sample_steps
    )
    es = energy_score(v_samples, target_v)

    t = torch.randint(0, model.n_diffusion_steps, (history_01.shape[0],), device=history_01.device)
    noise = torch.randn_like(target_v)
    sqrt_alpha = model.sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus = model.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    x_noisy = sqrt_alpha * target_v + sqrt_one_minus * noise
    noise_pred = model.denoiser(x_noisy, t, state)
    denoise = torch.nn.functional.mse_loss(noise_pred, noise)

    loss = es + lambda_denoise * denoise
    sample_delta = torch.sinh(v_samples) * local_scale.unsqueeze(1)
    metrics = {
        "terminal_es": es.detach(),
        "denoise_mse": denoise.detach(),
        "sample_v_std": v_samples.std(dim=1).mean().detach(),
        "sample_delta_std": sample_delta.std(dim=1).mean().detach(),
    }
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="212ad fine-tune 212ab diffusion on terminal ES")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=16)
    parser.add_argument("--train_sample_steps", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--lambda_denoise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, init_payload = load_pretrained_model(args.init_checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    print("212ad fine-tune 212ab diffusion on terminal ES")
    print(f"  init_checkpoint={args.init_checkpoint}")
    print(f"  train_samples={args.train_samples} train_sample_steps={args.train_sample_steps} lambda_denoise={args.lambda_denoise}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}
    cfg0 = init_payload["config"]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "terminal_es": 0.0, "denoise_mse": 0.0, "sample_v_std": 0.0, "sample_delta_std": 0.0}
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = terminal_training_loss(
                model,
                history_01,
                target_01,
                n_samples=args.train_samples,
                sample_steps=args.train_sample_steps,
                lambda_denoise=args.lambda_denoise,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["terminal_es"] += float(metrics["terminal_es"].item()) * batch
            running["denoise_mse"] += float(metrics["denoise_mse"].item()) * batch
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
                "type": "minimal_h1_conditional_diffusion_212ad_local_scale_asinh_terminal_es",
                "n_cells": cfg0["n_cells"],
                "history_feat_dim": cfg0["history_feat_dim"],
                "hidden_dim": cfg0["hidden_dim"],
                "gru_layers": cfg0["gru_layers"],
                "gru_dropout": cfg0["gru_dropout"],
                "denoiser_hidden": cfg0["denoiser_hidden"],
                "time_emb_dim": cfg0["time_emb_dim"],
                "history_len": cfg0["history_len"],
                "n_diffusion_steps": cfg0["n_diffusion_steps"],
                "ddim_steps": cfg0["ddim_steps"],
                "beta_start": cfg0["beta_start"],
                "beta_end": cfg0["beta_end"],
                "eval_samples": args.eval_samples,
                "ewma_alpha": cfg0["ewma_alpha"],
                "scale_floor": cfg0["scale_floor"],
                "include_scale_feature": cfg0["include_scale_feature"],
                "init_checkpoint": args.init_checkpoint,
                "train_samples": args.train_samples,
                "train_sample_steps": args.train_sample_steps,
                "lambda_denoise": args.lambda_denoise,
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
            f"termES={train_metrics['train_terminal_es']:.4f}  "
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
