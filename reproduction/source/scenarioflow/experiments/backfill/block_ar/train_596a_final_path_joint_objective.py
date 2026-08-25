#!/usr/bin/env python
"""596a: final-path joint-distribution objective finetune from the 510a AR frontier."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_353a_340c_full_rollout_energy_finetune import (  # noqa: E402
    sample_rollout_scores_with_grad,
)
from experiments.backfill.block_ar.train_391a_recent_rollout_energy_finetune import (  # noqa: E402
    build_recent_block,
)


def horizon_path_weights(
    horizon: int,
    *,
    end_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic normalized horizon weights for final-path losses."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    end = max(1e-6, float(end_weight))
    if horizon == 1:
        weights = torch.ones(1, device=device, dtype=dtype)
    else:
        weights = torch.linspace(1.0, end, horizon, device=device, dtype=dtype)
    return weights / weights.mean().clamp_min(1e-12)


def _weighted_flatten(
    paths: torch.Tensor,
    horizon_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if paths.ndim not in {3, 4}:
        raise ValueError("paths must have shape [B,T,C] or [B,K,T,C]")
    if horizon_weights is None:
        weighted = paths
        weights = torch.ones(paths.shape[-2], device=paths.device, dtype=paths.dtype)
    else:
        if horizon_weights.shape != (paths.shape[-2],):
            raise ValueError(
                f"horizon_weights must have shape ({paths.shape[-2]},), "
                f"got {tuple(horizon_weights.shape)}"
            )
        shape = (1,) * (paths.ndim - 2) + (paths.shape[-2], 1)
        weights = horizon_weights.to(device=paths.device, dtype=paths.dtype)
        weighted = paths * weights.view(shape)
    return weighted.reshape(*paths.shape[:-2], paths.shape[-2] * paths.shape[-1]), weights


def full_path_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score over the full future path in empirical normal-score coordinates."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    samples_flat, weights = _weighted_flatten(samples, horizon_weights)
    target_flat, _ = _weighted_flatten(target, horizon_weights)
    scale = torch.sqrt(weights.square().sum() * float(samples.shape[-1])).clamp_min(1e-12)
    target_dist = torch.sqrt(
        (samples_flat - target_flat[:, None, :]).pow(2).sum(dim=-1) + float(eps)
    ).mean(dim=1) / scale
    pair_dist = torch.cdist(samples_flat, samples_flat, p=2).mean(dim=(1, 2)) / scale
    score = target_dist - 0.5 * pair_dist
    return score.mean(), target_dist.mean(), pair_dist.mean()


def sliced_projection_wasserstein_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    n_projections: int,
    n_quantiles: int,
    horizon_weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batch-level sliced Wasserstein discrepancy between generated and realized paths."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target path dimensions do not match")
    if n_projections < 1:
        raise ValueError("n_projections must be positive")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    samples_flat, _weights = _weighted_flatten(samples, horizon_weights)
    target_flat, _ = _weighted_flatten(target, horizon_weights)
    gen = samples_flat.reshape(samples_flat.shape[0] * samples_flat.shape[1], -1)
    tgt = target_flat.reshape(target_flat.shape[0], -1)
    projections = torch.randn(
        int(n_projections),
        gen.shape[-1],
        device=gen.device,
        dtype=gen.dtype,
        generator=generator,
    )
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gen_proj = gen @ projections.T
    tgt_proj = tgt @ projections.T
    qs = torch.linspace(0.0, 1.0, int(n_quantiles), device=gen.device, dtype=gen.dtype)
    gen_q = torch.quantile(gen_proj, qs, dim=0)
    tgt_q = torch.quantile(tgt_proj, qs, dim=0)
    return torch.sqrt((gen_q - tgt_q).pow(2).mean() + 1e-12)


def joint_path_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    joint_weight: float,
    sw_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    n_projections: int,
    n_quantiles: int,
    energy_eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    target_scores = model.target_future_scores(future_norm)
    sampled_scores = sample_rollout_scores_with_grad(
        model=model,
        history_norm=history_norm,
        n_samples=train_sample_count,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    weights = horizon_path_weights(
        target_scores.shape[1],
        end_weight=horizon_end_weight,
        device=target_scores.device,
        dtype=target_scores.dtype,
    )
    energy, target_dist, pair_dist = full_path_energy_score(
        sampled_scores,
        target_scores,
        eps=energy_eps,
        horizon_weights=weights,
    )
    sw = sliced_projection_wasserstein_score(
        sampled_scores,
        target_scores,
        n_projections=n_projections,
        n_quantiles=n_quantiles,
        horizon_weights=weights,
    )
    joint = energy + float(sw_weight) * sw
    total = float(fm_anchor_weight) * fm_loss + float(joint_weight) * joint
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "sw": sw.detach(),
        "joint": joint.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
        "sample_h1_std": sampled_scores[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_scores[:, :, -1].std(unbiased=False).detach(),
        "target_h1_std": target_scores[:, 0].std(unbiased=False).detach(),
        "target_h30_std": target_scores[:, -1].std(unbiased=False).detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--joint_weight", type=float, default=0.05)
    parser.add_argument("--sw_weight", type=float, default=1.0)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=2.0)
    parser.add_argument("--n_projections", type=int, default=32)
    parser.add_argument("--n_quantiles", type=int, default=16)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=596)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")
    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_val = max(1, int(round(hist_01.shape[0] * float(args.holdout_frac))))
    train_hist, val_hist = hist_01[:-n_val], hist_01[-n_val:]
    train_fut, val_fut = fut_01[:-n_val], fut_01[-n_val:]
    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_fut),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    def run_epoch(loader: DataLoader, train_mode: bool, max_batches: int) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_batch, fut_batch in loader:
            if max_batches > 0 and n_batches >= max_batches:
                break
            hist_norm = normalize_iv(hist_batch.to(device, non_blocking=True)).view(
                hist_batch.shape[0],
                hist_batch.shape[1],
                -1,
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0],
                fut_batch.shape[1],
                -1,
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = joint_path_loss(
                    model,
                    hist_norm,
                    fut_norm,
                    train_sample_count=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    joint_weight=args.joint_weight,
                    sw_weight=args.sw_weight,
                    fm_anchor_weight=args.fm_anchor_weight,
                    horizon_end_weight=args.horizon_end_weight,
                    n_projections=args.n_projections,
                    n_quantiles=args.n_quantiles,
                    energy_eps=args.energy_eps,
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')} best_val: {payload.get('best_val')}")
    print(f"Recent windows: {hist_01.shape[0]} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {train_hist.shape[0]}/{val_hist.shape[0]}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, max_batches=args.max_train_batches)
        val_avg = run_epoch(val_loader, train_mode=False, max_batches=args.max_val_batches)
        scheduler.step()
        rec = {
            "epoch": int(epoch),
            **{f"train_{key}": value for key, value in train_avg.items()},
            **{f"val_{key}": value for key, value in val_avg.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        records.append(rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = int(epoch)
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)
    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, int(args.epochs), best_val)
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(train_hist.shape[0]),
        "n_val": int(val_hist.shape[0]),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "objective": {
            "loss": "final_path_energy_plus_sliced_projection_wasserstein",
            "joint_weight": float(args.joint_weight),
            "sw_weight": float(args.sw_weight),
            "fm_anchor_weight": float(args.fm_anchor_weight),
            "horizon_end_weight": float(args.horizon_end_weight),
            "n_projections": int(args.n_projections),
            "n_quantiles": int(args.n_quantiles),
            "train_sample_count": int(args.train_sample_count),
            "rollout_flow_steps": int(args.rollout_flow_steps),
        },
        "records": records,
    }
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
