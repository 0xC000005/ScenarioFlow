#!/usr/bin/env python
"""
Exp 88: Learned per-cell regime-adaptive scale with differentiable coverage loss.

Trains a 27-param PerCellRegimeScale module:
  scale[r,c] = 1 + bias + (alpha_global + delta[r,c]) * (vov_ratio - 1)

using regime-stratified Interval Score loss on cached model samples.

Usage:
    # Phase 1: Cache samples from frozen model
    PYTHONPATH=. python experiments/backfill/block_ar/train_percell_scale.py cache \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --n_samples 50 --max_batches 404 \
        --cache_dir data/percell_scale_cache --device cuda

    # Phase 2: Train scale head
    PYTHONPATH=. python experiments/backfill/block_ar/train_percell_scale.py train \
        --cache_dir data/percell_scale_cache --n_iters 500 --lr 0.01 \
        --output models/backfill/percell_scale_head/scale_head.pt

    # Phase 3: Evaluate on cached data
    PYTHONPATH=. python experiments/backfill/block_ar/train_percell_scale.py evaluate \
        --cache_dir data/percell_scale_cache \
        --scale_head models/backfill/percell_scale_head/scale_head.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# =============================================================================
# Architecture
# =============================================================================

class PerCellRegimeScale(nn.Module):
    """Per-cell regime-adaptive scale: extends scalar alpha to per-cell alpha.

    scale[r,c] = 1 + bias + (alpha_global + delta[r,c]) * (vov_ratio - 1)

    27 parameters total:
      - alpha_global (1): global regime sensitivity
      - delta (5x5=25): per-cell corrections to alpha
      - bias (1): global scale offset from 1.0
    """

    def __init__(self, init_alpha: float = 0.5, delta_clamp: float = 0.3,
                 use_bias: bool = True):
        super().__init__()
        self.alpha_global = nn.Parameter(torch.tensor(init_alpha))
        self.delta = nn.Parameter(torch.zeros(5, 5))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.delta_clamp = delta_clamp
        self.use_bias = use_bias

    def forward(self, vov_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vov_ratio: (B, 1) = vol_of_vol / global_mean_vol
        Returns:
            scale: (B, 5, 5)
        """
        delta_clamped = self.delta.clamp(-self.delta_clamp, self.delta_clamp)
        alpha_percell = self.alpha_global + delta_clamped  # (5, 5)
        bias = self.bias if self.use_bias else 0.0
        scale = 1.0 + bias + alpha_percell * (vov_ratio.unsqueeze(-1) - 1.0)
        return scale.clamp(0.5, 2.0)


# =============================================================================
# Loss
# =============================================================================

def interval_score(lower, upper, gt, alpha=0.1):
    """Interval Score for alpha-level CI.

    IS = (U - L) + (2/alpha) * max(0, L - y) + (2/alpha) * max(0, y - U)
    """
    width = upper - lower
    penalty_below = (2.0 / alpha) * torch.relu(lower - gt)
    penalty_above = (2.0 / alpha) * torch.relu(gt - upper)
    return width + penalty_below + penalty_above


def apply_percell_scale(samples, scale):
    """Apply per-cell scale to samples around ensemble mean.

    Args:
        samples: (N, S, T, 5, 5)
        scale: (N, 5, 5)
    Returns:
        corrected: (N, S, T, 5, 5)
    """
    mean = samples.mean(dim=1, keepdim=True)  # (N, 1, T, 5, 5)
    scale_bc = scale.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 5, 5)
    return (mean + scale_bc * (samples - mean)).clamp(0.0, 1.0)


# =============================================================================
# Phase 1: Cache Samples
# =============================================================================

def cache_samples(args):
    """Generate and cache samples from frozen model."""
    config = get_default_config()
    device = args.device
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)

    model = ConditionalBlockARDDPM(model_config)
    if "ema_params" in checkpoint and not args.no_ema:
        state_dict = model.state_dict()
        for name in state_dict:
            if name in checkpoint["ema_params"]:
                state_dict[name] = checkpoint["ema_params"][name]
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    data = np.load(config.data_path)
    surfaces = data["surface"]
    global_mean_vol = getattr(model_config, 'global_mean_vol', 0.0187)

    for split_name, start_idx, end_idx in [
        ("train", 0, config.train_end),
        ("val", config.val_start, config.val_end),
    ]:
        print(f"\n{'='*60}")
        print(f"Caching {split_name} split (indices {start_idx}-{end_idx})")
        print(f"{'='*60}")

        dataset = VolSurfaceDataset(
            surfaces, config.history_len, config.future_len,
            start_idx=start_idx, end_idx=end_idx,
        )
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        all_samples = []
        all_gt = []
        all_vov_ratio = []

        max_b = args.max_batches if split_name == "train" else 50
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(loader, desc=f"Sampling {split_name}", total=max_b)
            ):
                if batch_idx >= max_b:
                    break

                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"].to(device))

                samples = model.sample_batched(
                    history, n_samples=args.n_samples,
                    max_global_residual=0,
                )

                # Compute vol_of_vol ratio
                history_denorm = denormalize_iv(history)
                mean_iv = history_denorm.mean(dim=(-1, -2))
                daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                vol = daily_chg.std(dim=1)
                vov_ratio = (vol / global_mean_vol).unsqueeze(-1)

                all_samples.append(samples.cpu().half())
                all_gt.append(future_gt.cpu().half())
                all_vov_ratio.append(vov_ratio.cpu())

        samples_cat = torch.cat(all_samples, dim=0)
        gt_cat = torch.cat(all_gt, dim=0)
        vov_cat = torch.cat(all_vov_ratio, dim=0)

        out_path = cache_dir / f"{split_name}.pt"
        torch.save({
            "samples": samples_cat,
            "gt": gt_cat,
            "vov_ratio": vov_cat,
            "global_mean_vol": global_mean_vol,
        }, out_path)

        print(f"  Saved: {out_path}")
        print(f"  Samples: {samples_cat.shape}")
        print(f"  GT: {gt_cat.shape}")
        print(f"  VoV ratio range: [{vov_cat.min():.3f}, {vov_cat.max():.3f}]")
        print(f"  File size: {out_path.stat().st_size / 1e9:.2f} GB")


# =============================================================================
# Phase 2: Train Scale Head
# =============================================================================

def train_scale_head(args):
    """Train PerCellRegimeScale on cached samples with regime-stratified IS."""
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Load cached data
    print("Loading cached samples...")
    train_data = torch.load(cache_dir / "train.pt", weights_only=False)
    val_data = torch.load(cache_dir / "val.pt", weights_only=False)

    train_samples = train_data["samples"].float().to(device)
    train_gt = train_data["gt"].float().to(device)
    train_vov = train_data["vov_ratio"].float().to(device)

    val_samples = val_data["samples"].float().to(device)
    val_gt = val_data["gt"].float().to(device)
    val_vov = val_data["vov_ratio"].float().to(device)

    global_mean_vol = train_data.get("global_mean_vol", 0.0187)

    print(f"  Train: {train_samples.shape[0]} windows, "
          f"{train_samples.shape[1]} samples each")
    print(f"  Val:   {val_samples.shape[0]} windows")

    # Regime split thresholds (from train data)
    median_vov = train_vov.median().item()
    print(f"  Median VoV ratio: {median_vov:.3f}")

    # Initialize scale head
    use_bias = not getattr(args, 'no_bias', False)
    head = PerCellRegimeScale(
        init_alpha=args.init_alpha,
        delta_clamp=args.delta_clamp,
        use_bias=use_bias,
    ).to(device)
    print(f"\n  Scale head params: {sum(p.numel() for p in head.parameters())}")

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    alpha = args.alpha  # CI level (0.1 = 90% CI)

    best_val_loss = float('inf')
    best_state = None

    print(f"\nTraining: {args.n_iters} iters, lr={args.lr}, "
          f"delta_clamp={args.delta_clamp}, lambda_reg={args.lambda_reg}")
    print("=" * 70)

    # Mini-batch training to avoid OOM (quantile backward is memory-heavy)
    mini_batch_size = args.mini_batch_size
    N = train_samples.shape[0]

    for i in range(args.n_iters):
        head.train()
        optimizer.zero_grad()

        # Accumulate gradients over mini-batches
        total_loss_accum = 0.0
        n_batches = 0
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mb_samples = train_samples[start:end]
            mb_gt = train_gt[start:end]
            mb_vov = train_vov[start:end]

            # Forward: compute per-cell scale
            scale = head(mb_vov)  # (mb, 5, 5)
            corrected = apply_percell_scale(mb_samples, scale)

            # Compute quantiles
            q_lo = torch.quantile(corrected, alpha / 2, dim=1)
            q_hi = torch.quantile(corrected, 1 - alpha / 2, dim=1)

            # Regime-stratified IS loss
            calm_mask = mb_vov.squeeze(-1) <= median_vov
            turb_mask = ~calm_mask

            mb_loss = torch.tensor(0.0, device=device)
            for mask in [calm_mask, turb_mask]:
                if mask.sum() == 0:
                    continue
                is_vals = interval_score(
                    q_lo[mask], q_hi[mask], mb_gt[mask], alpha
                )
                mb_loss = mb_loss + is_vals.mean()

            # Scale loss by mini-batch proportion for correct accumulation
            mb_loss = mb_loss / ((N + mini_batch_size - 1) // mini_batch_size)
            mb_loss.backward()
            total_loss_accum += mb_loss.item()
            n_batches += 1

        # Add regularization (only once per outer step)
        reg = args.lambda_reg * head.delta.pow(2).mean()
        reg.backward()
        total_loss_accum += reg.item()

        optimizer.step()
        loss = total_loss_accum - reg.item()  # for logging

        # Logging
        if i % 50 == 0 or i == args.n_iters - 1:
            head.eval()
            with torch.no_grad():
                # Recompute train coverage in mini-batches (no grad needed)
                all_covered = []
                for start in range(0, N, mini_batch_size):
                    end = min(start + mini_batch_size, N)
                    mb_s = train_samples[start:end]
                    mb_g = train_gt[start:end]
                    mb_v = train_vov[start:end]
                    sc = head(mb_v)
                    corr = apply_percell_scale(mb_s, sc)
                    ql = torch.quantile(corr, alpha / 2, dim=1)
                    qh = torch.quantile(corr, 1 - alpha / 2, dim=1)
                    all_covered.append(((mb_g >= ql) & (mb_g <= qh)).float())
                covered_train = torch.cat(all_covered, dim=0)
                cov_train = covered_train.mean().item()

                # Val metrics (small enough for one pass)
                val_scale = head(val_vov)
                val_corrected = apply_percell_scale(val_samples, val_scale)
                val_q_lo = torch.quantile(val_corrected, alpha / 2, dim=1)
                val_q_hi = torch.quantile(val_corrected, 1 - alpha / 2, dim=1)

                val_calm = val_vov.squeeze(-1) <= median_vov
                val_turb = ~val_calm
                val_loss = torch.tensor(0.0, device=device)
                for mask in [val_calm, val_turb]:
                    if mask.sum() == 0:
                        continue
                    val_loss = val_loss + interval_score(
                        val_q_lo[mask], val_q_hi[mask], val_gt[mask], alpha
                    ).mean()

                covered_val = ((val_gt >= val_q_lo) & (val_gt <= val_q_hi)).float()
                cov_val = covered_val.mean().item()

                # Per-cell coverage on train
                cell_cov = covered_train.mean(dim=(0, 1))  # (5, 5)

                delta_clamped = head.delta.clamp(
                    -head.delta_clamp, head.delta_clamp)

                print(f"\nIter {i:4d}: train_IS={loss:.4f} "
                      f"val_IS={val_loss.item():.4f} "
                      f"train_cov={cov_train:.3f} val_cov={cov_val:.3f}")
                print(f"  alpha_global={head.alpha_global.item():.4f} "
                      f"bias={head.bias.item():.4f}")
                print(f"  delta range: [{delta_clamped.min():.4f}, "
                      f"{delta_clamped.max():.4f}] "
                      f"std={delta_clamped.std():.4f}")
                print(f"  cell coverage range: [{cell_cov.min():.3f}, "
                      f"{cell_cov.max():.3f}]")

                # Per-cell coverage grid
                if i % 100 == 0 or i == args.n_iters - 1:
                    print("  Per-cell coverage (train):")
                    for r in range(5):
                        print(f"    {' '.join(f'{cell_cov[r,c]:.3f}' for c in range(5))}")
                    print("  Per-cell alpha (global + delta):")
                    alpha_grid = (head.alpha_global + delta_clamped).cpu()
                    for r in range(5):
                        print(f"    {' '.join(f'{alpha_grid[r,c]:.3f}' for c in range(5))}")

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = {k: v.cpu().clone()
                                  for k, v in head.state_dict().items()}

    # Save best
    print(f"\n{'='*60}")
    print(f"Best val IS: {best_val_loss:.4f}")
    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    save_dict = {
        "state_dict": best_state,
        "global_mean_vol": global_mean_vol,
        "init_alpha": args.init_alpha,
        "delta_clamp": args.delta_clamp,
        "use_bias": use_bias,
        "best_val_is": best_val_loss,
        "n_iters": args.n_iters,
        "lr": args.lr,
        "lambda_reg": args.lambda_reg,
    }
    torch.save(save_dict, output_path)
    print(f"Saved: {output_path}")

    # Final evaluation on both splits
    for name, samples, gt, vov in [
        ("train", train_samples, train_gt, train_vov),
        ("val", val_samples, val_gt, val_vov),
    ]:
        print(f"\n--- {name} evaluation ---")
        evaluate_on_data(head, samples, gt, vov, median_vov, alpha, device)


# =============================================================================
# Phase 3: Evaluate
# =============================================================================

def evaluate_on_data(head, samples, gt, vov, median_vov, alpha, device):
    """Evaluate scale head on a dataset with regime-stratified diagnostics."""
    head.eval()
    with torch.no_grad():
        scale = head(vov)
        corrected = apply_percell_scale(samples, scale)

        q_lo = torch.quantile(corrected, alpha / 2, dim=1)
        q_hi = torch.quantile(corrected, 1 - alpha / 2, dim=1)

        covered = ((gt >= q_lo) & (gt <= q_hi)).float()
        cell_cov = covered.mean(dim=(0, 1))  # (5, 5)

        # Per-regime per-cell coverage
        calm_mask = vov.squeeze(-1) <= median_vov
        turb_mask = ~calm_mask

        for regime_name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
            if mask.sum() == 0:
                continue
            regime_cov = covered[mask].mean(dim=(0, 1))
            n_floor = (regime_cov < 0.70).sum().item()
            n_ceil = (regime_cov > 0.95).sum().item()
            print(f"  {regime_name}: coverage [{regime_cov.min():.3f}, "
                  f"{regime_cov.max():.3f}], "
                  f"floor(<70%)={n_floor}, ceil(>95%)={n_ceil}")

        # Per-horizon coverage
        horizons = [0, 6, 13, 29]
        for h_idx in horizons:
            if h_idx < covered.shape[1]:
                h_cov = covered[:, h_idx].mean().item()
                print(f"  h={h_idx+1:2d}: coverage={h_cov:.3f}")

        # Overall
        n_gate_pass = ((cell_cov >= 0.70) & (cell_cov <= 0.95)).sum().item()
        print(f"  Overall coverage: {covered.mean():.3f}")
        print(f"  Cells in [70%, 95%]: {n_gate_pass}/25")
        print(f"  CI width: {(q_hi - q_lo).mean():.4f}")

        # Scale statistics
        print(f"  Scale range: [{scale.min():.3f}, {scale.max():.3f}]")
        print(f"  Scale mean (calm): {scale[calm_mask].mean():.3f}")
        print(f"  Scale mean (turb): {scale[turb_mask].mean():.3f}")

        return {
            "coverage_overall": covered.mean().item(),
            "cell_coverage": cell_cov.cpu().numpy().tolist(),
            "ci_width": (q_hi - q_lo).mean().item(),
        }


def evaluate_scale_head(args):
    """Load trained scale head and evaluate on cached data."""
    cache_dir = Path(args.cache_dir)
    device = args.device

    # Load scale head
    ckpt = torch.load(args.scale_head, weights_only=False, map_location=device)
    head = PerCellRegimeScale(
        init_alpha=ckpt.get("init_alpha", 0.5),
        delta_clamp=ckpt.get("delta_clamp", 0.3),
        use_bias=ckpt.get("use_bias", True),
    )
    head.load_state_dict(ckpt["state_dict"])
    head = head.to(device)
    head.eval()

    global_mean_vol = ckpt.get("global_mean_vol", 0.0187)
    alpha = args.alpha

    # Print learned parameters
    delta = head.delta.clamp(-head.delta_clamp, head.delta_clamp)
    alpha_grid = head.alpha_global + delta
    print(f"\nLearned parameters:")
    print(f"  alpha_global: {head.alpha_global.item():.4f}")
    print(f"  bias: {head.bias.item():.4f}")
    print(f"  Per-cell alpha grid:")
    for r in range(5):
        print(f"    {' '.join(f'{alpha_grid[r,c].item():.3f}' for c in range(5))}")

    for split in ["train", "val"]:
        cache_path = cache_dir / f"{split}.pt"
        if not cache_path.exists():
            print(f"\n  {split}.pt not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating on {split} split")
        data = torch.load(cache_path, weights_only=False)
        samples = data["samples"].float().to(device)
        gt = data["gt"].float().to(device)
        vov = data["vov_ratio"].float().to(device)
        median_vov = vov.median().item()

        evaluate_on_data(head, samples, gt, vov, median_vov, alpha, device)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train per-cell regime-adaptive scale with IS loss"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Cache subcommand
    p_cache = subparsers.add_parser("cache", help="Cache samples from frozen model")
    p_cache.add_argument("--model_path", type=str, required=True)
    p_cache.add_argument("--no_ema", action="store_true")
    p_cache.add_argument("--n_samples", type=int, default=50)
    p_cache.add_argument("--max_batches", type=int, default=404)
    p_cache.add_argument("--cache_dir", type=str, default="data/percell_scale_cache")
    p_cache.add_argument("--device", type=str, default="cuda")

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train scale head on cached data")
    p_train.add_argument("--cache_dir", type=str, default="data/percell_scale_cache")
    p_train.add_argument("--n_iters", type=int, default=500)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--init_alpha", type=float, default=0.5)
    p_train.add_argument("--delta_clamp", type=float, default=0.3)
    p_train.add_argument("--lambda_reg", type=float, default=0.01)
    p_train.add_argument("--mini_batch_size", type=int, default=256,
                          help="Mini-batch size for gradient accumulation (avoid OOM)")
    p_train.add_argument("--no_bias", action="store_true",
                          help="Disable global bias term (force scale=1.0 at vov_ratio=1.0)")
    p_train.add_argument("--alpha", type=float, default=0.1,
                          help="CI level (0.1 = 90%% CI)")
    p_train.add_argument("--output", type=str,
                          default="models/backfill/percell_scale_head/scale_head.pt")
    p_train.add_argument("--device", type=str, default="cuda")

    # Evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained scale head")
    p_eval.add_argument("--cache_dir", type=str, default="data/percell_scale_cache")
    p_eval.add_argument("--scale_head", type=str, required=True)
    p_eval.add_argument("--alpha", type=float, default=0.1)
    p_eval.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "cache":
        cache_samples(args)
    elif args.command == "train":
        train_scale_head(args)
    elif args.command == "evaluate":
        evaluate_scale_head(args)


if __name__ == "__main__":
    main()
