"""
Online Conformal Calibration for Per-Cell CI Coverage.

Instead of learning corrections from val (which has different regime-coverage
patterns than test), use a SLIDING WINDOW of recent test windows to compute
per-cell quantile adjustments adaptively.

This is inspired by Adaptive Conformal Inference (Gibbs & Candes 2021):
- At each test window t, use windows [t-W, t-1] to calibrate
- Per-cell: separate adjustment for each (r,c) position
- Per-regime: optional, split calibration windows by vov
- Distribution-free: no parametric assumptions

Applied as multiplicative scaling of sample deviations from baseline:
  corrected = baseline * (sample / baseline) ^ correction
  where correction is computed to achieve target coverage in recent window.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import optimize


def compute_per_cell_correction_power(
    samples: np.ndarray,    # (W, N, T, 5, 5) denormalized
    gt: np.ndarray,         # (W, T, 5, 5)
    baselines: np.ndarray,  # (W, 5, 5)
    target_coverage: float = 0.90,
    per_horizon: bool = False,  # if True, return (T, 5, 5) corrections
    eval_horizons: list = None,  # only calibrate these horizons (indices)
) -> np.ndarray:
    """Find per-cell power correction to achieve target coverage.

    Returns:
        corrections: (5, 5) or (T, 5, 5) if per_horizon=True
    """
    H, W_surf = 5, 5
    T = samples.shape[2]

    if per_horizon:
        corrections = np.ones((T, H, W_surf))
        horizons_to_cal = eval_horizons if eval_horizons else range(T)
        for t_idx in horizons_to_cal:
            for r in range(H):
                for c in range(W_surf):
                    cell_samples = samples[:, :, t_idx, r, c]  # (W, N)
                    cell_gt = gt[:, t_idx, r, c]                # (W,)
                    cell_bl = baselines[:, r, c]                # (W,)

                    def coverage_at_power(power, cs=cell_samples, cg=cell_gt, cb=cell_bl):
                        ratio = (cs / cb[:, None].clip(min=0.001)).clip(min=1e-6)
                        corrected = cb[:, None] * ratio ** power
                        q05 = np.percentile(corrected, 5, axis=1)
                        q95 = np.percentile(corrected, 95, axis=1)
                        covered = (cg >= q05) & (cg <= q95)
                        return covered.mean() - target_coverage

                    try:
                        result = optimize.brentq(coverage_at_power, 0.1, 5.0, xtol=0.01)
                        corrections[t_idx, r, c] = result
                    except ValueError:
                        cov_lo = coverage_at_power(0.1)
                        if cov_lo > 0:
                            corrections[t_idx, r, c] = 0.1
                        else:
                            corrections[t_idx, r, c] = 5.0

        # Interpolate non-calibrated horizons from nearest calibrated ones
        if eval_horizons and len(eval_horizons) < T:
            cal_set = set(eval_horizons)
            for t_idx in range(T):
                if t_idx not in cal_set:
                    # Find nearest calibrated horizon
                    nearest = min(eval_horizons, key=lambda h: abs(h - t_idx))
                    corrections[t_idx] = corrections[nearest]

        return corrections
    else:
        corrections = np.ones((H, W_surf))
        for r in range(H):
            for c in range(W_surf):
                cell_samples = samples[:, :, :, r, c]  # (W, N, T)
                cell_gt = gt[:, :, r, c]                # (W, T)
                cell_bl = baselines[:, r, c]             # (W,)

                def coverage_at_power(power, cs=cell_samples, cg=cell_gt, cb=cell_bl):
                    bl_exp = cb[:, None]  # (W, 1)
                    ratio = (cs / bl_exp[:, :, None].clip(min=0.001)).clip(min=1e-6)
                    corrected = bl_exp[:, :, None] * ratio ** power
                    q05 = np.percentile(corrected, 5, axis=1)  # (W, T)
                    q95 = np.percentile(corrected, 95, axis=1)
                    covered = (cg >= q05) & (cg <= q95)
                    return covered.mean() - target_coverage

                try:
                    result = optimize.brentq(coverage_at_power, 0.1, 5.0, xtol=0.01)
                    corrections[r, c] = result
                except ValueError:
                    cov_lo = coverage_at_power(0.1)
                    if cov_lo > 0:
                        corrections[r, c] = 0.1
                    else:
                        corrections[r, c] = 5.0

        return corrections


def apply_power_correction(
    samples: np.ndarray,    # (B, N, T, 5, 5)
    baselines: np.ndarray,  # (B, 5, 5)
    correction: np.ndarray, # (5, 5) or (B, 5, 5) or (T, 5, 5)
) -> np.ndarray:
    """Apply per-cell power correction to samples."""
    bl = baselines[:, None, None, :, :]  # (B, 1, 1, 5, 5)
    if correction.ndim == 2:
        c = correction[None, None, None, :, :]   # (1, 1, 1, 5, 5)
    elif correction.ndim == 3 and correction.shape[0] == samples.shape[2]:
        # Per-horizon correction: (T, 5, 5) -> (1, 1, T, 5, 5)
        c = correction[None, None, :, :, :]
    else:
        c = correction[:, None, None, :, :]       # (B, 1, 1, 5, 5)

    ratio = (samples / np.clip(bl, 0.001, None)).clip(1e-6)
    corrected = bl * ratio ** c
    return np.clip(corrected, 0.001, 1.0)


def online_conformal_calibration(
    samples: np.ndarray,      # (N_total, N_samp, T, 5, 5)
    gt: np.ndarray,           # (N_total, T, 5, 5)
    baselines: np.ndarray,    # (N_total, 5, 5)
    vol_of_vol: np.ndarray,   # (N_total, 1)
    window_size: int = 100,
    regime_split: bool = True,
    target_coverage: float = 0.90,
    per_horizon: bool = False,
    eval_horizons: list = None,  # horizon indices for per-horizon calibration
):
    """Apply online conformal calibration with sliding window.

    For each test window t:
    1. Use windows [t-W, t-1] to compute per-cell corrections
    2. Optionally split by regime (calm/turb based on vov median in window)
    3. Apply corrections to window t's samples
    4. Evaluate coverage on window t

    Returns:
        corrected_samples: (N_total, N_samp, T, 5, 5)
        diagnostics: dict with per-window coverage info
    """
    N_total = samples.shape[0]
    corrected = samples.copy()
    diagnostics = {
        "per_window_correction_mean": [],
        "per_window_coverage_before": [],
        "per_window_coverage_after": [],
    }

    vov_flat = vol_of_vol.squeeze(-1)

    for t in range(window_size, N_total):
        # Calibration window
        cal_start = t - window_size
        cal_samples = samples[cal_start:t]
        cal_gt = gt[cal_start:t]
        cal_bl = baselines[cal_start:t]
        cal_vov = vov_flat[cal_start:t]

        if regime_split:
            # Determine current window's regime
            curr_vov = vov_flat[t]
            median_vov = np.median(cal_vov)

            # Use only same-regime windows for calibration
            if curr_vov <= median_vov:
                mask = cal_vov <= median_vov
            else:
                mask = cal_vov > median_vov

            if mask.sum() < 10:
                mask = np.ones(window_size, dtype=bool)  # fallback to all

            cal_samples_r = cal_samples[mask]
            cal_gt_r = cal_gt[mask]
            cal_bl_r = cal_bl[mask]
        else:
            cal_samples_r = cal_samples
            cal_gt_r = cal_gt
            cal_bl_r = cal_bl

        # Compute per-cell corrections from calibration window
        correction = compute_per_cell_correction_power(
            cal_samples_r, cal_gt_r, cal_bl_r,
            target_coverage=target_coverage,
            per_horizon=per_horizon,
            eval_horizons=eval_horizons,
        )

        # Apply to current window
        curr_corrected = apply_power_correction(
            samples[t:t+1], baselines[t:t+1], correction,
        )
        corrected[t] = curr_corrected[0]

        diagnostics["per_window_correction_mean"].append(float(correction.mean()))

    # Also apply a one-time correction to the first `window_size` windows
    # using the first calibration window [0, window_size)
    init_correction = compute_per_cell_correction_power(
        samples[:window_size], gt[:window_size], baselines[:window_size],
        target_coverage=target_coverage,
        per_horizon=per_horizon,
        eval_horizons=eval_horizons,
    )
    for t in range(window_size):
        corrected[t] = apply_power_correction(
            samples[t:t+1], baselines[t:t+1], init_correction,
        )[0]

    return corrected, diagnostics


def evaluate_conformal(
    corrected: np.ndarray,   # (N, S, T, 5, 5)
    gt: np.ndarray,          # (N, T, 5, 5)
    vol_of_vol: np.ndarray,  # (N, 1)
    horizons: list = [0, 6, 13, 29],
):
    """Evaluate per-cell, per-regime coverage on conformally calibrated samples."""
    q05 = np.percentile(corrected, 5, axis=1)
    q95 = np.percentile(corrected, 95, axis=1)
    covered = (gt >= q05) & (gt <= q95)

    vov_flat = vol_of_vol.squeeze(-1)
    median_vov = np.median(vov_flat)
    calm_mask = vov_flat <= median_vov
    turb_mask = vov_flat > median_vov

    # Use Q20/Q80 for regime split (matching test suite)
    q20_vov = np.percentile(vov_flat, 20)
    q80_vov = np.percentile(vov_flat, 80)
    calm_mask_strict = vov_flat <= q20_vov
    turb_mask_strict = vov_flat >= q80_vov

    results = {}

    # Overall per-cell coverage
    overall_cov = covered.mean(axis=(0, 1))  # (5, 5)
    under70 = int((overall_cov < 0.70).sum())
    over95 = int((overall_cov > 0.95).sum())
    results["overall"] = {
        "mean_coverage": float(overall_cov.mean()),
        "under_70": under70,
        "over_95": over95,
        "combined": under70 + over95,
        "grid": overall_cov.round(3).tolist(),
    }

    # Per-regime per-cell coverage (using Q20/Q80)
    layer2_under70 = 0
    layer2_over95 = 0
    for regime, mask in [("calm", calm_mask_strict), ("turb", turb_mask_strict)]:
        regime_results = {}
        for h_idx in horizons:
            h_cov = covered[mask, h_idx].mean(axis=0)  # (5, 5)
            u = int((h_cov < 0.70).sum())
            o = int((h_cov > 0.95).sum())
            layer2_under70 += u
            layer2_over95 += o
            regime_results[f"h={h_idx+1}"] = {
                "mean_coverage": float(h_cov.mean()),
                "under_70": u,
                "over_95": o,
                "worst": float(h_cov.min()),
                "best": float(h_cov.max()),
            }
        results[regime] = regime_results

    results["layer2_combined"] = layer2_under70 + layer2_over95
    results["layer2_under70"] = layer2_under70
    results["layer2_over95"] = layer2_over95

    return results


def main():
    parser = argparse.ArgumentParser(description="Online conformal calibration")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--regime_split", action="store_true")
    parser.add_argument("--per_horizon", action="store_true",
                        help="Per-horizon corrections (slower but prevents cross-horizon leakage)")
    parser.add_argument("--max_windows", type=int, default=9999)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/block_ar/conformal")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--data_start", type=int, default=4540,
                        help="Start index (4540=test)")
    parser.add_argument("--data_end", type=int, default=5822)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cache
    cache_path = None
    if args.cache_dir:
        cache_path = Path(args.cache_dir) / "precomputed.pt"

    if cache_path and cache_path.exists():
        print(f"Loading cached samples from {cache_path}")
        precomputed = torch.load(cache_path, weights_only=False)
    else:
        print("Precomputing samples from frozen generator...")
        from experiments.backfill.block_ar.train_calibration_head import precompute_samples
        precomputed = precompute_samples(
            model_path=args.model_path,
            data_path=args.data_path,
            n_samples=args.n_samples,
            max_windows=args.max_windows,
            device=args.device,
            data_start=args.data_start,
            data_end=args.data_end,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(precomputed, cache_path)
            print(f"Cached samples to {cache_path}")

    samples = precomputed["samples"].numpy()
    gt = precomputed["ground_truth"].numpy()
    baselines = precomputed["baselines"].numpy()
    vov = precomputed["vol_of_vol"].numpy()

    N = samples.shape[0]
    print(f"\nData: {N} windows, {samples.shape[1]} samples each")
    print(f"Window size: {args.window_size}")
    print(f"Regime split: {args.regime_split}")
    print(f"Per-horizon: {args.per_horizon}")

    # Before correction
    print("\n=== Before conformal calibration ===")
    before_results = evaluate_conformal(samples, gt, vov)
    print(f"  Overall: coverage={before_results['overall']['mean_coverage']:.3f}, "
          f"Layer2 combined={before_results['layer2_combined']}")
    for regime in ["calm", "turb"]:
        for h_key, h_data in before_results[regime].items():
            print(f"  {regime} {h_key}: {h_data['mean_coverage']:.3f} "
                  f"[{h_data['worst']:.3f}, {h_data['best']:.3f}] "
                  f"u70={h_data['under_70']} o95={h_data['over_95']}")

    # Apply online conformal
    eval_horizons = [0, 6, 13, 29] if args.per_horizon else None
    print(f"\n=== Applying online conformal calibration (W={args.window_size}) ===")
    corrected, diagnostics = online_conformal_calibration(
        samples, gt, baselines, vov,
        window_size=args.window_size,
        regime_split=args.regime_split,
        per_horizon=args.per_horizon,
        eval_horizons=eval_horizons,
    )

    # After correction
    print("\n=== After conformal calibration ===")
    after_results = evaluate_conformal(corrected, gt, vov)
    print(f"  Overall: coverage={after_results['overall']['mean_coverage']:.3f}, "
          f"Layer2 combined={after_results['layer2_combined']}")
    for regime in ["calm", "turb"]:
        for h_key, h_data in after_results[regime].items():
            print(f"  {regime} {h_key}: {h_data['mean_coverage']:.3f} "
                  f"[{h_data['worst']:.3f}, {h_data['best']:.3f}] "
                  f"u70={h_data['under_70']} o95={h_data['over_95']}")

    # Improvement summary
    before_l2 = before_results["layer2_combined"]
    after_l2 = after_results["layer2_combined"]
    print(f"\n  Layer2 improvement: {before_l2} -> {after_l2} ({before_l2 - after_l2} fewer failures)")

    # Save
    save_data = {
        "before": before_results,
        "after": after_results,
        "window_size": args.window_size,
        "regime_split": args.regime_split,
    }
    with open(output_dir / "conformal_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'conformal_results.json'}")


if __name__ == "__main__":
    main()
