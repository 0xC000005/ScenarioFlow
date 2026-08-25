"""
Phase 2: Learned Calibration Head for Per-Cell CI Width Correction.

Trains a condition-dependent MLP that outputs per-cell correction factors.
Works in OUTPUT SPACE with pinball loss directly targeting quantile calibration.

Key design choices:
- Operates on DENORMALIZED samples (output space), not noise predictions
- Uses pinball loss (quantile regression) targeting q05 and q95 per cell
- Condition-dependent: different corrections for calm vs turb via vol_of_vol
- Corrections applied as power operation: corrected = baseline * (sample/baseline)^c
- Power > 1 widens CI (for undercovered cells), < 1 narrows (for overcovered)

Workflow:
1. Precompute samples from frozen VS bestval generator
2. Train CalibrationHead MLP: condition(128) + vol_of_vol(1) -> correction(25)
3. Pinball loss at q=0.05 and q=0.95 per cell per timestep
4. Save trained CalibrationHead for inference integration
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diffusion.block_ar.block_ar_ddpm import (
    ConditionalBlockARDDPM,
    BlockARConfig,
    denormalize_iv,
    normalize_iv,
)


class CalibrationHead(nn.Module):
    """Condition-dependent per-cell correction for CI width.

    Input: condition(bottleneck_dim) + vol_of_vol(1) -> correction(5,5)
    Output: power exponent applied to sample/baseline ratio.

    correction > 1 = wider CIs (for undercovered cells)
    correction < 1 = narrower CIs (for overcovered cells)

    Architecture: structural base + regime-dependent delta.
    - base_net: condition -> structural correction (same for all regimes)
    - regime_net: condition + regime_features -> delta correction
    - final = base + regime_weight * delta
    This ensures the regime signal is NOT buried in 129 dims.
    """

    def __init__(self, cond_dim: int = 128, hidden_dim: int = 128, surface_h: int = 5, surface_w: int = 5):
        super().__init__()
        self.surface_h = surface_h
        self.surface_w = surface_w
        out_dim = surface_h * surface_w

        # Structural base: condition -> base correction
        self.base_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.base_net[-1].weight)
        nn.init.zeros_(self.base_net[-1].bias)

        # Regime-dependent delta: regime_features -> per-cell adjustment
        # Input: [vol_of_vol_normalized, regime_prob, mean_iv_level]
        self.regime_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        nn.init.zeros_(self.regime_net[-1].weight)
        nn.init.zeros_(self.regime_net[-1].bias)

        # Learned regime weight (starts at 1.0)
        self.regime_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, condition: torch.Tensor, vol_of_vol: torch.Tensor,
                vov_mean: float = 0.0, vov_std: float = 1.0,
                mean_iv: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            condition: (B, cond_dim) encoder output
            vol_of_vol: (B, 1) vol-of-vol scalar
            vov_mean, vov_std: normalization stats for vol_of_vol
            mean_iv: (B, 1) mean IV level (optional, defaults to 0)

        Returns:
            correction: (B, 5, 5) per-cell correction factors > 0
        """
        # Structural base
        base = self.base_net(condition)  # (B, 25)

        # Regime features with normalized vol_of_vol
        vov_norm = (vol_of_vol - vov_mean) / max(vov_std, 1e-6)  # z-scored
        regime_prob = torch.sigmoid(vov_norm * 2.0)  # ~0 for calm, ~1 for turb
        if mean_iv is None:
            mean_iv = torch.zeros_like(vol_of_vol)
        regime_feats = torch.cat([vov_norm, regime_prob, mean_iv], dim=-1)  # (B, 3)
        delta = self.regime_net(regime_feats)  # (B, 25)

        raw = base + self.regime_weight * delta  # (B, 25)
        # Softplus centered near 1.0: softplus(0.367) ≈ 1.0
        correction = torch.nn.functional.softplus(raw + 0.367)
        return correction.reshape(-1, self.surface_h, self.surface_w)


def pinball_loss_per_cell(sorted_samples, ground_truth, alpha):
    """Compute pinball loss at quantile alpha using sorted samples.

    Uses linear interpolation for differentiable quantile estimation.

    Args:
        sorted_samples: (B, N, T, H, W) samples sorted along dim=1
        ground_truth: (B, T, H, W)
        alpha: quantile level (e.g., 0.05 for lower, 0.95 for upper)

    Returns:
        loss: (H, W) per-cell pinball loss averaged over B, T
    """
    N = sorted_samples.shape[1]
    idx = alpha * (N - 1)
    lo = int(idx)
    hi = min(lo + 1, N - 1)
    frac = idx - lo

    # Differentiable quantile via linear interpolation
    q = sorted_samples[:, lo] * (1 - frac) + sorted_samples[:, hi] * frac  # (B, T, H, W)

    # Pinball loss: alpha * max(y-q, 0) + (1-alpha) * max(q-y, 0)
    residual = ground_truth - q
    loss = torch.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
    return loss.mean(dim=(0, 1))  # (H, W)


def apply_correction(samples, baselines, correction):
    """Apply per-cell power correction to samples.

    corrected[r,c] = baseline[r,c] * (sample[r,c] / baseline[r,c])^correction[r,c]

    In log-space: log(corrected/bl) = c * log(sample/bl)
    This scales the deviation from baseline by factor c.

    Args:
        samples: (B, N, T, 5, 5) denormalized samples
        baselines: (B, 5, 5) baseline IV surfaces
        correction: (B, 5, 5) per-cell correction factors

    Returns:
        corrected: (B, N, T, 5, 5) corrected samples
    """
    bl = baselines.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 5, 5)
    c = correction.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 5, 5)

    # ratio = sample / baseline, handle near-zero
    ratio = (samples / bl.clamp(min=0.001)).clamp(min=1e-6)

    # Apply power correction: ratio^c
    corrected = bl * ratio.pow(c)
    return corrected.clamp(0.001, 1.0)


def evaluate_coverage(samples, ground_truth, horizons=[0, 6, 13, 29], vol_of_vol=None):
    """Compute per-cell 90% CI coverage, optionally split by regime."""
    q05 = torch.quantile(samples, 0.05, dim=1)  # (B, T, 5, 5)
    q95 = torch.quantile(samples, 0.95, dim=1)  # (B, T, 5, 5)
    covered = (ground_truth >= q05) & (ground_truth <= q95)

    results = {}

    # Overall coverage
    for h in horizons:
        cov = covered[:, h].float().mean(dim=0)  # (5, 5)
        under70 = (cov < 0.70).sum().item()
        over95 = (cov > 0.95).sum().item()
        results[f"h={h+1}"] = {
            "mean_coverage": cov.mean().item(),
            "under_70": under70,
            "over_95": over95,
        }

    overall_cov = covered.float().mean(dim=(0, 1))  # (5, 5)
    under70_total = (overall_cov < 0.70).sum().item()
    over95_total = (overall_cov > 0.95).sum().item()
    results["overall"] = {
        "mean_coverage": overall_cov.mean().item(),
        "under_70": under70_total,
        "over_95": over95_total,
        "combined": under70_total + over95_total,
        "per_cell_coverage": overall_cov.numpy().round(3).tolist(),
    }

    # Per-regime coverage if vol_of_vol provided
    if vol_of_vol is not None:
        vov_flat = vol_of_vol.squeeze(-1)  # (B,)
        median_vov = vov_flat.median()
        calm_mask = vov_flat <= median_vov
        turb_mask = vov_flat > median_vov

        for regime, mask in [("calm", calm_mask), ("turb", turb_mask)]:
            if mask.sum() < 10:
                continue
            regime_cov = covered[mask].float().mean(dim=(0, 1))  # (5, 5)
            under70_r = (regime_cov < 0.70).sum().item()
            over95_r = (regime_cov > 0.95).sum().item()
            results[f"{regime}_overall"] = {
                "mean_coverage": regime_cov.mean().item(),
                "n_windows": mask.sum().item(),
                "under_70": under70_r,
                "over_95": over95_r,
                "combined": under70_r + over95_r,
                "per_cell_coverage": regime_cov.numpy().round(3).tolist(),
            }

    return results


def precompute_samples(
    model_path: str,
    data_path: str,
    n_samples: int = 50,
    max_windows: int = 500,
    device: str = "cuda",
    data_start: int = 4040,
    data_end: int = 4540,
):
    """Generate samples from frozen model and extract all needed data."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    config = BlockARConfig(**checkpoint["config"])
    model = ConditionalBlockARDDPM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # Load data
    data = np.load(data_path)
    surfaces = torch.tensor(data["surface"], dtype=torch.float32)

    val_start = data_start
    val_end = data_end
    history_len = config.history_len
    future_len = config.future_len

    all_samples = []
    all_gt = []
    all_conditions = []
    all_vol_of_vol = []
    all_baselines = []

    n_windows = min(max_windows, val_end - val_start - history_len - future_len + 1)
    batch_size = 8  # smaller batch for 50 samples

    with torch.no_grad():
        for start_idx in range(0, n_windows, batch_size):
            end_idx = min(start_idx + batch_size, n_windows)

            histories = []
            futures = []
            for i in range(start_idx, end_idx):
                global_idx = val_start + i
                h = surfaces[global_idx : global_idx + history_len]
                f = surfaces[global_idx + history_len : global_idx + history_len + future_len]
                if f.shape[0] < future_len:
                    continue
                histories.append(h)
                futures.append(f)

            if len(histories) == 0:
                continue

            history_raw = torch.stack(histories).to(device)  # (B, 30, 5, 5) in [0, 1]
            future_raw = torch.stack(futures).to(device)      # (B, 30, 5, 5) in [0, 1]
            B_actual = history_raw.shape[0]

            # Normalize history for model.sample() which expects [-1, 1]
            history_norm = normalize_iv(history_raw)

            # Generate samples (returns denormalized [0,1])
            samples = model.sample(
                history_norm, n_samples=n_samples,
            )  # (B, N, 30, 5, 5) denormalized

            # Get condition vector from encoder (match model.sample() conditioning)
            condition = model.encoder(history_norm, mask=None)  # (B, bottleneck_dim)
            if config.forward_only:
                condition = condition + model.encoder.null_embedding.expand(B_actual, -1)

            # Compute vol_of_vol from raw surfaces
            mean_iv = history_raw.mean(dim=(-1, -2))  # (B, T)
            daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
            vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)

            # Compute baseline from raw surfaces
            K = min(config.baseline_window, history_raw.shape[1])
            baseline = history_raw[:, -K:].mean(dim=1)  # (B, 5, 5)
            baseline = baseline.clamp(min=0.01)

            all_samples.append(samples.cpu())
            all_gt.append(future_raw.cpu())
            all_conditions.append(condition.cpu())
            all_vol_of_vol.append(vol.cpu())
            all_baselines.append(baseline.cpu())

            if (start_idx // batch_size) % 10 == 0:
                print(f"  Generated {end_idx}/{n_windows} windows")

    return {
        "samples": torch.cat(all_samples),       # (N_win, N_samp, 30, 5, 5)
        "ground_truth": torch.cat(all_gt),        # (N_win, 30, 5, 5)
        "conditions": torch.cat(all_conditions),  # (N_win, 128)
        "vol_of_vol": torch.cat(all_vol_of_vol),  # (N_win, 1)
        "baselines": torch.cat(all_baselines),    # (N_win, 5, 5)
    }


def train_calibration_head(
    precomputed: dict,
    cond_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    epochs: int = 300,
    device: str = "cuda",
    structural_only: bool = False,
):
    """Train CalibrationHead on precomputed samples with pinball loss."""
    samples = precomputed["samples"]        # (N, S, T, 5, 5)
    gt = precomputed["ground_truth"]        # (N, T, 5, 5)
    conditions = precomputed["conditions"]  # (N, cond_dim)
    vov = precomputed["vol_of_vol"]         # (N, 1)
    baselines = precomputed["baselines"]    # (N, 5, 5)

    N_windows = samples.shape[0]
    N_samp = samples.shape[1]
    print(f"Training on {N_windows} windows, {N_samp} samples each")

    # Evaluate BEFORE correction
    print("\n=== Before calibration ===")
    before_results = evaluate_coverage(samples, gt, vol_of_vol=vov)
    for k, v in before_results.items():
        print(f"  {k}: coverage={v['mean_coverage']:.3f}, "
              f"under70={v.get('under_70', 'N/A')}, over95={v.get('over_95', 'N/A')}")

    # Create calibration head
    head = CalibrationHead(cond_dim=cond_dim, hidden_dim=hidden_dim).to(device)
    if structural_only:
        # Freeze regime pathway — head learns only structural per-cell corrections
        head.regime_weight.data.fill_(0.0)
        head.regime_weight.requires_grad = False
        for p in head.regime_net.parameters():
            p.requires_grad = False
        print("Structural-only mode: regime pathway frozen")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, head.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Move data to GPU
    samples_gpu = samples.to(device)
    gt_gpu = gt.to(device)
    cond_gpu = conditions.to(device)
    vov_gpu = vov.to(device)
    bl_gpu = baselines.to(device)

    batch_size = min(32, N_windows)
    n_batches = (N_windows + batch_size - 1) // batch_size

    # Precompute regime split threshold for stratified loss
    median_vov = vov_gpu.squeeze(-1).median()
    print(f"Median vol_of_vol: {median_vov:.5f}")

    # Precompute vol_of_vol normalization stats and mean IV
    vov_mean = float(vov_gpu.mean())
    vov_std = float(vov_gpu.std())
    mean_iv_gpu = baselines.mean(dim=(-1, -2), keepdim=True).squeeze(-1).to(device)  # (N, 1)
    print(f"Vol_of_vol stats: mean={vov_mean:.5f}, std={vov_std:.5f}")

    best_combined = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        perm = torch.randperm(N_windows)

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            s_batch = samples_gpu[idx]    # (B, S, T, 5, 5)
            gt_batch = gt_gpu[idx]        # (B, T, 5, 5)
            c_batch = cond_gpu[idx]       # (B, cond_dim)
            v_batch = vov_gpu[idx]        # (B, 1)
            bl_batch = bl_gpu[idx]        # (B, 5, 5)

            # Get correction (with regime features)
            mi_batch = mean_iv_gpu[idx]  # (B, 1)
            correction = head(c_batch, v_batch, vov_mean=vov_mean, vov_std=vov_std, mean_iv=mi_batch)  # (B, 5, 5)

            # Apply correction
            corrected = apply_correction(s_batch, bl_batch, correction)

            # Sort along sample dim for quantile computation
            sorted_corrected, _ = corrected.sort(dim=1)

            # Regime-stratified pinball loss: compute separately for calm/turb
            # so the head learns different corrections per regime
            B_curr = s_batch.shape[0]
            vov_batch = v_batch.squeeze(-1)  # (B,)
            calm_b = vov_batch <= median_vov
            turb_b = vov_batch > median_vov

            loss = torch.tensor(0.0, device=device)
            for regime_mask in [calm_b, turb_b]:
                if regime_mask.sum() == 0:
                    continue
                sc_r = sorted_corrected[regime_mask]
                gt_r = gt_batch[regime_mask]
                loss_lo = pinball_loss_per_cell(sc_r, gt_r, alpha=0.05)
                loss_hi = pinball_loss_per_cell(sc_r, gt_r, alpha=0.95)
                loss = loss + (loss_lo + loss_hi).mean()

            # Light regularization: keep correction from exploding
            reg = ((correction - 1.0) ** 2).mean() * 0.001

            total_loss = loss + reg

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        epoch_loss /= n_batches

        if (epoch + 1) % 25 == 0 or epoch == 0:
            # Evaluate
            head.eval()
            with torch.no_grad():
                all_correction = head(cond_gpu, vov_gpu, vov_mean=vov_mean, vov_std=vov_std, mean_iv=mean_iv_gpu)  # (N, 5, 5)
                all_corrected = apply_correction(samples_gpu, bl_gpu, all_correction)
                results = evaluate_coverage(all_corrected.cpu(), gt, vol_of_vol=vov)

            combined = results["overall"]["combined"]
            calm_combined = results.get("calm_overall", {}).get("combined", "?")
            turb_combined = results.get("turb_overall", {}).get("combined", "?")

            # Best = minimize total combined, tiebreak on max(calm, turb)
            if isinstance(combined, (int, float)) and combined < best_combined:
                best_combined = combined
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_epoch = epoch + 1

            print(f"Epoch {epoch+1:3d} | loss={epoch_loss:.6f} | "
                  f"coverage={results['overall']['mean_coverage']:.3f} | "
                  f"combined={combined} (calm={calm_combined}, turb={turb_combined}) | "
                  f"correction=[{all_correction.min():.3f}, {all_correction.max():.3f}]")

    # Load best state (by combined failures)
    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"\nLoaded best checkpoint from epoch {best_epoch} (combined={best_combined})")
    head = head.to(device).eval()

    # Final evaluation
    print("\n=== After calibration ===")
    with torch.no_grad():
        all_correction = head(cond_gpu, vov_gpu, vov_mean=vov_mean, vov_std=vov_std, mean_iv=mean_iv_gpu)
        all_corrected = apply_correction(samples_gpu, bl_gpu, all_correction)
        after_results = evaluate_coverage(all_corrected.cpu(), gt, vol_of_vol=vov)

    for k, v in after_results.items():
        print(f"  {k}: coverage={v['mean_coverage']:.3f}, "
              f"under70={v.get('under_70', 'N/A')}, over95={v.get('over_95', 'N/A')}")

    print(f"\nCorrection stats: mean={all_correction.mean():.3f}, "
          f"std={all_correction.std():.3f}, "
          f"min={all_correction.min():.3f}, max={all_correction.max():.3f}")

    # Per-cell mean correction by regime
    vov_flat = vov.squeeze(-1)
    median_vov = vov_flat.median()
    calm_mask = vov_flat <= median_vov
    turb_mask = vov_flat > median_vov

    mean_correction_all = all_correction.mean(dim=0).cpu()
    print(f"\nPer-cell mean correction (all):\n{mean_correction_all.numpy().round(3)}")

    if calm_mask.sum() > 0:
        mean_calm = all_correction[calm_mask].mean(dim=0).cpu()
        print(f"\nPer-cell mean correction (calm):\n{mean_calm.numpy().round(3)}")

    if turb_mask.sum() > 0:
        mean_turb = all_correction[turb_mask].mean(dim=0).cpu()
        print(f"\nPer-cell mean correction (turb):\n{mean_turb.numpy().round(3)}")

    return head, before_results, after_results, vov_mean, vov_std


def main():
    parser = argparse.ArgumentParser(description="Train learned calibration head")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="models/backfill/calibration_head")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Dir to cache precomputed samples (skip regeneration)")
    parser.add_argument("--data_start", type=int, default=4040,
                        help="Start index for data window (4040=val, 4540=test)")
    parser.add_argument("--data_end", type=int, default=4540,
                        help="End index for data window (4540=val_end, 5822=test_end)")
    parser.add_argument("--structural_only", action="store_true",
                        help="Disable regime pathway (structural corrections only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached precomputed data
    cache_path = None
    if args.cache_dir:
        cache_path = Path(args.cache_dir) / "precomputed.pt"

    if cache_path and cache_path.exists():
        print(f"Loading cached samples from {cache_path}")
        precomputed = torch.load(cache_path, weights_only=False)
    else:
        print("Precomputing samples from frozen generator...")
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

    # Train
    head, before, after, vov_mean, vov_std = train_calibration_head(
        precomputed,
        cond_dim=precomputed["conditions"].shape[1],
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        structural_only=args.structural_only,
    )

    # Save (include normalization stats for inference)
    save_path = output_dir / "calibration_head.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "cond_dim": precomputed["conditions"].shape[1],
        "hidden_dim": args.hidden_dim,
        "vov_mean": vov_mean,
        "vov_std": vov_std,
        "before_results": before,
        "after_results": after,
    }, save_path)
    print(f"\nSaved calibration head to {save_path}")

    # Save results
    results_path = output_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump({"before": before, "after": after}, f, indent=2, default=str)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
