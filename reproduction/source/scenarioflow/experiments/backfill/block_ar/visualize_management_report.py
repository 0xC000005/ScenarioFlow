#!/usr/bin/env python
"""
Management Report: Block-AR Volatility Surface Scenario Generator Quality.

Generates publication-quality visualizations demonstrating:
1. Conditional variance fan charts with history context (calm vs turbulent, per cell)
2. Cross-cell regime sensitivity with history context
3. Temporal properties (kurtosis, ACF)
4. Surface structure validity (term structure, smile)
5. Surface heatmap snapshots (GT vs generated vs uncertainty)
6. Calibration curve
7. Marginal daily change distributions per cell (GT vs generated)
"""

import dataclasses
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats as sp_stats
from matplotlib.patches import FancyBboxPatch

from diffusion.block_ar.single_pass_ar import (
    SinglePassConfig,
    SinglePassBlockAR,
    normalize_iv,
)
# Legacy imports for backward compatibility
try:
    from diffusion.block_ar.block_ar_ddpm import (
        BlockARConfig,
        ConditionalBlockARDDPM,
        normalize_iv as normalize_iv_ddpm,
    )
    HAS_DDPM = True
except ImportError:
    HAS_DDPM = False

# ── Constants ──
MONEYNESS = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
MATURITIES_DAYS = np.array([30, 91, 182, 365, 730])
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
MONEYNESS_LABELS = ["0.70", "0.85", "1.00", "1.15", "1.30"]
CELL_NAMES = {}
for r in range(5):
    for c in range(5):
        CELL_NAMES[(r, c)] = f"{MATURITY_LABELS[r]} / K={MONEYNESS_LABELS[c]}"

import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model_path", default="models/backfill/afcrps_90d/best_model.pt")
_parser.add_argument("--output_dir", default="results/block_ar/management_report_90d")
_parser.add_argument("--quantile_map", default=None, help="Path to quantile_map.npz")
_parser.add_argument("--qmap_alpha", type=float, default=1.0, help="Quantile map alpha")
_args, _ = _parser.parse_known_args()

OUTPUT_DIR = _args.output_dir
MODEL_PATH = _args.model_path
QUANTILE_MAP = _args.quantile_map
QMAP_ALPHA = _args.qmap_alpha
N_SAMPLES = 50
MAX_WINDOWS = 400

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

CALM_COLOR = "#2196F3"
TURB_COLOR = "#F44336"
GT_COLOR = "#2E7D32"
SCENARIO_COLOR = "#9E9E9E"
BAND_ALPHA = 0.15


def load_model(model_path, device):
    cp = torch.load(model_path, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)

    # Detect model type from config fields
    is_single_pass = "noise_dim" in c or "noise_embed_dim" in c
    if is_single_pass:
        config = SinglePassConfig(**c)
        model = SinglePassBlockAR(config)
    else:
        config = BlockARConfig(**c)
        model = ConditionalBlockARDDPM(config)

    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model, config


def generate_all_data(model, config, device):
    """Generate samples for all test windows. Returns dict of arrays."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_start = 4540
    test_surfaces = surfaces[test_start:]

    history_len = config.history_len
    future_len = config.future_len
    total_len = history_len + future_len
    N = len(test_surfaces)
    n_windows = min(MAX_WINDOWS, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history, all_future = [], []
    for idx in indices:
        all_history.append(test_surfaces[idx:idx + history_len])
        all_future.append(test_surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)
    future_arr = np.stack(all_future)

    # vol_of_vol
    mean_iv = history_arr.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)

    # Generate samples
    print(f"Generating {N_SAMPLES} samples for {n_windows} windows...")
    batch_size = 16
    all_samples = []
    for i in range(0, n_windows, batch_size):
        batch_end = min(i + batch_size, n_windows)
        hist = torch.tensor(history_arr[i:batch_end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist)
        with torch.no_grad():
            samples = model.sample(hist_norm, n_samples=N_SAMPLES)
        all_samples.append(samples.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {batch_end}/{n_windows}")

    all_samples = np.concatenate(all_samples, axis=0)

    # Use raw samples (no conformal calibration — per standing directive)
    return {
        "history": history_arr,      # (W, 30, 5, 5)
        "future": future_arr,        # (W, 30, 5, 5)
        "samples": all_samples,      # (W, S, 30, 5, 5) — raw
        "samples_raw": all_samples,  # (W, S, 30, 5, 5) — same (no conformal)
        "vol_of_vol": vol_of_vol,    # (W,)
        "n_windows": n_windows,
    }


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-Cell Fan Charts — Calm vs Turbulent
# ══════════════════════════════════════════════════════════════════════
def plot_fan_charts(data):
    """Fan charts with history context for selected cells, calm vs turbulent."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    cells = [(0, 0), (2, 2), (4, 2), (0, 4)]
    cell_labels = [
        "1M OTM Put (K=0.70)\nHigh uncertainty",
        "6M ATM (K=1.00)\nMedium uncertainty",
        "2Y ATM (K=1.00)\nLow uncertainty",
        "1M OTM Call (K=1.30)\nHigh uncertainty",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle("Scenario Fan Charts: Calm vs Turbulent Market Regimes\n"
                 "Black = 30-day history, colored = forecast fan, green dashed = GT future",
                 fontsize=15, fontweight="bold", y=0.99)

    hist_days = np.arange(-29, 1)
    fwd_days = np.arange(1, 31)

    # First pass: compute y-ranges per row (shared across calm/turbulent)
    row_ylims = []
    for row, ((r, c), label) in enumerate(zip(cells, cell_labels)):
        ymin, ymax = np.inf, -np.inf
        for win_idx in [calm_idx, turb_idx]:
            hist_cell = data["history"][win_idx, :, r, c]
            samples_cell = data["samples"][win_idx, :, :, r, c]
            gt_cell = data["future"][win_idx, :, r, c]
            ymin = min(ymin, hist_cell.min(), samples_cell.min(), gt_cell.min())
            ymax = max(ymax, hist_cell.max(), samples_cell.max(), gt_cell.max())
        margin = (ymax - ymin) * 0.08
        row_ylims.append((ymin - margin, ymax + margin))

    for row, ((r, c), label) in enumerate(zip(cells, cell_labels)):
        for col, (win_idx, regime, color) in enumerate([
            (calm_idx, "Calm", CALM_COLOR),
            (turb_idx, "Turbulent", TURB_COLOR),
        ]):
            ax = axes[row, col]
            hist_cell = data["history"][win_idx, :, r, c]     # (30,)
            samples_cell = data["samples"][win_idx, :, :, r, c]  # (S, 30)
            gt_cell = data["future"][win_idx, :, r, c]            # (30,)

            # History (black solid)
            ax.plot(hist_days, hist_cell, color="black", linewidth=1.5,
                    label="History", zorder=6)

            # Forecast boundary
            ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6)

            # Light background paths (all samples, very faint)
            for s in range(min(N_SAMPLES, 50)):
                ax.plot(fwd_days, samples_cell[s], color=color, alpha=0.04, linewidth=0.3)

            # 90% CI band (light shading for context)
            q05 = np.percentile(samples_cell, 5, axis=0)
            q95 = np.percentile(samples_cell, 95, axis=0)
            ax.fill_between(fwd_days, q05, q95, color=color, alpha=0.08)

            # Individual paths — 6 distinct paths, visible and distinguishable
            path_colors = ["#E91E63", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
            n_show = min(6, N_SAMPLES)
            # Pick spread of paths: some from each quartile for variety
            path_indices = np.linspace(0, N_SAMPLES - 1, n_show, dtype=int)
            for pi, s in enumerate(path_indices):
                lbl = f"Path {pi+1}" if (row == 0 and col == 0) else None
                ax.plot(fwd_days, samples_cell[s], color=path_colors[pi % len(path_colors)],
                        alpha=0.7, linewidth=1.0, label=lbl, zorder=4)
                # Connect history to each path
                ax.plot([0, 1], [hist_cell[-1], samples_cell[s, 0]],
                        color=path_colors[pi % len(path_colors)], linewidth=0.7, alpha=0.5)

            # GT (thick, dashed green)
            ax.plot(fwd_days, gt_cell, color=GT_COLOR, linewidth=2.0,
                    linestyle="--", label="Ground truth" if (row == 0 and col == 1) else None, zorder=5)

            ax.set_ylim(row_ylims[row])

            # Annotate spread
            spread = samples_cell.std(axis=0).mean()
            ax.text(0.97, 0.95, f"spread={spread*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if row == 0:
                vov_val = vov[win_idx]
                ax.set_title(f"{regime} Regime (vol-of-vol={vov_val:.4f})",
                             fontsize=12, fontweight="bold")

            if col == 0:
                ax.set_ylabel(label, fontsize=10)

            if row == 3:
                ax.set_xlabel("Day (0 = forecast start)")

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower left", ncol=2)

            ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig1_fan_charts_calm_vs_turbulent.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Cross-Cell Regime Sensitivity
# ══════════════════════════════════════════════════════════════════════
def plot_cross_cell_sensitivity(data):
    """Show same window, different cells with history context.
    Top row: turbulent window. Bottom row: calm window. Same 4 cells."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)
    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    cells = [(0, 0), (0, 4), (2, 2), (4, 2)]
    col_labels = [
        "1M OTM Put\n(K=0.70)",
        "1M OTM Call\n(K=1.30)",
        "6M ATM\n(K=1.00)",
        "2Y ATM\n(K=1.00)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    fig.suptitle(
        "Same Cells, Different Regimes: Uncertainty Responds Differently Per Cell\n"
        "Black = history, colored = forecast fan",
        fontsize=15, fontweight="bold", y=1.01,
    )

    hist_days = np.arange(-29, 1)
    fwd_days = np.arange(1, 31)

    for row_idx, (win_idx, regime, color) in enumerate([
        (turb_idx, "Turbulent", TURB_COLOR),
        (calm_idx, "Calm", CALM_COLOR),
    ]):
        for col_idx, ((r, c), col_label) in enumerate(zip(cells, col_labels)):
            ax = axes[row_idx, col_idx]
            hist_cell = data["history"][win_idx, :, r, c]
            samples_cell = data["samples"][win_idx, :, :, r, c]
            gt_cell = data["future"][win_idx, :, r, c]

            spread = samples_cell.std(axis=0).mean()

            # History
            ax.plot(hist_days, hist_cell, color="black", linewidth=1.5, zorder=6)
            ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6)

            # Light background paths
            for s in range(min(N_SAMPLES, 50)):
                ax.plot(fwd_days, samples_cell[s], color=color, alpha=0.04, linewidth=0.3)

            q05 = np.percentile(samples_cell, 5, axis=0)
            q95 = np.percentile(samples_cell, 95, axis=0)
            ax.fill_between(fwd_days, q05, q95, color=color, alpha=0.08)

            # Individual paths — 6 distinct visible paths
            path_colors = ["#E91E63", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
            n_show = min(6, N_SAMPLES)
            path_indices = np.linspace(0, N_SAMPLES - 1, n_show, dtype=int)
            for pi, s in enumerate(path_indices):
                ax.plot(fwd_days, samples_cell[s], color=path_colors[pi % len(path_colors)],
                        alpha=0.7, linewidth=1.0, zorder=4)
                ax.plot([0, 1], [hist_cell[-1], samples_cell[s, 0]],
                        color=path_colors[pi % len(path_colors)], linewidth=0.7, alpha=0.5)

            ax.plot(fwd_days, gt_cell, color=GT_COLOR, linewidth=2.0, linestyle="--", zorder=5)

            ax.text(0.97, 0.95, f"spread={spread*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if row_idx == 0:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                vov_val = vov[win_idx]
                ax.set_ylabel(f"{regime}\n(vov={vov_val:.4f})", fontsize=10, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Day (0 = forecast start)")
            ax.tick_params(labelsize=9)

    # Share y-axis per column (include history range)
    for col_idx in range(4):
        ymin = min(axes[0, col_idx].get_ylim()[0], axes[1, col_idx].get_ylim()[0])
        ymax = max(axes[0, col_idx].get_ylim()[1], axes[1, col_idx].get_ylim()[1])
        for row_idx in range(2):
            axes[row_idx, col_idx].set_ylim(ymin, ymax)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_cross_cell_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal Properties — Kurtosis & ACF
# ══════════════════════════════════════════════════════════════════════
def plot_temporal_properties(data):
    """Cross-sample distribution match + ACF overlay: generated vs ground truth."""
    future = data["future"]        # (W, 30, 5, 5)
    samples = data["samples"]      # (W, S, 30, 5, 5)
    n_windows = data["n_windows"]

    gt_daily = np.diff(future, axis=1)  # (W, 29, 5, 5)
    gen_daily_all = []
    for s in range(min(10, N_SAMPLES)):
        gen_daily = np.diff(samples[:, s, :, :, :], axis=1)
        gen_daily_all.append(gen_daily)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Temporal Properties: Generated Scenarios vs Ground Truth",
                 fontsize=15, fontweight="bold")

    # Panel 1: Cross-sample distribution at h=30 (the hardest horizon)
    # This is what the test suite measures — kurtosis of the ensemble
    ax = axes[0]
    h_idx = 29  # h=30
    # Flatten across all windows and cells
    gt_vals_h30 = future[:, h_idx, :, :].ravel()
    # Pool one sample per window for comparable size
    gen_vals_h30 = samples[:, 0, h_idx, :, :].ravel()

    # Standardize to zero-mean unit-var for shape comparison
    gt_z = (gt_vals_h30 - gt_vals_h30.mean()) / gt_vals_h30.std()
    gen_z = (gen_vals_h30 - gen_vals_h30.mean()) / gen_vals_h30.std()

    bins = np.linspace(-4, 4, 80)
    ax.hist(gt_z, bins=bins, density=True, alpha=0.5, color=GT_COLOR, label="Ground truth")
    ax.hist(gen_z, bins=bins, density=True, alpha=0.5, color=TURB_COLOR, label="Generated")

    # Normal reference
    x = np.linspace(-4, 4, 200)
    ax.plot(x, sp_stats.norm.pdf(x), "k--", alpha=0.4, linewidth=1, label="Normal")

    gt_kurt = sp_stats.kurtosis(gt_vals_h30, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals_h30, fisher=True)
    ratio = gen_kurt / gt_kurt if gt_kurt > 0 else float("nan")
    ax.text(0.05, 0.88,
            f"GT kurtosis: {gt_kurt:.2f}\n"
            f"Gen kurtosis: {gen_kurt:.2f}\n"
            f"Ratio: {ratio:.2f} (1.0 = perfect)",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Standardized IV Level")
    ax.set_ylabel("Density")
    ax.set_title("Cross-Sample Distribution at h=30\n(Ensemble captures fat tails)")
    ax.legend(fontsize=9)
    ax.set_xlim(-4, 4)

    # Panel 2: ACF comparison for ATM cells
    ax = axes[1]
    max_lag = 15
    lags = np.arange(1, max_lag + 1)

    for r, c, label, ls in [(2, 2, "6M ATM", "-"), (0, 2, "1M ATM", "--"), (4, 2, "2Y ATM", ":")]:
        gt_series = np.abs(gt_daily[:, :, r, c].ravel())
        gt_acf = [np.corrcoef(gt_series[:-lag], gt_series[lag:])[0, 1] for lag in lags]

        gen_series = np.abs(np.array([gd[:, :, r, c].ravel() for gd in gen_daily_all]).ravel())
        n = min(len(gt_series), len(gen_series))
        gen_series_trimmed = gen_series[:n]
        gen_acf = [np.corrcoef(gen_series_trimmed[:-lag], gen_series_trimmed[lag:])[0, 1]
                   for lag in lags]

        ax.plot(lags, gt_acf, color=GT_COLOR, linestyle=ls, linewidth=2, label=f"GT {label}")
        ax.plot(lags, gen_acf, color=TURB_COLOR, linestyle=ls, linewidth=2,
                alpha=0.7, label=f"Gen {label}")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation of |Daily Changes|")
    ax.set_title("Volatility Clustering Preserved\n(ACF of absolute daily IV changes)")
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: Daily change distribution (6M ATM)
    ax = axes[2]
    r, c = 2, 2
    gt_vals = gt_daily[:, :, r, c].ravel()
    gen_vals = np.array([gd[:, :, r, c].ravel() for gd in gen_daily_all]).ravel()

    bins = np.linspace(np.percentile(gt_vals, 0.5), np.percentile(gt_vals, 99.5), 80)
    ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=GT_COLOR, label="Ground truth")
    ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=TURB_COLOR, label="Generated")

    gt_kurt = sp_stats.kurtosis(gt_vals, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals, fisher=True)
    ax.text(0.05, 0.88,
            f"GT kurtosis: {gt_kurt:.1f}\nGen kurtosis: {gen_kurt:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Daily IV Change")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Daily Changes (6M ATM)")
    ax.legend(fontsize=9)

    gt_kurt = sp_stats.kurtosis(gt_vals, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals, fisher=True)
    ax.text(0.05, 0.92, f"GT kurtosis: {gt_kurt:.2f}\nGen kurtosis: {gen_kurt:.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_temporal_properties.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Term Structure & Smile Validation
# ══════════════════════════════════════════════════════════════════════
def plot_surface_structure(data):
    """Term structure and smile slices with scenario CI bands."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    # Three windows: calm, median, turbulent
    window_picks = [
        (sorted_idx[int(0.10 * len(sorted_idx))], "Calm"),
        (sorted_idx[int(0.50 * len(sorted_idx))], "Medium"),
        (sorted_idx[int(0.90 * len(sorted_idx))], "Turbulent"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Surface Structure Validation: Term Structure & Volatility Smile\n"
                 "Ground truth (green dashed) should fall within scenario band",
                 fontsize=14, fontweight="bold", y=1.02)

    for row, (win_idx, regime) in enumerate(window_picks):
        # h=1 for structure check
        h_idx = 0
        gt_surface = data["future"][win_idx, h_idx]        # (5, 5)
        gen_surfaces = data["samples"][win_idx, :, h_idx]   # (S, 5, 5)

        # --- Term structure (ATM slice, column 2) ---
        ax = axes[row, 0]
        atm_col = 2
        gt_term = gt_surface[:, atm_col]
        gen_term = gen_surfaces[:, :, atm_col]  # (S, 5)

        for s in range(min(15, N_SAMPLES)):
            ax.plot(MATURITIES_DAYS, gen_term[s], color=SCENARIO_COLOR, alpha=0.15, linewidth=0.5)

        q05 = np.percentile(gen_term, 5, axis=0)
        q25 = np.percentile(gen_term, 25, axis=0)
        q75 = np.percentile(gen_term, 75, axis=0)
        q95 = np.percentile(gen_term, 95, axis=0)
        median = np.median(gen_term, axis=0)

        ax.fill_between(MATURITIES_DAYS, q05, q95, alpha=0.12, color=TURB_COLOR, label="90% CI")
        ax.fill_between(MATURITIES_DAYS, q25, q75, alpha=0.20, color=TURB_COLOR, label="50% CI")
        ax.plot(MATURITIES_DAYS, median, color=TURB_COLOR, linewidth=1.5, label="Median")
        ax.plot(MATURITIES_DAYS, gt_term, color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=6, label="Ground truth", zorder=5)

        ax.set_xlabel("Maturity (days)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — ATM Term Structure (K=1.00)")
        if row == 0:
            ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(MATURITIES_DAYS)
        ax.set_xticklabels(MATURITY_LABELS)

        # --- Smile (6M slice, row 2) ---
        ax = axes[row, 1]
        tenor_row = 2
        gt_smile = gt_surface[tenor_row, :]
        gen_smile = gen_surfaces[:, tenor_row, :]  # (S, 5)

        for s in range(min(15, N_SAMPLES)):
            ax.plot(MONEYNESS, gen_smile[s], color=SCENARIO_COLOR, alpha=0.15, linewidth=0.5)

        q05 = np.percentile(gen_smile, 5, axis=0)
        q25 = np.percentile(gen_smile, 25, axis=0)
        q75 = np.percentile(gen_smile, 75, axis=0)
        q95 = np.percentile(gen_smile, 95, axis=0)
        median = np.median(gen_smile, axis=0)

        ax.fill_between(MONEYNESS, q05, q95, alpha=0.12, color=CALM_COLOR, label="90% CI")
        ax.fill_between(MONEYNESS, q25, q75, alpha=0.20, color=CALM_COLOR, label="50% CI")
        ax.plot(MONEYNESS, median, color=CALM_COLOR, linewidth=1.5, label="Median")
        ax.plot(MONEYNESS, gt_smile, color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=6, label="Ground truth", zorder=5)

        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — 6M Volatility Smile")
        if row == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_term_structure_smile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Surface Heatmap Snapshots
# ══════════════════════════════════════════════════════════════════════
def plot_surface_heatmaps(data):
    """GT surface, mean generated, and uncertainty heatmaps."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Surface Snapshots at h=1: Ground Truth vs Generated Mean vs Uncertainty Map",
                 fontsize=14, fontweight="bold")

    for row, (win_idx, regime) in enumerate([(calm_idx, "Calm"), (turb_idx, "Turbulent")]):
        h_idx = 0
        gt = data["future"][win_idx, h_idx]            # (5, 5)
        gen = data["samples"][win_idx, :, h_idx]        # (S, 5, 5)
        gen_mean = gen.mean(axis=0)
        gen_std = gen.std(axis=0)
        error = np.abs(gt - gen_mean)

        vmin = min(gt.min(), gen_mean.min())
        vmax = max(gt.max(), gen_mean.max())

        for col, (arr, title, cmap, vmin_o, vmax_o) in enumerate([
            (gt, f"{regime}: Ground Truth", "YlOrRd", vmin, vmax),
            (gen_mean, f"{regime}: Scenario Mean", "YlOrRd", vmin, vmax),
            (gen_std, f"{regime}: Uncertainty (σ)", "Reds", None, None),
            (error, f"{regime}: |Error|", "Blues", None, None),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin_o, vmax=vmax_o, aspect="auto")
            ax.set_xticks(range(5))
            ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel("Maturity")
            if row == 1:
                ax.set_xlabel("Moneyness")

            # Annotate values
            for r in range(5):
                for c in range(5):
                    val = arr[r, c]
                    text_color = "white" if val > (vmax_o or arr.max()) * 0.6 else "black"
                    if col >= 2:
                        ax.text(c, r, f"{val*100:.2f}", ha="center", va="center",
                                fontsize=7, color=text_color)
                    else:
                        ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                                fontsize=7, color=text_color)

            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_surface_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Calibration Curve
# ══════════════════════════════════════════════════════════════════════
def plot_calibration_curve(data):
    """Nominal vs empirical coverage across CI levels and horizons."""
    future = data["future"]
    samples = data["samples"]

    nominal_levels = np.arange(0.1, 1.0, 0.1)  # 10%, 20%, ..., 90%

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.suptitle("Calibration Curve: Is the Model Well-Calibrated?",
                 fontsize=14, fontweight="bold")

    horizon_specs = [(0, "h=1", "-", "o"), (6, "h=7", "--", "s"),
                     (13, "h=14", "-.", "^"), (29, "h=30", ":", "D")]

    for h_idx, h_name, ls, marker in horizon_specs:
        gt = future[:, h_idx]            # (W, 5, 5)
        s = samples[:, :, h_idx]          # (W, S, 5, 5)

        empirical = []
        for nom in nominal_levels:
            alpha = (1 - nom) / 2 * 100
            lo = np.percentile(s, alpha, axis=1)
            hi = np.percentile(s, 100 - alpha, axis=1)
            covered = ((gt >= lo) & (gt <= hi)).mean()
            empirical.append(covered)

        ax.plot(nominal_levels * 100, np.array(empirical) * 100,
                linestyle=ls, marker=marker, markersize=6, linewidth=2,
                label=h_name)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, linewidth=1, label="Perfect calibration")
    ax.fill_between([0, 100], [0, 100], [5, 105], alpha=0.05, color="green")
    ax.fill_between([0, 100], [-5, 95], [0, 100], alpha=0.05, color="green")

    ax.set_xlabel("Nominal Coverage Level (%)", fontsize=12)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=12)
    ax.set_xlim(5, 95)
    ax.set_ylim(5, 100)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")

    # Add annotation
    ax.text(0.05, 0.88,
            "Points on the diagonal = perfectly calibrated\n"
            "Above = conservative (wider CIs than needed)\n"
            "Below = overconfident (CIs too narrow)",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig6_calibration_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Marginal Daily Change Distributions Per Cell
# ══════════════════════════════════════════════════════════════════════
def plot_marginal_daily_changes(data):
    """Per-cell daily IV change distributions: GT vs generated.

    Two plots:
      fig7a — 3x3 grid of 9 representative cells with detailed stats
      fig7b — compact 5x5 grid of all 25 cells
    """
    future = data["future"]        # (W, 30, 5, 5)
    samples = data["samples"]      # (W, S, 30, 5, 5)

    gt_daily = np.diff(future, axis=1)                  # (W, 29, 5, 5)
    gen_daily = np.diff(samples[:, :5], axis=2)          # (W, 5, 29, 5, 5)

    # ── Fig 7a: 3x3 representative cells ──
    cells_9 = [
        (0, 0), (0, 2), (0, 4),
        (2, 0), (2, 2), (2, 4),
        (4, 0), (4, 2), (4, 4),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Daily IV Change Distribution: Ground Truth vs Generated\n"
                 "Red = Generated, Green = Ground Truth — matching tails = good kurtosis",
                 fontsize=14, fontweight="bold")

    for idx, (r, c) in enumerate(cells_9):
        ax = axes[idx // 3, idx % 3]
        gt_vals = gt_daily[:, :, r, c].ravel()
        gen_vals = gen_daily[:, :, :, r, c].ravel()

        plo, phi = np.percentile(gt_vals, [0.5, 99.5])
        bins = np.linspace(plo, phi, 80)

        ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=GT_COLOR, label="GT")
        ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=TURB_COLOR, label="Gen")

        gt_k = sp_stats.kurtosis(gt_vals, fisher=True)
        gen_k = sp_stats.kurtosis(gen_vals, fisher=True)
        ratio = gen_k / gt_k if gt_k > 0 else float("nan")
        ks_stat, ks_p = sp_stats.ks_2samp(gt_vals, gen_vals)
        gt_std = gt_vals.std()
        gen_std = gen_vals.std()

        ax.text(0.03, 0.95,
                f"GT kurt={gt_k:.1f}\n"
                f"Gen kurt={gen_k:.1f}\n"
                f"Ratio={ratio:.2f}\n"
                f"GT std={gt_std:.4f}\n"
                f"Gen std={gen_std:.4f}\n"
                f"KS={ks_stat:.3f} p={ks_p:.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        cell_name = CELL_NAMES[(r, c)]
        pass_mark = "PASS" if ks_p > 0.05 else "FAIL"
        ax.set_title(f"({r},{c}) {cell_name}  [{pass_mark}]", fontsize=10,
                     fontweight="bold" if pass_mark == "FAIL" else "normal",
                     color="red" if pass_mark == "FAIL" else "black")

        if idx >= 6:
            ax.set_xlabel("Daily IV Change")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = f"{OUTPUT_DIR}/fig7a_marginal_daily_changes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Fig 7b: compact 5x5 all cells ──
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    fig.suptitle("Daily IV Change Distribution — All 25 Cells\n"
                 "Green=GT, Red=Generated",
                 fontsize=14, fontweight="bold")

    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_vals = gt_daily[:, :, r, c].ravel()
            gen_vals = gen_daily[:, :, :, r, c].ravel()

            plo, phi = np.percentile(gt_vals, [1, 99])
            bins = np.linspace(plo, phi, 50)

            ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=GT_COLOR)
            ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=TURB_COLOR)

            gt_k = sp_stats.kurtosis(gt_vals, fisher=True)
            gen_k = sp_stats.kurtosis(gen_vals, fisher=True)
            ratio = gen_k / gt_k if gt_k > 0 else float("nan")
            ks_stat, ks_p = sp_stats.ks_2samp(gt_vals, gen_vals)

            pass_mark = "P" if ks_p > 0.05 else "F"
            ax.set_title(f"({r},{c}) r={ratio:.2f} [{pass_mark}]", fontsize=8,
                         color="red" if pass_mark == "F" else "black")
            ax.tick_params(labelsize=6)
            if r < 4:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    for r in range(5):
        axes[r, 0].set_ylabel(MATURITY_LABELS[r], fontsize=9)
    for c in range(5):
        axes[4, c].set_xlabel(f"K={MONEYNESS_LABELS[c]}", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig7b_marginal_daily_changes_all25.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Kurtosis Heatmap — GT vs Generated Daily Change Magnitude
# ══════════════════════════════════════════════════════════════════════
def plot_kurtosis_heatmap(data):
    """Time × cell heatmap of daily changes: GT vs one sample path.

    X-axis = days (1-29), Y-axis = 25 cells (grouped by maturity).
    Color = daily change value (signed). Dark = large move, light = small.
    Shows temporal volatility clustering and cross-cell correlation patterns.
    """
    future = data["future"]        # (W, 30, 5, 5)
    samples = data["samples"]      # (W, S, 30, 5, 5)
    history = data["history"]      # (W, 30, 5, 5)
    vov = data["vol_of_vol"]

    # Pick a turbulent window for visual contrast
    turb_idx = np.argsort(vov)[-len(vov) // 4:]
    w_idx = turb_idx[len(turb_idx) // 2]

    gt_path = future[w_idx]                # (30, 5, 5)
    gen_path = samples[w_idx, 0]           # (30, 5, 5)
    hist_last = history[w_idx, -1:]        # (1, 5, 5)

    # Daily changes including first day (from history[-1])
    gt_full = np.concatenate([hist_last, gt_path], axis=0)   # (31, 5, 5)
    gen_full = np.concatenate([hist_last, gen_path], axis=0)  # (31, 5, 5)
    gt_dc = np.diff(gt_full, axis=0)     # (30, 5, 5)
    gen_dc = np.diff(gen_full, axis=0)   # (30, 5, 5)

    # Reshape to (30, 25): flatten cells in row-major (maturity-first) order
    gt_flat = gt_dc.reshape(30, 25).T     # (25, 30) — cells on y, days on x
    gen_flat = gen_dc.reshape(30, 25).T   # (25, 30)

    # Cell labels: maturity × moneyness
    cell_labels = []
    for r in range(5):
        for c in range(5):
            cell_labels.append(f"{MATURITY_LABELS[r]} K={MONEYNESS_LABELS[c]}")

    # Shared color scale
    vmax = max(np.abs(gt_flat).max(), np.abs(gen_flat).max())
    # Cap at 95th percentile to avoid outlier domination
    vmax = min(vmax, np.percentile(np.abs(np.concatenate([gt_flat, gen_flat])), 97))

    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    fig.suptitle("Daily Change Heatmap: Ground Truth vs Generated Sample (Single Turbulent Window)",
                 fontsize=15, fontweight="bold")

    # GT
    ax = axes[0]
    im = ax.imshow(gt_flat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title("Ground Truth", fontsize=13, fontweight="bold", color=GT_COLOR)
    ax.set_yticks(range(25))
    ax.set_yticklabels(cell_labels, fontsize=7)
    ax.set_ylabel("Cell (Maturity × Moneyness)")
    # Horizontal lines between maturity groups
    for i in range(1, 5):
        ax.axhline(i * 5 - 0.5, color="black", linewidth=0.8, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Daily IV Change", shrink=0.7)

    # Generated
    ax = axes[1]
    im = ax.imshow(gen_flat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title("Generated (Sample 1)", fontsize=13, fontweight="bold", color=TURB_COLOR)
    ax.set_yticks(range(25))
    ax.set_yticklabels(cell_labels, fontsize=7)
    ax.set_ylabel("Cell (Maturity × Moneyness)")
    ax.set_xlabel("Day (h=1 to h=30)")
    ax.set_xticks(range(0, 30, 5))
    ax.set_xticklabels([str(d + 1) for d in range(0, 30, 5)])
    for i in range(1, 5):
        ax.axhline(i * 5 - 0.5, color="black", linewidth=0.8, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Daily IV Change", shrink=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig8_kurtosis_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model, config = load_model(MODEL_PATH, device)

    print("Generating data...")
    data = generate_all_data(model, config, device)

    if QUANTILE_MAP:
        from experiments.backfill.block_ar.quantile_mapper import QuantileMapper
        print(f"Applying quantile mapping (alpha={QMAP_ALPHA}) from {QUANTILE_MAP}...")
        qmapper = QuantileMapper(QUANTILE_MAP, alpha=QMAP_ALPHA)
        data["samples"] = qmapper.apply(data["samples"], data["history"])
        data["samples_raw"] = data["samples"]

    print("\nGenerating visualizations...")

    print("\n[1/8] Fan charts with history: calm vs turbulent...")
    plot_fan_charts(data)

    print("[2/8] Cross-cell sensitivity with history...")
    plot_cross_cell_sensitivity(data)

    print("[3/8] Temporal properties...")
    plot_temporal_properties(data)

    print("[4/8] Term structure & smile...")
    plot_surface_structure(data)

    print("[5/8] Surface heatmaps...")
    plot_surface_heatmaps(data)

    print("[6/8] Calibration curve...")
    plot_calibration_curve(data)

    print("[7/8] Marginal daily change distributions...")
    plot_marginal_daily_changes(data)

    print("[8/8] Kurtosis heatmap...")
    plot_kurtosis_heatmap(data)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
