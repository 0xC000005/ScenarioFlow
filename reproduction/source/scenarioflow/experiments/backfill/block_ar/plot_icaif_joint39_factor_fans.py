#!/usr/bin/env python
"""Generate the compact joint-39 non-IV fan chart used by the ICAIF paper.

The calm and turbulent windows are selected from conditioning histories only.
For each non-IV factor, the model's causal local scale is ranked across the
validation split; the mean rank across factors gives a unit-free activity
score.  The displayed windows are nearest the 15th and 85th percentiles of
that score.  Realized futures never enter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)


DEFAULT_CHECKPOINT = (
    "models/backfill/"
    "734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
DEFAULT_OUTPUT = (
    "paper/camera_ready/icaif/illustrations/"
    "joint39_noniv_common_history_fans.png"
)

FACTOR_SPECS = (
    ("factor:spx", "SPX"),
    ("factor:vix", "VIX"),
    ("factor:us10y", "US 10Y"),
)

HISTORY_COLOR = "#202124"
CALM_COLOR = "#2864B7"
TURBULENT_COLOR = "#C64B28"
FUTURE_COLOR = "#14866D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7345)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--calm_quantile", type=float, default=0.15)
    parser.add_argument("--turbulent_quantile", type=float, default=0.85)

    # These values reproduce the evaluation contract in the paper.
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform",
        choices=["log_level", "bounded_logit"],
        default="bounded_logit",
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument(
        "--drift_feature_mode", choices=["none", "ewma_mean"], default="none"
    )
    return parser.parse_args()


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return columnwise ranks in [0, 1], preserving the input shape."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("activity inputs contain non-finite values")
    ranks = np.empty_like(values)
    grid = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="mergesort")
        ranks[order, column] = grid
    return ranks


def select_common_histories(
    factor_scale: np.ndarray,
    *,
    calm_quantile: float,
    turbulent_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select common calm/turbulent windows from history-derived local scales."""
    if not 0.0 < calm_quantile < turbulent_quantile < 1.0:
        raise ValueError("selection quantiles must satisfy 0 < calm < turbulent < 1")
    activity_score = percentile_ranks(np.log(np.maximum(factor_scale, 1e-12))).mean(
        axis=1
    )
    targets = np.quantile(activity_score, [calm_quantile, turbulent_quantile])
    calm_index = int(np.argmin(np.abs(activity_score - targets[0])))
    turbulent_distance = np.abs(activity_score - targets[1])
    turbulent_distance[calm_index] = np.inf
    turbulent_index = int(np.argmin(turbulent_distance))
    return np.asarray([calm_index, turbulent_index], dtype=np.int64), activity_score


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def generate_selected_samples(
    model: Any,
    selected: np.ndarray,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    increments = model.sample_batched(
        torch.from_numpy(history_level[selected]).to(device),
        torch.from_numpy(history_norm[selected]).to(device),
        torch.from_numpy(center[selected]).to(device),
        torch.from_numpy(scale[selected]).to(device),
        drift_feature=torch.from_numpy(drift_feature[selected]).to(device),
        n_samples=int(samples),
        n_steps=30,
        chunk_size=int(chunk_size),
        temperature=1.0,
    )
    return reconstruct_state_from_increments(
        history_raw[selected, -1, :],
        increments.detach().cpu().numpy(),
        specs,
    )


def plot_fans(
    output_path: Path,
    histories: np.ndarray,
    futures: np.ndarray,
    samples: np.ndarray,
    spec_names: list[str],
    end_dates: list[str],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 2.45), sharex=True)
    history_days = np.arange(-29, 1)
    future_days = np.arange(1, 31)
    colors = (CALM_COLOR, TURBULENT_COLOR)
    row_labels = ("Calm history", "Turbulent history")
    path_indices = (3, 41)

    legend_handles: dict[str, Any] = {}
    for column, (factor_name, display_name) in enumerate(FACTOR_SPECS):
        factor_index = spec_names.index(factor_name)
        for row in range(2):
            ax = axes[row, column]
            color = colors[row]
            history = histories[row, :, factor_index]
            future = futures[row, :, factor_index]
            factor_samples = samples[row, :, :, factor_index]
            q10, q25, q50, q75, q90 = np.percentile(
                factor_samples, [10, 25, 50, 75, 90], axis=0
            )

            line = ax.plot(
                history_days,
                history,
                color=HISTORY_COLOR,
                linewidth=1.25,
                label="History",
                zorder=5,
            )[0]
            legend_handles.setdefault("History", line)
            band90 = ax.fill_between(
                future_days,
                q10,
                q90,
                color=color,
                alpha=0.13,
                linewidth=0,
                label="10–90% band",
                zorder=1,
            )
            legend_handles.setdefault("10–90% band", band90)
            band50 = ax.fill_between(
                future_days,
                q25,
                q75,
                color=color,
                alpha=0.25,
                linewidth=0,
                label="IQR",
                zorder=2,
            )
            legend_handles.setdefault("IQR", band50)
            for path_number, path_index in enumerate(path_indices):
                scenario_line = ax.plot(
                    future_days,
                    factor_samples[path_index],
                    color=color,
                    alpha=0.38,
                    linewidth=0.65,
                    label="Joint sample paths" if path_number == 0 else None,
                    zorder=3,
                )[0]
                if path_number == 0:
                    legend_handles.setdefault("Joint sample paths", scenario_line)
            median = ax.plot(
                future_days,
                q50,
                color=color,
                linewidth=1.35,
                label="Median",
                zorder=4,
            )[0]
            legend_handles.setdefault("Median", median)
            realized = ax.plot(
                future_days,
                future,
                color=FUTURE_COLOR,
                linestyle=(0, (3, 2)),
                linewidth=1.15,
                label="Realized future",
                zorder=5,
            )[0]
            legend_handles.setdefault("Realized future", realized)

            ax.axvline(0.5, color="#8A8A8A", linestyle=":", linewidth=0.75)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0.015)
            if row == 0:
                ax.set_title(display_name, fontweight="semibold", pad=2.0)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontweight="semibold", labelpad=3.5)
                ax.text(
                    0.02,
                    0.94,
                    f"through {end_dates[row]}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.1,
                    color="#5F6368",
                )
    ordered_labels = (
        "History",
        "10–90% band",
        "IQR",
        "Joint sample paths",
        "Median",
        "Realized future",
    )
    fig.legend(
        [legend_handles[label] for label in ordered_labels],
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=6,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        top=0.915,
        bottom=0.175,
        wspace=0.25,
        hspace=0.12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_path.with_name(f"{output_path.stem}_selection.json")
    )

    model, payload = load_model(str(checkpoint_path), device)
    model.eval()
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = build_val_block(args, payload)
    spec_names = [spec.name for spec in specs]
    expected_names = [spec["name"] for spec in payload["state_specs"]]
    if spec_names != expected_names or len(spec_names) != 39:
        raise RuntimeError("rebuilt validation panel does not match the joint-39 checkpoint")

    selected, activity_score = select_common_histories(
        scale[:, int(args.iv_count) :],
        calm_quantile=float(args.calm_quantile),
        turbulent_quantile=float(args.turbulent_quantile),
    )
    generated = generate_selected_samples(
        model,
        selected,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        samples=int(args.samples),
        chunk_size=int(args.chunk_size),
        device=device,
    )

    _, _, dates = load_aligned_iv_factor_panel()
    raw_indices = block.indices[selected]
    history_end_positions = raw_indices + int(payload["config"]["history_len"]) - 1
    future_start_positions = history_end_positions + 1
    future_end_positions = (
        future_start_positions + int(payload["config"]["future_len"]) - 1
    )
    end_dates = [str(dates[position].date()) for position in history_end_positions]
    plot_fans(
        output_path,
        block.history_state[selected],
        block.future_state[selected],
        generated,
        spec_names,
        end_dates,
    )

    manifest = {
        "figure": str(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "state_scope": payload.get("state_scope"),
        "panel_channels": len(spec_names),
        "factor_channels": len(spec_names) - int(args.iv_count),
        "factors_plotted": [name for name, _ in FACTOR_SPECS],
        "validation_windows": int(history_raw.shape[0]),
        "history_days": int(payload["config"]["history_len"]),
        "future_days": int(payload["config"]["future_len"]),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "device": str(device),
        "selection_rule": (
            "history only: rank each non-IV channel's causal local scale across "
            "validation windows, average the 14 percentile ranks, and select "
            "windows nearest the requested calm/turbulent score quantiles"
        ),
        "realized_future_used_for_selection": False,
        "windows": {
            label: {
                "validation_index": int(selected[position]),
                "raw_panel_index": int(raw_indices[position]),
                "activity_score": float(activity_score[selected[position]]),
                "activity_score_percentile": float(
                    np.mean(activity_score <= activity_score[selected[position]])
                ),
                "history_start": str(dates[raw_indices[position]].date()),
                "history_end": end_dates[position],
                "future_start": str(dates[future_start_positions[position]].date()),
                "future_end": str(dates[future_end_positions[position]].date()),
            }
            for position, label in enumerate(("calm", "turbulent"))
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
