"""Portable recipes for the numerical figures used in the papers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..artifacts import ArtifactError, ArtifactResolver

MONEYNESS_LABELS = ("0.70", "0.85", "1.00", "1.15", "1.30")
MATURITY_LABELS = ("1M", "3M", "6M", "1Y", "2Y")
FACTOR_SPECS = (
    ("factor:spx", "SPX"),
    ("factor:vix", "VIX"),
    ("factor:us10y", "US 10Y"),
)

CALM_COLOR = "#2864B7"
TURBULENT_COLOR = "#C64B28"
GROUND_TRUTH_COLOR = "#2E7D32"
HISTORY_COLOR = "#202124"
FUTURE_COLOR = "#14866D"
IV_CALM_COLOR = "#2196F3"
IV_TURBULENT_COLOR = "#F44336"


class PaperPlotError(ValueError):
    """Raised when plot inputs or an output path are invalid."""


def _plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError as error:
        raise PaperPlotError(
            "paper plotting requires the scenarioflow[paper] extra"
        ) from error
    return plt, stats


def _output_path(path: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    output = Path(os.path.abspath(supplied))
    current = Path(output.anchor)
    for component in output.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise PaperPlotError("plot output path or an ancestor is a symlink")
    repository = Path(__file__).resolve().parents[3]
    protected = (repository / "paper_assets" / "figures").resolve()
    if output == protected or protected in output.parents:
        raise PaperPlotError("refusing to overwrite frozen paper figures")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_dir():
            raise IsADirectoryError
    except OSError as error:
        raise PaperPlotError("plot output path is unusable") from error
    return output


def _iv_arrays(
    history: np.ndarray, future: np.ndarray, samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_array = np.asarray(history, dtype=np.float64)
    future_array = np.asarray(future, dtype=np.float64)
    sample_array = np.asarray(samples, dtype=np.float64)
    if (
        history_array.ndim != 4
        or future_array.ndim != 4
        or sample_array.ndim != 5
        or history_array.shape[1:] != (30, 5, 5)
        or future_array.shape != history_array.shape
        or sample_array.shape[0] != history_array.shape[0]
        or sample_array.shape[2:] != (30, 5, 5)
        or sample_array.shape[1] < 1
        or not all(
            np.isfinite(array).all()
            for array in (history_array, future_array, sample_array)
        )
    ):
        raise PaperPlotError(
            "IV arrays must be finite [windows,30,5,5] histories/futures and "
            "[windows,samples,30,5,5] samples"
        )
    return history_array, future_array, sample_array


def _future_deltas(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    previous = np.concatenate([history[:, -1:], future[:, :-1]], axis=1)
    return future - previous


def _sample_deltas(history: np.ndarray, samples: np.ndarray) -> np.ndarray:
    previous = np.concatenate(
        (
            np.repeat(history[:, None, -1:], samples.shape[1], axis=1),
            samples[:, :, :-1],
        ),
        axis=2,
    )
    return samples - previous


def _encode_history_levels(history_state: np.ndarray, specs: list[Any]) -> np.ndarray:
    """Map raw states to the level coordinates used to train the checkpoint."""
    from ..state import encode_state

    return encode_state(history_state, specs).astype(np.float32)


def _prepare_iv_validation_block(data_dir: Path) -> Any:
    """Build the published validation windows for the frozen IV-only model."""
    from ..data import load_aligned_iv_factor_panel, official_train_val_indices
    from ..evaluation.current_panel import PUBLISHED_VALIDATION_PROTOCOL
    from ..state import build_increment_coordinate_block

    protocol = PUBLISHED_VALIDATION_PROTOCOL
    panel, columns, _dates = load_aligned_iv_factor_panel(data_dir=data_dir)
    _train_indices, validation_indices = official_train_val_indices(
        test_start=protocol.test_start,
        val_size=protocol.val_size,
        history_len=protocol.history_len,
        future_len=protocol.future_len,
    )
    return build_increment_coordinate_block(
        panel[:, : protocol.iv_count],
        columns[: protocol.iv_count],
        validation_indices,
        history_len=protocol.history_len,
        future_len=protocol.future_len,
        iv_count=protocol.iv_count,
        iv_transform="log_level",
    )


def plot_iv_marginal_daily_changes(
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot boundary-inclusive daily IV changes for nine representative cells."""
    plt, stats = _plotting()
    history, future, samples = _iv_arrays(history, future, samples)
    truth = _future_deltas(history, future)
    generated = _sample_deltas(history, samples[:, : min(5, samples.shape[1])])
    cells = (
        (0, 0),
        (0, 2),
        (0, 4),
        (2, 0),
        (2, 2),
        (2, 4),
        (4, 0),
        (4, 2),
        (4, 4),
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        "Daily IV P&L Distribution: Ground Truth vs Generated\n"
        "Includes h1 move from the last observed history level.",
        fontsize=14,
        fontweight="bold",
    )
    for index, (row, column) in enumerate(cells):
        axis = axes[index // 3, index % 3]
        truth_values = truth[:, :, row, column].ravel()
        generated_values = generated[:, :, :, row, column].ravel()
        span = max(
            np.percentile(np.abs(truth_values), 99.5),
            np.percentile(np.abs(generated_values), 99.5),
            1e-8,
        )
        bins = np.linspace(-span, span, 80)
        axis.hist(
            truth_values,
            bins=bins,
            density=True,
            alpha=0.5,
            color=GROUND_TRUTH_COLOR,
            label="GT",
        )
        axis.hist(
            generated_values,
            bins=bins,
            density=True,
            alpha=0.5,
            color=IV_TURBULENT_COLOR,
            label="Gen",
        )
        truth_kurtosis = float(stats.kurtosis(truth_values, fisher=True))
        generated_kurtosis = float(stats.kurtosis(generated_values, fisher=True))
        ratio = (
            generated_kurtosis / truth_kurtosis
            if truth_kurtosis > 0.0
            else float("nan")
        )
        ks_statistic = float(stats.ks_2samp(truth_values, generated_values).statistic)
        axis.text(
            0.03,
            0.95,
            f"GT kurt={truth_kurtosis:.1f}\n"
            f"Gen kurt={generated_kurtosis:.1f}\n"
            f"Ratio={ratio:.2f}\nKS={ks_statistic:.3f}",
            transform=axis.transAxes,
            fontsize=7,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
        )
        axis.set_title(
            f"({row},{column}) {MATURITY_LABELS[row]} / K={MONEYNESS_LABELS[column]}",
            fontsize=10,
        )
        if index >= 6:
            axis.set_xlabel("Daily IV P&L")
        if index % 3 == 0:
            axis.set_ylabel("Density")
        if index == 0:
            axis.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = _output_path(output_path)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_iv_cross_cell_sensitivity(
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    output_path: Path,
) -> Path:
    """Compare four IV cells after one calm and one turbulent history."""
    plt, _stats = _plotting()
    history, future, samples = _iv_arrays(history, future, samples)
    mean_iv = history.mean(axis=(-1, -2))
    activity = np.diff(mean_iv, axis=1).std(axis=1)
    ranked = np.argsort(activity)
    calm_index = int(ranked[int(0.10 * len(ranked))])
    turbulent_index = int(ranked[int(0.90 * len(ranked))])
    cells = ((0, 0), (0, 4), (2, 2), (4, 2))
    labels = (
        "1M OTM Put\n(K=0.70)",
        "1M OTM Call\n(K=1.30)",
        "6M ATM\n(K=1.00)",
        "2Y ATM\n(K=1.00)",
    )
    history_days = np.arange(-29, 1)
    future_days = np.arange(1, 31)
    path_colors = ("#E91E63", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    fig.suptitle(
        "Same Cells, Different Regimes: Uncertainty Responds Differently Per Cell\n"
        "Black = history, colored = forecast fan",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    regimes = (
        (turbulent_index, "Turbulent", IV_TURBULENT_COLOR),
        (calm_index, "Calm", IV_CALM_COLOR),
    )
    for plot_row, (window, regime, color) in enumerate(regimes):
        for plot_column, ((row, column), label) in enumerate(
            zip(cells, labels, strict=True)
        ):
            axis = axes[plot_row, plot_column]
            history_cell = history[window, :, row, column]
            sample_cell = samples[window, :, :, row, column]
            future_cell = future[window, :, row, column]
            spread = float(sample_cell.std(axis=0).mean())
            axis.plot(
                history_days, history_cell, color="black", linewidth=1.5, zorder=6
            )
            axis.axvline(0.5, color="gray", linestyle=":", alpha=0.6)
            for sample in sample_cell[:50]:
                axis.plot(future_days, sample, color=color, alpha=0.04, linewidth=0.3)
            q05, q95 = np.percentile(sample_cell, (5, 95), axis=0)
            axis.fill_between(future_days, q05, q95, color=color, alpha=0.08)
            path_indices = np.linspace(
                0, sample_cell.shape[0] - 1, min(6, sample_cell.shape[0]), dtype=int
            )
            for path_number, sample_index in enumerate(path_indices):
                path_color = path_colors[path_number % len(path_colors)]
                axis.plot(
                    future_days,
                    sample_cell[sample_index],
                    color=path_color,
                    alpha=0.7,
                    linewidth=1.0,
                    zorder=4,
                )
                axis.plot(
                    (0, 1),
                    (history_cell[-1], sample_cell[sample_index, 0]),
                    color=path_color,
                    linewidth=0.7,
                    alpha=0.5,
                )
            axis.plot(
                future_days,
                future_cell,
                color=GROUND_TRUTH_COLOR,
                linewidth=2.0,
                linestyle="--",
                zorder=5,
            )
            axis.text(
                0.97,
                0.95,
                f"spread={spread * 100:.1f}%",
                transform=axis.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
            )
            if plot_row == 0:
                axis.set_title(label, fontsize=11, fontweight="bold")
            if plot_column == 0:
                axis.set_ylabel(
                    f"{regime}\n(vov={activity[window]:.4f})",
                    fontsize=10,
                    fontweight="bold",
                )
            if plot_row == 1:
                axis.set_xlabel("Day (0 = forecast start)")
            axis.tick_params(labelsize=9)
    for column in range(4):
        lower = min(axes[0, column].get_ylim()[0], axes[1, column].get_ylim()[0])
        upper = max(axes[0, column].get_ylim()[1], axes[1, column].get_ylim()[1])
        axes[0, column].set_ylim(lower, upper)
        axes[1, column].set_ylim(lower, upper)
    fig.tight_layout()
    output = _output_path(output_path)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return stable columnwise ranks in the unit interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or array.shape[0] < 2:
        raise PaperPlotError("activity inputs must be a finite matrix with two rows")
    ranks = np.empty_like(array)
    grid = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float64)
    for column in range(array.shape[1]):
        order = np.argsort(array[:, column], kind="mergesort")
        ranks[order, column] = grid
    return ranks


def select_common_histories(
    factor_scale: np.ndarray,
    *,
    calm_quantile: float,
    turbulent_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select common calm and turbulent windows from history-derived scales."""
    if not 0.0 < calm_quantile < turbulent_quantile < 1.0:
        raise PaperPlotError("selection must satisfy 0 < calm < turbulent < 1")
    scale = np.asarray(factor_scale, dtype=np.float64)
    activity = percentile_ranks(np.log(np.maximum(scale, 1e-12))).mean(axis=1)
    targets = np.quantile(activity, (calm_quantile, turbulent_quantile))
    calm = int(np.argmin(np.abs(activity - targets[0])))
    turbulent_distance = np.abs(activity - targets[1])
    turbulent_distance[calm] = np.inf
    turbulent = int(np.argmin(turbulent_distance))
    return np.asarray((calm, turbulent), dtype=np.int64), activity


def build_factor_fan_selection(
    *,
    selected: np.ndarray,
    activity_score: np.ndarray,
    raw_indices: np.ndarray,
    data_dir: Path,
) -> dict[str, Any]:
    """Build the exact history-only selection evidence used by the ICAIF figure."""
    from ..data import load_aligned_iv_factor_panel

    selected = np.asarray(selected, dtype=np.int64)
    activity = np.asarray(activity_score, dtype=np.float64)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    if selected.shape != (2,) or activity.shape != raw_indices.shape:
        raise PaperPlotError("factor-fan selection arrays are inconsistent")
    _panel, _columns, dates = load_aligned_iv_factor_panel(data_dir=data_dir)
    chosen_raw = raw_indices[selected]
    history_end = chosen_raw + 29
    future_start = history_end + 1
    future_end = future_start + 29
    labels = ("calm", "turbulent")
    return {
        "figure": "paper/camera_ready/icaif/illustrations/joint39_noniv_common_history_fans.png",
        "checkpoint": "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt",
        "checkpoint_sha256": "35b3c416c89d0d928b2943406e64e0b687a0ded91c6fb379c8b78b5fecdfb217",
        "checkpoint_epoch": 1,
        "state_scope": "joint38",
        "panel_channels": 39,
        "factor_channels": 14,
        "factors_plotted": [name for name, _label in FACTOR_SPECS],
        "validation_windows": int(raw_indices.size),
        "history_days": 30,
        "future_days": 30,
        "samples": 64,
        "seed": 7345,
        "device": "cuda",
        "selection_rule": (
            "history only: rank each non-IV channel's causal local scale across "
            "validation windows, average the 14 percentile ranks, and select "
            "windows nearest the requested calm/turbulent score quantiles"
        ),
        "realized_future_used_for_selection": False,
        "windows": {
            label: {
                "validation_index": int(selected[position]),
                "raw_panel_index": int(chosen_raw[position]),
                "activity_score": float(activity[selected[position]]),
                "activity_score_percentile": float(
                    np.mean(activity <= activity[selected[position]])
                ),
                "history_start": str(dates[chosen_raw[position]].date()),
                "history_end": str(dates[history_end[position]].date()),
                "future_start": str(dates[future_start[position]].date()),
                "future_end": str(dates[future_end[position]].date()),
            }
            for position, label in enumerate(labels)
        },
    }


def plot_joint_factor_fans(
    histories: np.ndarray,
    futures: np.ndarray,
    samples: np.ndarray,
    spec_names: Sequence[str],
    end_dates: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot three factors for one calm and one turbulent conditioning history."""
    plt, _stats = _plotting()
    history = np.asarray(histories, dtype=np.float64)
    future = np.asarray(futures, dtype=np.float64)
    sample = np.asarray(samples, dtype=np.float64)
    names = list(spec_names)
    if (
        history.shape != (2, 30, len(names))
        or future.shape != history.shape
        or sample.shape != (2, 64, 30, len(names))
        or len(end_dates) != 2
        or any(name not in names for name, _label in FACTOR_SPECS)
        or not all(np.isfinite(array).all() for array in (history, future, sample))
    ):
        raise PaperPlotError("joint factor-fan arrays are inconsistent")

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
    legend_handles: dict[str, Any] = {}
    for column, (factor_name, display_name) in enumerate(FACTOR_SPECS):
        factor_index = names.index(factor_name)
        for row in range(2):
            axis = axes[row, column]
            factor_samples = sample[row, :, :, factor_index]
            q10, q25, q50, q75, q90 = np.percentile(
                factor_samples, (10, 25, 50, 75, 90), axis=0
            )
            line = axis.plot(
                history_days,
                history[row, :, factor_index],
                color=HISTORY_COLOR,
                linewidth=1.25,
                label="History",
                zorder=5,
            )[0]
            legend_handles.setdefault("History", line)
            band90 = axis.fill_between(
                future_days,
                q10,
                q90,
                color=colors[row],
                alpha=0.13,
                linewidth=0,
                label="10–90% band",
                zorder=1,
            )
            legend_handles.setdefault("10–90% band", band90)
            band50 = axis.fill_between(
                future_days,
                q25,
                q75,
                color=colors[row],
                alpha=0.25,
                linewidth=0,
                label="IQR",
                zorder=2,
            )
            legend_handles.setdefault("IQR", band50)
            for path_number, path_index in enumerate((3, 41)):
                scenario = axis.plot(
                    future_days,
                    factor_samples[path_index],
                    color=colors[row],
                    alpha=0.38,
                    linewidth=0.65,
                    label="Joint sample paths" if path_number == 0 else None,
                    zorder=3,
                )[0]
                if path_number == 0:
                    legend_handles.setdefault("Joint sample paths", scenario)
            median = axis.plot(
                future_days,
                q50,
                color=colors[row],
                linewidth=1.35,
                label="Median",
                zorder=4,
            )[0]
            legend_handles.setdefault("Median", median)
            realized = axis.plot(
                future_days,
                future[row, :, factor_index],
                color=FUTURE_COLOR,
                linestyle=(0, (3, 2)),
                linewidth=1.15,
                label="Realized future",
                zorder=5,
            )[0]
            legend_handles.setdefault("Realized future", realized)
            axis.axvline(0.5, color="#8A8A8A", linestyle=":", linewidth=0.75)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.015)
            if row == 0:
                axis.set_title(display_name, fontweight="semibold", pad=2.0)
            if column == 0:
                axis.set_ylabel(row_labels[row], fontweight="semibold", labelpad=3.5)
                axis.text(
                    0.02,
                    0.94,
                    f"through {end_dates[row]}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.1,
                    color="#5F6368",
                )
    labels = (
        "History",
        "10–90% band",
        "IQR",
        "Joint sample paths",
        "Median",
        "Realized future",
    )
    fig.legend(
        [legend_handles[label] for label in labels],
        labels,
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
    output = _output_path(output_path)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return output


def _sample_scenarioflow(
    *,
    checkpoint: Path,
    history_state: np.ndarray,
    history_increment: np.ndarray,
    future_increment: np.ndarray,
    specs: list[Any],
    n_samples: int,
    batch_size: int,
    chunk_size: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Generate raw paths through the portable model and state APIs."""
    import torch

    from ..model import load_model
    from ..state import normalize_increment_windows, reconstruct_state_from_increments

    normalized, _future, center, scale = normalize_increment_windows(
        history_increment,
        future_increment,
        half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
    )
    history_level = _encode_history_levels(history_state, specs)
    resolved_device = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )
    devices = [resolved_device] if resolved_device.type == "cuda" else []
    generated = []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(int(seed))
        model, _payload = load_model(checkpoint, resolved_device)
        model.eval()
        if int(model.cfg.n_cells) != history_state.shape[-1]:
            raise PaperPlotError("checkpoint channel count does not match plot data")
        for start in range(0, history_state.shape[0], batch_size):
            stop = min(start + batch_size, history_state.shape[0])
            sampled_increment = model.sample_batched(
                torch.from_numpy(history_level[start:stop]).to(resolved_device),
                torch.from_numpy(normalized[start:stop]).to(resolved_device),
                torch.from_numpy(center[start:stop]).to(resolved_device),
                torch.from_numpy(scale[start:stop]).to(resolved_device),
                drift_feature=torch.zeros_like(
                    torch.from_numpy(center[start:stop]).to(resolved_device)
                ),
                n_samples=int(n_samples),
                n_steps=30,
                chunk_size=int(chunk_size),
                temperature=1.0,
            )
            generated.append(
                reconstruct_state_from_increments(
                    history_state[start:stop, -1],
                    sampled_increment.detach().cpu().numpy(),
                    specs,
                )
            )
    return np.concatenate(generated, axis=0).astype(np.float32)


def _iv_plot_data(
    *,
    checkpoint: Path,
    data_dir: Path,
    n_windows: int,
    n_samples: int,
    batch_size: int,
    chunk_size: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation = _prepare_iv_validation_block(data_dir)
    count = min(int(n_windows), int(validation.history_state.shape[0]))
    if count < 1:
        raise PaperPlotError("n_windows must be positive")
    history_state = validation.history_state[:count]
    future_state = validation.future_state[:count]
    samples = _sample_scenarioflow(
        checkpoint=checkpoint,
        history_state=history_state,
        history_increment=validation.history_increment[:count],
        future_increment=validation.future_increment[:count],
        specs=validation.specs,
        n_samples=n_samples,
        batch_size=batch_size,
        chunk_size=chunk_size,
        seed=seed,
        device=device,
    )
    return (
        history_state.reshape(count, 30, 5, 5),
        future_state.reshape(count, 30, 5, 5),
        samples.reshape(count, n_samples, 30, 5, 5),
    )


def _joint_factor_data(
    *,
    checkpoint: Path,
    data_dir: Path,
    batch_size: int,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    from ..data import load_aligned_iv_factor_panel
    from ..evaluation.current_panel import prepare_validation_blocks
    from ..state import normalize_increment_windows

    _panel, _columns, _train, validation = prepare_validation_blocks(data_dir=data_dir)
    _history, _future, _center, scale = normalize_increment_windows(
        validation.history_increment,
        validation.future_increment,
        half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
    )
    selected, activity = select_common_histories(
        scale[:, 25:], calm_quantile=0.15, turbulent_quantile=0.85
    )
    samples = _sample_scenarioflow(
        checkpoint=checkpoint,
        history_state=validation.history_state[selected],
        history_increment=validation.history_increment[selected],
        future_increment=validation.future_increment[selected],
        specs=validation.specs,
        n_samples=64,
        batch_size=batch_size,
        chunk_size=chunk_size,
        seed=7345,
        device=device,
    )
    _source_panel, _source_columns, dates = load_aligned_iv_factor_panel(
        data_dir=data_dir
    )
    history_end_positions = validation.indices[selected] + 29
    end_dates = [str(dates[position].date()) for position in history_end_positions]
    selection = build_factor_fan_selection(
        selected=selected,
        activity_score=activity,
        raw_indices=validation.indices,
        data_dir=data_dir,
    )
    return (
        validation.history_state[selected],
        validation.future_state[selected],
        samples,
        [spec.name for spec in validation.specs],
        end_dates,
        selection,
    )


def _parser(*, prog: str) -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "recipe",
        choices=("iv-marginal", "iv-cross-cell", "joint-factor-fans"),
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=repository / "data")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=repository / "artifacts" / "manifest-v1.tsv",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=400)
    parser.add_argument("--samples", type=int, default=50)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "python -m scenarioflow.paper.plots",
) -> int:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.chunk_size < 1 or args.samples < 1:
        parser.error("--batch-size, --chunk-size, and --samples must be positive")
    try:
        resolver = ArtifactResolver.from_tsv(args.artifact_manifest)
        if args.recipe in {"iv-marginal", "iv-cross-cell"}:
            checkpoint = resolver.resolve("core.iv-only", args.artifact_root)
            history, future, samples = _iv_plot_data(
                checkpoint=checkpoint,
                data_dir=args.data_dir,
                n_windows=args.max_windows,
                n_samples=args.samples,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                seed=734,
                device=args.device,
            )
            if args.recipe == "iv-marginal":
                output = plot_iv_marginal_daily_changes(
                    history, future, samples, args.output
                )
            else:
                output = plot_iv_cross_cell_sensitivity(
                    history, future, samples, args.output
                )
            outputs = {"figure": str(output)}
        else:
            checkpoint = resolver.resolve("core.stage2", args.artifact_root)
            history, future, samples, names, dates, selection = _joint_factor_data(
                checkpoint=checkpoint,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                device=args.device,
            )
            output = plot_joint_factor_fans(
                history, future, samples, names, dates, args.output
            )
            selection_path = output.with_name(f"{output.stem}-selection.json")
            selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
            outputs = {"figure": str(output), "selection": str(selection_path)}
    except (ArtifactError, OSError, PaperPlotError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
