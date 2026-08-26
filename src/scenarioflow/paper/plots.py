"""Portable recipes for the numerical figures used in the papers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

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
MANAGEMENT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
}
JOINT_FAN_STYLE = {
    "font.size": 7.4,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.65,
}


class PaperPlotError(ValueError):
    """Raised when plot inputs or an output path are invalid."""


def _plotting(style: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError as error:
        raise PaperPlotError(
            "paper plotting requires the scenarioflow[paper] extra"
        ) from error
    plt.rcdefaults()
    plt.rcParams.update(style)
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


def plot_iv_marginal_daily_changes(
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the final ICAIF daily IV changes for nine representative cells."""
    plt, stats = _plotting(MANAGEMENT_STYLE)
    _history, future, samples = _iv_arrays(history, future, samples)
    truth = np.diff(future, axis=1)
    generated = np.diff(samples[:, : min(5, samples.shape[1])], axis=2)
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
        "Daily IV Change Distribution: Ground Truth vs Generated\n"
        "Red = Generated, Green = Ground Truth — matching tails = good kurtosis",
        fontsize=14,
        fontweight="bold",
    )
    for index, (row, column) in enumerate(cells):
        axis = axes[index // 3, index % 3]
        truth_values = truth[:, :, row, column].ravel()
        generated_values = generated[:, :, :, row, column].ravel()
        lower, upper = np.percentile(truth_values, (0.5, 99.5))
        bins = np.linspace(lower, upper, 80)
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
        ks_result = stats.ks_2samp(truth_values, generated_values)
        truth_std = float(truth_values.std())
        generated_std = float(generated_values.std())
        axis.text(
            0.03,
            0.95,
            f"GT kurt={truth_kurtosis:.1f}\n"
            f"Gen kurt={generated_kurtosis:.1f}\n"
            f"Ratio={ratio:.2f}\n"
            f"GT std={truth_std:.4f}\n"
            f"Gen std={generated_std:.4f}\n"
            f"KS={float(ks_result.statistic):.3f} p={float(ks_result.pvalue):.3f}",
            transform=axis.transAxes,
            fontsize=7,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
        )
        passed = ks_result.pvalue > 0.05
        axis.set_title(
            f"({row},{column}) {MATURITY_LABELS[row]} / K={MONEYNESS_LABELS[column]}  "
            f"[{'PASS' if passed else 'FAIL'}]",
            fontsize=10,
            fontweight="normal" if passed else "bold",
            color="black" if passed else "red",
        )
        if index >= 6:
            axis.set_xlabel("Daily IV Change")
        if index % 3 == 0:
            axis.set_ylabel("Density")
        if index == 0:
            axis.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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
    plt, _stats = _plotting(MANAGEMENT_STYLE)
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


def plot_joint_factor_fans(
    histories: np.ndarray,
    futures: np.ndarray,
    samples: np.ndarray,
    spec_names: Sequence[str],
    end_dates: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot three factors for one calm and one turbulent conditioning history."""
    plt, _stats = _plotting(JOINT_FAN_STYLE)
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
