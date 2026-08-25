#!/usr/bin/env python
"""Evaluate comparison baselines on the current real-VIX SNI panel.

This is the paper-facing rerun for baseline comparison metrics. It uses the
same current 39-state panel as the promoted SNI model:

  - 25 IV surface state variables;
  - 14 non-surface factor state variables, including real VIX;
  - state-coordinate increments for generation;
  - raw-state reconstruction before evaluation.

The old ``evaluate_baselines_38d.py`` benchmark is kept as historical evidence,
but it predates the real-VIX panel and uses 38 daily-change targets. This script
is the current-panel replacement for baseline comparison rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, ".")

from experiments.backfill.baselines.current_panel_deep_baselines import (  # noqa: E402
    load_current_panel_deep_baseline,
)
from experiments.backfill.baselines.joint_classical_baselines import (  # noqa: E402
    JointBootstrap,
    JointFilteredHS,
    JointGARCHCCC,
    JointPCAVAR,
    JointRandomWalk,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    panel_daily_changes,
    summarize_joint_quality,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


CLASSICAL_BASELINES = [
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
]
DEEP_BASELINES = ["deepvar", "timegrad", "path_diffusion"]
BASELINES = CLASSICAL_BASELINES + DEEP_BASELINES


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def median_abs_dev_from_half(values: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(np.abs(arr.reshape(-1) - 0.5)))


def make_block_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        iv_count=int(args.iv_count),
        max_train_windows=0,
        clean_nonpositive_log_levels=True,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
    )


class WindowHistoricalSim:
    """Conditional historical-window sampler on current-panel increments."""

    def __init__(
        self,
        history_increment: np.ndarray,
        future_increment: np.ndarray,
        history_state: np.ndarray,
        *,
        iv_count: int,
    ) -> None:
        self.history_increment = np.asarray(history_increment, dtype=np.float32)
        self.future_increment = np.asarray(future_increment, dtype=np.float32)
        self.history_state = np.asarray(history_state, dtype=np.float32)
        self.iv_count = int(iv_count)
        self.future_len = int(future_increment.shape[1])
        self.D = int(future_increment.shape[-1])
        self.features = np.asarray(
            [
                self._features(self.history_increment[i], self.history_state[i])
                for i in range(self.history_increment.shape[0])
            ],
            dtype=np.float64,
        )
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features_norm = (self.features - self.mean) / self.std

    def _features(self, history_increment: np.ndarray, history_state: np.ndarray) -> np.ndarray:
        iv = history_state[:, : self.iv_count].reshape(history_state.shape[0], 5, 5)
        mean_iv = float(iv[-5:].mean())
        mean_iv_ts = iv.mean(axis=(1, 2))
        vov = float(np.diff(mean_iv_ts).std()) if mean_iv_ts.shape[0] > 1 else 0.0
        slope = float(iv[-1, -1, 2] - iv[-1, 0, 2])
        spx_activity = float(np.abs(history_increment[-5:, self.iv_count]).mean())
        factor = history_state[:, self.iv_count :]
        yc_slope = float(factor[-1, 8] - factor[-1, 7]) if factor.shape[-1] > 8 else 0.0
        credit_width = float(factor[-1, 10] - factor[-1, 9]) if factor.shape[-1] > 10 else 0.0
        return np.array([mean_iv, vov, slope, spx_activity, yc_slope, credit_width])

    def sample_joint(
        self,
        history_changes: np.ndarray,
        *,
        n_samples: int,
        history_state: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        history_changes = np.asarray(history_changes, dtype=np.float32)
        history_state = np.asarray(history_state, dtype=np.float32)
        out = np.zeros(
            (history_changes.shape[0], int(n_samples), self.future_len, self.D),
            dtype=np.float32,
        )
        for b in range(history_changes.shape[0]):
            query = (self._features(history_changes[b], history_state[b]) - self.mean) / self.std
            dist = np.linalg.norm(self.features_norm - query[None, :], axis=1)
            # Stable soft nearest-neighbour weights.
            logits = -dist
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            idx = np.random.choice(
                self.future_increment.shape[0],
                size=int(n_samples),
                replace=True,
                p=weights,
            )
            out[b] = self.future_increment[idx]
        return out


def create_baseline(
    name: str,
    train_block: Any,
    train_daily_increment: np.ndarray,
    *,
    deep_checkpoint_dir: str | Path,
    device: str,
    deep_sample_steps: int,
    deep_chunk_size: int,
) -> Any:
    if name == "random_walk":
        return JointRandomWalk(train_daily_increment)
    if name == "bootstrap":
        return JointBootstrap(train_daily_increment)
    if name == "historical_sim":
        return WindowHistoricalSim(
            train_block.history_increment,
            train_block.future_increment,
            train_block.history_state,
            iv_count=25,
        )
    if name == "pca_var":
        return JointPCAVAR(train_daily_increment)
    if name == "garch_ccc":
        return JointGARCHCCC(train_daily_increment)
    if name == "filtered_hs":
        return JointFilteredHS(train_daily_increment)
    if name in DEEP_BASELINES:
        checkpoint_path = Path(deep_checkpoint_dir) / name / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"missing current-panel deep baseline checkpoint: {checkpoint_path}"
            )
        return load_current_panel_deep_baseline(
            checkpoint_path,
            device=device,
            sample_steps=int(deep_sample_steps),
            chunk_size=int(deep_chunk_size),
        )
    raise ValueError(f"unknown baseline {name!r}")


def compute_crps(samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int) -> dict[str, Any]:
    horizons = [1, 7, 14, 30]
    n, k, t_len, d = samples_delta.shape
    per_h: dict[str, list[float]] = {}
    for h in horizons:
        if h > t_len:
            continue
        t = h - 1
        s = samples_delta[:, :, t, :]
        y = gt_delta[:, t, :]
        mae = np.abs(s - y[:, None, :]).mean(axis=1)
        spread = np.zeros((n, d), dtype=np.float64)
        for i in range(k):
            spread += np.abs(s[:, i : i + 1, :] - s).sum(axis=1)
        spread /= float(k * k)
        per_h[str(h)] = (mae - 0.5 * spread).mean(axis=0).tolist()
    per_dim = np.asarray(list(per_h.values()), dtype=np.float64).mean(axis=0)
    return {
        "per_horizon": per_h,
        "per_dimension": per_dim.tolist(),
        "overall": float(per_dim.mean()),
        "iv_crps": float(per_dim[:iv_count].mean()),
        "factor_crps": float(per_dim[iv_count:].mean()),
    }


def compute_energy_score(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    train_delta: np.ndarray,
) -> dict[str, Any]:
    horizons = [1, 7, 14, 30]
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    s_z = (samples_delta - mean[None, None, None, :]) / std[None, None, None, :]
    y_z = (gt_delta - mean[None, None, :]) / std[None, None, :]
    n, k, t_len, _d = samples_delta.shape
    per_h: dict[str, float] = {}
    for h in horizons:
        if h > t_len:
            continue
        t = h - 1
        s = s_z[:, :, t, :]
        y = y_z[:, t, :]
        term1 = np.linalg.norm(s - y[:, None, :], axis=2).mean(axis=1)
        spread = np.zeros(n, dtype=np.float64)
        for i in range(k):
            spread += np.linalg.norm(s[:, i : i + 1, :] - s, axis=2).sum(axis=1)
        spread /= float(k * k)
        per_h[str(h)] = float((term1 - 0.5 * spread).mean())
    return {"per_horizon": per_h, "overall": float(np.mean(list(per_h.values())))}


def compute_variogram_score(
    samples_delta: np.ndarray,
    gt_delta: np.ndarray,
    train_delta: np.ndarray,
    *,
    p: float = 0.5,
) -> dict[str, Any]:
    mean = train_delta.mean(axis=0)
    std = train_delta.std(axis=0) + 1e-8
    t = min(samples_delta.shape[2], 30) - 1
    s = (samples_delta[:, :, t, :] - mean[None, None, :]) / std[None, None, :]
    y = (gt_delta[:, t, :] - mean[None, :]) / std[None, :]
    sample_vario = np.mean(np.abs(s[:, :, :, None] - s[:, :, None, :]) ** p, axis=1)
    gt_vario = np.abs(y[:, :, None] - y[:, None, :]) ** p
    triu = np.triu(np.ones((s.shape[-1], s.shape[-1]), dtype=bool), k=1)
    return {
        "variogram_score": float(((sample_vario - gt_vario) ** 2)[:, triu].mean()),
        "p": float(p),
        "horizon": int(t + 1),
    }


def compute_correlation_score(samples_delta: np.ndarray, gt_delta: np.ndarray, *, iv_count: int) -> dict[str, Any]:
    gt = gt_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    gen = samples_delta[..., iv_count:].reshape(-1, gt_delta.shape[-1] - iv_count)
    gt_corr = np.nan_to_num(np.corrcoef(gt.T), nan=0.0)
    gen_corr = np.nan_to_num(np.corrcoef(gen.T), nan=0.0)
    fro = float(np.linalg.norm(gen_corr - gt_corr, ord="fro"))
    fro_gt = float(np.linalg.norm(gt_corr, ord="fro"))
    ratio = fro / max(fro_gt, 1e-12)
    return {"frobenius_error": fro, "frobenius_ratio": ratio, "corr_score": float(1.0 - ratio)}


def compute_factor_ks(samples_delta: np.ndarray, gt_delta: np.ndarray, factor_names: list[str], *, iv_count: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    stats = []
    for j, name in enumerate(factor_names):
        gen = samples_delta[..., iv_count + j].reshape(-1)
        gt = gt_delta[..., iv_count + j].reshape(-1)
        stat, pval = sp_stats.ks_2samp(gen, gt)
        stats.append(float(stat))
        rows[name] = {"statistic": float(stat), "p_value": float(pval)}
    return {
        "per_factor": rows,
        "mean_statistic": float(np.mean(stats)) if stats else None,
        "median_statistic": float(np.median(stats)) if stats else None,
        "worst_statistic": float(np.max(stats)) if stats else None,
    }


def extract_iv_metrics(iv: dict[str, Any]) -> dict[str, Any]:
    coverage_overall = iv["coverage"]["overall"]
    cov90 = coverage_overall.get("0.9", coverage_overall.get(0.9))
    return {
        "cov90": cov90,
        "calibration_error": iv["coverage"]["calibration_error"],
        "level_ks_median": iv["distributional_fidelity"]["ks_level_test"]["median_stat"],
        "median_abs_dev": median_abs_dev_from_half(
            iv["distributional_fidelity"]["median_bias"].get("above_frac")
        ),
        "width_rho": iv["risk_state_allocation"]["history_width_spearman"],
        "mr_ratio": iv["mean_reversion"]["mr_gt_ratio"],
        "economic_ratio": iv["cointegration"].get(
            "gen_gt_ratio_legacy", iv["cointegration"].get("gen_gt_ratio")
        ),
        "acf_corr": iv["time_series"]["acf"]["acf_correlation"],
        "kurtosis_ratio": iv["time_series"]["kurtosis"]["kurtosis_ratio"],
        "jump_ks": iv["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def evaluate_one(
    name: str,
    model: Any,
    train_block: Any,
    val_block: Any,
    train_raw_delta: np.ndarray,
    *,
    batch: Any,
    samples: int,
    batch_size: int,
    iv_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    n_windows = int(val_block.history_increment.shape[0])
    all_inc: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, n_windows, int(batch_size)):
        end = min(start + int(batch_size), n_windows)
        kwargs: dict[str, Any] = {}
        if name == "historical_sim":
            kwargs["history_state"] = val_block.history_state[start:end]
        out = model.sample_joint(
            val_block.history_increment[start:end],
            n_samples=int(samples),
            **kwargs,
        )
        all_inc.append(np.asarray(out, dtype=np.float32))
        print(f"  {name}: generated windows {end}/{n_windows}", flush=True)
    sample_increment = np.concatenate(all_inc, axis=0)
    sample_raw = reconstruct_state_from_increments(
        val_block.history_state[:, -1, :],
        sample_increment,
        val_block.specs,
    ).astype(np.float32)
    generation_time = time.time() - t0

    raw_history = val_block.history_state.astype(np.float32)
    raw_future = val_block.future_state.astype(np.float32)
    gt_delta = panel_daily_changes(raw_history, raw_future)
    sample_prev = np.concatenate(
        [
            np.repeat(raw_history[:, None, -1:, :], sample_raw.shape[1], axis=1),
            sample_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    sample_delta = sample_raw - sample_prev

    cond_samples = sample_raw[..., :iv_count].reshape(
        n_windows, int(samples), sample_raw.shape[2], 5, 5
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()[:n_windows]
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(n_windows)}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    iv_results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
        data_path="data/vol_surface_with_ret.npz",
        test_start=4511,
        val_size=441,
        history_len=30,
        future_len=30,
        batch_size=int(batch_size),
        conditionality_samples=min(32, int(samples)),
        conditionality_max_batches=8,
        device=batch.history_01.device,
        eval_split="val",
    )
    factor_names = [spec.name for spec in val_block.specs[iv_count:]]
    joint_quality = summarize_joint_quality(
        raw_history,
        raw_future,
        sample_raw,
        factor_names,
        iv_count=int(iv_count),
    )
    result = {
        "baseline_name": name,
        "mode": "current_realvix_39_state_panel",
        "generation_time_s": float(generation_time),
        "n_windows": int(n_windows),
        "n_samples": int(samples),
        "state_specs": [asdict(spec) for spec in val_block.specs],
        "current_panel_iv": iv_results,
        "current_panel_iv_metrics": extract_iv_metrics(iv_results),
        "joint_panel_quality": joint_quality,
        "panel_b_crps": compute_crps(sample_delta, gt_delta, iv_count=int(iv_count)),
        "panel_b_energy_score": compute_energy_score(sample_delta, gt_delta, train_raw_delta),
        "panel_b_variogram_score": compute_variogram_score(sample_delta, gt_delta, train_raw_delta),
        "panel_b_correlation": compute_correlation_score(sample_delta, gt_delta, iv_count=int(iv_count)),
        "panel_b_factor_ks": compute_factor_ks(sample_delta, gt_delta, factor_names, iv_count=int(iv_count)),
    }
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_current_panel.json").write_text(
        json.dumps(make_serializable(result), indent=2),
        encoding="utf-8",
    )
    return make_serializable(result)


def write_summary(output_dir: Path, results: dict[str, Any]) -> None:
    rows = []
    for name, r in results.items():
        iv = r["current_panel_iv_metrics"]
        crps = r["panel_b_crps"]
        jq = r["joint_panel_quality"]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(crps["overall"], 5),
                    fmt(crps["iv_crps"], 5),
                    fmt(crps["factor_crps"], 5),
                    fmt(r["panel_b_energy_score"]["overall"], 4),
                    fmt(r["panel_b_variogram_score"]["variogram_score"], 5),
                    fmt(r["panel_b_correlation"]["corr_score"], 3),
                    fmt(iv["cov90"], 3),
                    fmt(iv["calibration_error"], 3),
                    fmt(iv["level_ks_median"], 3),
                    fmt(iv["median_abs_dev"], 3),
                    fmt(iv["width_rho"], 3),
                    fmt(iv["mr_ratio"], 3),
                    fmt(iv["economic_ratio"], 3),
                    fmt(iv["jump_ks"], 3),
                    fmt(iv["kurtosis_ratio"], 3),
                    fmt(iv["acf_corr"], 3),
                    fmt(jq["factor_delta_ks_mean"], 3),
                    fmt(jq["factor_tail_q99_ratio_median"], 3),
                ]
            )
            + " |"
        )
    header = (
        "| Baseline | CRPS | IV CRPS | Factor CRPS | Energy | Variogram | Corr | "
        "Cov90 | CalErr | Level KS med | Median dev | Width rho | MR ratio | "
        "Econ ratio | Jump KS | Kurtosis | ACF | Factor KS | Tail q99 ratio |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = header + "\n".join(rows) + "\n"
    summary = {
        "mode": "current_realvix_39_state_panel",
        "baselines": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(
        output_dir / "summary.md",
        "Current-panel baseline comparison rerun",
        [
            "- scope: current real-VIX 39-state panel",
            "- rows: classical and neural comparison models evaluated on the same validation panel",
            "- note: neural rows instantiate DeepVAR, TimeGrad, and CSDI mechanisms under the common current-panel scenario-generation protocol.",
            "",
            body,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=772)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="results/baselines_current_panel")
    parser.add_argument("--deep_checkpoint_dir", default="models/backfill/baselines/current_panel_deep")
    parser.add_argument("--deep_sample_steps", type=int, default=8)
    parser.add_argument("--deep_chunk_size", type=int, default=512)
    args = parser.parse_args()

    set_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_args = make_block_args(args)
    columns, metadata, train_block, val_block = build_blocks(block_args)
    if int(args.max_windows) > 0:
        val_block = type(val_block)(
            history_increment=val_block.history_increment[: int(args.max_windows)],
            future_increment=val_block.future_increment[: int(args.max_windows)],
            history_state=val_block.history_state[: int(args.max_windows)],
            future_state=val_block.future_state[: int(args.max_windows)],
            indices=val_block.indices[: int(args.max_windows)],
            specs=val_block.specs,
        )
    if len(val_block.specs) != 39:
        raise RuntimeError(f"expected current 39-state panel, got {len(val_block.specs)} states")
    if not any(spec.name == "factor:vix" for spec in val_block.specs):
        raise RuntimeError("current panel does not include factor:vix")

    # One contiguous-ish increment per train window avoids overweighting every
    # overlapped history/future day while preserving the official training split.
    train_daily_increment = train_block.history_increment[:, -1, :].astype(np.float32)
    train_raw_delta = panel_daily_changes(train_block.history_state, train_block.future_state).reshape(
        -1, len(train_block.specs)
    )
    batch = build_rollout_windows(
        data_path="data/vol_surface_with_ret.npz",
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(val_block.history_state.shape[0]),
        device=args.device,
        split="val",
    )
    history_align = np.max(
        np.abs(
            val_block.history_state[:, :, : int(args.iv_count)].reshape(val_block.history_state.shape[0], -1)
            - batch.history_01.detach().cpu().numpy().reshape(val_block.history_state.shape[0], -1)
        )
    )
    future_align = np.max(
        np.abs(
            val_block.future_state[:, :, : int(args.iv_count)].reshape(val_block.future_state.shape[0], -1)
            - batch.future_01.detach().cpu().numpy().reshape(val_block.future_state.shape[0], -1)
        )
    )
    if history_align > 1e-6 or future_align > 1e-6:
        raise RuntimeError(f"IV alignment failure: history={history_align}, future={future_align}")

    manifest = {
        "mode": "current_realvix_39_state_panel",
        "columns": columns,
        "metadata": metadata,
        "state_specs": [asdict(spec) for spec in val_block.specs],
        "n_train_windows": int(train_block.history_state.shape[0]),
        "n_val_windows": int(val_block.history_state.shape[0]),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "iv_alignment": {
            "history_max_abs_error": float(history_align),
            "future_max_abs_error": float(future_align),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(make_serializable(manifest), indent=2), encoding="utf-8")

    all_results: dict[str, Any] = {}
    for name in args.baselines:
        print("\n" + "=" * 72)
        print(f"Evaluating current-panel baseline: {name}")
        print("=" * 72)
        model = create_baseline(
            name,
            train_block,
            train_daily_increment,
            deep_checkpoint_dir=args.deep_checkpoint_dir,
            device=args.device,
            deep_sample_steps=int(args.deep_sample_steps),
            deep_chunk_size=int(args.deep_chunk_size),
        )
        all_results[name] = evaluate_one(
            name,
            model,
            train_block,
            val_block,
            train_raw_delta,
            batch=batch,
            samples=int(args.n_samples),
            batch_size=int(args.batch_size),
            iv_count=int(args.iv_count),
            output_dir=output_dir,
        )
    write_summary(output_dir, all_results)
    print(f"Wrote current-panel baseline rerun to {output_dir}")


if __name__ == "__main__":
    main()
