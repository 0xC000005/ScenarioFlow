#!/usr/bin/env python
"""403a: evaluate 392a with a separately fit marginal/regime calibration layer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    HistoryFutureDictDataset,
    build_multistep_windows,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (  # noqa: E402
    denormalize_iv,
    normalize_iv,
)


HORIZON = 30
GRID = 5


@dataclass
class CalibrationTables:
    levels: np.ndarray
    gen_quantiles: np.ndarray
    target_quantiles: np.ndarray
    vov_q20: float
    vov_q80: float
    alpha: float
    regime_bins: bool


def compute_vov(history_01: np.ndarray) -> np.ndarray:
    mean_iv = history_01.mean(axis=(2, 3))
    return np.diff(mean_iv, axis=1).std(axis=1)


def assign_bins(
    history_01: np.ndarray,
    vov_q20: float,
    vov_q80: float,
    regime_bins: bool,
) -> np.ndarray:
    if not regime_bins:
        return np.zeros(history_01.shape[0], dtype=np.int64)
    vov = compute_vov(history_01)
    bins = np.ones(history_01.shape[0], dtype=np.int64)
    bins[vov <= vov_q20] = 0
    bins[vov >= vov_q80] = 2
    return bins


def strictly_increasing(values: np.ndarray) -> np.ndarray:
    out = np.maximum.accumulate(values.astype(np.float64))
    eps = 1e-7
    for idx in range(1, out.shape[0]):
        if out[idx] <= out[idx - 1]:
            out[idx] = out[idx - 1] + eps
    return np.clip(out, 0.0, 1.0)


def build_recent_calibration_batch(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    calibration_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    val_start = max_train_idx - val_size
    n = min(int(calibration_windows), int(val_start))
    indices = np.arange(val_start - n, val_start)
    history_01, future_01 = build_multistep_windows(
        indices,
        surf_tensor,
        history_len,
        future_len,
    )
    return history_01, normalize_iv(history_01), future_01.view(history_01.shape[0], future_len, GRID, GRID)


@torch.no_grad()
def sample_native(
    model: torch.nn.Module,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        hist = history_norm[start:end].to(device)
        samples = model.sample_batched(
            hist,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        outputs.append(samples.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_calibration_tables(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history: np.ndarray,
    n_quantiles: int,
    alpha: float,
    regime_bins: bool,
    min_bin_windows: int,
) -> CalibrationTables:
    levels = np.linspace(0.001, 0.999, int(n_quantiles), dtype=np.float64)
    vov = compute_vov(calib_history)
    vov_q20 = float(np.quantile(vov, 0.2))
    vov_q80 = float(np.quantile(vov, 0.8))
    bins = assign_bins(calib_history, vov_q20, vov_q80, regime_bins)
    n_bins = 3 if regime_bins else 1
    gen_q = np.zeros((n_bins, HORIZON, GRID, GRID, len(levels)), dtype=np.float64)
    tgt_q = np.zeros_like(gen_q)
    all_idx = np.arange(calib_history.shape[0])

    for bin_id in range(n_bins):
        idx = np.where(bins == bin_id)[0]
        if idx.shape[0] < int(min_bin_windows):
            idx = all_idx
        for t in range(HORIZON):
            for i in range(GRID):
                for j in range(GRID):
                    gen_values = calib_samples[idx, :, t, i, j].reshape(-1)
                    target_values = calib_future[idx, t, i, j].reshape(-1)
                    gen_q[bin_id, t, i, j] = strictly_increasing(
                        np.quantile(gen_values, levels)
                    )
                    tgt_q[bin_id, t, i, j] = strictly_increasing(
                        np.quantile(target_values, levels)
                    )

    return CalibrationTables(
        levels=levels,
        gen_quantiles=gen_q,
        target_quantiles=tgt_q,
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        alpha=float(alpha),
        regime_bins=bool(regime_bins),
    )


def apply_calibration(
    samples: np.ndarray,
    history_01: np.ndarray,
    tables: CalibrationTables,
) -> np.ndarray:
    calibrated = np.empty_like(samples, dtype=np.float32)
    bins = assign_bins(history_01, tables.vov_q20, tables.vov_q80, tables.regime_bins)
    for b in range(samples.shape[0]):
        bin_id = int(bins[b])
        for t in range(samples.shape[2]):
            for i in range(GRID):
                for j in range(GRID):
                    values = samples[b, :, t, i, j].astype(np.float64)
                    gen_q = tables.gen_quantiles[bin_id, t, i, j]
                    tgt_q = tables.target_quantiles[bin_id, t, i, j]
                    u = np.interp(values, gen_q, tables.levels, left=tables.levels[0], right=tables.levels[-1])
                    mapped = np.interp(u, tables.levels, tgt_q)
                    out = (1.0 - tables.alpha) * values + tables.alpha * mapped
                    calibrated[b, :, t, i, j] = np.clip(out, 0.0, 1.0).astype(np.float32)
    return calibrated


class CalibratedSampler:
    def __init__(
        self,
        base_model: torch.nn.Module,
        tables: CalibrationTables,
        device: torch.device,
    ):
        self.base_model = base_model
        self.tables = tables
        self.device = device

    def eval(self) -> "CalibratedSampler":
        self.base_model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        history_device = history.device
        history_norm = history.to(self.device)
        with torch.no_grad():
            raw = self.base_model.sample_batched(
                history_norm,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
                history_is_normalized=history_is_normalized,
                **kwargs,
            )
        if history_is_normalized:
            history_01 = denormalize_iv(history_norm).detach().cpu().numpy()
        else:
            history_01 = history_norm.detach().cpu().numpy()
        calibrated = apply_calibration(raw.detach().cpu().numpy(), history_01, self.tables)
        return torch.from_numpy(calibrated).to(history_device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--calibration_windows", type=int, default=441)
    parser.add_argument("--calibration_samples", type=int, default=48)
    parser.add_argument("--n_quantiles", type=int, default=101)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

    calib_hist_01, calib_hist_norm, calib_future = build_recent_calibration_batch(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        calibration_windows=args.calibration_windows,
        device=device,
    )
    print("Fitting calibration samples")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables = fit_calibration_tables(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history=calib_hist_01.detach().cpu().numpy(),
        n_quantiles=args.n_quantiles,
        alpha=args.alpha,
        regime_bins=args.regime_bins,
        min_bin_windows=args.min_bin_windows,
    )
    model = CalibratedSampler(base_model, tables, device).eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    outputs: list[np.ndarray] = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        with torch.no_grad():
            samples = model.sample_batched(
                batch.history_norm[start:end],
                n_samples=args.samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
                history_is_normalized=True,
            )
        outputs.append(samples.detach().cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "calibration": {
                "kind": "pre_validation_empirical_quantile_map",
                "calibration_windows": args.calibration_windows,
                "calibration_samples": args.calibration_samples,
                "n_quantiles": args.n_quantiles,
                "alpha": args.alpha,
                "regime_bins": bool(args.regime_bins),
                "vov_q20": tables.vov_q20,
                "vov_q80": tables.vov_q80,
            },
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "regime_coverage": regime_coverage,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 11,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    lines = [
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- calibration: `pre-validation empirical quantile map, regime_bins={bool(args.regime_bins)}, alpha={args.alpha}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        "",
        "**Additional v2 Suites**",
        f"- time-series ACF corr: `{time_series['acf']['acf_correlation']:.3f}`",
        f"- block boundary ratio: `{block_ar['boundary_smoothness']['boundary_ratio']:.3f}`",
        f"- cointegration gen/GT ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- regime coverage overall: `{regime_coverage['overall_pass']}`",
        "",
        "**Fidelity / Structure**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "403a Calibrated Risk-System Validation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
