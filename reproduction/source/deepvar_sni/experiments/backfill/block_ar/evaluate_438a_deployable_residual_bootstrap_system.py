#!/usr/bin/env python
"""438a: deployable residual-error bootstrap risk system.

This evaluator turns the 435a oracle feasibility idea into a deployable policy:

    validation 392a median + sampled pre-validation forecast error + small 392a residual shape

The calibration bank is fit only on windows before validation. Validation futures are
used only by the evaluator after samples are frozen, never by the sampler.
"""

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
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    select_rollout_indices,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import (  # noqa: E402
    assign_bins,
    build_recent_calibration_batch,
    compute_vov,
    sample_native,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_iv_ewma_economic_link_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_risk_state_allocation_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)


HORIZON = 30
GRID = 5


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def history_key(history_norm: np.ndarray) -> bytes:
    return np.round(history_norm.astype(np.float32), 6).tobytes()


@dataclass(frozen=True)
class ResidualBank:
    errors: np.ndarray
    bins: np.ndarray
    vov_q20: float
    vov_q80: float
    regime_bins: bool
    min_bin_windows: int


def fit_residual_bank(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history_01: np.ndarray,
    regime_bins: bool,
    min_bin_windows: int,
) -> tuple[ResidualBank, dict[str, Any]]:
    """Fit a frozen forecast-error bank from pre-validation outcomes."""
    calib_median = np.median(calib_samples, axis=1)
    errors = (calib_future - calib_median).astype(np.float32)
    vov = compute_vov(calib_history_01)
    vov_q20 = float(np.quantile(vov, 0.20))
    vov_q80 = float(np.quantile(vov, 0.80))
    bins = assign_bins(calib_history_01, vov_q20, vov_q80, regime_bins)
    counts = {str(i): int((bins == i).sum()) for i in range(3 if regime_bins else 1)}
    meta = {
        "forecast_error_abs_mean": float(np.abs(errors).mean()),
        "forecast_error_abs_p90": float(np.quantile(np.abs(errors), 0.90)),
        "forecast_error_abs_p99": float(np.quantile(np.abs(errors), 0.99)),
        "forecast_error_std": float(errors.std()),
        "calibration_bin_counts": counts,
    }
    bank = ResidualBank(
        errors=errors,
        bins=bins.astype(np.int64),
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        regime_bins=bool(regime_bins),
        min_bin_windows=int(min_bin_windows),
    )
    return bank, meta


def draw_error_indices(
    bank: ResidualBank,
    val_bin: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.where(bank.bins == int(val_bin))[0]
    if idx.shape[0] < bank.min_bin_windows:
        idx = np.arange(bank.errors.shape[0])
    return rng.choice(idx, size=int(n_samples), replace=True)


def build_deployable_samples(
    val_base_samples: np.ndarray,
    val_history_01: np.ndarray,
    bank: ResidualBank,
    residual_shape_scale: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    val_median = np.median(val_base_samples, axis=1)
    val_residual_shape = val_base_samples - val_median[:, None]
    val_bins = assign_bins(
        val_history_01,
        bank.vov_q20,
        bank.vov_q80,
        bank.regime_bins,
    )

    n_windows, n_samples, horizon, rows, cols = val_base_samples.shape
    out = np.empty_like(val_base_samples, dtype=np.float32)
    sampled_bin_counts = {str(i): 0 for i in range(3 if bank.regime_bins else 1)}
    for w in range(n_windows):
        val_bin = int(val_bins[w])
        sampled_bin_counts[str(val_bin)] = sampled_bin_counts.get(str(val_bin), 0) + 1
        picks = draw_error_indices(bank, val_bin, n_samples, rng)
        sampled_errors = bank.errors[picks, :horizon, :rows, :cols]
        shaped = val_median[w, None] + sampled_errors
        shaped = shaped + float(residual_shape_scale) * val_residual_shape[w]
        out[w] = np.clip(shaped, 0.0, 1.0).astype(np.float32)

    meta = {
        "residual_shape_scale": float(residual_shape_scale),
        "validation_bin_counts": sampled_bin_counts,
        "sample_min": float(out.min()),
        "sample_median": float(np.median(out)),
        "sample_max": float(out.max()),
        "sample_std": float(out.std()),
    }
    return out, meta


class FixedDeployableSampler:
    """History-keyed sampler for full-suite conditionality tests.

    The arrays are precomputed from history-only inference plus a frozen
    pre-validation residual bank. The validation future is not available here.
    """

    def __init__(
        self,
        samples_by_key: dict[bytes, np.ndarray],
        *,
        allow_fallback: bool = False,
    ):
        if not samples_by_key:
            raise ValueError("samples_by_key must not be empty")
        self.samples_by_key = samples_by_key
        self.allow_fallback = bool(allow_fallback)
        self.fallback = next(iter(samples_by_key.values()))
        self.hit_count = 0
        self.missing_count = 0

    def eval(self) -> "FixedDeployableSampler":
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
        history_is_normalized: bool = True,
        **_: object,
    ) -> torch.Tensor:
        del chunk_size, history_is_normalized
        hist_np = history.detach().cpu().numpy()
        rows = []
        for hist in hist_np:
            key = history_key(hist)
            arr = self.samples_by_key.get(key)
            if arr is None:
                self.missing_count += 1
                if not self.allow_fallback:
                    raise KeyError(
                        "missing history key in FixedDeployableSampler; "
                        "audit would otherwise silently reuse an unrelated scenario deck"
                    )
                arr = self.fallback
            else:
                self.hit_count += 1
            if n_samples <= arr.shape[0]:
                out = arr[:n_samples, :n_steps]
            else:
                reps = int(np.ceil(n_samples / arr.shape[0]))
                out = np.tile(arr, (reps, 1, 1, 1))[:n_samples, :n_steps]
            rows.append(out)
        return torch.from_numpy(np.stack(rows, axis=0)).to(history.device)


def rollout_start_for_split(
    *,
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
    n_windows: int,
    eval_split: str,
) -> int:
    indices = select_rollout_indices(
        test_start=int(test_start),
        val_size=int(val_size),
        history_len=int(history_len),
        future_len=int(future_len),
        max_windows=int(n_windows),
        split=eval_split,
    )
    if indices.size == 0:
        raise ValueError(f"no rollout windows selected for split {eval_split!r}")
    return int(indices[0])


def run_suite(
    cond_samples: np.ndarray,
    batch: Any,
    model: FixedDeployableSampler,
    data_path: str,
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
    batch_size: int,
    conditionality_samples: int,
    conditionality_max_batches: int,
    device: torch.device,
    eval_split: str = "val",
) -> dict[str, Any]:
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    raw = np.load(data_path)
    returns = raw["ret"].astype(np.float64)
    rollout_start = rollout_start_for_split(
        test_start=test_start,
        val_size=val_size,
        history_len=history_len,
        future_len=future_len,
        n_windows=int(ground_truth.shape[0]),
        eval_split=eval_split,
    )

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        model,
        cond_loader,
        n_samples=conditionality_samples,
        max_batches=conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=history_len,
        future_len=future_len,
    )
    iv_ewma_economic_link = run_iv_ewma_economic_link_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=history_len,
        future_len=future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    risk_state_allocation = run_risk_state_allocation_tests(
        cond_samples,
        ground_truth,
        history_01,
    )
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "iv_ewma_economic_link": iv_ewma_economic_link,
        "regime_coverage": regime_coverage,
        "risk_state_allocation": risk_state_allocation,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    return results


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
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--residual_shape_scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=438)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
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
    print("Sampling calibration forecasts")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    bank, bank_meta = fit_residual_bank(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history_01=calib_hist_01.detach().cpu().numpy(),
        regime_bins=bool(args.regime_bins),
        min_bin_windows=args.min_bin_windows,
    )

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

    print("Sampling validation learned core")
    val_base_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    cond_samples, policy_meta = build_deployable_samples(
        val_base_samples=val_base_samples,
        val_history_01=batch.history_01.detach().cpu().numpy(),
        bank=bank,
        residual_shape_scale=args.residual_shape_scale,
        seed=args.seed,
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(hist_norm_np.shape[0])}
    model = FixedDeployableSampler(samples_by_key).eval()

    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=model,
        data_path=args.data_path,
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        batch_size=args.batch_size,
        conditionality_samples=args.conditionality_samples,
        conditionality_max_batches=args.conditionality_max_batches,
        device=device,
        eval_split="val",
    )

    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size
    results["config"] = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "deployability": {
            "kind": "pre_validation_residual_error_bootstrap",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_per_window_oracle_miss_placement": False,
            "calibration_split": "pre_validation",
            "calibration_windows": int(args.calibration_windows),
            "calibration_samples": int(args.calibration_samples),
            "base_center": "validation_392a_sample_median",
            "residual_error_source": "pre_validation_realized_future_minus_392a_sample_median",
            "regime_bins": bool(args.regime_bins),
            "min_bin_windows": int(args.min_bin_windows),
            "vov_q20": bank.vov_q20,
            "vov_q80": bank.vov_q80,
            "not_base_learned_law": True,
            "calibrated_risk_policy": True,
            **bank_meta,
            **policy_meta,
        },
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "conditionality_samples": int(args.conditionality_samples),
        "rollout_start": int(rollout_start),
        "seed": int(args.seed),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")

    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    distributional = results["distributional_fidelity"]
    regime = results["regime_coverage"]
    risk_state = results["risk_state_allocation"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    pathwise = results["pathwise_jump_realism"]
    lines = [
        "- policy: `pre-validation residual-error bootstrap around frozen 392a median`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- regime bins: `{bool(args.regime_bins)}`",
        f"- calibration windows/samples: `{args.calibration_windows}` / `{args.calibration_samples}`",
        f"- residual shape scale: `{args.residual_shape_scale:.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- risk-state allocation: `{risk_state['overall_pass']}`",
        f"- risk-state observable response: `{risk_state.get('observable_state_response_pass', False)}`",
        f"- risk-state oracle future alignment: `{risk_state.get('oracle_future_alignment_pass', False)}`",
        f"- risk-state width/history rho: `{risk_state.get('history_width_spearman', float('nan')):.3f}`",
        f"- risk-state width/future rho: `{risk_state['future_width_spearman']:.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "438a Deployable Residual Bootstrap Risk System", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
