#!/usr/bin/env python
"""
213a: dedicated H=1 conditional-distribution quality suite.

Goal:
  Evaluate one-day-ahead conditional distribution quality using the parts of the
  repo-wide 30d suite that remain meaningful for H=1.

Included:
  1. Surface validity
  2. Coverage + stress-subset coverage
  3. Conditionality vs shuffled history
  4. One-step move-size shape
  5. Regime and worst-cell coverage
  6. Unconditional marginal realism
  7. Cross-cell dependence and factor concentration
  8. Mean-reversion realism
  9. Extreme-move realism

Excluded as not meaningful for H=1:
  - multi-horizon ACF / temporal-shape persistence
  - block-AR rollout behavior
  - IV-EWMA cointegration over forward paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import ks_2samp, kurtosis

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    compute_move_size_profile,
    run_ci_coverage_tests,
    run_mean_reversion_tests,
    run_surface_validity_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    load_model as load_212b_model,
)
from experiments.backfill.block_ar.train_212c_h1_mean_residual_direct_delta import (
    load_model as load_212c_model,
)
from experiments.backfill.block_ar.train_212d_h1_state_basis_residual_direct_delta import (
    load_model as load_212d_model,
)
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import (
    load_model as load_212e_model,
)
from experiments.backfill.block_ar.train_212f_h1_asymmetric_modulated_direct_delta_quantile import (
    load_model as load_212f_model,
)
from experiments.backfill.block_ar.train_212g_h1_asymmetric_modulated_direct_delta_scale_supervision import (
    load_model as load_212g_model,
)
from experiments.backfill.block_ar.train_212h_h1_asymmetric_modulated_direct_delta_cvar_quantile import (
    load_model as load_212h_model,
)
from experiments.backfill.block_ar.train_212q_h1_asymmetric_modulated_direct_delta_learned_output_scale import (
    load_model as load_212q_model,
)
from experiments.backfill.block_ar.train_212r_h1_asymmetric_modulated_direct_delta_cvar_energy import (
    load_model as load_212r_model,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    load_model as load_212s_model,
)
from experiments.backfill.block_ar.train_212u_h1_minimal_direct_stochastic_delta_raw_output import (
    load_model as load_212u_model,
)
from experiments.backfill.block_ar.train_212v_h1_minimal_direct_stochastic_delta_tail_incidence_sampling import (
    load_model as load_212v_model,
)
from experiments.backfill.block_ar.train_212w_h1_minimal_direct_stochastic_delta_student_t_latent import (
    load_model as load_212w_model,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    load_model as load_212x_model,
)
from experiments.backfill.block_ar.train_212y_h1_minimal_direct_stochastic_delta_local_scale_asinh import (
    load_model as load_212y_model,
)
from experiments.backfill.block_ar.train_212z_h1_minimal_direct_stochastic_delta_local_scale_asinh_crps import (
    load_model as load_212z_model,
)
from experiments.backfill.block_ar.train_212aa_h1_minimal_direct_stochastic_delta_local_scale_asinh_tailcrps import (
    load_model as load_212aa_model,
)
from experiments.backfill.block_ar.train_212ac_h1_rectified_flow_local_scale_asinh import (
    load_model as load_212ac_model,
)
from experiments.backfill.block_ar.train_212ad_h1_conditional_diffusion_local_scale_asinh_terminal_es import (
    load_model as load_212ad_model,
)
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import (
    load_model as load_212ae_model,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    load_model as load_212af_model,
)
from experiments.backfill.block_ar.train_212ag_h1_conditional_flow_local_scale_asinh_mean_aux import (
    load_model as load_212ag_model,
)
from experiments.backfill.block_ar.train_212ah_h1_conditional_flow_local_scale_asinh_nll_mean import (
    load_model as load_212ah_model,
)
from experiments.backfill.block_ar.train_212ai_h1_conditional_flow_local_scale_asinh_staged_nll import (
    load_model as load_212ai_model,
)
from experiments.backfill.block_ar.train_212aj_h1_conditional_flow_local_scale_asinh_locscale_staged_nll import (
    load_model as load_212aj_model,
)
from experiments.backfill.block_ar.train_212ak_h1_conditional_flow_local_scale_asinh_location_staged_nll import (
    load_model as load_212ak_model,
)
from experiments.backfill.block_ar.train_212al_h1_conditional_flow_local_scale_asinh_location_frozen_stage import (
    load_model as load_212al_model,
)
from experiments.backfill.block_ar.train_212am_h1_conditional_flow_logit_drift_staged_nll import (
    load_model as load_212am_model,
)
from experiments.backfill.block_ar.train_212an_h1_conditional_flow_logit_drift_boundary_scale_staged_nll import (
    load_model as load_212an_model,
)
from experiments.backfill.block_ar.train_212ao_h1_conditional_flow_local_scale_asinh_staged_nll_bound_penalty import (
    load_model as load_212ao_model,
)
from experiments.backfill.block_ar.train_223a_generated_history_finetune import (
    load_model as load_223_model,
)
from experiments.backfill.block_ar.train_226a_factor_decoupled_flow import (
    load_model as load_226a_model,
)
from experiments.backfill.block_ar.train_212ab_h1_conditional_diffusion_local_scale_asinh import (
    load_model as load_212ab_model,
)


LoaderFn = Callable[[str, torch.device], tuple[torch.nn.Module, dict[str, Any]]]


def _get_loader(model_type: str) -> LoaderFn:
    mapping: dict[str, LoaderFn] = {
        "212b": load_212b_model,
        "212c": load_212c_model,
        "212d": load_212d_model,
        "212e": load_212e_model,
        "212f": load_212f_model,
        "212g": load_212g_model,
        "212h": load_212h_model,
        "212q": load_212q_model,
        "212r": load_212r_model,
        "212s": load_212s_model,
        "212u": load_212u_model,
        "212v": load_212v_model,
        "212w": load_212w_model,
        "212x": load_212x_model,
        "212y": load_212y_model,
        "212z": load_212z_model,
        "212aa": load_212aa_model,
        "212ac": load_212ac_model,
        "212ad": load_212ad_model,
        "212ae": load_212ae_model,
        "212af": load_212af_model,
        "212ag": load_212ag_model,
        "212ah": load_212ah_model,
        "212ai": load_212ai_model,
        "212aj": load_212aj_model,
        "212ak": load_212ak_model,
        "212al": load_212al_model,
        "212am": load_212am_model,
        "212an": load_212an_model,
        "212ao": load_212ao_model,
        "223a": load_223_model,
        "223b": load_223_model,
        "212ab": load_212ab_model,
        "226a": load_226a_model,
    }
    if model_type not in mapping:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return mapping[model_type]


def _sample_model(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
    paired_noise_histories: torch.Tensor | None = None,
) -> np.ndarray:
    """Sample next-day IV surfaces in batches.

    Returns shape (N, S, 1, 5, 5).
    """
    out = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        batch_hist = history_01[start:end]
        noise = None
        if paired_noise_histories is not None:
            noise = paired_noise_histories[start:end]
        samp = model.sample_next_iv(batch_hist, n_samples=n_samples, noise=noise)
        samp_np = samp.detach().cpu().numpy()
        if samp_np.ndim == 3:
            samp_np = samp_np.reshape(samp_np.shape[0], samp_np.shape[1], 1, 5, 5)
        elif samp_np.ndim == 5:
            pass
        else:
            raise ValueError(f"Unexpected sample_next_iv output shape: {samp_np.shape}")
        out.append(samp_np)
    return np.concatenate(out, axis=0)


def _paired_noise(model: torch.nn.Module, n_windows: int, n_samples: int, device: torch.device) -> torch.Tensor:
    noise_dim = getattr(model, "noise_dim", None)
    if noise_dim is None:
        raise ValueError("Model does not expose noise_dim for paired-noise sampling.")
    return torch.randn(n_windows, n_samples, int(noise_dim), device=device)


def _compute_h1_delta_arrays(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prev = history[:, -1]  # (N, 5, 5)
    gt_delta = ground_truth[:, 0] - prev  # (N, 5, 5)
    sample_delta = cond_samples[:, :, 0] - prev[:, None]  # (N, S, 5, 5)
    return gt_delta, sample_delta


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_h1_conditionality_tests(
    cond_samples: np.ndarray,
    shuffled_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 3: CONDITIONALITY")
    print("=" * 60)

    cond_lower = np.quantile(cond_samples, 0.05, axis=1)[:, 0]
    cond_upper = np.quantile(cond_samples, 0.95, axis=1)[:, 0]
    cond_median = np.median(cond_samples, axis=1)[:, 0]

    shuf_lower = np.quantile(shuffled_samples, 0.05, axis=1)[:, 0]
    shuf_upper = np.quantile(shuffled_samples, 0.95, axis=1)[:, 0]
    shuf_median = np.median(shuffled_samples, axis=1)[:, 0]

    gt_next = ground_truth[:, 0]
    cond_width = cond_upper - cond_lower
    shuf_width = shuf_upper - shuf_lower

    avg_cond_width = float(cond_width.mean())
    avg_shuf_width = float(shuf_width.mean())
    mean_abs_window_width_change = float(
        np.abs(cond_width.mean(axis=(1, 2)) - shuf_width.mean(axis=(1, 2))).mean()
    )

    avg_cond_mae = float(np.abs(cond_median - gt_next).mean())
    avg_shuf_mae = float(np.abs(shuf_median - gt_next).mean())
    mae_reduction_pct = (
        (avg_shuf_mae - avg_cond_mae) / max(avg_shuf_mae, 1e-8) * 100.0
    )
    mae_pass = mae_reduction_pct > 5.0

    cond_cell_width = cond_width.mean(axis=0)
    shuf_cell_width = shuf_width.mean(axis=0)
    cond_cell_mae = np.abs(cond_median - gt_next).mean(axis=0)
    shuf_cell_mae = np.abs(shuf_median - gt_next).mean(axis=0)

    cell_width_ratio = cond_cell_width / np.maximum(shuf_cell_width, 1e-8)
    worst_cell_width_ratio = float(cell_width_ratio.max())
    worst_cell_wr_pass = worst_cell_width_ratio < 1.20

    cell_mae_reduction = np.where(
        shuf_cell_mae > 1e-8,
        (shuf_cell_mae - cond_cell_mae) / shuf_cell_mae * 100.0,
        0.0,
    )
    worst_cell_mae_reduction = float(cell_mae_reduction.min())
    worst_cell_mae_pass = worst_cell_mae_reduction > -10.0

    mean_iv = history.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = float(np.quantile(vol_of_vol, 0.20))
    q80 = float(np.quantile(vol_of_vol, 0.80))
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80
    calm_avg_width = float(cond_width[calm_mask].mean()) if calm_mask.any() else 0.0
    turb_avg_width = float(cond_width[turb_mask].mean()) if turb_mask.any() else 0.0
    turb_calm_ratio = turb_avg_width / max(calm_avg_width, 1e-8)
    turb_calm_pass = turb_calm_ratio > 1.15

    realized_max_abs = np.abs(gt_next - history[:, -1]).max(axis=(1, 2))
    cond_window_width = cond_width.mean(axis=(1, 2))
    shuf_window_width = shuf_width.mean(axis=(1, 2))
    cond_width_realized_corr = _safe_corr(cond_window_width, realized_max_abs)
    shuf_width_realized_corr = _safe_corr(shuf_window_width, realized_max_abs)
    width_realized_corr_gain = cond_width_realized_corr - shuf_width_realized_corr

    print(f"  Mean |window-width change| vs shuffled: {mean_abs_window_width_change:.4f} (informational)")
    print(f"    Cond width:    {avg_cond_width:.4f}")
    print(f"    Shuffled width:{avg_shuf_width:.4f}")
    print(
        f"  MAE reduction vs shuffled: {mae_reduction_pct:.1f}% "
        f"(target >5%) {'PASS' if mae_pass else 'FAIL'}"
    )
    print(
        f"  Worst-cell width ratio: {worst_cell_width_ratio:.3f} "
        f"(target <1.20) {'PASS' if worst_cell_wr_pass else 'FAIL'}"
    )
    print(
        f"  Worst-cell MAE reduction: {worst_cell_mae_reduction:.1f}% "
        f"(target >-10%) {'PASS' if worst_cell_mae_pass else 'FAIL'}"
    )
    print(
        f"  Turb/Calm width ratio: {turb_calm_ratio:.3f} "
        f"(target >1.15) {'PASS' if turb_calm_pass else 'FAIL'}"
    )
    print(
        f"  Width vs realized-max corr: cond={cond_width_realized_corr:.3f}, "
        f"shuffled={shuf_width_realized_corr:.3f}, gain={width_realized_corr_gain:.3f} (informational)"
    )

    overall_pass = mae_pass and worst_cell_wr_pass and worst_cell_mae_pass and turb_calm_pass
    return {
        "mean_abs_window_width_change_vs_shuffled": mean_abs_window_width_change,
        "avg_cond_width": avg_cond_width,
        "avg_shuffled_width": avg_shuf_width,
        "mae_reduction_pct_vs_shuffled": float(mae_reduction_pct),
        "mae_pass": bool(mae_pass),
        "per_cell_width_ratio": cell_width_ratio.tolist(),
        "worst_cell_width_ratio": float(worst_cell_width_ratio),
        "worst_cell_wr_pass": bool(worst_cell_wr_pass),
        "per_cell_mae_reduction_pct": cell_mae_reduction.tolist(),
        "worst_cell_mae_reduction_pct": float(worst_cell_mae_reduction),
        "worst_cell_mae_pass": bool(worst_cell_mae_pass),
        "n_calm": int(calm_mask.sum()),
        "n_turb": int(turb_mask.sum()),
        "vol_of_vol_q20": q20,
        "vol_of_vol_q80": q80,
        "turb_calm_width_ratio": float(turb_calm_ratio),
        "turb_calm_pass": bool(turb_calm_pass),
        "width_realizedmax_corr_cond": float(cond_width_realized_corr),
        "width_realizedmax_corr_shuffled": float(shuf_width_realized_corr),
        "width_realizedmax_corr_gain": float(width_realized_corr_gain),
        "overall_pass": bool(overall_pass),
    }


def run_h1_move_size_shape_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 4: MOVE-SIZE SHAPE")
    print("=" * 60)

    gt_delta, sample_delta = _compute_h1_delta_arrays(cond_samples, ground_truth, history)
    gt_abs = np.abs(gt_delta.reshape(-1))
    gen_abs = np.abs(sample_delta.reshape(-1))

    gt_kurt = float(kurtosis(gt_delta.reshape(-1), fisher=True))
    gen_kurt = float(kurtosis(sample_delta.reshape(-1), fisher=True))
    kurt_ratio = gen_kurt / max(gt_kurt, 1e-8)
    kurt_gate_lo = 0.80
    kurt_gate_hi = 1.25
    kurt_pass = kurt_gate_lo <= kurt_ratio <= kurt_gate_hi

    print(f"  GT kurtosis:  {gt_kurt:.3f}")
    print(f"  Gen kurtosis: {gen_kurt:.3f}")
    print(
        f"  Kurtosis ratio: {kurt_ratio:.3f} "
        f"(target {kurt_gate_lo:.2f}-{kurt_gate_hi:.2f}) "
        f"{'PASS' if kurt_pass else 'FAIL'}"
    )

    per_cell_q99_ratio = np.zeros((5, 5), dtype=np.float64)
    per_cell_pass = np.zeros((5, 5), dtype=bool)
    n_samp = min(10, cond_samples.shape[1])
    for r in range(5):
        for c in range(5):
            gt_cell_abs = np.abs(gt_delta[:, r, c].ravel())
            gt_q99 = float(np.quantile(gt_cell_abs, 0.99))
            gen_q99_est = []
            for s_idx in range(n_samp):
                gen_cell_abs = np.abs(sample_delta[:, s_idx, r, c].ravel())
                gen_q99_est.append(float(np.quantile(gen_cell_abs, 0.99)))
            gen_q99 = float(np.median(gen_q99_est))
            ratio = gen_q99 / max(gt_q99, 1e-12)
            per_cell_q99_ratio[r, c] = ratio
            per_cell_pass[r, c] = np.isfinite(ratio) and (0.5 <= ratio <= 2.0)
    n_tail_pass = int(per_cell_pass.sum())
    tail_pass = n_tail_pass >= 20
    print(f"  Per-cell q99(|ΔIV|) scale: {n_tail_pass}/25 cells "
          f"(gate >=20) {'PASS' if tail_pass else 'FAIL'}")

    move_size_profile = compute_move_size_profile(gt_abs, gen_abs)
    move_size_pass = bool(move_size_profile["pass"])
    print("  Move-size shares:")
    for label, disp in [
        ("very_small_moves", "<=0.005"),
        ("small_moves", "<=0.010"),
        ("moderate_moves", "<=0.020"),
        ("large_moves", "<=0.050"),
    ]:
        bucket = move_size_profile[label]
        print(
            f"    {disp}: ratio={bucket['ratio']:.3f} "
            f"(GT={bucket['gt_share']:.1%}, gen={bucket['gen_share']:.1%}) "
            f"{'PASS' if bucket['pass'] else 'FAIL'}"
        )

    overall_pass = kurt_pass and tail_pass and move_size_pass
    return {
        "kurtosis": {
            "gt_kurtosis": gt_kurt,
            "gen_kurtosis": gen_kurt,
            "kurtosis_ratio": float(kurt_ratio),
            "gate_lo": kurt_gate_lo,
            "gate_hi": kurt_gate_hi,
            "pass": bool(kurt_pass),
        },
        "tail_scale": {
            "per_cell_q99_ratio": per_cell_q99_ratio.tolist(),
            "n_pass": int(n_tail_pass),
            "gate_lo": 0.5,
            "gate_hi": 2.0,
            "pass": bool(tail_pass),
        },
        "move_size_profile": move_size_profile,
        "overall_pass": bool(overall_pass),
    }


def run_h1_regime_coverage_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 7: REGIME AND WORST-CELL COVERAGE")
    print("=" * 60)

    lower = np.quantile(cond_samples, 0.05, axis=1)[:, 0]
    upper = np.quantile(cond_samples, 0.95, axis=1)[:, 0]
    covered = (ground_truth[:, 0] >= lower) & (ground_truth[:, 0] <= upper)

    mean_iv = history.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = float(np.quantile(vol_of_vol, 0.20))
    q80 = float(np.quantile(vol_of_vol, 0.80))
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    layer1 = {}
    layer1_pass = True
    for name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        cov = float(covered[mask].mean()) if mask.any() else float("nan")
        passed = bool(np.isfinite(cov) and cov > 0.65)
        layer1[name] = {"coverage": cov, "pass": passed, "n_windows": int(mask.sum())}
        layer1_pass = layer1_pass and passed
        print(
            f"  {name:5s} average coverage: {cov:.1%} "
            f"(target >65%) {'PASS' if passed else 'FAIL'}"
        )

    layer2 = {}
    layer2_pass = True
    for name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        if not mask.any():
            layer2[name] = {"pass": False}
            layer2_pass = False
            continue
        cell_cov = covered[mask].mean(axis=0)
        worst = float(cell_cov.min())
        best = float(cell_cov.max())
        worst_idx = np.unravel_index(cell_cov.argmin(), (5, 5))
        best_idx = np.unravel_index(cell_cov.argmax(), (5, 5))
        passed = bool((worst >= 0.70) and (best <= 0.95))
        layer2[name] = {
            "grid": cell_cov.tolist(),
            "worst": worst,
            "worst_cell": list(worst_idx),
            "best": best,
            "best_cell": list(best_idx),
            "pass": passed,
        }
        layer2_pass = layer2_pass and passed
        print(
            f"  {name:5s} worst cell {worst_idx}={worst:.1%}, "
            f"best cell {best_idx}={best:.1%} "
            f"(gate [70%,95%]) {'PASS' if passed else 'FAIL'}"
        )

    overall_pass = layer1_pass and layer2_pass
    return {
        "layer1_regime_coverage": layer1,
        "layer1_pass": bool(layer1_pass),
        "layer2_regime_cell_coverage": layer2,
        "layer2_pass": bool(layer2_pass),
        "overall_pass": bool(overall_pass),
        "notes": "Layer 3 persistent undercoverage omitted for H=1 because there is no multi-horizon persistence to average over.",
    }


def run_h1_distributional_fidelity_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 8: UNCONDITIONAL MARGINAL REALISM")
    print("=" * 60)

    N, S = cond_samples.shape[:2]
    prev = history[:, -1]
    gt_diff = ground_truth[:, 0] - prev
    n_samp_ks = min(5, S)
    gen_diff = cond_samples[:, :n_samp_ks, 0] - prev[:, None]

    ks_grid = np.zeros((5, 5), dtype=np.float64)
    ks_pass_grid = np.zeros((5, 5), dtype=bool)
    KS_GATE = 0.15
    for r in range(5):
        for c in range(5):
            stat, _p = ks_2samp(gt_diff[:, r, c].ravel(), gen_diff[:, :, r, c].ravel())
            ks_grid[r, c] = stat
            ks_pass_grid[r, c] = stat < KS_GATE
    n_ks_pass = int(ks_pass_grid.sum())
    ks_overall_pass = n_ks_pass >= 15
    print(f"  Daily-change KS pass cells: {n_ks_pass}/25 {'PASS' if ks_overall_pass else 'FAIL'}")

    level_ks_grid = np.zeros((5, 5), dtype=np.float64)
    level_ks_pass_grid = np.zeros((5, 5), dtype=bool)
    for r in range(5):
        for c in range(5):
            stat, _p = ks_2samp(ground_truth[:, 0, r, c].ravel(), cond_samples[:, :n_samp_ks, 0, r, c].ravel())
            level_ks_grid[r, c] = stat
            level_ks_pass_grid[r, c] = stat < KS_GATE
    n_level_ks_pass = int(level_ks_pass_grid.sum())
    level_ks_overall_pass = n_level_ks_pass >= 15
    print(f"  Level KS pass cells: {n_level_ks_pass}/25 {'PASS' if level_ks_overall_pass else 'FAIL'}")

    median_pred = np.median(cond_samples, axis=1)[:, 0]
    gt_next = ground_truth[:, 0]
    above_frac = (median_pred > gt_next).mean(axis=0)
    mean_bias = (median_pred - gt_next).mean(axis=0)
    bias_pass_grid = (above_frac >= 0.30) & (above_frac <= 0.70)
    n_bias_pass = int(bias_pass_grid.sum())
    bias_frac_pass = n_bias_pass >= 20
    bias_mag_pass_grid = np.abs(mean_bias) < 0.03
    n_bias_mag_pass = int(bias_mag_pass_grid.sum())
    bias_mag_pass = n_bias_mag_pass >= 22
    bias_overall_pass = bias_frac_pass and bias_mag_pass

    q05 = np.quantile(cond_samples, 0.05, axis=1)[:, 0]
    q95 = np.quantile(cond_samples, 0.95, axis=1)[:, 0]
    covered = (gt_next >= q05) & (gt_next <= q95)
    per_window_cov = covered.mean(axis=(1, 2))
    n_bad_windows = int((per_window_cov < 0.50).sum())
    pct_bad = n_bad_windows / max(N, 1)
    window_floor_pass = pct_bad < 0.05

    at_ceiling = float((cond_samples >= 0.99).mean())
    at_floor = float((cond_samples <= 0.001).mean())
    ceiling_per_cell = (cond_samples >= 0.99).mean(axis=(0, 1, 2))
    floor_per_cell = (cond_samples <= 0.001).mean(axis=(0, 1, 2))
    worst_cell_ceiling = float(ceiling_per_cell.max())
    worst_cell_floor = float(floor_per_cell.max())
    explosion_pass = (at_ceiling < 0.02) and (at_floor < 0.02) and (worst_cell_ceiling < 0.05) and (worst_cell_floor < 0.05)

    cell_mae = np.abs(median_pred - gt_next).mean(axis=0)
    mae_pass_grid = cell_mae < 0.10
    n_mae_pass = int(mae_pass_grid.sum())
    mae_overall_pass = n_mae_pass >= 20

    overall_pass = (
        ks_overall_pass
        and level_ks_overall_pass
        and bias_overall_pass
        and window_floor_pass
        and explosion_pass
        and mae_overall_pass
    )
    return {
        "ks_test": {
            "ks_grid": ks_grid.tolist(),
            "ks_gate": KS_GATE,
            "n_pass": int(n_ks_pass),
            "worst_stat": float(ks_grid.max()),
            "median_stat": float(np.median(ks_grid)),
            "pass": bool(ks_overall_pass),
        },
        "ks_level_test": {
            "ks_grid": level_ks_grid.tolist(),
            "ks_gate": KS_GATE,
            "n_pass": int(n_level_ks_pass),
            "worst_stat": float(level_ks_grid.max()),
            "median_stat": float(np.median(level_ks_grid)),
            "pass": bool(level_ks_overall_pass),
        },
        "median_bias": {
            "above_frac": above_frac.tolist(),
            "mean_bias": mean_bias.tolist(),
            "n_pass": int(n_bias_pass),
            "frac_pass": bool(bias_frac_pass),
            "n_mag_pass": int(n_bias_mag_pass),
            "mag_pass": bool(bias_mag_pass),
            "pass": bool(bias_overall_pass),
        },
        "window_floor": {
            "n_bad_windows": int(n_bad_windows),
            "pct_bad": float(pct_bad),
            "worst_window_cov": float(per_window_cov.min()),
            "p10_cov": float(np.percentile(per_window_cov, 10)),
            "pass": bool(window_floor_pass),
        },
        "explosion": {
            "at_ceiling": at_ceiling,
            "at_floor": at_floor,
            "ceiling_per_cell": ceiling_per_cell.tolist(),
            "worst_cell_ceiling": worst_cell_ceiling,
            "worst_cell_floor": worst_cell_floor,
            "pass": bool(explosion_pass),
        },
        "cell_mae": {
            "mae_grid": cell_mae.tolist(),
            "n_pass": int(n_mae_pass),
            "worst_mae": float(cell_mae.max()),
            "pass": bool(mae_overall_pass),
        },
        "overall_pass": bool(overall_pass),
    }


def run_h1_cross_cell_dependence_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 9: CROSS-CELL DEPENDENCE AND FACTOR CONCENTRATION")
    print("=" * 60)

    prev = history[:, -1]
    gt_changes = (ground_truth[:, 0] - prev).reshape(ground_truth.shape[0], -1)
    n_corr_samples = min(5, cond_samples.shape[1])
    gen_corr_mats = []
    for s_idx in range(n_corr_samples):
        gen_changes = (cond_samples[:, s_idx, 0] - prev).reshape(cond_samples.shape[0], -1)
        gen_corr_mats.append(np.corrcoef(gen_changes.T))
    gen_corr = np.mean(gen_corr_mats, axis=0)
    gt_corr = np.corrcoef(gt_changes.T)

    n_cells = gt_changes.shape[1]
    mask = np.triu(np.ones((n_cells, n_cells), dtype=bool), k=1)
    gt_mean_corr = float(gt_corr[mask].mean())
    gen_mean_corr = float(gen_corr[mask].mean())
    corr_ratio = gen_mean_corr / gt_mean_corr if abs(gt_mean_corr) > 1e-8 else float("inf")
    corr_pass = 0.5 <= corr_ratio <= 2.0

    def eff_rank(eigvals: np.ndarray) -> float:
        p = eigvals / max(eigvals.sum(), 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    gt_e = np.maximum(np.linalg.eigvalsh(gt_corr)[::-1], 0.0)
    gen_e = np.maximum(np.linalg.eigvalsh(gen_corr)[::-1], 0.0)
    gt_rank = eff_rank(gt_e)
    gen_rank = eff_rank(gen_e)
    rank_ratio = gen_rank / max(gt_rank, 1e-8)
    rank_pass = 0.5 <= rank_ratio <= 3.0

    overall_pass = corr_pass and rank_pass
    print(f"  Mean correlation ratio: {corr_ratio:.3f} {'PASS' if corr_pass else 'FAIL'}")
    print(f"  Factor breadth ratio:   {rank_ratio:.3f} {'PASS' if rank_pass else 'FAIL'}")
    return {
        "gt_mean_corr": gt_mean_corr,
        "gen_mean_corr": gen_mean_corr,
        "corr_ratio": float(corr_ratio),
        "corr_pass": bool(corr_pass),
        "gt_eff_rank": float(gt_rank),
        "gen_eff_rank": float(gen_rank),
        "rank_ratio": float(rank_ratio),
        "rank_pass": bool(rank_pass),
        "frob_dist": float(np.linalg.norm(gen_corr - gt_corr, "fro")),
        "gt_pc1_var": float(gt_e[0] / max(gt_e.sum(), 1e-10)),
        "gen_pc1_var": float(gen_e[0] / max(gen_e.sum(), 1e-10)),
        "overall_pass": bool(overall_pass),
    }


def run_h1_extreme_move_tests(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("H1 SUITE 11: EXTREME-MOVE REALISM")
    print("=" * 60)

    gt_delta, sample_delta = _compute_h1_delta_arrays(cond_samples, ground_truth, history)
    gt_path_max = np.abs(gt_delta).max(axis=(1, 2))
    gen_path_max = np.abs(sample_delta).max(axis=(2, 3)).reshape(-1)
    maxjump_ks, _ = ks_2samp(gt_path_max, gen_path_max)
    gt_q90 = float(np.quantile(gt_path_max, 0.90))
    gt_q99 = float(np.quantile(gt_path_max, 0.99))
    gen_q90 = float(np.quantile(gen_path_max, 0.90))
    gen_q99 = float(np.quantile(gen_path_max, 0.99))
    q90_ratio = gen_q90 / max(gt_q90, 1e-12)
    q99_ratio = gen_q99 / max(gt_q99, 1e-12)
    maxjump_ks_pass = maxjump_ks < 0.20
    qtail_pass = (0.5 <= q90_ratio <= 2.0) and (0.5 <= q99_ratio <= 2.0)

    print(
        f"  Max-|ΔIV| KS: {maxjump_ks:.3f} "
        f"(gate <0.20) {'PASS' if maxjump_ks_pass else 'FAIL'}"
    )
    print(
        f"  q90 ratio: {q90_ratio:.3f}, q99 ratio: {q99_ratio:.3f} "
        f"{'PASS' if qtail_pass else 'FAIL'}"
    )

    per_cell_q99_ratio = np.zeros((5, 5), dtype=np.float64)
    per_cell_pass = np.zeros((5, 5), dtype=bool)
    n_jump_samples = min(10, cond_samples.shape[1])
    for r in range(5):
        for c in range(5):
            gt_abs = np.abs(gt_delta[:, r, c].ravel())
            gt_cell_q99 = float(np.quantile(gt_abs, 0.99))
            gen_est = []
            for s_idx in range(n_jump_samples):
                gen_abs = np.abs(sample_delta[:, s_idx, r, c].ravel())
                gen_est.append(float(np.quantile(gen_abs, 0.99)))
            gen_cell_q99 = float(np.median(gen_est))
            ratio = gen_cell_q99 / max(gt_cell_q99, 1e-12)
            per_cell_q99_ratio[r, c] = ratio
            per_cell_pass[r, c] = np.isfinite(ratio) and (0.5 <= ratio <= 2.0)
    n_cell_pass = int(per_cell_pass.sum())
    per_cell_overall_pass = n_cell_pass >= 20

    global_gt_q99 = float(np.quantile(np.abs(gt_delta).ravel(), 0.99))
    gt_window_extreme = (np.abs(gt_delta) >= global_gt_q99).any(axis=(1, 2))
    gen_window_extreme = (np.abs(sample_delta) >= global_gt_q99).any(axis=(2, 3)).reshape(-1)
    gt_rate = float(gt_window_extreme.mean())
    gen_rate = float(gen_window_extreme.mean())
    incidence_ratio = gen_rate / max(gt_rate, 1e-12)
    incidence_pass = 0.5 <= incidence_ratio <= 2.0

    overall_pass = maxjump_ks_pass and qtail_pass and per_cell_overall_pass and incidence_pass
    return {
        "pathwise_max_jump": {
            "ks_stat": float(maxjump_ks),
            "ks_gate": 0.20,
            "q90_ratio": float(q90_ratio),
            "q99_ratio": float(q99_ratio),
            "pass": bool(maxjump_ks_pass and qtail_pass),
        },
        "per_cell_q99": {
            "ratio_grid": per_cell_q99_ratio.tolist(),
            "n_pass": int(n_cell_pass),
            "gate_lo": 0.5,
            "gate_hi": 2.0,
            "pass": bool(per_cell_overall_pass),
        },
        "window_extreme_incidence": {
            "gt_rate": gt_rate,
            "gen_rate": gen_rate,
            "ratio": float(incidence_ratio),
            "pass": bool(incidence_pass),
        },
        "overall_pass": bool(overall_pass),
    }


def summarize_suite(results: dict[str, Any]) -> dict[str, Any]:
    suites = [
        ("surface_validity", results["surface_validity"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("move_size_shape", results["move_size_shape"]["overall_pass"]),
        ("regime_coverage", results["regime_coverage"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_dependence", results["cross_cell_dependence"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("extreme_move_realism", results["extreme_move_realism"]["overall_pass"]),
    ]
    n_pass = int(sum(int(flag) for _, flag in suites))
    return {
        "suite_passes": {name: bool(flag) for name, flag in suites},
        "n_pass": n_pass,
        "n_total": len(suites),
        "overall_pass": bool(n_pass == len(suites)),
    }


def build_markdown_report(results: dict[str, Any]) -> str:
    summary = results["summary"]
    cov = results["coverage"]
    cond = results["conditionality"]
    shape = results["move_size_shape"]
    reg = results["regime_coverage"]
    df = results["distributional_fidelity"]
    dep = results["cross_cell_dependence"]
    mr = results["mean_reversion"]
    jump = results["extreme_move_realism"]

    mp = shape["move_size_profile"]
    overall_cov = cov["overall"].get("0.9", cov["overall"].get(0.9))
    lines = [
        "# 213a H=1 Conditional Distribution Suite",
        "",
        f"- Model: `{results['config']['model_type']}`",
        f"- Checkpoint: `{results['config']['checkpoint']}`",
        f"- Validation windows: `{results['config']['n_val_windows']}`",
        f"- Samples per window: `{results['config']['eval_samples']}`",
        f"- Score: `{summary['n_pass']}/{summary['n_total']}`",
        "",
        "## Key metrics",
        "",
        f"- Overall 90% coverage: `{overall_cov:.4f}`",
        f"- Realized q99 90% coverage: `{cov['stress_coverage']['realized_q99_coverage_90']:.4f}`",
        f"- Conditional MAE reduction vs shuffled: `{cond['mae_reduction_pct_vs_shuffled']:.1f}%`",
        f"- Width/realized-max corr gain vs shuffled: `{cond['width_realizedmax_corr_gain']:.3f}`",
        f"- Turb/calm width ratio: `{cond['turb_calm_width_ratio']:.3f}`",
        f"- Kurtosis ratio: `{shape['kurtosis']['kurtosis_ratio']:.3f}`",
        f"- Move shares ratio: `<=0.005 {mp['very_small_moves']['ratio']:.3f}, <=0.010 {mp['small_moves']['ratio']:.3f}, <=0.020 {mp['moderate_moves']['ratio']:.3f}, <=0.050 {mp['large_moves']['ratio']:.3f}`",
        f"- Calm average coverage: `{reg['layer1_regime_coverage']['calm']['coverage']:.4f}`",
        f"- Turb average coverage: `{reg['layer1_regime_coverage']['turb']['coverage']:.4f}`",
        f"- Daily-change KS pass cells: `{df['ks_test']['n_pass']}/25`",
        f"- Level KS pass cells: `{df['ks_level_test']['n_pass']}/25`",
        f"- Cross-cell correlation ratio: `{dep['corr_ratio']:.3f}`",
        f"- Factor breadth ratio: `{dep['rank_ratio']:.3f}`",
        f"- Mean-reversion slope ratio: `{mr['mr_gt_ratio']:.3f}`",
        f"- Extreme-move max-jump KS: `{jump['pathwise_max_jump']['ks_stat']:.3f}`",
        "",
        "## Suite passes",
        "",
    ]
    for name, passed in summary["suite_passes"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="213a H=1 conditional-distribution suite")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "212b",
            "212c",
            "212d",
            "212e",
            "212f",
            "212g",
            "212h",
            "212q",
            "212r",
            "212s",
            "212u",
            "212v",
            "212w",
            "212x",
            "212y",
            "212z",
            "212aa",
            "212ac",
            "212ad",
            "212ae",
            "212af",
            "212ag",
            "212ah",
            "212ai",
            "212aj",
            "212ak",
            "212al",
            "212am",
            "212an",
            "212ao",
            "212ab",
            "226a",
        ],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--shuffle_seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader_fn = _get_loader(args.model_type)
    model, payload = loader_fn(args.checkpoint, device)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    history_np = history_01.detach().cpu().numpy()
    ground_truth_np = target_01.detach().cpu().numpy().reshape(target_01.shape[0], 1, 5, 5)

    paired_noise = _paired_noise(model, history_01.shape[0], args.eval_samples, device)
    cond_samples = _sample_model(model, history_01, args.eval_samples, args.batch_size, paired_noise)
    rng = np.random.default_rng(args.shuffle_seed)
    shuffled_idx = rng.permutation(history_01.shape[0])
    shuffled_history = history_01[shuffled_idx]
    shuffled_samples = _sample_model(model, shuffled_history, args.eval_samples, args.batch_size, paired_noise)

    results: dict[str, Any] = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "epoch": int(payload.get("epoch", -1)),
            "history_len": args.history_len,
            "test_start": args.test_start,
            "val_size": args.val_size,
            "n_val_windows": int(history_01.shape[0]),
            "eval_samples": args.eval_samples,
            "shuffle_seed": args.shuffle_seed,
            "thresholds": {"q95": q95, "q99": q99},
        },
        "surface_validity": run_surface_validity_tests(cond_samples, ground_truth_np),
        "coverage": run_ci_coverage_tests(cond_samples, ground_truth_np, horizons=[1], ci_levels=[0.5, 0.8, 0.9, 0.95]),
        "conditionality": run_h1_conditionality_tests(cond_samples, shuffled_samples, ground_truth_np, history_np),
        "move_size_shape": run_h1_move_size_shape_tests(cond_samples, ground_truth_np, history_np),
        "regime_coverage": run_h1_regime_coverage_tests(cond_samples, ground_truth_np, history_np),
        "distributional_fidelity": run_h1_distributional_fidelity_tests(cond_samples, ground_truth_np, history_np),
        "cross_cell_dependence": run_h1_cross_cell_dependence_tests(cond_samples, ground_truth_np, history_np),
        "mean_reversion": run_mean_reversion_tests(cond_samples, ground_truth_np, history_np),
        "extreme_move_realism": run_h1_extreme_move_tests(cond_samples, ground_truth_np, history_np),
    }

    q05 = np.quantile(cond_samples, 0.05, axis=1)[:, 0]
    q95s = np.quantile(cond_samples, 0.95, axis=1)[:, 0]
    prev = history_np[:, -1]
    target_delta_abs = np.abs(ground_truth_np[:, 0] - prev)
    q95_mask = target_delta_abs >= q95
    q99_mask = target_delta_abs >= q99
    results["coverage"]["stress_coverage"] = {
        "realized_q95_coverage_90": float(((ground_truth_np[:, 0][q95_mask] >= q05[q95_mask]) & (ground_truth_np[:, 0][q95_mask] <= q95s[q95_mask])).mean()) if q95_mask.any() else float("nan"),
        "realized_q99_coverage_90": float(((ground_truth_np[:, 0][q99_mask] >= q05[q99_mask]) & (ground_truth_np[:, 0][q99_mask] <= q95s[q99_mask])).mean()) if q99_mask.any() else float("nan"),
        "q95_cell_count": int(q95_mask.sum()),
        "q99_cell_count": int(q99_mask.sum()),
        "q99_gate": 0.58,
        "q99_pass": bool((((ground_truth_np[:, 0][q99_mask] >= q05[q99_mask]) & (ground_truth_np[:, 0][q99_mask] <= q95s[q99_mask])).mean()) >= 0.58) if q99_mask.any() else False,
    }
    results["summary"] = summarize_suite(results)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown_report(results))

    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
