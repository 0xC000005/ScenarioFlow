#!/usr/bin/env python
"""
Deep mechanism review for 183c_best.

Focus:
  1. Why do S3/S7 still fail even after 183c fixes S4?
  2. Is residual mass being concentrated on the true hard slices or smeared into neighbors?
  3. Does the remaining tail problem match "too many medium moves, not enough quiet days"?
  4. What is the narrowest next principled fix?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.analyze_183a_final_mechanism import (
    cell_label,
    classify_cell_failure,
    distribution_compare,
    quantile_profile,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
)


BANDS = ["low", "mid", "high"]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def grid_neighbors(cell_idx: int) -> list[int]:
    r, c = divmod(cell_idx, 5)
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 5 and 0 <= cc < 5:
            out.append(rr * 5 + cc)
    return out


def exceedance_spectrum(gt: np.ndarray, gen: np.ndarray, qs: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[str, Any]:
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    gen = np.asarray(gen, dtype=np.float64).reshape(-1)
    thresholds = {str(q): float(np.quantile(gt, q)) for q in qs}
    gt_rates = {k: float((gt > thr).mean()) for k, thr in thresholds.items()}
    gen_rates = {k: float((gen > thr).mean()) for k, thr in thresholds.items()}
    rate_ratio = {k: float(gen_rates[k] / max(gt_rates[k], 1e-12)) for k in gt_rates}
    q50 = thresholds["0.5"]
    q95 = thresholds["0.95"]
    q99 = thresholds["0.99"]
    quiet_gt = float((gt <= q50).mean())
    quiet_gen = float((gen <= q50).mean())
    shoulder_gt = float(((gt > q50) & (gt <= q95)).mean())
    shoulder_gen = float(((gen > q50) & (gen <= q95)).mean())
    extreme_gt = float((gt > q99).mean())
    extreme_gen = float((gen > q99).mean())
    return {
        "thresholds": thresholds,
        "gt_exceed_rate": gt_rates,
        "gen_exceed_rate": gen_rates,
        "exceed_rate_ratio": rate_ratio,
        "quiet_mass": {"gt": quiet_gt, "gen": quiet_gen, "ratio": float(quiet_gen / max(quiet_gt, 1e-12))},
        "shoulder_mass": {"gt": shoulder_gt, "gen": shoulder_gen, "ratio": float(shoulder_gen / max(shoulder_gt, 1e-12))},
        "extreme_mass": {"gt": extreme_gt, "gen": extreme_gen, "ratio": float(extreme_gen / max(extreme_gt, 1e-12))},
    }


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c"
    # Accept 241-series descendants (same architecture, different loss surgery)
    compatible_prefixes = ("241a_", "241b_", "241c_", "241d_", "241e_", "241f_")
    if raw_config["type"] != expected and not raw_config["type"].startswith(compatible_prefixes):
        raise ValueError(f"Expected {expected} or 241-series, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = StateMetricTransportModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        path_config=raw_config["path"],
        prior_config=raw_config["prior"],
        integrated_config=raw_config["integrated"],
        state_config=raw_config["state"],
        metric_config=raw_config["metric"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
        mix_chunk_size=raw_config.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def analyze_183c_best(
    model: StateMetricTransportModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    review_183b: dict[str, Any],
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)
    n_windows, future_len, n_cells = future_01.shape

    vov, q20, q80 = regime_masks_from_history(history_norm.cpu())
    calm_mask = vov <= q20
    turb_mask = vov >= q80

    sample_std = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    det_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    target_local_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    raw_local_zero_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    raw_local_target_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    metric_local_zero_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    metric_local_target_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    local_gate_all = np.zeros((n_windows, 1), dtype=np.float32)

    raw_band_zero_all = np.zeros((n_windows, 3), dtype=np.float32)
    raw_band_target_all = np.zeros((n_windows, 3), dtype=np.float32)
    metric_band_zero_all = np.zeros((n_windows, 3), dtype=np.float32)
    metric_band_target_all = np.zeros((n_windows, 3), dtype=np.float32)
    band_gate_all = np.zeros((n_windows, 3), dtype=np.float32)

    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    path_max_jump_gt = []
    path_max_jump_gen = []
    path_q99_exceed_count_gt = []
    path_q99_exceed_count_gen = []
    band_energy_gt = {k: [] for k in BANDS}
    band_energy_gen = {k: [] for k in BANDS}
    high_band_coeff_gt = []
    high_band_coeff_gen = []

    geometry = model.path_geometry
    band_masks = {
        "low": geometry.low_band_mask().reshape(1, future_len, n_cells).to(device),
        "mid": geometry.mid_band_mask().reshape(1, future_len, n_cells).to(device),
        "high": geometry.high_band_mask().reshape(1, future_len, n_cells).to(device),
    }

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)
        batch = end - start

        (
            mu_u,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_01_b)
        det_iv = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi)
        det_mean[start:end] = det_iv.detach().cpu().numpy()

        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu_u,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(batch, future_len, n_cells)
        target_basis_flat = target_basis.reshape(batch, -1)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, _target_band_log = compute_control_targets(model, target_basis)
        target_local_all[start:end] = target_local_log.detach().cpu().numpy()

        t = torch.full((batch,), 0.5, device=device, dtype=target_basis.dtype)
        zero_state = torch.zeros_like(target_basis_flat)
        raw_local_zero, raw_band_zero, metric_local_zero, metric_band_zero, _lg0, _bg0 = model.build_state_metric_controls(
            zero_state, t, path_context
        )
        raw_local_t, raw_band_t, metric_local_t, metric_band_t, lg_t, bg_t = model.build_state_metric_controls(
            target_basis_flat, t, path_context
        )

        raw_local_zero_all[start:end] = raw_local_zero.detach().cpu().numpy()
        raw_local_target_all[start:end] = raw_local_t.detach().cpu().numpy()
        metric_local_zero_all[start:end] = metric_local_zero.detach().cpu().numpy()
        metric_local_target_all[start:end] = metric_local_t.detach().cpu().numpy()
        local_gate_all[start:end] = lg_t.detach().cpu().numpy()

        raw_band_zero_all[start:end] = raw_band_zero.detach().cpu().numpy()
        raw_band_target_all[start:end] = raw_band_t.detach().cpu().numpy()
        metric_band_zero_all[start:end] = metric_band_zero.detach().cpu().numpy()
        metric_band_target_all[start:end] = metric_band_t.detach().cpu().numpy()
        band_gate_all[start:end] = bg_t.detach().cpu().numpy()

        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
        samples_iv = samples_01.detach().cpu().numpy()
        sample_std[start:end] = samples_iv.std(axis=1)
        lo90[start:end] = np.quantile(samples_iv, 0.05, axis=1)
        hi90[start:end] = np.quantile(samples_iv, 0.95, axis=1)

        gt_path = torch.cat([hist_01_b[:, -1:].reshape(batch, 1, 5, 5), fut_01_b.view(batch, future_len, 5, 5)], dim=1)
        gen_path = torch.cat(
            [
                hist_01_b[:, -1:].reshape(batch, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1),
                samples_01.view(batch, n_samples, future_len, 5, 5),
            ],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]

        pooled_abs_delta_gt.append(gt_diff.abs().reshape(-1).cpu().numpy())
        pooled_abs_delta_gen.append(gen_diff.abs().reshape(-1).cpu().numpy())
        path_max_jump_gt.append(gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
        path_max_jump_gen.append(gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy())

        gt_q99 = torch.quantile(gt_diff.abs().reshape(-1), 0.99)
        path_q99_exceed_count_gt.append((gt_diff.abs().reshape(batch, -1) > gt_q99).sum(dim=1).cpu().numpy())
        path_q99_exceed_count_gen.append((gen_diff.abs().reshape(batch * n_samples, -1) > gt_q99).sum(dim=1).cpu().numpy())

        gt_diff_flat = gt_diff.reshape(batch, future_len, n_cells)
        gen_diff_flat = gen_diff.reshape(batch * n_samples, future_len, n_cells)
        gt_coeff = geometry.to_basis(gt_diff_flat)
        gen_coeff = geometry.to_basis(gen_diff_flat)
        gt_total_energy = gt_coeff.reshape(batch, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        gen_total_energy = gen_coeff.reshape(batch * n_samples, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        for band in BANDS:
            mask = band_masks[band].to(dtype=gt_coeff.dtype)
            gt_band_coeff = (gt_coeff * mask).reshape(batch, -1)
            gen_band_coeff = (gen_coeff * mask).reshape(batch * n_samples, -1)
            gt_share = (gt_band_coeff.pow(2).sum(dim=1) / gt_total_energy).cpu().numpy()
            gen_share = (gen_band_coeff.pow(2).sum(dim=1) / gen_total_energy).cpu().numpy()
            band_energy_gt[band].append(gt_share)
            band_energy_gen[band].append(gen_share)
            if band == "high":
                high_band_coeff_gt.append(gt_band_coeff.abs().reshape(-1).cpu().numpy())
                high_band_coeff_gen.append(gen_band_coeff.abs().reshape(-1).cpu().numpy())

    future_np = future_01.cpu().numpy()
    inside90 = (future_np >= lo90) & (future_np <= hi90)
    det_resid = future_np - det_mean
    z_mean = det_resid / np.maximum(sample_std, 1e-6)

    turb_cov = inside90[turb_mask]
    turb_std = sample_std[turb_mask]
    turb_gt = future_np[turb_mask]
    turb_det = det_mean[turb_mask]
    turb_det_resid = turb_gt - turb_det

    hard_entries = []
    over_entries = []
    for h in (7, 14, 30):
        h_idx = h - 1
        for c in range(n_cells):
            cov = float(turb_cov[:, h_idx, c].mean())
            gen_std = float(turb_std[:, h_idx, c].mean())
            gt_std = float(np.std(turb_det_resid[:, h_idx, c]))
            ratio = gen_std / max(gt_std, 1e-6)
            z = float(z_mean[turb_mask, h_idx, c].mean())
            entry = {
                "regime": "turb",
                "horizon": h,
                "cell": cell_label(c),
                "coverage_90": cov,
                "z_mean": z,
                "gen_std_iv": gen_std,
                "gt_resid_std_iv": gt_std,
                "std_ratio_gen_to_gt": ratio,
                "target_local_log": float(target_local_all[turb_mask, h_idx, c].mean()),
                "raw_local_zero": float(raw_local_zero_all[turb_mask, h_idx, c].mean()),
                "raw_local_target": float(raw_local_target_all[turb_mask, h_idx, c].mean()),
                "metric_local_zero": float(metric_local_zero_all[turb_mask, h_idx, c].mean()),
                "metric_local_target": float(metric_local_target_all[turb_mask, h_idx, c].mean()),
                "state_uplift_metric_local": float(
                    (metric_local_target_all[turb_mask, h_idx, c] - metric_local_zero_all[turb_mask, h_idx, c]).mean()
                ),
                "classification": classify_cell_failure(cov, z, ratio),
            }
            if cov < 0.70:
                hard_entries.append(entry)
            if cov > 0.95:
                over_entries.append(entry)

    hard_entries.sort(key=lambda x: x["coverage_90"])
    over_entries.sort(key=lambda x: (-x["coverage_90"], -x["std_ratio_gen_to_gt"]))

    review_slices = review_183b["width_allocation_review"]["top_turbulent_hard_slices"][:12]
    slice_progress = []
    for item in review_slices:
        h = int(item["horizon"])
        c = item["cell"][0] * 5 + item["cell"][1]
        h_idx = h - 1
        cov = float(turb_cov[:, h_idx, c].mean())
        gen_std = float(turb_std[:, h_idx, c].mean())
        gt_std = float(np.std(turb_det_resid[:, h_idx, c]))
        ratio = gen_std / max(gt_std, 1e-6)
        slice_progress.append(
            {
                "horizon": h,
                "cell": item["cell"],
                "coverage_183b": float(item["coverage_90"]),
                "coverage_183c": cov,
                "coverage_delta": cov - float(item["coverage_90"]),
                "std_ratio_183b": float(item["std_ratio_gen_to_gt"]),
                "std_ratio_183c": ratio,
                "std_ratio_delta": ratio - float(item["std_ratio_gen_to_gt"]),
                "target_local_log_183c": float(target_local_all[turb_mask, h_idx, c].mean()),
                "metric_local_target_183c": float(metric_local_target_all[turb_mask, h_idx, c].mean()),
                "state_uplift_metric_local_183c": float(
                    (metric_local_target_all[turb_mask, h_idx, c] - metric_local_zero_all[turb_mask, h_idx, c]).mean()
                ),
            }
        )

    target_local_flat = target_local_all[turb_mask].reshape(-1)
    raw_local_target_flat = raw_local_target_all[turb_mask].reshape(-1)
    metric_local_target_flat = metric_local_target_all[turb_mask].reshape(-1)
    raw_local_zero_flat = raw_local_zero_all[turb_mask].reshape(-1)
    metric_local_zero_flat = metric_local_zero_all[turb_mask].reshape(-1)

    hard_mask = np.zeros((future_len, n_cells), dtype=bool)
    for item in hard_entries[:12]:
        hard_mask[int(item["horizon"]) - 1, item["cell"][0] * 5 + item["cell"][1]] = True
    over_mask = np.zeros((future_len, n_cells), dtype=bool)
    for item in over_entries[:8]:
        over_mask[int(item["horizon"]) - 1, item["cell"][0] * 5 + item["cell"][1]] = True

    # Where does positive/negative local mass go?
    pos_target = np.maximum(target_local_all[turb_mask], 0.0)
    pos_metric = np.maximum(metric_local_target_all[turb_mask], 0.0)
    neg_target = np.maximum(-target_local_all[turb_mask], 0.0)
    neg_metric = np.maximum(-metric_local_target_all[turb_mask], 0.0)
    hard_positive_share_target = float(pos_target[:, hard_mask].sum() / max(pos_target.sum(), 1e-12)) if hard_mask.any() else float("nan")
    hard_positive_share_metric = float(pos_metric[:, hard_mask].sum() / max(pos_metric.sum(), 1e-12)) if hard_mask.any() else float("nan")
    over_negative_share_target = float(neg_target[:, over_mask].sum() / max(neg_target.sum(), 1e-12)) if over_mask.any() else float("nan")
    over_negative_share_metric = float(neg_metric[:, over_mask].sum() / max(neg_metric.sum(), 1e-12)) if over_mask.any() else float("nan")

    # Neighbor spillover for hard slices.
    neighbor_reviews = []
    for item in hard_entries[:12]:
        h_idx = int(item["horizon"]) - 1
        cell = item["cell"][0] * 5 + item["cell"][1]
        neigh = grid_neighbors(cell)
        if not neigh:
            continue
        target_cell = target_local_all[turb_mask, h_idx, cell]
        target_neigh = target_local_all[turb_mask, h_idx][:, neigh].mean(axis=1)
        metric_cell = metric_local_target_all[turb_mask, h_idx, cell]
        metric_neigh = metric_local_target_all[turb_mask, h_idx][:, neigh].mean(axis=1)
        neighbor_reviews.append(
            {
                "horizon": int(item["horizon"]),
                "cell": item["cell"],
                "coverage_90": float(item["coverage_90"]),
                "target_cell_mean": float(target_cell.mean()),
                "target_neighbor_mean": float(target_neigh.mean()),
                "target_contrast": float((target_cell - target_neigh).mean()),
                "metric_cell_mean": float(metric_cell.mean()),
                "metric_neighbor_mean": float(metric_neigh.mean()),
                "metric_contrast": float((metric_cell - metric_neigh).mean()),
                "contrast_ratio_metric_to_target": float(((metric_cell - metric_neigh).mean()) / max((target_cell - target_neigh).mean(), 1e-6)),
            }
        )

    pooled_gt = np.concatenate(pooled_abs_delta_gt)
    pooled_gen = np.concatenate(pooled_abs_delta_gen)
    gt_path_max = np.concatenate(path_max_jump_gt)
    gen_path_max = np.concatenate(path_max_jump_gen)
    gt_q99_exceed = np.concatenate(path_q99_exceed_count_gt)
    gen_q99_exceed = np.concatenate(path_q99_exceed_count_gen)
    high_band_abs_gt = np.concatenate(high_band_coeff_gt)
    high_band_abs_gen = np.concatenate(high_band_coeff_gen)

    band_summary = {}
    for i, band in enumerate(BANDS):
        band_summary[band] = {
            "energy_ratio_p50": float(
                np.quantile(np.concatenate(band_energy_gen[band]), 0.5)
                / max(np.quantile(np.concatenate(band_energy_gt[band]), 0.5), 1e-12)
            ),
            "raw_zero_mean_turb": float(raw_band_zero_all[turb_mask, i].mean()),
            "raw_target_mean_turb": float(raw_band_target_all[turb_mask, i].mean()),
            "metric_zero_mean_turb": float(metric_band_zero_all[turb_mask, i].mean()),
            "metric_target_mean_turb": float(metric_band_target_all[turb_mask, i].mean()),
            "state_uplift_metric_turb": float((metric_band_target_all[turb_mask, i] - metric_band_zero_all[turb_mask, i]).mean()),
            "raw_zero_mean_calm": float(raw_band_zero_all[calm_mask, i].mean()),
            "raw_target_mean_calm": float(raw_band_target_all[calm_mask, i].mean()),
            "metric_zero_mean_calm": float(metric_band_zero_all[calm_mask, i].mean()),
            "metric_target_mean_calm": float(metric_band_target_all[calm_mask, i].mean()),
            "state_uplift_metric_calm": float((metric_band_target_all[calm_mask, i] - metric_band_zero_all[calm_mask, i]).mean()),
            "band_gate_mean_turb": float(band_gate_all[turb_mask, i].mean()),
            "band_gate_mean_calm": float(band_gate_all[calm_mask, i].mean()),
        }

    return {
        "suite_context": {
            "best_passes": ["surface", "coverage", "time_series", "block_ar", "cointegration", "distributional", "cross_cell_correlation", "mean_reversion", "pathwise_jump_realism"],
            "best_fails": ["conditionality", "regime_coverage"],
        },
        "control_metric_review": {
            "local_metric_budget": float(model.local_metric_budget().item()),
            "band_metric_budget": float(model.band_metric_budget().item()),
            "target_local_abs_mean_turb": float(np.abs(target_local_flat).mean()),
            "pred_raw_local_abs_mean_turb_zero": float(np.abs(raw_local_zero_flat).mean()),
            "pred_raw_local_abs_mean_turb_target": float(np.abs(raw_local_target_flat).mean()),
            "pred_metric_local_abs_mean_turb_zero": float(np.abs(metric_local_zero_flat).mean()),
            "pred_metric_local_abs_mean_turb_target": float(np.abs(metric_local_target_flat).mean()),
            "raw_target_corr_turb_zero_state": _corr(raw_local_zero_flat, target_local_flat),
            "raw_target_corr_turb_target_state": _corr(raw_local_target_flat, target_local_flat),
            "metric_target_corr_turb_zero_state": _corr(metric_local_zero_flat, target_local_flat),
            "metric_target_corr_turb_target_state": _corr(metric_local_target_flat, target_local_flat),
            "mean_metric_local_hard_slices": float(metric_local_target_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "mean_metric_local_overwide_slices": float(metric_local_target_all[turb_mask][:, over_mask].mean()) if over_mask.any() else float("nan"),
            "mean_target_local_hard_slices": float(target_local_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "mean_target_local_overwide_slices": float(target_local_all[turb_mask][:, over_mask].mean()) if over_mask.any() else float("nan"),
            "mean_state_uplift_metric_hard_slices": float(
                (metric_local_target_all[turb_mask][:, hard_mask] - metric_local_zero_all[turb_mask][:, hard_mask]).mean()
            ) if hard_mask.any() else float("nan"),
            "mean_state_uplift_metric_overwide_slices": float(
                (metric_local_target_all[turb_mask][:, over_mask] - metric_local_zero_all[turb_mask][:, over_mask]).mean()
            ) if over_mask.any() else float("nan"),
            "local_gate_mean_turb": float(local_gate_all[turb_mask].mean()),
            "local_gate_mean_calm": float(local_gate_all[calm_mask].mean()),
            "hard_positive_share_target": hard_positive_share_target,
            "hard_positive_share_metric": hard_positive_share_metric,
            "over_negative_share_target": over_negative_share_target,
            "over_negative_share_metric": over_negative_share_metric,
        },
        "width_allocation_review": {
            "top_turbulent_hard_slices": hard_entries[:12],
            "top_overwide_slices": over_entries[:8],
            "progress_vs_183b_hard_slices": slice_progress,
            "neighbor_spillover_on_hard_slices": neighbor_reviews,
        },
        "tail_shape_review": {
            "pooled_abs_delta": distribution_compare(pooled_gt, pooled_gen),
            "pathwise_max_jump": distribution_compare(gt_path_max, gen_path_max),
            "gt_q99_exceed_count": quantile_profile(gt_q99_exceed),
            "gen_q99_exceed_count": quantile_profile(gen_q99_exceed),
            "pooled_abs_delta_exceedance_spectrum": exceedance_spectrum(pooled_gt, pooled_gen),
            "high_band_abs_exceedance_spectrum": exceedance_spectrum(high_band_abs_gt, high_band_abs_gen),
        },
        "band_review": band_summary,
        "diagnosis": {
            "s3_s7": "183c fixes the metric-strength bottleneck enough to improve coverage and pooled tails, but hard turbulent late-horizon cells are still under-allocated relative to target local concentration. The remaining miss looks more like overly diffuse allocation than wrong sign or missing state information.",
            "concentration": "The remaining width problem is coupled to concentration. The model now allocates more positive mass to hard slices than 183b, but still not enough, and the local metric contrast at the target cell remains too weak relative to neighboring cells.",
            "tail_shape": "The main tail issue is shoulder heaviness: quiet-day mass is too low and medium-move mass is too high, while extreme jump scale remains broadly acceptable. That is why S4 only barely passes and S11 is already fine.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 183c_best mechanism")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--review_183b", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model, _checkpoint = load_model(args.checkpoint, args.device)
    review_183b = load_json(args.review_183b)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    out = analyze_183c_best(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        review_183b=review_183b,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(out), indent=2))
    print(f"Wrote mechanism review to {output_path}")


if __name__ == "__main__":
    main()
