#!/usr/bin/env python
"""
Focused mechanistic analysis for 170d.

Questions:
  1. Are remaining S2/S7 failures driven more by mean bias or variance misallocation?
  2. Do failures concentrate in specific regime x horizon x cell slices?
  3. Does 170d route regime information mainly through global scale, or also through covariance shape?
  4. What does the evidence imply for the next model after 170d?

This script is diagnostic-only. It does not change model weights.
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    normalize_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    StructuredJointStudentTModel,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


SELECT_HORIZONS = [1, 7, 14, 30]
REGIME_NAMES = ["all", "calm", "turb"]


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[mask].mean())


def frob_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.square(a - b).sum()))


def sample_cov(samples: torch.Tensor) -> torch.Tensor:
    """
    Args:
        samples: (B, S, C)
    Returns:
        cov: (B, C, C)
    """
    centered = samples - samples.mean(dim=1, keepdim=True)
    denom = max(samples.shape[1] - 1, 1)
    return torch.matmul(centered.transpose(1, 2), centered) / denom


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    if raw_config["type"] != "structured_joint_student_t_170d":
        raise ValueError(f"Expected structured_joint_student_t_170d, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = StructuredJointStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


def build_test_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(data_path)
    dataset = VolSurfaceDataset(
        data["surface"],
        history_len,
        future_len,
        start_idx=test_start,
    )
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    histories = []
    futures = []
    for i in range(n):
        item = dataset[i]
        histories.append(item["history"])
        futures.append(item["future"])
    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    return vov, q20, q80


def classify_cell_failure(coverage: float, z_mean: float, z_std: float) -> str:
    """
    Heuristic classification, used only for diagnosis.
    z_mean/z_std are computed on IV-scale standardized residuals.
    """
    if coverage > 0.95:
        if z_std < 0.90 and abs(z_mean) < 0.25:
            return "overwide"
        if z_std < 0.90:
            return "overwide_plus_bias"
        return "high_coverage_mixed"
    if coverage < 0.70:
        if abs(z_mean) > 0.50 and z_std <= 1.15:
            return "bias_dominant"
        if z_std > 1.15 and abs(z_mean) <= 0.50:
            return "underwide_dominant"
        return "bias_plus_underwide"
    return "pass"


@torch.no_grad()
def analyze_170d(
    model: StructuredJointStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01_all = denormalize_iv(history_norm).reshape(history_norm.shape[0], history_norm.shape[1], -1)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n_windows, future_len, n_cells = future_01_all.shape
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    pred_mean_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_std_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_median_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    pred_mean_u = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_var_u = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    residual_u = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    white_resid = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    global_scale = np.zeros(n_windows, dtype=np.float32)
    mean_time_var = np.zeros((n_windows, future_len), dtype=np.float32)
    mean_cell_var = np.zeros((n_windows, n_cells), dtype=np.float32)
    time_offdiag_corr = np.zeros(n_windows, dtype=np.float32)
    cell_offdiag_corr = np.zeros(n_windows, dtype=np.float32)
    time_eff_rank = np.zeros(n_windows, dtype=np.float32)
    cell_eff_rank = np.zeros(n_windows, dtype=np.float32)

    pred_cov_sum = {name: {str(h): np.zeros((n_cells, n_cells), dtype=np.float64) for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    pred_cov_count = {name: {str(h): 0 for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    emp_cov_sum = {name: {str(h): np.zeros((n_cells, n_cells), dtype=np.float64) for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    emp_cov_count = {name: {str(h): 0 for h in SELECT_HORIZONS} for name in REGIME_NAMES}

    row0 = 0
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        fut_norm_b = future_norm[start:end]
        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(fut_norm_b).reshape(end - start, future_len, n_cells)

        mu_u, time_factor, time_diag, cell_factor, cell_diag, scale, nu = model.forward_from_history(hist_01_b)
        cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)

        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )

        diag_t = torch.diagonal(cov_t, dim1=-2, dim2=-1)
        diag_c = torch.diagonal(cov_c, dim1=-2, dim2=-1)
        marginal_var_u = diag_t.unsqueeze(-1) * diag_c.unsqueeze(1)

        diff_u = target_u - mu_u
        white_t = torch.linalg.solve_triangular(chol_t, diff_u, upper=False)
        white = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)

        samples_01 = model.sample_batched(hist_norm_b, n_samples=n_samples)
        samples_iv = samples_01.reshape(end - start, n_samples, future_len, n_cells)
        mean_iv_b = samples_iv.mean(dim=1)
        std_iv_b = samples_iv.std(dim=1, unbiased=False)
        median_iv_b = samples_iv.median(dim=1).values
        lo90_b = torch.quantile(samples_iv, 0.05, dim=1)
        hi90_b = torch.quantile(samples_iv, 0.95, dim=1)

        pred_mean_iv[row0:end] = mean_iv_b.detach().cpu().numpy()
        pred_std_iv[row0:end] = std_iv_b.detach().cpu().numpy()
        pred_median_iv[row0:end] = median_iv_b.detach().cpu().numpy()
        lo90_iv[row0:end] = lo90_b.detach().cpu().numpy()
        hi90_iv[row0:end] = hi90_b.detach().cpu().numpy()

        pred_mean_u[row0:end] = mu_u.detach().cpu().numpy()
        pred_var_u[row0:end] = marginal_var_u.detach().cpu().numpy()
        residual_u[row0:end] = diff_u.detach().cpu().numpy()
        white_resid[row0:end] = white.detach().cpu().numpy()

        scale_np = scale.detach().cpu().numpy()
        cov_t_np = cov_t.detach().cpu().numpy()
        cov_c_np = cov_c.detach().cpu().numpy()
        global_scale[row0:end] = scale_np
        mean_time_var[row0:end] = np.diagonal(cov_t_np, axis1=1, axis2=2)
        mean_cell_var[row0:end] = np.diagonal(cov_c_np, axis1=1, axis2=2)

        for b in range(end - start):
            t_corr = cov_t_np[b] / np.outer(np.sqrt(np.clip(np.diag(cov_t_np[b]), 1e-10, None)), np.sqrt(np.clip(np.diag(cov_t_np[b]), 1e-10, None)))
            c_corr = cov_c_np[b] / np.outer(np.sqrt(np.clip(np.diag(cov_c_np[b]), 1e-10, None)), np.sqrt(np.clip(np.diag(cov_c_np[b]), 1e-10, None)))
            np.fill_diagonal(t_corr, 1.0)
            np.fill_diagonal(c_corr, 1.0)
            time_offdiag_corr[row0 + b] = mean_offdiag_corr(cov_t_np[b])
            cell_offdiag_corr[row0 + b] = mean_offdiag_corr(cov_c_np[b])
            time_eff_rank[row0 + b] = eff_rank_from_matrix(cov_t_np[b])
            cell_eff_rank[row0 + b] = eff_rank_from_matrix(cov_c_np[b])

        for regime_name, mask in masks.items():
            local_mask = mask[start:end]
            if not np.any(local_mask):
                continue
            for h in SELECT_HORIZONS:
                hidx = h - 1
                samples_h = samples_iv[local_mask, :, hidx, :]
                pred_cov = sample_cov(samples_h).mean(dim=0).detach().cpu().numpy()
                resid_h = (fut_01_b[local_mask, hidx, :] - mean_iv_b[local_mask, hidx, :]).detach().cpu().numpy()
                if resid_h.shape[0] >= 2:
                    emp_cov = np.cov(resid_h, rowvar=False, bias=True)
                else:
                    emp_cov = np.zeros((n_cells, n_cells), dtype=np.float64)
                pred_cov_sum[regime_name][str(h)] += pred_cov
                pred_cov_count[regime_name][str(h)] += 1
                emp_cov_sum[regime_name][str(h)] += emp_cov
                emp_cov_count[regime_name][str(h)] += 1

        row0 = end

    target_iv = future_01_all.detach().cpu().numpy()
    resid_iv = target_iv - pred_mean_iv
    pred_var_iv = np.square(pred_std_iv)
    z_iv = resid_iv / np.clip(pred_std_iv, 1e-6, None)
    covered90 = (target_iv >= lo90_iv) & (target_iv <= hi90_iv)

    regime_horizon_summary: dict[str, dict[str, Any]] = {}
    for regime_name, mask in masks.items():
        regime_horizon_summary[regime_name] = {}
        for h in SELECT_HORIZONS:
            hidx = h - 1
            m = mask
            cov = covered90[m, hidx, :]
            res = resid_iv[m, hidx, :]
            z = z_iv[m, hidx, :]
            std = pred_std_iv[m, hidx, :]
            width = hi90_iv[m, hidx, :] - lo90_iv[m, hidx, :]
            regime_horizon_summary[regime_name][str(h)] = {
                "coverage_90": float(cov.mean()),
                "signed_bias_mean": float(res.mean()),
                "abs_bias_mean": float(np.abs(res).mean()),
                "pred_std_mean": float(std.mean()),
                "pred_width90_mean": float(width.mean()),
                "rmse": float(np.sqrt(np.square(res).mean())),
                "z_mean": float(z.mean()),
                "z_std": float(z.std()),
                "variance_ratio_pred_over_realized": float(
                    np.mean(pred_var_iv[m, hidx, :]) / max(np.mean(np.square(res)), 1e-8)
                ),
            }

    overcovered = []
    undercovered = []
    class_counts = {}
    for regime_name, mask in masks.items():
        for h in SELECT_HORIZONS:
            hidx = h - 1
            cov_vec = covered90[mask, hidx, :]
            z_vec = z_iv[mask, hidx, :]
            res_vec = resid_iv[mask, hidx, :]
            std_vec = pred_std_iv[mask, hidx, :]
            for cell in range(n_cells):
                coverage = float(cov_vec[:, cell].mean())
                z_mean = float(z_vec[:, cell].mean())
                z_std = float(z_vec[:, cell].std())
                label = classify_cell_failure(coverage, z_mean, z_std)
                class_counts[label] = class_counts.get(label, 0) + 1
                record = {
                    "regime": regime_name,
                    "horizon": h,
                    "cell": cell,
                    "coverage_90": coverage,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "abs_bias_mean": float(np.abs(res_vec[:, cell]).mean()),
                    "pred_std_mean": float(std_vec[:, cell].mean()),
                    "classification": label,
                }
                if coverage > 0.95:
                    overcovered.append(record)
                elif coverage < 0.70:
                    undercovered.append(record)

    overcovered.sort(key=lambda x: x["coverage_90"], reverse=True)
    undercovered.sort(key=lambda x: x["coverage_90"])

    regime_routing = {}
    for regime_name, mask in masks.items():
        regime_routing[regime_name] = {
            "n_windows": int(mask.sum()),
            "global_scale_mean": float(global_scale[mask].mean()),
            "time_var_mean": float(mean_time_var[mask].mean()),
            "cell_var_mean": float(mean_cell_var[mask].mean()),
            "time_offdiag_corr_mean": float(time_offdiag_corr[mask].mean()),
            "cell_offdiag_corr_mean": float(cell_offdiag_corr[mask].mean()),
            "time_eff_rank_mean": float(time_eff_rank[mask].mean()),
            "cell_eff_rank_mean": float(cell_eff_rank[mask].mean()),
        }

    regime_shape_check = {}
    for regime_name in REGIME_NAMES:
        regime_shape_check[regime_name] = {}
        for h in SELECT_HORIZONS:
            if pred_cov_count[regime_name][str(h)] == 0 or emp_cov_count[regime_name][str(h)] == 0:
                continue
            pred_cov = pred_cov_sum[regime_name][str(h)] / pred_cov_count[regime_name][str(h)]
            emp_cov = emp_cov_sum[regime_name][str(h)] / emp_cov_count[regime_name][str(h)]
            regime_shape_check[regime_name][str(h)] = {
                "pred_mean_offdiag_corr": mean_offdiag_corr(pred_cov),
                "emp_mean_offdiag_corr": mean_offdiag_corr(emp_cov),
                "pred_eff_rank": eff_rank_from_matrix(pred_cov),
                "emp_eff_rank": eff_rank_from_matrix(emp_cov),
                "frob_distance": frob_norm(pred_cov, emp_cov),
            }

    white_summary = {}
    for regime_name, mask in masks.items():
        white_flat = white_resid[mask].reshape(-1)
        white_summary[regime_name] = {
            "mean": float(white_flat.mean()),
            "std": float(white_flat.std()),
            "mean_abs": float(np.abs(white_flat).mean()),
            "p95_abs": float(np.quantile(np.abs(white_flat), 0.95)),
        }

    next_step_inference = {
        "mean_bias_is_primary": bool(
            abs(regime_horizon_summary["all"]["30"]["z_mean"]) > 0.25
            or abs(regime_horizon_summary["turb"]["30"]["z_mean"]) > 0.25
        ),
        "variance_misallocation_is_primary": bool(
            regime_horizon_summary["all"]["30"]["variance_ratio_pred_over_realized"] > 1.15
            or regime_horizon_summary["all"]["30"]["variance_ratio_pred_over_realized"] < 0.85
            or regime_horizon_summary["turb"]["30"]["variance_ratio_pred_over_realized"] > 1.15
            or regime_horizon_summary["turb"]["30"]["variance_ratio_pred_over_realized"] < 0.85
        ),
        "regime_scale_routing_is_weak": bool(
            regime_horizon_summary["turb"]["30"]["pred_std_mean"]
            / max(regime_horizon_summary["calm"]["30"]["pred_std_mean"], 1e-8)
            < 1.20
        ),
        "regime_shape_routing_is_weak": bool(
            abs(
                regime_routing["turb"]["cell_offdiag_corr_mean"]
                - regime_routing["calm"]["cell_offdiag_corr_mean"]
            )
            < 0.03
        ),
    }

    return {
        "metadata": {
            "n_windows": n_windows,
            "future_len": future_len,
            "n_cells": n_cells,
            "n_samples": n_samples,
            "q20_vov": q20,
            "q80_vov": q80,
        },
        "regime_horizon_summary": regime_horizon_summary,
        "failure_class_counts": class_counts,
        "top_overcovered_slices": overcovered[:20],
        "top_undercovered_slices": undercovered[:20],
        "regime_routing": regime_routing,
        "regime_shape_check": regime_shape_check,
        "white_residual_summary": white_summary,
        "next_step_inference": next_step_inference,
    }


def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis for 170d")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)
    analysis = analyze_170d(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    analysis["checkpoint_epoch"] = checkpoint.get("epoch")
    analysis["model_path"] = args.model_path

    out_path = Path(args.output_dir) / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved analysis to {out_path}")
    print(json.dumps(make_serializable(analysis["next_step_inference"]), indent=2))


if __name__ == "__main__":
    main()
