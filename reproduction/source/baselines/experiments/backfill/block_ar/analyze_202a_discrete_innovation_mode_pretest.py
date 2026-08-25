#!/usr/bin/env python
"""
Pretest whether 201b teacher-forced residuals contain a useful discrete innovation mode.

Questions:
  1. Does a small discrete mixture improve held-out residual density over a single mode?
  2. Are the inferred modes semantically different in a useful way, not just wider vs narrower?
  3. Do inferred modes have temporal persistence under teacher forcing?
  4. Are oracle mode labels predictable from context and useful for localization/template fit?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import (
    auc_roc_binary,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    reshape_history,
)
from experiments.backfill.block_ar.train_201b_underfit_aware_selffed_rollout_student_t import (
    TransformerRolloutTailStudentTARModel,
)


def build_split_indices(
    surface_len: int,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
) -> dict[str, np.ndarray]:
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    test_indices = np.arange(test_start, surface_len - history_len - future_len + 1)
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def regime_masks_from_history_np(history_01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    mean_iv = history_01.mean(axis=(2, 3))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(axis=1)
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    return vov, calm, turb, q20, q80


def build_window_metadata(
    history_01: np.ndarray,
    future_flat: np.ndarray,
    q80_vov_train: float | None = None,
    q80_h30_turb_train: float | None = None,
) -> dict[str, Any]:
    vov, calm, turb, q20_vov, q80_vov = regime_masks_from_history_np(history_01)
    prev = history_01[:, -1].reshape(history_01.shape[0], -1)
    h30_energy = np.abs(future_flat[:, -1, :] - prev).sum(axis=1)
    if q80_vov_train is None:
        q80_vov_use = q80_vov
    else:
        q80_vov_use = q80_vov_train
        calm = vov <= q20_vov
        turb = vov >= q80_vov_use
    if np.any(turb):
        q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8)) if q80_h30_turb_train is None else q80_h30_turb_train
        hard_late = turb & (h30_energy >= q80_h30_turb)
    else:
        q80_h30_turb = float("nan") if q80_h30_turb_train is None else q80_h30_turb_train
        hard_late = np.zeros_like(turb, dtype=bool)
    return {
        "vov": vov,
        "calm": calm,
        "turb": turb,
        "hard_late": hard_late,
        "q20_vov": float(q20_vov),
        "q80_vov": float(q80_vov_use),
        "q80_h30_turb": float(q80_h30_turb),
    }


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TransformerRolloutTailStudentTARModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    model_type = raw_config["type"]
    if model_type != "transformer_underfit_aware_selffed_rollout_student_t_201b":
        raise ValueError(f"Expected 201b checkpoint, got {model_type}")
    model = TransformerRolloutTailStudentTARModel(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def fit_multiclass_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_classes: int,
    seed: int = 42,
) -> dict[str, float]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(32, x_train.shape[1]))),
            (
                "logit",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    test_proba = pipe.predict_proba(x_test)
    test_pred = test_proba.argmax(axis=1)
    top3 = np.argsort(test_proba, axis=1)[:, -min(3, max_classes) :]
    majority = np.bincount(y_train, minlength=max_classes).argmax()
    return {
        "test_top1_acc": float((test_pred == y_test).mean()),
        "test_top3_acc": float(np.mean([y in row for y, row in zip(y_test, top3)])),
        "chance_top1": 1.0 / float(max_classes),
        "majority_top1": float((y_test == majority).mean()),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def collect_teacher_forced_step_records(
    model: TransformerRolloutTailStudentTARModel,
    history_01: torch.Tensor,
    future_flat: torch.Tensor,
    window_meta: dict[str, np.ndarray],
    split_name: str,
    q95: float,
    q99: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {
        "split": [],
        "window_idx": [],
        "step": [],
        "cond_feat": [],
        "whitened_resid": [],
        "abs_delta_pattern": [],
        "abs_delta_norm": [],
        "q99_any": [],
        "q95_frac": [],
        "top_cell": [],
        "effective_support": [],
        "resid_norm": [],
        "calm": [],
        "turb": [],
        "hard_late_window": [],
        "hard_late_h30": [],
    }

    n_windows = history_01.shape[0]
    future_len = future_flat.shape[1]
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist = history_01[start:end].to(device)
        fut = future_flat[start:end].to(device)
        context = hist.clone()
        batch_window_idx = np.arange(start, end)

        for step in range(future_len):
            with torch.no_grad():
                cond = model.encode(context)
                if isinstance(cond, tuple):
                    cond = cond[0]
                mu, factor, diag, scale, nu = model.forward_from_history(context)
                cov = model.covariance(factor, diag, scale)

            prev_01 = reshape_history(context)[:, -1]
            target_t = fut[:, step]
            target_u = iv_to_unconstrained(
                target_t,
                lo=model.support_lo,
                hi=model.support_hi,
                eps=model.support_eps,
            )
            resid = target_u - mu
            chol = torch.linalg.cholesky(cov)
            whiten = torch.linalg.solve_triangular(chol, resid.unsqueeze(-1), upper=False).squeeze(-1)

            delta_abs = (target_t - prev_01).abs()
            delta_sum = delta_abs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            delta_norm = delta_abs / delta_sum
            entropy = -(delta_norm * torch.log(delta_norm.clamp_min(1e-8))).sum(dim=-1)
            eff_support = torch.exp(entropy)

            out["split"].append(np.full(end - start, split_name))
            out["window_idx"].append(batch_window_idx.copy())
            out["step"].append(np.full(end - start, step, dtype=np.int64))
            out["cond_feat"].append(cond.detach().cpu().numpy())
            out["whitened_resid"].append(whiten.detach().cpu().numpy())
            out["abs_delta_pattern"].append(delta_abs.detach().cpu().numpy())
            out["abs_delta_norm"].append(delta_norm.detach().cpu().numpy())
            out["q99_any"].append((delta_abs >= q99).any(dim=-1).detach().cpu().numpy().astype(np.int64))
            out["q95_frac"].append((delta_abs >= q95).float().mean(dim=-1).detach().cpu().numpy())
            out["top_cell"].append(delta_abs.argmax(dim=-1).detach().cpu().numpy())
            out["effective_support"].append(eff_support.detach().cpu().numpy())
            out["resid_norm"].append(whiten.norm(dim=-1).detach().cpu().numpy())
            out["calm"].append(window_meta["calm"][batch_window_idx].astype(np.int64))
            out["turb"].append(window_meta["turb"][batch_window_idx].astype(np.int64))
            out["hard_late_window"].append(window_meta["hard_late"][batch_window_idx].astype(np.int64))
            out["hard_late_h30"].append((window_meta["hard_late"][batch_window_idx] & (step == future_len - 1)).astype(np.int64))

            next_frame = target_t.view(end - start, 1, 5, 5)
            context = torch.cat([context[:, 1:], next_frame], dim=1)

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def sample_fit_indices(n: int, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def fit_gmm_family(
    z_train: np.ndarray,
    z_val: np.ndarray,
    z_test: np.ndarray,
    max_modes: int,
    pca_dim: int,
    max_fit_samples: int | None,
    seed: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    fit_idx = sample_fit_indices(z_train.shape[0], max_fit_samples, seed)
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train[fit_idx])
    pca = PCA(n_components=min(pca_dim, z_train.shape[1]), random_state=seed)
    z_train_pca = pca.fit_transform(z_train_scaled)
    z_val_pca = pca.transform(scaler.transform(z_val))
    z_test_pca = pca.transform(scaler.transform(z_test))
    z_train_all_pca = pca.transform(scaler.transform(z_train))

    results: dict[int, Any] = {}
    best_k = 1
    best_val = -np.inf
    best_gmm: GaussianMixture | None = None
    for k in range(1, max_modes + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-4,
            n_init=2,
            max_iter=300,
            random_state=seed,
        )
        gmm.fit(z_train_pca)
        val_score = float(gmm.score(z_val_pca))
        results[k] = {
            "train_avg_loglik_fit_subset": float(gmm.score(z_train_pca)),
            "train_avg_loglik_all": float(gmm.score(z_train_all_pca)),
            "val_avg_loglik": val_score,
            "test_avg_loglik": float(gmm.score(z_test_pca)),
            "fit_subset_bic": float(gmm.bic(z_train_pca)),
            "fit_subset_aic": float(gmm.aic(z_train_pca)),
            "weights": gmm.weights_.tolist(),
            "min_weight": float(gmm.weights_.min()),
            "max_weight": float(gmm.weights_.max()),
        }
        if val_score > best_val:
            best_val = val_score
            best_k = k
            best_gmm = gmm

    assert best_gmm is not None
    payload = {
        "best_k": int(best_k),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "gmm_weights": best_gmm.weights_,
        "gmm_means": best_gmm.means_,
        "gmm_covariances": best_gmm.covariances_,
        "gmm_precisions_cholesky": best_gmm.precisions_cholesky_,
    }
    return results, payload


def transform_with_payload(z: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    scaled = (z - payload["scaler_mean"]) / payload["scaler_scale"]
    centered = scaled - payload["pca_mean"]
    return centered @ payload["pca_components"].T


def gmm_predict_proba(z_pca: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    weights = payload["gmm_weights"]
    means = payload["gmm_means"]
    covs = payload["gmm_covariances"]
    n, d = z_pca.shape
    k = means.shape[0]
    logp = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        diff = z_pca - means[j]
        cov = covs[j]
        chol = np.linalg.cholesky(cov)
        solve = np.linalg.solve(chol, diff.T).T
        mahal = np.sum(solve * solve, axis=1)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        logp[:, j] = np.log(weights[j] + 1e-12) - 0.5 * (d * np.log(2.0 * np.pi) + logdet + mahal)
    logp = logp - logp.max(axis=1, keepdims=True)
    probs = np.exp(logp)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def summarize_modes(
    records: dict[str, np.ndarray],
    mode_assign: np.ndarray,
    n_modes: int,
) -> dict[str, Any]:
    top_cell = records["top_cell"]
    abs_norm = records["abs_delta_norm"]
    global_top_counts = np.bincount(top_cell, minlength=25)
    global_top1 = float(global_top_counts.max() / max(global_top_counts.sum(), 1))
    mode_summaries: dict[str, Any] = {}
    for mode in range(n_modes):
        mask = mode_assign == mode
        if not np.any(mask):
            mode_summaries[str(mode)] = {"count": 0}
            continue
        top_counts = np.bincount(top_cell[mask], minlength=25)
        mode_summaries[str(mode)] = {
            "count": int(mask.sum()),
            "weight": float(mask.mean()),
            "resid_norm_mean": float(records["resid_norm"][mask].mean()),
            "q99_any_rate": float(records["q99_any"][mask].mean()),
            "q95_frac_mean": float(records["q95_frac"][mask].mean()),
            "effective_support_mean": float(records["effective_support"][mask].mean()),
            "hard_late_h30_rate": float(records["hard_late_h30"][mask].mean()),
            "calm_rate": float(records["calm"][mask].mean()),
            "turb_rate": float(records["turb"][mask].mean()),
            "step29_rate": float((records["step"][mask] == 29).mean()),
            "top_cell_majority_share": float(top_counts.max() / max(top_counts.sum(), 1)),
            "top_cell_majority": int(top_counts.argmax()),
            "mean_abs_pattern_top3": np.argsort(abs_norm[mask].mean(axis=0))[-3:][::-1].tolist(),
        }
    return {
        "global_top_cell_majority_share": global_top1,
        "modes": mode_summaries,
    }


def summarize_persistence(
    records: dict[str, np.ndarray],
    mode_assign: np.ndarray,
    n_windows: int,
    future_len: int,
    subset_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mode_mat = np.full((n_windows, future_len), -1, dtype=np.int64)
    mode_mat[records["window_idx"], records["step"]] = mode_assign
    valid = mode_mat[:, :-1] >= 0
    if subset_mask is not None:
        valid = valid & subset_mask[:, None]
    lhs = mode_mat[:, :-1][valid]
    rhs = mode_mat[:, 1:][valid]
    if lhs.size == 0:
        return {"stay_prob": float("nan"), "iid_baseline": float("nan"), "lift": float("nan")}
    weights = np.bincount(mode_assign, minlength=int(mode_assign.max()) + 1).astype(np.float64)
    weights = weights / max(weights.sum(), 1.0)
    iid_baseline = float((weights**2).sum())
    stay = float((lhs == rhs).mean())
    return {
        "stay_prob": stay,
        "iid_baseline": iid_baseline,
        "lift": float(stay - iid_baseline),
    }


def template_metrics(
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    train_modes: np.ndarray,
    test_modes: np.ndarray,
    n_modes: int,
) -> dict[str, Any]:
    global_template = train_records["abs_delta_norm"].mean(axis=0)
    mode_templates = np.stack(
        [
            train_records["abs_delta_norm"][train_modes == k].mean(axis=0)
            if np.any(train_modes == k)
            else global_template
            for k in range(n_modes)
        ],
        axis=0,
    )

    global_top_probs = np.bincount(train_records["top_cell"], minlength=25).astype(np.float64)
    global_top_probs /= max(global_top_probs.sum(), 1.0)
    mode_top_probs = []
    for k in range(n_modes):
        mask = train_modes == k
        counts = np.bincount(train_records["top_cell"][mask], minlength=25).astype(np.float64) if np.any(mask) else global_top_probs.copy()
        counts /= max(counts.sum(), 1.0)
        mode_top_probs.append(counts)
    mode_top_probs = np.stack(mode_top_probs, axis=0)

    pred_global = np.repeat(global_template[None, :], test_records["abs_delta_norm"].shape[0], axis=0)
    pred_mode = mode_templates[test_modes]
    global_mae = np.abs(pred_global - test_records["abs_delta_norm"]).mean(axis=1)
    mode_mae = np.abs(pred_mode - test_records["abs_delta_norm"]).mean(axis=1)

    global_top_rank = np.argsort(global_top_probs)[::-1]
    mode_top_rank = np.argsort(mode_top_probs, axis=1)[:, ::-1]
    top_cell = test_records["top_cell"]

    def subset_summary(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {}
        g1 = float((top_cell[mask] == global_top_rank[0]).mean())
        m1 = float((top_cell[mask] == mode_top_rank[test_modes[mask], 0]).mean())
        g3 = float(np.mean([y in global_top_rank[:3] for y in top_cell[mask]]))
        m3 = float(np.mean([y in row[:3] for y, row in zip(top_cell[mask], mode_top_rank[test_modes[mask]])]))
        return {
            "global_template_mae": float(global_mae[mask].mean()),
            "oracle_mode_template_mae": float(mode_mae[mask].mean()),
            "template_mae_improvement_pct": float((global_mae[mask].mean() - mode_mae[mask].mean()) / max(global_mae[mask].mean(), 1e-8)),
            "global_top1_acc": g1,
            "oracle_mode_top1_acc": m1,
            "global_top3_acc": g3,
            "oracle_mode_top3_acc": m3,
        }

    return {
        "all_steps": subset_summary(np.ones(test_records["top_cell"].shape[0], dtype=bool)),
        "q99_any_steps": subset_summary(test_records["q99_any"] == 1),
        "hard_late_h30_steps": subset_summary(test_records["hard_late_h30"] == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="202a discrete innovation mode pretest on 201b residuals")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_modes", type=int, default=6)
    parser.add_argument("--pca_dim", type=int, default=8)
    parser.add_argument("--max_fit_samples", type=int, default=60000)
    parser.add_argument("--max_probe_train_steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/202_design/202a_discrete_innovation_mode_pretest",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    surfaces = np.load(args.data_path)["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    model, payload = load_model(args.checkpoint, device)

    split_indices = build_split_indices(
        surface_len=surfaces.shape[0],
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
    )
    surf_tensor = torch.from_numpy(surfaces).to(device)

    split_records: dict[str, dict[str, np.ndarray]] = {}
    split_meta: dict[str, dict[str, Any]] = {}
    train_thresholds: dict[str, float] | None = None
    split_window_counts: dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        idx = split_indices[split_name]
        hist_01, future_flat = build_multistep_windows(idx, surf_tensor, args.history_len, args.future_len)
        hist_np = hist_01.detach().cpu().numpy()
        future_np = future_flat.detach().cpu().numpy()
        if split_name == "train":
            meta = build_window_metadata(hist_np, future_np)
            train_thresholds = {
                "q80_vov": meta["q80_vov"],
                "q80_h30_turb": meta["q80_h30_turb"],
            }
        else:
            assert train_thresholds is not None
            meta = build_window_metadata(
                hist_np,
                future_np,
                q80_vov_train=train_thresholds["q80_vov"],
                q80_h30_turb_train=train_thresholds["q80_h30_turb"],
            )
        split_meta[split_name] = meta
        split_window_counts[split_name] = int(hist_np.shape[0])
        split_records[split_name] = collect_teacher_forced_step_records(
            model=model,
            history_01=hist_01,
            future_flat=future_flat,
            window_meta=meta,
            split_name=split_name,
            q95=q95,
            q99=q99,
            batch_size=args.batch_size,
            device=device,
        )

    gmm_results, gmm_payload = fit_gmm_family(
        z_train=split_records["train"]["whitened_resid"],
        z_val=split_records["val"]["whitened_resid"],
        z_test=split_records["test"]["whitened_resid"],
        max_modes=args.max_modes,
        pca_dim=args.pca_dim,
        max_fit_samples=args.max_fit_samples,
        seed=args.seed,
    )
    best_k = int(gmm_payload["best_k"])

    assignments: dict[str, np.ndarray] = {}
    responsibilities: dict[str, np.ndarray] = {}
    for split_name, rec in split_records.items():
        z_pca = transform_with_payload(rec["whitened_resid"], gmm_payload)
        proba = gmm_predict_proba(z_pca, gmm_payload)
        responsibilities[split_name] = proba
        assignments[split_name] = proba.argmax(axis=1)

    if split_records["train"]["cond_feat"].shape[0] > args.max_probe_train_steps:
        probe_idx = np.sort(rng.choice(split_records["train"]["cond_feat"].shape[0], size=args.max_probe_train_steps, replace=False))
    else:
        probe_idx = np.arange(split_records["train"]["cond_feat"].shape[0])
    mode_probe = fit_multiclass_probe(
        x_train=split_records["train"]["cond_feat"][probe_idx],
        y_train=assignments["train"][probe_idx],
        x_test=split_records["test"]["cond_feat"],
        y_test=assignments["test"],
        max_classes=best_k,
        seed=args.seed,
    )

    train_mode_probs = responsibilities["train"]
    hard_late_binary_probe = {
        "mode0_prob_auc": auc_roc_binary(split_records["test"]["hard_late_h30"], train_mode_probs.mean(axis=0)[0] * np.ones_like(split_records["test"]["hard_late_h30"], dtype=np.float64))
    }
    # More useful held-out oracle read: max mode confidence on hard-late steps vs others.
    hard_late_binary_probe = {
        "hard_late_positive_rate_test": float(split_records["test"]["hard_late_h30"].mean()),
        "mode_confidence_auc": auc_roc_binary(
            split_records["test"]["hard_late_h30"],
            responsibilities["test"].max(axis=1),
        ),
    }

    template_summary = template_metrics(
        train_records=split_records["train"],
        test_records=split_records["test"],
        train_modes=assignments["train"],
        test_modes=assignments["test"],
        n_modes=best_k,
    )

    persistence_summary = {
        split_name: {
            "all_steps": summarize_persistence(rec, assignments[split_name], split_window_counts[split_name], args.future_len, subset_mask=None),
            "turb_windows": summarize_persistence(
                rec,
                assignments[split_name],
                split_window_counts[split_name],
                args.future_len,
                subset_mask=split_meta[split_name]["turb"],
            ),
            "hard_late_windows": summarize_persistence(
                rec,
                assignments[split_name],
                split_window_counts[split_name],
                args.future_len,
                subset_mask=split_meta[split_name]["hard_late"],
            ),
        }
        for split_name, rec in split_records.items()
    }

    mode_semantics = {
        split_name: summarize_modes(split_records[split_name], assignments[split_name], best_k)
        for split_name in ("train", "val", "test")
    }

    global_train_ll = gmm_results[1]["train_avg_loglik_all"]
    best_val_ll = gmm_results[best_k]["val_avg_loglik"]
    summary: dict[str, Any] = {
        "config": {
            "checkpoint": args.checkpoint,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "q95": q95,
            "q99": q99,
            "train_windows": split_window_counts["train"],
            "val_windows": split_window_counts["val"],
            "test_windows": split_window_counts["test"],
            "train_steps": int(split_records["train"]["step"].shape[0]),
            "val_steps": int(split_records["val"]["step"].shape[0]),
            "test_steps": int(split_records["test"]["step"].shape[0]),
            "max_fit_samples": args.max_fit_samples,
            "pca_dim": args.pca_dim,
            "best_checkpoint_epoch": int(payload.get("epoch", -1)),
        },
        "gmm_model_selection": {
            "results": gmm_results,
            "best_k_by_val_loglik": best_k,
            "val_loglik_gain_vs_k1": float(gmm_results[best_k]["val_avg_loglik"] - gmm_results[1]["val_avg_loglik"]),
            "test_loglik_gain_vs_k1": float(gmm_results[best_k]["test_avg_loglik"] - gmm_results[1]["test_avg_loglik"]),
            "train_loglik_gain_vs_k1": float(gmm_results[best_k]["train_avg_loglik_all"] - gmm_results[1]["train_avg_loglik_all"]),
        },
        "mode_semantics": mode_semantics,
        "mode_persistence": persistence_summary,
        "history_to_mode_predictability": mode_probe,
        "hard_late_mode_confidence_probe": hard_late_binary_probe,
        "oracle_mode_usefulness": template_summary,
        "split_meta": {
            split_name: {
                "vov_q80": split_meta[split_name]["q80_vov"],
                "h30_turb_q80": split_meta[split_name]["q80_h30_turb"],
                "calm_windows": int(split_meta[split_name]["calm"].sum()),
                "turb_windows": int(split_meta[split_name]["turb"].sum()),
                "hard_late_windows": int(split_meta[split_name]["hard_late"].sum()),
            }
            for split_name in split_meta
        },
        "interpretation": {
            "read": (
                "A useful discrete innovation mode is supported if a small residual mixture beats k=1 on held-out "
                "log likelihood, mode-conditioned templates beat the global template on held-out hard/q99 steps, "
                "mode labels are history-predictable above chance, and inferred modes show persistence beyond iid."
            ),
            "headline": (
                "best_k="
                f"{best_k}, val_ll_gain={gmm_results[best_k]['val_avg_loglik'] - gmm_results[1]['val_avg_loglik']:.4f}, "
                f"test_template_gain_hard={template_summary['hard_late_h30_steps'].get('template_mae_improvement_pct', float('nan')):.4f}"
            ),
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
