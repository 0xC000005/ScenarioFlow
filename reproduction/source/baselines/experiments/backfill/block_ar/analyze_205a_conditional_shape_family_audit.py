#!/usr/bin/env python
"""
Audit whether dangerous-window conditional innovation shapes form a small coherent family
after factoring out mean / covariance / severity effects.

This is the direct follow-up to 202a/202b:
  - use 201b teacher-forced conditional state
  - remove broad mean/cov effects via whitened residual direction
  - remove severity via norm normalization
  - test local conditional families around dangerous histories

Question:
  Is dangerous-window GT innovation uncertainty:
    1. a small coherent family of shapes conditional on history?
    2. or still genuinely diffuse / weakly identifiable from current inputs?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import (
    make_serializable,
)
from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_split_indices,
    build_window_metadata,
    load_model,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import reshape_history


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, eps, None)


def safe_entropy_from_probs(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return float("nan")
    return float(-(probs * np.log(probs)).sum())


def effective_rank(x: np.ndarray) -> float:
    if x.shape[0] <= 1:
        return 1.0
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    probs = eigvals / eigvals.sum()
    return float(np.exp(-(probs * np.log(probs)).sum()))


def cosine_max(query: np.ndarray, family: np.ndarray) -> float:
    sims = family @ query
    return float(sims.max()) if sims.size else float("nan")


def greedy_prototypes(x: np.ndarray, n_proto: int) -> np.ndarray:
    if x.shape[0] <= n_proto:
        return x
    dists_to_mean = ((x - x.mean(axis=0, keepdims=True)) ** 2).sum(axis=1)
    selected = [int(dists_to_mean.argmin())]
    min_dist = ((x - x[selected[0] : selected[0] + 1]) ** 2).sum(axis=1)
    while len(selected) < n_proto:
        next_idx = int(min_dist.argmax())
        selected.append(next_idx)
        cur = ((x - x[next_idx : next_idx + 1]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, cur)
    return x[selected]


def prototype_min_mae(query: np.ndarray, protos: np.ndarray) -> float:
    mae = np.abs(protos - query[None, :]).mean(axis=1)
    return float(mae.min())


def collect_teacher_forced_records(
    model,
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
        "resid_norm": [],
        "whiten_dir": [],
        "signed_delta_norm": [],
        "top_cell": [],
        "q95_any": [],
        "q99_any": [],
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
            delta = target_t - prev_01
            delta_abs = delta.abs()
            delta_norm = delta / delta_abs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            target_u = iv_to_unconstrained(
                target_t,
                lo=model.support_lo,
                hi=model.support_hi,
                eps=model.support_eps,
            )
            resid = target_u - mu
            chol = torch.linalg.cholesky(cov)
            whiten = torch.linalg.solve_triangular(chol, resid.unsqueeze(-1), upper=False).squeeze(-1)
            whiten_norm = whiten.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            whiten_dir = whiten / whiten_norm

            out["split"].append(np.full(end - start, split_name))
            out["window_idx"].append(batch_window_idx.copy())
            out["step"].append(np.full(end - start, step, dtype=np.int64))
            out["cond_feat"].append(cond.detach().cpu().numpy())
            out["resid_norm"].append(whiten_norm.squeeze(-1).detach().cpu().numpy())
            out["whiten_dir"].append(whiten_dir.detach().cpu().numpy())
            out["signed_delta_norm"].append(delta_norm.detach().cpu().numpy())
            out["top_cell"].append(delta_abs.argmax(dim=-1).detach().cpu().numpy())
            out["q95_any"].append((delta_abs >= q95).any(dim=-1).detach().cpu().numpy().astype(np.int64))
            out["q99_any"].append((delta_abs >= q99).any(dim=-1).detach().cpu().numpy().astype(np.int64))
            out["calm"].append(window_meta["calm"][batch_window_idx].astype(np.int64))
            out["turb"].append(window_meta["turb"][batch_window_idx].astype(np.int64))
            out["hard_late_window"].append(window_meta["hard_late"][batch_window_idx].astype(np.int64))
            out["hard_late_h30"].append((window_meta["hard_late"][batch_window_idx] & (step == future_len - 1)).astype(np.int64))

            next_frame = target_t.view(end - start, 1, 5, 5)
            context = torch.cat([context[:, 1:], next_frame], dim=1)

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def fit_conditional_nn(train_feat: np.ndarray, pca_dim: int, knn: int) -> tuple[dict[str, Any], NearestNeighbors]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    pca = PCA(n_components=min(pca_dim, train_feat.shape[1]), random_state=42)
    train_z = pca.fit_transform(train_scaled)
    nn = NearestNeighbors(n_neighbors=min(knn, train_z.shape[0]), metric="euclidean")
    nn.fit(train_z)
    return {"scaler": scaler, "pca": pca, "train_z": train_z}, nn


def transform_feat(x: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    return bundle["pca"].transform(bundle["scaler"].transform(x))


def evaluate_subset(
    subset_name: str,
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    knn: int,
    n_proto: int,
    max_queries: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size == 0 or test_idx.size == 0:
        return {
            "subset": subset_name,
            "n_train_pool": int(train_idx.size),
            "n_test_queries": int(test_idx.size),
            "status": "empty",
        }

    if test_idx.size > max_queries:
        test_idx = np.sort(rng.choice(test_idx, size=max_queries, replace=False))

    pool_feat = train_records["cond_feat"][train_idx]
    bundle, nn = fit_conditional_nn(pool_feat, pca_dim=32, knn=knn)
    query_feat = transform_feat(test_records["cond_feat"][test_idx], bundle)
    _, nbr_pos = nn.kneighbors(query_feat, return_distance=True)
    nbr_global = train_idx[nbr_pos]

    pool_top_cell = train_records["top_cell"][train_idx]
    top_counts = np.bincount(pool_top_cell, minlength=25).astype(np.float64)
    top_probs_global = top_counts / max(top_counts.sum(), 1.0)
    global_top_rank = np.argsort(top_probs_global)[::-1]
    global_template = train_records["signed_delta_norm"][train_idx].mean(axis=0)
    global_protos = greedy_prototypes(train_records["signed_delta_norm"][train_idx], n_proto=n_proto)

    local_top1_hits = []
    local_top3_hits = []
    global_top1_hits = []
    global_top3_hits = []
    local_template_mae = []
    global_template_mae = []
    local_proto_mae = []
    global_proto_mae = []
    local_topcell_entropy = []
    local_topcell_majority = []
    local_rank = []
    random_rank = []
    local_cos_signed = []
    random_cos_signed = []
    local_cos_whiten = []
    random_cos_whiten = []

    pool_signed = normalize_rows(train_records["signed_delta_norm"][train_idx])
    pool_whiten = normalize_rows(train_records["whiten_dir"][train_idx])
    random_knn = min(knn, train_idx.size)

    for q_abs_idx, neighbors_abs in zip(test_idx, nbr_global):
        q_top = int(test_records["top_cell"][q_abs_idx])
        q_signed = test_records["signed_delta_norm"][q_abs_idx]
        q_signed_unit = normalize_rows(q_signed[None, :])[0]
        q_whiten = normalize_rows(test_records["whiten_dir"][q_abs_idx][None, :])[0]

        n_local = neighbors_abs.shape[0]
        rand_abs = train_idx[rng.choice(train_idx.size, size=n_local, replace=False)]

        local_top = train_records["top_cell"][neighbors_abs]
        local_counts = np.bincount(local_top, minlength=25).astype(np.float64)
        local_probs = local_counts / max(local_counts.sum(), 1.0)
        local_ranked = np.argsort(local_probs)[::-1]

        local_top1_hits.append(float(q_top == local_ranked[0]))
        local_top3_hits.append(float(q_top in local_ranked[:3]))
        global_top1_hits.append(float(q_top == global_top_rank[0]))
        global_top3_hits.append(float(q_top in global_top_rank[:3]))

        local_template = train_records["signed_delta_norm"][neighbors_abs].mean(axis=0)
        local_template_mae.append(float(np.abs(local_template - q_signed).mean()))
        global_template_mae.append(float(np.abs(global_template - q_signed).mean()))

        local_protos = greedy_prototypes(train_records["signed_delta_norm"][neighbors_abs], n_proto=n_proto)
        local_proto_mae.append(prototype_min_mae(q_signed, local_protos))
        global_proto_mae.append(prototype_min_mae(q_signed, global_protos))

        local_topcell_entropy.append(safe_entropy_from_probs(local_probs))
        local_topcell_majority.append(float(local_probs.max()))

        local_wh = pool_whiten[np.searchsorted(train_idx, neighbors_abs)]
        rand_wh = pool_whiten[np.searchsorted(train_idx, rand_abs)]
        local_rank.append(effective_rank(local_wh))
        random_rank.append(effective_rank(rand_wh))

        local_signed = pool_signed[np.searchsorted(train_idx, neighbors_abs)]
        rand_signed = pool_signed[np.searchsorted(train_idx, rand_abs)]
        local_cos_signed.append(cosine_max(q_signed_unit, local_signed))
        random_cos_signed.append(cosine_max(q_signed_unit, rand_signed))
        local_cos_whiten.append(cosine_max(q_whiten, local_wh))
        random_cos_whiten.append(cosine_max(q_whiten, rand_wh))

    return {
        "subset": subset_name,
        "n_train_pool": int(train_idx.size),
        "n_test_queries": int(test_idx.size),
        "knn": int(min(knn, train_idx.size)),
        "n_prototypes": int(n_proto),
        "global_top1_acc": float(np.mean(global_top1_hits)),
        "local_top1_acc": float(np.mean(local_top1_hits)),
        "global_top3_acc": float(np.mean(global_top3_hits)),
        "local_top3_acc": float(np.mean(local_top3_hits)),
        "local_top1_lift": float(np.mean(local_top1_hits) - np.mean(global_top1_hits)),
        "local_top3_lift": float(np.mean(local_top3_hits) - np.mean(global_top3_hits)),
        "global_template_mae": float(np.mean(global_template_mae)),
        "local_template_mae": float(np.mean(local_template_mae)),
        "template_mae_improvement_pct": float(
            (np.mean(global_template_mae) - np.mean(local_template_mae)) / max(np.mean(global_template_mae), 1e-8)
        ),
        "global_proto_mae": float(np.mean(global_proto_mae)),
        "local_proto_mae": float(np.mean(local_proto_mae)),
        "proto_mae_improvement_pct": float(
            (np.mean(global_proto_mae) - np.mean(local_proto_mae)) / max(np.mean(global_proto_mae), 1e-8)
        ),
        "local_topcell_majority_share_mean": float(np.mean(local_topcell_majority)),
        "local_topcell_entropy_mean": float(np.mean(local_topcell_entropy)),
        "local_whiten_effective_rank_mean": float(np.mean(local_rank)),
        "random_whiten_effective_rank_mean": float(np.mean(random_rank)),
        "effective_rank_gap_vs_random": float(np.mean(local_rank) - np.mean(random_rank)),
        "local_nearest_cos_signed_mean": float(np.mean(local_cos_signed)),
        "random_nearest_cos_signed_mean": float(np.mean(random_cos_signed)),
        "signed_cos_gap_vs_random": float(np.mean(local_cos_signed) - np.mean(random_cos_signed)),
        "local_nearest_cos_whiten_mean": float(np.mean(local_cos_whiten)),
        "random_nearest_cos_whiten_mean": float(np.mean(random_cos_whiten)),
        "whiten_cos_gap_vs_random": float(np.mean(local_cos_whiten) - np.mean(random_cos_whiten)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="205a conditional shape-family audit on dangerous 201b innovations")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--n_prototypes", type=int, default=3)
    parser.add_argument("--max_query_steps", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/205_design/205a_conditional_shape_family_audit",
    )
    args = parser.parse_args()

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

    train_history, train_future = build_multistep_windows(
        split_indices["train"], surf_tensor, args.history_len, args.future_len
    )
    test_history, test_future = build_multistep_windows(
        split_indices["test"], surf_tensor, args.history_len, args.future_len
    )

    train_future_flat = train_future.view(train_future.shape[0], train_future.shape[1], -1).detach().cpu().numpy()
    test_future_flat = test_future.view(test_future.shape[0], test_future.shape[1], -1).detach().cpu().numpy()
    train_history_np = train_history.detach().cpu().numpy()
    test_history_np = test_history.detach().cpu().numpy()

    train_meta = build_window_metadata(train_history_np, train_future_flat)
    test_meta = build_window_metadata(
        test_history_np,
        test_future_flat,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    train_records = collect_teacher_forced_records(
        model=model,
        history_01=train_history,
        future_flat=train_future.view(train_future.shape[0], train_future.shape[1], -1),
        window_meta=train_meta,
        split_name="train",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )
    test_records = collect_teacher_forced_records(
        model=model,
        history_01=test_history,
        future_flat=test_future.view(test_future.shape[0], test_future.shape[1], -1),
        window_meta=test_meta,
        split_name="test",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )

    subset_defs = {
        "all_q99_steps": (
            train_records["q99_any"] == 1,
            test_records["q99_any"] == 1,
        ),
        "turb_q99_steps": (
            (train_records["q99_any"] == 1) & (train_records["turb"] == 1),
            (test_records["q99_any"] == 1) & (test_records["turb"] == 1),
        ),
        "hard_window_q99_steps": (
            (train_records["q99_any"] == 1) & (train_records["hard_late_window"] == 1),
            (test_records["q99_any"] == 1) & (test_records["hard_late_window"] == 1),
        ),
        "hard_h30_all_steps": (
            train_records["hard_late_h30"] == 1,
            test_records["hard_late_h30"] == 1,
        ),
    }

    subset_results = {}
    for name, (train_mask, test_mask) in subset_defs.items():
        subset_results[name] = evaluate_subset(
            subset_name=name,
            train_records=train_records,
            test_records=test_records,
            train_mask=train_mask,
            test_mask=test_mask,
            knn=args.knn,
            n_proto=args.n_prototypes,
            max_queries=args.max_query_steps,
            seed=args.seed,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "model_type": payload["config"]["type"],
        "thresholds": {"q95": q95, "q99": q99},
        "window_counts": {
            "train_windows": int(train_history.shape[0]),
            "test_windows": int(test_history.shape[0]),
            "train_hard_late_windows": int(train_meta["hard_late"].sum()),
            "test_hard_late_windows": int(test_meta["hard_late"].sum()),
        },
        "subset_results": subset_results,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
