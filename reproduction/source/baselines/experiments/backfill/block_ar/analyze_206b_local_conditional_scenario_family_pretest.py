#!/usr/bin/env python
"""
Pretest a directly local conditional scenario-family mechanism on dangerous 201b steps.

Goal:
  Compare a local top-K scenario family against the current 201b teacher-forced
  uncertainty allocation on the same dangerous next-step targets.
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

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import make_serializable
from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_split_indices,
    build_window_metadata,
    load_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import reshape_history


def fit_feature_space(train_feat: np.ndarray, pca_dim: int) -> tuple[dict[str, Any], np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    pca = PCA(n_components=min(pca_dim, train_feat.shape[1]), random_state=42)
    train_z = pca.fit_transform(train_scaled)
    return {"scaler": scaler, "pca": pca}, train_z


def transform_feat(x: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    return bundle["pca"].transform(bundle["scaler"].transform(x))


def farthest_first_subset(x: np.ndarray, n_select: int) -> np.ndarray:
    if x.shape[0] <= n_select:
        return np.arange(x.shape[0], dtype=np.int64)
    center = x.mean(axis=0, keepdims=True)
    d = ((x - center) ** 2).sum(axis=1)
    selected = [int(d.argmin())]
    min_d = ((x - x[selected[0] : selected[0] + 1]) ** 2).sum(axis=1)
    while len(selected) < n_select:
        nxt = int(min_d.argmax())
        selected.append(nxt)
        cur = ((x - x[nxt : nxt + 1]) ** 2).sum(axis=1)
        min_d = np.minimum(min_d, cur)
    return np.array(selected, dtype=np.int64)


@torch.no_grad()
def collect_width_maps(
    model,
    history_01: torch.Tensor,
    future_flat: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out = {
        "std_map": [],
        "top1_std_cell": [],
        "top3_std_cells": [],
        "step": [],
        "window_idx": [],
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
            mu, factor, diag, scale, nu = model.forward_from_history(context)
            cov = model.covariance(factor, diag, scale)
            std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8))
            top3 = std.topk(k=min(3, std.shape[1]), dim=-1).indices.detach().cpu().numpy()
            out["std_map"].append(std.detach().cpu().numpy())
            out["top1_std_cell"].append(top3[:, 0])
            out["top3_std_cells"].append(top3)
            out["step"].append(np.full(end - start, step, dtype=np.int64))
            out["window_idx"].append(batch_window_idx.copy())

            next_frame = fut[:, step].view(end - start, 1, 5, 5)
            context = torch.cat([context[:, 1:], next_frame], dim=1)

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def eval_subset(
    subset_name: str,
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    width_records: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    knn: int,
    family_size: int,
    max_queries: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size == 0 or test_idx.size == 0:
        return {"subset": subset_name, "status": "empty"}

    if test_idx.size > max_queries:
        test_idx = np.sort(rng.choice(test_idx, size=max_queries, replace=False))

    feat_bundle, train_z = fit_feature_space(train_records["cond_feat"][train_idx], pca_dim=32)
    nn = NearestNeighbors(n_neighbors=min(knn, train_idx.size), metric="euclidean")
    nn.fit(train_z)
    test_z = transform_feat(test_records["cond_feat"][test_idx], feat_bundle)
    _, nbr_pos = nn.kneighbors(test_z, return_distance=True)
    nbr_idx = train_idx[nbr_pos]

    global_patterns = train_records["signed_delta_norm"][train_idx]
    global_select = farthest_first_subset(global_patterns, family_size)
    global_family = global_patterns[global_select]
    global_top_union = np.unique(np.abs(global_family).argmax(axis=1))

    local_best_mae = []
    global_best_mae = []
    width_top1_hits = []
    width_top3_hits = []
    local_union_top1_hits = []
    global_union_top1_hits = []
    local_union_top3_hits = []
    global_union_top3_hits = []
    local_union_size = []
    local_family_dispersion = []

    for q_abs, neighbors_abs in zip(test_idx, nbr_idx):
        q_pattern = test_records["signed_delta_norm"][q_abs]
        q_top = int(test_records["top_cell"][q_abs])
        width_top1 = int(width_records["top1_std_cell"][q_abs])
        width_top3 = width_records["top3_std_cells"][q_abs]

        fam = train_records["signed_delta_norm"][neighbors_abs]
        fam_sel = farthest_first_subset(fam, family_size)
        fam = fam[fam_sel]
        fam_top = np.abs(fam).argmax(axis=1)
        fam_union = np.unique(fam_top)

        local_best_mae.append(float(np.abs(fam - q_pattern[None, :]).mean(axis=1).min()))
        global_best_mae.append(float(np.abs(global_family - q_pattern[None, :]).mean(axis=1).min()))

        width_top1_hits.append(float(q_top == width_top1))
        width_top3_hits.append(float(q_top in width_top3))
        local_union_top1_hits.append(float(q_top in fam_union))
        global_union_top1_hits.append(float(q_top in global_top_union))

        local_top3_cells = np.unique(np.argsort(np.abs(fam), axis=1)[:, -3:].reshape(-1))
        global_top3_cells = np.unique(np.argsort(np.abs(global_family), axis=1)[:, -3:].reshape(-1))
        local_union_top3_hits.append(float(q_top in local_top3_cells))
        global_union_top3_hits.append(float(q_top in global_top3_cells))
        local_union_size.append(float(fam_union.size))

        centered = fam - fam.mean(axis=0, keepdims=True)
        local_family_dispersion.append(float(np.sqrt((centered**2).mean())))

    return {
        "subset": subset_name,
        "n_train_pool": int(train_idx.size),
        "n_test_queries": int(test_idx.size),
        "knn": int(min(knn, train_idx.size)),
        "family_size": int(family_size),
        "width_top1_acc": float(np.mean(width_top1_hits)),
        "width_top3_acc": float(np.mean(width_top3_hits)),
        "local_family_union_top1_acc": float(np.mean(local_union_top1_hits)),
        "global_family_union_top1_acc": float(np.mean(global_union_top1_hits)),
        "local_family_union_top3_acc": float(np.mean(local_union_top3_hits)),
        "global_family_union_top3_acc": float(np.mean(global_union_top3_hits)),
        "local_vs_width_top1_lift": float(np.mean(local_union_top1_hits) - np.mean(width_top1_hits)),
        "local_vs_width_top3_lift": float(np.mean(local_union_top3_hits) - np.mean(width_top3_hits)),
        "local_best_proto_mae": float(np.mean(local_best_mae)),
        "global_best_proto_mae": float(np.mean(global_best_mae)),
        "local_vs_global_proto_improvement_pct": float(
            (np.mean(global_best_mae) - np.mean(local_best_mae)) / max(np.mean(global_best_mae), 1e-8)
        ),
        "local_family_top_union_size_mean": float(np.mean(local_union_size)),
        "local_family_dispersion_mean": float(np.mean(local_family_dispersion)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="206b local conditional scenario-family pretest")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--family_size", type=int, default=3)
    parser.add_argument("--max_queries", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/206_design/206b_local_conditional_scenario_family_pretest",
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

    train_hist, train_future = build_multistep_windows(split_indices["train"], surf_tensor, args.history_len, args.future_len)
    test_hist, test_future = build_multistep_windows(split_indices["test"], surf_tensor, args.history_len, args.future_len)

    train_future_np = train_future.view(train_future.shape[0], train_future.shape[1], -1).detach().cpu().numpy()
    test_future_np = test_future.view(test_future.shape[0], test_future.shape[1], -1).detach().cpu().numpy()
    train_hist_np = train_hist.detach().cpu().numpy()
    test_hist_np = test_hist.detach().cpu().numpy()
    train_meta = build_window_metadata(train_hist_np, train_future_np)
    test_meta = build_window_metadata(
        test_hist_np,
        test_future_np,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    train_records = collect_teacher_forced_records(
        model=model,
        history_01=train_hist,
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
        history_01=test_hist,
        future_flat=test_future.view(test_future.shape[0], test_future.shape[1], -1),
        window_meta=test_meta,
        split_name="test",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )
    width_records = collect_width_maps(
        model=model,
        history_01=test_hist,
        future_flat=test_future.view(test_future.shape[0], test_future.shape[1], -1),
        batch_size=args.batch_size,
        device=device,
    )

    subset_defs = {
        "turb_q99_steps": (
            (train_records["q99_any"] == 1) & (train_records["turb"] == 1),
            (test_records["q99_any"] == 1) & (test_records["turb"] == 1),
        ),
        "hard_window_q99_steps": (
            (train_records["q99_any"] == 1) & (train_records["hard_late_window"] == 1),
            (test_records["q99_any"] == 1) & (test_records["hard_late_window"] == 1),
        ),
        "hard_h30_steps": (
            train_records["hard_late_h30"] == 1,
            test_records["hard_late_h30"] == 1,
        ),
    }

    subset_results = {}
    for name, (train_mask, test_mask) in subset_defs.items():
        subset_results[name] = eval_subset(
            subset_name=name,
            train_records=train_records,
            test_records=test_records,
            width_records=width_records,
            train_mask=train_mask,
            test_mask=test_mask,
            knn=args.knn,
            family_size=args.family_size,
            max_queries=args.max_queries,
            seed=args.seed,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "model_type": payload["config"]["type"],
        "thresholds": {"q95": q95, "q99": q99},
        "subset_results": subset_results,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
