#!/usr/bin/env python
"""
208a: H=1 teacher-guided local-family Transformer Student-t.

Narrow one-step branch:
  - keep the 207b local-family law
  - remove rollout entirely
  - train directly on one-step NLL
  - add teacher-guided gate/family supervision
  - add a one-step spectrum loss to reduce shoulder smear
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_window_metadata,
    load_model as load_teacher_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.analyze_206b_local_conditional_scenario_family_pretest import (
    farthest_first_subset,
    fit_feature_space,
    transform_feat,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    maybe_load_decoder_warm_start,
)
from experiments.backfill.block_ar.train_207a_local_conditional_family_student_t import (
    TransformerLocalConditionalFamilyARModel,
    evaluate_family_statistics,
)


PERMUTATIONS_3 = list(itertools.permutations(range(3)))


def softmax_np(x: np.ndarray, temp: float) -> np.ndarray:
    z = -x / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / np.clip(e.sum(), 1e-8, None)


def pad_family(fam: np.ndarray, family_size: int) -> np.ndarray:
    if fam.shape[0] >= family_size:
        return fam[:family_size]
    if fam.shape[0] == 0:
        return np.zeros((family_size, 25), dtype=np.float32)
    pad = np.repeat(fam[-1:], family_size - fam.shape[0], axis=0)
    return np.concatenate([fam, pad], axis=0)


def build_teacher_targets_for_split(
    train_records: dict[str, np.ndarray],
    split_records: dict[str, np.ndarray],
    n_windows: int,
    family_size: int,
    knn: int,
    target_temp: float,
    exclude_self: bool,
) -> dict[str, torch.Tensor]:
    gate_targets = np.zeros((n_windows,), dtype=np.float32)
    family_mask = np.zeros((n_windows,), dtype=np.float32)
    family_shapes = np.zeros((n_windows, family_size, 25), dtype=np.float32)
    family_probs = np.zeros((n_windows, family_size), dtype=np.float32)

    gate_targets[split_records["window_idx"]] = split_records["q95_any"].astype(np.float32)

    train_pool_mask = train_records["q99_any"] == 1
    train_pool_idx = np.flatnonzero(train_pool_mask)
    if train_pool_idx.size == 0:
        return {
            "gate_targets": torch.from_numpy(gate_targets),
            "family_mask": torch.from_numpy(family_mask),
            "family_shapes": torch.from_numpy(family_shapes),
            "family_probs": torch.from_numpy(family_probs),
        }

    feat_bundle, train_z = fit_feature_space(train_records["cond_feat"][train_pool_idx], pca_dim=32)
    nn = NearestNeighbors(n_neighbors=min(knn + int(exclude_self), train_pool_idx.size), metric="euclidean")
    nn.fit(train_z)
    split_z = transform_feat(split_records["cond_feat"], feat_bundle)
    _, nbr_pos = nn.kneighbors(split_z, return_distance=True)
    nbr_idx = train_pool_idx[nbr_pos]

    for q_abs in range(split_records["window_idx"].shape[0]):
        if split_records["q99_any"][q_abs] != 1:
            continue

        window_idx = int(split_records["window_idx"][q_abs])
        neighbors = nbr_idx[q_abs]
        if exclude_self:
            neighbors = neighbors[neighbors != q_abs]
        if neighbors.size == 0:
            neighbors = nbr_idx[q_abs][:1]

        fam = train_records["signed_delta_norm"][neighbors]
        fam = pad_family(fam, family_size=max(family_size, 1))
        fam_sel = farthest_first_subset(fam, family_size)
        fam = pad_family(fam[fam_sel], family_size)

        query = split_records["signed_delta_norm"][q_abs]
        mae = np.abs(fam - query[None, :]).mean(axis=1)
        order = np.argsort(mae)
        fam = fam[order]
        mae = mae[order]
        probs = softmax_np(mae, temp=target_temp).astype(np.float32)

        family_mask[window_idx] = 1.0
        family_shapes[window_idx] = fam.astype(np.float32)
        family_probs[window_idx] = probs

    return {
        "gate_targets": torch.from_numpy(gate_targets),
        "family_mask": torch.from_numpy(family_mask),
        "family_shapes": torch.from_numpy(family_shapes),
        "family_probs": torch.from_numpy(family_probs),
    }


def build_teacher_guidance_targets(
    teacher_checkpoint: str,
    train_history: torch.Tensor,
    train_target: torch.Tensor,
    val_history: torch.Tensor,
    val_target: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
    batch_size: int,
    device: torch.device,
    family_size: int,
    knn: int,
    target_temp: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    teacher_model, _payload = load_teacher_model(teacher_checkpoint, device)
    train_future = train_target.unsqueeze(1)
    val_future = val_target.unsqueeze(1)
    train_future_np = train_future.detach().cpu().numpy()
    val_future_np = val_future.detach().cpu().numpy()
    train_history_np = train_history.detach().cpu().numpy()
    val_history_np = val_history.detach().cpu().numpy()

    train_meta = build_window_metadata(train_history_np, train_future_np)
    val_meta = build_window_metadata(
        val_history_np,
        val_future_np,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    train_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=train_history,
        future_flat=train_future,
        window_meta=train_meta,
        split_name="train",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )
    val_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=val_history,
        future_flat=val_future,
        window_meta=val_meta,
        split_name="val",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )

    train_targets = build_teacher_targets_for_split(
        train_records=train_records,
        split_records=train_records,
        n_windows=train_history.shape[0],
        family_size=family_size,
        knn=knn,
        target_temp=target_temp,
        exclude_self=True,
    )
    val_targets = build_teacher_targets_for_split(
        train_records=train_records,
        split_records=val_records,
        n_windows=val_history.shape[0],
        family_size=family_size,
        knn=knn,
        target_temp=target_temp,
        exclude_self=False,
    )
    return train_targets, val_targets


def one_step_teacher_guidance_objective(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    gate_targets: torch.Tensor,
    family_mask: torch.Tensor,
    family_shapes_target: torch.Tensor,
    family_probs_target: torch.Tensor,
    gate_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _mu, _factor, _diag, _scale, _nu = model.forward_from_history(history_01)
    family = model.last_family_params()

    pos_weight = history_01.new_tensor(gate_pos_weight)
    gate_t = gate_targets.to(history_01.device)
    gate_loss = F.binary_cross_entropy_with_logits(
        family["gate_logit"],
        gate_t,
        pos_weight=pos_weight,
        reduction="mean",
    )

    shape_loss = history_01.new_tensor(0.0)
    prob_loss = history_01.new_tensor(0.0)
    top1_match = history_01.new_tensor(0.0)
    active_rate = (family_mask > 0.5).float().mean()

    mask = family_mask.to(history_01.device) > 0.5
    if mask.any():
        pred_shapes = family["family_shapes"][mask]
        pred_logits = family["family_logits"][mask]
        tgt_shapes = family_shapes_target.to(history_01.device)[mask]
        tgt_probs = family_probs_target.to(history_01.device)[mask]

        shape_losses = []
        prob_losses = []
        top1_matches = []
        for i in range(pred_shapes.shape[0]):
            best_cost = None
            best_probs = None
            best_top1 = None
            best_cost_value = None
            for perm in PERMUTATIONS_3:
                perm_idx = list(perm)
                aligned_shapes = tgt_shapes[i, perm_idx]
                aligned_probs = tgt_probs[i, perm_idx]
                shape_cost = (aligned_probs * (pred_shapes[i] - aligned_shapes).abs().mean(dim=-1)).sum()
                shape_cost_value = float(shape_cost.detach().item())
                if best_cost is None or shape_cost_value < best_cost_value:
                    best_cost = shape_cost
                    best_probs = aligned_probs
                    best_top1 = float(pred_logits[i].argmax().item() == int(aligned_probs.argmax().item()))
                    best_cost_value = shape_cost_value
            shape_losses.append(best_cost)
            prob_losses.append(-(best_probs * F.log_softmax(pred_logits[i], dim=-1)).sum())
            top1_matches.append(best_top1)

        shape_loss = torch.stack(shape_losses).mean()
        prob_loss = torch.stack(prob_losses).mean()
        top1_match = history_01.new_tensor(top1_matches).mean()

    total = gate_loss + shape_loss + prob_loss
    metrics = {
        "teacher_guidance_total_loss": total.detach(),
        "teacher_gate_bce": gate_loss.detach(),
        "teacher_family_shape_loss": shape_loss.detach(),
        "teacher_family_prob_loss": prob_loss.detach(),
        "teacher_gate_target_rate": gate_targets.mean().detach(),
        "teacher_family_mask_rate": family_mask.mean().detach(),
        "teacher_family_top1_match": top1_match.detach(),
        "teacher_family_active_rate": active_rate.detach(),
    }
    return total, metrics


def sample_reparameterized_next_iv(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    mu, factor, diag, scale, nu = model.forward_from_history(history_01)
    samples_u = [
        model.reparameterized_next_u(mu, factor, diag, scale, nu)
        for _ in range(n_samples)
    ]
    samples_u = torch.stack(samples_u, dim=1)
    return unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)


def h1_spectrum_objective(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    q50_threshold: float,
    q95_threshold: float,
    q99_threshold: float,
    n_samples: int,
    indicator_temp: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prev = history_01[:, -1].reshape(history_01.shape[0], -1)
    samples_01 = sample_reparameterized_next_iv(model, history_01, n_samples=n_samples)

    target_abs = (target_01 - prev).abs().reshape(-1)
    sample_abs = (samples_01 - prev[:, None, :]).abs().reshape(-1)
    temp = max(indicator_temp, 1e-5)

    def smooth_leq(x: torch.Tensor, thr: float) -> torch.Tensor:
        return torch.sigmoid((thr - x) / temp)

    def smooth_geq(x: torch.Tensor, thr: float) -> torch.Tensor:
        return torch.sigmoid((x - thr) / temp)

    quiet_t = smooth_leq(target_abs, q50_threshold).mean()
    quiet_s = smooth_leq(sample_abs, q50_threshold).mean()

    shoulder_t = (smooth_geq(target_abs, q50_threshold) * smooth_leq(target_abs, q95_threshold)).mean()
    shoulder_s = (smooth_geq(sample_abs, q50_threshold) * smooth_leq(sample_abs, q95_threshold)).mean()

    extreme_t = smooth_geq(target_abs, q99_threshold).mean()
    extreme_s = smooth_geq(sample_abs, q99_threshold).mean()

    quiet_loss = (quiet_s - quiet_t).pow(2)
    shoulder_loss = (shoulder_s - shoulder_t).pow(2)
    extreme_loss = (extreme_s - extreme_t).pow(2)
    total = quiet_loss + shoulder_loss + extreme_loss
    metrics = {
        "spectrum_loss": total.detach(),
        "quiet_loss": quiet_loss.detach(),
        "shoulder_loss": shoulder_loss.detach(),
        "extreme_loss": extreme_loss.detach(),
        "quiet_target": quiet_t.detach(),
        "quiet_sample": quiet_s.detach(),
        "shoulder_target": shoulder_t.detach(),
        "shoulder_sample": shoulder_s.detach(),
        "extreme_target": extreme_t.detach(),
        "extreme_sample": extreme_s.detach(),
    }
    return total, metrics


def compute_h1_shape_stats(
    gt_delta: np.ndarray,
    sample_delta: np.ndarray,
) -> dict[str, float]:
    gt_abs = np.abs(gt_delta.reshape(-1))
    ro_abs = np.abs(sample_delta.reshape(-1))
    gt_q50 = float(np.quantile(gt_abs, 0.5))
    gt_q95 = float(np.quantile(gt_abs, 0.95))
    gt_q99 = float(np.quantile(gt_abs, 0.99))

    def kurtosis(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        m = x.mean()
        v = ((x - m) ** 2).mean()
        if v <= 1e-12:
            return float("nan")
        return float(((x - m) ** 4).mean() / (v ** 2))

    gt_kurt = kurtosis(gt_delta.reshape(-1))
    ro_kurt = kurtosis(sample_delta.reshape(-1))

    # Management-facing absolute move thresholds in raw IV delta space.
    move_thresholds = {
        "very_small_0p005": 0.005,
        "small_0p010": 0.01,
        "moderate_0p020": 0.02,
        "large_0p050": 0.05,
    }

    move_size_stats: dict[str, float] = {}
    for name, thr in move_thresholds.items():
        gt_share = float((gt_abs <= thr).mean())
        ro_share = float((ro_abs <= thr).mean())
        move_size_stats[f"{name}_gt_share"] = gt_share
        move_size_stats[f"{name}_sample_share"] = ro_share
        move_size_stats[f"{name}_ratio"] = float(ro_share / max(gt_share, 1e-8))

    out = {
        "quiet_ratio": float((ro_abs <= gt_q50).mean() / max((gt_abs <= gt_q50).mean(), 1e-8)),
        "shoulder_ratio": float(
            (((ro_abs > gt_q50) & (ro_abs <= gt_q95)).mean())
            / max((((gt_abs > gt_q50) & (gt_abs <= gt_q95)).mean()), 1e-8)
        ),
        "extreme_ratio": float((ro_abs > gt_q99).mean() / max((gt_abs > gt_q99).mean(), 1e-8)),
        "q50_ratio": float(np.quantile(ro_abs, 0.5) / max(np.quantile(gt_abs, 0.5), 1e-8)),
        "q95_ratio": float(np.quantile(ro_abs, 0.95) / max(np.quantile(gt_abs, 0.95), 1e-8)),
        "q99_ratio": float(np.quantile(ro_abs, 0.99) / max(np.quantile(gt_abs, 0.99), 1e-8)),
        "gt_kurtosis": float(gt_kurt),
        "sample_kurtosis": float(ro_kurt),
        "kurtosis_ratio": float(ro_kurt / max(gt_kurt, 1e-8)),
    }
    out.update(move_size_stats)
    return out


@torch.no_grad()
def evaluate_h1(
    model: TransformerLocalConditionalFamilyARModel,
    loader: DataLoader,
    q50_threshold: float,
    q95_threshold: float,
    q99_threshold: float,
    gate_pos_weight: float,
    spectrum_samples: int,
    indicator_temp: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    total = {
        "val_nll": 0.0,
        "val_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
        "val_realized_q95_coverage_90": 0.0,
        "val_realized_q99_coverage_90": 0.0,
        "val_teacher_guidance_total_loss": 0.0,
        "val_teacher_gate_bce": 0.0,
        "val_teacher_family_shape_loss": 0.0,
        "val_teacher_family_prob_loss": 0.0,
        "val_teacher_family_top1_match": 0.0,
        "val_spectrum_loss": 0.0,
        "val_quiet_loss": 0.0,
        "val_shoulder_loss": 0.0,
        "val_extreme_loss": 0.0,
    }
    total_q95 = 0
    total_q99 = 0
    gt_delta_all = []
    sample_delta_all = []
    total_count = 0

    for history_01, target_01, gate_targets, family_mask, family_shapes, family_probs in loader:
        mu, factor, diag, scale, nu = model.forward_from_history(history_01)
        target_u = iv_to_unconstrained(
            target_01,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        nll = model.student_t_nll(target_u, mu, factor, diag, scale, nu)
        mean_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        guidance_loss, guidance_metrics = one_step_teacher_guidance_objective(
            model,
            history_01,
            gate_targets,
            family_mask,
            family_shapes,
            family_probs,
            gate_pos_weight=gate_pos_weight,
        )
        spectrum_loss, spectrum_metrics = h1_spectrum_objective(
            model,
            history_01,
            target_01,
            q50_threshold=q50_threshold,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            n_samples=spectrum_samples,
            indicator_temp=indicator_temp,
        )

        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        q05 = samples.quantile(0.05, dim=1)
        q95 = samples.quantile(0.95, dim=1)
        coverage = ((target_01 >= q05) & (target_01 <= q95)).float().mean()
        width = (q95 - q05).mean()

        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_abs = (target_01 - prev).abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold
        q95_cov = ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean() if q95_mask.any() else target_01.new_tensor(0.0)
        q99_cov = ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean() if q99_mask.any() else target_01.new_tensor(0.0)

        batch_size = history_01.shape[0]
        total["val_nll"] += float(nll.mean().item()) * batch_size
        total["val_mae"] += float((mean_iv - target_01).abs().mean().item()) * batch_size
        total["val_coverage_90"] += float(coverage.item()) * batch_size
        total["val_width_90"] += float(width.item()) * batch_size
        total["val_teacher_guidance_total_loss"] += float(guidance_metrics["teacher_guidance_total_loss"].item()) * batch_size
        total["val_teacher_gate_bce"] += float(guidance_metrics["teacher_gate_bce"].item()) * batch_size
        total["val_teacher_family_shape_loss"] += float(guidance_metrics["teacher_family_shape_loss"].item()) * batch_size
        total["val_teacher_family_prob_loss"] += float(guidance_metrics["teacher_family_prob_loss"].item()) * batch_size
        total["val_teacher_family_top1_match"] += float(guidance_metrics["teacher_family_top1_match"].item()) * batch_size
        total["val_spectrum_loss"] += float(spectrum_metrics["spectrum_loss"].item()) * batch_size
        total["val_quiet_loss"] += float(spectrum_metrics["quiet_loss"].item()) * batch_size
        total["val_shoulder_loss"] += float(spectrum_metrics["shoulder_loss"].item()) * batch_size
        total["val_extreme_loss"] += float(spectrum_metrics["extreme_loss"].item()) * batch_size
        total["val_realized_q95_coverage_90"] += float(q95_cov.item()) * int(q95_mask.any(dim=-1).sum().item())
        total["val_realized_q99_coverage_90"] += float(q99_cov.item()) * int(q99_mask.any(dim=-1).sum().item())
        total_q95 += int(q95_mask.any(dim=-1).sum().item())
        total_q99 += int(q99_mask.any(dim=-1).sum().item())
        total_count += batch_size

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev[:, None, :]).detach().cpu().numpy())

    out = {k: v / max(total_count, 1) for k, v in total.items()}
    out["val_realized_q95_coverage_90"] = out["val_realized_q95_coverage_90"] * total_count / max(total_q95, 1)
    out["val_realized_q99_coverage_90"] = out["val_realized_q99_coverage_90"] * total_count / max(total_q99, 1)

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    out.update({f"val_h1_{k}": v for k, v in shape.items()})
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TransformerLocalConditionalFamilyARModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_teacher_guided_local_family_student_t_208a":
        raise ValueError(f"Unexpected model type: {raw_config['type']}")
    model = TransformerLocalConditionalFamilyARModel(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="208a H=1 teacher-guided local-family Student-t")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--enc_d_model", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_d_model", type=int, default=128)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--decoder_warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--teacher_checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--teacher_knn", type=int, default=32)
    parser.add_argument("--teacher_target_temp", type=float, default=0.15)
    parser.add_argument("--teacher_guidance_weight", type=float, default=0.25)
    parser.add_argument("--gate_pos_weight", type=float, default=3.0)
    parser.add_argument("--n_family", type=int, default=3)
    parser.add_argument("--family_scale_floor", type=float, default=5e-4)
    parser.add_argument("--init_gate_prob", type=float, default=0.10)
    parser.add_argument("--init_family_scale", type=float, default=0.05)
    parser.add_argument("--family_shape_scale", type=float, default=0.05)
    parser.add_argument("--family_nu", type=float, default=4.0)
    parser.add_argument("--gate_temperature", type=float, default=0.5)
    parser.add_argument("--family_temperature", type=float, default=0.6)
    parser.add_argument("--gate_prob_eps", type=float, default=1e-4)
    parser.add_argument("--spectrum_weight", type=float, default=1.0)
    parser.add_argument("--spectrum_samples", type=int, default=16)
    parser.add_argument("--spectrum_temp", type=float, default=0.01)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[:4511], axis=0).reshape(-1))
    q50_threshold = float(np.quantile(train_abs_delta, 0.50))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    hist_len = args.history_len
    test_start = 4511
    max_train_idx = test_start - hist_len - 30
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, hist_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, hist_len)

    train_targets, val_targets = build_teacher_guidance_targets(
        teacher_checkpoint=args.teacher_checkpoint,
        train_history=train_hist,
        train_target=train_target,
        val_history=val_hist,
        val_target=val_target,
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        batch_size=args.batch_size,
        device=device,
        family_size=args.n_family,
        knn=args.teacher_knn,
        target_temp=args.teacher_target_temp,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_hist,
            train_target,
            train_targets["gate_targets"].to(device),
            train_targets["family_mask"].to(device),
            train_targets["family_shapes"].to(device),
            train_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_hist,
            val_target,
            val_targets["gate_targets"].to(device),
            val_targets["family_mask"].to(device),
            val_targets["family_shapes"].to(device),
            val_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    family_val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    encoder_config = dict(
        input_dim=25,
        d_model=args.enc_d_model,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.enc_dropout,
        bottleneck_dim=128,
        max_len=max(hist_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.dec_d_model,
        n_heads=args.dec_heads,
        n_layers=args.dec_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
        n_family=args.n_family,
        family_scale_floor=args.family_scale_floor,
        init_gate_prob=args.init_gate_prob,
        init_family_scale=args.init_family_scale,
        family_shape_scale=args.family_shape_scale,
        family_nu=args.family_nu,
        gate_temperature=args.gate_temperature,
        family_temperature=args.family_temperature,
        gate_prob_eps=args.gate_prob_eps,
    )

    model = TransformerLocalConditionalFamilyARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_decoder_warm_start(model, args.decoder_warm_start)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 76}")
    print("208a: H=1 teacher-guided local-family Student-t")
    print(f"{'=' * 76}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  Thresholds: q50={q50_threshold:.5f}, q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Teacher KNN / family size: {args.teacher_knn} / {args.n_family}")
    print(f"  Teacher guidance weight: {args.teacher_guidance_weight}")
    print(f"  Spectrum weight / samples: {args.spectrum_weight} / {args.spectrum_samples}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "train_nll": 0.0,
            "train_mae": 0.0,
            "train_teacher_guidance_total_loss": 0.0,
            "train_teacher_gate_bce": 0.0,
            "train_teacher_family_shape_loss": 0.0,
            "train_teacher_family_prob_loss": 0.0,
            "train_teacher_family_top1_match": 0.0,
            "train_spectrum_loss": 0.0,
            "train_quiet_loss": 0.0,
            "train_shoulder_loss": 0.0,
            "train_extreme_loss": 0.0,
        }
        nb = 0

        for history_01, target_01, gate_targets, family_mask, family_shapes, family_probs in train_loader:
            optimizer.zero_grad()

            mu, factor, diag, scale, nu = model.forward_from_history(history_01)
            target_u = iv_to_unconstrained(
                target_01,
                lo=model.support_lo,
                hi=model.support_hi,
                eps=model.support_eps,
            )
            nll = model.student_t_nll(target_u, mu, factor, diag, scale, nu).mean()
            mean_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
            mae = (mean_iv - target_01).abs().mean()

            guidance_loss, guidance_metrics = one_step_teacher_guidance_objective(
                model,
                history_01,
                gate_targets,
                family_mask,
                family_shapes,
                family_probs,
                gate_pos_weight=args.gate_pos_weight,
            )
            spectrum_loss, spectrum_metrics = h1_spectrum_objective(
                model,
                history_01,
                target_01,
                q50_threshold=q50_threshold,
                q95_threshold=q95_threshold,
                q99_threshold=q99_threshold,
                n_samples=args.spectrum_samples,
                indicator_temp=args.spectrum_temp,
            )

            loss = nll + args.teacher_guidance_weight * guidance_loss + args.spectrum_weight * spectrum_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep["train_nll"] += float(nll.item())
            ep["train_mae"] += float(mae.item())
            ep["train_teacher_guidance_total_loss"] += float(guidance_metrics["teacher_guidance_total_loss"].item())
            ep["train_teacher_gate_bce"] += float(guidance_metrics["teacher_gate_bce"].item())
            ep["train_teacher_family_shape_loss"] += float(guidance_metrics["teacher_family_shape_loss"].item())
            ep["train_teacher_family_prob_loss"] += float(guidance_metrics["teacher_family_prob_loss"].item())
            ep["train_teacher_family_top1_match"] += float(guidance_metrics["teacher_family_top1_match"].item())
            ep["train_spectrum_loss"] += float(spectrum_metrics["spectrum_loss"].item())
            ep["train_quiet_loss"] += float(spectrum_metrics["quiet_loss"].item())
            ep["train_shoulder_loss"] += float(spectrum_metrics["shoulder_loss"].item())
            ep["train_extreme_loss"] += float(spectrum_metrics["extreme_loss"].item())
            nb += 1

        scheduler.step()

        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q50_threshold=q50_threshold,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            gate_pos_weight=args.gate_pos_weight,
            spectrum_samples=args.spectrum_samples,
            indicator_temp=args.spectrum_temp,
            eval_samples=args.val_samples,
        )
        family_metrics = evaluate_family_statistics(model, family_val_loader)
        val_metrics.update(family_metrics)

        selection_score = (
            val_metrics["val_nll"]
            + args.teacher_guidance_weight * val_metrics["val_teacher_guidance_total_loss"]
            + args.spectrum_weight * val_metrics["val_spectrum_loss"]
        )
        val_metrics["selection_score"] = selection_score

        elapsed = time.time() - t0
        row = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_sec": elapsed}
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_teacher_guided_local_family_student_t_208a",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, Path(args.output_dir) / "final_model.pt")

        if selection_score < best_score:
            best_score = selection_score
            best_metrics = row
            torch.save(ckpt, Path(args.output_dir) / "best_model.pt")
            best_flag = "  *best"
        else:
            best_flag = ""

        print(
            f"Ep {epoch:>3d}  "
            f"train_nll={train_metrics['train_nll']:.4f}  "
            f"train_tg={train_metrics['train_teacher_guidance_total_loss']:.4f}  "
            f"train_sp={train_metrics['train_spectrum_loss']:.4f}  "
            f"val_nll={val_metrics['val_nll']:.4f}  "
            f"val_cov90={val_metrics['val_coverage_90']:.4f}  "
            f"val_q99cov={val_metrics['val_realized_q99_coverage_90']:.4f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"gate={val_metrics['val_gate_prob_mean']:.3f}  "
            f"fam_top1={val_metrics['val_family_top1_mean']:.3f}  "
            f"sel={selection_score:.4f}  "
            f"({elapsed:.1f}s){best_flag}"
        )

        with open(Path(args.output_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if best_metrics is not None:
        print("\nBest metrics:")
        for k, v in best_metrics.items():
            if k == "epoch":
                print(f"  {k}: {v}")
            elif isinstance(v, (float, int)):
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
