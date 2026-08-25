#!/usr/bin/env python
"""
201b: same Transformer AR Student-t model as 201a, but with a narrower objective fix.

Changes vs 201a:
  - underfit-aware tail weighting instead of raw realized-tail weighting
  - self-fed rollout likelihood instead of rollout parameter-consistency
  - easy-case overwidth penalty
  - fixed ex-ante selection score aligned with the training objective
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    make_serializable,
    iv_to_unconstrained,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TransformerRolloutTailStudentTARModel,
    evaluate_rollout_subset,
    maybe_load_decoder_warm_start,
    reshape_history,
)


def compute_step_severity(
    prev_01: torch.Tensor,
    target_t: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
) -> torch.Tensor:
    delta_abs = (target_t - prev_01).abs()
    q95_frac = (delta_abs >= q95_threshold).float().mean(dim=-1)
    mean_excess = F.relu(delta_abs.mean(dim=-1) / max(q95_threshold, 1e-6) - 1.0)
    max_excess = F.relu(delta_abs.max(dim=-1).values / max(q99_threshold, 1e-6) - 1.0)
    return q95_frac + 0.5 * mean_excess + 0.5 * max_excess


def compute_underfit_weight_and_overwidth(
    model: TransformerRolloutTailStudentTARModel,
    prev_01: torch.Tensor,
    target_t: torch.Tensor,
    target_u: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    objective_config: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    severity = compute_step_severity(
        prev_01=prev_01,
        target_t=target_t,
        q95_threshold=objective_config["q95_threshold"],
        q99_threshold=objective_config["q99_threshold"],
    )

    marginal_std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-6))
    z = (target_u - mu).abs() / marginal_std
    z_gate = objective_config["underfit_z_gate"]
    underfit_margin = F.relu(z - z_gate) / z_gate
    underfit = underfit_margin.mean(dim=-1) + 0.5 * underfit_margin.max(dim=-1).values

    severity_det = severity.detach()
    underfit_det = underfit.detach()
    weight = 1.0 + objective_config["tail_weight"] * severity_det * underfit_det

    width_proxy = marginal_std.mean(dim=-1)
    easy_factor = torch.exp(-severity_det) * torch.exp(-underfit_det)
    overwidth_pen = (easy_factor * width_proxy).mean()

    aux = {
        "severity_mean": severity_det.mean(),
        "underfit_mean": underfit_det.mean(),
        "step_weight_mean": weight.mean(),
        "width_proxy_mean": width_proxy.mean(),
        "overwidth_penalty": overwidth_pen,
    }
    return weight, overwidth_pen, aux


def teacher_forced_multistep_objective(
    model: TransformerRolloutTailStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_flat = reshape_history(history_01)
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    future_len = future_flat.shape[1]
    context = history_flat

    total_loss = 0.0
    total_nll = 0.0
    total_mae = 0.0
    total_overwidth = 0.0
    total_severity = 0.0
    total_underfit = 0.0
    total_step_weight = 0.0
    total_scale = 0.0
    total_shape_var = 0.0
    total_nu = 0.0
    total_attn_top1 = 0.0

    for step in range(future_len):
        mu, factor, diag, scale, nu, attn = model.forward_from_history(context, return_attention=True)
        cov = model.covariance(factor, diag, scale)

        prev_01 = context[:, -1]
        target_t = future_flat[:, step]
        target_u = iv_to_unconstrained(target_t, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
        nll_t = model.student_t_nll(target_u, mu, factor, diag, scale, nu)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        step_weight, overwidth_pen, aux = compute_underfit_weight_and_overwidth(
            model=model,
            prev_01=prev_01,
            target_t=target_t,
            target_u=target_u,
            mu=mu,
            cov=cov,
            objective_config=objective_config,
        )
        weighted_nll = (step_weight * nll_t).mean()
        weighted_mae = (step_weight * (mu_iv - target_t).abs().mean(dim=-1)).mean()

        factor_norm, diag_norm, avg_var = model.normalized_components(factor, diag)
        step_loss = (
            weighted_nll
            + objective_config["tail_mae_weight"] * weighted_mae
            + objective_config["overwidth_weight"] * overwidth_pen
        )
        total_loss = total_loss + step_loss
        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_overwidth = total_overwidth + overwidth_pen
        total_severity = total_severity + aux["severity_mean"]
        total_underfit = total_underfit + aux["underfit_mean"]
        total_step_weight = total_step_weight + aux["step_weight_mean"]
        total_scale = total_scale + scale.mean()
        total_shape_var = total_shape_var + avg_var.mean()
        total_nu = total_nu + nu.mean()
        total_attn_top1 = total_attn_top1 + attn.max(dim=1).values.mean()

        context = torch.cat([context[:, 1:], target_t.unsqueeze(1)], dim=1)

    scale_fac = 1.0 / future_len
    metrics = {
        "teacher_total_loss": total_loss * scale_fac,
        "multistep_nll": total_nll * scale_fac,
        "multistep_mae": total_mae * scale_fac,
        "overwidth_penalty": total_overwidth * scale_fac,
        "severity_mean": total_severity * scale_fac,
        "underfit_mean": total_underfit * scale_fac,
        "step_weight_mean": total_step_weight * scale_fac,
        "scale_mean": total_scale * scale_fac,
        "shape_avg_var": total_shape_var * scale_fac,
        "nu_mean": total_nu * scale_fac,
        "attention_top1": total_attn_top1 * scale_fac,
    }
    return total_loss * scale_fac, metrics


def selffed_rollout_likelihood_objective(
    model: TransformerRolloutTailStudentTARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    objective_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_flat = reshape_history(history_01)
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    rollout_steps = int(min(objective_config["rollout_steps"], future_flat.shape[1]))

    ro_context = history_flat
    total_loss = 0.0
    total_nll = 0.0
    total_mae = 0.0
    total_overwidth = 0.0
    total_severity = 0.0
    total_underfit = 0.0
    total_step_weight = 0.0

    for step in range(rollout_steps):
        mu, factor, diag, scale, nu = model.forward_from_history(ro_context)
        cov = model.covariance(factor, diag, scale)

        target_t = future_flat[:, step]
        target_u = iv_to_unconstrained(target_t, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
        prev_01 = ro_context[:, -1]
        nll_t = model.student_t_nll(target_u, mu, factor, diag, scale, nu)
        mu_iv = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)

        step_weight, overwidth_pen, aux = compute_underfit_weight_and_overwidth(
            model=model,
            prev_01=prev_01,
            target_t=target_t,
            target_u=target_u,
            mu=mu,
            cov=cov,
            objective_config=objective_config,
        )
        weighted_nll = (step_weight * nll_t).mean()
        weighted_mae = (step_weight * (mu_iv - target_t).abs().mean(dim=-1)).mean()
        step_loss = (
            weighted_nll
            + objective_config["tail_mae_weight"] * weighted_mae
            + objective_config["overwidth_weight"] * overwidth_pen
        )

        total_loss = total_loss + step_loss
        total_nll = total_nll + nll_t.mean()
        total_mae = total_mae + (mu_iv - target_t).abs().mean()
        total_overwidth = total_overwidth + overwidth_pen
        total_severity = total_severity + aux["severity_mean"]
        total_underfit = total_underfit + aux["underfit_mean"]
        total_step_weight = total_step_weight + aux["step_weight_mean"]

        sample_u = model.reparameterized_next_u(mu, factor, diag, scale, nu)
        sample_iv = unconstrained_to_iv(sample_u, lo=model.support_lo, hi=model.support_hi)
        ro_context = torch.cat([ro_context[:, 1:], sample_iv.unsqueeze(1)], dim=1)

    scale_fac = 1.0 / max(rollout_steps, 1)
    metrics = {
        "rollout_total_loss": total_loss * scale_fac,
        "rollout_nll": total_nll * scale_fac,
        "rollout_mae": total_mae * scale_fac,
        "rollout_overwidth_penalty": total_overwidth * scale_fac,
        "rollout_severity_mean": total_severity * scale_fac,
        "rollout_underfit_mean": total_underfit * scale_fac,
        "rollout_step_weight_mean": total_step_weight * scale_fac,
    }
    return total_loss * scale_fac, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: TransformerRolloutTailStudentTARModel,
    val_loader: DataLoader,
    objective_config: dict[str, float],
) -> dict[str, float]:
    model.eval()
    total = {
        "val_teacher_total_loss": 0.0,
        "val_multistep_nll": 0.0,
        "val_multistep_mae": 0.0,
        "val_overwidth_penalty": 0.0,
        "val_severity_mean": 0.0,
        "val_underfit_mean": 0.0,
        "val_step_weight_mean": 0.0,
        "val_scale_mean": 0.0,
        "val_shape_avg_var": 0.0,
        "val_nu_mean": 0.0,
        "val_attention_top1": 0.0,
    }
    total_count = 0
    for history_01, future_01 in val_loader:
        loss, metrics = teacher_forced_multistep_objective(model, history_01, future_01, objective_config)
        bs = history_01.shape[0]
        total["val_teacher_total_loss"] += loss.item() * bs
        total["val_multistep_nll"] += metrics["multistep_nll"].item() * bs
        total["val_multistep_mae"] += metrics["multistep_mae"].item() * bs
        total["val_overwidth_penalty"] += metrics["overwidth_penalty"].item() * bs
        total["val_severity_mean"] += metrics["severity_mean"].item() * bs
        total["val_underfit_mean"] += metrics["underfit_mean"].item() * bs
        total["val_step_weight_mean"] += metrics["step_weight_mean"].item() * bs
        total["val_scale_mean"] += metrics["scale_mean"].item() * bs
        total["val_shape_avg_var"] += metrics["shape_avg_var"].item() * bs
        total["val_nu_mean"] += metrics["nu_mean"].item() * bs
        total["val_attention_top1"] += metrics["attention_top1"].item() * bs
        total_count += bs
    return {k: v / max(total_count, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate_selffed_rollout(
    model: TransformerRolloutTailStudentTARModel,
    val_loader: DataLoader,
    objective_config: dict[str, float],
) -> dict[str, float]:
    model.eval()
    total = {
        "val_rollout_total_loss": 0.0,
        "val_rollout_nll": 0.0,
        "val_rollout_mae": 0.0,
        "val_rollout_overwidth_penalty": 0.0,
        "val_rollout_severity_mean": 0.0,
        "val_rollout_underfit_mean": 0.0,
        "val_rollout_step_weight_mean": 0.0,
    }
    total_count = 0
    for history_01, future_01 in val_loader:
        loss, metrics = selffed_rollout_likelihood_objective(model, history_01, future_01, objective_config)
        bs = history_01.shape[0]
        total["val_rollout_total_loss"] += loss.item() * bs
        total["val_rollout_nll"] += metrics["rollout_nll"].item() * bs
        total["val_rollout_mae"] += metrics["rollout_mae"].item() * bs
        total["val_rollout_overwidth_penalty"] += metrics["rollout_overwidth_penalty"].item() * bs
        total["val_rollout_severity_mean"] += metrics["rollout_severity_mean"].item() * bs
        total["val_rollout_underfit_mean"] += metrics["rollout_underfit_mean"].item() * bs
        total["val_rollout_step_weight_mean"] += metrics["rollout_step_weight_mean"].item() * bs
        total_count += bs
    return {k: v / max(total_count, 1) for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="201b underfit-aware self-fed rollout Student-t")
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--tail_weight", type=float, default=6.0)
    parser.add_argument("--tail_mae_weight", type=float, default=0.05)
    parser.add_argument("--underfit_z_gate", type=float, default=1.8)
    parser.add_argument("--overwidth_weight", type=float, default=0.02)
    parser.add_argument("--rollout_weight", type=float, default=0.25)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--rollout_warmup_epochs", type=int, default=3)
    parser.add_argument("--rollout_ramp_epochs", type=int, default=3)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--decoder_warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
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
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    hist_len = args.history_len
    future_len = args.future_len
    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
    )

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
    )
    objective_config = dict(
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        tail_weight=args.tail_weight,
        tail_mae_weight=args.tail_mae_weight,
        underfit_z_gate=args.underfit_z_gate,
        overwidth_weight=args.overwidth_weight,
        rollout_steps=args.rollout_steps,
    )

    model = TransformerRolloutTailStudentTARModel(
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
    print("201b: Transformer AR Student-t with underfit-aware self-fed rollout training")
    print(f"{'=' * 76}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Tail thresholds: q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Underfit z-gate: {args.underfit_z_gate:.3f}")
    print(f"  Overwidth weight: {args.overwidth_weight}")
    print(f"  Self-fed rollout steps: {args.rollout_steps}, weight={args.rollout_weight}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": args.lr_encoder,
                "weight_decay": args.weight_decay_encoder,
            },
            {
                "params": model.decoder.parameters(),
                "lr": args.lr_decoder,
                "weight_decay": args.weight_decay_decoder,
            },
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
            "teacher_total_loss": 0.0,
            "multistep_nll": 0.0,
            "multistep_mae": 0.0,
            "overwidth_penalty": 0.0,
            "severity_mean": 0.0,
            "underfit_mean": 0.0,
            "step_weight_mean": 0.0,
            "scale_mean": 0.0,
            "shape_avg_var": 0.0,
            "nu_mean": 0.0,
            "attention_top1": 0.0,
            "rollout_total_loss": 0.0,
            "rollout_nll": 0.0,
            "rollout_mae": 0.0,
            "rollout_overwidth_penalty": 0.0,
            "rollout_severity_mean": 0.0,
            "rollout_underfit_mean": 0.0,
            "rollout_step_weight_mean": 0.0,
        }
        nb = 0

        if epoch <= args.rollout_warmup_epochs:
            rollout_scale = 0.0
        else:
            progress = (epoch - args.rollout_warmup_epochs) / max(args.rollout_ramp_epochs, 1)
            rollout_scale = float(min(max(progress, 0.0), 1.0))

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            teacher_loss, teacher_metrics = teacher_forced_multistep_objective(model, history_01, future_01, objective_config)
            loss = teacher_loss
            rollout_metrics = {
                "rollout_total_loss": torch.tensor(0.0, device=history_01.device),
                "rollout_nll": torch.tensor(0.0, device=history_01.device),
                "rollout_mae": torch.tensor(0.0, device=history_01.device),
                "rollout_overwidth_penalty": torch.tensor(0.0, device=history_01.device),
                "rollout_severity_mean": torch.tensor(0.0, device=history_01.device),
                "rollout_underfit_mean": torch.tensor(0.0, device=history_01.device),
                "rollout_step_weight_mean": torch.tensor(0.0, device=history_01.device),
            }
            if rollout_scale > 0.0 and args.rollout_weight > 0.0 and args.rollout_steps > 0:
                rollout_loss, rollout_metrics = selffed_rollout_likelihood_objective(model, history_01, future_01, objective_config)
                loss = loss + (args.rollout_weight * rollout_scale) * rollout_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in ep:
                if k in teacher_metrics:
                    ep[k] += teacher_metrics[k].item()
                elif k in rollout_metrics:
                    ep[k] += rollout_metrics[k].item()
            nb += 1

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep.items()}
        train_metrics["train_rollout_scale"] = rollout_scale

        val_teacher = evaluate_teacher_forced(model, val_loader, objective_config)
        val_rollout = evaluate_selffed_rollout(model, val_loader, objective_config)
        rollout_diag = evaluate_rollout_subset(
            model,
            val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )

        selection_score = val_teacher["val_teacher_total_loss"] + args.rollout_weight * val_rollout["val_rollout_total_loss"]

        elapsed = time.time() - t0
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score
            best_metrics = {**val_teacher, **val_rollout, **rollout_diag, "selection_score": selection_score}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": {
                        "type": "transformer_underfit_aware_selffed_rollout_student_t_201b",
                        "encoder": encoder_config,
                        "decoder": decoder_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "objective": objective_config,
                    },
                    "metrics": best_metrics,
                },
                Path(args.output_dir) / "best_model.pt",
            )

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_teacher,
            **val_rollout,
            **rollout_diag,
            "selection_score": selection_score,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        with open(Path(args.output_dir) / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"Ep {epoch:>3d}  "
            f"train_tf={train_metrics['train_teacher_total_loss']:.4f}  "
            f"train_ro={train_metrics['train_rollout_total_loss']:.4f}  "
            f"val_tf={val_teacher['val_teacher_total_loss']:.4f}  "
            f"val_ro={val_rollout['val_rollout_total_loss']:.4f}  "
            f"roll_cov90={rollout_diag['rollout_cov90']:.4f}  "
            f"roll_mae={rollout_diag['rollout_mae']:.4f}  "
            f"roll_tc={rollout_diag['rollout_turb_calm_ratio']:.3f}  "
            f"rank={rollout_diag['rollout_rank_ratio_h30']:.2f}  "
            f"attn={val_teacher['val_attention_top1']:.3f}  "
            f"rs={rollout_scale:.2f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": {
                "type": "transformer_underfit_aware_selffed_rollout_student_t_201b",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
                "history_len": hist_len,
                "future_len": future_len,
                "objective": objective_config,
            },
            "metrics": row,
        },
        Path(args.output_dir) / "final_model.pt",
    )

    print(f"\nBest selection score: {best_score:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"val_rollout_total_loss={best_metrics['val_rollout_total_loss']:.4f}, "
            f"rollout_cov90={best_metrics['rollout_cov90']:.4f}, "
            f"rollout_width90={best_metrics['rollout_width90']:.4f}, "
            f"rollout_turb_calm_ratio={best_metrics['rollout_turb_calm_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
