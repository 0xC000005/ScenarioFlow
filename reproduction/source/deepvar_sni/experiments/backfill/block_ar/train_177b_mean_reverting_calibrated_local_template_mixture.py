#!/usr/bin/env python
"""
177b: Mean-reverting local-template mixture with a tiny shared local width correction.

Motivation:
  - 177a fixes mean reversion and preserves the broader H7 realism story
  - 177a still fails S2 due to mild, stable best-cell overcoverage
  - the next move should preserve 177a's drift law and only add a narrowly
    targeted uncertainty-reallocation mechanism
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

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    make_serializable,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
    effective_rank,
)
from experiments.backfill.block_ar.train_176b_shared_local_template_mixture import (
    evaluate_joint_subset as _unused_evaluate_joint_subset,
)
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import (
    MeanRevertingSharedLocalTemplateMixtureDecoder,
    MeanRevertingSharedLocalTemplateMixtureStudentTModel,
    evaluate_joint_subset,
)


class MeanRevertingCalibratedLocalTemplateMixtureDecoder(MeanRevertingSharedLocalTemplateMixtureDecoder):
    """177a decoder plus a tiny shared local log-variance correction map."""

    def __init__(
        self,
        *args,
        shared_calibration_clip: float = 0.08,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shared_calibration_clip = shared_calibration_clip
        self.shared_calibration_template = torch.nn.Parameter(torch.zeros(self.n_frames, self.n_cells))
        torch.nn.init.normal_(self.shared_calibration_template, mean=0.0, std=2e-3)

    def forward(self, cond: torch.Tensor, prev_u: torch.Tensor, hist_mean_u: torch.Tensor):
        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            local_delta_components,
            gate_logits,
        ) = super().forward(cond, prev_u, hist_mean_u)

        shared_cal = torch.tanh(self.shared_calibration_template) * self.shared_calibration_clip
        shared_cal = shared_cal - shared_cal.mean()
        local_delta_components = local_delta_components + shared_cal.unsqueeze(0).unsqueeze(0)
        local_delta_components = torch.tanh(local_delta_components) * self.local_delta_clip
        local_delta_components = local_delta_components - local_delta_components.mean(
            dim=(2, 3), keepdim=True
        )
        return (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            local_delta_components,
            gate_logits,
        )


class MeanRevertingCalibratedLocalTemplateMixtureStudentTModel(
    MeanRevertingSharedLocalTemplateMixtureStudentTModel
):
    """177a backbone with a tiny shared local uncertainty reallocation map."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: dict,
        flow_config: dict,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-5,
        base_nu: float = 8.0,
    ):
        base_decoder_config = dict(decoder_config)
        base_decoder_config.pop("shared_calibration_clip", None)
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=base_decoder_config,
            flow_config=flow_config,
            support_lo=support_lo,
            support_hi=support_hi,
            support_eps=support_eps,
            cov_jitter=cov_jitter,
            base_nu=base_nu,
        )
        self.decoder = MeanRevertingCalibratedLocalTemplateMixtureDecoder(**decoder_config)
        self.decoder_config = decoder_config


def joint_nll_loss(
    model: MeanRevertingCalibratedLocalTemplateMixtureStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    shared_calibration_penalty: float,
):
    from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained, unconstrained_to_iv

    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        base_local_delta,
        local_delta_components,
        gate_logits,
    ) = model.forward_from_history(history_01)
    logprob, posterior, aux = model.mixture_log_prob(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        flow_context=flow_context,
        local_delta_components=local_delta_components,
        gate_logits=gate_logits,
    )

    pred_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi)
    prior = F.softmax(gate_logits, dim=-1)
    usage = prior.mean(dim=0)
    target_usage = torch.full_like(usage, 1.0 / usage.numel())
    usage_penalty = (usage - target_usage).pow(2).mean()

    nll = (-logprob).mean()
    local_pen = base_local_delta.pow(2).mean()
    template_pen = model.decoder.local_template_bank.pow(2).mean()
    shared_pen = model.decoder.shared_calibration_template.pow(2).mean()
    loss = (
        nll
        + local_var_penalty * local_pen
        + template_penalty * template_pen
        + gate_balance_penalty * usage_penalty
        + shared_calibration_penalty * shared_pen
    )

    posterior_det = posterior.detach()
    weighted_local_delta_rms = (posterior_det * aux["local_delta_rms"]).sum(dim=1).mean()
    weighted_local_scale_min = (posterior_det * aux["local_scale_min"]).sum(dim=1).mean()
    weighted_local_scale_max = (posterior_det * aux["local_scale_max"]).sum(dim=1).mean()
    weighted_white_std = (posterior_det * aux["white_std"]).sum(dim=1).mean()
    weighted_z_std = (posterior_det * aux["z_std"]).sum(dim=1).mean()
    shared_rms = (
        torch.tanh(model.decoder.shared_calibration_template) * model.decoder.shared_calibration_clip
    ).pow(2).mean().sqrt()

    metrics = {
        "joint_nll": nll,
        "total_loss": loss,
        "local_var_penalty": local_pen,
        "template_penalty": template_pen,
        "gate_usage_penalty": usage_penalty,
        "shared_calibration_penalty": shared_pen,
        "joint_mae": (pred_01 - future_01).abs().mean(),
        "time_eff_rank": effective_rank(aux["cov_t"]).mean(),
        "cell_eff_rank": effective_rank(aux["cov_c"]).mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": (posterior_det * aux["flow_logdet"]).sum(dim=1).mean(),
        "white_std_mean": weighted_white_std,
        "z_std_mean": weighted_z_std,
        "local_delta_rms": weighted_local_delta_rms,
        "local_scale_min": weighted_local_scale_min,
        "local_scale_max": weighted_local_scale_max,
        "prior_entropy": (-(prior * torch.log(prior.clamp_min(1e-8))).sum(dim=-1)).mean(),
        "posterior_entropy": (-(posterior_det * torch.log(posterior_det.clamp_min(1e-8))).sum(dim=-1)).mean(),
        "gate_max_prob": prior.max(dim=-1).values.mean(),
        "shared_calibration_rms": shared_rms,
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: MeanRevertingCalibratedLocalTemplateMixtureStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    template_penalty: float,
    gate_balance_penalty: float,
    shared_calibration_penalty: float,
) -> dict:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_joint_nll": 0.0,
        "val_local_var_penalty": 0.0,
        "val_template_penalty": 0.0,
        "val_gate_usage_penalty": 0.0,
        "val_shared_calibration_penalty": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_flow_logdet_mean": 0.0,
        "val_white_std_mean": 0.0,
        "val_z_std_mean": 0.0,
        "val_local_delta_rms": 0.0,
        "val_local_scale_min": 0.0,
        "val_local_scale_max": 0.0,
        "val_prior_entropy": 0.0,
        "val_posterior_entropy": 0.0,
        "val_gate_max_prob": 0.0,
        "val_shared_calibration_rms": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        _loss, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            template_penalty=template_penalty,
            gate_balance_penalty=gate_balance_penalty,
            shared_calibration_penalty=shared_calibration_penalty,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            metric_key = key.replace("val_", "")
            totals[key] += metrics[metric_key].item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="177b: mean-reverting calibrated local-template mixture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--template_penalty", type=float, default=0.5)
    parser.add_argument("--gate_balance_penalty", type=float, default=5.0)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--shared_calibration_clip", type=float, default=0.08)
    parser.add_argument("--shared_calibration_penalty", type=float, default=2.0)
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

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
        n_components=args.n_components,
        drift_strength_max=args.drift_strength_max,
        equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
        shared_calibration_clip=args.shared_calibration_clip,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = MeanRevertingCalibratedLocalTemplateMixtureStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {
            "train_total_loss": 0.0,
            "train_joint_nll": 0.0,
            "train_local_var_penalty": 0.0,
            "train_template_penalty": 0.0,
            "train_gate_usage_penalty": 0.0,
            "train_shared_calibration_penalty": 0.0,
            "train_joint_mae": 0.0,
            "train_time_eff_rank": 0.0,
            "train_cell_eff_rank": 0.0,
            "train_scale_mean": 0.0,
            "train_flow_logdet_mean": 0.0,
            "train_white_std_mean": 0.0,
            "train_z_std_mean": 0.0,
            "train_local_delta_rms": 0.0,
            "train_local_scale_min": 0.0,
            "train_local_scale_max": 0.0,
            "train_prior_entropy": 0.0,
            "train_posterior_entropy": 0.0,
            "train_gate_max_prob": 0.0,
            "train_shared_calibration_rms": 0.0,
        }
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
                template_penalty=args.template_penalty,
                gate_balance_penalty=args.gate_balance_penalty,
                shared_calibration_penalty=args.shared_calibration_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                metric_key = key.replace("train_", "")
                totals[key] += metrics[metric_key].item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
            template_penalty=args.template_penalty,
            gate_balance_penalty=args.gate_balance_penalty,
            shared_calibration_penalty=args.shared_calibration_penalty,
        )
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_total_loss"] < best_val
        if is_best:
            best_val = val_metrics["val_total_loss"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": best_val,
                    "config": {
                        "type": "mean_reverting_calibrated_local_template_mixture_residual_flow_structured_joint_student_t_177b",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "local_var_penalty": args.local_var_penalty,
                        "template_penalty": args.template_penalty,
                        "gate_balance_penalty": args.gate_balance_penalty,
                        "shared_calibration_penalty": args.shared_calibration_penalty,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"mr={joint_metrics['joint_sample_mr_ratio']:.3f}  "
            f"shared_rms={val_metrics['val_shared_calibration_rms']:.3f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "mean_reverting_calibrated_local_template_mixture_residual_flow_structured_joint_student_t_177b",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "local_var_penalty": args.local_var_penalty,
            "template_penalty": args.template_penalty,
            "gate_balance_penalty": args.gate_balance_penalty,
            "shared_calibration_penalty": args.shared_calibration_penalty,
        },
        "best_val_total_loss": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    print(f"\nBest val total loss: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_sample_mr_ratio={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
