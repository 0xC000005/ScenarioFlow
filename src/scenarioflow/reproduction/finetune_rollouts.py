"""Fine-tune ScenarioFlow with the frozen free-rollout Energy objective."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..model import (
    enable_adaptive_graph_velocity_mixer,
    enable_conditional_base_noise_scale,
    enable_group_head_velocity_readout,
    enable_group_residual_velocity_readout,
    load_model,
    save_checkpoint,
)
from .losses import (
    effective_readout_iv_count,
    normalized_rollout_energy_loss,
    state_tail_sampling_weights,
)
from .panel import prepare_panel_blocks, select_normalized_innovation_scope
from .train_next_step import _selected_parameter_hash

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def stage2_optimizer_update_hash(
    checkpoint: str | Path,
    tensors: tuple[torch.Tensor, ...],
    *,
    selected_parameters: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Run the frozen single Stage 2 AdamW update and hash selected weights."""
    history_level, history_norm, future_level, future_norm, center, scale, drift = (
        tensors
    )
    model, _payload = load_model(checkpoint, torch.device("cpu"))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    torch.manual_seed(int(seed))
    optimizer.zero_grad(set_to_none=True)
    loss, _metrics = normalized_rollout_energy_loss(
        model,
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift,
        train_sample_count=2,
        rollout_flow_steps=4,
        energy_weight=0.2,
        marginal_crps_weight=0.0,
        variogram_weight=0.0,
        variogram_power=0.5,
        fm_anchor_weight=1.0,
        horizon_end_weight=1.2,
        energy_eps=1e-6,
        temperature=1.0,
        level_energy_weight=0.0,
        channel_level_energy_weight=0.05,
        channel_level_energy_coordinate="level",
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return _selected_parameter_hash(model, selected_parameters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/cache/models/core/stage1/best_model.pt",
    )
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument(
        "--state_scope",
        choices=["iv_only", "joint38", "anchor_only"],
        default="joint38",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform",
        choices=["log_level", "bounded_logit"],
        default="bounded_logit",
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument(
        "--drift_feature_mode", choices=["none", "ewma_mean"], default="none"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=2)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_weight", type=float, default=0.2)
    parser.add_argument("--marginal_crps_weight", type=float, default=0.0)
    parser.add_argument("--interval_score_weight", type=float, default=0.0)
    parser.add_argument("--interval_alpha", type=float, default=0.1)
    parser.add_argument("--variogram_weight", type=float, default=0.0)
    parser.add_argument("--variogram_power", type=float, default=0.5)
    parser.add_argument("--level_marginal_crps_weight", type=float, default=0.0)
    parser.add_argument(
        "--level_marginal_crps_coordinate",
        choices=["level", "scaled_delta"],
        default="scaled_delta",
    )
    parser.add_argument("--level_energy_weight", type=float, default=0.0)
    parser.add_argument("--channel_level_energy_weight", type=float, default=0.05)
    parser.add_argument(
        "--channel_level_energy_coordinate",
        choices=["level", "scaled_delta"],
        default="level",
    )
    parser.add_argument("--condition_rollout_contrast_weight", type=float, default=0.0)
    parser.add_argument("--condition_rollout_contrast_margin", type=float, default=0.0)
    parser.add_argument(
        "--condition_rollout_negative_mode",
        choices=["roll", "nearest_history"],
        default="roll",
    )
    parser.add_argument("--risk_state_weight", type=float, default=0.0)
    parser.add_argument("--risk_state_rank_weight", type=float, default=0.0)
    parser.add_argument("--mixed_support_loss_weight", type=float, default=0.0)
    parser.add_argument("--dispersion_calibration_weight", type=float, default=0.0)
    parser.add_argument(
        "--dispersion_calibration_mode",
        choices=["window", "channel", "window_channel"],
        default="window",
    )
    parser.add_argument("--free_running_fm_weight", type=float, default=0.0)
    parser.add_argument("--free_running_fm_prefix_steps", type=int, default=0)
    parser.add_argument("--state_tail_sampler_weight", type=float, default=0.0)
    parser.add_argument("--state_tail_sampler_quantile", type=float, default=0.8)
    parser.add_argument(
        "--velocity_readout_mode",
        choices=["shared", "group_residual", "group_head"],
        default="shared",
    )
    parser.add_argument(
        "--velocity_mixer_mode",
        choices=["checkpoint", "adaptive_graph_residual"],
        default="checkpoint",
    )
    parser.add_argument("--adaptive_graph_k", type=int, default=8)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=1.2)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--base_noise_rho", type=float, default=None)
    parser.add_argument("--conditional_base_noise_scale", action="store_true")
    parser.add_argument("--base_noise_scale_min", type=float, default=0.5)
    parser.add_argument("--base_noise_scale_max", type=float, default=2.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=REPOSITORY_ROOT / "build/training/stage2",
    )
    return parser


def _finetune_objective(
    args: argparse.Namespace,
    model: Any,
    train_state_tail_weights: np.ndarray,
) -> dict[str, Any]:
    return {
        "base": "flow_matching_mse",
        "rollout_energy_coordinate": "normalized_innovation",
        "train_sample_count": int(args.train_sample_count),
        "rollout_flow_steps": int(args.rollout_flow_steps),
        "energy_weight": float(args.energy_weight),
        "marginal_crps_weight": float(args.marginal_crps_weight),
        "interval_score_weight": float(args.interval_score_weight),
        "interval_alpha": float(args.interval_alpha),
        "variogram_weight": float(args.variogram_weight),
        "variogram_power": float(args.variogram_power),
        "level_marginal_crps_weight": float(args.level_marginal_crps_weight),
        "level_marginal_crps_coordinate": args.level_marginal_crps_coordinate,
        "level_energy_weight": float(args.level_energy_weight),
        "channel_level_energy_weight": float(args.channel_level_energy_weight),
        "channel_level_energy_coordinate": args.channel_level_energy_coordinate,
        "condition_rollout_contrast_weight": float(
            args.condition_rollout_contrast_weight
        ),
        "condition_rollout_contrast_margin": float(
            args.condition_rollout_contrast_margin
        ),
        "condition_rollout_negative_mode": args.condition_rollout_negative_mode,
        "risk_state_weight": float(args.risk_state_weight),
        "risk_state_rank_weight": float(args.risk_state_rank_weight),
        "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        "mixed_support_observation": getattr(
            model.cfg, "mixed_support_observation", "none"
        ),
        "dispersion_calibration_weight": float(args.dispersion_calibration_weight),
        "dispersion_calibration_mode": args.dispersion_calibration_mode,
        "free_running_fm_weight": float(args.free_running_fm_weight),
        "free_running_fm_prefix_steps": int(args.free_running_fm_prefix_steps),
        "state_tail_sampler_weight": float(args.state_tail_sampler_weight),
        "state_tail_sampler_quantile": float(args.state_tail_sampler_quantile),
        "state_tail_sampler_mean": float(train_state_tail_weights.mean()),
        "state_tail_sampler_max": float(train_state_tail_weights.max()),
        "base_noise_rho": float(model.cfg.base_noise_rho),
        "conditional_base_noise_scale": bool(model.cfg.conditional_base_noise_scale),
        "base_noise_scale_min": float(model.cfg.base_noise_scale_min),
        "base_noise_scale_max": float(model.cfg.base_noise_scale_max),
        "velocity_readout_mode": args.velocity_readout_mode,
        "velocity_mixer_mode": getattr(model.cfg, "velocity_mixer_mode", "transformer"),
        "adaptive_graph_k": int(
            getattr(model.cfg, "adaptive_graph_k", args.adaptive_graph_k)
        ),
        "fm_anchor_weight": float(args.fm_anchor_weight),
        "horizon_end_weight": float(args.horizon_end_weight),
    }


def _loss(
    model,
    batch,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return normalized_rollout_energy_loss(
        model,
        *batch[:6],
        batch[6],
        train_sample_count=args.train_sample_count,
        rollout_flow_steps=args.rollout_flow_steps,
        energy_weight=args.energy_weight,
        marginal_crps_weight=args.marginal_crps_weight,
        interval_score_weight=args.interval_score_weight,
        interval_alpha=args.interval_alpha,
        variogram_weight=args.variogram_weight,
        variogram_power=args.variogram_power,
        level_marginal_crps_weight=args.level_marginal_crps_weight,
        level_marginal_crps_coordinate=args.level_marginal_crps_coordinate,
        level_energy_weight=args.level_energy_weight,
        channel_level_energy_weight=args.channel_level_energy_weight,
        channel_level_energy_coordinate=args.channel_level_energy_coordinate,
        condition_rollout_contrast_weight=args.condition_rollout_contrast_weight,
        condition_rollout_contrast_margin=args.condition_rollout_contrast_margin,
        condition_rollout_negative_mode=args.condition_rollout_negative_mode,
        risk_state_weight=args.risk_state_weight,
        risk_state_rank_weight=args.risk_state_rank_weight,
        mixed_support_weight=args.mixed_support_loss_weight,
        dispersion_calibration_weight=args.dispersion_calibration_weight,
        dispersion_calibration_mode=args.dispersion_calibration_mode,
        free_running_fm_weight=args.free_running_fm_weight,
        free_running_fm_prefix_steps=args.free_running_fm_prefix_steps,
        fm_anchor_weight=args.fm_anchor_weight,
        horizon_end_weight=args.horizon_end_weight,
        energy_eps=args.energy_eps,
        temperature=args.temperature,
    )


def _run_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    args: argparse.Namespace,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    count = 0
    for batch_index, values in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = tuple(value.to(device) for value in values)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = _loss(model, batch, args)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        count += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())
    return {key: value / max(count, 1) for key, value in totals.items()}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(args.checkpoint, device)
    if args.velocity_mixer_mode == "adaptive_graph_residual":
        enable_adaptive_graph_velocity_mixer(model, k_neighbors=args.adaptive_graph_k)
    if args.base_noise_rho is not None:
        model.cfg.base_noise_rho = float(args.base_noise_rho)
    if bool(args.conditional_base_noise_scale):
        enable_conditional_base_noise_scale(
            model,
            scale_min=float(args.base_noise_scale_min),
            scale_max=float(args.base_noise_scale_max),
        )
    if args.velocity_readout_mode in {"group_residual", "group_head"}:
        readout_iv_count = effective_readout_iv_count(
            payload.get("state_scope", args.state_scope),
            n_cells=int(model.cfg.n_cells),
            iv_count=int(args.iv_count),
        )
        if args.velocity_readout_mode == "group_residual":
            enable_group_residual_velocity_readout(model, iv_count=readout_iv_count)
        else:
            enable_group_head_velocity_readout(model, iv_count=readout_iv_count)
    args.history_len = int(model.cfg.history_len)
    args.future_len = int(model.cfg.future_len)
    args.state_scope = payload.get("state_scope", args.state_scope)
    normalization = payload.get("normalization", {})
    args.iv_transform = normalization.get(
        "iv_transform", payload.get("iv_transform", args.iv_transform)
    )
    args.iv_lower_bound = float(
        normalization.get(
            "iv_lower_bound", payload.get("iv_lower_bound", args.iv_lower_bound)
        )
    )
    args.iv_upper_bound = float(
        normalization.get(
            "iv_upper_bound", payload.get("iv_upper_bound", args.iv_upper_bound)
        )
    )
    args.scale_floor = float(normalization.get("scale_floor", args.scale_floor))
    args.center_mode = normalization.get("center_mode", args.center_mode)
    args.drift_feature_mode = normalization.get(
        "drift_feature_mode", args.drift_feature_mode
    )
    half_life = normalization.get("scale_half_life", args.scale_half_life)
    scale_half_life = (
        None if half_life is None or float(half_life) <= 0.0 else float(half_life)
    )
    _columns, panel_metadata, train_block, val_block = prepare_panel_blocks(
        data_dir=args.data_dir,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        max_train_windows=args.max_train_windows,
        clean_nonpositive_log_levels=args.clean_nonpositive_log_levels,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=args.iv_lower_bound,
        iv_upper_bound=args.iv_upper_bound,
    )
    train = select_normalized_innovation_scope(
        train_block,
        args.state_scope,
        args.iv_count,
        scale_half_life=scale_half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    validation = select_normalized_innovation_scope(
        val_block,
        args.state_scope,
        args.iv_count,
        scale_half_life=scale_half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    train_sampler = None
    train_shuffle = True
    train_state_tail_weights = np.ones(int(train[0].shape[0]), dtype=np.float32)
    if float(args.state_tail_sampler_weight) > 0.0:
        train_state_tail_weights = state_tail_sampling_weights(
            train[0],
            tail_quantile=float(args.state_tail_sampler_quantile),
            strength=float(args.state_tail_sampler_weight),
        )
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(train_state_tail_weights, dtype=torch.double),
            num_samples=int(train_state_tail_weights.shape[0]),
            replacement=True,
        )
        train_shuffle = False
    train_loader = DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in train[:7])),
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in validation[:7])),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    objective = _finetune_objective(args, model, train_state_tail_weights)
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": payload.get(
            "model_coordinate", "state_aware_normalized_innovation"
        ),
        "normalization": normalization,
        "finetune_objective": objective,
        "iv_transform": args.iv_transform,
        "iv_lower_bound": args.iv_lower_bound,
        "iv_upper_bound": args.iv_upper_bound,
        "iv_count": args.iv_count,
        "state_specs": payload.get("state_specs", []),
        "panel_metadata": panel_metadata,
        "source_checkpoint": str(args.checkpoint),
    }
    best_val = float("inf")
    best_epoch = -1
    records = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            args=args,
            device=device,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                None,
                args=args,
                device=device,
                max_batches=args.max_val_batches,
            )
        record = {
            "epoch": epoch,
            "elapsed_s": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        records.append(record)
        val_total = val_metrics.get("total", float("inf"))
        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                model.cfg,
                epoch,
                best_val,
                extra=extra,
            )
        print(json.dumps(record), flush=True)
    save_checkpoint(
        args.output_dir / "final_model.pt",
        model,
        model.cfg,
        args.epochs,
        best_val,
        extra=extra,
    )
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "config": asdict(model.cfg),
                "best_epoch": best_epoch,
                "best_val_total": best_val,
                "history": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
