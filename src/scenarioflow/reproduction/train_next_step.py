"""Train the frozen ScenarioFlow next-step flow-matching stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..model import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    load_model,
    save_checkpoint,
)
from .panel import prepare_panel_blocks, select_normalized_innovation_scope

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _selected_parameter_hash(
    model: torch.nn.Module, selected_parameters: Sequence[str]
) -> np.ndarray:
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in selected_parameters:
        value = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def stage1_loss(
    checkpoint: str | Path,
    tensors: tuple[torch.Tensor, ...],
    *,
    seed: int,
) -> np.ndarray:
    """Evaluate the exact teacher-forced Stage 1 objective on one batch."""
    history_level, history_norm, future_level, future_norm, center, scale, drift = (
        tensors
    )
    model, _payload = load_model(checkpoint, torch.device("cpu"))
    model.train()
    torch.manual_seed(int(seed))
    loss, _metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift_feature=drift,
    )
    return np.asarray([loss.detach().item()], dtype=np.float32)


def stage1_optimizer_update_hash(
    checkpoint: str | Path,
    tensors: tuple[torch.Tensor, ...],
    *,
    selected_parameters: Sequence[str],
    seed: int,
    learning_rate: float,
    weight_decay: float,
    clip_grad_norm: float,
) -> np.ndarray:
    """Run the frozen single Stage 1 AdamW update and hash selected weights."""
    history_level, history_norm, future_level, future_norm, center, scale, drift = (
        tensors
    )
    model, _payload = load_model(checkpoint, torch.device("cpu"))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    torch.manual_seed(int(seed))
    optimizer.zero_grad(set_to_none=True)
    loss, _metrics = model.training_loss(
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift_feature=drift,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
    optimizer.step()
    return _selected_parameter_hash(model, selected_parameters)


def fit_empirical_quantiles(
    history: np.ndarray,
    future: np.ndarray,
    *,
    n_quantiles: int,
    cdf_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate(
        [
            np.asarray(history, dtype=np.float64).reshape(-1, history.shape[-1]),
            np.asarray(future, dtype=np.float64).reshape(-1, future.shape[-1]),
        ],
        axis=0,
    )
    levels = np.linspace(
        float(cdf_eps),
        1.0 - float(cdf_eps),
        int(n_quantiles),
        dtype=np.float64,
    )
    quantiles = np.quantile(flat, levels, axis=0).T.astype(np.float32)
    return quantiles, levels.astype(np.float32)


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "source_column": spec.source_column,
        "source_index": int(spec.source_index),
        "transform": spec.transform,
        "reference_increment_column": spec.reference_increment_column,
        "reference_increment_index": (
            None
            if spec.reference_increment_index is None
            else int(spec.reference_increment_index)
        ),
        "lower_bound": getattr(spec, "lower_bound", None),
        "upper_bound": getattr(spec, "upper_bound", None),
    }


def _model_coordinate_from_innovation_coordinate(innovation_coordinate: str) -> str:
    if innovation_coordinate == "score":
        return "state_aware_normalized_innovation_score"
    if innovation_coordinate == "hybrid_sticky_score":
        return "state_aware_normalized_innovation_hybrid_score"
    if innovation_coordinate == "hybrid_tail_asinh":
        return "state_aware_normalized_innovation_tail_asinh"
    return "state_aware_normalized_innovation"


def _checkpoint_metadata(
    args: argparse.Namespace,
    cfg: GenericStateAwareNormalizedInnovationFMConfig,
    *,
    scale_half_life: float | None,
    sticky_score_report: dict[str, Any],
    tail_asinh_report: dict[str, Any],
    mixed_support_report: dict[str, Any],
    train_future_weight: np.ndarray,
    val_future_weight: np.ndarray,
    train_no_update_target: np.ndarray,
    val_no_update_target: np.ndarray,
    state_specs: list[dict[str, Any]],
    panel_metadata: dict[str, Any],
) -> dict[str, Any]:
    normalization = {
        "coordinate": "encoded_increment",
        "center_mode": args.center_mode,
        "drift_feature_mode": args.drift_feature_mode,
        "scale_method": "ewma_rms",
        "scale_half_life": scale_half_life,
        "scale_floor": float(args.scale_floor),
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "innovation_coordinate": args.innovation_coordinate,
        "sticky_score": sticky_score_report,
        "tail_asinh": tail_asinh_report,
        "sticky_observation_loss": args.sticky_observation_loss,
        "sticky_observation_zero_weight": float(args.sticky_observation_zero_weight),
        "mixed_support": mixed_support_report,
        "mixed_support_observation": args.mixed_support_observation,
        "mixed_support_zero_eps": float(args.mixed_support_zero_eps),
        "mixed_support_zero_rate_gate": float(args.mixed_support_zero_rate_gate),
        "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        "train_future_weight_mean": float(np.mean(train_future_weight)),
        "val_future_weight_mean": float(np.mean(val_future_weight)),
        "train_no_update_target_mean": float(np.mean(train_no_update_target)),
        "val_no_update_target_mean": float(np.mean(val_no_update_target)),
    }
    return {
        "state_scope": args.state_scope,
        "model_coordinate": _model_coordinate_from_innovation_coordinate(
            args.innovation_coordinate
        ),
        "normalization": normalization,
        "training_objective": {
            "base": "flow_matching_mse",
            "condition_contrast_weight": float(args.condition_contrast_weight),
            "condition_contrast_margin": float(args.condition_contrast_margin),
            "risk_state_dim": int(args.risk_state_dim),
            "risk_state_weight": float(args.risk_state_weight),
            "risk_state_rank_weight": float(args.risk_state_rank_weight),
            "base_noise_rho": float(cfg.base_noise_rho),
            "conditional_base_noise_scale": bool(cfg.conditional_base_noise_scale),
            "base_noise_scale_min": float(cfg.base_noise_scale_min),
            "base_noise_scale_max": float(cfg.base_noise_scale_max),
            "flow_coordinate": args.innovation_coordinate,
            "sticky_score": sticky_score_report,
            "tail_asinh": tail_asinh_report,
            "sticky_observation_loss": args.sticky_observation_loss,
            "sticky_observation_zero_weight": float(
                args.sticky_observation_zero_weight
            ),
            "mixed_support": mixed_support_report,
            "mixed_support_observation": args.mixed_support_observation,
            "mixed_support_loss_weight": float(args.mixed_support_loss_weight),
        },
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "iv_count": int(args.iv_count),
        "state_specs": state_specs,
        "panel_metadata": panel_metadata,
        "positive_level_policy": args.positive_level_policy,
        "sticky_score": sticky_score_report,
        "tail_asinh": tail_asinh_report,
        "mixed_support": mixed_support_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument(
        "--state_scope",
        choices=["iv_only", "anchor_only", "joint38"],
        default="joint38",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
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
    parser.add_argument(
        "--innovation_coordinate",
        choices=[
            "normalized",
            "score",
            "hybrid_sticky_score",
            "hybrid_tail_asinh",
        ],
        default="normalized",
    )
    parser.add_argument("--sticky_score_zero_eps", type=float, default=1e-10)
    parser.add_argument("--sticky_score_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--tail_asinh_scale_quantile", type=float, default=0.5)
    parser.add_argument("--tail_asinh_min_scale", type=float, default=0.5)
    parser.add_argument(
        "--sticky_observation_loss",
        choices=["none", "nonzero_mask"],
        default="none",
    )
    parser.add_argument("--sticky_observation_zero_weight", type=float, default=0.0)
    parser.add_argument(
        "--mixed_support_observation",
        choices=["none", "bernoulli_no_update"],
        default="none",
    )
    parser.add_argument("--mixed_support_zero_eps", type=float, default=1e-10)
    parser.add_argument("--mixed_support_zero_rate_gate", type=float, default=0.25)
    parser.add_argument("--mixed_support_loss_weight", type=float, default=1.0)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.05)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--base_noise_rho", type=float, default=0.0)
    parser.add_argument("--conditional_base_noise_scale", action="store_true")
    parser.add_argument("--base_noise_scale_min", type=float, default=0.5)
    parser.add_argument("--base_noise_scale_max", type=float, default=2.0)
    parser.add_argument("--prefix_feature_mode", default="scale")
    parser.add_argument("--condition_contrast_weight", type=float, default=0.0)
    parser.add_argument("--condition_contrast_margin", type=float, default=0.0)
    parser.add_argument("--risk_state_dim", type=int, default=0)
    parser.add_argument("--risk_state_weight", type=float, default=0.0)
    parser.add_argument("--risk_state_rank_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=REPOSITORY_ROOT / "build/training/stage1",
    )
    return parser


def _validation_loss(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss, _metrics = model.training_loss(
                *(value.to(device) for value in batch[:6]),
                drift_feature=batch[6].to(device),
            )
            total += float(loss.item()) * int(batch[0].shape[0])
            count += int(batch[0].shape[0])
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    half_life = (
        None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life)
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
        scale_half_life=half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    validation = select_normalized_innovation_scope(
        val_block,
        args.state_scope,
        args.iv_count,
        scale_half_life=half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    level_quantiles, quantile_levels = fit_empirical_quantiles(
        train[0], train[2], n_quantiles=args.n_quantiles, cdf_eps=args.cdf_eps
    )
    cfg = GenericStateAwareNormalizedInnovationFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=int(train[0].shape[-1]),
        memory_dim=args.memory_dim,
        memory_layers=args.memory_layers,
        memory_heads=args.memory_heads,
        memory_ff=args.memory_ff,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_ff=args.token_ff,
        time_dim=args.time_dim,
        model_dropout=args.model_dropout,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        base_noise_rho=args.base_noise_rho,
        conditional_base_noise_scale=args.conditional_base_noise_scale,
        base_noise_scale_min=args.base_noise_scale_min,
        base_noise_scale_max=args.base_noise_scale_max,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
        prefix_feature_mode=args.prefix_feature_mode,
        innovation_coordinate=args.innovation_coordinate,
        risk_state_dim=args.risk_state_dim,
        mixed_support_observation=args.mixed_support_observation,
        conditioning_mode="prefix",
    )
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg).to(device)
    model.set_level_quantiles(
        torch.from_numpy(level_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
    train_loader = DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in train[:7])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
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
    best_val = float("inf")
    best_epoch = -1
    records = []
    start_time = time.time()
    disabled_report: dict[str, Any] = {
        "policy": "disabled",
        "selected_names": [],
    }
    extra = _checkpoint_metadata(
        args,
        cfg,
        scale_half_life=half_life,
        sticky_score_report=dict(disabled_report),
        tail_asinh_report=dict(disabled_report),
        mixed_support_report=dict(disabled_report),
        train_future_weight=np.ones((1,), dtype=np.float32),
        val_future_weight=np.ones((1,), dtype=np.float32),
        train_no_update_target=np.zeros((1,), dtype=np.float32),
        val_no_update_target=np.zeros((1,), dtype=np.float32),
        state_specs=[_spec_to_dict(spec) for spec in train[8]],
        panel_metadata=panel_metadata,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = model.training_loss(
                *(value.to(device) for value in batch[:6]),
                drift_feature=batch[6].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * int(batch[0].shape[0])
            count += int(batch[0].shape[0])
        val_loss = _validation_loss(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "val_loss": val_loss,
            "elapsed_s": time.time() - start_time,
        }
        records.append(record)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                cfg,
                epoch,
                best_val,
                extra=extra,
            )
        print(json.dumps(record), flush=True)
    save_checkpoint(
        args.output_dir / "final_model.pt",
        model,
        cfg,
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
                "config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "history": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
