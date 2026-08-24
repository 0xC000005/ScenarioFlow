"""Reusable Stage 1/Stage 2 training steps and a portable CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .losses import normalized_rollout_energy_loss
from .model import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    enable_adaptive_graph_velocity_mixer,
    load_model,
    save_checkpoint,
)
from .state import normalize_increment_windows


def load_training_model(
    checkpoint: str | Path,
    *,
    stage: str,
    device: torch.device,
    graph_k: int = 8,
) -> tuple[GenericStateAwareNormalizedInnovationFlowMatching, dict[str, Any]]:
    """Load an ordinary checkpoint, then perform the frozen graph continuation."""
    model, payload = load_model(checkpoint, device)
    if stage == "graph":
        enable_adaptive_graph_velocity_mixer(model, k_neighbors=int(graph_k))
        payload = dict(payload)
        payload["graph_migration"] = {
            "source": "ordinary_stage2_checkpoint",
            "method": "output_preserving_zero_residual",
            "k_neighbors": int(graph_k),
        }
    return model, payload


@torch.no_grad()
def initialize_level_quantiles(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    future_level_values: torch.Tensor,
) -> None:
    """Fit the model's empirical level-score table from training-only levels."""
    if history_level_values.shape[-1] != model.cfg.n_cells:
        raise ValueError("history_level_values channel count does not match the model")
    if future_level_values.shape[-1] != model.cfg.n_cells:
        raise ValueError("future_level_values channel count does not match the model")
    values = torch.cat(
        [
            history_level_values.reshape(-1, model.cfg.n_cells),
            future_level_values.reshape(-1, model.cfg.n_cells),
        ],
        dim=0,
    )
    quantile_levels = np.linspace(
        float(model.cfg.cdf_eps),
        1.0 - float(model.cfg.cdf_eps),
        int(model.cfg.n_quantiles),
        dtype=np.float64,
    )
    quantiles = np.quantile(
        values.detach().cpu().numpy().astype(np.float64),
        quantile_levels,
        axis=0,
    ).T.astype(np.float32)
    model.set_level_quantiles(
        torch.from_numpy(quantiles).to(values.device),
        torch.from_numpy(quantile_levels.astype(np.float32)).to(values.device),
    )


def _finish_step(
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    optimizer: torch.optim.Optimizer,
    *,
    clip_grad: float = 0.0,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if float(clip_grad) > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
    optimizer.step()
    return {key: float(value.detach().item()) for key, value in metrics.items()}


def stage1_training_step(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    optimizer: torch.optim.Optimizer,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    future_level_values: torch.Tensor,
    future_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
    future_no_update_target: torch.Tensor | None = None,
    *,
    clip_grad: float = 0.0,
    **objective: Any,
) -> dict[str, float]:
    """Run one Stage 1 teacher-forced flow-matching optimization step."""
    model.train()
    loss, metrics = model.training_loss(
        history_level_values,
        history_normalized_innovation,
        future_level_values,
        future_normalized_innovation,
        center,
        scale,
        drift_feature=drift_feature,
        future_no_update_target=future_no_update_target,
        **objective,
    )
    return _finish_step(loss, metrics, model, optimizer, clip_grad=clip_grad)


def stage2_training_step(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    optimizer: torch.optim.Optimizer,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    future_level_values: torch.Tensor,
    future_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor | None = None,
    future_no_update_target: torch.Tensor | None = None,
    *,
    clip_grad: float = 0.0,
    **objective: Any,
) -> dict[str, float]:
    """Run one Stage 2 free-rollout distributional optimization step."""
    model.train()
    loss, metrics = normalized_rollout_energy_loss(
        model,
        history_level_values,
        history_normalized_innovation,
        future_level_values,
        future_normalized_innovation,
        center,
        scale,
        drift_feature=drift_feature,
        future_no_update_target=future_no_update_target,
        **objective,
    )
    return _finish_step(loss, metrics, model, optimizer, clip_grad=clip_grad)


def _load_batch(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    keys = (
        "history_level_values",
        "history_normalized_innovation",
        "future_level_values",
        "future_normalized_innovation",
        "center",
        "scale",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"training batch is missing arrays: {missing}")
        return tuple(
            torch.from_numpy(np.asarray(payload[key], dtype=np.float32)).to(device)
            for key in keys
        )


def _block_batch(block: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    history_norm, future_norm, center, scale = normalize_increment_windows(
        block.history_increment.astype(np.float32),
        block.future_increment.astype(np.float32),
        half_life=None,
        scale_floor=1e-4,
        center_mode="zero",
    )
    arrays = (
        block.history_state,
        history_norm,
        block.future_state,
        future_norm,
        center,
        scale,
    )
    return tuple(
        torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for value in arrays
    )


def _panel_batches(
    data_dir: Path, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    from .evaluation.current_panel import prepare_validation_blocks

    _panel, _columns, train, validation = prepare_validation_blocks(data_dir=data_dir)
    return _block_batch(train, device), _block_batch(validation, device)


def _frozen_stage1_config(
    n_cells: int, *, smoke_model: bool = False
) -> GenericStateAwareNormalizedInnovationFMConfig:
    if smoke_model:
        return GenericStateAwareNormalizedInnovationFMConfig(
            history_len=30,
            future_len=30,
            n_cells=int(n_cells),
            memory_dim=16,
            memory_layers=1,
            memory_heads=1,
            memory_ff=32,
            token_dim=16,
            token_layers=1,
            token_heads=1,
            token_ff=32,
            model_dropout=0.0,
            time_dim=8,
            flow_steps=2,
            prefix_feature_mode="scale",
        )
    return GenericStateAwareNormalizedInnovationFMConfig(
        history_len=30,
        future_len=30,
        n_cells=int(n_cells),
        memory_dim=128,
        memory_layers=3,
        memory_heads=4,
        memory_ff=256,
        token_dim=128,
        token_layers=3,
        token_heads=4,
        token_ff=256,
        model_dropout=0.05,
        time_dim=32,
        flow_steps=16,
        prefix_feature_mode="scale",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["stage1", "stage2", "graph"], default="stage1"
    )
    parser.add_argument(
        "--batch-npz", type=Path, help="Portable six-array training batch"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--config", type=Path, help="JSON model config for a new Stage 1 model"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint to continue or fine-tune"
    )
    parser.add_argument("--output", type=Path, help="Output checkpoint path")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument(
        "--smoke-model",
        action="store_true",
        help="use an explicitly non-paper tiny architecture for bounded DAG smoke",
    )
    parser.add_argument("--config-overrides", type=Path)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=662)
    parser.add_argument("--train-sample-count", type=int, default=4)
    parser.add_argument("--rollout-flow-steps", type=int, default=8)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--channel-level-energy-weight", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--free-running-fm-weight", type=float, default=0.0)
    parser.add_argument("--free-running-fm-prefix-steps", type=int, default=0)
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument(
        "--channel-level-energy-coordinate",
        choices=("level", "scaled_delta"),
        default="level",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.batch_npz is None) == (args.data_dir is None) or args.output is None:
        parser.error("provide exactly one of --batch-npz or --data-dir, plus --output")
    if args.stage == "stage1" and args.checkpoint is not None:
        parser.error("Stage 1 creates a new model")
    if args.stage != "stage1" and args.checkpoint is None:
        parser.error("Stage 2 and graph continuation require --checkpoint")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    preloaded_batches: (
        tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None
    ) = None
    load_metadata: dict[str, Any] = {}
    if args.checkpoint is not None:
        overrides = (
            None
            if args.config_overrides is None
            else json.loads(args.config_overrides.read_text(encoding="utf-8"))
        )
        if overrides is not None and args.stage == "graph":
            parser.error(
                "graph continuation uses the package migration, not config overrides"
            )
        model, load_metadata = load_training_model(
            args.checkpoint,
            stage=args.stage,
            device=device,
            graph_k=int(args.graph_k),
        )
    else:
        config = (
            json.loads(args.config.read_text(encoding="utf-8"))
            if args.config is not None
            else None
        )
        if config is None and args.data_dir is None:
            parser.error("Stage 1 with --batch-npz requires --config")
        if config is None:
            preloaded_batches = _panel_batches(args.data_dir, device)
            train_preview, _validation_preview = preloaded_batches
            config = asdict(
                _frozen_stage1_config(
                    int(train_preview[0].shape[-1]),
                    smoke_model=bool(args.smoke_model),
                )
            )
        model = GenericStateAwareNormalizedInnovationFlowMatching(
            GenericStateAwareNormalizedInnovationFMConfig(**config)
        ).to(device)
    if args.data_dir is not None:
        batch, validation_batch = preloaded_batches or _panel_batches(
            args.data_dir, device
        )
    else:
        batch = _load_batch(args.batch_npz, device)
        validation_batch = batch
    if args.max_train_windows > 0:
        batch = tuple(value[: args.max_train_windows] for value in batch)
    if args.max_val_windows > 0:
        validation_batch = tuple(
            value[: args.max_val_windows] for value in validation_batch
        )
    if not batch[0].shape[0]:
        parser.error("the training batch must contain at least one window")
    if args.stage == "stage1":
        initialize_level_quantiles(model, batch[0], batch[2])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    metrics: dict[str, float] = {}
    training_steps = int(args.epochs if args.epochs is not None else args.steps)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    window_count = int(batch[0].shape[0])
    batch_size = min(int(args.batch_size), window_count)
    if batch_size <= 0:
        parser.error("--batch-size must be positive")
    best_validation = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(training_steps):
        permutation = torch.randperm(window_count, generator=generator)
        for start in range(0, window_count, batch_size):
            positions = permutation[start : start + batch_size].to(device)
            if args.drop_last and positions.numel() < batch_size:
                continue
            mini_batch = tuple(value.index_select(0, positions) for value in batch)
            if args.stage == "stage1":
                metrics = stage1_training_step(
                    model, optimizer, *mini_batch, clip_grad=float(args.clip_grad)
                )
            else:
                metrics = stage2_training_step(
                    model,
                    optimizer,
                    *mini_batch,
                    train_sample_count=int(args.train_sample_count),
                    rollout_flow_steps=int(args.rollout_flow_steps),
                    energy_weight=float(args.energy_weight),
                    channel_level_energy_weight=float(args.channel_level_energy_weight),
                    channel_level_energy_coordinate=args.channel_level_energy_coordinate,
                    free_running_fm_weight=float(args.free_running_fm_weight),
                    free_running_fm_prefix_steps=int(args.free_running_fm_prefix_steps),
                    fm_anchor_weight=1.0,
                    horizon_end_weight=1.2,
                    energy_eps=1e-6,
                    temperature=1.0,
                    clip_grad=float(args.clip_grad),
                )
        model.eval()
        validation_total = 0.0
        validation_count = 0
        validation_metric_totals: dict[str, float] = {}
        with torch.no_grad():
            for start in range(0, validation_batch[0].shape[0], batch_size):
                validation_mini_batch = tuple(
                    value[start : start + batch_size] for value in validation_batch
                )
                if args.stage == "stage1":
                    loss, validation_metrics = model.training_loss(
                        *validation_mini_batch
                    )
                else:
                    loss, validation_metrics = normalized_rollout_energy_loss(
                        model,
                        *validation_mini_batch,
                        train_sample_count=int(args.train_sample_count),
                        rollout_flow_steps=int(args.rollout_flow_steps),
                        energy_weight=float(args.energy_weight),
                        channel_level_energy_weight=float(
                            args.channel_level_energy_weight
                        ),
                        channel_level_energy_coordinate=(
                            args.channel_level_energy_coordinate
                        ),
                        free_running_fm_weight=float(args.free_running_fm_weight),
                        free_running_fm_prefix_steps=int(
                            args.free_running_fm_prefix_steps
                        ),
                        fm_anchor_weight=1.0,
                        horizon_end_weight=1.2,
                        energy_eps=1e-6,
                        temperature=1.0,
                    )
                size = int(validation_mini_batch[0].shape[0])
                validation_total += float(loss) * size
                validation_count += size
                for key, value in validation_metrics.items():
                    validation_metric_totals[key] = (
                        validation_metric_totals.get(key, 0.0)
                        + float(value.detach()) * size
                    )
        validation_value = validation_total / validation_count
        validation_metrics_mean = {
            key: value / validation_count
            for key, value in validation_metric_totals.items()
        }
        if np.isfinite(validation_value) and validation_value < best_validation:
            best_validation = validation_value
            best_metrics = validation_metrics_mean
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                args.output,
                model,
                model.cfg,
                epoch=epoch + 1,
                best_val=best_validation,
                extra={
                    "training_stage": args.stage,
                    "best_epoch": epoch,
                    **(
                        {"graph_migration": load_metadata["graph_migration"]}
                        if "graph_migration" in load_metadata
                        else {}
                    ),
                },
            )
    if not args.output.is_file():
        raise RuntimeError("training did not produce a finite validation checkpoint")
    metrics.update({f"validation_{key}": value for key, value in best_metrics.items()})
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
