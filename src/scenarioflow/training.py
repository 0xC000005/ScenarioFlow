"""Reusable Stage 1/Stage 2 training steps and a portable CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .losses import normalized_rollout_energy_loss
from .model import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    load_model,
    save_checkpoint,
)


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
    quantile_levels = model.quantile_levels.to(device=values.device, dtype=values.dtype)
    quantiles = torch.quantile(values, quantile_levels, dim=0).transpose(0, 1)
    model.set_level_quantiles(quantiles, quantile_levels)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument(
        "--batch-npz", type=Path, help="Portable six-array training batch"
    )
    parser.add_argument(
        "--config", type=Path, help="JSON model config for a new Stage 1 model"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint to continue or fine-tune"
    )
    parser.add_argument("--output", type=Path, help="Output checkpoint path")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=662)
    parser.add_argument("--train-sample-count", type=int, default=4)
    parser.add_argument("--rollout-flow-steps", type=int, default=8)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_npz is None or args.output is None:
        parser.error("--batch-npz and --output are required to run training")
    if (args.config is None) == (args.checkpoint is None):
        parser.error("provide exactly one of --config or --checkpoint")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    if args.checkpoint is not None:
        model, _ = load_model(args.checkpoint, device)
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        model = GenericStateAwareNormalizedInnovationFlowMatching(
            GenericStateAwareNormalizedInnovationFMConfig(**config)
        ).to(device)
    batch = _load_batch(args.batch_npz, device)
    if args.config is not None:
        initialize_level_quantiles(model, batch[0], batch[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    metrics: dict[str, float] = {}
    for _ in range(int(args.steps)):
        if args.stage == "stage1":
            metrics = stage1_training_step(model, optimizer, *batch)
        else:
            metrics = stage2_training_step(
                model,
                optimizer,
                *batch,
                train_sample_count=int(args.train_sample_count),
                rollout_flow_steps=int(args.rollout_flow_steps),
                energy_weight=float(args.energy_weight),
                fm_anchor_weight=1.0,
                horizon_end_weight=1.2,
                energy_eps=1e-6,
                temperature=1.0,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.output,
        model,
        model.cfg,
        epoch=int(args.steps),
        best_val=float(metrics.get("total", float("nan"))),
        extra={"training_stage": args.stage},
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
