"""Checkpoint loading and deterministic numerical path inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import load_model
from .state import UnifiedVariableSpec, reconstruct_state_from_normalized_increments


@torch.no_grad()
def sample_normalized_innovations(
    *,
    checkpoint: str | Path,
    history_level_values: torch.Tensor | np.ndarray,
    history_normalized_innovation: torch.Tensor | np.ndarray,
    center: torch.Tensor | np.ndarray,
    scale: torch.Tensor | np.ndarray,
    drift_feature: torch.Tensor | np.ndarray | None = None,
    n_samples: int = 50,
    n_steps: int | None = None,
    flow_steps: int | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load a compatible checkpoint and sample normalized-innovation paths."""
    resolved_device = torch.device(device)
    model, _ = load_model(checkpoint, resolved_device)

    def tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=resolved_device)

    history_level = tensor(history_level_values)
    history_normalized = tensor(history_normalized_innovation)
    center_tensor = tensor(center)
    scale_tensor = tensor(scale)
    drift_tensor = None if drift_feature is None else tensor(drift_feature)
    horizon = int(model.cfg.future_len if n_steps is None else n_steps)
    integration_steps = int(model.cfg.flow_steps if flow_steps is None else flow_steps)
    devices = [resolved_device] if resolved_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        original_flow_steps = int(model.cfg.flow_steps)
        model.cfg.flow_steps = integration_steps
        try:
            sampled_increment = model.sample_batched(
                history_level,
                history_normalized,
                center_tensor,
                scale_tensor,
                drift_feature=drift_tensor,
                n_samples=int(n_samples),
                n_steps=horizon,
                chunk_size=int(n_samples),
                temperature=float(temperature),
            )
        finally:
            model.cfg.flow_steps = original_flow_steps
    return (
        (sampled_increment - center_tensor[:, None, None, :])
        / scale_tensor[:, None, None, :].clamp_min(1e-8)
    ).detach()


def reconstruct_sampled_states(
    sampled_normalized_innovation: torch.Tensor | np.ndarray,
    *,
    last_history_state: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    specs: list[UnifiedVariableSpec],
) -> np.ndarray:
    """Reconstruct raw scenario paths from sampled normalized innovations."""
    sampled = (
        sampled_normalized_innovation.detach().cpu().numpy()
        if isinstance(sampled_normalized_innovation, torch.Tensor)
        else np.asarray(sampled_normalized_innovation)
    )
    return reconstruct_state_from_normalized_increments(
        last_history_state, sampled, center, scale, specs
    )


def _load_inputs(path: Path) -> dict[str, np.ndarray]:
    required = (
        "history_level_values",
        "history_normalized_innovation",
        "center",
        "scale",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"inference input is missing arrays: {missing}")
        return {key: np.asarray(payload[key], dtype=np.float32) for key in required}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="ScenarioFlow checkpoint")
    parser.add_argument("--input", type=Path, help="Portable four-array NPZ input")
    parser.add_argument("--output", type=Path, help="Output NPZ path")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint is None or args.input is None or args.output is None:
        parser.error("--checkpoint, --input, and --output are required for inference")
    inputs = _load_inputs(args.input)
    samples = sample_normalized_innovations(
        checkpoint=args.checkpoint,
        **inputs,
        n_samples=int(args.samples),
        n_steps=args.steps,
        flow_steps=args.flow_steps,
        temperature=float(args.temperature),
        seed=int(args.seed),
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        normalized_innovation=samples.cpu().numpy(),
        seed=np.asarray(int(args.seed), dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(samples.shape),
                "seed": args.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
