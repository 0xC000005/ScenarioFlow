"""Released-checkpoint loading and conditional ScenarioFlow deck generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..model import (
    GenericStateAwareNormalizedInnovationFlowMatching,
    load_model,
)
from ..state import reconstruct_state_from_increments
from .metrics import evaluate_deck
from .panel import prepare_panel_blocks, select_normalized_innovation_scope

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_checkpoint_strict(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[GenericStateAwareNormalizedInnovationFlowMatching, dict[str, Any]]:
    """Load a released ordinary or adaptive-graph checkpoint without key drift."""
    model, payload = load_model(checkpoint, device)
    checkpoint_state = dict(payload["model_state_dict"])
    model_state = model.state_dict()
    for name in ("no_update_mask", "no_update_prior_rate"):
        if name not in checkpoint_state:
            checkpoint_state[name] = model_state[name]
    model.load_state_dict(checkpoint_state, strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def velocity_output(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level: torch.Tensor,
    history_normalized: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    drift_feature: torch.Tensor,
    *,
    seed: int,
) -> np.ndarray:
    """Frozen two-row velocity probe used by the parity contract."""
    model.eval()
    level_scores = model.level_values_to_scores(history_level)
    flow_coordinate = model._to_flow_coordinate(history_normalized)
    memory = model._encode_prefix(
        level_scores, flow_coordinate, center, scale, drift_feature
    )[:, -1]
    torch.manual_seed(int(seed))
    state = torch.randn(2, model.cfg.n_cells, dtype=torch.float32)
    time = torch.tensor([0.25, 0.75], dtype=torch.float32)
    values = model.velocity(state, level_scores[:, -1], memory, time)
    return values.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def generate_deck(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level: np.ndarray,
    history_normalized: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    temperature: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Generate and reconstruct complete raw-state scenario paths."""
    model.eval()
    torch.manual_seed(int(seed))
    chunks: list[np.ndarray] = []
    for start in range(0, int(history_level.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(history_level.shape[0]))
        sampled_increment = model.sample_batched(
            torch.from_numpy(history_level[start:end]).to(device),
            torch.from_numpy(history_normalized[start:end]).to(device),
            torch.from_numpy(center[start:end]).to(device),
            torch.from_numpy(scale[start:end]).to(device),
            drift_feature=torch.from_numpy(drift_feature[start:end]).to(device),
            n_samples=int(samples),
            n_steps=int(n_steps),
            chunk_size=int(chunk_size),
            temperature=float(temperature),
        )
        increment_array = sampled_increment.detach().cpu().numpy()
        panel_array = reconstruct_state_from_increments(
            history_raw[start:end, -1, :], increment_array, specs
        )
        chunks.append(panel_array.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/cache/models/core/stage2/best_model.pt",
    )
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=REPOSITORY_ROOT / "build/inference/scenarioflow",
    )
    parser.add_argument("--max-windows", "--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n-steps", "--n_steps", type=int, default=30)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=8)
    parser.add_argument(
        "--sample-temperature", "--sample_temperature", type=float, default=1.0
    )
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing released checkpoint: {args.checkpoint}")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, payload = load_checkpoint_strict(args.checkpoint, device)
    normalization = payload.get("normalization", {})
    half_life = normalization.get("scale_half_life")
    scale_half_life = (
        None if half_life is None or float(half_life) <= 0.0 else float(half_life)
    )
    _columns, _metadata, train_block, validation_block = prepare_panel_blocks(
        data_dir=args.data_dir,
        history_len=int(model.cfg.history_len),
        future_len=int(model.cfg.future_len),
        iv_count=int(payload.get("iv_count", 25)),
        positive_level_policy=payload.get("positive_level_policy", "reference_based"),
        iv_transform=normalization.get(
            "iv_transform", payload.get("iv_transform", "bounded_logit")
        ),
        iv_lower_bound=float(
            normalization.get("iv_lower_bound", payload.get("iv_lower_bound", 1e-4))
        ),
        iv_upper_bound=float(
            normalization.get("iv_upper_bound", payload.get("iv_upper_bound", 1.0))
        ),
    )
    selected = select_normalized_innovation_scope(
        validation_block,
        payload.get("state_scope", "joint38"),
        int(payload.get("iv_count", 25)),
        scale_half_life=scale_half_life,
        scale_floor=float(normalization.get("scale_floor", 1e-4)),
        center_mode=normalization.get("center_mode", "zero"),
        drift_feature_mode=normalization.get("drift_feature_mode", "none"),
    )
    (
        history_level,
        history_normalized,
        _future_level,
        _future_normalized,
        center,
        scale,
        drift,
        history_raw,
        specs,
    ) = selected
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    if n_windows < 1:
        raise ValueError("--max-windows must select at least one validation window")
    deck = generate_deck(
        model,
        history_level[:n_windows],
        history_normalized[:n_windows],
        center[:n_windows],
        scale[:n_windows],
        drift[:n_windows],
        history_raw[:n_windows],
        specs,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        temperature=float(args.sample_temperature),
        seed=int(args.seed),
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "scenarios.npz"
    np.savez_compressed(
        output_path,
        scenarios=deck,
        window_indices=validation_block.indices[:n_windows],
        seed=np.asarray(int(args.seed), dtype=np.int64),
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "output": str(output_path),
        "shape": list(deck.shape),
        "seed": int(args.seed),
    }
    if int(args.n_steps) == int(model.cfg.future_len):
        metrics, screens = evaluate_deck(
            raw_history=history_raw[:n_windows],
            raw_future=validation_block.future_state[:n_windows],
            sample_raw=deck,
            train_block=train_block,
            factor_names=[
                spec.name for spec in specs[int(payload.get("iv_count", 25)) :]
            ],
            iv_count=int(payload.get("iv_count", 25)),
        )
        result["metrics"] = metrics.tolist()
        result["screens"] = screens.tolist()
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
