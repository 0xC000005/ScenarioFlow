"""Evaluate a clean ScenarioFlow checkpoint under the final panel contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from ..reproduction.inference import generate_deck, load_checkpoint_strict
from ..reproduction.panel import (
    prepare_panel_blocks,
    select_normalized_innovation_scope,
)
from .evaluation import evaluate_samples
from .window_scores import add_window_score_argument

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=("scenarioflow", "scenarioflow_graph"),
        default="scenarioflow",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument("--state_scope", default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--iv_transform",
        choices=("log_level", "bounded_logit"),
        default="bounded_logit",
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument(
        "--positive_level_policy",
        choices=("reference_based", "observed_positive"),
        default="reference_based",
    )
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    add_window_score_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, payload = load_checkpoint_strict(args.checkpoint, device)
    normalization = payload.get("normalization", {})
    half_life = normalization.get("scale_half_life")
    scale_half_life = (
        None if half_life is None or float(half_life) <= 0 else float(half_life)
    )
    _columns, _metadata, train_block, validation_block = prepare_panel_blocks(
        data_dir=args.data_dir,
        history_len=int(model.cfg.history_len),
        future_len=int(model.cfg.future_len),
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        positive_level_policy=payload.get(
            "positive_level_policy", args.positive_level_policy
        ),
        iv_transform=normalization.get(
            "iv_transform", payload.get("iv_transform", args.iv_transform)
        ),
        iv_lower_bound=float(
            normalization.get(
                "iv_lower_bound", payload.get("iv_lower_bound", args.iv_lower_bound)
            )
        ),
        iv_upper_bound=float(
            normalization.get(
                "iv_upper_bound", payload.get("iv_upper_bound", args.iv_upper_bound)
            )
        ),
    )
    if args.max_windows > 0:
        stop = args.max_windows
        validation_block = replace(
            validation_block,
            history_increment=validation_block.history_increment[:stop],
            future_increment=validation_block.future_increment[:stop],
            history_state=validation_block.history_state[:stop],
            future_state=validation_block.future_state[:stop],
            indices=validation_block.indices[:stop],
        )
    selected = select_normalized_innovation_scope(
        validation_block,
        payload.get("state_scope", args.state_scope),
        args.iv_count,
        scale_half_life=scale_half_life,
        scale_floor=float(normalization.get("scale_floor", 1e-4)),
        center_mode=normalization.get("center_mode", "zero"),
        drift_feature_mode=normalization.get("drift_feature_mode", "none"),
    )
    sample_raw = generate_deck(
        model,
        selected[0],
        selected[1],
        selected[4],
        selected[5],
        selected[6],
        selected[7],
        selected[8],
        samples=args.samples,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        temperature=args.sample_temperature,
        seed=args.seed,
        device=device,
    )
    evaluate_samples(
        candidate=args.candidate,
        train_block=train_block,
        validation_block=validation_block,
        sample_raw=sample_raw,
        output_dir=args.output_dir,
        save_window_scores=args.save_window_scores,
    )


if __name__ == "__main__":
    main()
