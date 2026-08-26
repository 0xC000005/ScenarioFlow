"""Evaluate DeepVAR using the ScenarioFlow observation interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..reproduction.panel import prepare_panel_blocks
from .baselines.deepvar_sni import DeepVARSNIInterfaceBaseline
from .evaluation import sample_and_evaluate
from .window_scores import add_window_score_argument

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
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
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deep_checkpoint", type=Path, required=True)
    parser.add_argument("--deep_chunk_size", type=int, default=512)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    add_window_score_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _columns, _metadata, train_block, validation_block = prepare_panel_blocks(
        data_dir=args.data_dir,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=args.iv_lower_bound,
        iv_upper_bound=args.iv_upper_bound,
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
    model = DeepVARSNIInterfaceBaseline.load(
        args.deep_checkpoint, device=args.device, chunk_size=args.deep_chunk_size
    )
    sample_and_evaluate(
        candidate="deepvar_sni",
        model=model,
        train_block=train_block,
        validation_block=validation_block,
        samples=args.n_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_window_scores=args.save_window_scores,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
