"""Evaluate final classical and deep baselines on the released 39-state panel."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..reproduction.panel import prepare_panel_blocks
from .baselines.classical import (
    JointBootstrap,
    JointFilteredHistoricalSimulation,
    JointGaussianGARCHCCC,
    JointPCAVAR,
    JointRandomWalk,
    WindowHistoricalSimulation,
)
from .baselines.deep import load_current_panel_deep_baseline
from .evaluation import sample_and_evaluate
from .window_scores import add_window_score_argument

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CLASSICAL = (
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
)
DEEP = ("deepvar", "timegrad", "path_diffusion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument(
        "--baselines", nargs="+", choices=CLASSICAL + DEEP, required=True
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
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--deep_checkpoint_dir", type=Path)
    parser.add_argument("--deep_sample_steps", type=int, default=8)
    parser.add_argument("--deep_chunk_size", type=int, default=512)
    add_window_score_argument(parser)
    return parser


def _create_model(name: str, train_block: Any, args: argparse.Namespace) -> Any:
    daily = train_block.history_increment[:, -1].astype(np.float32)
    if name == "random_walk":
        return JointRandomWalk(daily, args.future_len)
    if name == "bootstrap":
        return JointBootstrap(daily, args.future_len)
    if name == "historical_sim":
        return WindowHistoricalSimulation(
            train_block.history_increment,
            train_block.future_increment,
            train_block.history_state,
            iv_count=args.iv_count,
        )
    if name == "pca_var":
        return JointPCAVAR(daily, args.future_len)
    if name == "garch_ccc":
        return JointGaussianGARCHCCC(daily, args.future_len)
    if name == "filtered_hs":
        return JointFilteredHistoricalSimulation(daily, args.future_len)
    if args.deep_checkpoint_dir is None:
        raise ValueError("--deep_checkpoint_dir is required for neural baselines")
    return load_current_panel_deep_baseline(
        args.deep_checkpoint_dir / name / "best_model.pt",
        device=args.device,
        sample_steps=args.deep_sample_steps,
        chunk_size=args.deep_chunk_size,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    columns, metadata, train_block, validation_block = prepare_panel_blocks(
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
        stop = int(args.max_windows)
        validation_block = replace(
            validation_block,
            history_increment=validation_block.history_increment[:stop],
            future_increment=validation_block.future_increment[:stop],
            history_state=validation_block.history_state[:stop],
            future_state=validation_block.future_state[:stop],
            indices=validation_block.indices[:stop],
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.baselines:
        model = _create_model(name, train_block, args)
        results[name] = sample_and_evaluate(
            candidate=name,
            model=model,
            train_block=train_block,
            validation_block=validation_block,
            samples=args.n_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_window_scores=args.save_window_scores,
            seed=args.seed,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "current_realvix_39_state_panel",
                "columns": columns,
                "metadata": metadata,
                "baselines": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
