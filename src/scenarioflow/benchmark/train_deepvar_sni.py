"""Train DeepVAR with ScenarioFlow's history-local innovation normalization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..reproduction.panel import prepare_panel_blocks
from ..state import normalize_increment_windows
from .baselines.deep import build_model

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
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=("zero", "ewma_mean"), default="zero")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    return parser


def _loss_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    clip: float,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    for history, future in loader:
        history = history.to(device)
        future = future.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = model(history, future)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total += float(loss.detach()) * history.shape[0]
        count += int(history.shape[0])
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    _columns, metadata, train_block, validation_block = prepare_panel_blocks(
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
    half_life = None if args.scale_half_life <= 0 else args.scale_half_life
    train_history, train_future, _center, _scale = normalize_increment_windows(
        train_block.history_increment,
        train_block.future_increment,
        half_life=half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
    )
    validation_history, validation_future, _center, _scale = (
        normalize_increment_windows(
            validation_block.history_increment,
            validation_block.future_increment,
            half_life=half_life,
            scale_floor=args.scale_floor,
            center_mode=args.center_mode,
        )
    )
    if args.max_train_windows > 0:
        train_history = train_history[: args.max_train_windows]
        train_future = train_future[: args.max_train_windows]
    if args.max_val_windows > 0:
        validation_history = validation_history[: args.max_val_windows]
        validation_future = validation_future[: args.max_val_windows]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_history), torch.from_numpy(train_future)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_history), torch.from_numpy(validation_future)
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    config = {
        "model_type": "deepvar",
        "input_dim": len(train_block.specs),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "rank": args.rank,
        "sample_steps": args.sample_steps,
    }
    normalization = {
        "coordinate": "encoded_increment",
        "center_mode": args.center_mode,
        "scale_method": "ewma_rms",
        "scale_half_life": half_life,
        "scale_floor": args.scale_floor,
        "iv_transform": args.iv_transform,
        "iv_lower_bound": args.iv_lower_bound,
        "iv_upper_bound": args.iv_upper_bound,
        "innovation_coordinate": "normalized",
    }
    model = build_model(
        "deepvar",
        input_dim=len(train_block.specs),
        future_len=args.future_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rank=args.rank,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint = args.output_dir / "deepvar_sni/best_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = _loss_epoch(model, train_loader, device, optimizer, args.grad_clip)
        with torch.no_grad():
            validation_loss = _loss_epoch(
                model, validation_loader, device, None, args.grad_clip
            )
        if not np.isfinite(validation_loss):
            raise RuntimeError("DeepVAR ScenarioFlow-interface loss became non-finite")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            ),
            flush=True,
        )
        if validation_loss < best:
            best = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_type": "deepvar",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalization": normalization,
                    "best_val_loss": best,
                    "best_epoch": best_epoch,
                },
                checkpoint,
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "current_realvix_39_state_panel",
                "variant": "deepvar_sni_interface",
                "seed": args.seed,
                "metadata": metadata,
                "state_specs": [asdict(spec) for spec in train_block.specs],
                "checkpoint": str(checkpoint),
                "best_epoch": best_epoch,
                "best_val_loss": best,
                "normalization": normalization,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
