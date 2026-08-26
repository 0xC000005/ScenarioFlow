"""Train the final DeepVAR, TimeGrad, and path-diffusion baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ..reproduction.panel import prepare_panel_blocks
from .baselines.deep import build_model, save_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASELINES = ("deepvar", "timegrad", "path_diffusion")


class IncrementWindowDataset(Dataset):
    def __init__(
        self,
        history: np.ndarray,
        future: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        max_windows: int = 0,
    ) -> None:
        if int(max_windows) > 0:
            history = history[: int(max_windows)]
            future = future[: int(max_windows)]
        self.history = ((history - mean[None, None]) / std[None, None]).astype(
            np.float32
        )
        self.future = ((future - mean[None, None]) / std[None, None]).astype(np.float32)

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {"history": self.history[index], "future": self.future[index]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", "--data_dir", type=Path, default=REPOSITORY_ROOT / "data"
    )
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=BASELINES)
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--n_diffusion_steps", type=int, default=40)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    return parser


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    clip: float,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
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
    values = np.concatenate(
        [
            train_block.history_increment.reshape(-1, len(train_block.specs)),
            train_block.future_increment.reshape(-1, len(train_block.specs)),
        ]
    )
    mean = values.mean(axis=0).astype(np.float32)
    std = (values.std(axis=0) + 1e-6).astype(np.float32)
    train_data = IncrementWindowDataset(
        train_block.history_increment,
        train_block.future_increment,
        mean=mean,
        std=std,
        max_windows=args.max_train_windows,
    )
    validation_data = IncrementWindowDataset(
        validation_block.history_increment,
        validation_block.future_increment,
        mean=mean,
        std=std,
        max_windows=args.max_val_windows,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "mode": "current_realvix_39_state_panel",
        "columns": columns,
        "metadata": metadata,
        "seed": args.seed,
        "state_specs": [asdict(spec) for spec in train_block.specs],
        "baselines": {},
    }
    for name in args.baselines:
        config = {
            "model_type": name,
            "input_dim": len(train_block.specs),
            "history_len": args.history_len,
            "future_len": args.future_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "rank": args.rank,
            "cond_dim": args.cond_dim,
            "n_diffusion_steps": args.n_diffusion_steps,
            "sample_steps": args.sample_steps,
        }
        model = build_model(
            name,
            input_dim=len(train_block.specs),
            future_len=args.future_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            rank=args.rank,
            cond_dim=args.cond_dim,
            n_diffusion_steps=args.n_diffusion_steps,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = float("inf")
        best_epoch = -1
        checkpoint = args.output_dir / name / "best_model.pt"
        for epoch in range(1, args.epochs + 1):
            train_loss = _epoch(model, train_loader, device, optimizer, args.grad_clip)
            with torch.no_grad():
                validation_loss = _epoch(
                    model, validation_loader, device, None, args.grad_clip
                )
            print(
                json.dumps(
                    {
                        "baseline": name,
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
                save_checkpoint(
                    checkpoint,
                    model,
                    model_type=name,
                    mean=mean,
                    std=std,
                    config=config,
                    best_val_loss=best,
                )
        manifest["baselines"][name] = {
            "checkpoint": str(checkpoint),
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "config": config,
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
