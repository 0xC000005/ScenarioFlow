#!/usr/bin/env python
"""Train neural baselines on the current real-VIX 39-state panel."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")

from experiments.backfill.baselines.current_panel_deep_baselines import (  # noqa: E402
    build_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


BASELINES = ["deepvar", "timegrad", "path_diffusion"]


class IncrementWindowDataset(Dataset):
    def __init__(
        self,
        history_increment: np.ndarray,
        future_increment: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        max_windows: int = 0,
    ) -> None:
        if max_windows and max_windows > 0:
            history_increment = history_increment[: int(max_windows)]
            future_increment = future_increment[: int(max_windows)]
        self.history = ((history_increment - mean[None, None, :]) / std[None, None, :]).astype(
            np.float32
        )
        self.future = ((future_increment - mean[None, None, :]) / std[None, None, :]).astype(
            np.float32
        )

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {"history": self.history[idx], "future": self.future[idx]}


def make_block_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        iv_count=int(args.iv_count),
        max_train_windows=0,
        clean_nonpositive_log_levels=True,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
    )


def train_one(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    *,
    input_dim: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = {
        "model_type": model_type,
        "input_dim": int(input_dim),
        "history_len": int(args.history_len),
        "future_len": int(args.future_len),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "rank": int(args.rank),
        "cond_dim": int(args.cond_dim),
        "n_diffusion_steps": int(args.n_diffusion_steps),
        "sample_steps": int(args.sample_steps),
    }
    model_kwargs = {
        k: v for k, v in config.items() if k not in {"model_type", "input_dim", "future_len"}
    }
    model = build_model(
        model_type,
        input_dim=input_dim,
        future_len=int(args.future_len),
        **model_kwargs,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    best_epoch = -1
    ckpt_path = output_dir / model_type / "best_model.pt"
    (output_dir / model_type).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            history = batch["history"].to(args.device)
            future = batch["future"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(history, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            total += float(loss.detach().cpu()) * history.shape[0]
            n += int(history.shape[0])
        train_loss = total / max(n, 1)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(args.device)
                future = batch["future"].to(args.device)
                loss = model(history, future)
                val_total += float(loss.detach().cpu()) * history.shape[0]
                val_n += int(history.shape[0])
        val_loss = val_total / max(val_n, 1)
        print(
            f"{model_type} epoch {epoch}/{args.epochs}: "
            f"train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                model,
                model_type=model_type,
                mean=mean,
                std=std,
                config=config,
                best_val_loss=best_val,
            )

    return {
        "model_type": model_type,
        "checkpoint": str(ckpt_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_params": int(n_params),
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
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
    parser.add_argument("--seed", type=int, default=774)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="models/backfill/baselines/current_panel_deep")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _columns, metadata, train_block, val_block = build_blocks(make_block_args(args))
    train_values = np.concatenate(
        [
            train_block.history_increment.reshape(-1, len(train_block.specs)),
            train_block.future_increment.reshape(-1, len(train_block.specs)),
        ],
        axis=0,
    )
    mean = train_values.mean(axis=0).astype(np.float32)
    std = (train_values.std(axis=0) + 1e-6).astype(np.float32)
    train_ds = IncrementWindowDataset(
        train_block.history_increment,
        train_block.future_increment,
        mean=mean,
        std=std,
        max_windows=int(args.max_train_windows),
    )
    val_ds = IncrementWindowDataset(
        val_block.history_increment,
        val_block.future_increment,
        mean=mean,
        std=std,
        max_windows=int(args.max_val_windows),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "current_realvix_39_state_panel",
        "seed": int(args.seed),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "state_specs": [asdict(spec) for spec in train_block.specs],
        "metadata": metadata,
        "baselines": {},
    }
    for model_type in args.baselines:
        print("\n" + "=" * 72)
        print(f"Training current-panel neural baseline: {model_type}")
        print("=" * 72)
        manifest["baselines"][model_type] = train_one(
            model_type,
            train_loader,
            val_loader,
            mean,
            std,
            args,
            input_dim=len(train_block.specs),
            output_dir=output_dir,
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(make_serializable(manifest), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote neural baseline checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
