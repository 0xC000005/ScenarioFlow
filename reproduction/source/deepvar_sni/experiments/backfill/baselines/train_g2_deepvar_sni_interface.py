#!/usr/bin/env python
"""G2: train DeepVAR inside the SNI observation interface.

Apples-to-apples width control for the ICAIF review (R3). This is a ONE-AXIS SWAP
relative to ``train_current_panel_deep_baselines.py``: the global per-dim z-score of
encoded increments is replaced by the SNI per-window EWMA-RMS local-scale
normalization (``normalize_increment_windows``, center=0). The DeepVAR model class,
the state map ``g_d`` (via ``build_blocks``), the reconstruction path, and the entire
training recipe (epochs / batch / lr / seed / capacity) are held identical to the
current-panel DeepVAR baseline. Only the normalizer differs.

New files / new output dir only. Existing scripts and every incumbent checkpoint
(734a/739a, ``current_panel_deep``) are untouched.

Run:
  PYTHONPATH=. python experiments/backfill/baselines/train_g2_deepvar_sni_interface.py \
      --output_dir models/backfill/baselines/g2_sni_interface --device cuda
"""

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
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    make_serializable,
)
from experiments.backfill.block_ar.normalized_innovation_662_utils import (  # noqa: E402
    normalize_increment_windows,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


class NormalizedInnovationWindowDataset(Dataset):
    """Yields per-window SNI-normalized innovations (history + future)."""

    def __init__(
        self,
        history_norm: np.ndarray,
        future_norm: np.ndarray,
        *,
        max_windows: int = 0,
    ) -> None:
        if max_windows and max_windows > 0:
            history_norm = history_norm[: int(max_windows)]
            future_norm = future_norm[: int(max_windows)]
        self.history = np.asarray(history_norm, dtype=np.float32)
        self.future = np.asarray(future_norm, dtype=np.float32)

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


def train_deepvar_sni(
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    *,
    input_dim: int,
    normalization: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    config = {
        "model_type": "deepvar",
        "input_dim": int(input_dim),
        "history_len": int(args.history_len),
        "future_len": int(args.future_len),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "rank": int(args.rank),
        "sample_steps": int(args.sample_steps),
    }
    model = build_model(
        "deepvar",
        input_dim=input_dim,
        future_len=int(args.future_len),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        rank=int(args.rank),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = Adam(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    best_val = float("inf")
    best_epoch = -1
    model_dir = output_dir / "deepvar_sni"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "best_model.pt"

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
            f"deepvar_sni epoch {epoch}/{args.epochs}: "
            f"train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )
        if not np.isfinite(val_loss):
            raise RuntimeError(
                f"deepvar_sni val loss non-finite at epoch {epoch} "
                "(heavy-tailed normalized innovations destabilized NLL — "
                "invoke the TimeGrad fallback rather than tuning the floor)"
            )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            payload = {
                "model_type": "deepvar",
                "model_state_dict": model.state_dict(),
                "config": config,
                "normalization": normalization,
                "best_val_loss": float(best_val),
                "best_epoch": int(best_epoch),
            }
            torch.save(payload, ckpt_path)

    if best_epoch < 0:
        raise RuntimeError("deepvar_sni training produced no finite checkpoint")

    return {
        "model_type": "deepvar_sni",
        "checkpoint": str(ckpt_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_params": int(n_params),
        "config": config,
        "normalization": normalization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data / block args (identical defaults to the current-panel DeepVAR recipe).
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit"
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    # SNI observation interface (matched byte-for-byte to 734a's normalization block).
    parser.add_argument("--scale_half_life", type=float, default=0.0)  # <=0 -> None -> H/3
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    # Training recipe (identical to current-panel DeepVAR).
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
    parser.add_argument("--seed", type=int, default=774)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir", default="models/backfill/baselines/g2_sni_interface"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    scale_half_life = None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life)
    normalization = {
        "coordinate": "encoded_increment",
        "center_mode": str(args.center_mode),
        "drift_feature_mode": "none",
        "scale_method": "ewma_rms",
        "scale_half_life": scale_half_life,
        "scale_floor": float(args.scale_floor),
        "iv_transform": str(args.iv_transform),
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "innovation_coordinate": "normalized",
        "state_scope": "joint39_full_block",
    }

    _columns, metadata, train_block, val_block = build_blocks(make_block_args(args))
    if len(train_block.specs) != 39:
        raise RuntimeError(f"expected 39-state panel, got {len(train_block.specs)}")

    # SNI per-window normalization on the FULL 39-channel block (no joint38 subset).
    train_hist_norm, train_fut_norm, _c_tr, _s_tr = normalize_increment_windows(
        train_block.history_increment,
        train_block.future_increment,
        half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=str(args.center_mode),
    )
    val_hist_norm, val_fut_norm, _c_va, _s_va = normalize_increment_windows(
        val_block.history_increment,
        val_block.future_increment,
        half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=str(args.center_mode),
    )

    train_ds = NormalizedInnovationWindowDataset(
        train_hist_norm, train_fut_norm, max_windows=int(args.max_train_windows)
    )
    val_ds = NormalizedInnovationWindowDataset(
        val_hist_norm, val_fut_norm, max_windows=int(args.max_val_windows)
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("Training G2 DeepVAR inside the SNI observation interface")
    print("=" * 72)
    summary = train_deepvar_sni(
        train_loader,
        val_loader,
        args,
        input_dim=len(train_block.specs),
        normalization=normalization,
        output_dir=output_dir,
    )

    manifest = {
        "mode": "current_realvix_39_state_panel",
        "variant": "deepvar_sni_interface",
        "seed": int(args.seed),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "normalization": normalization,
        "state_specs": [asdict(spec) for spec in train_block.specs],
        "metadata": metadata,
        "baseline": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(make_serializable(manifest), indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote deepvar_sni checkpoint to {output_dir / 'deepvar_sni' / 'best_model.pt'} "
        f"(best_epoch={summary['best_epoch']}, n_params={summary['n_params']})"
    )


if __name__ == "__main__":
    main()
