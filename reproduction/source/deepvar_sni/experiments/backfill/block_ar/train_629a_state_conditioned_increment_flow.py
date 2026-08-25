#!/usr/bin/env python
"""629a: train a state-conditioned encoded-increment transition flow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (  # noqa: E402
    GenericStateConditionedIncrementFMConfig,
    GenericStateConditionedIncrementFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import encode_state  # noqa: E402
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    IncrementCoordinateBlock,
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
    fit_empirical_quantiles,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


def select_state_increment_scope(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    if scope == "joint38":
        specs = block.specs
        return (
            encode_state(block.history_state, specs).astype(np.float32),
            block.history_increment,
            encode_state(block.future_state, specs).astype(np.float32),
            block.future_increment,
            block.history_state,
            specs,
        )
    if scope == "iv_only":
        specs = block.specs[:iv_count]
        return (
            encode_state(block.history_state[..., :iv_count], specs).astype(np.float32),
            block.history_increment[..., :iv_count],
            encode_state(block.future_state[..., :iv_count], specs).astype(np.float32),
            block.future_increment[..., :iv_count],
            block.history_state[..., :iv_count],
            specs,
        )
    raise ValueError(f"unknown state_scope {scope!r}")


def eval_loss(
    model: GenericStateConditionedIncrementFlowMatching,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for history_state, history_increment, future_state, future_increment in loader:
            loss, _metrics = model.training_loss(
                history_state.to(device),
                history_increment.to(device),
                future_state.to(device),
                future_increment.to(device),
            )
            batch_n = int(history_state.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
    return total / max(count, 1)


@torch.no_grad()
def sample_smoke(
    model: GenericStateConditionedIncrementFlowMatching,
    history_level: np.ndarray,
    history_increment: np.ndarray,
    history_state_raw: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    steps: int,
    chunk_size: int,
    device: torch.device,
    iv_count: int,
) -> dict[str, Any]:
    hist_level = torch.from_numpy(history_level[:4]).to(device)
    hist_increment = torch.from_numpy(history_increment[:4]).to(device)
    sampled_increment = model.sample_batched(
        hist_level,
        hist_increment,
        n_samples=int(samples),
        n_steps=int(steps),
        chunk_size=int(chunk_size),
    )
    increments = sampled_increment.detach().cpu().numpy()
    states = reconstruct_state_from_increments(history_state_raw[:4, -1, :], increments, specs)
    report: dict[str, Any] = {
        "sample_increment_shape": list(increments.shape),
        "sample_state_shape": list(states.shape),
        "finite_increment_rate": float(np.isfinite(increments).mean()),
        "finite_state_rate": float(np.isfinite(states).mean()),
        "iv_min": float(np.nanmin(states[..., :iv_count])),
        "iv_max": float(np.nanmax(states[..., :iv_count])),
    }
    if states.shape[-1] > iv_count:
        report.update(
            {
                "factor_min": float(np.nanmin(states[..., iv_count:])),
                "factor_max": float(np.nanmax(states[..., iv_count:])),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.05)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale"], default="basic")
    parser.add_argument("--conditional_source_scale", action="store_true")
    parser.add_argument("--source_scale_min", type=float, default=0.5)
    parser.add_argument("--source_scale_max", type=float, default=2.0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=629)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_level, train_increment, train_future_level, train_future_increment, _train_raw, train_specs = (
        select_state_increment_scope(train_block, args.state_scope, int(args.iv_count))
    )
    val_level, val_increment, val_future_level, val_future_increment, val_raw, val_specs = select_state_increment_scope(
        val_block,
        args.state_scope,
        int(args.iv_count),
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    level_quantiles, quantile_levels = fit_empirical_quantiles(
        train_level,
        train_future_level,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    increment_quantiles, _ = fit_empirical_quantiles(
        train_increment,
        train_future_increment,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    cfg = GenericStateConditionedIncrementFMConfig(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        n_cells=int(train_level.shape[-1]),
        memory_dim=int(args.memory_dim),
        memory_layers=int(args.memory_layers),
        memory_heads=int(args.memory_heads),
        memory_ff=int(args.memory_ff),
        token_dim=int(args.token_dim),
        token_layers=int(args.token_layers),
        token_heads=int(args.token_heads),
        token_ff=int(args.token_ff),
        time_dim=int(args.time_dim),
        model_dropout=float(args.model_dropout),
        flow_steps=int(args.flow_steps),
        sample_temperature=float(args.sample_temperature),
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
        prefix_feature_mode=args.prefix_feature_mode,
        conditional_source_scale=bool(args.conditional_source_scale),
        source_scale_min=float(args.source_scale_min),
        source_scale_max=float(args.source_scale_max),
        conditioning_mode="prefix",
    )
    model = GenericStateConditionedIncrementFlowMatching(cfg).to(device)
    model.set_empirical_quantiles(
        torch.from_numpy(level_quantiles).to(device),
        torch.from_numpy(increment_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_level),
            torch.from_numpy(train_increment),
            torch.from_numpy(train_future_level),
            torch.from_numpy(train_future_increment),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_level),
            torch.from_numpy(val_increment),
            torch.from_numpy(val_future_level),
            torch.from_numpy(val_future_increment),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    t0 = time.time()
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": "state_conditioned_encoded_increment",
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        metric_sums: dict[str, float] = {}
        for history_level, history_increment, future_level, future_increment in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.training_loss(
                history_level.to(device),
                history_increment.to(device),
                future_level.to(device),
                future_increment.to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_n = int(history_level.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * batch_n
        train_loss = total / max(count, 1)
        val_loss = eval_loss(model, val_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "elapsed_s": float(time.time() - t0),
        }
        for key, value in metric_sums.items():
            record[f"train_{key}"] = float(value / max(count, 1))
        history_records.append(record)
        is_best = val_loss < best_val
        if is_best:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            save_checkpoint(str(best_path), model, cfg, epoch, best_val, extra=extra)
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}"
            f"{' best' if is_best else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(str(final_path), model, cfg, int(args.epochs), best_val, extra=extra)
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_increment[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=min(int(args.iv_count), int(train_level.shape[-1])),
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "model_coordinate": "state_conditioned_encoded_increment",
        "n_state_vars": int(train_level.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_level.shape),
        "val_shape": list(val_level.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(history_records[-1]["val_loss"] if history_records else float("nan")),
        "sample_smoke": smoke,
        "panel_metadata": panel_metadata,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(make_serializable(history_records), indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(make_serializable(summary), indent=2),
        encoding="utf-8",
    )
    (output_dir / "args.json").write_text(json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
