#!/usr/bin/env python
"""628a: train a generic AR transition flow on encoded daily changes."""

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

from diffusion.block_ar.generic_empirical_score_transition_flow_matching import (  # noqa: E402
    GenericEmpiricalScoreTransitionFMConfig,
    GenericEmpiricalScoreTransitionFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (
    official_train_val_indices,
)  # noqa: E402
from experiments.backfill.block_ar._panel_law_535_utils import (
    load_aligned_iv_factor_panel,
)  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    IncrementCoordinateBlock,
    build_increment_coordinate_block,
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
    eval_loss,
    fit_empirical_quantiles,
)


def select_increment_scope(
    block: IncrementCoordinateBlock,
    scope: str,
    iv_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    if scope == "joint38":
        return (
            block.history_increment,
            block.future_increment,
            block.history_state,
            block.future_state,
            block.specs,
        )
    if scope == "iv_only":
        return (
            block.history_increment[..., :iv_count],
            block.future_increment[..., :iv_count],
            block.history_state[..., :iv_count],
            block.future_state[..., :iv_count],
            block.specs[:iv_count],
        )
    raise ValueError(f"unknown state_scope {scope!r}")


def build_blocks(
    args: argparse.Namespace,
) -> tuple[
    list[str], dict[str, Any], IncrementCoordinateBlock, IncrementCoordinateBlock
]:
    panel, columns, dates = load_aligned_iv_factor_panel()
    positive_level_policy = getattr(args, "positive_level_policy", "reference_based")
    iv_transform = getattr(args, "iv_transform", "log_level")
    iv_lower_bound = float(getattr(args, "iv_lower_bound", 1e-4))
    iv_upper_bound = float(getattr(args, "iv_upper_bound", 1.0))
    cleaning_report: dict[str, Any] = {"enabled": False}
    if args.clean_nonpositive_log_levels:
        panel, cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
        )
    train_indices, val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(args.history_len),
        future_len=int(args.future_len),
    )
    if int(args.max_train_windows) > 0:
        train_indices = train_indices[-int(args.max_train_windows) :]
    train_block = build_increment_coordinate_block(
        panel,
        columns,
        train_indices,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        iv_count=int(args.iv_count),
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    val_block = build_increment_coordinate_block(
        panel,
        columns,
        val_indices,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        iv_count=int(args.iv_count),
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    metadata = {
        "dates_start": str(dates[0]) if len(dates) else None,
        "dates_end": str(dates[-1]) if len(dates) else None,
        "source_columns": columns,
        "positive_level_policy": positive_level_policy,
        "iv_transform": iv_transform,
        "iv_lower_bound": iv_lower_bound,
        "iv_upper_bound": iv_upper_bound,
        "cleaning_report": cleaning_report,
        "train_indices_start": int(train_indices[0]) if len(train_indices) else None,
        "train_indices_end": int(train_indices[-1]) if len(train_indices) else None,
        "val_indices_start": int(val_indices[0]) if len(val_indices) else None,
        "val_indices_end": int(val_indices[-1]) if len(val_indices) else None,
    }
    return columns, metadata, train_block, val_block


@torch.no_grad()
def sample_smoke(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    history_increment: np.ndarray,
    history_state: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    steps: int,
    chunk_size: int,
    device: torch.device,
    iv_count: int,
) -> dict[str, Any]:
    model.eval()
    n = min(4, int(history_increment.shape[0]))
    hist = torch.from_numpy(history_increment[:n]).to(device)
    sampled_increment = model.sample_batched(
        hist,
        n_samples=int(samples),
        n_steps=int(steps),
        chunk_size=int(chunk_size),
    )
    increments = sampled_increment.detach().cpu().numpy()
    states = reconstruct_state_from_increments(
        history_state[:n, -1, :], increments, specs
    )
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
    parser.add_argument(
        "--state_scope", choices=["iv_only", "joint38"], default="joint38"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
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
    parser.add_argument("--path_source_corr", type=float, default=0.0)
    parser.add_argument("--path_source_ar", type=float, default=0.0)
    parser.add_argument(
        "--prefix_feature_mode", choices=["basic", "scale"], default="basic"
    )
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=628)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_history, train_future, train_state, _train_future_state, train_specs = (
        select_increment_scope(
            train_block,
            args.state_scope,
            int(args.iv_count),
        )
    )
    val_history, val_future, val_state, _val_future_state, val_specs = (
        select_increment_scope(
            val_block,
            args.state_scope,
            int(args.iv_count),
        )
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    value_quantiles, quantile_levels = fit_empirical_quantiles(
        train_history,
        train_future,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    cfg = GenericEmpiricalScoreTransitionFMConfig(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        n_cells=int(train_history.shape[-1]),
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
        path_source_corr=float(args.path_source_corr),
        path_source_ar=float(args.path_source_ar),
        conditioning_mode="prefix",
    )
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg).to(device)
    model.set_empirical_quantiles(
        torch.from_numpy(value_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_history), torch.from_numpy(train_future)),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_history), torch.from_numpy(val_future)),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    t0 = time.time()
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": "encoded_increment",
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "positive_level_policy": args.positive_level_policy,
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        metric_sums: dict[str, float] = {}
        for history, future in train_loader:
            history = history.to(device)
            future = future.to(device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.training_loss(history, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_n = int(history.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
            for key, value in metrics.items():
                metric_sums[key] = (
                    metric_sums.get(key, 0.0) + float(value.item()) * batch_n
                )
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
    save_checkpoint(
        str(final_path), model, cfg, int(args.epochs), best_val, extra=extra
    )
    smoke = sample_smoke(
        model,
        val_history[: int(args.sample_windows)],
        val_state[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=min(int(args.iv_count), int(train_history.shape[-1])),
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "model_coordinate": "encoded_increment",
        "n_state_vars": int(train_history.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_history.shape),
        "val_shape": list(val_history.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(
            history_records[-1]["val_loss"] if history_records else float("nan")
        ),
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
    (output_dir / "args.json").write_text(
        json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8"
    )
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
