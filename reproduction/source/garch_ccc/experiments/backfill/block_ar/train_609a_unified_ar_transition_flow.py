#!/usr/bin/env python
"""609a: train a generic 510a-style AR transition flow on IV or joint state panels."""

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
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    UnifiedIncrementBlock,
    build_unified_increment_block,
    clean_nonpositive_log_level_factors,
    decode_state,
    encode_state,
)


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "source_column": spec.source_column,
        "source_index": int(spec.source_index),
        "transform": spec.transform,
        "reference_increment_column": spec.reference_increment_column,
        "reference_increment_index": (
            None if spec.reference_increment_index is None else int(spec.reference_increment_index)
        ),
        "lower_bound": getattr(spec, "lower_bound", None),
        "upper_bound": getattr(spec, "upper_bound", None),
    }


def select_scope(
    block: UnifiedIncrementBlock,
    scope: str,
    iv_count: int,
    value_coordinate: str = "raw",
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if value_coordinate == "raw":
        history = block.history_state
        future = block.future_state
    elif value_coordinate == "encoded":
        history = encode_state(block.history_state, block.specs).astype(np.float32)
        future = encode_state(block.future_state, block.specs).astype(np.float32)
    else:
        raise ValueError(f"unknown value_coordinate {value_coordinate!r}")
    if scope == "joint38":
        return history, future, block.specs
    if scope == "iv_only":
        return history[..., :iv_count], future[..., :iv_count], block.specs[:iv_count]
    raise ValueError(f"unknown state_scope {scope!r}")


def fit_empirical_quantiles(
    history: np.ndarray,
    future: np.ndarray,
    *,
    n_quantiles: int,
    cdf_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate(
        [
            np.asarray(history, dtype=np.float64).reshape(-1, history.shape[-1]),
            np.asarray(future, dtype=np.float64).reshape(-1, future.shape[-1]),
        ],
        axis=0,
    )
    levels = np.linspace(float(cdf_eps), 1.0 - float(cdf_eps), int(n_quantiles), dtype=np.float64)
    quantiles = np.quantile(flat, levels, axis=0).T.astype(np.float32)
    return quantiles, levels.astype(np.float32)


def build_blocks(args: argparse.Namespace) -> tuple[list[str], Any, UnifiedIncrementBlock, UnifiedIncrementBlock]:
    panel, columns, dates = load_aligned_iv_factor_panel()
    cleaning_report: dict[str, Any] = {"enabled": False}
    if args.clean_nonpositive_log_levels:
        panel, cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
        )
    train_indices, val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(args.history_len),
        future_len=int(args.future_len),
    )
    if args.max_train_windows > 0:
        train_indices = train_indices[-int(args.max_train_windows) :]
    train_block = build_unified_increment_block(
        panel,
        columns,
        train_indices,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        iv_count=int(args.iv_count),
    )
    val_block = build_unified_increment_block(
        panel,
        columns,
        val_indices,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        iv_count=int(args.iv_count),
    )
    metadata = {
        "dates_start": str(dates[0]) if len(dates) else None,
        "dates_end": str(dates[-1]) if len(dates) else None,
        "source_columns": columns,
        "cleaning_report": cleaning_report,
        "train_indices_start": int(train_indices[0]) if len(train_indices) else None,
        "train_indices_end": int(train_indices[-1]) if len(train_indices) else None,
        "val_indices_start": int(val_indices[0]) if len(val_indices) else None,
        "val_indices_end": int(val_indices[-1]) if len(val_indices) else None,
    }
    return columns, metadata, train_block, val_block


def eval_loss(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for history, future in loader:
            history = history.to(device)
            future = future.to(device)
            loss, _metrics = model.training_loss(history, future)
            total += float(loss.item()) * int(history.shape[0])
            count += int(history.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def sample_smoke(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    history: np.ndarray,
    *,
    samples: int,
    steps: int,
    chunk_size: int,
    device: torch.device,
    iv_count: int,
    specs: list[Any] | None = None,
    value_coordinate: str = "raw",
) -> dict[str, Any]:
    model.eval()
    hist = torch.from_numpy(history[: min(4, history.shape[0])]).to(device)
    out = model.sample_batched(hist, n_samples=int(samples), n_steps=int(steps), chunk_size=int(chunk_size))
    arr = out.detach().cpu().numpy()
    report_arr = decode_state(arr, specs).astype(np.float32) if value_coordinate == "encoded" and specs is not None else arr
    report: dict[str, Any] = {
        "sample_shape": list(arr.shape),
        "finite_rate": float(np.isfinite(arr).mean()),
        "decoded": bool(value_coordinate == "encoded"),
        "iv_min": float(np.nanmin(report_arr[..., :iv_count])),
        "iv_max": float(np.nanmax(report_arr[..., :iv_count])),
    }
    if report_arr.shape[-1] > iv_count:
        report.update(
            {
                "factor_min": float(np.nanmin(report_arr[..., iv_count:])),
                "factor_max": float(np.nanmax(report_arr[..., iv_count:])),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--value_coordinate", choices=["raw", "encoded"], default="raw")
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
    parser.add_argument("--path_source_corr", type=float, default=0.0)
    parser.add_argument("--path_source_ar", type=float, default=0.0)
    parser.add_argument("--conditional_source_scale", action="store_true")
    parser.add_argument("--source_scale_min", type=float, default=0.25)
    parser.add_argument("--source_scale_max", type=float, default=4.0)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale"], default="basic")
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=609)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_history, train_future, train_specs = select_scope(
        train_block,
        args.state_scope,
        int(args.iv_count),
        args.value_coordinate,
    )
    val_history, val_future, val_specs = select_scope(
        val_block,
        args.state_scope,
        int(args.iv_count),
        args.value_coordinate,
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
        conditional_source_scale=bool(args.conditional_source_scale),
        source_scale_min=float(args.source_scale_min),
        source_scale_max=float(args.source_scale_max),
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
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    t0 = time.time()
    extra = {
        "state_scope": args.state_scope,
        "value_coordinate": args.value_coordinate,
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
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
        val_history[: int(args.sample_windows)],
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
        specs=train_specs,
        value_coordinate=args.value_coordinate,
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "value_coordinate": args.value_coordinate,
        "n_state_vars": int(train_history.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_history.shape),
        "val_shape": list(val_history.shape),
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
