#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOCAL_FACTOR_NAMES = ["ret", "price", "slopes", "skews", "levels"]


def iv_channel_names() -> list[str]:
    return [f"iv_{row}_{col}" for row in range(5) for col in range(5)]


def _as_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _check_same_length(arrays: Iterable[tuple[str, np.ndarray]]) -> int:
    lengths = {name: int(array.shape[0]) for name, array in arrays}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"arrays must share the same time length, got {lengths}")
    return unique_lengths.pop()


def fill_missing_by_column(
    panel: np.ndarray,
    *,
    names: Iterable[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    panel = np.asarray(panel, dtype=np.float32)
    if panel.ndim != 2:
        raise ValueError("panel must have shape (time, channels)")
    names = [str(name) for name in names]
    if len(names) != panel.shape[1]:
        raise ValueError("names length must match panel channel count")

    finite_before = np.isfinite(panel)
    filled = panel.copy()
    columns_with_missing: list[str] = []
    missing_counts: dict[str, int] = {}
    for col_idx, name in enumerate(names):
        col = filled[:, col_idx]
        finite_idx = np.flatnonzero(np.isfinite(col))
        missing_count = int((~np.isfinite(col)).sum())
        if missing_count == 0:
            continue
        if finite_idx.size == 0:
            raise ValueError(f"column {name} has no finite values")
        columns_with_missing.append(name)
        missing_counts[name] = missing_count

        first_idx = int(finite_idx[0])
        col[:first_idx] = col[first_idx]
        last_value = float(col[first_idx])
        for row_idx in range(first_idx + 1, col.shape[0]):
            if np.isfinite(col[row_idx]):
                last_value = float(col[row_idx])
            else:
                col[row_idx] = last_value
        filled[:, col_idx] = col

    return filled, {
        "finite_rate_before": float(finite_before.mean()),
        "finite_rate_after": float(np.isfinite(filled).mean()),
        "columns_with_missing": columns_with_missing,
        "missing_counts": missing_counts,
        "method": "per_column_backfill_then_forward_fill",
    }


def build_local_factor_panel(
    surface: np.ndarray,
    ret: np.ndarray,
    price: np.ndarray,
    slopes: np.ndarray,
    skews: np.ndarray,
    levels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    surface = _as_float_array(surface, name="surface")
    if surface.ndim != 3 or surface.shape[1:] != (5, 5):
        raise ValueError("surface must have shape (time, 5, 5)")
    factors = [
        _as_float_array(ret, name="ret"),
        _as_float_array(price, name="price"),
        _as_float_array(slopes, name="slopes"),
        _as_float_array(skews, name="skews"),
        _as_float_array(levels, name="levels"),
    ]
    _check_same_length([("surface", surface), *zip(LOCAL_FACTOR_NAMES, factors, strict=True)])
    panel = np.concatenate([surface.reshape(surface.shape[0], 25), np.stack(factors, axis=1)], axis=1)
    return panel.astype(np.float32), iv_channel_names() + LOCAL_FACTOR_NAMES


def build_broad_factor_panel(
    surface: np.ndarray,
    factor_levels: np.ndarray,
    factor_returns: np.ndarray,
    *,
    level_names: Iterable[str],
    return_names: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    surface = _as_float_array(surface, name="surface")
    factor_levels = _as_float_array(factor_levels, name="factor_levels")
    factor_returns = _as_float_array(factor_returns, name="factor_returns")
    if surface.ndim != 3 or surface.shape[1:] != (5, 5):
        raise ValueError("surface must have shape (time, 5, 5)")
    if factor_levels.ndim != 2:
        raise ValueError("factor_levels must have shape (time, channels)")
    if factor_returns.ndim != 2:
        raise ValueError("factor_returns must have shape (time, channels)")
    _check_same_length(
        [
            ("surface", surface),
            ("factor_levels", factor_levels),
            ("factor_returns", factor_returns),
        ]
    )
    level_names = [str(name) for name in level_names]
    return_names = [str(name) for name in return_names]
    if len(level_names) != factor_levels.shape[1]:
        raise ValueError("level_names length must match factor_levels channel count")
    if len(return_names) != factor_returns.shape[1]:
        raise ValueError("return_names length must match factor_returns channel count")

    panel = np.concatenate(
        [surface.reshape(surface.shape[0], 25), factor_levels, factor_returns],
        axis=1,
    )
    names = (
        iv_channel_names()
        + [f"factor_level:{name}" for name in level_names]
        + [f"factor_return:{name}" for name in return_names]
    )
    return panel.astype(np.float32), names


def window_boundary_summary(
    *,
    n_obs: int,
    history_len: int,
    future_lens: Iterable[int],
    test_start: int,
    val_size: int,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for future_len in future_lens:
        future_len = int(future_len)
        max_train_idx = int(test_start) - int(history_len) - future_len
        val_start = max_train_idx - int(val_size)
        train_windows = max(0, val_start)
        val_windows = int(val_size) if val_start >= 0 else max(0, max_train_idx)
        first_val_start = val_start if val_start >= 0 else 0
        last_val_start = max_train_idx - 1 if val_windows > 0 else None
        last_val_future_end = (
            int(last_val_start + history_len + future_len + 1)
            if last_val_start is not None
            else None
        )
        summary[str(future_len)] = {
            "history_len": int(history_len),
            "future_len": future_len,
            "n_obs": int(n_obs),
            "test_start": int(test_start),
            "max_train_idx_exclusive": int(max_train_idx),
            "train_windows": int(train_windows),
            "val_windows": int(val_windows),
            "first_val_start_index": int(first_val_start) if val_windows > 0 else None,
            "last_val_start_index": int(last_val_start) if last_val_start is not None else None,
            "last_val_future_end_exclusive": last_val_future_end,
            "no_test_leakage": bool(last_val_future_end is not None and last_val_future_end <= test_start),
            "fits_available_history": bool(test_start + future_len <= n_obs),
        }
    return summary


def _finite_rate(panel: np.ndarray) -> float:
    return float(np.isfinite(panel).mean())


def _read_iv_dates(path: str) -> tuple[str | None, str | None, int | None]:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return None, None, None
    try:
        import pandas as pd

        frame = pd.read_parquet(parquet_path)
    except Exception:
        return None, None, None
    if "date" in frame.columns:
        dates = frame["date"]
    elif "Date" in frame.columns:
        dates = frame["Date"]
    else:
        dates = frame.index
    if len(dates) == 0:
        return None, None, 0
    return str(dates.iloc[0]), str(dates.iloc[-1]), int(len(dates))


def build_readiness_report(
    *,
    iv_data_path: str,
    factor_data_path: str,
    iv_dates_path: str,
    history_len: int,
    future_lens: Iterable[int],
    test_start: int,
    val_size: int,
) -> dict[str, Any]:
    iv_raw = np.load(iv_data_path)
    surface = iv_raw["surface"].astype(np.float32)
    local_panel, local_names = build_local_factor_panel(
        surface,
        iv_raw["ret"],
        iv_raw["price"],
        iv_raw["slopes"],
        iv_raw["skews"],
        iv_raw["levels"],
    )

    factor_raw = np.load(factor_data_path, allow_pickle=True)
    factor_obs = int(factor_raw["levels"].shape[0])
    aligned_obs = min(int(surface.shape[0]), factor_obs)
    level_names = factor_raw["level_columns"].astype(str).tolist()
    return_names = factor_raw["return_columns"].astype(str).tolist()
    factor_levels_filled, level_missing = fill_missing_by_column(
        factor_raw["levels"],
        names=level_names,
    )
    factor_returns_filled, return_missing = fill_missing_by_column(
        factor_raw["returns"],
        names=return_names,
    )
    broad_panel, broad_names = build_broad_factor_panel(
        surface[:aligned_obs],
        factor_levels_filled[:aligned_obs],
        factor_returns_filled[:aligned_obs],
        level_names=level_names,
        return_names=return_names,
    )
    factor_dates = factor_raw["dates"]
    iv_date_start, iv_date_end, iv_date_count = _read_iv_dates(iv_dates_path)

    return {
        "scope": "factor_panel_readiness_no_training",
        "iv_data_path": iv_data_path,
        "factor_data_path": factor_data_path,
        "local_factor_names": LOCAL_FACTOR_NAMES,
        "broad_factor_level_names": level_names,
        "broad_factor_return_names": return_names,
        "iv": {
            "surface_shape": list(surface.shape),
            "date_start": iv_date_start,
            "date_end": iv_date_end,
            "date_count": iv_date_count,
        },
        "factor_panel": {
            "date_start": str(factor_dates[0]),
            "date_end": str(factor_dates[-1]),
            "date_count": int(len(factor_dates)),
            "factor_only_observations_after_truncation": int(factor_obs - aligned_obs),
            "level_missing": level_missing,
            "return_missing": return_missing,
        },
        "local_panel": {
            "shape": list(local_panel.shape),
            "channel_count": int(local_panel.shape[1]),
            "finite_rate": _finite_rate(local_panel),
            "channel_names": local_names,
        },
        "broad_panel": {
            "shape": list(broad_panel.shape),
            "channel_count": int(broad_panel.shape[1]),
            "finite_rate": _finite_rate(broad_panel),
            "channel_names": broad_names,
        },
        "window_boundaries": window_boundary_summary(
            n_obs=aligned_obs,
            history_len=history_len,
            future_lens=future_lens,
            test_start=test_start,
            val_size=val_size,
        ),
        "interpretation": {
            "local_stack": "25 IV channels plus 5 attached SPX/IV summary factors",
            "broad_stack": "25 IV channels plus 13 factor levels plus 13 factor returns/diffs",
            "limitation": "This audit proves mechanical panel readiness only; it does not train or validate a joint factor scenario law.",
        },
    }


def _write_markdown(report: dict[str, Any], output_md: str) -> None:
    lines = [
        "# 569a Factor Panel Readiness",
        "",
        "## Factor Scope",
        "",
        f"- local panel: `{report['local_panel']['channel_count']}` channels, shape `{tuple(report['local_panel']['shape'])}`",
        f"- broad panel: `{report['broad_panel']['channel_count']}` channels, shape `{tuple(report['broad_panel']['shape'])}`",
        f"- broad factor levels: `{', '.join(report['broad_factor_level_names'])}`",
        f"- broad factor returns/diffs: `{', '.join(report['broad_factor_return_names'])}`",
        "",
        "## Horizon Window Readiness",
        "",
        "| future days | train windows | val windows | last val future end | no test leakage |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for future_len, item in report["window_boundaries"].items():
        lines.append(
            f"| `{future_len}` | `{item['train_windows']}` | `{item['val_windows']}` | "
            f"`{item['last_val_future_end_exclusive']}` | `{item['no_test_leakage']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Factor stacking is mechanically ready for local and broad panels.",
            "- Long-horizon labels can be framed without test leakage under the official split.",
            "- This is not evidence that a trained joint factor law will pass the 11-suite.",
        ]
    )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="569a factor-panel readiness audit")
    parser.add_argument("--iv_data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--factor_data_path", default="data/multi_factor_data.npz")
    parser.add_argument(
        "--iv_dates_path",
        default="data/spx_vol_surface_history_full_data_fixed.parquet",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_lens", type=int, nargs="+", default=[30, 60, 90, 252])
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument(
        "--output_json",
        default="results/autoresearch/569a_factor_panel_readiness/readiness.json",
    )
    parser.add_argument(
        "--output_md",
        default="results/autoresearch/569a_factor_panel_readiness/readiness.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_readiness_report(
        iv_data_path=args.iv_data_path,
        factor_data_path=args.factor_data_path,
        iv_dates_path=args.iv_dates_path,
        history_len=args.history_len,
        future_lens=args.future_lens,
        test_start=args.test_start,
        val_size=args.val_size,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, str(output_md))
    print(json.dumps({"status": "ok", "output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
