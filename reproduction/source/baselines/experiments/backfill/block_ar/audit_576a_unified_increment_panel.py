#!/usr/bin/env python
"""576a: audit unified IV+factor future-increment data framing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.audit_569a_factor_panel_readiness import (  # noqa: E402
    window_boundary_summary,
)


@dataclass(frozen=True)
class UnifiedVariableSpec:
    name: str
    source_column: str
    source_index: int
    transform: str
    reference_increment_column: str | None = None
    reference_increment_index: int | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class UnifiedIncrementBlock:
    history_state: np.ndarray
    future_state: np.ndarray
    future_increment: np.ndarray
    reconstructed_future_state: np.ndarray
    indices: np.ndarray
    specs: list[UnifiedVariableSpec]


def is_return_like_column(column: str) -> bool:
    return column.endswith("_logret") or column.endswith("_diff") or "return:" in column


def _base_factor_name(column: str) -> str:
    return column.removeprefix("factor:")


def build_unified_variable_specs(
    columns: list[str],
    *,
    panel: np.ndarray | None = None,
    iv_count: int = 25,
    eps: float = 1e-8,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "log_level",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> list[UnifiedVariableSpec]:
    """Build one state-variable list without duplicated level/return targets."""
    if len(columns) < iv_count:
        raise ValueError("columns shorter than iv_count")
    if positive_level_policy not in {"reference_based", "observed_positive"}:
        raise ValueError(
            "positive_level_policy must be 'reference_based' or 'observed_positive'"
        )
    if iv_transform not in {"log_level", "bounded_logit"}:
        raise ValueError("iv_transform must be 'log_level' or 'bounded_logit'")
    if iv_transform == "bounded_logit" and not iv_lower_bound < iv_upper_bound:
        raise ValueError("iv_lower_bound must be less than iv_upper_bound")
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    specs: list[UnifiedVariableSpec] = []

    for idx in range(iv_count):
        column = columns[idx]
        if not column.startswith("iv:"):
            raise ValueError(f"expected IV column at {idx}, got {column}")
        specs.append(
            UnifiedVariableSpec(
                name=column,
                source_column=column,
                source_index=idx,
                transform=iv_transform,
                lower_bound=float(iv_lower_bound) if iv_transform == "bounded_logit" else None,
                upper_bound=float(iv_upper_bound) if iv_transform == "bounded_logit" else None,
            )
        )

    for idx in range(iv_count, len(columns)):
        column = columns[idx]
        if is_return_like_column(column):
            continue
        base = _base_factor_name(column)
        logret_name = f"factor:{base}_logret"
        diff_name = f"factor:{base}_diff"
        values = None if panel is None else np.asarray(panel[:, idx], dtype=np.float64)
        observed_positive = (
            positive_level_policy == "observed_positive"
            and values is not None
            and np.nanmin(values) > float(eps)
        )
        if logret_name in name_to_idx or observed_positive:
            if values is not None and np.nanmin(values) < -float(eps):
                transform = "diff_level"
                reference_column = None
                reference_index = None
            else:
                transform = "log_level"
                reference_column = logret_name if logret_name in name_to_idx else None
                reference_index = name_to_idx.get(logret_name)
        elif diff_name in name_to_idx:
            transform = "diff_level"
            reference_column = diff_name
            reference_index = name_to_idx[diff_name]
        else:
            transform = "diff_level"
            reference_column = None
            reference_index = None
        specs.append(
            UnifiedVariableSpec(
                name=column,
                source_column=column,
                source_index=idx,
                transform=transform,
                reference_increment_column=reference_column,
                reference_increment_index=reference_index,
            )
        )
    return specs


def clean_nonpositive_log_level_factors(
    panel: np.ndarray,
    columns: list[str],
    *,
    iv_count: int = 25,
    eps: float = 1e-8,
    positive_level_policy: str = "reference_based",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean zero-filled nonnegative level series before log-coordinate use.

    Negative-valued series are not forced into log coordinates. They are left
    unchanged and later assigned a difference-coordinate transform by
    `build_unified_variable_specs`.
    """
    if positive_level_policy not in {"reference_based", "observed_positive"}:
        raise ValueError(
            "positive_level_policy must be 'reference_based' or 'observed_positive'"
        )
    cleaned = np.asarray(panel, dtype=np.float32).copy()
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    cleaned_columns: dict[str, int] = {}
    diff_fallback_columns: list[str] = []
    for idx in range(iv_count, len(columns)):
        column = columns[idx]
        if is_return_like_column(column):
            continue
        base = _base_factor_name(column)
        has_log_reference = f"factor:{base}_logret" in name_to_idx
        if not has_log_reference and positive_level_policy != "observed_positive":
            continue
        values = cleaned[:, idx].astype(np.float64)
        if np.nanmin(values) < -float(eps):
            diff_fallback_columns.append(column)
            continue
        bad = ~np.isfinite(values) | (values <= float(eps))
        if not np.any(bad):
            continue
        good_idx = np.flatnonzero(~bad)
        if good_idx.size == 0:
            raise ValueError(f"cannot clean {column}: no positive finite values")
        filled = values.copy()
        first = int(good_idx[0])
        filled[:first] = filled[first]
        last_value = float(filled[first])
        for row in range(first + 1, filled.shape[0]):
            if np.isfinite(filled[row]) and filled[row] > float(eps):
                last_value = float(filled[row])
            else:
                filled[row] = last_value
        cleaned[:, idx] = filled.astype(np.float32)
        cleaned_columns[column] = int(np.sum(bad))
    return cleaned, {
        "enabled": True,
        "eps": float(eps),
        "cleaned_columns": cleaned_columns,
        "diff_fallback_columns": diff_fallback_columns,
        "n_cleaned_values": int(sum(cleaned_columns.values())),
        "n_cleaned_columns": int(len(cleaned_columns)),
        "n_diff_fallback_columns": int(len(diff_fallback_columns)),
    }


def _state_from_panel(
    panel: np.ndarray, specs: list[UnifiedVariableSpec]
) -> np.ndarray:
    panel = np.asarray(panel, dtype=np.float64)
    state = np.stack([panel[:, spec.source_index] for spec in specs], axis=1)
    return state.astype(np.float64)


def encode_state(
    state: np.ndarray,
    specs: list[UnifiedVariableSpec],
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != len(specs):
        raise ValueError("state last dimension must match specs")
    encoded = np.empty_like(state, dtype=np.float64)
    for idx, spec in enumerate(specs):
        values = state[..., idx]
        if spec.transform == "log_level":
            encoded[..., idx] = np.log(np.maximum(values, eps))
        elif spec.transform == "bounded_logit":
            if spec.lower_bound is None or spec.upper_bound is None:
                raise ValueError(f"bounded_logit spec {spec.name} requires bounds")
            lower = float(spec.lower_bound)
            upper = float(spec.upper_bound)
            width = upper - lower
            unit = (values - lower) / max(width, eps)
            unit = np.clip(unit, eps, 1.0 - eps)
            encoded[..., idx] = np.log(unit) - np.log1p(-unit)
        elif spec.transform == "diff_level":
            encoded[..., idx] = values
        else:
            raise ValueError(f"unknown transform {spec.transform}")
    return encoded


def decode_state(encoded: np.ndarray, specs: list[UnifiedVariableSpec]) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.shape[-1] != len(specs):
        raise ValueError("encoded last dimension must match specs")
    state = np.empty_like(encoded, dtype=np.float64)
    for idx, spec in enumerate(specs):
        values = encoded[..., idx]
        if spec.transform == "log_level":
            state[..., idx] = np.exp(values)
        elif spec.transform == "bounded_logit":
            if spec.lower_bound is None or spec.upper_bound is None:
                raise ValueError(f"bounded_logit spec {spec.name} requires bounds")
            lower = float(spec.lower_bound)
            upper = float(spec.upper_bound)
            clipped = np.clip(values, -60.0, 60.0)
            unit = 1.0 / (1.0 + np.exp(-clipped))
            state[..., idx] = lower + (upper - lower) * unit
        elif spec.transform == "diff_level":
            state[..., idx] = values
        else:
            raise ValueError(f"unknown transform {spec.transform}")
    return state


def build_unified_increment_block(
    panel: np.ndarray,
    columns: list[str],
    indices: Iterable[int],
    *,
    history_len: int,
    future_len: int,
    iv_count: int = 25,
    positive_level_policy: str = "reference_based",
    iv_transform: str = "log_level",
    iv_lower_bound: float = 1e-4,
    iv_upper_bound: float = 1.0,
) -> UnifiedIncrementBlock:
    panel = np.asarray(panel, dtype=np.float64)
    specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=iv_count,
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    state_panel = _state_from_panel(panel, specs)
    encoded_panel = encode_state(state_panel, specs)

    history_state: list[np.ndarray] = []
    future_state: list[np.ndarray] = []
    future_increment: list[np.ndarray] = []
    reconstructed: list[np.ndarray] = []
    index_array = np.asarray(list(indices), dtype=np.int64)
    for raw_idx in index_array:
        start = int(raw_idx)
        hist_start = start
        hist_end = start + int(history_len)
        fut_end = hist_end + int(future_len)
        if hist_start < 0 or fut_end > panel.shape[0]:
            raise ValueError(f"window {start} exceeds panel bounds")

        hist_state = state_panel[hist_start:hist_end]
        fut_state = state_panel[hist_end:fut_end]
        encoded_path = encoded_panel[hist_end - 1 : fut_end]
        increments = np.diff(encoded_path, axis=0)
        reconstructed_state = decode_state(
            encoded_panel[hist_end - 1][None, :] + np.cumsum(increments, axis=0),
            specs,
        )

        history_state.append(hist_state)
        future_state.append(fut_state)
        future_increment.append(increments)
        reconstructed.append(reconstructed_state)

    return UnifiedIncrementBlock(
        history_state=np.asarray(history_state, dtype=np.float32),
        future_state=np.asarray(future_state, dtype=np.float32),
        future_increment=np.asarray(future_increment, dtype=np.float32),
        reconstructed_future_state=np.asarray(reconstructed, dtype=np.float32),
        indices=index_array,
        specs=specs,
    )


def _finite_rate(array: np.ndarray) -> float:
    return float(np.isfinite(array).mean()) if array.size else float("nan")


def _reference_increment_errors(
    panel: np.ndarray,
    block: UnifiedIncrementBlock,
    *,
    history_len: int,
    future_len: int,
) -> dict[str, Any]:
    errors: list[float] = []
    by_column: dict[str, float] = {}
    for spec_idx, spec in enumerate(block.specs):
        if (
            spec.reference_increment_index is None
            or spec.reference_increment_column is None
        ):
            continue
        ref_chunks = []
        for start in block.indices:
            hist_end = int(start) + int(history_len)
            ref_chunks.append(
                panel[
                    hist_end : hist_end + int(future_len),
                    spec.reference_increment_index,
                ]
            )
        reference = np.asarray(ref_chunks, dtype=np.float64)
        diff = np.abs(reference - block.future_increment[..., spec_idx])
        max_error = float(np.nanmax(diff)) if diff.size else float("nan")
        by_column[spec.name] = max_error
        errors.append(max_error)
    finite = np.asarray(
        [value for value in errors if np.isfinite(value)], dtype=np.float64
    )
    return {
        "reference_increment_count": int(finite.size),
        "reference_increment_max_abs_error": (
            float(np.max(finite)) if finite.size else float("nan")
        ),
        "reference_increment_median_abs_error": (
            float(np.median(finite)) if finite.size else float("nan")
        ),
        "reference_increment_max_abs_error_by_state": by_column,
    }


def summarize_unified_increment_block(
    panel: np.ndarray,
    block: UnifiedIncrementBlock,
    *,
    columns: list[str],
    history_len: int,
    future_len: int,
    iv_count: int = 25,
) -> dict[str, Any]:
    target_names = [spec.name for spec in block.specs]
    duplicate_target_names = sorted(
        {name for name in target_names if target_names.count(name) > 1}
    )
    return_like_targets = [name for name in target_names if is_return_like_column(name)]
    reconstruction_error = np.abs(block.future_state - block.reconstructed_future_state)
    floor_hits = 0
    for spec_idx, spec in enumerate(block.specs):
        if spec.transform == "log_level":
            floor_hits += int((block.future_state[..., spec_idx] <= 1e-8).sum())
    summary = {
        "n_windows": int(block.history_state.shape[0]),
        "history_shape": list(block.history_state.shape),
        "future_state_shape": list(block.future_state.shape),
        "future_increment_shape": list(block.future_increment.shape),
        "reconstructed_future_state_shape": list(
            block.reconstructed_future_state.shape
        ),
        "source_panel_channel_count": int(len(columns)),
        "target_state_channel_count": int(len(block.specs)),
        "iv_state_count": int(sum(spec.name.startswith("iv:") for spec in block.specs)),
        "factor_state_count": int(
            sum(spec.name.startswith("factor:") for spec in block.specs)
        ),
        "no_duplicate_target_names": not duplicate_target_names,
        "duplicate_target_names": duplicate_target_names,
        "no_return_like_targets": not return_like_targets,
        "return_like_targets": return_like_targets,
        "finite_history_rate": _finite_rate(block.history_state),
        "finite_future_rate": _finite_rate(block.future_state),
        "finite_increment_rate": _finite_rate(block.future_increment),
        "finite_reconstruction_rate": _finite_rate(block.reconstructed_future_state),
        "reconstruction_max_abs_error": float(np.nanmax(reconstruction_error)),
        "reconstruction_mean_abs_error": float(np.nanmean(reconstruction_error)),
        "log_transform_floor_hits": int(floor_hits),
        "index_start": int(block.indices[0]) if block.indices.size else None,
        "index_end": int(block.indices[-1]) if block.indices.size else None,
        "history_len": int(history_len),
        "future_len": int(future_len),
        "iv_count_argument": int(iv_count),
    }
    summary.update(
        _reference_increment_errors(
            np.asarray(panel, dtype=np.float64),
            block,
            history_len=history_len,
            future_len=future_len,
        )
    )
    return summary


def build_audit_report(
    *,
    history_len: int,
    future_lens: Iterable[int],
    test_start: int,
    val_size: int,
    iv_count: int = 25,
    clean_nonpositive_log_levels: bool = False,
    positive_level_policy: str = "reference_based",
) -> dict[str, Any]:
    panel, columns, dates = load_aligned_iv_factor_panel()
    cleaning_report: dict[str, Any] = {"enabled": False}
    if clean_nonpositive_log_levels:
        panel, cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=iv_count,
            positive_level_policy=positive_level_policy,
        )
    specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=iv_count,
        positive_level_policy=positive_level_policy,
    )
    horizons: dict[str, Any] = {}
    for future_len in future_lens:
        train_indices, val_indices = official_train_val_indices(
            test_start=test_start,
            val_size=val_size,
            history_len=history_len,
            future_len=int(future_len),
        )
        train_block = build_unified_increment_block(
            panel,
            columns,
            train_indices,
            history_len=history_len,
            future_len=int(future_len),
            iv_count=iv_count,
            positive_level_policy=positive_level_policy,
        )
        val_block = build_unified_increment_block(
            panel,
            columns,
            val_indices,
            history_len=history_len,
            future_len=int(future_len),
            iv_count=iv_count,
            positive_level_policy=positive_level_policy,
        )
        horizons[str(int(future_len))] = {
            "train": summarize_unified_increment_block(
                panel,
                train_block,
                columns=columns,
                history_len=history_len,
                future_len=int(future_len),
                iv_count=iv_count,
            ),
            "val": summarize_unified_increment_block(
                panel,
                val_block,
                columns=columns,
                history_len=history_len,
                future_len=int(future_len),
                iv_count=iv_count,
            ),
        }
    return {
        "scope": "576a_unified_increment_panel_framing_audit",
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "n_observations": int(panel.shape[0]),
        "source_panel_shape": list(panel.shape),
        "source_panel_columns": list(columns),
        "state_variable_count": int(len(specs)),
        "state_variables": [asdict(spec) for spec in specs],
        "positive_level_policy": positive_level_policy,
        "cleaning": cleaning_report,
        "split_boundaries": window_boundary_summary(
            n_obs=int(panel.shape[0]),
            history_len=history_len,
            future_lens=[int(value) for value in future_lens],
            test_start=test_start,
            val_size=val_size,
        ),
        "horizons": horizons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_lens", type=int, nargs="+", default=[30, 60, 90, 152])
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true")
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_audit_report(
        history_len=args.history_len,
        future_lens=args.future_lens,
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        clean_nonpositive_log_levels=args.clean_nonpositive_log_levels,
        positive_level_policy=args.positive_level_policy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(report["split_boundaries"]), indent=2))
    for horizon, payload in report["horizons"].items():
        val = payload["val"]
        print(
            f"h={horizon}: state_vars={val['target_state_channel_count']} "
            f"val_windows={val['n_windows']} recon_max={val['reconstruction_max_abs_error']:.3e} "
            f"ref_max={val['reference_increment_max_abs_error']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
