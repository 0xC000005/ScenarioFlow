#!/usr/bin/env python
"""Evaluate the promoted SNI joint checkpoint on current-panel paper metrics."""

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

sys.path.insert(0, ".")

from analysis_code.window_score_artifact import (  # noqa: E402
    add_window_score_argument,
    maybe_write_window_score_artifact,
)
from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.baselines.evaluate_current_panel_baselines import (  # noqa: E402
    compute_correlation_score,
    compute_crps,
    compute_energy_score,
    compute_factor_ks,
    compute_variogram_score,
    extract_iv_metrics,
    make_block_args,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    panel_daily_changes,
    summarize_joint_quality,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


@torch.no_grad()
def generate_full_panel_samples(
    model: Any,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    *,
    samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    sample_temperature: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, int(history_level.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(history_level.shape[0]))
        sampled_increment = model.sample_batched(
            torch.from_numpy(history_level[start:end]).to(device),
            torch.from_numpy(history_norm[start:end]).to(device),
            torch.from_numpy(center[start:end]).to(device),
            torch.from_numpy(scale[start:end]).to(device),
            drift_feature=torch.from_numpy(drift_feature[start:end]).to(device),
            n_samples=int(samples),
            n_steps=int(n_steps),
            chunk_size=int(chunk_size),
            temperature=float(sample_temperature),
        )
        increment_arr = sampled_increment.detach().cpu().numpy()
        panel_arr = reconstruct_state_from_increments(
            history_raw[start:end, -1, :], increment_arr, specs
        )
        chunks.append(panel_arr.astype(np.float32))
        print(f"  SNI generated windows {end}/{history_level.shape[0]}", flush=True)
    return np.concatenate(chunks, axis=0)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_result_summary(output_dir: Path, result: dict[str, Any]) -> None:
    iv = result["current_panel_iv_metrics"]
    crps = result["panel_b_crps"]
    jq = result["joint_panel_quality"]
    rows = [
        "- scope: current real-VIX 39-state panel",
        "- row: evaluated SNI joint checkpoint with the same paper metrics as the classical baselines",
        f"- checkpoint: `{result['checkpoint']}`",
        "",
        "| Model | CRPS | IV CRPS | Factor CRPS | Energy | Variogram | Corr | "
        "Cov90 | CalErr | Level KS med | Median dev | Width rho | MR ratio | "
        "Econ ratio | Jump KS | Kurtosis | ACF | Factor KS | Tail q99 ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                "sni_joint",
                fmt(crps["overall"], 5),
                fmt(crps["iv_crps"], 5),
                fmt(crps["factor_crps"], 5),
                fmt(result["panel_b_energy_score"]["overall"], 4),
                fmt(result["panel_b_variogram_score"]["variogram_score"], 5),
                fmt(result["panel_b_correlation"]["corr_score"], 3),
                fmt(iv["cov90"], 3),
                fmt(iv["calibration_error"], 3),
                fmt(iv["level_ks_median"], 3),
                fmt(iv["median_abs_dev"], 3),
                fmt(iv["width_rho"], 3),
                fmt(iv["mr_ratio"], 3),
                fmt(iv["economic_ratio"], 3),
                fmt(iv["jump_ks"], 3),
                fmt(iv["kurtosis_ratio"], 3),
                fmt(iv["acf_corr"], 3),
                fmt(jq["factor_delta_ks_mean"], 3),
                fmt(jq["factor_tail_q99_ratio_median"], 3),
            ]
        )
        + " |",
    ]
    write_markdown_summary(output_dir / "summary.md", "Current-panel SNI comparison metrics", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--state_scope", choices=["iv_only", "joint38", "anchor_only"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
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
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=773)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="results/baselines_current_panel/sni_joint")
    add_window_score_argument(parser)
    args = parser.parse_args()

    set_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    if payload.get("state_scope") != "joint38":
        raise RuntimeError(f"expected joint checkpoint, got state_scope={payload.get('state_scope')}")

    history_level, history_norm, center, scale, drift_feature, history_raw, specs, val_block = build_val_block(args, payload)
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n_windows]
    history_norm = history_norm[:n_windows]
    center = center[:n_windows]
    scale = scale[:n_windows]
    drift_feature = drift_feature[:n_windows]
    history_raw = history_raw[:n_windows]
    val_block = type(val_block)(
        history_increment=val_block.history_increment[:n_windows],
        future_increment=val_block.future_increment[:n_windows],
        history_state=val_block.history_state[:n_windows],
        future_state=val_block.future_state[:n_windows],
        indices=val_block.indices[:n_windows],
        specs=val_block.specs,
    )
    if len(specs) != 39:
        raise RuntimeError(f"expected current 39-state SNI checkpoint, got {len(specs)} states")

    block_args = make_block_args(args)
    _columns, metadata, train_block, _current_val_block = build_blocks(block_args)
    train_raw_delta = panel_daily_changes(train_block.history_state, train_block.future_state).reshape(
        -1, len(train_block.specs)
    )
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(n_windows),
        device=device,
        split=args.eval_split,
    )
    history_align = float(
        np.max(
            np.abs(
                val_block.history_state[:, :, : int(args.iv_count)].reshape(n_windows, -1)
                - batch.history_01.detach().cpu().numpy().reshape(n_windows, -1)
            )
        )
    )
    future_align = float(
        np.max(
            np.abs(
                val_block.future_state[:, :, : int(args.iv_count)].reshape(n_windows, -1)
                - batch.future_01.detach().cpu().numpy().reshape(n_windows, -1)
            )
        )
    )
    if history_align > 1e-6 or future_align > 1e-6:
        raise RuntimeError(f"IV alignment failure: history={history_align}, future={future_align}")

    print(f"Generating promoted SNI samples: windows={n_windows}, samples={args.samples}")
    t0 = time.time()
    sample_raw = generate_full_panel_samples(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        sample_temperature=float(args.sample_temperature),
    )
    generation_time = time.time() - t0

    raw_history = val_block.history_state.astype(np.float32)
    raw_future = val_block.future_state.astype(np.float32)
    gt_delta = panel_daily_changes(raw_history, raw_future)
    sample_prev = np.concatenate(
        [
            np.repeat(raw_history[:, None, -1:, :], sample_raw.shape[1], axis=1),
            sample_raw[:, :, :-1, :],
        ],
        axis=2,
    )
    sample_delta = sample_raw - sample_prev

    cond_samples = sample_raw[..., : int(args.iv_count)].reshape(
        n_windows, int(args.samples), sample_raw.shape[2], 5, 5
    )
    hist_norm_np = batch.history_norm.detach().cpu().numpy()[:n_windows]
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(n_windows)}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    iv_results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
        data_path=args.data_path,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        batch_size=int(args.batch_size),
        conditionality_samples=int(args.conditionality_samples),
        conditionality_max_batches=int(args.conditionality_max_batches),
        device=device,
        eval_split=args.eval_split,
    )
    factor_names = [spec.name for spec in specs[int(args.iv_count) :]]
    joint_quality = summarize_joint_quality(
        raw_history,
        raw_future,
        sample_raw,
        factor_names,
        iv_count=int(args.iv_count),
    )
    result = {
        "baseline_name": "sni_joint",
        "display_name": "SNI flow (ours)",
        "mode": "current_realvix_39_state_panel",
        "checkpoint": args.checkpoint,
        "generation_time_s": float(generation_time),
        "n_windows": int(n_windows),
        "n_samples": int(args.samples),
        "state_specs": [asdict(spec) for spec in specs],
        "metadata": metadata,
        "alignment": {
            "history_max_abs_error": history_align,
            "future_max_abs_error": future_align,
        },
        "current_panel_iv": iv_results,
        "current_panel_iv_metrics": extract_iv_metrics(iv_results),
        "joint_panel_quality": joint_quality,
        "panel_b_crps": compute_crps(sample_delta, gt_delta, iv_count=int(args.iv_count)),
        "panel_b_energy_score": compute_energy_score(sample_delta, gt_delta, train_raw_delta),
        "panel_b_variogram_score": compute_variogram_score(sample_delta, gt_delta, train_raw_delta),
        "panel_b_correlation": compute_correlation_score(sample_delta, gt_delta, iv_count=int(args.iv_count)),
        "panel_b_factor_ks": compute_factor_ks(sample_delta, gt_delta, factor_names, iv_count=int(args.iv_count)),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maybe_write_window_score_artifact(
        enabled=bool(args.save_window_scores),
        output_path=output_dir / "window_scores.npz",
        samples_delta=sample_delta,
        ground_truth_delta=gt_delta,
        train_delta=train_raw_delta,
        conditional_iv_samples=cond_samples,
        ground_truth_iv=batch.future_01.detach().cpu().numpy()[:n_windows],
        history_iv=batch.history_01.detach().cpu().numpy()[:n_windows],
        raw_history=raw_history,
        sample_raw=sample_raw,
        window_indices=val_block.indices,
        variable_names=[spec.name for spec in specs],
        existing_result=result,
        iv_count=int(args.iv_count),
    )
    (output_dir / "results_current_panel.json").write_text(
        json.dumps(make_serializable(result), indent=2), encoding="utf-8"
    )
    write_result_summary(output_dir, make_serializable(result))
    print(f"Wrote promoted SNI current-panel metrics to {output_dir}")


if __name__ == "__main__":
    main()
