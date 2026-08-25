#!/usr/bin/env python
"""662a: score state-aware normalized-innovation checkpoints on IV 11-suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    select_rollout_indices,
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)


def build_val_block(
    args: argparse.Namespace, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any], Any]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    positive_level_policy = payload.get(
        "positive_level_policy",
        payload.get("panel_metadata", {}).get(
            "positive_level_policy",
            getattr(args, "positive_level_policy", "reference_based"),
        ),
    )
    if args.clean_nonpositive_log_levels:
        panel, _cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
        )
    iv_transform = payload.get(
        "iv_transform",
        payload.get("normalization", {}).get(
            "iv_transform",
            payload.get("panel_metadata", {}).get(
                "iv_transform",
                getattr(args, "iv_transform", "log_level"),
            ),
        ),
    )
    iv_lower_bound = float(
        payload.get(
            "iv_lower_bound",
            payload.get("normalization", {}).get(
                "iv_lower_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_lower_bound",
                    getattr(args, "iv_lower_bound", 1e-4),
                ),
            ),
        )
    )
    iv_upper_bound = float(
        payload.get(
            "iv_upper_bound",
            payload.get("normalization", {}).get(
                "iv_upper_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_upper_bound",
                    getattr(args, "iv_upper_bound", 1.0),
                ),
            ),
        )
    )
    indices = select_rollout_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        max_windows=int(args.max_windows) if int(args.max_windows) > 0 else None,
        split=args.eval_split,
    )
    block = build_increment_coordinate_block(
        panel,
        columns,
        indices,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        iv_count=int(args.iv_count),
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    norm_cfg = payload.get("normalization", {})
    scale_half_life = norm_cfg.get("scale_half_life", getattr(args, "scale_half_life", 0.0))
    if scale_half_life is not None and float(scale_half_life) <= 0.0:
        scale_half_life = None
    scale_floor = float(norm_cfg.get("scale_floor", getattr(args, "scale_floor", 1e-4)))
    center_mode = norm_cfg.get("center_mode", getattr(args, "center_mode", "zero"))
    drift_feature_mode = norm_cfg.get("drift_feature_mode", getattr(args, "drift_feature_mode", "none"))
    (
        history_level,
        history_norm,
        _future_level,
        _future_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
    ) = select_normalized_innovation_scope(
        block,
        payload.get("state_scope", args.state_scope),
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=scale_floor,
        center_mode=center_mode,
        drift_feature_mode=drift_feature_mode,
    )
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in specs]
    if expected and expected != actual:
        raise RuntimeError("checkpoint state specs do not match rebuilt validation specs")
    return (
        history_level.astype(np.float32),
        history_norm.astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
        drift_feature.astype(np.float32),
        history_raw.astype(np.float32),
        specs,
        block,
    )


def alignment_diagnostics(block: Any, batch: Any, n_windows: int) -> dict[str, float]:
    history_iv = block.history_state[:n_windows, :, :25].reshape(n_windows, -1)
    future_iv = block.future_state[:n_windows, :, :25].reshape(n_windows, -1)
    batch_history = batch.history_01.detach().cpu().numpy()[:n_windows].reshape(n_windows, -1)
    batch_future = batch.future_01.detach().cpu().numpy()[:n_windows].reshape(n_windows, -1)
    return {
        "history_max_abs_error": float(np.max(np.abs(history_iv - batch_history))),
        "future_max_abs_error": float(np.max(np.abs(future_iv - batch_future))),
        "history_mean_abs_error": float(np.mean(np.abs(history_iv - batch_history))),
        "future_mean_abs_error": float(np.mean(np.abs(future_iv - batch_future))),
    }


@torch.no_grad()
def generate_iv_samples(
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
    iv_count: int,
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
        arr = panel_arr[..., :iv_count]
        chunks.append(arr.reshape(end - start, int(samples), int(n_steps), 5, 5).astype(np.float32))
        print(f"  generated windows {end}/{history_level.shape[0]}", flush=True)
    return np.concatenate(chunks, axis=0)


def summarize_results(results: dict[str, Any], alignment: dict[str, float]) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    iv_link = results.get("iv_ewma_economic_link", {})
    return [
        "- source: `662a state-aware normalized-innovation AR flow`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        f"- history alignment max error: `{alignment['history_max_abs_error']:.3e}`",
        f"- future alignment max error: `{alignment['future_max_abs_error']:.3e}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
        f"- IV-EWMA economic link: `{iv_link.get('overall_pass', False)}` "
        f"(slope ratio `{iv_link.get('slope_ratio', float('nan')):.3f}`, "
        f"rho ratio `{iv_link.get('spearman_ratio', float('nan')):.3f}`)",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=662)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    history_level, history_norm, center, scale, drift_feature, history_raw, specs, block = build_val_block(args, payload)
    cfg = payload["config"]
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n_windows]
    history_norm = history_norm[:n_windows]
    center = center[:n_windows]
    scale = scale[:n_windows]
    drift_feature = drift_feature[:n_windows]
    history_raw = history_raw[:n_windows]
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(n_windows),
        device=device,
        split=args.eval_split,
    )
    alignment = alignment_diagnostics(block, batch, n_windows)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_max_abs_error"] > 1e-6:
        raise RuntimeError(f"662a/full-suite alignment failed: {alignment}")

    print(f"Generating {args.samples} samples for {n_windows} windows")
    t0 = time.time()
    cond_samples = generate_iv_samples(
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
        iv_count=int(args.iv_count),
        sample_temperature=float(args.sample_temperature),
    )
    generation_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()[:n_windows]
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(n_windows)}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
        data_path=args.data_path,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        batch_size=int(args.batch_size),
        conditionality_samples=int(args.conditionality_samples),
        conditionality_max_batches=int(args.conditionality_max_batches),
        device=device,
        eval_split=args.eval_split,
    )
    results["config"] = {
        "mode": "662a_state_aware_normalized_innovation_flow",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "checkpoint_config": cfg,
        "state_scope": payload.get("state_scope", args.state_scope),
        "model_coordinate": payload.get("model_coordinate", "state_aware_normalized_innovation"),
        "normalization": payload.get("normalization", {}),
        "n_windows": int(n_windows),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "sample_temperature": float(args.sample_temperature),
        "generation_time_s": float(generation_time),
        "device": str(device),
        "seed": int(args.seed),
        "eval_split": args.eval_split,
        "alignment": alignment,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(
        out_md,
        f"662a State-Aware Normalized-Innovation AR Flow Official IV 11-Suite ({args.eval_split})",
        summarize_results(results, alignment),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
