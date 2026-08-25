#!/usr/bin/env python
"""
220h: Full 11-suite multi-horizon validation for recursive one-day kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    rollout_samples_in_batches,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_iv_ewma_economic_link_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_risk_state_allocation_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    ordered = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("time_series", results["time_series"]["overall_pass"]),
        ("block_ar", results["block_ar"]["overall_pass"]),
        ("cointegration", results["cointegration"]["overall_pass"]),
        ("regime_coverage", results["regime_coverage"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in ordered if not passed]
    return sum(int(passed) for _name, passed in ordered), failed


def main() -> None:
    parser = argparse.ArgumentParser(description="220h full 11-suite multi-horizon validation")
    parser.add_argument("--model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sample_temperature_override",
        type=float,
        default=None,
        help="Optionally override cfg.sample_temperature at evaluation time.",
    )
    parser.add_argument(
        "--transport_strength_override",
        type=float,
        default=None,
        help="Optionally override cfg.transport_strength at evaluation time.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--force_native_anchor", action="store_true",
                        help="Force native-path rollout (model.sample_batched) with inference "
                             "anchor(0.50) for any 227a-family checkpoint. Use this to evaluate "
                             "229a honestly under its production rollout regime.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    if args.sample_temperature_override is not None:
        if hasattr(model, "cfg") and hasattr(model.cfg, "sample_temperature"):
            model.cfg.sample_temperature = float(args.sample_temperature_override)
        elif hasattr(model, "sample_temperature"):
            model.sample_temperature = float(args.sample_temperature_override)
        else:
            raise ValueError(
                f"Model type {args.model_type} does not expose sample_temperature"
            )
        print(
            f"[eval override] sample_temperature={float(args.sample_temperature_override):.4f}"
        )
    if args.transport_strength_override is not None:
        if hasattr(model, "cfg") and hasattr(model.cfg, "transport_strength"):
            model.cfg.transport_strength = float(args.transport_strength_override)
        else:
            raise ValueError(
                f"Model type {args.model_type} does not expose transport_strength"
            )
        print(
            f"[eval override] transport_strength={float(args.transport_strength_override):.4f}"
        )

    # Keep inference-time anchor overrides aligned with evaluate_220b so the 11-suite
    # uses the same rollout regime as the 7-suite comparisons.
    if args.model_type in {'232a', '232b', '232c', '232d'}:
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print("[eval override] 232 variant: use_scale_anchor=True, alpha=0.50 at inference")

    if args.model_type.startswith('233a'):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a {args.model_type}: use_scale_anchor=True, alpha=0.50 at inference")

    if args.model_type.startswith("233a_v1_2"):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a_v1_2 {args.model_type}: use_scale_anchor=True, alpha=0.50")

    if args.force_native_anchor and args.model_type == '227a':
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print("[eval override] --force_native_anchor: use_scale_anchor=True alpha=0.50, native path")

    wrapper = OneDayKernelRolloutWrapper(model).eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    use_native = hasattr(model, 'sample_batched') and (
        hasattr(model, 'temporal_adapter')
        or args.model_type == '183c'
        or args.model_type in {'231a', '231b', '231c', '232a', '232b', '232c', '232d'}
        or args.model_type.startswith('233a')
        or args.model_type.startswith('240a')
        or args.model_type.startswith('240b')
        or args.model_type.startswith('240c')
        or args.model_type.startswith(("250", "251", "252", "253", "254", "255", "256", "257", "258", "260", "261", "262", "263", "264", "266", "267", "268", "269", "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289", "290", "291", "293", "294", "295", "296", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349", "350", "351", "352", "353", "354", "421", "460", "470", "471", "476", "481", "483", "492", "544"))
        or args.force_native_anchor
    )
    if use_native:
        from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
        print("Using native sample_batched")
        outputs = []
        n_steps = batch.future_01.shape[1]
        for start in range(0, batch.history_01.shape[0], args.batch_size):
            end = min(start + args.batch_size, batch.history_01.shape[0])
            hist_batch = normalize_iv(batch.history_01[start:end])
            with torch.no_grad():
                samp = model.sample_batched(
                    hist_batch,
                    n_samples=args.samples,
                    n_steps=n_steps,
                    chunk_size=args.chunk_size,
                )
            outputs.append(samp.cpu().numpy())
        cond_samples = np.concatenate(outputs, axis=0)
    else:
        cond_samples = rollout_samples_in_batches(
            wrapper=wrapper,
            history_norm=batch.history_norm,
            n_samples=args.samples,
            n_steps=batch.future_01.shape[1],
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    # Native multi-day models already implement sample_batched; one-day kernels need the wrapper.
    cond_model = model if use_native else wrapper
    conditionality = run_conditionality_tests(
        cond_model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    iv_ewma_economic_link = run_iv_ewma_economic_link_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    risk_state_allocation = run_risk_state_allocation_tests(
        cond_samples,
        ground_truth,
        history_01,
    )
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
            "sample_temperature_override": args.sample_temperature_override,
            "transport_strength_override": args.transport_strength_override,
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "iv_ewma_economic_link": iv_ewma_economic_link,
        "regime_coverage": regime_coverage,
        "risk_state_allocation": risk_state_allocation,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 11,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- MR ratio h1: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- MR ratio h30: `{mean_reversion.get('full_horizon', {}).get('per_horizon', {}).get(30, {}).get('ratio', float('nan')):.3f}`",
        "",
        "**Additional v2 Suites**",
        f"- time-series ACF corr: `{time_series['acf']['acf_correlation']:.3f}`",
        f"- block boundary ratio: `{block_ar['boundary_smoothness']['boundary_ratio']:.3f}`",
        f"- cointegration gen/GT ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- IV-EWMA economic link: `{iv_ewma_economic_link['overall_pass']}` "
        f"(slope ratio `{iv_ewma_economic_link.get('slope_ratio', float('nan')):.3f}`, "
        f"rho ratio `{iv_ewma_economic_link.get('spearman_ratio', float('nan')):.3f}`)",
        f"- regime coverage overall: `{regime_coverage['overall_pass']}`",
        f"- risk-state allocation: `{risk_state_allocation['overall_pass']}`",
        f"- risk-state observable response: `{risk_state_allocation.get('observable_state_response_pass', False)}`",
        f"- risk-state oracle future alignment: `{risk_state_allocation.get('oracle_future_alignment_pass', False)}`",
        f"- risk-state width/history rho: `{risk_state_allocation.get('history_width_spearman', float('nan')):.3f}`",
        f"- risk-state width/future rho: `{risk_state_allocation['future_width_spearman']:.3f}`",
        "",
        "**Fidelity / Structure**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220h Full 11-Suite Multi-Horizon Validation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
