#!/usr/bin/env python
"""G2: evaluate DeepVAR-in-SNI-interface with the UNMODIFIED metric battery.

The existing ``load_current_panel_deep_baseline`` hard-codes global de-z-scoring, so
the SNI-interface variant cannot reuse it. This driver reproduces the data-prep of
``evaluate_current_panel_baselines.main()`` verbatim (same splits, windows, sample
counts, seed, IV-alignment guard) and calls the UNMODIFIED ``evaluate_one`` — so every
metric, including the joint-panel width-Spearman, is computed byte-for-byte identically
to the reference baseline table. Only the model plugged in differs.

New file only. Nothing in ``evaluate_current_panel_baselines.py`` is edited.

Run:
  PYTHONPATH=. python experiments/backfill/baselines/evaluate_g2_deepvar_sni_interface.py \
      --deep_checkpoint models/backfill/baselines/g2_sni_interface/deepvar_sni/best_model.pt \
      --output_dir results/baselines_g2_sni_interface --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from analysis_code.window_score_artifact import add_window_score_argument  # noqa: E402
from experiments.backfill.baselines.evaluate_current_panel_baselines import (  # noqa: E402
    evaluate_one,
    make_block_args,
)
from experiments.backfill.baselines.g2_sni_interface_baseline import (  # noqa: E402
    DeepVARSNIInterfaceBaseline,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    panel_daily_changes,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--iv_transform",
        choices=["log_level", "bounded_logit"],
        default="bounded_logit",
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=772)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--deep_checkpoint",
        default="models/backfill/baselines/g2_sni_interface/deepvar_sni/best_model.pt",
    )
    parser.add_argument("--deep_chunk_size", type=int, default=512)
    parser.add_argument("--output_dir", default="results/baselines_g2_sni_interface")
    add_window_score_argument(parser)
    args = parser.parse_args()

    set_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_args = make_block_args(args)
    columns, metadata, train_block, val_block = build_blocks(block_args)
    if int(args.max_windows) > 0:
        val_block = type(val_block)(
            history_increment=val_block.history_increment[: int(args.max_windows)],
            future_increment=val_block.future_increment[: int(args.max_windows)],
            history_state=val_block.history_state[: int(args.max_windows)],
            future_state=val_block.future_state[: int(args.max_windows)],
            indices=val_block.indices[: int(args.max_windows)],
            specs=val_block.specs,
        )
    if len(val_block.specs) != 39:
        raise RuntimeError(
            f"expected current 39-state panel, got {len(val_block.specs)} states"
        )
    if not any(spec.name == "factor:vix" for spec in val_block.specs):
        raise RuntimeError("current panel does not include factor:vix")

    train_raw_delta = panel_daily_changes(
        train_block.history_state, train_block.future_state
    ).reshape(-1, len(train_block.specs))

    batch = build_rollout_windows(
        data_path="data/vol_surface_with_ret.npz",
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(val_block.history_state.shape[0]),
        device=args.device,
        split="val",
    )
    history_align = np.max(
        np.abs(
            val_block.history_state[:, :, : int(args.iv_count)].reshape(
                val_block.history_state.shape[0], -1
            )
            - batch.history_01.detach()
            .cpu()
            .numpy()
            .reshape(val_block.history_state.shape[0], -1)
        )
    )
    future_align = np.max(
        np.abs(
            val_block.future_state[:, :, : int(args.iv_count)].reshape(
                val_block.future_state.shape[0], -1
            )
            - batch.future_01.detach()
            .cpu()
            .numpy()
            .reshape(val_block.future_state.shape[0], -1)
        )
    )
    if history_align > 1e-6 or future_align > 1e-6:
        raise RuntimeError(
            f"IV alignment failure: history={history_align}, future={future_align}"
        )

    manifest = {
        "mode": "current_realvix_39_state_panel",
        "variant": "deepvar_sni_interface",
        "columns": columns,
        "metadata": metadata,
        "state_specs": [asdict(spec) for spec in val_block.specs],
        "n_train_windows": int(train_block.history_state.shape[0]),
        "n_val_windows": int(val_block.history_state.shape[0]),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "checkpoint": str(args.deep_checkpoint),
        "iv_alignment": {
            "history_max_abs_error": float(history_align),
            "future_max_abs_error": float(future_align),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(make_serializable(manifest), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("Evaluating G2 baseline: deepvar_sni (DeepVAR in SNI interface)")
    print("=" * 72)
    model = DeepVARSNIInterfaceBaseline.load(
        args.deep_checkpoint,
        device=args.device,
        chunk_size=int(args.deep_chunk_size),
    )
    result = evaluate_one(
        "deepvar_sni",
        model,
        train_block,
        val_block,
        train_raw_delta,
        batch=batch,
        samples=int(args.n_samples),
        batch_size=int(args.batch_size),
        iv_count=int(args.iv_count),
        output_dir=output_dir,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        save_window_scores=bool(args.save_window_scores),
    )

    width = result["joint_panel_quality"]["conditional_panel"][
        "history_activity_width_spearman"
    ]
    iv_width = result["current_panel_iv_metrics"]["width_rho"]
    crps = result["panel_b_crps"]["overall"]
    print(
        f"\ndeepvar_sni: joint width-Spearman={width:.4f} "
        f"(IV width_rho={iv_width:.4f}) CRPS={crps:.5f}"
    )
    print(
        f"Wrote G2 result to {output_dir / 'deepvar_sni' / 'results_current_panel.json'}"
    )


if __name__ == "__main__":
    main()
