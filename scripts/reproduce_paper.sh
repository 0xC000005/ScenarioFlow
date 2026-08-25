#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MODE="${1:-all}"

if [[ "$MODE" != "all" && "$MODE" != "tables" && "$MODE" != "plots" ]]; then
  echo "usage: $0 [all|tables|plots]" >&2
  exit 2
fi

"$PYTHON" "$ROOT/scripts/run_reproduction.py" --verify-sources

if [[ "$MODE" == "all" || "$MODE" == "tables" ]]; then
  RESULT_ROOT="$ROOT/build/reproduction/results"
  DAG="$RESULT_ROOT/command_dag.json"
  if [[ ! -f "$DAG" ]]; then
    echo "Run ./scripts/reproduce.sh before rendering the multi-seed table." >&2
    exit 2
  fi

  (
    cd "$ROOT/reproduction/source/paper"
    PYTHONPATH=. "$PYTHON" analysis_code/fmts_multiseed_aggregate.py \
      --protocol-manifest "$DAG" \
      --output-dir "$RESULT_ROOT/aggregate"
  )

  FMTS_STAGE="$ROOT/build/paper/fmts-source"
  mkdir -p \
    "$FMTS_STAGE/paper/scripts" \
    "$FMTS_STAGE/paper/camera_ready/neurips_workshop" \
    "$FMTS_STAGE/results/fmts_multiseed_20260805/aggregate"
  cp "$ROOT/reproduction/source/paper/paper/scripts/generate_fmts_multiseed_appendix.py" \
    "$FMTS_STAGE/paper/scripts/generate_fmts_multiseed_appendix.py"
  cp "$RESULT_ROOT/aggregate/summary.json" \
    "$FMTS_STAGE/results/fmts_multiseed_20260805/aggregate/summary.json"
  (
    cd "$FMTS_STAGE"
    "$PYTHON" paper/scripts/generate_fmts_multiseed_appendix.py
  )

  ICAIF_STAGE="$ROOT/build/paper/icaif-source"
  mkdir -p \
    "$ICAIF_STAGE/paper/scripts" \
    "$ICAIF_STAGE/results/baselines_current_panel/sni_joint" \
    "$ICAIF_STAGE/results/baselines_current_panel/sni_graph_residual"
  cp "$ROOT/reproduction/source/paper/paper/scripts/generate_paper_tables.py" \
    "$ICAIF_STAGE/paper/scripts/generate_paper_tables.py"
  cp "$ROOT/results/frozen/benchmark/current-panel/summary.json" \
    "$ICAIF_STAGE/results/baselines_current_panel/summary.json"
  cp "$ROOT/results/frozen/benchmark/current-panel/scenarioflow/results.json" \
    "$ICAIF_STAGE/results/baselines_current_panel/sni_joint/results_current_panel.json"
  cp "$ROOT/results/frozen/benchmark/current-panel/scenarioflow-graph/results.json" \
    "$ICAIF_STAGE/results/baselines_current_panel/sni_graph_residual/results_current_panel.json"
  "$PYTHON" "$ROOT/scripts/render_icaif_table.py" \
    --source "$ICAIF_STAGE/paper/scripts/generate_paper_tables.py" \
    --output-dir "$ICAIF_STAGE/paper/generated_tables"
fi

if [[ "$MODE" == "all" || "$MODE" == "plots" ]]; then
  CHECKPOINT="$ROOT/artifacts/cache/models/core/stage2/best_model.pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Fetch artifact core.stage2 before rendering the plots." >&2
    exit 2
  fi

  MANAGEMENT_OUT="$ROOT/build/paper/icaif-management"
  (
    cd "$ROOT/reproduction/source/scenarioflow"
    "$PYTHON" experiments/backfill/block_ar/visualize_management_report.py \
      --model_path "$CHECKPOINT" \
      --output_dir "$MANAGEMENT_OUT"
    "$PYTHON" experiments/backfill/block_ar/plot_icaif_joint39_factor_fans.py \
      --checkpoint "$CHECKPOINT" \
      --output "$ROOT/build/paper/joint-factor-fans.png" \
      --manifest "$ROOT/build/paper/joint-factor-fans-selection.json"
  )
fi
