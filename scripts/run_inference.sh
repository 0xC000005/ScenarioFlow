#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="$ROOT/artifacts/cache/models/core/stage2/best_model.pt"
OUTPUT_DIR="$ROOT/build/inference/scenarioflow"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_inference.sh [EVALUATOR OPTIONS]

Generate conditional ScenarioFlow paths from the packaged numerical histories
using the released Stage 2 weights. Additional options are passed to the
preserved numerical evaluator.
EOF
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Run ./scripts/setup.sh before inference." >&2
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Fetch the core.stage2 artifact before inference." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT/reproduction/source/scenarioflow"
exec "$PYTHON" \
  experiments/backfill/baselines/evaluate_sni_current_panel_metrics.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --save-window-scores \
  "$@"
