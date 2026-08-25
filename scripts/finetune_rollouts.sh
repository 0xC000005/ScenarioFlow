#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-1001}"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_reproduction.py" \
  --node-id "train:scenarioflow:stage2:${SEED}"
