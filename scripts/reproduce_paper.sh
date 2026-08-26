#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"
if [[ "$MODE" != "all" && "$MODE" != "tables" && "$MODE" != "plots" ]]; then
  echo "usage: $0 [all|tables|plots] [--artifact-root PATH] [--output-dir PATH]" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" -m scenarioflow.cli paper "$MODE" "$@"
