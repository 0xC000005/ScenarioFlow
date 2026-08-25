#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: ./scripts/check_installation.sh

Verify the packaged data, preserved source snapshots, and original command routes.
This check does not train a model or reproduce the paper results.

Options:
  -h, --help    Show this help message
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

cd "$ROOT_DIR"
uv run scenarioflow setup --check
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_reproduction.py" \
  --verify-sources
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_reproduction.py" \
  --node-id train:scenarioflow:stage1:1001 \
  --dry-run >/dev/null
echo "original reproduction routes: ok"
