#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 ARCHIVED_NODE_ID" >&2
  exit 2
fi
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_reproduction.py" --node-id "$1"
