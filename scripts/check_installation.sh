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
"$ROOT_DIR/.venv/bin/python" - "$ROOT_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = (
    "vol_surface_with_ret.npz",
    "spx_vol_surface_history_full_data_fixed.parquet",
    "multi_factor_data.npz",
    "multi_factor_levels.parquet",
    "multi_factor_returns.parquet",
)
for snapshot in ("scenarioflow", "baselines", "garch_ccc", "deepvar_sni"):
    data = root / "reproduction/source" / snapshot / "data"
    if not data.is_symlink():
        raise SystemExit(f"preserved data route is not a symlink: {data}")
    for filename in files:
        with (data / filename).open("rb") as handle:
            handle.read(1)
print("preserved data routes: ok")
PY
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_reproduction.py" \
  --verify-sources
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_reproduction.py" \
  --node-id train:scenarioflow:stage1:1001 \
  --dry-run >/dev/null
echo "original reproduction routes: ok"
