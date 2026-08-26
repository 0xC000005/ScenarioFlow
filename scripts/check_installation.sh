#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: ./scripts/check_installation.sh

Verify the package, data, artifact manifest, and public numerical command routes.
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
for filename in files:
    with (root / "data" / filename).open("rb") as handle:
        handle.read(1)
if not (root / "data/CHECKSUMS.tsv").is_file():
    raise SystemExit("missing data checksum manifest")
if not (root / "artifacts/manifest-v1.tsv").is_file():
    raise SystemExit("missing artifact manifest")
print("package, data, and artifact manifest: ok")
PY
for module in train_next_step finetune_rollouts inference; do
  "$ROOT_DIR/.venv/bin/python" -c "import scenarioflow.reproduction.$module"
  "$ROOT_DIR/.venv/bin/python" -m "scenarioflow.reproduction.$module" --help >/dev/null
done
echo "public numerical routes: ok"
