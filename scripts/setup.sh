#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="core"
DEFAULT_BASE_URL="https://github.com/0xC000005/ScenarioFlow/releases/download/v1.0.0"
BASE_URL="${SCENARIOFLOW_ARTIFACT_BASE_URL:-$DEFAULT_BASE_URL}"
BASE_URL_OVERRIDDEN=0
if [[ -n "${SCENARIOFLOW_ARTIFACT_BASE_URL:-}" ]]; then
  BASE_URL_OVERRIDDEN=1
fi
BUNDLE_DIR=""
ARTIFACT_ROOT="$ROOT_DIR/artifacts/cache"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [OPTIONS]

Install ScenarioFlow and fetch a checksum-verified artifact profile.

This installs the numerical core used for training, released-checkpoint replay,
and benchmark planning. Full multi-seed reproduction additionally requires:
  uv sync --locked --extra baseline-classical

Options:
  --profile NAME       Artifact profile to fetch (default: core)
  --base-url URL       Override the GitHub Release URL
  --bundle-dir PATH    Use a local release bundle instead
  --artifact-root PATH Installation directory for artifacts
  -h, --help           Show this help message

The v1.0.0 GitHub Release is used by default. Set
SCENARIOFLOW_ARTIFACT_BASE_URL to use another host.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?--profile requires a value}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:?--base-url requires a value}"
      BASE_URL_OVERRIDDEN=1
      shift 2
      ;;
    --bundle-dir)
      BUNDLE_DIR="${2:?--bundle-dir requires a value}"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="${2:?--artifact-root requires a value}"
      shift 2
      ;;
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
done

if [[ -n "$BUNDLE_DIR" && "$BASE_URL_OVERRIDDEN" -eq 1 ]]; then
  echo "Use either --base-url or --bundle-dir, not both." >&2
  exit 2
fi
if [[ -n "$BUNDLE_DIR" ]]; then
  BASE_URL=""
fi
if [[ -n "$BUNDLE_DIR" && ! -d "$BUNDLE_DIR" ]]; then
  echo "Artifact bundle not found: $BUNDLE_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
uv sync --locked

SOURCE_ARGS=(--base-url "$BASE_URL")
if [[ -n "$BUNDLE_DIR" ]]; then
  SOURCE_ARGS=(--bundle-dir "$BUNDLE_DIR")
fi

if ! uv run scenarioflow fetch \
  --profile "$PROFILE" \
  "${SOURCE_ARGS[@]}" \
  --artifact-root "$ARTIFACT_ROOT"; then
  if [[ -z "$BUNDLE_DIR" ]]; then
    BUNDLE_NAME="scenarioflow-complete-v1.0.0.zip"
    if [[ "$PROFILE" == "core" ]]; then
      BUNDLE_NAME="scenarioflow-core-v1.0.0.zip"
    fi
    cat >&2 <<EOF

Automatic artifact download failed.

Download $BUNDLE_NAME from:
  https://github.com/0xC000005/ScenarioFlow/releases/tag/v1.0.0

Extract it into one directory, then rerun:
  ./scripts/setup.sh --profile $PROFILE --bundle-dir /path/to/downloaded-release-assets
EOF
  fi
  exit 1
fi
uv run scenarioflow setup --check --artifact-root "$ARTIFACT_ROOT"
