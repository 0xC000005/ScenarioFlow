#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="core"
BASE_URL="${SCENARIOFLOW_ARTIFACT_BASE_URL:-}"
BUNDLE_DIR=""
ARTIFACT_ROOT="$ROOT_DIR/artifacts/cache"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [OPTIONS]

Install ScenarioFlow and fetch a checksum-verified artifact profile.

Options:
  --profile NAME       Artifact profile to fetch (default: core)
  --base-url URL       Base URL of the hosted artifact bundle
  --bundle-dir PATH    Local artifact bundle with manifest paths preserved
  --artifact-root PATH Installation directory for artifacts
  -h, --help           Show this help message

Set SCENARIOFLOW_ARTIFACT_BASE_URL instead of passing --base-url each time.
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

if [[ -n "$BASE_URL" && -n "$BUNDLE_DIR" ]]; then
  echo "Use either --base-url or --bundle-dir, not both." >&2
  exit 2
fi
if [[ -z "$BASE_URL" && -z "$BUNDLE_DIR" ]]; then
  echo "Pass --base-url or --bundle-dir to fetch release artifacts." >&2
  exit 2
fi
if [[ -n "$BUNDLE_DIR" && ! -d "$BUNDLE_DIR" ]]; then
  echo "Artifact bundle not found: $BUNDLE_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
uv sync \
  --extra dev \
  --extra baseline-classical \
  --extra demo-server \
  --extra demo-openai \
  --extra paper

SOURCE_ARGS=(--base-url "$BASE_URL")
if [[ -n "$BUNDLE_DIR" ]]; then
  SOURCE_ARGS=(--bundle-dir "$BUNDLE_DIR")
fi

uv run scenarioflow fetch \
  --profile "$PROFILE" \
  "${SOURCE_ARGS[@]}" \
  --artifact-root "$ARTIFACT_ROOT"
uv run scenarioflow setup --check --artifact-root "$ARTIFACT_ROOT"

