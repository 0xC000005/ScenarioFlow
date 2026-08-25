#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETACH=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run_demo.sh [--detach]

Build and start the ScenarioFlow landing page and both demos with Docker Compose.

Options:
  --detach       Run the services in the background
  -h, --help     Show this help message

Narrative-to-scenario requires OPENAI_API_KEY. Scenario-to-narrative requires an
authenticated Codex CLI installation under SCENARIOFLOW_CODEX_HOME or ~/.codex.
The site is served at http://127.0.0.1:8766 by default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
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

cd "$ROOT_DIR"
mkdir -p artifacts/cache

if [[ -z "${SCENARIOFLOW_CODEX_HOME:-}" ]]; then
  if [[ -d "$HOME/.codex" ]]; then
    export SCENARIOFLOW_CODEX_HOME="$HOME/.codex"
  else
    mkdir -p .codex-unavailable
    export SCENARIOFLOW_CODEX_HOME="$ROOT_DIR/.codex-unavailable"
  fi
fi

COMPOSE_ARGS=(up --build)
if [[ "$DETACH" -eq 1 ]]; then
  COMPOSE_ARGS+=(--detach)
fi

docker compose "${COMPOSE_ARGS[@]}" \
  frontend narrative-to-scenario scenario-to-narrative

