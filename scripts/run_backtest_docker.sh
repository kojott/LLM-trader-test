#!/usr/bin/env bash
set -euo pipefail

# Runs the backtest inside Docker with start/end overrides and an optional
# system prompt file. Results are written to data-backtest/run-<id> on the host.

usage() {
  cat <<'EOF'
Usage: scripts/run_backtest_docker.sh <start> <end> [system_prompt_file]

Arguments:
  <start>                UTC timestamp for BACKTEST_START (e.g. 2024-01-01T00:00:00Z)
  <end>                  UTC timestamp for BACKTEST_END   (e.g. 2024-01-07T00:00:00Z)
  [system_prompt_file]   Optional path to a prompt file. Use "-" to skip.

Environment overrides (optional):
  DOCKER_IMAGE           Docker image to run (default: tradebot)
  DOCKER_ENV_FILE        Env file passed via --env-file (default: .env in repo root)
  BACKTEST_INTERVAL      Interval override forwarded to the container
  BACKTEST_RUN_ID        Custom run identifier (default: run-<timestamp>-<random>)

Examples:
  scripts/run_backtest_docker.sh 2024-01-01T00:00:00Z 2024-01-07T00:00:00Z prompts/backtest_rules.txt
  BACKTEST_INTERVAL=5m scripts/run_backtest_docker.sh 2024-02-01T00:00:00Z 2024-02-05T00:00:00Z -
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

START_ARG=$1
END_ARG=$2
PROMPT_ARG=${3:-}

if [[ -z "$START_ARG" || -z "$END_ARG" ]]; then
  echo "error: start and end timestamps are required." >&2
  usage >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required but not found in PATH." >&2
  exit 1
fi

IMAGE_NAME=${DOCKER_IMAGE:-tradebot}
ENV_FILE=${DOCKER_ENV_FILE:-${REPO_ROOT}/.env}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file '$ENV_FILE' not found. Set DOCKER_ENV_FILE or create .env." >&2
  exit 1
fi

# Auto-build the image if missing.
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image '$IMAGE_NAME' not found. Building..."
  docker build -t "$IMAGE_NAME" "$REPO_ROOT"
fi

DATA_BACKTEST_DIR="${REPO_ROOT}/data-backtest"
mkdir -p "$DATA_BACKTEST_DIR"

# Generate a unique run identifier if not provided.
RUN_ID=${BACKTEST_RUN_ID:-}
if [[ -z "$RUN_ID" ]]; then
  timestamp=$(date -u +%Y%m%d-%H%M%S)
  random_suffix=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6 || true)
  if [[ -z "$random_suffix" ]]; then
    random_suffix=$(printf '%06x' "$(($RANDOM * $RANDOM))")
  fi
  RUN_ID="run-${timestamp}-${random_suffix}"
fi

# Sanitise for Docker container names (must start with alphanumeric).
container_suffix=$(echo "$RUN_ID" | tr -cs '[:alnum:]._-' '-' | sed 's/^[^[:alnum:]]//')
if [[ -z "$container_suffix" ]]; then
  container_suffix=$(date +%s)
fi
CONTAINER_NAME="tradebot-backtest-${container_suffix}"

VOLUME_MOUNTS=(
  "-v" "${DATA_BACKTEST_DIR}:/app/data-backtest"
)

ENV_VARS=(
  "--env" "BACKTEST_START=${START_ARG}"
  "--env" "BACKTEST_END=${END_ARG}"
  "--env" "BACKTEST_RUN_ID=${RUN_ID}"
  "--env" "BACKTEST_DATA_DIR=/app/data-backtest"
  "--env" "BACKTEST_DISABLE_TELEGRAM=true"
)

if [[ -n "${BACKTEST_INTERVAL:-}" ]]; then
  ENV_VARS+=("--env" "BACKTEST_INTERVAL=${BACKTEST_INTERVAL}")
fi

# Handle optional system prompt file argument.
if [[ -n "$PROMPT_ARG" && "$PROMPT_ARG" != "-" ]]; then
  if [[ ! -f "$PROMPT_ARG" ]]; then
    echo "error: system prompt file '$PROMPT_ARG' does not exist." >&2
    exit 1
  fi
  PROMPT_ABS=$(cd "$(dirname "$PROMPT_ARG")" && pwd)/$(basename "$PROMPT_ARG")

  PROMPT_FILENAME=$(basename "$PROMPT_ABS")
  PROMPT_CONTAINER_DIR="/app/prompts"
  PROMPT_CONTAINER_PATH="${PROMPT_CONTAINER_DIR}/${PROMPT_FILENAME}"

  VOLUME_MOUNTS+=("-v" "${PROMPT_ABS}:${PROMPT_CONTAINER_PATH}:ro")
  ENV_VARS+=("--env" "BACKTEST_SYSTEM_PROMPT_FILE=${PROMPT_CONTAINER_PATH}")

  echo "Using system prompt file: ${PROMPT_ABS}"
else
  echo "No system prompt override provided."
fi

echo "Starting Docker backtest run '${RUN_ID}'..."
echo "Container name: ${CONTAINER_NAME}"
echo "Start: ${START_ARG}"
echo "End:   ${END_ARG}"

set -x
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --env-file "${ENV_FILE}" \
  "${ENV_VARS[@]}" \
  "${VOLUME_MOUNTS[@]}" \
  "${IMAGE_NAME}" \
  python backtest.py
set +x

echo "Backtest run complete."
echo "Results available in: ${DATA_BACKTEST_DIR}/${RUN_ID}"
