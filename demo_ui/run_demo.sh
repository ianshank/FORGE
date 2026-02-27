#!/usr/bin/env bash
# run_demo.sh — Cross-platform FORGE demo UI launcher (Linux / macOS)
#
# Usage:
#   ./demo_ui/run_demo.sh                # default port 8765
#   ./demo_ui/run_demo.sh --port 9000   # custom port
#
# Windows users: use demo_ui/run_demo.ps1 instead.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (overridable via env vars)
# ---------------------------------------------------------------------------
FORGE_HOST="${FORGE_HOST:-127.0.0.1}"
FORGE_PORT="${FORGE_PORT:-8765}"
FORGE_LOG_LEVEL="${FORGE_LOG_LEVEL:-info}"

# Resolve script location so this works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse CLI args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            FORGE_PORT="$2"
            shift 2
            ;;
        --host)
            FORGE_HOST="$2"
            shift 2
            ;;
        --log-level)
            FORGE_LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--port PORT] [--host HOST] [--log-level LEVEL]"
            echo "  Default: http://127.0.0.1:8765  (log-level: info)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Check Python
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+." >&2
    exit 1
fi

PYTHON="python3"
PY_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python ${PY_VERSION} at $(command -v ${PYTHON})"

# ---------------------------------------------------------------------------
# Install backend dependencies (idempotent)
# ---------------------------------------------------------------------------
echo ""
echo "Installing backend dependencies..."
"${PYTHON}" -m pip install \
    --requirement "${SCRIPT_DIR}/backend/requirements.txt" \
    --quiet \
    --disable-pip-version-check

# ---------------------------------------------------------------------------
# Start uvicorn in background
# ---------------------------------------------------------------------------
APP_URL="http://${FORGE_HOST}:${FORGE_PORT}"
echo ""
echo "Starting FORGE demo UI on ${APP_URL} ..."
echo "(Press Ctrl+C to stop)"
echo ""

"${PYTHON}" -m uvicorn \
    "demo_ui.backend.main:app" \
    --host "${FORGE_HOST}" \
    --port "${FORGE_PORT}" \
    --log-level "${FORGE_LOG_LEVEL}" \
    --app-dir "${REPO_ROOT}" &

UVICORN_PID=$!

# Give uvicorn a moment to start
sleep 1

# Check it's actually running
if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "ERROR: uvicorn failed to start. Check the logs above." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Open browser
# ---------------------------------------------------------------------------
echo "Opening ${APP_URL} in your browser..."
if command -v open &>/dev/null; then
    # macOS
    open "${APP_URL}" &
elif command -v xdg-open &>/dev/null; then
    # Linux (X11 / Wayland)
    xdg-open "${APP_URL}" &
elif command -v wslview &>/dev/null; then
    # WSL
    wslview "${APP_URL}" &
else
    echo "(Could not auto-open browser — navigate to ${APP_URL} manually)"
fi

# ---------------------------------------------------------------------------
# Wait for uvicorn to exit (or Ctrl+C)
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Stopping FORGE demo UI..."
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

wait "${UVICORN_PID}"
