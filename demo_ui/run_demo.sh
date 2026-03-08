#!/usr/bin/env bash
# FORGE Demo UI — Cross-platform launcher (Bash equivalent of run_demo.ps1)
#
# Usage:
#   bash demo_ui/run_demo.sh
#   bash demo_ui/run_demo.sh --port 9000
#   bash demo_ui/run_demo.sh --no-browser
#   bash demo_ui/run_demo.sh --skip-install

set -euo pipefail

# --- Defaults ---
PORT=8765
NO_BROWSER=false
SKIP_INSTALL=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)      PORT="$2"; shift 2 ;;
    --no-browser)  NO_BROWSER=true; shift ;;
    --skip-install) SKIP_INSTALL=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--port PORT] [--no-browser] [--skip-install]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS="$SCRIPT_DIR/backend/requirements.txt"
URL="http://127.0.0.1:$PORT"

echo ""
echo "================================================================"
echo "   FORGE Demo UI"
echo "================================================================"
echo ""

# ----- Step 1: Check prerequisites -----
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "ERROR: Python 3 is required but not found on PATH."
  exit 1
fi
PYTHON="$(command -v python3 2>/dev/null || command -v python)"

# ----- Step 2: Install dependencies -----
if [ "$SKIP_INSTALL" = false ]; then
  echo "[1/3] Installing dependencies..."
  "$PYTHON" -m pip install -r "$REQUIREMENTS" --quiet
  echo "      Done."
else
  echo "[1/3] Skipping dependency install (--skip-install)."
fi

# ----- Step 3: Launch server -----
echo "[2/3] Starting FastAPI server on $URL ..."

cd "$FORGE_ROOT"
"$PYTHON" -m uvicorn demo_ui.backend.main:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --reload \
  --log-level info &
SERVER_PID=$!

# Give the server time to bind
sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: Server process exited prematurely."
  exit 1
fi

echo "      Server PID: $SERVER_PID"

# ----- Step 4: Open browser -----
if [ "$NO_BROWSER" = false ]; then
  echo "[3/3] Opening browser at $URL ..."
  if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" 2>/dev/null &
  elif command -v open &>/dev/null; then
    open "$URL" 2>/dev/null &
  else
    echo "      Could not detect browser opener. Please visit $URL manually."
  fi
fi

echo ""
echo "================================================================"
echo "   FORGE Demo UI is running at: $URL"
echo "   Press Ctrl+C to stop the server."
echo "================================================================"
echo ""

# Wait and cleanup on exit
cleanup() {
  echo ""
  echo "Shutting down server (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  echo "Done. Goodbye!"
}
trap cleanup EXIT INT TERM

wait "$SERVER_PID"
