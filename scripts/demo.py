#!/usr/bin/env python3
"""Unified investor demo launcher for FORGE.

Orchestrates three services:
1. forge-server (Rust Axum WebSocket server) — if a pre-built binary exists
2. Demo UI (FastAPI SSE server) — lightweight browser-based demo
3. Optional training loop with live dashboard metrics

Usage::

    python scripts/demo.py                          # Launch demo UI only
    python scripts/demo.py --with-training           # Demo UI + training loop
    python scripts/demo.py --mode demo-ui            # Default: FastAPI demo UI
    python scripts/demo.py --mode dashboard          # React dashboard (requires npm)
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

DEFAULT_DEMO_PORT: int = int(os.getenv("FORGE_DEMO_UI_PORT", "8765"))
DEFAULT_SERVER_PORT: int = int(os.getenv("FORGE_SERVER_PORT", "8080"))
DEFAULT_HOST: str = os.getenv("FORGE_BIND_HOST", "127.0.0.1")
_STARTUP_WAIT_S: int = int(os.getenv("FORGE_STARTUP_WAIT_SECS", "2"))
_MODE_CHOICES = ("demo-ui", "dashboard")
_TRAINING_EPISODES: str = os.getenv("FORGE_TRAINING_EPISODES", "1000")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch FORGE investor demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo-ui",
        choices=list(_MODE_CHOICES),
        help="Which UI to launch",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_DEMO_PORT, help="Demo UI port"
    )
    parser.add_argument(
        "--server-port", type=int, default=DEFAULT_SERVER_PORT, help="forge-server port"
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Bind host"
    )
    parser.add_argument(
        "--with-training", action="store_true", help="Also start a training loop"
    )
    parser.add_argument(
        "--agent", type=str, default="random", help="Agent type for training"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )
    return parser.parse_args()


def _start_process(
    cmd: list[str], label: str, logger: logging.Logger
) -> subprocess.Popen[bytes]:
    """Start a subprocess and register cleanup."""
    logger.info("Starting %s: %s", label, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda p=proc, lbl=label: _cleanup(p, lbl, logger))  # type: ignore[misc]
    return proc


def _cleanup(
    proc: subprocess.Popen[bytes], label: str, logger: logging.Logger
) -> None:
    """Terminate a subprocess if still running."""
    if proc.poll() is None:
        logger.info("Shutting down %s (PID %d)", label, proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _find_server_binary() -> Path | None:
    """Locate the forge-server binary, preferring release builds."""
    for profile in ("release", "debug"):
        path = Path(f"target/{profile}/forge-server")
        if path.exists():
            return path
    return None


def _start_forge_server(
    binary: Path, logger: logging.Logger
) -> subprocess.Popen[bytes]:
    """Start the Rust forge-server binary."""
    proc = _start_process([str(binary)], "forge-server", logger)
    time.sleep(_STARTUP_WAIT_S)
    return proc


def _start_ui(
    mode: str, host: str, port: int, logger: logging.Logger
) -> subprocess.Popen[bytes]:
    """Start the chosen UI server (demo-ui or dashboard)."""
    if mode == "dashboard":
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=str(dashboard_dir),
        )
        atexit.register(_cleanup, proc, "dashboard", logger)
        return proc
    return _start_process(
        [sys.executable, "-m", "uvicorn", "demo_ui.backend.main:app",
         "--host", host, "--port", str(port)],
        "demo-ui",
        logger,
    )


def _wait_for_processes(
    procs: list[subprocess.Popen[bytes]], logger: logging.Logger
) -> None:
    """Block until Ctrl+C, logging any process exits."""
    while True:
        for proc in procs:
            if proc.poll() is not None:
                logger.warning(
                    "Process PID %d exited with code %d",
                    proc.pid,
                    proc.returncode or 0,
                )
        time.sleep(2)


def main() -> None:
    """Launch the FORGE investor demo."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("forge.demo")

    procs: list[subprocess.Popen[bytes]] = []
    server_url = f"http://{args.host}:{args.server_port}"

    # Step 1: Start forge-server (Rust) if binary exists
    server_bin = _find_server_binary()
    if server_bin is not None:
        procs.append(_start_forge_server(server_bin, logger))
        logger.info("forge-server available at %s", server_url)
    else:
        logger.info("forge-server binary not found — build with 'cargo build -p forge-server'")

    # Step 2: Start the chosen UI
    demo_url = f"http://{args.host}:{args.port}"
    procs.append(_start_ui(args.mode, args.host, args.port, logger))
    time.sleep(_STARTUP_WAIT_S)

    # Step 3: Optional training loop
    if args.with_training:
        train_cmd = [sys.executable, "scripts/train.py", "--agent", args.agent,
                     "--episodes", _TRAINING_EPISODES]
        if server_bin is not None:
            train_cmd.extend(["--dashboard-url", server_url])
        procs.append(_start_process(train_cmd, "training", logger))

    # Step 4: Open browser
    if not args.no_browser:
        logger.info("Opening browser at %s", demo_url)
        webbrowser.open(demo_url)

    logger.info("FORGE demo running at %s — press Ctrl+C to stop", demo_url)
    try:
        _wait_for_processes(procs, logger)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
