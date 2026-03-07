#!/usr/bin/env python3
"""Investor demo launcher — starts the FORGE demo UI server.

Usage::

    python scripts/demo.py [--port 8000]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch FORGE demo")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


def main() -> None:
    """Launch the demo server."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("forge.demo")

    logger.info("Starting FORGE demo on %s:%d", args.host, args.port)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demo_ui.backend.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
