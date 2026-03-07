#!/usr/bin/env python3
"""Replay file viewer for FORGE simulation recordings.

Usage::

    python scripts/replay_viewer.py --file replay.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="View FORGE replay files")
    parser.add_argument("--file", type=str, required=True, help="Replay file path")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    return parser.parse_args()


def main() -> None:
    """View replay file contents."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("forge.replay")

    path = Path(args.file)
    if not path.exists():
        logger.error("Replay file not found: %s", path)
        return

    with open(path) as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get("frames", [])
    logger.info("Replay: %d frames", len(frames))

    if args.summary:
        if frames:
            logger.info("First tick: %s", frames[0].get("tick", "?"))
            logger.info("Last tick: %s", frames[-1].get("tick", "?"))
        return

    for frame in frames:
        tick = frame.get("tick", "?")
        agents = frame.get("agent_positions", [])
        events = frame.get("events", [])
        logger.info("Tick %s: %d agents, %d events", tick, len(agents), len(events))


if __name__ == "__main__":
    main()
