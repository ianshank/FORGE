#!/usr/bin/env python3
"""Evaluation and benchmarking script for FORGE agents.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/latest --episodes 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a FORGE agent")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Eval episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    """Run evaluation."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("forge.evaluate")

    logger.info(
        "Evaluating checkpoint=%s over %d episodes", args.checkpoint, args.episodes
    )

    # Evaluation placeholder
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
