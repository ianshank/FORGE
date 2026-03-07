#!/usr/bin/env python3
"""CLI training entrypoint for FORGE agents.

Usage::

    python scripts/train.py --config forge.toml --agent mappo --episodes 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add python/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a FORGE agent")
    parser.add_argument(
        "--config", type=str, default="forge.toml", help="Path to config file"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "mcts", "mappo", "hybrid"],
        help="Agent type to train",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    """Run training loop."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("forge.train")

    from forge.config import ForgeConfig
    from forge.utils.seed import set_all_seeds

    config = ForgeConfig.from_file(args.config)
    set_all_seeds(args.seed)

    logger.info(
        "Starting training: agent=%s, episodes=%d, seed=%d",
        args.agent,
        args.episodes,
        args.seed,
    )
    logger.info("Config: grid_size=%d", config.simulation.grid_size)

    # Training loop placeholder — requires environment integration
    for episode in range(1, args.episodes + 1):
        if episode % max(1, args.episodes // 10) == 0:
            logger.info("Episode %d/%d", episode, args.episodes)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
