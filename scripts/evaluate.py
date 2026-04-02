#!/usr/bin/env python3
"""Evaluation and benchmarking script for FORGE agents.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/latest --episodes 100
    python scripts/evaluate.py --dry-run --agent random --episodes 5
    python scripts/evaluate.py --config forge.toml --agent mcts --episodes 50

Prints evaluation results as JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge.config import ForgeConfig
from forge.evaluation import EvalConfig, Evaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a FORGE agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to agent checkpoint (optional; random agent used if omitted)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode (small env, few steps)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to forge.toml config file",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "mcts"],
        help="Agent type to use when no checkpoint is provided",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _make_random_agent(num_actions: int = 8) -> object:
    """Return a simple random-action agent backed by a mock."""
    import random  # noqa: PLC0415

    class _RandomAgent:
        """Uniform-random action agent."""

        def act(self, observation: object) -> tuple[int, dict]:
            return random.randint(0, num_actions - 1), {}

        def learn(self, batch: dict) -> dict:
            return {}

    return _RandomAgent()


def _make_mcts_agent() -> object:
    """Return a stub MCTS agent (placeholder until native bindings available)."""

    class _MctsAgent:
        """MCTS stub — falls back to action 0 until native bindings are ready."""

        def act(self, observation: object) -> tuple[int, dict]:
            return 0, {"mcts_stub": True}

        def learn(self, batch: dict) -> dict:
            return {}

    return _MctsAgent()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _make_mock_env(max_episode_length: int = 100, num_actions: int = 8) -> object:
    """Create a lightweight mock environment for use when native env is unavailable.

    The mock runs episodes of exactly *max_episode_length* steps returning
    a fixed reward of 1.0 per step.
    """
    env = MagicMock()
    obs = MagicMock(name="obs")
    env.reset.return_value = (obs, {})
    env.observation_space = {"n": num_actions}
    env.action_space = {"n": num_actions}

    step_count = [0]

    def _step(_action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
        step_count[0] += 1
        terminated = step_count[0] >= max_episode_length
        if terminated:
            step_count[0] = 0
        return obs, 1.0, terminated, False, {"tier": 1, "success": terminated}

    env.step.side_effect = _step
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run evaluation and print results as JSON."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("forge.evaluate")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    forge_config = ForgeConfig.from_file(args.config)
    if args.dry_run:
        forge_config.dry_run.enabled = True
    sim = forge_config.effective_simulation()

    # ------------------------------------------------------------------
    # Build EvalConfig
    # ------------------------------------------------------------------
    num_episodes: int
    if args.episodes is not None:
        num_episodes = args.episodes
    elif forge_config.dry_run.enabled:
        num_episodes = forge_config.dry_run.max_episodes
    else:
        num_episodes = 10  # sensible default

    seed = args.seed if args.seed is not None else sim.seed
    eval_config = EvalConfig(
        num_episodes=num_episodes,
        seed=seed,
        determinism_check=not forge_config.dry_run.enabled,
        log_per_episode=args.log_level == "DEBUG",
    )

    logger.info(
        "Starting evaluation: agent=%s episodes=%d seed=%d dry_run=%s",
        args.agent,
        eval_config.num_episodes,
        eval_config.seed,
        forge_config.dry_run.enabled,
    )

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    env: object
    try:
        import forge  # noqa: F401,PLC0415

        logger.debug("Native forge extension available — using real environment")
        # When native bindings are present, construct the real env here.
        # For now fall through to the mock until bindings are wired.
        raise ImportError("real env wiring not yet implemented")
    except ImportError:
        logger.info("Native env unavailable — using mock environment")
        env = _make_mock_env(
            max_episode_length=sim.max_episode_length,
        )

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    agent: object
    if args.checkpoint:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        # Placeholder: real checkpoint loading goes here.
        agent = _make_random_agent()
    elif args.agent == "mcts":
        agent = _make_mcts_agent()
    else:
        agent = _make_random_agent()

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    evaluator = Evaluator(eval_config)
    result = evaluator.evaluate(env, agent)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output = result.to_dict()
    print(json.dumps(output, indent=2))
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
