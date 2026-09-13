#!/usr/bin/env python3
"""Compare random vs lawnmower (and optional SAC) on orchard_coverage.

Does not train SAC for a long budget. Pass ``--sac-timesteps N`` with a
small N to smoke the CleanRL script, or omit it.

Usage::
    python examples/run_orchard_coverage_baselines.py --episodes 2
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any

from forge.baselines.coverage import (
    home_from_config,
    lawnmower_action_ids,
    orchard_env_config,
)

logger = logging.getLogger(__name__)


def _run_policy(actions: list[int], config: dict[str, Any], seed: int) -> dict[str, float]:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env = ForgeGymnasiumEnv(config=config)
    obs, info = env.reset(seed=seed)
    _ = obs
    total = 0.0
    steps = 0
    for action in actions:
        _obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    completed = 0.0
    if isinstance(info, dict) and info.get("tasks_completed"):
        completed = 1.0
    return {"return": total, "steps": float(steps), "task_flag": completed}


def _random_actions(n: int, action_n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(action_n) for _ in range(n)]


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--sac-timesteps",
        type=int,
        default=0,
        help="If >0, print the SAC command for configs/training/sac_orchard.toml",
    )
    args = parser.parse_args()

    config = orchard_env_config()
    world = config.get("world") or {}
    width = int(world.get("width", 16))
    height = int(world.get("height", 16))
    margin = int(world.get("geofence_margin", 0))
    home = home_from_config(config)
    lawn = lawnmower_action_ids(width, height, home, margin=margin)[: args.max_steps]
    logger.info("lawnmower plan length=%d home=%s margin=%d", len(lawn), home, margin)

    try:
        from forge_env.gymnasium_env import ForgeGymnasiumEnv
    except ImportError:
        logger.warning("native forge_env missing; printing plan only")
        if args.sac_timesteps:
            logger.info(
                "SAC: python examples/train_sac_cleanrl.py "
                "--config configs/training/sac_orchard.toml --total-timesteps %d",
                args.sac_timesteps,
            )
        return 0

    probe = ForgeGymnasiumEnv(config=config)
    action_n = int(probe.action_space.n)
    probe.close()

    for ep in range(args.episodes):
        seed = args.seed + ep
        rnd = _run_policy(_random_actions(args.max_steps, action_n, seed), config, seed)
        lwn = _run_policy(lawn, config, seed)
        logger.info("episode=%d random=%s lawnmower=%s", ep, rnd, lwn)

    if args.sac_timesteps:
        logger.info(
            "SAC: python examples/train_sac_cleanrl.py "
            "--config configs/training/sac_orchard.toml --total-timesteps %d",
            args.sac_timesteps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
