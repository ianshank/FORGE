#!/usr/bin/env python3
"""CLI training entrypoint for FORGE agents.

Wires the ForgeGymnasiumEnv to MAPPO, MCTS, or Random agents and runs
the training loop. All hyperparameters flow through ForgeConfig.

Usage::

    python scripts/train.py --config forge.toml --agent mappo --num-updates 10
    python scripts/train.py --agent random --episodes 100
    python scripts/train.py --agent mcts --episodes 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add python/ to path so forge and forge_env packages are importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logger = logging.getLogger("forge.train")


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
        choices=["random", "mcts", "mappo"],
        help="Agent type to train",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    parser.add_argument(
        "--num-updates",
        type=int,
        default=10,
        help="PPO update iterations (mappo only)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes per eval",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def _flatten_obs(obs: dict) -> "np.ndarray":
    """Flatten a dict observation into a 1-D float32 numpy array.

    Keys are sorted for deterministic ordering.
    """
    import numpy as np  # noqa: PLC0415

    parts = [
        np.asarray(obs[key], dtype=np.float32).ravel() for key in sorted(obs.keys())
    ]
    return np.concatenate(parts)


def _compute_obs_dim(env: object) -> int:
    """Compute the flat observation dimensionality from the environment."""
    obs, _info = env.reset()  # type: ignore[attr-defined]
    return _flatten_obs(obs).shape[0]


def _train_mappo(env: object, config: object, args: argparse.Namespace) -> None:
    """Train a MAPPO agent with PPO rollout collection and updates."""
    import numpy as np  # noqa: PLC0415

    from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig
    from forge.training.checkpointing import CheckpointManager
    from forge.training.trainer import PPOTrainer, PPOTrainerConfig

    obs_dim = _compute_obs_dim(env)
    action_dim = env.action_space.n  # type: ignore[attr-defined]
    logger.info("Env obs_dim=%d, action_dim=%d", obs_dim, action_dim)

    mappo_config = MAPPOConfig.from_forge_config(config)
    agent = MAPPOAgent(mappo_config, obs_dim=obs_dim, action_dim=action_dim)

    trainer_config = PPOTrainerConfig.from_forge_config(config)
    trainer_config.checkpoint_dir = args.checkpoint_dir
    trainer = PPOTrainer(agent=agent, config=trainer_config)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    def reset_fn() -> np.ndarray:
        obs, _info = env.reset()  # type: ignore[attr-defined]
        return _flatten_obs(obs)

    def step_fn(action: int) -> tuple:
        obs, reward, terminated, truncated, info = env.step(action)  # type: ignore[attr-defined]
        return _flatten_obs(obs), reward, terminated, truncated, info

    logger.info("Starting MAPPO training: %d updates", args.num_updates)
    all_metrics = trainer.train(reset_fn, step_fn, num_updates=args.num_updates)

    if all_metrics:
        checkpoint_mgr.save(
            agent, episode=trainer.episode_count, metrics=all_metrics[-1]
        )
        logger.info("Final checkpoint saved to %s", args.checkpoint_dir)

    logger.info(
        "Training complete: %d updates, %d episodes, %d total steps",
        len(all_metrics),
        trainer.episode_count,
        trainer.total_steps,
    )
    if all_metrics:
        final = all_metrics[-1]
        logger.info(
            "Final metrics: policy_loss=%.4f, value_loss=%.4f, entropy=%.4f",
            final.get("policy_loss", 0.0),
            final.get("value_loss", 0.0),
            final.get("entropy", 0.0),
        )


def _train_basic(
    env: object,
    agent: object,
    args: argparse.Namespace,
) -> None:
    """Run episodes for non-learning agents (random, mcts)."""
    log_interval = max(1, args.episodes // 10)

    for episode in range(1, args.episodes + 1):
        obs, _info = env.reset()  # type: ignore[attr-defined]
        flat_obs = _flatten_obs(obs)
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _trace = agent.act(flat_obs)  # type: ignore[attr-defined]
            obs, reward, terminated, truncated, _info = env.step(action)  # type: ignore[attr-defined]
            flat_obs = _flatten_obs(obs)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        if episode % log_interval == 0:
            logger.info(
                "Episode %d/%d: reward=%.2f, steps=%d",
                episode,
                args.episodes,
                total_reward,
                steps,
            )

    logger.info("Training complete: %d episodes", args.episodes)


def main() -> None:
    """Run training loop."""
    args = parse_args()

    from forge.utils.logging_config import setup_logging  # noqa: PLC0415

    setup_logging(level=args.log_level)

    from forge.config import ForgeConfig  # noqa: PLC0415
    from forge.utils.seed import set_all_seeds  # noqa: PLC0415

    config = ForgeConfig.from_file(args.config)
    set_all_seeds(args.seed)

    logger.info(
        "Starting training: agent=%s, seed=%d, config=%s",
        args.agent,
        args.seed,
        args.config,
    )

    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    env = ForgeGymnasiumEnv(config=config.to_rust_config())

    if args.agent == "mappo":
        _train_mappo(env, config, args)
    elif args.agent == "random":
        from forge.agents.base_agent import AgentConfig  # noqa: PLC0415
        from forge.agents.random_agent import RandomAgent  # noqa: PLC0415

        action_dim = env.action_space.n
        agent = RandomAgent(
            config=AgentConfig(name="random"),
            action_space_size=action_dim,
            seed=args.seed,
        )
        _train_basic(env, agent, args)
    elif args.agent == "mcts":
        from forge.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: PLC0415

        action_dim = env.action_space.n
        agent = MCTSAgent(
            config=MCTSConfig(name="mcts"),
            action_space_size=action_dim,
            seed=args.seed,
        )
        _train_basic(env, agent, args)

    env.close()
    logger.info("Environment closed")


if __name__ == "__main__":
    main()
