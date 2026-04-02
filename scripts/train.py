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
from typing import Any

# Add python/ to path so forge and forge_env packages are importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logger = logging.getLogger("forge.train")

# --- CLI argument defaults (no magic numbers) ---
_DEFAULT_EPISODES = 100
_DEFAULT_NUM_UPDATES = 10
_DEFAULT_SEED = 42
_DEFAULT_CHECKPOINT_DIR = "checkpoints"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_MAX_EPISODE_STEPS = 1000
_DEFAULT_DASHBOARD_URL = ""
_AGENT_CHOICES = ("random", "mcts", "mappo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a FORGE agent")
    parser.add_argument("--config", type=str, default="forge.toml", help="Path to config file")
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=list(_AGENT_CHOICES),
        help="Agent type to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=_DEFAULT_EPISODES,
        help="Training episodes",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=_DEFAULT_NUM_UPDATES,
        help="PPO update iterations (mappo only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=_DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=_DEFAULT_LOG_LEVEL,
        help="Logging level",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=_DEFAULT_DASHBOARD_URL,
        help="URL of forge-server for live dashboard metrics (e.g. http://localhost:8080)",
    )
    return parser.parse_args(argv)


def _create_env(config: Any) -> Any:
    """Create and return a ForgeGymnasiumEnv from config.

    Args:
        config: A ForgeConfig instance.

    Returns:
        A ForgeGymnasiumEnv environment.

    Raises:
        RuntimeError: If environment creation fails.
    """
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    try:
        rust_config = config.to_rust_config()
        env = ForgeGymnasiumEnv(config=rust_config)
    except Exception as exc:
        msg = f"Failed to create environment: {exc}"
        raise RuntimeError(msg) from exc

    logger.info("Environment created: action_space=%s", env.action_space)
    return env


def _train_mappo(env: Any, config: Any, args: argparse.Namespace) -> None:
    """Train a MAPPO agent with PPO rollout collection and updates.

    Args:
        env: A ForgeGymnasiumEnv instance.
        config: A ForgeConfig instance.
        args: Parsed CLI arguments.
    """
    import numpy as np  # noqa: PLC0415, TC002
    from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig  # noqa: PLC0415
    from forge.training.checkpointing import CheckpointManager  # noqa: PLC0415
    from forge.training.trainer import PPOTrainer, PPOTrainerConfig  # noqa: PLC0415
    from forge.utils.observation import compute_obs_dim, flatten_obs  # noqa: PLC0415

    obs_dim = compute_obs_dim(env)
    action_dim: int = env.action_space.n
    logger.info("Env obs_dim=%d, action_dim=%d", obs_dim, action_dim)

    mappo_config = MAPPOConfig.from_forge_config(config)
    agent = MAPPOAgent(mappo_config, obs_dim=obs_dim, action_dim=action_dim)

    trainer_config = PPOTrainerConfig.from_forge_config(config)
    trainer_config.checkpoint_dir = args.checkpoint_dir
    trainer = PPOTrainer(agent=agent, config=trainer_config)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    def reset_fn() -> np.ndarray:
        obs, _info = env.reset()
        return flatten_obs(obs)

    def step_fn(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = env.step(action)
        return flatten_obs(obs), reward, terminated, truncated, info

    # Optional dashboard client for live metrics streaming
    dashboard = None
    if getattr(args, "dashboard_url", ""):
        from forge.utils.dashboard_client import DashboardClient  # noqa: PLC0415

        dashboard = DashboardClient(args.dashboard_url)

    logger.info("Starting MAPPO training: %d updates", args.num_updates)
    all_metrics = trainer.train(reset_fn, step_fn, num_updates=args.num_updates)

    # Post each update's metrics to the dashboard
    for metrics in all_metrics:
        if dashboard is not None:
            dashboard.post_training_metrics(
                episode=int(metrics.get("episodes", 0)),
                total_steps=int(metrics.get("total_steps", 0)),
                mean_reward=metrics.get("mean_reward", 0.0),
                loss_policy=metrics.get("policy_loss", 0.0),
                loss_value=metrics.get("value_loss", 0.0),
                entropy=metrics.get("entropy", 0.0),
            )

    if all_metrics:
        checkpoint_mgr.save(agent, episode=trainer.episode_count, metrics=all_metrics[-1])
        logger.info("Final checkpoint saved to %s", args.checkpoint_dir)

    if dashboard is not None:
        dashboard.close()

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
    env: Any,
    agent: Any,
    args: argparse.Namespace,
) -> None:
    """Run episodes for non-learning agents (random, mcts).

    Args:
        env: A Gymnasium-compatible environment.
        agent: An agent implementing ``act(obs) -> (action, trace)``.
        args: Parsed CLI arguments.
    """
    from forge.utils.observation import flatten_obs  # noqa: PLC0415

    dashboard = None
    if getattr(args, "dashboard_url", ""):
        from forge.utils.dashboard_client import DashboardClient  # noqa: PLC0415

        dashboard = DashboardClient(args.dashboard_url)

    log_interval = max(1, args.episodes // 10)
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        obs, _info = env.reset()
        flat_obs = flatten_obs(obs)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < _DEFAULT_MAX_EPISODE_STEPS:
            action, _trace = agent.act(flat_obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            flat_obs = flatten_obs(obs)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        total_steps += steps

        if episode % log_interval == 0:
            logger.info(
                "Episode %d/%d: reward=%.2f, steps=%d",
                episode,
                args.episodes,
                total_reward,
                steps,
            )
            if dashboard is not None:
                dashboard.post_training_metrics(
                    episode=episode,
                    total_steps=total_steps,
                    mean_reward=total_reward,
                )

    if dashboard is not None:
        dashboard.close()

    logger.info("Training complete: %d episodes", args.episodes)


def main(argv: list[str] | None = None) -> None:
    """Run the FORGE training pipeline.

    Args:
        argv: Optional argument list for testing. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)

    from forge.utils.logging_config import setup_logging  # noqa: PLC0415

    setup_logging(level=args.log_level)

    from forge.config import ForgeConfig  # noqa: PLC0415
    from forge.utils.seed import set_all_seeds  # noqa: PLC0415

    try:
        config = ForgeConfig.from_file(args.config)
    except Exception as exc:
        logger.error("Failed to load config from '%s': %s", args.config, exc)
        sys.exit(1)

    set_all_seeds(args.seed)

    logger.info(
        "Starting training: agent=%s, seed=%d, config=%s",
        args.agent,
        args.seed,
        args.config,
    )

    env = _create_env(config)

    try:
        if args.agent == "mappo":
            _train_mappo(env, config, args)
        elif args.agent == "random":
            from forge.agents.base_agent import AgentConfig  # noqa: PLC0415
            from forge.agents.random_agent import RandomAgent  # noqa: PLC0415

            action_dim: int = env.action_space.n
            agent = RandomAgent(
                config=AgentConfig(name="random"),
                action_space_size=action_dim,
                seed=args.seed,
            )
            _train_basic(env, agent, args)
        elif args.agent == "mcts":
            from forge.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: PLC0415

            action_dim = env.action_space.n
            mcts_agent = MCTSAgent(
                config=MCTSConfig(name="mcts"),
                action_space_size=action_dim,
                seed=args.seed,
            )
            _train_basic(env, mcts_agent, args)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
    finally:
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()
