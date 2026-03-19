"""PPO training template for FORGE environments.

Provides a ready-to-use skeleton for training a Proximal Policy Optimization
(PPO) agent on a FORGE environment using Stable Baselines3. When SB3 is not
installed, the script falls back to a random-action baseline so you can still
verify that the environment works correctly.

Usage:
    # With Stable Baselines3 installed:
    python train_ppo.py --timesteps 100000 --seed 42

    # Without SB3 (random baseline):
    python train_ppo.py --seed 42

    # With custom world size and episode length:
    python train_ppo.py --width 64 --height 64 --max-steps 1000
"""

import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- Optional: Stable Baselines3 -------------------------------------------
try:
    from stable_baselines3 import PPO

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None

# --- FORGE environment ------------------------------------------------------
try:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
except ImportError:
    ForgeGymnasiumEnv = None
    logger.warning(
        "Could not import ForgeGymnasiumEnv from forge_env.gymnasium_env. "
        "Make sure the forge-python crate is built and installed: "
        "cd crates/forge-python && maturin develop"
    )

# --- FORGE wrappers ---------------------------------------------------------
try:
    from forge_env.wrappers import (
        FlattenObservationWrapper,
        RecordEpisodeStatistics,
        TimeLimit,
    )
except ImportError:
    FlattenObservationWrapper = None
    RecordEpisodeStatistics = None
    TimeLimit = None
    logger.warning(
        "Could not import wrappers from forge_env.wrappers. "
        "FlattenObservationWrapper, TimeLimit, and RecordEpisodeStatistics "
        "will not be available."
    )

# --- Default hyperparameters (override via CLI args) ------------------------
_DEFAULT_WORLD_WIDTH = 32
_DEFAULT_WORLD_HEIGHT = 32
_DEFAULT_NUM_AGENTS = 1
_DEFAULT_MAX_STEPS = 500
_DEFAULT_PPO_N_STEPS = 256
_DEFAULT_PPO_BATCH_SIZE = 64
_DEFAULT_PPO_N_EPOCHS = 4
_DEFAULT_PPO_LEARNING_RATE = 3e-4
_DEFAULT_TIMESTEPS = 50_000
_DEFAULT_EVAL_EPISODES = 5
_DEFAULT_BASELINE_STEPS = 1000
_DEFAULT_SEED = 42
_DEFAULT_EVAL_SEED_OFFSET = 1000


def make_env(
    seed: int = 0,
    width: int = _DEFAULT_WORLD_WIDTH,
    height: int = _DEFAULT_WORLD_HEIGHT,
    num_agents: int = _DEFAULT_NUM_AGENTS,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> object | None:
    """Create a FORGE Gymnasium environment with standard wrappers.

    The environment is wrapped with:
    - TimeLimit: caps each episode at ``max_steps`` steps.
    - FlattenObservationWrapper: flattens nested observations into a 1-D array.
    - RecordEpisodeStatistics: tracks episode return and length.

    Args:
        seed: Random seed for reproducibility.
        width: World grid width.
        height: World grid height.
        num_agents: Number of agents in the environment.
        max_steps: Maximum steps per episode (TimeLimit).

    Returns:
        A wrapped ForgeGymnasiumEnv instance, or None if imports failed.
    """
    if ForgeGymnasiumEnv is None:
        logger.error("ForgeGymnasiumEnv is not available. Cannot create environment.")
        return None

    config = {
        "world": {
            "width": width,
            "height": height,
        },
        "agents": {
            "num_agents": num_agents,
        },
    }

    env = ForgeGymnasiumEnv(config=config, seed=seed)

    if TimeLimit is not None:
        env = TimeLimit(env, max_steps=max_steps)
    if FlattenObservationWrapper is not None:
        env = FlattenObservationWrapper(env)
    if RecordEpisodeStatistics is not None:
        env = RecordEpisodeStatistics(env)

    return env


def train_with_sb3(args: argparse.Namespace) -> None:
    """Train a PPO agent using Stable Baselines3.

    Args:
        args: Parsed CLI arguments containing all hyperparameters.
    """
    env = make_env(
        seed=args.seed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    if env is None:
        return

    logger.info("Training PPO for %d timesteps (seed=%d)...", args.timesteps, args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
    )

    model.learn(total_timesteps=args.timesteps)

    # Evaluate the trained agent
    logger.info("--- Evaluation ---")
    eval_env = make_env(
        seed=args.seed + _DEFAULT_EVAL_SEED_OFFSET,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    if eval_env is None:
        return

    episode_rewards = []
    for ep in range(args.eval_episodes):
        obs, _info = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        logger.info("  Episode %d: reward = %.3f", ep + 1, total_reward)

    eval_env.close()
    env.close()

    logger.info("Mean evaluation reward: %.3f", np.mean(episode_rewards))
    logger.info("Std evaluation reward:  %.3f", np.std(episode_rewards))


def run_random_baseline(args: argparse.Namespace) -> None:
    """Run a random-action baseline when SB3 is not available.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info(
        "Stable Baselines3 is not installed. To train with PPO, install it: "
        "pip install stable-baselines3"
    )
    logger.info("Running random baseline for %d steps instead...", args.baseline_steps)

    env = make_env(
        seed=args.seed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    if env is None:
        return

    _obs, _info = env.reset()
    episode_rewards: list[float] = []
    current_episode_reward = 0.0

    for _step in range(1, args.baseline_steps + 1):
        action = env.action_space.sample()
        _obs, reward, terminated, truncated, _info = env.step(action)
        current_episode_reward += reward

        if terminated or truncated:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            _obs, _info = env.reset()

    # Account for any in-progress episode
    if current_episode_reward != 0.0:
        episode_rewards.append(current_episode_reward)

    env.close()

    if episode_rewards:
        logger.info("--- Random Baseline Results ---")
        logger.info("  Episodes completed: %d", len(episode_rewards))
        logger.info("  Mean episode reward: %.3f", np.mean(episode_rewards))
        logger.info("  Std episode reward:  %.3f", np.std(episode_rewards))
        logger.info("  Min episode reward:  %.3f", np.min(episode_rewards))
        logger.info("  Max episode reward:  %.3f", np.max(episode_rewards))
    else:
        logger.warning("No episodes completed within the given steps.")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all configurable hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a FORGE environment (or run a random baseline)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=_DEFAULT_TIMESTEPS,
        help=f"Total training timesteps for PPO (default: {_DEFAULT_TIMESTEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=_DEFAULT_WORLD_WIDTH,
        help=f"World grid width (default: {_DEFAULT_WORLD_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=_DEFAULT_WORLD_HEIGHT,
        help=f"World grid height (default: {_DEFAULT_WORLD_HEIGHT}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {_DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=_DEFAULT_PPO_N_STEPS,
        help=f"PPO rollout buffer size (default: {_DEFAULT_PPO_N_STEPS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_PPO_BATCH_SIZE,
        help=f"PPO minibatch size (default: {_DEFAULT_PPO_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=_DEFAULT_PPO_N_EPOCHS,
        help=f"PPO optimization epochs per update (default: {_DEFAULT_PPO_N_EPOCHS}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_DEFAULT_PPO_LEARNING_RATE,
        help=f"PPO learning rate (default: {_DEFAULT_PPO_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=_DEFAULT_EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {_DEFAULT_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--baseline-steps",
        type=int,
        default=_DEFAULT_BASELINE_STEPS,
        help=f"Steps for random baseline (default: {_DEFAULT_BASELINE_STEPS}).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()

    if SB3_AVAILABLE:
        train_with_sb3(args)
    else:
        run_random_baseline(args)
