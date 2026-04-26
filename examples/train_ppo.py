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

    # With metrics callback and TensorBoard logging:
    python train_ppo.py --timesteps 100000 --logger tensorboard --log-dir runs/ppo
"""

import argparse
from typing import Any

import numpy as np

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
    print(
        "WARNING: Could not import ForgeGymnasiumEnv from forge_env.gymnasium_env.\n"
        "Make sure the forge-python crate is built and installed:\n"
        "  cd crates/forge-python && maturin develop\n"
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
    print(
        "WARNING: Could not import wrappers from forge_env.wrappers.\n"
        "FlattenObservationWrapper, TimeLimit, and RecordEpisodeStatistics "
        "will not be available.\n"
    )

# --- FORGE SB3 callbacks (optional) ----------------------------------------
try:
    from forge_env.sb3_callbacks import ForgeMetricsCallback

    _HAS_CALLBACKS = True
except ImportError:
    _HAS_CALLBACKS = False


def make_env(seed: int = 0, max_steps: int = 500):
    """Create a FORGE Gymnasium environment with standard wrappers.

    The environment is wrapped with:
    - TimeLimit: caps each episode at ``max_steps`` steps.
    - FlattenObservationWrapper: flattens nested observations into a 1-D array.
    - RecordEpisodeStatistics: tracks episode return and length.

    Args:
        seed: Random seed for reproducibility.
        max_steps: Maximum steps per episode before truncation.

    Returns:
        A wrapped ForgeGymnasiumEnv instance, or None if imports failed.
    """
    if ForgeGymnasiumEnv is None:
        print("ForgeGymnasiumEnv is not available. Cannot create environment.")
        return None

    config = {
        "world": {
            "width": 32,
            "height": 32,
        },
        "agents": {
            "num_agents": 1,
        },
    }

    env = ForgeGymnasiumEnv(config=config)
    env.reset(seed=seed)

    if TimeLimit is not None:
        env = TimeLimit(env, max_steps=max_steps)
    if FlattenObservationWrapper is not None:
        env = FlattenObservationWrapper(env)
    if RecordEpisodeStatistics is not None:
        env = RecordEpisodeStatistics(env)

    return env


def train_with_sb3(timesteps: int, seed: int, logger_backend: str = "none", log_dir: str = "runs/ppo") -> None:
    """Train a PPO agent using Stable Baselines3.

    Args:
        timesteps: Total number of training timesteps.
        seed: Random seed for reproducibility.
        logger_backend: Experiment logger (``"none"``, ``"tensorboard"``,
            ``"wandb"``, or ``"mlflow"``).
        log_dir: Directory for TensorBoard / MLflow logs.
    """
    env = make_env(seed=seed)
    if env is None:
        return

    # Build optional ForgeMetricsCallback
    callbacks = []
    if _HAS_CALLBACKS:
        forge_logger = None
        if logger_backend != "none":
            try:
                from forge.training.loggers import make_logger
                logger_kwargs: dict[str, Any]
                if logger_backend == "tensorboard":
                    logger_kwargs = {"log_dir": log_dir}
                elif logger_backend == "wandb":
                    logger_kwargs = {"project": "forge-ppo"}
                else:  # mlflow
                    logger_kwargs = {"experiment_name": "forge-ppo"}
                forge_logger = make_logger(logger_backend, **logger_kwargs)
            except ImportError as exc:
                print(f"Logger '{logger_backend}' unavailable: {exc}")
        callbacks.append(ForgeMetricsCallback(forge_logger=forge_logger, log_freq=1000))

    print(f"Training PPO for {timesteps} timesteps (seed={seed})...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=timesteps, callback=callbacks or None)

    # Evaluate the trained agent for a few episodes
    print("\n--- Evaluation ---")
    eval_env = make_env(seed=seed + 1000)
    if eval_env is None:
        return

    episode_rewards = []
    for ep in range(5):
        obs, _info = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.3f}")

    eval_env.close()
    env.close()

    print(f"\nMean evaluation reward: {np.mean(episode_rewards):.3f}")
    print(f"Std evaluation reward:  {np.std(episode_rewards):.3f}")


def run_random_baseline(seed, num_steps=1000):
    """Run a random-action baseline when SB3 is not available.

    Args:
        seed: Random seed for reproducibility.
        num_steps: Number of steps to run.
    """
    print(
        "Stable Baselines3 is not installed. To train with PPO, install it:\n"
        "  pip install stable-baselines3\n"
    )
    print(f"Running random baseline for {num_steps} steps instead...\n")

    env = make_env(seed=seed)
    if env is None:
        return

    _obs, _info = env.reset()
    episode_rewards = []
    current_episode_reward = 0.0

    for _step in range(1, num_steps + 1):
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
        print("--- Random Baseline Results ---")
        print(f"  Episodes completed: {len(episode_rewards)}")
        print(f"  Mean episode reward: {np.mean(episode_rewards):.3f}")
        print(f"  Std episode reward:  {np.std(episode_rewards):.3f}")
        print(f"  Min episode reward:  {np.min(episode_rewards):.3f}")
        print(f"  Max episode reward:  {np.max(episode_rewards):.3f}")
    else:
        print("No episodes completed within the given steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a FORGE environment (or run a random baseline)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps for PPO (default: 50000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--logger",
        choices=["none", "tensorboard", "wandb", "mlflow"],
        default="none",
        help="Experiment tracking backend (default: none).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/ppo",
        help="Directory for TensorBoard/MLflow logs (default: runs/ppo).",
    )
    args = parser.parse_args()

    if SB3_AVAILABLE:
        train_with_sb3(
            timesteps=args.timesteps,
            seed=args.seed,
            logger_backend=args.logger,
            log_dir=args.log_dir,
        )
    else:
        run_random_baseline(seed=args.seed)
