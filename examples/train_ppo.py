"""PPO training script for FORGE environments.

Supports Stable Baselines3 (default) with optional W&B / MLflow logging.
Falls back to a random-action baseline when SB3 is not installed.

Usage::

    # Basic (SB3 required):
    python examples/train_ppo.py --timesteps 100000 --seed 42

    # With W&B:
    python examples/train_ppo.py --timesteps 100000 --wandb --wandb-project forge-ppo

    # With MLflow:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    python examples/train_ppo.py --timesteps 100000 --mlflow

    # Save metrics CSV:
    python examples/train_ppo.py --timesteps 100000 --output-dir runs/exp1

    # Random baseline (no SB3):
    python examples/train_ppo.py --seed 42

    # Smoke test (fast CI):
    python examples/train_ppo.py --timesteps 1000 --no-render
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback as SB3BaseCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: Stable Baselines3
# ---------------------------------------------------------------------------
try:
    import stable_baselines3 as _sb3  # noqa: F401  (availability flag only)

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ---------------------------------------------------------------------------
# FORGE imports
# ---------------------------------------------------------------------------
try:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
except ImportError:
    ForgeGymnasiumEnv = None  # type: ignore[assignment, misc]
    logger.warning(
        "ForgeGymnasiumEnv not available — run `maturin develop` in crates/forge-python"
    )

try:
    from forge_env.callbacks import (
        CompositeCallback,
        ConsoleCallback,
        CsvCallback,
        EpisodeStats,
        MLflowCallback,
        WandbCallback,
    )
    from forge_env.wrappers import (
        FlattenObservationWrapper,
        RecordEpisodeStatistics,
        TimeLimit,
    )

    FORGE_EXTRAS_AVAILABLE = True
except ImportError:
    FORGE_EXTRAS_AVAILABLE = False
    logger.warning("forge_env extras (callbacks/wrappers) not available.")


# ---------------------------------------------------------------------------
# SB3 bridge callback
# ---------------------------------------------------------------------------


class _ForgeCallbackBridge:  # noqa: N801  (acts like SB3 callback, not a data class)
    """Adapts SB3's BaseCallback to fire forge_env LoggingCallbacks on episode end.

    At class definition time SB3 may not be installed, so we do NOT inherit from
    BaseCallback statically.  The instance is only constructed inside
    ``train_with_sb3`` which is guarded by ``SB3_AVAILABLE``.
    """

    def __init__(self, forge_callback: Any, verbose: int = 0) -> None:
        self._forge_cb = forge_callback
        self._episode = 0
        self._ep_start_step = 0
        self._ep_start_time: float = 0.0
        self.n_calls: int = 0
        self.num_timesteps: int = 0
        self.locals: dict[str, Any] = {}

    def _on_training_start(self) -> None:
        import time

        self._ep_start_time = time.monotonic()
        self._forge_cb.on_training_start()

    def _on_step(self) -> bool:
        import time

        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                duration = time.monotonic() - self._ep_start_time or 1e-6
                fps = ep.get("l", 1) / duration
                stats = EpisodeStats(
                    episode=self._episode,
                    total_steps=self.num_timesteps,
                    episode_length=ep.get("l", 0),
                    episode_return=float(ep.get("r", 0.0)),
                    fps=fps,
                )
                self._forge_cb.on_episode_end(stats)
                self._episode += 1
                self._ep_start_time = time.monotonic()
        return True

    def _on_training_end(self) -> None:
        self._forge_cb.on_training_end()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(
    seed: int = 0,
    world_size: int = 32,
    num_agents: int = 1,
    max_steps: int = 500,
) -> Any:
    """Create a wrapped FORGE Gymnasium environment.

    Parameters
    ----------
    seed:           Random seed.
    world_size:     Width and height of the square world grid.
    num_agents:     Number of agents in the simulation.
    max_steps:      Episode truncation length (TimeLimit wrapper).

    Returns
    -------
    A wrapped gymnasium-compatible env, or ``None`` on import failure.
    """
    if ForgeGymnasiumEnv is None:
        logger.error("ForgeGymnasiumEnv not available; cannot create env.")
        return None

    config: dict[str, Any] = {
        "world": {"width": world_size, "height": world_size},
        "agents": {"num_agents": num_agents},
    }
    env: Any = ForgeGymnasiumEnv(config=config)
    env.reset(seed=seed)  # seed is not a constructor arg — pass to reset

    if FORGE_EXTRAS_AVAILABLE:
        env = TimeLimit(env, max_steps=max_steps)
        env = FlattenObservationWrapper(env)
        env = RecordEpisodeStatistics(env)

    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_with_sb3(args: argparse.Namespace) -> None:
    """Run SB3 PPO training with optional logging callbacks."""
    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 is not installed; cannot train.")
        sys.exit(1)

    env = make_env(
        seed=args.seed,
        world_size=args.world_size,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
    )
    if env is None:
        sys.exit(1)

    # Build callback chain
    callbacks: list[Any] = [ConsoleCallback(log_every=args.log_every)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    csv_cb = CsvCallback(output_path=csv_path)
    callbacks.append(csv_cb)

    if args.wandb:
        callbacks.append(
            WandbCallback(
                project=args.wandb_project,
                run_name=args.run_name,
                config=vars(args),
            )
        )
    if args.mlflow:
        callbacks.append(MLflowCallback(experiment_name=args.mlflow_experiment))

    composite = CompositeCallback(callbacks)
    sb3_bridge = _ForgeCallbackBridge(composite)

    logger.info("Training PPO for %d timesteps (seed=%d)…", args.timesteps, args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0 if args.no_render else 1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
    )
    model.learn(total_timesteps=args.timesteps, callback=sb3_bridge)
    model_path = output_dir / "ppo_forge.zip"
    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Quick evaluation
    if not args.no_render:
        _evaluate(model, args)

    env.close()


def _evaluate(model: Any, args: argparse.Namespace, n_episodes: int = 5) -> None:
    """Run evaluation episodes and print results."""
    eval_env = make_env(seed=args.seed + 1000, world_size=args.world_size)
    if eval_env is None:
        return

    rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        logger.info("Eval episode %d: return=%.3f", ep + 1, total)

    eval_env.close()
    print(f"\n{'─' * 40}")
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Mean return : {np.mean(rewards):.3f}")
    print(f"  Std return  : {np.std(rewards):.3f}")
    print(f"{'─' * 40}")


def run_random_baseline(args: argparse.Namespace) -> None:
    """Random-action baseline for when SB3 is not available."""
    print(
        "stable-baselines3 not installed. Running random baseline…\n"
        "Install SB3: pip install 'forge-env[sb3]'"
    )
    env = make_env(seed=args.seed)
    if env is None:
        return

    env.reset()
    episode_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(1, args.timesteps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        ep_reward += float(reward)
        if terminated or truncated:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            env.reset()

    if ep_reward != 0.0:
        episode_rewards.append(ep_reward)

    env.close()
    if episode_rewards:
        print(
            f"Random baseline over {len(episode_rewards)} episodes:\n"
            f"  Mean return: {np.mean(episode_rewards):.3f}\n"
            f"  Std return:  {np.std(episode_rewards):.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ── Environment
    env_g = parser.add_argument_group("Environment")
    env_g.add_argument("--world-size", type=int, default=32, metavar="N",
                       help="Square world grid size (default: 32)")
    env_g.add_argument("--num-agents", type=int, default=1, metavar="N")
    env_g.add_argument("--max-steps", type=int, default=500, metavar="N",
                       help="Episode truncation length (default: 500)")
    # ── Training
    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--timesteps", type=int, default=100_000,
                         help="Total SB3 timesteps (default: 100000)")
    train_g.add_argument("--seed", type=int, default=42)
    train_g.add_argument("--n-steps", type=int, default=256,
                         help="PPO rollout length per update (default: 256)")
    train_g.add_argument("--batch-size", type=int, default=64)
    train_g.add_argument("--n-epochs", type=int, default=4)
    train_g.add_argument("--learning-rate", type=float, default=3e-4)
    train_g.add_argument("--gamma", type=float, default=0.99)
    # ── Output
    out_g = parser.add_argument_group("Output")
    out_g.add_argument("--output-dir", type=Path, default=Path("runs/default"),
                       help="Directory for model checkpoints and metrics CSV")
    out_g.add_argument("--run-name", type=str, default=None)
    out_g.add_argument("--log-every", type=int, default=10,
                       help="Console log frequency in episodes (default: 10)")
    out_g.add_argument("--no-render", action="store_true",
                       help="Suppress SB3 verbose output and skip evaluation")
    # ── Logging
    log_g = parser.add_argument_group("Logging")
    log_g.add_argument("--wandb", action="store_true",
                       help="Enable W&B logging (requires wandb installed)")
    log_g.add_argument("--wandb-project", type=str, default="forge-ppo")
    log_g.add_argument("--mlflow", action="store_true",
                       help="Enable MLflow logging (requires mlflow installed)")
    log_g.add_argument("--mlflow-experiment", type=str, default="forge-ppo")
    log_g.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if SB3_AVAILABLE:
        train_with_sb3(args)
    else:
        run_random_baseline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
