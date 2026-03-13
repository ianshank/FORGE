"""CleanRL-style PPO training script for FORGE environments.

Implements Proximal Policy Optimisation (PPO) with Generalised Advantage
Estimation (GAE) in a single self-contained file, following the CleanRL
coding philosophy.  All hyperparameters are loaded from a TOML config file
and can be further overridden via CLI flags — no magic numbers anywhere.

Usage
-----
::

    # Default hyperparameters (configs/training/ppo_default.toml):
    python train_ppo_cleanrl.py

    # Custom config + overrides:
    python train_ppo_cleanrl.py \\
        --config configs/training/ppo_default.toml \\
        --total-timesteps 500000 \\
        --n-envs 4 \\
        --logger tensorboard \\
        --log-dir runs/forge_ppo

    # W&B logging:
    python train_ppo_cleanrl.py --logger wandb --wandb-project my-forge-runs

References
----------
- CleanRL: https://github.com/vwxyzjn/cleanrl
- PPO paper: Schulman et al., 2017 (https://arxiv.org/abs/1707.06347)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging setup — must come before any module imports so that structured
# log output is visible during import-time side effects.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — all guarded with clear error messages
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:
    logger.error("numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    logger.error("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from forge_env.vecenv import ForgeSyncVecEnv, make_forge_vec_env
    from forge_env.wrappers import FlattenObservationWrapper, RecordEpisodeStatistics, TimeLimit
except ImportError as exc:
    logger.error(
        "forge_env not found: %s\n"
        "Build with: cd crates/forge-python && maturin develop",
        exc,
    )
    sys.exit(1)

try:
    from forge_env.utils import seed_everything
except ImportError:
    def seed_everything(seed: int) -> None:  # type: ignore[misc]
        import random  # noqa: PLC0415
        random.seed(seed)
        np.random.seed(seed)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "training" / "ppo_default.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file; returns an empty dict if the file does not exist."""
    if not path.is_file():
        logger.warning("Config file not found: %s — using defaults", path)
        return {}
    try:
        import tomllib  # noqa: PLC0415
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _build_argparser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    hp = defaults.get("hyperparams", {})
    log_cfg = defaults.get("logging", {})
    curr_cfg = defaults.get("curriculum", {})

    p = argparse.ArgumentParser(
        description="CleanRL PPO for FORGE environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH,
                   help="Path to training TOML config file.")

    # Env
    p.add_argument("--env-width", type=int, default=32, help="Grid world width.")
    p.add_argument("--env-height", type=int, default=32, help="Grid world height.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--n-envs", type=int, default=hp.get("n_envs", 1),
                   help="Number of parallel envs.")
    p.add_argument("--seed", type=int, default=0, help="Global random seed.")

    # PPO hyperparams
    p.add_argument("--total-timesteps", type=int,
                   default=hp.get("total_timesteps", 1_000_000))
    p.add_argument("--learning-rate", type=float,
                   default=hp.get("learning_rate", 3e-4))
    p.add_argument("--n-steps", type=int, default=hp.get("n_steps", 2048),
                   help="Steps per env per rollout.")
    p.add_argument("--batch-size", type=int, default=hp.get("batch_size", 64))
    p.add_argument("--n-epochs", type=int, default=hp.get("n_epochs", 10))
    p.add_argument("--gamma", type=float, default=hp.get("gamma", 0.99))
    p.add_argument("--gae-lambda", type=float, default=hp.get("gae_lambda", 0.95))
    p.add_argument("--clip-range", type=float, default=hp.get("clip_range", 0.2))
    p.add_argument("--ent-coef", type=float, default=hp.get("ent_coef", 0.01))
    p.add_argument("--vf-coef", type=float, default=hp.get("vf_coef", 0.5))
    p.add_argument("--max-grad-norm", type=float,
                   default=hp.get("max_grad_norm", 0.5))

    # Logging
    p.add_argument("--log-freq", type=int, default=log_cfg.get("log_freq", 1000))
    p.add_argument("--logger", choices=["none", "wandb", "mlflow", "tensorboard"],
                   default="none", help="Experiment tracking backend.")
    p.add_argument("--wandb-project", type=str, default="forge-ppo",
                   help="W&B project name (only used when --logger=wandb).")
    p.add_argument("--log-dir", type=str, default="runs/forge_ppo",
                   help="TensorBoard / MLflow log directory.")

    # Curriculum
    p.add_argument("--curriculum", action="store_true",
                   default=curr_cfg.get("enabled", False))
    p.add_argument("--curriculum-target", type=float,
                   default=curr_cfg.get("target_success_rate", 0.7))
    p.add_argument("--curriculum-window", type=int,
                   default=curr_cfg.get("window_size", 100))

    return p


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------


class _ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for discrete action spaces.

    Args:
        obs_dim: Dimensionality of the flattened observation.
        action_dim: Number of discrete actions.
        hidden_sizes: Hidden layer sizes for the shared MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(current, h), nn.Tanh()])
            current = h
        self.shared = nn.Sequential(*layers)
        self.actor_head = nn.Linear(current, action_dim)
        self.critic_head = nn.Linear(current, 1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action, log_prob, entropy, value.

        Args:
            obs: Observation tensor of shape ``(batch, obs_dim)``.
            action: Optional pre-sampled action (used during PPO update).

        Returns:
            ``(action, log_prob, entropy, value)``
        """
        features = self.shared(obs)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic_head(features).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the value estimate for an observation.

        Args:
            obs: Observation tensor.

        Returns:
            Value tensor of shape ``(batch,)``.
        """
        return self.critic_head(self.shared(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the PPO training loop.

    Args:
        args: Parsed CLI arguments (see :func:`_build_argparser`).
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Build vectorised environment ---
    env_config = {
        "world": {"width": args.env_width, "height": args.env_height},
        "agents": {"num_agents": 1},
    }
    vec_env = make_forge_vec_env(
        config=env_config,
        n_envs=args.n_envs,
        seed=args.seed,
        wrapper_fns=[
            lambda e: TimeLimit(e, max_steps=args.max_steps),
            FlattenObservationWrapper,
            RecordEpisodeStatistics,
        ],
    )
    assert isinstance(vec_env, ForgeSyncVecEnv)
    action_dim = int(vec_env.action_space.n)

    # Probe obs_dim from a reset
    obs_batch, _ = vec_env.reset(seed=args.seed)
    obs_arr = obs_batch if isinstance(obs_batch, np.ndarray) else next(iter(obs_batch.values()))
    obs_dim = int(np.prod(obs_arr.shape[1:]))

    # --- Build model and optimiser ---
    model = _ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Optional experiment logger ---
    forge_logger = None
    if args.logger != "none":
        try:
            from forge.training.loggers import make_logger  # noqa: PLC0415
            forge_logger = make_logger(
                args.logger,
                **({"project": args.wandb_project} if args.logger == "wandb"
                   else {"experiment_name": "forge-ppo", "run_name": f"ppo-{args.seed}"}
                   if args.logger == "mlflow"
                   else {"log_dir": args.log_dir}),
            )
        except ImportError as exc:
            logger.warning("Could not initialise logger '%s': %s", args.logger, exc)

    # --- Rollout buffer dimensions ---
    rollout_steps = args.n_steps  # per env
    total_steps_per_update = rollout_steps * args.n_envs
    num_updates = args.total_timesteps // total_steps_per_update

    # Allocate rollout buffer tensors
    obs_buf = torch.zeros(rollout_steps, args.n_envs, obs_dim, device=device)
    actions_buf = torch.zeros(rollout_steps, args.n_envs, dtype=torch.long, device=device)
    log_probs_buf = torch.zeros(rollout_steps, args.n_envs, device=device)
    rewards_buf = torch.zeros(rollout_steps, args.n_envs, device=device)
    dones_buf = torch.zeros(rollout_steps, args.n_envs, device=device)
    values_buf = torch.zeros(rollout_steps, args.n_envs, device=device)

    global_step = 0
    start_time = time.perf_counter()
    episode_returns: list[float] = []

    # Flatten helper that also handles dict obs
    def _flatten(raw_obs: Any) -> torch.Tensor:
        if isinstance(raw_obs, dict):
            parts = [np.asarray(raw_obs[k], dtype=np.float32).reshape(args.n_envs, -1)
                     for k in sorted(raw_obs.keys())]
            flat = np.concatenate(parts, axis=-1)
        else:
            flat = np.asarray(raw_obs, dtype=np.float32).reshape(args.n_envs, -1)
        return torch.tensor(flat, device=device)

    current_obs = _flatten(obs_batch)
    current_done = torch.zeros(args.n_envs, device=device)

    for update in range(1, num_updates + 1):
        # --- Rollout collection ---
        for step in range(rollout_steps):
            global_step += args.n_envs
            obs_buf[step] = current_obs
            dones_buf[step] = current_done

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(current_obs)
                values_buf[step] = value

            actions_buf[step] = action
            log_probs_buf[step] = log_prob

            raw_obs, reward, terminated, truncated, infos = vec_env.step(
                action.cpu().numpy()
            )
            rewards_buf[step] = torch.tensor(reward, device=device)
            current_done = torch.tensor(
                (terminated | truncated).astype(np.float32), device=device
            )
            current_obs = _flatten(raw_obs)

            # Collect episode statistics
            for info in infos:
                if "episode" in info:
                    episode_returns.append(float(info["episode"]["r"]))

        # --- GAE computation ---
        with torch.no_grad():
            next_value = model.get_value(current_obs)

        advantages = torch.zeros_like(rewards_buf, device=device)
        last_gae = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(rollout_steps)):
            next_non_terminal = 1.0 - (
                current_done if t == rollout_steps - 1 else dones_buf[t + 1]
            )
            next_val = next_value if t == rollout_steps - 1 else values_buf[t + 1]
            delta = (
                rewards_buf[t]
                + args.gamma * next_val * next_non_terminal
                - values_buf[t]
            )
            last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_buf

        # --- PPO update ---
        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_actions = actions_buf.reshape(-1)
        flat_log_probs = log_probs_buf.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        for _ in range(args.n_epochs):
            # Shuffle mini-batches
            idx = torch.randperm(total_steps_per_update, device=device)
            for start in range(0, total_steps_per_update, args.batch_size):
                mb_idx = idx[start : start + args.batch_size]
                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                # Normalise advantages within mini-batch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                _, new_log_prob, entropy, new_value = model.get_action_and_value(
                    mb_obs, mb_actions
                )
                ratio = (new_log_prob - mb_old_log_probs).exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(
                    1.0 - args.clip_range, 1.0 + args.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                vf_loss = ((new_value - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy_loss

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimiser.step()

        # --- Logging ---
        if update % max(1, args.log_freq // total_steps_per_update) == 0:
            fps = int(global_step / (time.perf_counter() - start_time))
            mean_return = np.mean(episode_returns[-100:]) if episode_returns else 0.0
            logger.info(
                "update=%d global_step=%d fps=%d mean_ep_return=%.3f",
                update, global_step, fps, mean_return,
            )
            if forge_logger is not None:
                forge_logger.log(
                    {
                        "train/mean_ep_return": mean_return,
                        "train/fps": float(fps),
                        "train/policy_loss": float(pg_loss),
                        "train/value_loss": float(vf_loss),
                    },
                    step=global_step,
                )

    vec_env.close()
    if forge_logger is not None:
        forge_logger.close()
    logger.info("Training complete. Total timesteps: %d", global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and start training.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    # Load config file first so that defaults flow into argparser.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _load_toml(pre_args.config)

    parser = _build_argparser(config_defaults)
    args = parser.parse_args(argv)

    # FORGE_ env-var overrides (mirrors the Rust-side convention)
    for attr in vars(args):
        env_key = f"FORGE_TRAINING_{attr.upper().replace('-', '_')}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            current = getattr(args, attr)
            try:
                if isinstance(current, bool):
                    setattr(args, attr, env_val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(args, attr, int(env_val))
                elif isinstance(current, float):
                    setattr(args, attr, float(env_val))
                else:
                    setattr(args, attr, env_val)
            except ValueError:
                logger.warning("Invalid env override %s=%s; keeping default", env_key, env_val)

    train(args)


if __name__ == "__main__":
    main()
