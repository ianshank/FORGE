"""CleanRL-style SAC training script for FORGE environments.

Implements Soft Actor-Critic (SAC) with a replay buffer in a single
self-contained file following the CleanRL philosophy.  All hyperparameters
flow from a TOML config file and CLI flags — no magic numbers.

SAC is an off-policy algorithm well-suited for environments with dense
rewards.  This implementation uses a discrete action variant (Christodoulou,
2019) that avoids the need for a continuous action space.

Usage
-----
::

    # Default hyperparameters (configs/training/sac_default.toml):
    python train_sac_cleanrl.py

    # Override + TensorBoard logging:
    python train_sac_cleanrl.py \\
        --total-timesteps 500000 \\
        --learning-rate 1e-4 \\
        --logger tensorboard \\
        --log-dir runs/forge_sac

References
----------
- SAC paper: Haarnoja et al., 2018 (https://arxiv.org/abs/1801.01290)
- Discrete SAC: Christodoulou, 2019 (https://arxiv.org/abs/1910.07207)
- CleanRL: https://github.com/vwxyzjn/cleanrl
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    logger.error("numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    import torch.optim as optim
except ImportError:
    logger.error("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.wrappers import FlattenObservationWrapper, RecordEpisodeStatistics, TimeLimit
except ImportError as exc:
    logger.error(
        "forge_env not found: %s\nBuild with: cd crates/forge-python && maturin develop",
        exc,
    )
    sys.exit(1)

try:
    from forge_env.utils import seed_everything
except ImportError:
    def seed_everything(seed: int) -> None:  # type: ignore[misc]
        random.seed(seed)
        np.random.seed(seed)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "training" / "sac_default.toml"


def _load_toml(path: Path) -> dict[str, Any]:
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

    p = argparse.ArgumentParser(
        description="CleanRL discrete SAC for FORGE environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)

    # Env
    p.add_argument("--env-width", type=int, default=32)
    p.add_argument("--env-height", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    # SAC hyperparams
    p.add_argument("--total-timesteps", type=int,
                   default=hp.get("total_timesteps", 1_000_000))
    p.add_argument("--learning-rate", type=float,
                   default=hp.get("learning_rate", 3e-4))
    p.add_argument("--buffer-size", type=int,
                   default=hp.get("buffer_size", 1_000_000))
    p.add_argument("--batch-size", type=int, default=hp.get("batch_size", 256))
    p.add_argument("--learning-starts", type=int,
                   default=hp.get("learning_starts", 1000))
    p.add_argument("--gamma", type=float, default=hp.get("gamma", 0.99))
    p.add_argument("--tau", type=float, default=hp.get("tau", 0.005))
    p.add_argument("--train-freq", type=int, default=hp.get("train_freq", 1))
    p.add_argument("--gradient-steps", type=int,
                   default=hp.get("gradient_steps", 1))
    p.add_argument("--ent-coef", type=str,
                   default=str(hp.get("ent_coef", "auto")),
                   help="Entropy coef: 'auto' or a float.")
    p.add_argument("--target-entropy", type=str,
                   default=str(hp.get("target_entropy", "auto")),
                   help="Target entropy: 'auto' or a float.")

    # Logging
    p.add_argument("--log-freq", type=int, default=log_cfg.get("log_freq", 1000))
    p.add_argument("--logger", choices=["none", "wandb", "mlflow", "tensorboard"],
                   default="none")
    p.add_argument("--wandb-project", type=str, default="forge-sac")
    p.add_argument("--log-dir", type=str, default="runs/forge_sac")

    return p


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class _ReplayBuffer:
    """Simple circular replay buffer for off-policy SAC.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of the flattened observation.
        device: PyTorch device.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self._capacity = capacity
        self._device = device
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: float,
    ) -> None:
        """Add a single transition."""
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict of tensors: obs, next_obs, actions, rewards, dones.
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.tensor(self._obs[idx], device=self._device),
            "next_obs": torch.tensor(self._next_obs[idx], device=self._device),
            "actions": torch.tensor(self._actions[idx], dtype=torch.long, device=self._device),
            "rewards": torch.tensor(self._rewards[idx], device=self._device),
            "dones": torch.tensor(self._dones[idx], device=self._device),
        }

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Discrete SAC networks
# ---------------------------------------------------------------------------


class _SoftQNetwork(nn.Module):
    """Q-network returning Q-values for all actions simultaneously.

    Args:
        obs_dim: Flattened observation dimension.
        action_dim: Number of discrete actions.
        hidden_sizes: Hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(current, h), nn.ReLU()])
            current = h
        layers.append(nn.Linear(current, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape ``(batch, action_dim)``."""
        return self.net(obs)


class _Actor(nn.Module):
    """Stochastic policy returning action probabilities.

    Args:
        obs_dim: Flattened observation dimension.
        action_dim: Number of discrete actions.
        hidden_sizes: Hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(current, h), nn.ReLU()])
            current = h
        layers.append(nn.Linear(current, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action probabilities, log-probabilities, and sampled action.

        Args:
            obs: Observation tensor.

        Returns:
            ``(action_probs, log_probs, actions)``
        """
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)  # Greedy for evaluation; stochastic handled in loss
        return probs, log_probs, actions


# ---------------------------------------------------------------------------
# SAC training loop
# ---------------------------------------------------------------------------


def _flatten_obs(raw: Any, n_envs: int = 1) -> np.ndarray:
    """Flatten a (possibly dict) observation to a 1-D float32 array."""
    if isinstance(raw, dict):
        parts = [np.asarray(raw[k], dtype=np.float32).ravel() for k in sorted(raw.keys())]
        return np.concatenate(parts)
    return np.asarray(raw, dtype=np.float32).ravel()


def train(args: argparse.Namespace) -> None:
    """Run the discrete SAC training loop.

    Args:
        args: Parsed CLI arguments.
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Build single environment ---
    env_config = {
        "world": {"width": args.env_width, "height": args.env_height},
        "agents": {"num_agents": 1},
    }
    env: Any = ForgeGymnasiumEnv(config=env_config)
    env = TimeLimit(env, max_steps=args.max_steps)
    env = FlattenObservationWrapper(env)
    env = RecordEpisodeStatistics(env)

    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(np.asarray(obs).shape))
    action_dim = int(env.action_space.n)

    replay_buffer = _ReplayBuffer(args.buffer_size, obs_dim, device)

    actor = _Actor(obs_dim, action_dim).to(device)
    qf1 = _SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = _SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = _SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = _SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optim = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )

    # Auto temperature (entropy coef)
    auto_ent = args.ent_coef == "auto"
    if auto_ent:
        target_entropy: float = (
            -float(action_dim) if args.target_entropy == "auto" else float(args.target_entropy)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = float(args.ent_coef)
        log_alpha = None  # type: ignore[assignment]
        alpha_optim = None  # type: ignore[assignment]
        target_entropy = 0.0  # unused

    # --- Optional experiment logger ---
    forge_logger = None
    if args.logger != "none":
        try:
            from forge.training.loggers import make_logger  # noqa: PLC0415
            forge_logger = make_logger(
                args.logger,
                **({"project": args.wandb_project} if args.logger == "wandb"
                   else {"experiment_name": "forge-sac", "run_name": f"sac-{args.seed}"}
                   if args.logger == "mlflow"
                   else {"log_dir": args.log_dir}),
            )
        except ImportError as exc:
            logger.warning("Could not initialise logger '%s': %s", args.logger, exc)

    start_time = time.perf_counter()
    episode_returns: deque[float] = deque(maxlen=100)
    current_obs = _flatten_obs(obs)

    for global_step in range(1, args.total_timesteps + 1):
        # --- Action selection ---
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(current_obs, device=device).unsqueeze(0)
            with torch.no_grad():
                probs, _, _ = actor(obs_t)
                action = int(torch.multinomial(probs, 1).item())

        next_raw, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)
        next_obs_arr = _flatten_obs(next_raw)

        replay_buffer.add(current_obs, next_obs_arr, action, float(reward), done)

        if done:
            if "episode" in info:
                episode_returns.append(float(info["episode"]["r"]))
            current_obs = _flatten_obs(env.reset(seed=None)[0])
        else:
            current_obs = next_obs_arr

        # --- SAC update ---
        if global_step >= args.learning_starts and global_step % args.train_freq == 0:
            for _ in range(args.gradient_steps):
                batch = replay_buffer.sample(args.batch_size)
                obs_b = batch["obs"]
                next_obs_b = batch["next_obs"]
                actions_b = batch["actions"]
                rewards_b = batch["rewards"]
                dones_b = batch["dones"]

                with torch.no_grad():
                    next_probs, next_log_probs, _ = actor(next_obs_b)
                    qf1_next = qf1_target(next_obs_b)
                    qf2_next = qf2_target(next_obs_b)
                    min_q_next = torch.min(qf1_next, qf2_next)
                    # Expectation over next-state distribution
                    v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=-1)
                    td_target = rewards_b + (1.0 - dones_b) * args.gamma * v_next

                qf1_vals = qf1(obs_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                qf2_vals = qf2(obs_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                qf_loss = F.mse_loss(qf1_vals, td_target) + F.mse_loss(qf2_vals, td_target)

                q_optim.zero_grad()
                qf_loss.backward()
                q_optim.step()

                # Actor update
                probs_b, log_probs_b, _ = actor(obs_b)
                with torch.no_grad():
                    min_q = torch.min(qf1(obs_b), qf2(obs_b))
                actor_loss = (probs_b * (alpha * log_probs_b - min_q)).sum(dim=-1).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # Temperature update
                if auto_ent and log_alpha is not None and alpha_optim is not None:
                    with torch.no_grad():
                        _, log_pi, _ = actor(obs_b)
                        log_pi_a = (probs_b * log_pi).sum(dim=-1)
                    alpha_loss = (-log_alpha.exp() * (log_pi_a + target_entropy)).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()

                # Soft target update
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # --- Logging ---
        if global_step % args.log_freq == 0:
            fps = int(global_step / (time.perf_counter() - start_time))
            mean_return = float(np.mean(list(episode_returns))) if episode_returns else 0.0
            logger.info(
                "step=%d fps=%d mean_ep_return=%.3f alpha=%.4f",
                global_step, fps, mean_return, alpha,
            )
            if forge_logger is not None:
                forge_logger.log(
                    {
                        "train/mean_ep_return": mean_return,
                        "train/fps": float(fps),
                        "train/alpha": alpha,
                    },
                    step=global_step,
                )

    env.close()
    if forge_logger is not None:
        forge_logger.close()
    logger.info("SAC training complete. Total steps: %d", args.total_timesteps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and start training.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _load_toml(pre_args.config)

    parser = _build_argparser(config_defaults)
    args = parser.parse_args(argv)

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
