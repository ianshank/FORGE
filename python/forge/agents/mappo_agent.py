"""MAPPO (Multi-Agent PPO) agent implementation.

Uses an ActorCriticNetwork with PPO clipped surrogate objective
and GAE advantage estimation. All hyperparameters flow through config.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.models.policy_network import ActorCriticNetwork
from forge.utils.device import get_device
from forge_env.wrappers import DEFAULT_REWARD_EPSILON

logger = logging.getLogger(__name__)

DEFAULT_OBS_DIM = 64
DEFAULT_ACTION_DIM = 8


@dataclass
class MAPPOConfig(AgentConfig):
    """Configuration for MAPPO agent — extends AgentConfig."""

    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    batch_size: int = 256
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"

    @classmethod
    def from_forge_config(cls, forge_config: Any) -> MAPPOConfig:
        """Build MAPPOConfig from a ForgeConfig instance."""
        tc = forge_config.training
        return cls(
            learning_rate=tc.learning_rate,
            gamma=tc.gamma,
            gae_lambda=tc.gae_lambda,
            clip_ratio=tc.clip_ratio,
            epochs=tc.epochs,
            batch_size=tc.batch_size,
            entropy_coeff=tc.entropy_coeff,
            value_coeff=tc.value_coeff,
            max_grad_norm=tc.max_grad_norm,
        )


class MAPPOAgent(BaseAgent):
    """Multi-Agent PPO agent with actor-critic architecture.

    Implements the PPO clipped surrogate objective with GAE
    advantage estimation. Compatible with the BaseAgent interface.

    Args:
        config: MAPPOConfig with all hyperparameters.
        obs_dim: Observation dimensionality (overrides config if provided).
        action_dim: Number of discrete actions (overrides config if provided).
    """

    def __init__(
        self,
        config: MAPPOConfig,
        obs_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        super().__init__(config)
        self.mappo_config = config
        self._obs_dim = obs_dim or config.obs_dim
        self._action_dim = action_dim or config.action_dim

        device = config.device
        if device == "auto":
            device = get_device()

        self.network = ActorCriticNetwork(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_sizes=config.hidden_sizes,
            learning_rate=config.learning_rate,
            device=device,
        )
        self._device = device
        logger.info(
            "MAPPOAgent initialized: obs_dim=%d, action_dim=%d, device=%s",
            self._obs_dim,
            self._action_dim,
            device,
        )

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality."""
        return self._obs_dim

    @property
    def device(self) -> str:
        """Compute device."""
        return self._device

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action using the actor-critic policy.

        Args:
            observation: Flat observation array.

        Returns:
            (action_id, info_dict) where info_dict contains log_prob and value.
        """
        import torch  # noqa: PLC0415

        self.network.eval_mode()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=torch.device(self._device)
            ).unsqueeze(0)
            action, log_prob, entropy, value = self.network.get_action_and_value(obs_tensor)

        self._step_count += 1
        return int(action.item()), {
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    def act_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch action selection for vectorized environments.

        Args:
            observations: Array of shape (batch, obs_dim).
            deterministic: If True, use argmax instead of sampling.

        Returns:
            actions: Shape (batch,).
            log_probs: Shape (batch,).
            entropies: Shape (batch,).
            values: Shape (batch,).
        """
        import torch  # noqa: PLC0415

        self.network.eval_mode()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=torch.device(self._device)
            )
            actions, log_probs, entropies, values = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            entropies.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform PPO update on a batch of rollout data.

        Expected batch keys:
            observations: (N, obs_dim)
            actions: (N,)
            old_log_probs: (N,)
            advantages: (N,)
            returns: (N,)

        Returns:
            Dict of training metrics.
        """
        import torch  # noqa: PLC0415

        self.network.train_mode()
        device = torch.device(self._device)

        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(
            batch["old_log_probs"], dtype=torch.float32, device=device
        )
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + DEFAULT_REWARD_EPSILON)

        cfg = self.mappo_config
        n_samples = obs.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _epoch in range(cfg.epochs):
            # Shuffle indices for minibatch updates
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, cfg.batch_size):
                end = min(start + cfg.batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    mb_obs, action=mb_actions
                )
                values = values.squeeze(-1)

                # PPO clipped surrogate loss
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                policy_loss = -torch.min(
                    ratio * mb_advantages, clipped_ratio * mb_advantages
                ).mean()

                # Value loss (clipped)
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + cfg.value_coeff * value_loss
                    - cfg.entropy_coeff * entropy_loss
                )

                self.network.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.network.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl
                num_updates += 1

        num_updates = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
        }

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Shape (T,) rewards at each timestep.
            values: Shape (T,) value estimates at each timestep.
            dones: Shape (T,) episode termination flags.
            next_value: Bootstrap value for the last state.

        Returns:
            advantages: Shape (T,) GAE advantage estimates.
            returns: Shape (T,) discounted returns (advantages + values).
        """
        cfg = self.mappo_config
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(dones[t])
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + cfg.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str) -> None:
        """Save agent state and network weights."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save network weights
        weights_path = str(p.with_suffix(".pt"))
        self.network.save(weights_path)

        # Save agent metadata
        meta_path = str(p.with_suffix(".json"))
        with Path(meta_path).open("w") as f:
            json.dump(
                {
                    "config": self.config.__dict__,
                    "step_count": self._step_count,
                    "obs_dim": self._obs_dim,
                    "action_dim": self._action_dim,
                },
                f,
            )
        logger.info("MAPPOAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load agent state and network weights."""
        # Load network weights
        weights_path = str(Path(path).with_suffix(".pt"))
        self.network.load(weights_path)

        # Load agent metadata
        meta_path = str(Path(path).with_suffix(".json"))
        if Path(meta_path).exists():
            with Path(meta_path).open() as f:
                data = json.load(f)
            self._step_count = data.get("step_count", 0)
        logger.info("MAPPOAgent loaded from %s", path)
