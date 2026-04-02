"""Policy network implementations for FORGE agents.

Provides an abstract PolicyNetwork base class and a concrete implementation:
- RandomPolicyNetwork: uniform random baseline implementing the PolicyNetwork ABC

Also defines:
- ActorCriticNetwork: PyTorch actor-critic for PPO/MAPPO training (does not implement PolicyNetwork)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from forge.config import DEFAULT_ACTION_SIZE, DEFAULT_HIDDEN_SIZES

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class PolicyNetwork(ABC):
    """Abstract base class for policy networks."""

    @abstractmethod
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities from an observation."""

    @abstractmethod
    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step. Returns metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""


class RandomPolicyNetwork(PolicyNetwork):
    """Policy network that returns uniform random action probabilities."""

    def __init__(self, action_size: int = DEFAULT_ACTION_SIZE) -> None:
        self.action_size = action_size
        logger.info("RandomPolicyNetwork initialized with action_size=%d", action_size)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Return uniform action probabilities."""
        result: np.ndarray = np.ones(self.action_size, dtype=np.float32) / self.action_size
        return result

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op training step."""
        return {}

    def save(self, path: str) -> None:
        """No-op save."""
        logger.debug("RandomPolicyNetwork.save called (no-op)")

    def load(self, path: str) -> None:
        """No-op load."""
        logger.debug("RandomPolicyNetwork.load called (no-op)")


class ActorCriticNetwork:
    """PyTorch actor-critic network for PPO/MAPPO training.

    Architecture:
    - Shared MLP encoder over flat observation vector
    - Actor head: produces action logits (Discrete action space)
    - Critic head: produces scalar state value estimate

    All dimensions are configurable — no hardcoded layer sizes.

    Args:
        obs_dim: Dimensionality of the flat observation vector.
        action_dim: Number of discrete actions.
        hidden_sizes: List of hidden layer widths for the shared encoder.
        learning_rate: Optimizer learning rate.
        device: Torch device string ("cpu", "cuda", "mps").
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        if hidden_sizes is None:
            hidden_sizes = list(DEFAULT_HIDDEN_SIZES)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Build shared encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.Tanh())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)

        # Actor head: observation -> action logits
        self.actor_head = nn.Linear(prev_dim, action_dim).to(self.device)

        # Critic head: observation -> scalar value
        self.critic_head = nn.Linear(prev_dim, 1).to(self.device)

        # Collect all parameters for the optimizer
        all_params = (
            list(self.encoder.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate)

        total_params = sum(p.numel() for p in all_params)
        logger.info(
            "ActorCriticNetwork: obs_dim=%d, action_dim=%d, hidden=%s, params=%d, device=%s",
            obs_dim,
            action_dim,
            hidden_sizes,
            total_params,
            device,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            action_logits: Shape (batch, action_dim).
            value: Shape (batch, 1).
        """
        features = self.encoder(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and compute associated quantities for PPO.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).
            action: If provided, compute log_prob for this action instead of sampling.
            deterministic: If True, select argmax action instead of sampling.

        Returns:
            action: Selected action tensor of shape (batch,).
            log_prob: Log probability of the action, shape (batch,).
            entropy: Policy entropy, shape (batch,).
            value: State value estimate, shape (batch, 1).
        """
        import torch  # noqa: PLC0415
        from torch.distributions import Categorical  # noqa: PLC0415

        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = torch.argmax(action_logits, dim=-1) if deterministic else dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value only (for GAE bootstrap).

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            value: Shape (batch, 1).
        """
        features = self.encoder(obs)
        return self.critic_head(features)

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        import torch  # noqa: PLC0415

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor_head": self.actor_head.state_dict(),
                "critic_head": self.critic_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )
        logger.info("ActorCriticNetwork saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Validates that the loaded checkpoint dimensions match the current
        network configuration to prevent silent shape mismatches.
        """
        import torch  # noqa: PLC0415

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # Validate dimensions for backwards compatibility
        saved_obs = checkpoint.get("obs_dim")
        saved_act = checkpoint.get("action_dim")
        if saved_obs is not None and saved_obs != self.obs_dim:
            msg = f"Checkpoint obs_dim={saved_obs} does not match network obs_dim={self.obs_dim}"
            raise ValueError(msg)
        if saved_act is not None and saved_act != self.action_dim:
            msg = (
                f"Checkpoint action_dim={saved_act} does not match "
                f"network action_dim={self.action_dim}"
            )
            raise ValueError(msg)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("ActorCriticNetwork loaded from %s", path)

    def train_mode(self) -> None:
        """Set all modules to training mode."""
        self.encoder.train()
        self.actor_head.train()
        self.critic_head.train()

    def eval_mode(self) -> None:
        """Set all modules to evaluation mode."""
        self.encoder.eval()
        self.actor_head.eval()
        self.critic_head.eval()

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return all trainable parameters."""
        return (
            list(self.encoder.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
        )
