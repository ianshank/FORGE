"""MuZero world model: learned latent-space dynamics for planning.

Combines the three MuZero networks (representation, dynamics, prediction)
into a single model that satisfies the :class:`~forge.models.world_model.WorldModel`
ABC and provides MuZero-specific inference methods.

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel

    config = MuZeroConfig(obs_dim=920, action_dim=75)
    model = MuZeroWorldModel(config)

    # Initial inference: observation -> (latent, policy, value)
    output = model.initial_inference(observation)

    # Recurrent inference: (latent, action) -> (next_latent, reward, policy, value)
    output = model.recurrent_inference(output.latent_state, action=3)
"""
from __future__ import annotations

__all__ = ["MuZeroWorldModel", "NetworkOutput"]

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forge.models.muzero_config import MuZeroConfig
from forge.models.muzero_networks import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    scalar_to_support,
    support_to_scalar,
)
from forge.models.world_model import WorldModel

if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)

DEFAULT_MUZERO_WEIGHT_FILE = "muzero/final.pt"


@dataclass
class NetworkOutput:
    """Output of a MuZero inference step.

    Attributes:
        latent_state: Latent state vector of shape ``(latent_dim,)``.
        reward: Scalar reward prediction (0.0 for initial inference).
        policy_logits: Action logits of shape ``(action_dim,)``.
        value: Scalar value prediction.
    """

    latent_state: np.ndarray
    reward: float
    policy_logits: np.ndarray
    value: float


class MuZeroWorldModel(WorldModel):
    """MuZero world model combining representation, dynamics, and prediction.

    This model enables planning in latent space: the MCTS planner calls
    :meth:`initial_inference` at the root and :meth:`recurrent_inference`
    at each tree expansion, avoiding the need to simulate the full environment.

    Args:
        config: MuZero configuration. If ``None``, uses defaults.
    """

    def __init__(self, config: MuZeroConfig | None = None) -> None:
        import torch  # noqa: PLC0415

        self._config = config or MuZeroConfig()
        c = self._config
        self._device = torch.device(c.device)

        self.representation = RepresentationNetwork(c)
        self.dynamics = DynamicsNetwork(c)
        self.prediction = PredictionNetwork(c)

        # Collect all parameters for training
        self._all_params = (
            self.representation.parameters()
            + self.dynamics.parameters()
            + self.prediction.parameters()
        )

        total_params = sum(p.numel() for p in self._all_params)
        logger.info(
            "MuZeroWorldModel: obs=%d, act=%d, latent=%d, total_params=%d, device=%s",
            c.obs_dim, c.action_dim, c.latent_dim, total_params, c.device,
        )

    @property
    def config(self) -> MuZeroConfig:
        """Return the MuZero configuration."""
        return self._config

    def _action_to_onehot(self, action: int) -> torch.Tensor:
        """Convert a discrete action index to a one-hot tensor.

        Args:
            action: Action index in ``[0, action_dim)``.

        Returns:
            One-hot tensor of shape ``(action_dim,)``.

        Raises:
            ValueError: If action is out of bounds.
        """
        import torch  # noqa: PLC0415

        if not 0 <= action < self._config.action_dim:
            msg = (
                f"Invalid action index: {action}. "
                f"Expected [0, {self._config.action_dim})."
            )
            raise ValueError(msg)
        vec = torch.zeros(self._config.action_dim, device=self._device)
        vec[action] = 1.0
        return vec

    def initial_inference(self, observation: np.ndarray) -> NetworkOutput:
        """Run representation + prediction on a raw observation.

        Args:
            observation: Flat observation array of shape ``(obs_dim,)``.

        Returns:
            :class:`NetworkOutput` with latent state, policy, and value.
            Reward is 0.0 (no transition occurred).
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self._device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            latent = self.representation.forward(obs_t)
            policy_logits, value_logits = self.prediction.forward(latent)
            value = support_to_scalar(value_logits, self._config.value_support_size)

        latent_np: np.ndarray = latent.squeeze(0).cpu().numpy()
        policy_np: np.ndarray = policy_logits.squeeze(0).cpu().numpy()
        return NetworkOutput(
            latent_state=latent_np,
            reward=0.0,
            policy_logits=policy_np,
            value=float(value.squeeze().item()),
        )

    def recurrent_inference(
        self, latent_state: np.ndarray, action: int
    ) -> NetworkOutput:
        """Run dynamics + prediction from a latent state and action.

        Args:
            latent_state: Latent state array of shape ``(latent_dim,)``.
            action: Discrete action index.

        Returns:
            :class:`NetworkOutput` with next latent state, reward,
            policy, and value.
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            state_t = torch.as_tensor(
                latent_state, dtype=torch.float32, device=self._device
            )
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            action_oh = self._action_to_onehot(action).unsqueeze(0)
            next_latent, reward_logits = self.dynamics.forward(state_t, action_oh)
            policy_logits, value_logits = self.prediction.forward(next_latent)

            reward = support_to_scalar(
                reward_logits, self._config.reward_support_size
            )
            value = support_to_scalar(
                value_logits, self._config.value_support_size
            )

        next_np: np.ndarray = next_latent.squeeze(0).cpu().numpy()
        policy_np: np.ndarray = policy_logits.squeeze(0).cpu().numpy()
        return NetworkOutput(
            latent_state=next_np,
            reward=float(reward.squeeze().item()),
            policy_logits=policy_np,
            value=float(value.squeeze().item()),
        )

    # --- WorldModel ABC methods ---

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict the next latent state (WorldModel ABC compliance).

        Args:
            state: Current latent state of shape ``(latent_dim,)``.
            action: Discrete action index.

        Returns:
            Next latent state of shape ``(latent_dim,)``.
        """
        output = self.recurrent_inference(state, action)
        return output.latent_state

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single MuZero training step.

        Unrolls the dynamics network for ``num_unroll_steps`` steps and
        computes policy, value, and reward losses.

        Expected batch keys:
            observations: ``(N, obs_dim)`` — initial observations
            actions: ``(N, K)`` — action sequences (K = num_unroll_steps)
            target_values: ``(N, K+1)`` — target values at each step
            target_rewards: ``(N, K)`` — target rewards at each step
            target_policies: ``(N, K+1, action_dim)`` — target policy distributions

        Returns:
            Dictionary of training metrics.
        """
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        c = self._config
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self._device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self._device)
        target_values = torch.as_tensor(
            batch["target_values"], dtype=torch.float32, device=self._device
        )
        target_rewards = torch.as_tensor(
            batch["target_rewards"], dtype=torch.float32, device=self._device
        )
        target_policies = torch.as_tensor(
            batch["target_policies"], dtype=torch.float32, device=self._device
        )

        # Initial inference
        latent = self.representation.forward(obs)
        policy_logits, value_logits = self.prediction.forward(latent)

        # Initial step losses
        target_value_dist = scalar_to_support(target_values[:, 0], c.value_support_size)
        value_loss = nn.functional.cross_entropy(value_logits, target_value_dist)

        log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.mean(torch.sum(target_policies[:, 0] * log_probs, dim=-1))

        reward_loss = torch.tensor(0.0, device=self._device)

        # Unroll dynamics
        num_steps = min(c.num_unroll_steps, actions.shape[1])
        for k in range(num_steps):
            action_oh = nn.functional.one_hot(
                actions[:, k], num_classes=c.action_dim
            ).float()

            latent, rew_logits = self.dynamics.forward(latent, action_oh)
            pol_logits, val_logits = self.prediction.forward(latent)

            # Reward loss
            target_rew_dist = scalar_to_support(target_rewards[:, k], c.reward_support_size)
            reward_loss = reward_loss + nn.functional.cross_entropy(rew_logits, target_rew_dist)

            # Value loss
            target_val_dist = scalar_to_support(target_values[:, k + 1], c.value_support_size)
            value_loss = value_loss + nn.functional.cross_entropy(val_logits, target_val_dist)

            # Policy loss
            log_p = torch.log_softmax(pol_logits, dim=-1)
            policy_loss = policy_loss + (
                -torch.mean(torch.sum(target_policies[:, k + 1] * log_p, dim=-1))
            )

        # Scale losses by number of unroll steps
        scale = 1.0 / (num_steps + 1)
        total_loss = scale * (value_loss + policy_loss + reward_loss)

        # L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in self._all_params)
        total_loss = total_loss + c.weight_decay * l2_reg

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item() * scale),
            "value_loss": float(value_loss.item() * scale),
            "reward_loss": float(reward_loss.item() * scale),
            "l2_reg": float(l2_reg.item()),
        }

    def save(self, path: str) -> None:
        """Save all MuZero network weights to disk.

        Args:
            path: File path for the checkpoint.
        """
        import torch  # noqa: PLC0415

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "representation": self.representation.modules_list.state_dict(),
                "dynamics": self.dynamics.modules_list.state_dict(),
                "prediction": self.prediction.modules_list.state_dict(),
                "config": {
                    "obs_dim": self._config.obs_dim,
                    "action_dim": self._config.action_dim,
                    "latent_dim": self._config.latent_dim,
                    "hidden_dim": self._config.hidden_dim,
                    "num_blocks": self._config.num_blocks,
                    "reward_support_size": self._config.reward_support_size,
                    "value_support_size": self._config.value_support_size,
                    "grid_height": self._config.grid_height,
                    "grid_width": self._config.grid_width,
                    "grid_channels": self._config.grid_channels,
                    "vector_dim": self._config.vector_dim,
                },
            },
            path,
        )
        logger.info("MuZeroWorldModel saved to %s", path)

    def load(self, path: str) -> None:
        """Load MuZero network weights from disk.

        Validates that saved dimensions match the current configuration.

        Args:
            path: File path to the checkpoint.

        Raises:
            ValueError: If checkpoint dimensions don't match config.
        """
        import torch  # noqa: PLC0415

        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        saved_cfg = checkpoint.get("config", {})

        for key in ("obs_dim", "action_dim", "latent_dim"):
            saved_val = saved_cfg.get(key)
            actual_val = getattr(self._config, key)
            if saved_val is not None and saved_val != actual_val:
                msg = f"Checkpoint {key}={saved_val} != model {key}={actual_val}"
                raise ValueError(msg)

        self.representation.modules_list.load_state_dict(checkpoint["representation"])
        self.dynamics.modules_list.load_state_dict(checkpoint["dynamics"])
        self.prediction.modules_list.load_state_dict(checkpoint["prediction"])
        logger.info("MuZeroWorldModel loaded from %s", path)

    def all_parameters(self) -> list[torch.nn.Parameter]:
        """Return all trainable parameters across all networks."""
        return list(self._all_params)

    def to(self, device: "torch.device | str") -> "MuZeroWorldModel":
        """Move every sub-network's parameters to ``device``. Mirrors
        :meth:`torch.nn.Module.to` so a caller can write
        ``model.to(trainer.device)`` regardless of whether ``model`` is
        a ``nn.Module`` or this wrapper.

        Updates the cached ``self._device`` so future
        ``.config.device``-aware paths stay consistent.

        Returns ``self`` for chainability (matches
        ``nn.Module.to``'s contract).
        """
        import torch  # noqa: PLC0415

        resolved = torch.device(device) if not isinstance(device, torch.device) else device
        self.representation.modules_list.to(resolved)
        self.dynamics.modules_list.to(resolved)
        self.prediction.modules_list.to(resolved)
        self._device = resolved
        return self
