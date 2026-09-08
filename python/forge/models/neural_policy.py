"""Neural MCTS policy network for learned action priors and value estimation.

Implements the :class:`PolicyNetwork` ABC with a PyTorch MLP that produces
softmax action probabilities and a scalar value estimate.  Weights can be
loaded from NumPy ``.npz`` archives via
:class:`~forge.utils.weight_loader.WeightLoader`.

Usage::

    from forge.models.neural_policy import NeuralMCTSPolicy, NeuralPolicyConfig
    from forge.utils.weight_loader import WeightLoader

    policy = NeuralMCTSPolicy(NeuralPolicyConfig())
    policy.load_from_npz(WeightLoader(), "mcts/policy_init.npz")
    priors = policy.forward(observation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from forge.config import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM
from forge.models.policy_network import PolicyNetwork

if TYPE_CHECKING:
    import numpy as np

    from forge.utils.weight_loader import WeightLoader

logger = logging.getLogger(__name__)

DEFAULT_POLICY_HIDDEN_SIZES: list[int] = [128, 128]
DEFAULT_POLICY_WEIGHT_FILE = "mcts/policy_init.npz"


@dataclass
class NeuralPolicyConfig:
    """Configuration for :class:`NeuralMCTSPolicy`.

    All dimensions are configurable — no hard-coded layer sizes.
    """

    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    hidden_sizes: list[int] = field(
        default_factory=lambda: list(DEFAULT_POLICY_HIDDEN_SIZES)
    )
    learning_rate: float = 3e-4
    device: str = "cpu"


class NeuralMCTSPolicy(PolicyNetwork):
    """MLP policy network that outputs action priors and a value estimate.

    Architecture:
        - Shared encoder: MLP with Tanh activations
        - Policy head: Linear -> softmax action probabilities
        - Value head: Linear -> scalar state value

    Args:
        config: Policy configuration.
    """

    def __init__(self, config: NeuralPolicyConfig | None = None) -> None:
        import torch
        from torch import nn

        self._config = config or NeuralPolicyConfig()
        c = self._config
        self._device = torch.device(c.device)

        # Build shared encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = c.obs_dim
        for h in c.hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.Tanh())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers).to(self._device)

        # Policy head: -> action logits
        self.policy_head = nn.Linear(prev_dim, c.action_dim).to(self._device)

        # Value head: -> scalar value
        self.value_head = nn.Linear(prev_dim, 1).to(self._device)

        # Optimizer
        all_params = (
            list(self.encoder.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=c.learning_rate)

        total_params = sum(p.numel() for p in all_params)
        logger.info(
            "NeuralMCTSPolicy: obs=%d, act=%d, hidden=%s, params=%d, device=%s",
            c.obs_dim,
            c.action_dim,
            c.hidden_sizes,
            total_params,
            c.device,
        )

    @property
    def config(self) -> NeuralPolicyConfig:
        """Return the policy configuration."""
        return self._config

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities from an observation.

        Args:
            obs: Observation array of shape ``(obs_dim,)`` or ``(batch, obs_dim)``.

        Returns:
            Action probability distribution of shape ``(action_dim,)``
            or ``(batch, action_dim)``.
        """
        import torch

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
            squeezed = obs_t.dim() == 1
            if squeezed:
                obs_t = obs_t.unsqueeze(0)
            features = self.encoder(obs_t)
            logits = self.policy_head(features)
            probs = torch.softmax(logits, dim=-1)
            if squeezed:
                probs = probs.squeeze(0)
        result: np.ndarray = probs.cpu().numpy()
        return result

    def evaluate(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute action priors and state value.

        This method matches the interface expected by MCTS policy-value
        evaluation, returning both a probability distribution over actions
        and a scalar value estimate.

        Args:
            obs: Observation array of shape ``(obs_dim,)``.

        Returns:
            Tuple of (action_priors, value) where action_priors has shape
            ``(action_dim,)`` and value is a scalar float.
        """
        import torch

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            features = self.encoder(obs_t)
            logits = self.policy_head(features)
            probs = torch.softmax(logits, dim=-1)
            value = self.value_head(features)
        priors: np.ndarray = probs.squeeze(0).cpu().numpy()
        return priors, float(value.squeeze().item())

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step with policy and value losses.

        Expected batch keys:
            observations: ``(N, obs_dim)``
            target_priors: ``(N, action_dim)`` — target policy distribution
            target_values: ``(N,)`` — target state values

        Returns:
            Dictionary of training metrics.
        """
        import torch
        from torch import nn

        obs = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self._device
        )
        target_priors = torch.as_tensor(
            batch["target_priors"], dtype=torch.float32, device=self._device
        )
        target_values = torch.as_tensor(
            batch["target_values"], dtype=torch.float32, device=self._device
        )

        features = self.encoder(obs)
        logits = self.policy_head(features)
        log_probs = torch.log_softmax(logits, dim=-1)
        values = self.value_head(features).squeeze(-1)

        # Cross-entropy policy loss
        policy_loss = -torch.mean(torch.sum(target_priors * log_probs, dim=-1))
        # MSE value loss
        value_loss = nn.functional.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "loss": float(loss.item()),
        }

    def save(self, path: str) -> None:
        """Save policy weights to disk."""
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_dim": self._config.obs_dim,
                "action_dim": self._config.action_dim,
            },
            path,
        )
        logger.info("NeuralMCTSPolicy saved to %s", path)

    def load(self, path: str) -> None:
        """Load policy weights from disk."""
        import torch

        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        saved_obs = checkpoint.get("obs_dim")
        saved_act = checkpoint.get("action_dim")
        if saved_obs is not None and saved_obs != self._config.obs_dim:
            msg = (
                f"Checkpoint obs_dim={saved_obs} != "
                f"policy obs_dim={self._config.obs_dim}"
            )
            raise ValueError(msg)
        if saved_act is not None and saved_act != self._config.action_dim:
            msg = (
                f"Checkpoint action_dim={saved_act} != "
                f"policy action_dim={self._config.action_dim}"
            )
            raise ValueError(msg)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("NeuralMCTSPolicy loaded from %s", path)

    def load_from_npz(
        self,
        loader: WeightLoader,
        filename: str = DEFAULT_POLICY_WEIGHT_FILE,
    ) -> None:
        """Load policy weights from a NumPy ``.npz`` archive on HuggingFace.

        The ``.npz`` archive is expected to contain arrays that map to the
        encoder, policy head, and value head parameters.  Arrays are loaded
        by positional index matching the network's parameter order.

        Args:
            loader: A configured :class:`WeightLoader`.
            filename: Path within the repository.
        """
        import torch

        data = loader.load_npz(filename)

        # Load arrays into parameters by matching sorted keys to param order
        all_params = (
            list(self.encoder.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        sorted_keys = sorted(data.keys())

        if len(sorted_keys) != len(all_params):
            logger.warning(
                "Weight file %s contains %d arrays but network expects %d parameters; "
                "extra arrays or missing parameters will be ignored.",
                filename,
                len(sorted_keys),
                len(all_params),
            )

        loaded = 0
        for key, param in zip(sorted_keys, all_params):
            arr = data[key]
            tensor = torch.as_tensor(arr, dtype=torch.float32, device=self._device)
            if tensor.shape == param.shape:
                with torch.no_grad():
                    param.copy_(tensor)
                loaded += 1
            else:
                logger.warning(
                    "Shape mismatch for %s: npz=%s, param=%s — skipping",
                    key,
                    tensor.shape,
                    param.shape,
                )

        logger.info(
            "NeuralMCTSPolicy loaded %d/%d arrays from %s",
            loaded,
            len(sorted_keys),
            filename,
        )
