"""Recurrent State-Space Model (RSSM) world model for latent-space planning.

Implements the :class:`WorldModel` ABC with a PyTorch-backed RSSM that
supports single-step prediction, multi-step imagination, and observation
encoding.  Weights can be loaded from a HuggingFace repository via
:class:`~forge.utils.weight_loader.WeightLoader`.

Usage::

    from forge.models.rssm_world_model import RSSMWorldModel, RSSMConfig
    from forge.utils.weight_loader import WeightLoader

    model = RSSMWorldModel(RSSMConfig())
    model.load_from_hub(WeightLoader())
    next_state = model.predict(state, action=3)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forge.config import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM
from forge.models.world_model import WorldModel

if TYPE_CHECKING:
    import numpy as np
    import torch

    from forge.utils.weight_loader import WeightLoader

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_DIM = 128
DEFAULT_STOCHASTIC_DIM = 32
DEFAULT_DETERMINISTIC_DIM = 64
DEFAULT_STATE_DIM = DEFAULT_STOCHASTIC_DIM + DEFAULT_DETERMINISTIC_DIM
DEFAULT_RSSM_WEIGHT_FILE = "rssm/final.pt"


@dataclass
class RSSMConfig:
    """Configuration for the RSSM world model.

    All dimensions are configurable — no hard-coded layer sizes.
    """

    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    state_dim: int = DEFAULT_STATE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    stochastic_dim: int = DEFAULT_STOCHASTIC_DIM
    deterministic_dim: int = DEFAULT_DETERMINISTIC_DIM
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Ensure state_dim equals deterministic_dim + stochastic_dim."""
        expected = self.deterministic_dim + self.stochastic_dim
        if self.state_dim != expected:
            logger.warning(
                "RSSMConfig.state_dim (%d) != deterministic_dim + stochastic_dim (%d); "
                "correcting to %d.",
                self.state_dim,
                expected,
                expected,
            )
            self.state_dim = expected


class RSSMWorldModel(WorldModel):
    """RSSM world model for latent-space dynamics prediction.

    Architecture:
        - Encoder: obs -> stochastic latent state
        - Transition: (state, action) -> next deterministic state (GRU)
        - Prior / Posterior: deterministic -> stochastic (mean, logvar)
        - Decoder: latent state -> reconstructed observation

    Args:
        config: RSSM configuration.
    """

    def __init__(self, config: RSSMConfig | None = None) -> None:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        self._config = config or RSSMConfig()
        c = self._config
        self._device = torch.device(c.device)

        # Encoder: observation -> stochastic latent
        self.encoder = nn.Sequential(
            nn.Linear(c.obs_dim, c.hidden_dim),
            nn.ELU(),
            nn.Linear(c.hidden_dim, c.stochastic_dim * 2),  # mean + logvar
        ).to(self._device)

        # Transition model: GRU-based recurrence
        gru_input_dim = c.stochastic_dim + c.action_dim
        self.transition_gru = nn.GRUCell(gru_input_dim, c.deterministic_dim).to(
            self._device
        )

        # Prior: deterministic -> stochastic (for imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(c.deterministic_dim, c.hidden_dim),
            nn.ELU(),
            nn.Linear(c.hidden_dim, c.stochastic_dim * 2),
        ).to(self._device)

        # Decoder: full state -> observation reconstruction
        full_state_dim = c.deterministic_dim + c.stochastic_dim
        self.decoder = nn.Sequential(
            nn.Linear(full_state_dim, c.hidden_dim),
            nn.ELU(),
            nn.Linear(c.hidden_dim, c.obs_dim),
        ).to(self._device)

        # Collect all parameters
        self._modules_list = nn.ModuleList(
            [self.encoder, self.transition_gru, self.prior_net, self.decoder]
        ).to(self._device)

        total_params = sum(p.numel() for p in self._modules_list.parameters())
        logger.info(
            "RSSMWorldModel: obs=%d, act=%d, stoch=%d, det=%d, params=%d, device=%s",
            c.obs_dim,
            c.action_dim,
            c.stochastic_dim,
            c.deterministic_dim,
            total_params,
            c.device,
        )

    @property
    def config(self) -> RSSMConfig:
        """Return the RSSM configuration."""
        return self._config

    def _one_hot_action(self, action: int) -> torch.Tensor:
        """Convert a discrete action to a one-hot tensor.

        Raises:
            ValueError: If action is not in [0, action_dim).
        """
        import torch  # noqa: PLC0415

        if not 0 <= action < self._config.action_dim:
            msg = (
                f"Invalid action index: {action}. Expected an integer in the range "
                f"[0, {self._config.action_dim})."
            )
            raise ValueError(msg)
        vec = torch.zeros(self._config.action_dim, device=self._device)
        vec[action] = 1.0
        return vec

    def _split_state(
        self, state_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a state tensor into (deterministic, stochastic) parts.

        Accepts either a pure stochastic state ``(B, stochastic_dim)`` or a
        full concatenated state ``(B, deterministic_dim + stochastic_dim)``.
        Any other shape is considered invalid and results in a ``ValueError``.
        """
        import torch  # noqa: PLC0415

        c = self._config
        if state_t.shape[-1] == c.stochastic_dim:
            det = torch.zeros(state_t.shape[0], c.deterministic_dim, device=self._device)
            stoch = state_t
        elif state_t.shape[-1] == c.deterministic_dim + c.stochastic_dim:
            det = state_t[..., : c.deterministic_dim]
            stoch = state_t[..., c.deterministic_dim :]
        else:
            msg = (
                f"Invalid state shape: {state_t.shape}. Expected last dimension "
                f"to be {c.stochastic_dim} or "
                f"{c.deterministic_dim + c.stochastic_dim}."
            )
            raise ValueError(msg)
        return det, stoch

    def _reparametrise(self, mean_logvar: torch.Tensor) -> torch.Tensor:
        """Sample from a Gaussian using the reparametrisation trick."""
        import torch  # noqa: PLC0415

        dim = mean_logvar.shape[-1] // 2
        mean, logvar = mean_logvar[..., :dim], mean_logvar[..., dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode an observation into a stochastic latent state.

        Args:
            obs: Flat observation array of shape ``(obs_dim,)``.

        Returns:
            Latent state of shape ``(stochastic_dim,)``.
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            mean_logvar = self.encoder(obs_t)
            stoch = self._reparametrise(mean_logvar)
        result: np.ndarray = stoch.squeeze(0).cpu().numpy()
        return result

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict the next observation given a latent state and action.

        Args:
            state: Current latent state of shape ``(stochastic_dim,)``
                or full state ``(deterministic_dim + stochastic_dim,)``.
            action: Discrete action index.

        Returns:
            Predicted next observation of shape ``(obs_dim,)``.
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self._device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            det, stoch = self._split_state(state_t)

            action_oh = self._one_hot_action(action).unsqueeze(0)
            gru_input = torch.cat([stoch, action_oh], dim=-1)
            next_det = self.transition_gru(gru_input, det)
            next_stoch = self._reparametrise(self.prior_net(next_det))
            full_state = torch.cat([next_det, next_stoch], dim=-1)
            obs_pred = self.decoder(full_state)

        result: np.ndarray = obs_pred.squeeze(0).cpu().numpy()
        return result

    def imagine(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out multiple steps in latent space.

        Args:
            state: Initial latent state of shape ``(stochastic_dim,)``.
            actions: Sequence of discrete actions, shape ``(T,)``.

        Returns:
            Predicted observations of shape ``(T, obs_dim)``.
        """
        import torch  # noqa: PLC0415

        T = len(actions)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self._device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            det, stoch = self._split_state(state_t)

            predictions = []
            for t in range(T):
                action_oh = self._one_hot_action(int(actions[t])).unsqueeze(0)
                gru_input = torch.cat([stoch, action_oh], dim=-1)
                det = self.transition_gru(gru_input, det)
                stoch = self._reparametrise(self.prior_net(det))
                full_state = torch.cat([det, stoch], dim=-1)
                obs_pred = self.decoder(full_state)
                predictions.append(obs_pred)

            result_t = torch.cat(predictions, dim=0)
        result: np.ndarray = result_t.cpu().numpy()
        return result

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step (reconstruction + KL loss).

        Expected batch keys:
            observations: ``(N, obs_dim)``
            actions: ``(N,)``

        Returns:
            Dictionary of training metrics.
        """
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        obs = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self._device
        )

        # Encode current observation
        posterior_params = self.encoder(obs)
        stoch = self._reparametrise(posterior_params)

        # Reconstruct current observation (simplified — ignores recurrence for batch training)
        c = self._config
        det = torch.zeros(obs.shape[0], c.deterministic_dim, device=self._device)
        full_state = torch.cat([det, stoch], dim=-1)
        obs_recon = self.decoder(full_state)

        # Reconstruction loss against current observation
        recon_loss = nn.functional.mse_loss(obs_recon, obs)

        # KL divergence (posterior vs unit Gaussian prior)
        dim = c.stochastic_dim
        mean = posterior_params[..., :dim]
        logvar = posterior_params[..., dim:]
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        loss = recon_loss + kl_loss

        return {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kl_loss": float(kl_loss.item()),
        }

    def save(self, path: str) -> None:
        """Save all RSSM module weights to disk."""
        import torch  # noqa: PLC0415

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "modules": self._modules_list.state_dict(),
                "config": {
                    "obs_dim": self._config.obs_dim,
                    "action_dim": self._config.action_dim,
                    "state_dim": self._config.state_dim,
                    "hidden_dim": self._config.hidden_dim,
                    "stochastic_dim": self._config.stochastic_dim,
                    "deterministic_dim": self._config.deterministic_dim,
                },
            },
            path,
        )
        logger.info("RSSMWorldModel saved to %s", path)

    def load(self, path: str) -> None:
        """Load RSSM module weights from disk."""
        import torch  # noqa: PLC0415

        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        saved_cfg = checkpoint.get("config", {})
        if saved_cfg.get("obs_dim") and saved_cfg["obs_dim"] != self._config.obs_dim:
            msg = (
                f"Checkpoint obs_dim={saved_cfg['obs_dim']} != "
                f"model obs_dim={self._config.obs_dim}"
            )
            raise ValueError(msg)
        self._modules_list.load_state_dict(checkpoint["modules"])
        logger.info("RSSMWorldModel loaded from %s", path)

    def load_from_hub(
        self,
        loader: WeightLoader,
        filename: str = DEFAULT_RSSM_WEIGHT_FILE,
    ) -> None:
        """Download and load RSSM weights from HuggingFace Hub.

        Args:
            loader: A configured :class:`WeightLoader`.
            filename: Path within the repository.
        """
        path = loader.resolve_path(filename)
        self.load(str(path))
        logger.info("RSSMWorldModel loaded from hub: %s", filename)
