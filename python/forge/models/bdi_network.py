"""Belief-Desire-Intention (BDI) encoder networks.

Four sub-networks that compose a BDI cognitive architecture:
    - **Belief encoder**: observation -> belief state
    - **Desire encoder**: (observation, belief) -> desire state
    - **Intention predictor**: (belief, desire) -> intention state
    - **Affect estimator**: (observation, belief) -> affect state

All weights can be loaded from ``.npz`` archives via
:class:`~forge.utils.weight_loader.WeightLoader`.

Usage::

    from forge.models.bdi_network import BDINetwork, BDIConfig
    from forge.utils.weight_loader import WeightLoader

    bdi = BDINetwork(BDIConfig())
    bdi.load_from_hub(WeightLoader())
    state = bdi.forward(observation)
    print(state.belief.shape, state.intention.shape)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from forge.config import DEFAULT_OBS_DIM

if TYPE_CHECKING:
    import numpy as np
    import torch

    from forge.utils.weight_loader import WeightLoader

logger = logging.getLogger(__name__)

DEFAULT_BELIEF_DIM = 32
DEFAULT_DESIRE_DIM = 16
DEFAULT_INTENTION_DIM = 16
DEFAULT_AFFECT_DIM = 8
DEFAULT_BDI_HIDDEN_SIZES: list[int] = [64, 64]

DEFAULT_BELIEF_WEIGHT_FILE = "bdi/belief.npz"
DEFAULT_DESIRE_WEIGHT_FILE = "bdi/desire.npz"
DEFAULT_INTENTION_WEIGHT_FILE = "bdi/intention.npz"
DEFAULT_AFFECT_WEIGHT_FILE = "bdi/affect.npz"


@dataclass
class BDIConfig:
    """Configuration for :class:`BDINetwork`.

    All dimensions are configurable — no hard-coded layer sizes.
    """

    obs_dim: int = DEFAULT_OBS_DIM
    belief_dim: int = DEFAULT_BELIEF_DIM
    desire_dim: int = DEFAULT_DESIRE_DIM
    intention_dim: int = DEFAULT_INTENTION_DIM
    affect_dim: int = DEFAULT_AFFECT_DIM
    hidden_sizes: list[int] = field(
        default_factory=lambda: list(DEFAULT_BDI_HIDDEN_SIZES)
    )
    device: str = "cpu"


@dataclass
class BDIState:
    """Output of a full BDI forward pass.

    Attributes:
        belief: Belief state array of shape ``(belief_dim,)``.
        desire: Desire state array of shape ``(desire_dim,)``.
        intention: Intention state array of shape ``(intention_dim,)``.
        affect: Affect state array of shape ``(affect_dim,)``.
    """

    belief: np.ndarray
    desire: np.ndarray
    intention: np.ndarray
    affect: np.ndarray


def _to_tensor(
    arr: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Convert a numpy array to a 2-D (1, N) float32 tensor on *device*.

    If the array is already 1-D it is unsqueezed to add a batch dimension.
    """
    import torch

    t = torch.as_tensor(arr, dtype=torch.float32, device=torch.device(device))
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    device: str,
) -> torch.nn.Sequential:
    """Build an MLP with ELU activations."""
    import torch
    from torch import nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers).to(torch.device(device))


class BDINetwork:
    """Belief-Desire-Intention encoder network.

    Composes four sub-networks:
        1. Belief: obs -> belief
        2. Desire: (obs, belief) -> desire
        3. Intention: (belief, desire) -> intention
        4. Affect: (obs, belief) -> affect

    Args:
        config: BDI configuration.
    """

    def __init__(self, config: BDIConfig | None = None) -> None:
        from torch import nn

        self._config = config or BDIConfig()
        c = self._config

        self.belief_net = _build_mlp(c.obs_dim, c.belief_dim, c.hidden_sizes, c.device)
        self.desire_net = _build_mlp(
            c.obs_dim + c.belief_dim, c.desire_dim, c.hidden_sizes, c.device
        )
        self.intention_net = _build_mlp(
            c.belief_dim + c.desire_dim, c.intention_dim, c.hidden_sizes, c.device
        )
        self.affect_net = _build_mlp(
            c.obs_dim + c.belief_dim, c.affect_dim, c.hidden_sizes, c.device
        )

        self._modules_list = nn.ModuleList(
            [self.belief_net, self.desire_net, self.intention_net, self.affect_net]
        )

        total_params = sum(p.numel() for p in self._modules_list.parameters())
        logger.info(
            "BDINetwork: obs=%d, belief=%d, desire=%d, intention=%d, affect=%d, "
            "params=%d, device=%s",
            c.obs_dim,
            c.belief_dim,
            c.desire_dim,
            c.intention_dim,
            c.affect_dim,
            total_params,
            c.device,
        )

    @property
    def config(self) -> BDIConfig:
        """Return the BDI configuration."""
        return self._config

    def encode_belief(self, obs: np.ndarray) -> np.ndarray:
        """Encode an observation into a belief state.

        Args:
            obs: Observation array of shape ``(obs_dim,)``.

        Returns:
            Belief state of shape ``(belief_dim,)``.
        """
        import torch

        with torch.no_grad():
            obs_t = _to_tensor(obs, self._config.device)
            result = self.belief_net(obs_t)
        out: np.ndarray = result.squeeze(0).cpu().numpy()
        return out

    def encode_desire(self, obs: np.ndarray, belief: np.ndarray) -> np.ndarray:
        """Encode observation and belief into a desire state.

        Args:
            obs: Observation array of shape ``(obs_dim,)``.
            belief: Belief state of shape ``(belief_dim,)``.

        Returns:
            Desire state of shape ``(desire_dim,)``.
        """
        import torch

        with torch.no_grad():
            obs_t = _to_tensor(obs, self._config.device)
            belief_t = _to_tensor(belief, self._config.device)
            combined = torch.cat([obs_t, belief_t], dim=-1)
            result = self.desire_net(combined)
        out: np.ndarray = result.squeeze(0).cpu().numpy()
        return out

    def predict_intention(
        self, belief: np.ndarray, desire: np.ndarray
    ) -> np.ndarray:
        """Predict intention from belief and desire states.

        Args:
            belief: Belief state of shape ``(belief_dim,)``.
            desire: Desire state of shape ``(desire_dim,)``.

        Returns:
            Intention state of shape ``(intention_dim,)``.
        """
        import torch

        with torch.no_grad():
            belief_t = _to_tensor(belief, self._config.device)
            desire_t = _to_tensor(desire, self._config.device)
            combined = torch.cat([belief_t, desire_t], dim=-1)
            result = self.intention_net(combined)
        out: np.ndarray = result.squeeze(0).cpu().numpy()
        return out

    def estimate_affect(self, obs: np.ndarray, belief: np.ndarray) -> np.ndarray:
        """Estimate affect from observation and belief.

        Args:
            obs: Observation array of shape ``(obs_dim,)``.
            belief: Belief state of shape ``(belief_dim,)``.

        Returns:
            Affect state of shape ``(affect_dim,)``.
        """
        import torch

        with torch.no_grad():
            obs_t = _to_tensor(obs, self._config.device)
            belief_t = _to_tensor(belief, self._config.device)
            combined = torch.cat([obs_t, belief_t], dim=-1)
            result = self.affect_net(combined)
        out: np.ndarray = result.squeeze(0).cpu().numpy()
        return out

    def forward(self, obs: np.ndarray) -> BDIState:
        """Run the full BDI pipeline.

        Pipeline: obs -> belief -> desire -> intention; (obs, belief) -> affect.

        Args:
            obs: Observation array of shape ``(obs_dim,)``.

        Returns:
            :class:`BDIState` containing all four latent representations.
        """
        belief = self.encode_belief(obs)
        desire = self.encode_desire(obs, belief)
        intention = self.predict_intention(belief, desire)
        affect = self.estimate_affect(obs, belief)
        return BDIState(
            belief=belief, desire=desire, intention=intention, affect=affect
        )

    def save(self, path: str) -> None:
        """Save all BDI module weights to disk."""
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "modules": self._modules_list.state_dict(),
                "config": {
                    "obs_dim": self._config.obs_dim,
                    "belief_dim": self._config.belief_dim,
                    "desire_dim": self._config.desire_dim,
                    "intention_dim": self._config.intention_dim,
                    "affect_dim": self._config.affect_dim,
                },
            },
            path,
        )
        logger.info("BDINetwork saved to %s", path)

    def load(self, path: str) -> None:
        """Load BDI module weights from disk."""
        import torch

        checkpoint = torch.load(
            path, map_location=torch.device(self._config.device), weights_only=True
        )
        saved_cfg = checkpoint.get("config", {})
        if (
            saved_cfg.get("obs_dim")
            and saved_cfg["obs_dim"] != self._config.obs_dim
        ):
            msg = (
                f"Checkpoint obs_dim={saved_cfg['obs_dim']} != "
                f"network obs_dim={self._config.obs_dim}"
            )
            raise ValueError(msg)
        self._modules_list.load_state_dict(checkpoint["modules"])
        logger.info("BDINetwork loaded from %s", path)

    def _load_sub_network_from_npz(
        self,
        net: torch.nn.Sequential,
        loader: WeightLoader,
        filename: str,
    ) -> None:
        """Load weights for a single sub-network from an .npz archive."""
        import torch

        data = loader.load_npz(filename)
        params = list(net.parameters())
        sorted_keys = sorted(data.keys())

        if len(sorted_keys) != len(params):
            logger.warning(
                "Mismatch between %s arrays (%d) and network parameters (%d). "
                "Extra arrays or missing parameters will be ignored.",
                filename,
                len(sorted_keys),
                len(params),
            )

        loaded = 0
        for key, param in zip(sorted_keys, params):
            arr = data[key]
            tensor = torch.as_tensor(
                arr, dtype=torch.float32, device=torch.device(self._config.device)
            )
            if tensor.shape == param.shape:
                with torch.no_grad():
                    param.copy_(tensor)
                loaded += 1
            else:
                logger.warning(
                    "Shape mismatch for %s in %s: npz=%s, param=%s — skipping",
                    key,
                    filename,
                    tensor.shape,
                    param.shape,
                )
        logger.info("Loaded %d/%d arrays from %s", loaded, len(sorted_keys), filename)

    def load_from_hub(
        self,
        loader: WeightLoader,
        *,
        belief_file: str = DEFAULT_BELIEF_WEIGHT_FILE,
        desire_file: str = DEFAULT_DESIRE_WEIGHT_FILE,
        intention_file: str = DEFAULT_INTENTION_WEIGHT_FILE,
        affect_file: str = DEFAULT_AFFECT_WEIGHT_FILE,
    ) -> None:
        """Download and load all BDI weights from HuggingFace Hub.

        Args:
            loader: A configured :class:`WeightLoader`.
            belief_file: Path for belief encoder weights.
            desire_file: Path for desire encoder weights.
            intention_file: Path for intention predictor weights.
            affect_file: Path for affect estimator weights.
        """
        self._load_sub_network_from_npz(self.belief_net, loader, belief_file)
        self._load_sub_network_from_npz(self.desire_net, loader, desire_file)
        self._load_sub_network_from_npz(self.intention_net, loader, intention_file)
        self._load_sub_network_from_npz(self.affect_net, loader, affect_file)
        logger.info("BDINetwork loaded all weights from hub")
