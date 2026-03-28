"""MouseDroid composite agent integrating RSSM, BDI, and Constitutional RL.

Composes pre-trained sub-components from the ``ianshank/mousedroid-weights``
HuggingFace repository into a single :class:`BaseAgent`-compatible agent.

Components:
    - RSSM world model for latent-space dynamics
    - BDI encoder for belief/desire/intention/affect
    - Neural MCTS policy for action prior estimation
    - Constitutional RL (actor-critic) for PPO fine-tuning

Usage::

    from forge.agents.mousedroid_agent import MouseDroidAgent, MouseDroidConfig

    agent = MouseDroidAgent(MouseDroidConfig())
    agent.load_from_hub()
    action, info = agent.act(observation)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.config import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM
from forge.models.bdi_network import (
    DEFAULT_AFFECT_DIM,
    DEFAULT_BDI_HIDDEN_SIZES,
    DEFAULT_BELIEF_DIM,
    DEFAULT_DESIRE_DIM,
    DEFAULT_INTENTION_DIM,
    BDIConfig,
    BDINetwork,
)
from forge.models.neural_policy import (
    DEFAULT_POLICY_HIDDEN_SIZES,
    NeuralMCTSPolicy,
    NeuralPolicyConfig,
)
from forge.models.rssm_world_model import (
    DEFAULT_DETERMINISTIC_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_STATE_DIM,
    DEFAULT_STOCHASTIC_DIM,
    RSSMConfig,
    RSSMWorldModel,
)
from forge.utils.weight_loader import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    WeightLoader,
    WeightLoaderConfig,
)

if TYPE_CHECKING:
    from forge.models.policy_network import ActorCriticNetwork

logger = logging.getLogger(__name__)

DEFAULT_CONSTITUTIONAL_HIDDEN_SIZES: list[int] = [256, 256]

# PPO / constitutional RL training defaults
DEFAULT_PPO_CLIP_RATIO: float = 0.2
DEFAULT_CONSTITUTIONAL_VALUE_COEFF: float = 0.5
DEFAULT_CONSTITUTIONAL_ENTROPY_COEFF: float = 0.01
DEFAULT_CONSTITUTIONAL_GRAD_CLIP: float = 0.5

# Numerical stability epsilon for intention modulation
DEFAULT_INTENTION_MODULATION_EPSILON: float = 1e-8

# File suffixes used when saving/loading composite agent checkpoints
_RSSM_SUFFIX = ".rssm.pt"
_BDI_SUFFIX = ".bdi.pt"
_POLICY_SUFFIX = ".policy.pt"
_CONSTITUTIONAL_SUFFIX = ".constitutional.pt"


@dataclass
class MouseDroidConfig(AgentConfig):
    """Configuration for :class:`MouseDroidAgent`.

    Aggregates all sub-component configurations into a single dataclass.
    Extends :class:`AgentConfig` for compatibility with the agent framework.
    """

    name: str = "mousedroid"

    # HuggingFace Hub
    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    cache_dir: str | None = None

    # Shared dimensions
    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    device: str = "auto"

    # RSSM world model
    rssm_state_dim: int = DEFAULT_STATE_DIM
    rssm_hidden_dim: int = DEFAULT_HIDDEN_DIM
    rssm_stochastic_dim: int = DEFAULT_STOCHASTIC_DIM
    rssm_deterministic_dim: int = DEFAULT_DETERMINISTIC_DIM

    # BDI
    belief_dim: int = DEFAULT_BELIEF_DIM
    desire_dim: int = DEFAULT_DESIRE_DIM
    intention_dim: int = DEFAULT_INTENTION_DIM
    affect_dim: int = DEFAULT_AFFECT_DIM
    bdi_hidden_sizes: list[int] = field(
        default_factory=lambda: list(DEFAULT_BDI_HIDDEN_SIZES)
    )

    # MCTS policy
    policy_hidden_sizes: list[int] = field(
        default_factory=lambda: list(DEFAULT_POLICY_HIDDEN_SIZES)
    )

    # Constitutional RL
    constitutional_hidden_sizes: list[int] = field(
        default_factory=lambda: list(DEFAULT_CONSTITUTIONAL_HIDDEN_SIZES)
    )

    # PPO training hyperparameters
    ppo_clip_ratio: float = DEFAULT_PPO_CLIP_RATIO
    constitutional_value_coeff: float = DEFAULT_CONSTITUTIONAL_VALUE_COEFF
    constitutional_entropy_coeff: float = DEFAULT_CONSTITUTIONAL_ENTROPY_COEFF
    constitutional_grad_clip: float = DEFAULT_CONSTITUTIONAL_GRAD_CLIP

    # Numerical stability
    intention_modulation_epsilon: float = DEFAULT_INTENTION_MODULATION_EPSILON

    # Reproducibility
    seed: int | None = None

    # Behaviour — set to True to auto-download weights on construction
    auto_download: bool = False


class MouseDroidAgent(BaseAgent):
    """Composite agent integrating RSSM, BDI, neural policy, and constitutional RL.

    On construction, creates all sub-components.  Call :meth:`load_from_hub`
    to download and load pre-trained weights from HuggingFace.

    Args:
        config: MouseDroid configuration.
    """

    def __init__(self, config: MouseDroidConfig) -> None:
        super().__init__(config)
        self._md_config = config

        device = config.device
        if device == "auto":
            from forge.utils.device import get_device  # noqa: PLC0415

            device = get_device()
        self._device = device

        # Build sub-components
        self._world_model = RSSMWorldModel(
            RSSMConfig(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                state_dim=config.rssm_state_dim,
                hidden_dim=config.rssm_hidden_dim,
                stochastic_dim=config.rssm_stochastic_dim,
                deterministic_dim=config.rssm_deterministic_dim,
                device=device,
            )
        )

        self._bdi = BDINetwork(
            BDIConfig(
                obs_dim=config.obs_dim,
                belief_dim=config.belief_dim,
                desire_dim=config.desire_dim,
                intention_dim=config.intention_dim,
                affect_dim=config.affect_dim,
                hidden_sizes=config.bdi_hidden_sizes,
                device=device,
            )
        )

        self._policy = NeuralMCTSPolicy(
            NeuralPolicyConfig(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                hidden_sizes=config.policy_hidden_sizes,
                device=device,
            )
        )

        # Constitutional RL policy (actor-critic for PPO fine-tuning)
        from forge.models.policy_network import ActorCriticNetwork  # noqa: PLC0415

        self._constitutional = ActorCriticNetwork(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_sizes=config.constitutional_hidden_sizes,
            device=device,
        )

        self._rng = np.random.default_rng(config.seed)

        logger.info(
            "MouseDroidAgent initialised: obs=%d, act=%d, device=%s",
            config.obs_dim,
            config.action_dim,
            device,
        )

        if config.auto_download:
            try:
                self.load_from_hub()
            except (ImportError, OSError) as exc:
                logger.warning(
                    "auto_download enabled but weight loading failed: %s. "
                    "Call load_from_hub() manually when dependencies are available.",
                    exc,
                )

    # ----- Properties -----

    @property
    def mousedroid_config(self) -> MouseDroidConfig:
        """Return the MouseDroid configuration."""
        return self._md_config

    @property
    def world_model(self) -> RSSMWorldModel:
        """Return the RSSM world model sub-component."""
        return self._world_model

    @property
    def bdi(self) -> BDINetwork:
        """Return the BDI network sub-component."""
        return self._bdi

    @property
    def policy(self) -> NeuralMCTSPolicy:
        """Return the neural MCTS policy sub-component."""
        return self._policy

    @property
    def constitutional_policy(self) -> ActorCriticNetwork:
        """Return the constitutional RL actor-critic network."""
        return self._constitutional

    @property
    def device(self) -> str:
        """Return the compute device."""
        return self._device

    # ----- BaseAgent interface -----

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action using the BDI-conditioned policy.

        Pipeline:
            1. BDI encoding: observation -> belief/desire/intention/affect
            2. Policy evaluation: observation -> action priors + value
            3. Action selection: sample from priors

        Args:
            observation: Flat observation array of shape ``(obs_dim,)``.

        Returns:
            ``(action_id, info_dict)`` where info contains BDI state,
            policy priors, and value estimate.
        """
        # BDI encoding
        bdi_state = self._bdi.forward(observation)

        # Policy evaluation
        priors, value = self._policy.evaluate(observation)

        # Modulate priors with per-action intention scaling
        eps = self._md_config.intention_modulation_epsilon
        intention_abs = np.abs(bdi_state.intention).astype(priors.dtype)
        action_scale = np.resize(intention_abs, priors.shape)
        scale_mean = float(action_scale.mean())
        if scale_mean > 0.0:
            action_scale = action_scale / (scale_mean + eps)
        else:
            action_scale = np.ones_like(priors)
        modulated_priors = priors * action_scale
        total = float(modulated_priors.sum())
        if total > eps:
            modulated_priors = modulated_priors / total
            # Clip any negative FP artefacts then renormalize to a valid distribution
            modulated_priors = np.clip(modulated_priors, 0.0, None)
            sum_probs = float(modulated_priors.sum())
            if sum_probs == 0.0:
                modulated_priors = np.full_like(priors, 1.0 / len(priors))
            else:
                modulated_priors = modulated_priors / sum_probs
        else:
            modulated_priors = np.full_like(priors, 1.0 / len(priors))

        # Sample action from modulated distribution
        action = int(self._rng.choice(len(modulated_priors), p=modulated_priors))

        self._step_count += 1

        return action, {
            "value": float(value),
            "priors": priors.tolist(),
            "belief_norm": float(np.linalg.norm(bdi_state.belief)),
            "desire_norm": float(np.linalg.norm(bdi_state.desire)),
            "intention_norm": float(np.linalg.norm(bdi_state.intention)),
            "affect_norm": float(np.linalg.norm(bdi_state.affect)),
        }

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Update the constitutional RL policy with PPO.

        Delegates to the ActorCriticNetwork for policy/value updates.

        Expected batch keys:
            observations: ``(N, obs_dim)``
            actions: ``(N,)``
            old_log_probs: ``(N,)``
            advantages: ``(N,)``
            returns: ``(N,)``

        Returns:
            Dictionary of training metrics.
        """
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        dev = torch.device(self._device)
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=dev)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=dev)
        old_log_probs = torch.as_tensor(
            batch["old_log_probs"], dtype=torch.float32, device=dev
        )
        advantages = torch.as_tensor(
            batch["advantages"], dtype=torch.float32, device=dev
        )
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=dev)

        # Forward through constitutional policy
        action_logits, values = self._constitutional.forward(obs)
        values = values.squeeze(-1)

        from torch.distributions import Categorical  # noqa: PLC0415

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO clipped objective
        clip = self._md_config.ppo_clip_ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = (
            policy_loss
            + self._md_config.constitutional_value_coeff * value_loss
            - self._md_config.constitutional_entropy_coeff * entropy
        )

        self._constitutional.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._constitutional.parameters(),
            self._md_config.constitutional_grad_clip,
        )
        self._constitutional.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "loss": float(loss.item()),
        }

    # ----- Hub loading -----

    def load_from_hub(self, loader: WeightLoader | None = None) -> None:
        """Download and load all pre-trained weights from HuggingFace Hub.

        Creates a :class:`WeightLoader` if none is provided, using the
        agent's configuration.

        Args:
            loader: Optional pre-configured weight loader.
        """
        if loader is None:
            loader = WeightLoader(
                WeightLoaderConfig(
                    repo_id=self._md_config.repo_id,
                    revision=self._md_config.revision,
                    cache_dir=self._md_config.cache_dir,
                )
            )

        self._world_model.load_from_hub(loader)
        self._bdi.load_from_hub(loader)
        self._policy.load_from_npz(loader)

        # Load constitutional RL weights (policy.npz + value.npz)
        self._load_constitutional_from_hub(loader)

        logger.info("MouseDroidAgent: all weights loaded from hub")

    def _load_constitutional_from_hub(self, loader: WeightLoader) -> None:
        """Load constitutional RL policy and value weights from .npz files."""
        import torch  # noqa: PLC0415

        # Load policy weights into encoder + actor head
        policy_data = loader.load_npz("policy.npz")
        encoder_params = list(self._constitutional.encoder.parameters())
        actor_params = list(self._constitutional.actor_head.parameters())
        all_policy_params = encoder_params + actor_params
        sorted_keys = sorted(policy_data.keys())

        if len(sorted_keys) != len(all_policy_params):
            logger.warning(
                "Mismatch between policy.npz arrays (%d) and policy parameters (%d). "
                "Some parameters may remain uninitialized or some arrays may be unused.",
                len(sorted_keys),
                len(all_policy_params),
            )

        loaded = 0
        for key, param in zip(sorted_keys, all_policy_params):
            arr = policy_data[key]
            tensor = torch.as_tensor(
                arr, dtype=torch.float32, device=torch.device(self._device)
            )
            if tensor.shape == param.shape:
                with torch.no_grad():
                    param.copy_(tensor)
                loaded += 1
            else:
                logger.warning(
                    "Shape mismatch for policy param %s: npz=%s, param=%s — skipping",
                    key, tensor.shape, param.shape,
                )
        logger.info(
            "Constitutional policy: loaded %d/%d arrays", loaded, len(sorted_keys)
        )

        # Load value weights into critic head
        value_data = loader.load_npz("value.npz")
        critic_params = list(self._constitutional.critic_head.parameters())
        sorted_value_keys = sorted(value_data.keys())

        if len(sorted_value_keys) != len(critic_params):
            logger.warning(
                "Mismatch between value.npz arrays (%d) and critic parameters (%d). "
                "Some parameters may remain uninitialized or some arrays may be unused.",
                len(sorted_value_keys),
                len(critic_params),
            )

        loaded_v = 0
        for key, param in zip(sorted_value_keys, critic_params):
            arr = value_data[key]
            tensor = torch.as_tensor(
                arr, dtype=torch.float32, device=torch.device(self._device)
            )
            if tensor.shape == param.shape:
                with torch.no_grad():
                    param.copy_(tensor)
                loaded_v += 1
            else:
                logger.warning(
                    "Shape mismatch for value param %s: npz=%s, param=%s — skipping",
                    key, tensor.shape, param.shape,
                )
        logger.info(
            "Constitutional value: loaded %d/%d arrays", loaded_v, len(sorted_value_keys)
        )

    # ----- Save / Load -----

    def save(self, path: str) -> None:
        """Save all sub-component weights and metadata."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        self._world_model.save(str(base.with_suffix(_RSSM_SUFFIX)))
        self._bdi.save(str(base.with_suffix(_BDI_SUFFIX)))
        self._policy.save(str(base.with_suffix(_POLICY_SUFFIX)))
        self._constitutional.save(str(base.with_suffix(_CONSTITUTIONAL_SUFFIX)))

        # Metadata — serialize full config; key dimensions validated on load
        meta = {
            "step_count": self._step_count,
            "config": asdict(self._md_config),
        }
        with base.with_suffix(".json").open("w") as f:
            json.dump(meta, f)

        logger.info("MouseDroidAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load all sub-component weights and metadata."""
        base = Path(path)

        self._world_model.load(str(base.with_suffix(_RSSM_SUFFIX)))
        self._bdi.load(str(base.with_suffix(_BDI_SUFFIX)))
        self._policy.load(str(base.with_suffix(_POLICY_SUFFIX)))
        self._constitutional.load(str(base.with_suffix(_CONSTITUTIONAL_SUFFIX)))

        meta_path = base.with_suffix(".json")
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            self._step_count = meta.get("step_count", 0)
            saved_cfg = meta.get("config", {})
            for key in ("obs_dim", "action_dim"):
                saved_val = saved_cfg.get(key)
                current_val = getattr(self._md_config, key, None)
                if saved_val is not None and saved_val != current_val:
                    logger.warning(
                        "Checkpoint %s=%d does not match current config %s=%d; "
                        "architecture mismatch may cause errors.",
                        key,
                        saved_val,
                        key,
                        current_val,
                    )

        logger.info("MouseDroidAgent loaded from %s", path)
