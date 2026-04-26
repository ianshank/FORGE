"""Adapter layer bridging AlphaGalerkin models to FORGE Gymnasium environments.

AlphaGalerkin is an optional dependency; all imports are guarded behind a
try/except so FORGE continues to function when it is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional AlphaGalerkin import
# ---------------------------------------------------------------------------

ALPHAGALERKIN_AVAILABLE = False
try:
    # Probe import: we only need to know whether AlphaGalerkin is installed.
    # The actual `GameInterface` symbol is unused in this module — the adapter
    # talks to AlphaGalerkin via duck-typed objects. Aliasing to `_` makes the
    # intent explicit and silences ruff F401.
    from src.games.interface import GameInterface as _AlphaGalerkinProbe  # noqa: F401

    ALPHAGALERKIN_AVAILABLE = True
except ImportError:
    logger.debug("AlphaGalerkin not available")


# ---------------------------------------------------------------------------
# ForgeGameState
# ---------------------------------------------------------------------------


@dataclass
class ForgeGameState:
    """Immutable snapshot of a FORGE environment step, compatible with AG tree search."""

    observation: np.ndarray
    done: bool
    reward: float
    info: dict[str, Any]
    move_number: int = 0


# ---------------------------------------------------------------------------
# ForgeGameAdapter
# ---------------------------------------------------------------------------


class ForgeGameAdapter:
    """Wraps a FORGE Gymnasium environment for AlphaGalerkin-style tree search.

    AlphaGalerkin expects a functional interface over game states; this adapter
    bridges FORGE's stateful Gymnasium API to that contract.
    """

    def __init__(self, env: Any) -> None:
        """Initialise the adapter with a Gymnasium-compatible FORGE environment.

        Parameters
        ----------
        env:
            A Gymnasium environment (or compatible mock) with ``reset()``,
            ``step()``, ``observation_space``, and ``action_space`` attributes.
        """
        self._env = env

    @property
    def action_space_size(self) -> int:
        """Return the number of discrete actions in the environment."""
        space = self._env.action_space
        # Prefer attribute access (Gymnasium spaces.Discrete, SimpleNamespace),
        # fall back to mapping access for backwards compatibility.
        n = getattr(space, "n", None)
        if n is None:
            try:
                n = space["n"]
            except (TypeError, KeyError) as exc:
                msg = (
                    "Unsupported action_space type: expected an object with "
                    "an 'n' attribute or a mapping with key 'n'."
                )
                raise TypeError(msg) from exc
        return int(n)

    def initial_state(self) -> ForgeGameState:
        """Reset the environment and return the initial ``ForgeGameState``."""
        obs, info = self._env.reset()
        observation = np.asarray(obs, dtype=np.float32)
        return ForgeGameState(
            observation=observation,
            done=False,
            reward=0.0,
            info=info if isinstance(info, dict) else {},
            move_number=0,
        )

    def apply_action(self, state: ForgeGameState, action: int) -> ForgeGameState:
        """Apply *action* to the environment and return a new ``ForgeGameState``.

        .. warning::

            This method mutates the underlying Gymnasium environment.  It is
            **not** suitable for MCTS-style tree search where multiple branches
            must be explored from the same state.  A future version should
            snapshot/clone the environment before stepping to support true
            functional semantics.

        Parameters
        ----------
        state:
            The current game state (used only for ``move_number``).
        action:
            The discrete action index to apply.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        observation = np.asarray(obs, dtype=np.float32)
        return ForgeGameState(
            observation=observation,
            done=done,
            reward=float(reward),
            info=info if isinstance(info, dict) else {},
            move_number=state.move_number + 1,
        )

    def get_legal_actions(self, state: ForgeGameState) -> list[int]:
        """Return the list of legal action indices for *state*.

        If ``state.info`` contains an ``action_mask`` key (a sequence of
        booleans/ints of length ``action_space_size``), only the actions
        whose mask entry is truthy are returned.  Otherwise all actions are
        considered legal.
        """
        mask = state.info.get("action_mask")
        if mask is not None:
            return [i for i, allowed in enumerate(mask) if allowed]
        return list(range(self.action_space_size))

    def is_terminal(self, state: ForgeGameState) -> bool:
        """Return ``True`` if *state* represents a terminal game position."""
        return state.done

    def to_tensor(self, state: ForgeGameState) -> np.ndarray:
        """Return the observation array for *state* (suitable for model input)."""
        return state.observation


# ---------------------------------------------------------------------------
# AlphaGalerkinAgent
# ---------------------------------------------------------------------------


class AlphaGalerkinAgent(BaseAgent):
    """Wraps an AlphaGalerkin model as a FORGE ``BaseAgent``.

    The agent runs the AG model's forward pass to obtain action logits,
    applies softmax, and samples an action.  Learning is delegated to
    AlphaGalerkin's own trainer pipeline; ``learn()`` is therefore a no-op
    that logs a warning and returns an empty metrics dict.
    """

    def __init__(
        self,
        config: AgentConfig,
        model: Any,
        action_space_size: int,
    ) -> None:
        """Initialise the agent.

        Parameters
        ----------
        config:
            Standard FORGE agent configuration.
        model:
            An AlphaGalerkin model object with a callable forward interface
            (``model(tensor)`` returning logits).
        action_space_size:
            Number of discrete actions in the environment.
        """
        super().__init__(config)
        self._model = model
        self._action_space_size = action_space_size

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action via the AG model's forward pass.

        Performs a softmax over the returned logits and samples an action.

        Parameters
        ----------
        observation:
            A flat or shaped numpy observation array.

        Returns
        -------
        tuple[int, dict[str, Any]]
            ``(action_id, info)`` where *info* contains the raw logits and
            computed probabilities.
        """
        import torch  # lazy import — torch is optional at module level

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, num_samples=1).item())
        self._step_count += 1
        return action, {
            "logits": logits.squeeze(0).tolist(),
            "probs": probs.squeeze(0).tolist(),
        }

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op — AlphaGalerkin uses its own trainer pipeline.

        Logs a warning to make the no-op visible during development.
        """
        logger.warning(
            "AlphaGalerkinAgent.learn() called but AlphaGalerkin uses its own "
            "trainer; ignoring batch of size %d.",
            len(next(iter(batch.values()), [])) if batch else 0,
        )
        return {}
