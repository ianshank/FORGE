"""OpenEnv sidecar over in-process ``ForgeEnv``.

Rewards and termination stay inside the FORGE step loop (the same
``forge-task`` predicates attached by the scenario compiler). This is
an eval / LLM-tool distribution adapter: Observation objects carry
``reward`` / ``done``. Training remains on PyO3 / Gymnasium.

``openenv`` is optional. Without it this module still exposes the
contract so unit tests and in-process callers do not need the Hub SDK.
Do not put OpenEnv on the PyO3 hot path or replace ``Env`` / ``FlatObsEnv``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openenv.core.env_server import Environment as _SdkEnvironment

    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False
    _SdkEnvironment = None


@dataclass
class ForgeAction:
    """Discrete FORGE action for the OpenEnv wire format."""

    action_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeObservation:
    """OpenEnv observation: payload plus ``reward`` / ``done``."""

    observation: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    done: bool = False
    reward: float | None = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeState:
    """Episode metadata for OpenEnv ``state``."""

    episode_id: str = ""
    step_count: int = 0
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)


class _EnvironmentBase:
    """Fallback Environment contract when the OpenEnv SDK is absent."""


_EnvBase = _SdkEnvironment if HAS_OPENENV and _SdkEnvironment is not None else _EnvironmentBase


def _new_episode_id() -> str:
    return str(uuid.uuid4())


class ForgeOpenEnv(_EnvBase):  # type: ignore[misc, valid-type]
    """In-process OpenEnv wrapper around ``ForgeEnv`` / Gymnasium."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        inner: Any | None = None,
        rubric: Any | None = None,
    ) -> None:
        if HAS_OPENENV:
            super().__init__(rubric=rubric)
        else:
            super().__init__()
            if rubric is not None:
                logger.debug("rubric ignored without the OpenEnv SDK")
        self._config = dict(config or {})
        if inner is not None:
            self._env = inner
        else:
            try:
                from forge_env.forge_env import ForgeEnv
            except ImportError as exc:
                msg = (
                    "forge_env native module not found. "
                    "Install with maturin develop, or pass inner=..."
                )
                raise ImportError(msg) from exc
            self._env = ForgeEnv(config=self._config or None)
        self._state = ForgeState(episode_id=_new_episode_id(), config=self._config)

    def reset(self, seed: int | None = None, **kwargs: Any) -> ForgeObservation:
        _ = kwargs
        reset_rubric = getattr(self, "_reset_rubric", None)
        if callable(reset_rubric):
            reset_rubric()
        result = self._env.reset(seed=seed)
        obs, info = _unpack_reset(result)
        self._state = ForgeState(
            episode_id=_new_episode_id(),
            step_count=0,
            seed=seed,
            config=self._config,
        )
        return ForgeObservation(
            observation=obs,
            info=info,
            terminated=False,
            truncated=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: ForgeAction, **kwargs: Any) -> ForgeObservation:
        _ = kwargs
        packed = self._env.step(int(action.action_id))
        obs, reward, terminated, truncated, info = _unpack_step(packed)
        self._state.step_count += 1
        done = bool(terminated or truncated)
        observation = ForgeObservation(
            observation=obs,
            info=info,
            terminated=bool(terminated),
            truncated=bool(truncated),
            done=done,
            reward=float(reward),
        )
        apply_rubric = getattr(self, "_apply_rubric", None)
        if callable(apply_rubric):
            apply_rubric(action, observation)
        return observation

    @property
    def state(self) -> ForgeState:
        return self._state

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()


def _unpack_reset(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        obs_dict = dict(obs) if isinstance(obs, Mapping) else {"value": obs}
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        return obs_dict, info_dict
    if isinstance(result, Mapping):
        return dict(result), {}
    return {"value": result}, {}


def _unpack_step(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        obs_dict = dict(obs) if isinstance(obs, Mapping) else {"value": obs}
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        return obs_dict, float(reward), bool(terminated), bool(truncated), info_dict
    msg = f"inner env.step must return a Gymnasium 5-tuple, got {type(result)!r}"
    raise TypeError(msg)


def create_forge_openenv_app(
    config: Mapping[str, Any] | None = None,
    env_factory: Callable[[], ForgeOpenEnv] | None = None,
) -> Any:
    """FastAPI app via OpenEnv ``create_app``. Requires the optional SDK."""
    try:
        from openenv.core.env_server import create_app
    except ImportError as exc:
        raise ImportError(
            "openenv is not installed. Training remains on PyO3; "
            "pip-install the OpenEnv SDK only to serve this sidecar."
        ) from exc

    factory = env_factory or (lambda: ForgeOpenEnv(config=config))
    return create_app(
        factory,
        ForgeAction,
        ForgeObservation,
        env_name="forge",
    )
