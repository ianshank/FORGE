"""Explicit Gymnasium registry integration for FORGE.

Most of the RL ecosystem instantiates environments by id — ``gymnasium.make``,
Stable-Baselines3's ``make_vec_env``, CleanRL's ``--env-id`` — so an environment
that cannot be looked up by id is awkward to consume even when its API is
perfectly compliant. Registering an id also gives the env a ``spec``, which lets
``gymnasium.utils.env_checker.check_env`` exercise alternative render modes it
otherwise skips.

Registration is a **function you call**, never an import side effect. FORGE's
extensibility invariant is explicit registration (``docs/CHARTER.md`` Invariant 1:
"no auto-registration via static-import side effects"), and this is the same
rule applied to the Python surface. It also matches Gymnasium's own modern
idiom, where a plugin package exposes registration for the caller to invoke
rather than mutating a global registry the moment it is imported.

Typical use::

    import gymnasium as gym
    from forge_env import register_envs

    register_envs()
    env = gym.make("Forge-v0")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["FORGE_ENV_ENTRY_POINT", "FORGE_ENV_ID", "is_registered", "register_envs"]

#: Gymnasium id for the single-agent FORGE environment. Versioned per
#: Gymnasium's ``Name-vN`` convention: bump the suffix if the observation or
#: action space changes incompatibly, so pinned downstream code keeps working.
FORGE_ENV_ID: str = "Forge-v0"

#: Import path Gymnasium uses to construct the environment.
FORGE_ENV_ENTRY_POINT: str = "forge_env.gymnasium_env:ForgeGymnasiumEnv"


def is_registered(env_id: str = FORGE_ENV_ID) -> bool:
    """Return whether ``env_id`` is present in the Gymnasium registry.

    Args:
        env_id: The id to look up.

    Returns:
        ``True`` if registered, ``False`` if not or if gymnasium is unavailable.
    """
    try:
        import gymnasium
    except ImportError:
        return False
    return env_id in gymnasium.registry


def register_envs(
    *,
    env_id: str = FORGE_ENV_ID,
    entry_point: str = FORGE_ENV_ENTRY_POINT,
    kwargs: Mapping[str, Any] | None = None,
    force: bool = False,
) -> str:
    """Register FORGE's environment id with Gymnasium.

    Idempotent by default: registering an id that is already present is a no-op,
    so calling this from several entry points in one process is safe.

    Args:
        env_id: Id to register under. Override to expose a preconfigured variant
            alongside the default (for example a fixed-seed evaluation env).
        entry_point: Import path Gymnasium constructs from.
        kwargs: Constructor keyword arguments baked into the registration, such
            as ``{"config": {...}}`` for a preconfigured variant.
        force: Re-register even when ``env_id`` is already present.

    Returns:
        The registered ``env_id``, so callers can chain into ``gymnasium.make``.

    Raises:
        ImportError: If gymnasium is not installed.
    """
    try:
        import gymnasium
    except ImportError as exc:  # pragma: no cover - exercised only without gymnasium
        raise ImportError(
            "gymnasium is required to register FORGE environments. "
            "Install it with: pip install gymnasium"
        ) from exc

    if env_id in gymnasium.registry and not force:
        logger.debug("Gymnasium id %r is already registered; leaving it as is.", env_id)
        return env_id

    gymnasium.register(
        id=env_id,
        entry_point=entry_point,
        kwargs=dict(kwargs) if kwargs else {},
    )
    logger.debug("Registered Gymnasium id %r -> %s.", env_id, entry_point)
    return env_id
