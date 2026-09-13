"""Build Gymnasium spaces from the native descriptor, and fit observations to them.

Both public wrappers — :mod:`forge_env.gymnasium_env` (single-agent) and
:mod:`forge_env.pettingzoo_env` (multi-agent Parallel API) — expose the *same*
underlying observation and action spaces, because both wrap the same native
``ForgeEnv``. This module is that shared definition, so the two wrappers cannot
drift apart and neither restates a shape, bound, or dtype of its own.

Two responsibilities, deliberately kept together because they are two halves of
one contract:

**Building the spaces.** ``ForgeEnv.observation_space`` / ``.action_space``
return plain dicts describing shapes, bounds, and dtypes, all derived Rust-side
from :class:`ForgeConfig` (``crates/forge-python/src/spaces.rs``). Every value
here is read from that descriptor rather than restated, so changing
``default_vision_radius``, ``default_carry_capacity``, ``comm_buffer_size``,
``comm_vocab_size``, or the enabled action families in config automatically
changes the Python spaces. The module-level fallbacks are used only when a
descriptor key is absent, which keeps older native builds (and the test doubles
in ``tests/python/conftest.py``, which mock a descriptor carrying only the flat
convenience keys) working unchanged.

**Fitting observations to them.** A declared space is a promise, and upstream
compliance suites check it: ``gymnasium.utils.env_checker.check_env`` and
``pettingzoo.test.parallel_api_test`` both assert every returned observation is
contained in its declared space. The native env returns Python scalars, tuples,
and a *variable-length* message list, none of which satisfy a fixed-shape
``Box`` as-is. :func:`fit_observation` performs that coercion in one place for
both wrappers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover - exercised only without gymnasium
    HAS_GYMNASIUM = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt
    from gymnasium import spaces as spaces_t

__all__ = [
    "DEFAULT_ACTION_COUNT",
    "DEFAULT_CARRY_CAPACITY",
    "DEFAULT_GRID_CHANNELS",
    "DEFAULT_NUM_DAY_PHASES",
    "DEFAULT_VIEW_SIDE",
    "DEFAULT_VISION_RADIUS",
    "DISCRETE_OBSERVATION_KEYS",
    "INVENTORY_FIELDS_PER_SLOT",
    "MESSAGE_PAD_VALUE",
    "POSITION_DIMENSIONS",
    "UINT16_MAX",
    "VARIABLE_LENGTH_OBSERVATION_KEYS",
    "build_action_space",
    "build_observation_space",
    "fit_observation",
]

# ---------------------------------------------------------------------------
# Fallbacks. Used ONLY when the native descriptor omits a key -- every one of
# these mirrors a Rust-side constant, and the descriptor is authoritative when
# present. They exist so a descriptor from an older native build, or a test
# double that supplies only the flat convenience keys, still yields a usable
# space instead of a KeyError.
# ---------------------------------------------------------------------------

#: ``forge_types::constants::DEFAULT_VISION_RADIUS``.
DEFAULT_VISION_RADIUS: int = 5
#: A vision radius of ``r`` yields a ``(2r + 1)``-square egocentric view.
DEFAULT_VIEW_SIDE: int = 2 * DEFAULT_VISION_RADIUS + 1
#: ``forge_types::constants::OBS_FEATURES_PER_TILE``.
DEFAULT_GRID_CHANNELS: int = 7
#: ``forge_types::constants::DEFAULT_CARRY_CAPACITY``.
DEFAULT_CARRY_CAPACITY: int = 10
#: ``forge_types::constants::NUM_DAY_PHASES``.
DEFAULT_NUM_DAY_PHASES: int = 4
#: ``Action::space_size(0, false)`` -- base actions, no comm vocabulary, no drone.
DEFAULT_ACTION_COUNT: int = 40

#: Each inventory slot is a ``(item_type, count)`` pair.
INVENTORY_FIELDS_PER_SLOT: int = 2
#: Positions are ``(x, y)`` on the grid.
POSITION_DIMENSIONS: int = 2
#: Upper bound for the ``uint16``-typed observation components.
UINT16_MAX: int = 65535

#: Observation components modelled as :class:`~gymnasium.spaces.Discrete`
#: rather than :class:`~gymnasium.spaces.Box`.
DISCRETE_OBSERVATION_KEYS: frozenset[str] = frozenset({"day_phase"})

#: Components the native env returns with a run-varying length, which
#: :func:`fit_observation` pads or truncates to the declared fixed shape. The
#: message buffer holds only the messages actually received this tick, while the
#: declared space is sized by ``comm_buffer_size``.
VARIABLE_LENGTH_OBSERVATION_KEYS: frozenset[str] = frozenset({"messages"})

#: Fill value for padding a short variable-length component. ``0`` is the
#: "no message" slot in the comm vocabulary.
MESSAGE_PAD_VALUE: int = 0


def _require_gymnasium() -> None:
    """Raise a directive ImportError when gymnasium is absent.

    Raises:
        ImportError: If gymnasium (and its numpy dependency) is not installed.
    """
    if not HAS_GYMNASIUM:
        raise ImportError(
            "gymnasium is required to build FORGE observation/action spaces. "
            "Install it with: pip install gymnasium"
        )


def _component(descriptor: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return the nested descriptor entry for ``key``, or an empty mapping."""
    entry = descriptor.get(key)
    if isinstance(entry, Mapping):
        return entry
    return {}


def _grid_view_shape(descriptor: Mapping[str, Any]) -> tuple[int, ...]:
    """Resolve the grid-view shape from the descriptor.

    Prefers the nested ``grid_view.shape``; falls back to the flat
    ``grid_view_{height,width,channels}`` convenience keys the native descriptor
    also publishes, then to the module fallbacks.
    """
    nested = _component(descriptor, "grid_view").get("shape")
    if nested is not None:
        return tuple(int(dim) for dim in nested)
    return (
        int(descriptor.get("grid_view_height", DEFAULT_VIEW_SIDE)),
        int(descriptor.get("grid_view_width", DEFAULT_VIEW_SIDE)),
        int(descriptor.get("grid_view_channels", DEFAULT_GRID_CHANNELS)),
    )


def _inventory_shape(descriptor: Mapping[str, Any]) -> tuple[int, ...]:
    """Resolve the inventory shape as ``(capacity, fields_per_slot)``."""
    nested = _component(descriptor, "inventory").get("shape")
    if nested is not None:
        return tuple(int(dim) for dim in nested)
    return (
        int(descriptor.get("inventory_capacity", DEFAULT_CARRY_CAPACITY)),
        INVENTORY_FIELDS_PER_SLOT,
    )


def _shape_of(
    descriptor: Mapping[str, Any], key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    """Resolve a component's shape, falling back to ``default``."""
    nested = _component(descriptor, key).get("shape")
    if nested is None:
        return default
    return tuple(int(dim) for dim in nested)


def _day_phase_count(descriptor: Mapping[str, Any]) -> int:
    """Resolve the number of day phases from the descriptor's inclusive bound."""
    high = _component(descriptor, "day_phase").get("high")
    if high is None:
        return DEFAULT_NUM_DAY_PHASES
    # The descriptor publishes an inclusive upper bound; Discrete takes a count.
    return int(high) + 1


def _dtype_of(descriptor: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve a component dtype from the descriptor, with fallback on absence/parse failure."""
    dtype_name = _component(descriptor, key).get("dtype")
    if dtype_name is None:
        return np.dtype(fallback)
    try:
        return np.dtype(dtype_name)
    except TypeError:
        logger.debug("Invalid dtype %r for %r; using fallback %r.", dtype_name, key, fallback)
        return np.dtype(fallback)


def _bounds_of(
    descriptor: Mapping[str, Any], key: str, low_fallback: Any, high_fallback: Any
) -> tuple[Any, Any]:
    """Resolve a component's low/high bounds from the descriptor with fallbacks."""
    component = _component(descriptor, key)
    return component.get("low", low_fallback), component.get("high", high_fallback)


def build_observation_space(descriptor: Mapping[str, Any]) -> spaces_t.Dict:
    """Build the Gymnasium observation space from a native descriptor.

    Args:
        descriptor: The mapping returned by ``ForgeEnv.observation_space``.

    Returns:
        A :class:`~gymnasium.spaces.Dict` whose entries match what
        :func:`fit_observation` produces.

    Raises:
        ImportError: If gymnasium is not installed.
    """
    _require_gymnasium()

    grid_low, grid_high = _bounds_of(descriptor, "grid_view", 0, np.iinfo(np.uint8).max)
    inventory_low, inventory_high = _bounds_of(descriptor, "inventory", 0, UINT16_MAX)
    health_low, health_high = _bounds_of(descriptor, "health", 0.0, 1.0)
    stamina_low, stamina_high = _bounds_of(descriptor, "stamina", 0.0, 1.0)
    position_low, position_high = _bounds_of(descriptor, "position", 0, UINT16_MAX)
    messages_low, messages_high = _bounds_of(descriptor, "messages", 0, UINT16_MAX)

    return spaces.Dict(
        {
            "grid_view": spaces.Box(
                low=grid_low,
                high=grid_high,
                shape=_grid_view_shape(descriptor),
                dtype=_dtype_of(descriptor, "grid_view", np.uint8),
            ),
            "inventory": spaces.Box(
                low=inventory_low,
                high=inventory_high,
                shape=_inventory_shape(descriptor),
                dtype=_dtype_of(descriptor, "inventory", np.uint16),
            ),
            "health": spaces.Box(
                low=health_low,
                high=health_high,
                shape=_shape_of(descriptor, "health", ()),
                dtype=_dtype_of(descriptor, "health", np.float32),
            ),
            "stamina": spaces.Box(
                low=stamina_low,
                high=stamina_high,
                shape=_shape_of(descriptor, "stamina", ()),
                dtype=_dtype_of(descriptor, "stamina", np.float32),
            ),
            "position": spaces.Box(
                low=position_low,
                high=position_high,
                shape=_shape_of(descriptor, "position", (POSITION_DIMENSIONS,)),
                dtype=_dtype_of(descriptor, "position", np.uint16),
            ),
            "messages": spaces.Box(
                low=messages_low,
                high=int(messages_high),
                shape=_shape_of(descriptor, "messages", (0,)),
                dtype=_dtype_of(descriptor, "messages", np.uint16),
            ),
            "day_phase": spaces.Discrete(_day_phase_count(descriptor)),
        }
    )


def build_action_space(descriptor: Mapping[str, Any]) -> spaces_t.Discrete:
    """Build the Gymnasium action space from a native descriptor.

    Args:
        descriptor: The mapping returned by ``ForgeEnv.action_space``.

    Returns:
        A :class:`~gymnasium.spaces.Discrete` sized by the descriptor's ``n``.

    Raises:
        ImportError: If gymnasium is not installed.
    """
    _require_gymnasium()
    return spaces.Discrete(int(descriptor.get("n", DEFAULT_ACTION_COUNT)))


def _fit_variable_length(
    value: Any, shape: tuple[int, ...], dtype: Any
) -> npt.NDArray[Any]:
    """Pad or truncate a variable-length sequence to a fixed 1-D ``shape``."""
    (width,) = shape
    values = list(value) if isinstance(value, Sequence) else list(np.asarray(value).ravel())
    if len(values) > width:
        logger.debug(
            "Truncating variable-length observation component from %d to %d entries.",
            len(values),
            width,
        )
        values = values[:width]
    elif len(values) < width:
        values = values + [MESSAGE_PAD_VALUE] * (width - len(values))
    return np.asarray(values, dtype=dtype).reshape(shape)


def fit_observation(
    observation: Mapping[str, Any], observation_space: spaces_t.Dict
) -> dict[str, Any]:
    """Coerce a native observation into values contained in ``observation_space``.

    The native env returns Python scalars (``health``), tuples (``position``),
    and a variable-length list (``messages``). Upstream compliance suites assert
    ``space.contains(obs)``, which requires exact dtypes and shapes, so each
    component is cast to its declared space here rather than in each wrapper.

    Unknown keys are ignored so the returned keyset exactly matches the declared
    :class:`~gymnasium.spaces.Dict`.

    Args:
        observation: The mapping returned by the native ``reset``/``step``.
        observation_space: The space the result must be contained in.

    Returns:
        A new dict whose values satisfy ``observation_space.contains(...)``.
    """
    fitted: dict[str, Any] = {}
    for key, space in observation_space.spaces.items():
        if key not in observation:
            logger.debug("Observation key %r missing from native observation.", key)
            continue
        value = observation[key]
        if key in DISCRETE_OBSERVATION_KEYS:
            fitted[key] = int(value)
            continue
        if key in VARIABLE_LENGTH_OBSERVATION_KEYS:
            fitted[key] = _fit_variable_length(value, space.shape, space.dtype)
            continue
        fitted[key] = np.asarray(value, dtype=space.dtype).reshape(space.shape)
    return fitted
