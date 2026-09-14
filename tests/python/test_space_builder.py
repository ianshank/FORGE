"""Unit tests for the shared space builder and observation fitter.

:mod:`forge_env.space_builder` is the single definition both public wrappers
depend on, so its edge cases are worth testing directly rather than only through
whichever wrapper happens to exercise them:

* the **descriptor fallbacks**, which keep older native builds and the flat-key
  test doubles working, and which no realistic end-to-end test reaches because
  the real descriptor always populates every key;
* the **variable-length fitting**, where a short message buffer must pad and an
  over-long one must truncate — the padding path runs constantly, the truncation
  path essentially never, and it is the one that would silently corrupt an
  observation if it were wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

np = pytest.importorskip("numpy")
spaces = pytest.importorskip("gymnasium.spaces")

from forge_env.space_builder import (  # noqa: E402  (after importorskip)
    DEFAULT_ACTION_COUNT,
    DEFAULT_CARRY_CAPACITY,
    DEFAULT_GRID_CHANNELS,
    DEFAULT_NUM_DAY_PHASES,
    DEFAULT_VIEW_SIDE,
    INVENTORY_FIELDS_PER_SLOT,
    MESSAGE_PAD_VALUE,
    POSITION_DIMENSIONS,
    build_action_space,
    build_observation_space,
    fit_observation,
)


class TestBuildsFromTheDescriptor:
    """Shapes and bounds come from the native descriptor, not from literals."""

    def test_nested_descriptor_entries_drive_the_shapes(self) -> None:
        descriptor: dict[str, Any] = {
            "grid_view": {"shape": (3, 3, 2), "low": 1, "high": 2, "dtype": "uint16"},
            "inventory": {"shape": (4, 2), "low": 2, "high": 3, "dtype": "uint8"},
            "health": {"low": -1.0, "high": 2.0, "dtype": "float64"},
            "stamina": {"low": -2.0, "high": 3.0, "dtype": "float64"},
            "position": {"shape": (2,), "low": 4, "high": 5, "dtype": "uint8"},
            "messages": {"shape": (5,), "low": 6, "high": 9, "dtype": "uint8"},
            "day_phase": {"high": 7},
        }
        space = build_observation_space(descriptor)

        assert space.spaces["grid_view"].shape == (3, 3, 2)
        assert space.spaces["grid_view"].dtype == np.uint16
        assert int(space.spaces["grid_view"].low.min()) == 1
        assert int(space.spaces["grid_view"].high.max()) == 2
        assert space.spaces["inventory"].shape == (4, 2)
        assert space.spaces["inventory"].dtype == np.uint8
        assert int(space.spaces["inventory"].low.min()) == 2
        assert int(space.spaces["inventory"].high.max()) == 3
        assert float(space.spaces["health"].low) == -1.0
        assert float(space.spaces["health"].high) == 2.0
        assert space.spaces["health"].dtype == np.float64
        assert float(space.spaces["stamina"].low) == -2.0
        assert float(space.spaces["stamina"].high) == 3.0
        assert space.spaces["stamina"].dtype == np.float64
        assert int(space.spaces["position"].low.min()) == 4
        assert int(space.spaces["position"].high.max()) == 5
        assert space.spaces["position"].dtype == np.uint8
        assert space.spaces["messages"].shape == (5,)
        assert int(space.spaces["messages"].low.min()) == 6
        assert int(space.spaces["messages"].high.max()) == 9
        assert space.spaces["messages"].dtype == np.uint8
        # The descriptor publishes an inclusive bound; Discrete takes a count.
        assert space.spaces["day_phase"].n == 8

    def test_flat_convenience_keys_drive_the_shapes(self) -> None:
        """The native descriptor also publishes flat dimension keys."""
        descriptor: dict[str, Any] = {
            "grid_view_height": 5,
            "grid_view_width": 7,
            "grid_view_channels": 3,
            "inventory_capacity": 6,
        }
        space = build_observation_space(descriptor)

        assert space.spaces["grid_view"].shape == (5, 7, 3)
        assert space.spaces["inventory"].shape == (6, INVENTORY_FIELDS_PER_SLOT)

    def test_empty_descriptor_falls_back_to_the_declared_defaults(self) -> None:
        """An older native build missing every key must still yield a space."""
        space = build_observation_space({})

        assert space.spaces["grid_view"].shape == (
            DEFAULT_VIEW_SIDE,
            DEFAULT_VIEW_SIDE,
            DEFAULT_GRID_CHANNELS,
        )
        assert space.spaces["inventory"].shape == (
            DEFAULT_CARRY_CAPACITY,
            INVENTORY_FIELDS_PER_SLOT,
        )
        assert space.spaces["position"].shape == (POSITION_DIMENSIONS,)
        assert space.spaces["day_phase"].n == DEFAULT_NUM_DAY_PHASES
        assert build_action_space({}).n == DEFAULT_ACTION_COUNT

    def test_non_mapping_descriptor_entry_is_ignored(self) -> None:
        """A malformed entry falls back rather than raising at construction."""
        space = build_observation_space({"grid_view": "not-a-mapping"})
        assert space.spaces["grid_view"].shape == (
            DEFAULT_VIEW_SIDE,
            DEFAULT_VIEW_SIDE,
            DEFAULT_GRID_CHANNELS,
        )

    def test_action_space_uses_the_descriptor_count(self) -> None:
        assert build_action_space({"type": "Discrete", "n": 17}).n == 17

    def test_invalid_dtype_falls_back_to_default(self) -> None:
        space = build_observation_space({"grid_view": {"shape": (2, 2, 3), "dtype": "not-a-dtype"}})
        assert space.spaces["grid_view"].dtype == np.uint8


class TestFitObservation:
    """Every fitted value must be contained in its declared space."""

    @staticmethod
    def _space() -> Any:
        return build_observation_space(
            {
                "grid_view": {"shape": (2, 2, 3)},
                "inventory": {"shape": (2, 2)},
                "position": {"shape": (2,)},
                "messages": {"shape": (4,), "high": 15},
                "day_phase": {"high": 3},
            }
        )

    def test_python_scalars_and_tuples_become_contained_arrays(self) -> None:
        space = self._space()
        raw = {
            "grid_view": np.zeros((2, 2, 3), dtype=np.uint8),
            "inventory": np.zeros((2, 2), dtype=np.uint16),
            "health": 0.5,
            "stamina": 1.0,
            "position": (3, 4),
            "messages": [],
            "day_phase": 2,
        }
        fitted = fit_observation(raw, space)

        assert space.contains(fitted), "fitted observation is not in its declared space"
        assert fitted["position"].dtype == np.uint16
        assert fitted["health"].dtype == np.float32
        assert isinstance(fitted["day_phase"], int)

    def test_short_message_buffer_is_padded(self) -> None:
        space = self._space()
        fitted = fit_observation({"messages": [7, 8]}, space)

        assert fitted["messages"].shape == (4,)
        assert list(fitted["messages"]) == [7, 8, MESSAGE_PAD_VALUE, MESSAGE_PAD_VALUE]

    def test_over_long_message_buffer_is_truncated(self) -> None:
        """The rare path: a buffer longer than the declared space."""
        space = self._space()
        fitted = fit_observation({"messages": [1, 2, 3, 4, 5, 6]}, space)

        assert fitted["messages"].shape == (4,)
        assert list(fitted["messages"]) == [1, 2, 3, 4]

    def test_exact_length_message_buffer_is_unchanged(self) -> None:
        space = self._space()
        fitted = fit_observation({"messages": [1, 2, 3, 4]}, space)
        assert list(fitted["messages"]) == [1, 2, 3, 4]

    def test_non_sequence_variable_length_value_is_accepted(self) -> None:
        """A native build returning an array rather than a list still fits."""
        space = self._space()
        fitted = fit_observation({"messages": np.array([5, 6], dtype=np.uint16)}, space)
        assert list(fitted["messages"]) == [5, 6, MESSAGE_PAD_VALUE, MESSAGE_PAD_VALUE]

    def test_unknown_keys_are_dropped(self) -> None:
        """Fitted observations must match the declared Dict-space key set exactly."""
        space = self._space()
        fitted = fit_observation({"a_brand_new_component": object(), "health": 0.5}, space)
        assert "a_brand_new_component" not in fitted
        assert "health" in fitted


class TestGymnasiumIsRequired:
    """Absent gymnasium must produce a directive error, not an AttributeError."""

    def test_build_observation_space_reports_the_missing_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from forge_env import space_builder

        monkeypatch.setattr(space_builder, "HAS_GYMNASIUM", False)
        with pytest.raises(ImportError, match="gymnasium is required"):
            space_builder.build_observation_space({})

    def test_build_action_space_reports_the_missing_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from forge_env import space_builder

        monkeypatch.setattr(space_builder, "HAS_GYMNASIUM", False)
        with pytest.raises(ImportError, match="gymnasium is required"):
            space_builder.build_action_space({})
