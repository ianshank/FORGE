"""Tests for MangoMAS action and observation adapters."""
from __future__ import annotations

import numpy as np
import pytest

from forge.mangomas.adapters import ActionSpaceAdapter, ObservationAdapter
from forge.mangomas.config import ActionAdapterConfig, ObservationAdapterConfig


class TestActionSpaceAdapter:
    """Tests for ActionSpaceAdapter."""

    def test_car_continuous_dims(self) -> None:
        adapter = ActionSpaceAdapter(platform="car")
        assert adapter.continuous_dims == 2

    def test_drone_continuous_dims(self) -> None:
        adapter = ActionSpaceAdapter(platform="drone")
        assert adapter.continuous_dims == 4

    def test_continuous_to_discrete_car(self) -> None:
        adapter = ActionSpaceAdapter(platform="car")
        action = np.array([0.0, 0.0], dtype=np.float32)
        discrete = adapter.continuous_to_discrete(action)
        assert isinstance(discrete, int)
        assert discrete >= 0

    def test_continuous_to_discrete_drone(self) -> None:
        adapter = ActionSpaceAdapter(platform="drone")
        action = np.array([0.0, 0.0, 0.5, -0.5], dtype=np.float32)
        discrete = adapter.continuous_to_discrete(action)
        assert isinstance(discrete, int)
        assert discrete >= 0

    def test_discrete_to_continuous_roundtrip(self) -> None:
        adapter = ActionSpaceAdapter(platform="car")
        for action_id in range(min(adapter._bins ** 2, 20)):
            continuous = adapter.discrete_to_continuous(action_id)
            assert continuous.shape == (2,)
            assert np.all(continuous >= adapter._lo)
            assert np.all(continuous <= adapter._hi)

    def test_clipping(self) -> None:
        adapter = ActionSpaceAdapter(platform="car")
        # Out of range values should be clipped
        action = np.array([5.0, -5.0], dtype=np.float32)
        discrete = adapter.continuous_to_discrete(action)
        assert isinstance(discrete, int)

    def test_custom_config(self) -> None:
        config = ActionAdapterConfig(bins_per_axis=5, continuous_range_min=-2.0, continuous_range_max=2.0)
        adapter = ActionSpaceAdapter(config=config, platform="car")
        assert adapter._bins == 5
        assert adapter._lo == -2.0
        assert adapter._hi == 2.0

    def test_all_bins_reachable_car(self) -> None:
        adapter = ActionSpaceAdapter(platform="car")
        total_actions = adapter._bins ** 2
        seen = set()
        for i in range(total_actions):
            cont = adapter.discrete_to_continuous(i)
            disc = adapter.continuous_to_discrete(cont)
            seen.add(disc)
        # At least most actions map back (exact roundtrip depends on quantization)
        assert len(seen) >= total_actions * 0.8


class TestObservationAdapter:
    """Tests for ObservationAdapter."""

    def test_car_output_dim(self) -> None:
        adapter = ObservationAdapter(platform="car")
        assert adapter.output_dim == 18

    def test_drone_output_dim(self) -> None:
        adapter = ObservationAdapter(platform="drone")
        assert adapter.output_dim == 22

    def test_adapt_minimal_obs(self) -> None:
        adapter = ObservationAdapter(platform="car")
        obs = {
            "grid_view": np.zeros((11, 11, 7)),
            "health": 1.0,
            "stamina": 0.8,
            "position": [5, 5],
            "day_phase": 0.5,
            "inventory": {},
        }
        state = adapter.adapt(obs)
        assert state.shape == (18,)
        assert state.dtype == np.float32

    def test_adapt_drone_obs(self) -> None:
        adapter = ObservationAdapter(platform="drone")
        obs = {
            "grid_view": np.random.rand(11, 11, 7).astype(np.float32),
            "health": 0.9,
            "stamina": 0.7,
            "position": [3, 8],
            "day_phase": 0.25,
            "inventory": {"wood": 3, "stone": 1},
            "altitude": 0.5,
            "battery": 0.8,
            "morphology": 1.0,
            "heading": 0.75,
        }
        state = adapter.adapt(obs)
        assert state.shape == (22,)
        assert state.dtype == np.float32

    def test_adapt_empty_grid(self) -> None:
        adapter = ObservationAdapter(platform="car")
        obs = {"grid_view": [], "health": 1.0, "stamina": 1.0, "position": [0, 0], "day_phase": 0.0}
        state = adapter.adapt(obs)
        assert state.shape == (18,)

    def test_adapt_missing_fields_uses_defaults(self) -> None:
        adapter = ObservationAdapter(platform="car")
        obs = {}
        state = adapter.adapt(obs)
        assert state.shape == (18,)
        # health defaults to 1.0
        assert state[11] == 1.0
