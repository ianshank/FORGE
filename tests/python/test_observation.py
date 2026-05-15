"""Tests for forge.utils.observation module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from forge.utils.observation import compute_obs_dim, flatten_obs

logger = logging.getLogger(__name__)


class TestFlattenObs:
    """Tests for the flatten_obs function."""

    def test_flatten_obs_simple_dict(self) -> None:
        """Basic dict with array values is flattened and sorted by key."""
        obs = {"position": np.array([1.0, 2.0]), "health": np.array([0.5])}
        result = flatten_obs(obs)

        # Sorted keys: health, position -> [0.5, 1.0, 2.0]
        expected = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_flatten_obs_scalar_values(self) -> None:
        """Scalar values are treated as single-element arrays."""
        obs = {"a": 1.0, "b": 2.0}
        result = flatten_obs(obs)

        expected = np.array([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_flatten_obs_empty_dict(self) -> None:
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot flatten empty"):
            flatten_obs({})

    def test_flatten_obs_nested(self) -> None:
        """2-D arrays are ravelled during flattening."""
        obs = {"grid": np.array([[1, 2], [3, 4]])}
        result = flatten_obs(obs)

        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_flatten_obs_dtype(self) -> None:
        """Output is always float32."""
        obs = {"x": np.array([1, 2, 3], dtype=np.int64)}
        result = flatten_obs(obs)

        assert result.dtype == np.float32

    def test_flatten_obs_sorted_keys(self) -> None:
        """Keys are sorted alphabetically for deterministic ordering."""
        obs = {"z": np.array([3.0]), "a": np.array([1.0]), "m": np.array([2.0])}
        result = flatten_obs(obs)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestComputeObsDim:
    """Tests for the compute_obs_dim function."""

    def test_compute_obs_dim_with_real_env(self) -> None:
        """Computes correct dimensionality from a real FORGE environment."""
        pytest.importorskip("gymnasium")
        try:
            from forge_env import gymnasium_env
            from forge_env.gymnasium_env import ForgeGymnasiumEnv
        except ImportError as exc:
            pytest.skip(f"forge_env native extension not available: {exc}")

        if gymnasium_env._NativeEnv is None:
            pytest.skip("forge_env running in pure-Python mode (no native backend)")

        env = ForgeGymnasiumEnv()
        dim = compute_obs_dim(env)
        assert isinstance(dim, int)
        assert dim > 0
        env.close()

    def test_compute_obs_dim_reset_failure(self) -> None:
        """RuntimeError is raised when env.reset() fails."""
        mock_env = MagicMock()
        mock_env.reset.side_effect = RuntimeError("env broken")

        with pytest.raises(RuntimeError, match="Failed to probe"):
            compute_obs_dim(mock_env)
