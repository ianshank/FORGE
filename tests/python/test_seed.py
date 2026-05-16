"""Tests for forge.utils.seed module."""

from __future__ import annotations

import logging
import random

import numpy as np

from forge.utils.seed import derive_seed, set_all_seeds

logger = logging.getLogger(__name__)


class TestSetAllSeeds:
    """Tests for the set_all_seeds function."""

    def test_set_all_seeds_deterministic(self) -> None:
        """Same seed produces the same random sequence."""
        set_all_seeds(42)
        vals_a = [random.random() for _ in range(5)]

        set_all_seeds(42)
        vals_b = [random.random() for _ in range(5)]

        assert vals_a == vals_b

    def test_set_all_seeds_numpy(self) -> None:
        """NumPy random is seeded deterministically."""
        set_all_seeds(123)
        arr_a = np.random.rand(5)

        set_all_seeds(123)
        arr_b = np.random.rand(5)

        np.testing.assert_array_equal(arr_a, arr_b)

    def test_set_all_seeds_different_seeds(self) -> None:
        """Different seeds produce different random sequences."""
        set_all_seeds(1)
        val_a = random.random()

        set_all_seeds(2)
        val_b = random.random()

        assert val_a != val_b


class TestDeriveSeed:
    """Tests for the derive_seed function."""

    def test_derive_seed_deterministic(self) -> None:
        """Same inputs always produce the same output."""
        s1 = derive_seed(42, "env")
        s2 = derive_seed(42, "env")

        assert s1 == s2

    def test_derive_seed_different_components(self) -> None:
        """Different component names produce different seeds."""
        s1 = derive_seed(42, "env")
        s2 = derive_seed(42, "agent")

        assert s1 != s2

    def test_derive_seed_different_bases(self) -> None:
        """Different base seeds produce different derived seeds."""
        s1 = derive_seed(1, "env")
        s2 = derive_seed(2, "env")

        assert s1 != s2

    def test_derive_seed_within_range(self) -> None:
        """Derived seed is within valid range [0, 2**32 - 1]."""
        seed = derive_seed(999, "component")

        assert 0 <= seed <= 2**32 - 1
