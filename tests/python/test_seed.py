"""Tests for forge.utils.seed module."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest

from forge.utils.seed import (
    CUBLAS_WORKSPACE_CONFIG_ENV_VAR,
    DEFAULT_CUBLAS_WORKSPACE_CONFIG,
    DEFAULT_DETERMINISTIC,
    PYTHONHASHSEED_ENV_VAR,
    derive_seed,
    seed_torch,
    set_all_seeds,
)

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


class TestTorchSeeding:
    """Torch must be seeded on CPU-only hosts, not just under CUDA.

    Regression guard: ``set_all_seeds`` / ``seed_everything`` previously
    called only ``torch.cuda.manual_seed_all`` inside an
    ``if torch.cuda.is_available()`` guard, which left torch's CPU
    generator entirely unseeded on every CI runner.
    """

    def test_set_all_seeds_makes_torch_reproducible(self) -> None:
        """Same seed => identical torch tensors (CPU generator)."""
        torch = pytest.importorskip("torch")

        set_all_seeds(4242)
        a = torch.rand(8)

        set_all_seeds(4242)
        b = torch.rand(8)

        assert torch.equal(a, b), "torch CPU generator was not seeded by set_all_seeds"

    def test_different_seeds_give_different_torch_tensors(self) -> None:
        """Sanity check that the assertion above is not vacuously true."""
        torch = pytest.importorskip("torch")

        set_all_seeds(1)
        a = torch.rand(8)
        set_all_seeds(2)
        b = torch.rand(8)

        assert not torch.equal(a, b)

    def test_seed_torch_reports_whether_torch_was_seeded(self) -> None:
        """``seed_torch`` returns True when torch is importable."""
        pytest.importorskip("torch")
        assert seed_torch(7) is True

    def test_seed_torch_without_torch_returns_false(self) -> None:
        """Torch is optional: absence is reported, never raised."""
        import sys
        from unittest.mock import patch

        # `sys.modules["torch"] = None` makes `import torch` raise ImportError.
        with patch.dict(sys.modules, {"torch": None}):
            assert seed_torch(7) is False


class TestStrictDeterminism:
    """The strict-determinism switch is opt-in and off by default."""

    def test_default_is_off(self) -> None:
        assert DEFAULT_DETERMINISTIC is False

    def test_off_by_default_leaves_torch_flags_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain ``set_all_seeds`` must not flip global torch state."""
        torch = pytest.importorskip("torch")
        monkeypatch.delenv(PYTHONHASHSEED_ENV_VAR, raising=False)

        torch.use_deterministic_algorithms(False)
        set_all_seeds(11)

        assert torch.are_deterministic_algorithms_enabled() is False
        assert PYTHONHASHSEED_ENV_VAR not in os.environ

    def test_opt_in_enables_torch_determinism_and_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.delenv(PYTHONHASHSEED_ENV_VAR, raising=False)
        monkeypatch.delenv(CUBLAS_WORKSPACE_CONFIG_ENV_VAR, raising=False)
        # Restore global torch state for every later test in the session.
        monkeypatch.setattr(
            torch,
            "use_deterministic_algorithms",
            torch.use_deterministic_algorithms,
        )

        try:
            set_all_seeds(11, deterministic=True)

            assert torch.are_deterministic_algorithms_enabled() is True
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
            assert os.environ[PYTHONHASHSEED_ENV_VAR] == "11"
            assert (
                os.environ[CUBLAS_WORKSPACE_CONFIG_ENV_VAR] == DEFAULT_CUBLAS_WORKSPACE_CONFIG
            )
        finally:
            torch.use_deterministic_algorithms(False)

    def test_existing_cublas_config_is_not_overridden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator's own workspace config wins over our default."""
        torch = pytest.importorskip("torch")
        monkeypatch.setenv(CUBLAS_WORKSPACE_CONFIG_ENV_VAR, ":16:8")

        try:
            set_all_seeds(11, deterministic=True)
            assert os.environ[CUBLAS_WORKSPACE_CONFIG_ENV_VAR] == ":16:8"
        finally:
            torch.use_deterministic_algorithms(False)


def test_fallback_constants_match_canonical_definitions() -> None:
    """``forge_env.utils`` duplicates these for its no-``forge`` fallback path.

    They cannot be imported there (that branch exists precisely because
    ``forge.utils.seed`` is unavailable), so pin them together here.
    """
    from forge_env import utils as fe_utils

    assert fe_utils.DEFAULT_DETERMINISTIC == DEFAULT_DETERMINISTIC
    assert fe_utils.PYTHONHASHSEED_ENV_VAR == PYTHONHASHSEED_ENV_VAR
    assert fe_utils.CUBLAS_WORKSPACE_CONFIG_ENV_VAR == CUBLAS_WORKSPACE_CONFIG_ENV_VAR
    assert fe_utils.DEFAULT_CUBLAS_WORKSPACE_CONFIG == DEFAULT_CUBLAS_WORKSPACE_CONFIG


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
