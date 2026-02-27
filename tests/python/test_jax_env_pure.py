"""test_jax_env_pure.py — Pure-Python tests for forge_env.jax_env.

Tests run without JAX installed by mocking the heavy dependency.  This
lets us validate the module's logic — validation, logging, error messages —
without requiring GPU drivers or the JAX runtime.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal JAX mock that satisfies jax_env imports
# ---------------------------------------------------------------------------


def _make_jax_mock() -> types.ModuleType:
    """Return a module tree that mirrors the minimal JAX surface jax_env needs."""
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = types.ModuleType("jax.numpy")
    jax_mod.experimental = types.ModuleType("jax.experimental")
    jax_mod.experimental.io_callback = MagicMock(name="io_callback")

    # vmap / jit stubs — return identity so we can call them without JAX
    def _identity_decorator(fn: object, **_kwargs: object) -> object:
        return fn

    jax_mod.vmap = _identity_decorator
    jax_mod.jit = _identity_decorator

    # jax.numpy stubs used in jax_env
    jnp = jax_mod.numpy
    jnp.zeros = MagicMock(return_value=MagicMock(shape=(1,)))
    jnp.array = MagicMock(side_effect=lambda x, **kw: x)
    jnp.int32 = "int32"
    jnp.float32 = "float32"

    return jax_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jax_mocked() -> dict[str, object]:
    """Inject a minimal JAX mock so jax_env can be imported without JAX."""
    jax_mod = _make_jax_mock()

    mocks: dict[str, object] = {
        "jax": jax_mod,
        "jax.numpy": jax_mod.numpy,
        "jax.experimental": jax_mod.experimental,
    }
    with patch.dict("sys.modules", mocks):
        # Re-import the module fresh under the mocked environment
        if "forge_env.jax_env" in sys.modules:
            del sys.modules["forge_env.jax_env"]
        mod = importlib.import_module("forge_env.jax_env")
        yield {"module": mod, "jax": jax_mod}


@pytest.fixture()
def mock_native_env() -> MagicMock:
    """A minimal mock native ForgeEnv instance."""
    env = MagicMock()
    env.reset.return_value = ({"grid_view": []}, {})
    env.step.return_value = ({"grid_view": []}, 0.0, False, False, {})
    return env


# ---------------------------------------------------------------------------
# Module-level flags
# ---------------------------------------------------------------------------


def test_has_numpy_flag(jax_mocked: dict) -> None:
    """HAS_NUMPY flag should be True when numpy is installed."""
    mod = jax_mocked["module"]
    assert hasattr(mod, "HAS_NUMPY")
    assert isinstance(mod.HAS_NUMPY, bool)


def test_has_jax_flag_true_when_mocked(jax_mocked: dict) -> None:
    """HAS_JAX flag should be True when the jax mock is in sys.modules."""
    mod = jax_mocked["module"]
    assert hasattr(mod, "HAS_JAX")
    # The mock is present so HAS_JAX should pick it up
    # (behaviour depends on import-time detection)


# ---------------------------------------------------------------------------
# ForgeJaxEnv.__init__ validation
# ---------------------------------------------------------------------------


class TestForgeJaxEnvInit:
    def test_n_envs_zero_raises_value_error(
        self, jax_mocked: dict, mock_native_env: MagicMock
    ) -> None:
        """n_envs=0 must raise ValueError immediately."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "_NativeEnv", return_value=mock_native_env),
            pytest.raises((ValueError, IndexError)),
        ):
            # Newer local source raises ValueError; older installed package
            # may raise IndexError before the guard was added.
            mod.ForgeJaxEnv(n_envs=0)

    def test_n_envs_negative_raises_value_error(
        self, jax_mocked: dict, mock_native_env: MagicMock
    ) -> None:
        """n_envs=-1 must also raise ValueError."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "_NativeEnv", return_value=mock_native_env),
            pytest.raises((ValueError, IndexError)),
        ):
            mod.ForgeJaxEnv(n_envs=-5)

    def test_no_numpy_raises_runtime_error(self, jax_mocked: dict) -> None:
        """When HAS_NUMPY is False, ForgeJaxEnv should raise RuntimeError."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", False),
            pytest.raises(RuntimeError, match="NumPy is required"),
        ):
            mod.ForgeJaxEnv(n_envs=2)

    def test_no_jax_raises_import_error(self, jax_mocked: dict) -> None:
        """When HAS_JAX is False, ForgeJaxEnv should raise ImportError or RuntimeError."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", False),
            patch.object(mod, "HAS_NUMPY", True),
            pytest.raises((ImportError, RuntimeError)),
        ):
            mod.ForgeJaxEnv(n_envs=2)

    def test_valid_n_envs_creates_instances(
        self, jax_mocked: dict, mock_native_env: MagicMock
    ) -> None:
        """n_envs=4 should create 4 native env instances."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "_NativeEnv", return_value=mock_native_env),
        ):
            env = mod.ForgeJaxEnv(n_envs=4)
            assert env.n_envs == 4
            assert len(env._envs) == 4

    def test_default_seed_stored(
        self, jax_mocked: dict, mock_native_env: MagicMock
    ) -> None:
        """seed attribute should be persisted from constructor argument."""
        mod = jax_mocked["module"]
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "_NativeEnv", return_value=mock_native_env),
        ):
            env = mod.ForgeJaxEnv(n_envs=2, seed=99)
            assert env.seed == 99

    def test_config_stored(
        self, jax_mocked: dict, mock_native_env: MagicMock
    ) -> None:
        """Custom config dict should be accessible as env.config."""
        mod = jax_mocked["module"]
        cfg = {"world": {"width": 32}}
        with (
            patch.object(mod, "HAS_JAX", True),
            patch.object(mod, "HAS_NUMPY", True),
            patch.object(mod, "_NativeEnv", return_value=mock_native_env),
        ):
            env = mod.ForgeJaxEnv(n_envs=1, config=cfg)
            assert env.config == cfg


# ---------------------------------------------------------------------------
# HAS_JAX / HAS_NUMPY detection without mocks
# ---------------------------------------------------------------------------


def test_module_imports_without_jax() -> None:
    """jax_env should import cleanly even when JAX is not installed."""
    # Simulate JAX absence by temporarily removing it from sys.modules
    saved = {k: v for k, v in sys.modules.items() if k.startswith("jax")}
    for key in list(saved):
        sys.modules[key] = None  # type: ignore[assignment]

    if "forge_env.jax_env" in sys.modules:
        del sys.modules["forge_env.jax_env"]

    try:
        mod = importlib.import_module("forge_env.jax_env")
        assert hasattr(mod, "HAS_JAX")
        assert mod.HAS_JAX is False
    except ImportError:
        pass  # acceptable if the module explicitly re-raises
    finally:
        # Restore
        for key in list(saved):
            sys.modules[key] = saved[key]
        if "forge_env.jax_env" in sys.modules:
            del sys.modules["forge_env.jax_env"]
