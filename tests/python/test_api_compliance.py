"""Real Gymnasium env_checker and PettingZoo parallel_api_test gates."""

from __future__ import annotations

import pytest


def _skip_if_no_native() -> None:
    """Skip when the forge_env native extension is not built."""
    from forge_env import gymnasium_env

    if gymnasium_env._NativeEnv is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")


def test_gymnasium_env_checker() -> None:
    """``gymnasium.utils.env_checker.check_env`` must accept ForgeGymnasiumEnv."""
    pytest.importorskip("gymnasium")
    _skip_if_no_native()
    from gymnasium.utils.env_checker import check_env

    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    check_env(ForgeGymnasiumEnv(), skip_render_check=True)


def test_pettingzoo_parallel_api() -> None:
    """``pettingzoo.test.parallel_api_test`` must accept ForgeParallelEnv."""
    pytest.importorskip("pettingzoo")
    _skip_if_no_native()
    from pettingzoo.test import parallel_api_test

    from forge_env.pettingzoo_env import ForgeParallelEnv

    parallel_api_test(ForgeParallelEnv(n_agents=2), num_cycles=200)
