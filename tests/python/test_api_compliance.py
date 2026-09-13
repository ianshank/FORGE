"""Upstream RL-ecosystem API compliance gates.

These run the *real* compliance suites shipped by Gymnasium and PettingZoo
against FORGE's wrappers. That distinction matters: ``forge_env.utils.check_env``
is a lightweight shape check this repository wrote itself, and passing it says
nothing about whether Stable-Baselines3, CleanRL, RLlib, or a PettingZoo-based
trainer can consume the environment. Only the upstream suites answer that, so
only they can back a compliance claim in the README or release notes.

What each suite enforces, beyond "the tuple has the right length":

``gymnasium.utils.env_checker.check_env``
    The env inherits :class:`gymnasium.Env`; ``reset`` accepts ``seed`` and
    ``options`` as keyword arguments and seeds ``np_random``; two resets with
    the same seed produce equal observations; and every returned observation is
    *contained in* the declared observation space, which means exact dtypes and
    shapes rather than merely plausible values.

``pettingzoo.test.parallel_api_test``
    ``possible_agents`` / ``agents`` bookkeeping, agents never revive once
    retired, per-agent observation/reward/termination/truncation/info dicts, and
    ``observation_space(agent)`` returning the identical object each call so
    per-agent space seeding works.

Both suites are skipped, not failed, when their package is missing, so the
no-native and no-extras CI jobs stay green; the dedicated ``api-compliance`` job
installs both and is where these actually run.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest

#: Cycles driven by the PettingZoo suite. Enough to cover reset, many steps, and
#: the terminal transition, without making the gate slow.
PARALLEL_API_TEST_CYCLES: int = 200

#: Agent count for the multi-agent suite. Two is the smallest count that would
#: expose the "only agent_0's action is applied" class of bug.
COMPLIANCE_AGENT_COUNT: int = 2

#: Deterministic seed so a failure is reproducible from the test name alone.
COMPLIANCE_SEED: int = 42


def _require_native() -> None:
    """Skip when the compiled extension is unavailable."""
    forge_env = pytest.importorskip("forge_env")
    if getattr(forge_env, "ForgeEnv", None) is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")


def _make_gymnasium_env(**kwargs: Any) -> Any:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    return ForgeGymnasiumEnv(config={"world": {"seed": COMPLIANCE_SEED}}, **kwargs)


class TestGymnasiumCompliance:
    """FORGE's single-agent wrapper against Gymnasium's own checker."""

    def test_passes_gymnasium_env_checker(self) -> None:
        """The full upstream checker, including the render-mode checks."""
        pytest.importorskip("gymnasium")
        _require_native()
        from gymnasium.utils.env_checker import check_env

        env = _make_gymnasium_env(render_mode="ascii")
        try:
            check_env(env)
        finally:
            env.close()

    def test_is_a_gymnasium_env_subclass(self) -> None:
        """Inheritance is asserted by the checker and by every gymnasium wrapper."""
        gymnasium = pytest.importorskip("gymnasium")
        _require_native()

        env = _make_gymnasium_env()
        try:
            assert isinstance(env, gymnasium.Env)
        finally:
            env.close()

    def test_observations_are_contained_in_declared_space(self) -> None:
        """`reset` and `step` observations must satisfy `space.contains`."""
        pytest.importorskip("gymnasium")
        _require_native()

        env = _make_gymnasium_env()
        try:
            obs, _info = env.reset(seed=COMPLIANCE_SEED)
            assert env.observation_space.contains(obs), (
                "reset() observation is not contained in the declared observation_space"
            )
            obs, _reward, _term, _trunc, _info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), (
                "step() observation is not contained in the declared observation_space"
            )
        finally:
            env.close()

    def test_unwrapped_follows_the_gymnasium_contract(self) -> None:
        """`unwrapped` yields the base env; the native handle lives on `native`."""
        pytest.importorskip("gymnasium")
        _require_native()

        env = _make_gymnasium_env()
        try:
            assert env.unwrapped is env
            assert env.native is env._env
        finally:
            env.close()


class TestGymnasiumRegistration:
    """`gymnasium.make` support, which is how the ecosystem instantiates envs.

    Stable-Baselines3's `make_vec_env`, CleanRL's `--env-id`, and RLlib all
    construct environments by id, so an unregistered env is awkward to consume
    even when its API is compliant. Registration also gives the env a `spec`,
    which is what lets the upstream checker exercise alternative render modes.
    """

    def test_register_envs_is_idempotent(self) -> None:
        gymnasium = pytest.importorskip("gymnasium")
        from forge_env.registration import FORGE_ENV_ID, is_registered, register_envs

        assert register_envs() == FORGE_ENV_ID
        assert is_registered(FORGE_ENV_ID)
        # A second call must not raise or duplicate the entry.
        assert register_envs() == FORGE_ENV_ID
        assert FORGE_ENV_ID in gymnasium.registry

    def test_registration_is_not_an_import_side_effect(self) -> None:
        """Charter Invariant 1: registration is explicit, never on import.

        Asserted against a subprocess so an earlier test in this session that
        already called ``register_envs()`` cannot mask a regression.
        """
        pytest.importorskip("gymnasium")
        import subprocess
        import sys

        probe = (
            "import gymnasium, forge_env;"
            "from forge_env.registration import FORGE_ENV_ID;"
            "print(FORGE_ENV_ID in gymnasium.registry)"
        )
        python_dir = pathlib.Path(__file__).resolve().parents[2] / "python"
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{python_dir}:{existing_pythonpath}" if existing_pythonpath else str(python_dir)
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
        )
        assert result.stdout.strip() == "False", (
            "importing forge_env registered a Gymnasium id as a side effect; "
            "registration must stay explicit via register_envs()"
        )

    def test_make_produces_a_checker_clean_env(self) -> None:
        """The env built through `gymnasium.make` also passes the checker."""
        gymnasium = pytest.importorskip("gymnasium")
        _require_native()
        from gymnasium.utils.env_checker import check_env

        from forge_env.registration import FORGE_ENV_ID, register_envs

        register_envs()
        env = gymnasium.make(FORGE_ENV_ID)
        try:
            assert env.spec is not None
            check_env(env.unwrapped)
        finally:
            env.close()


class TestPettingZooCompliance:
    """FORGE's multi-agent wrapper against PettingZoo's own Parallel API suite."""

    def test_passes_parallel_api_test(self) -> None:
        """The full upstream Parallel API suite."""
        pytest.importorskip("pettingzoo")
        _require_native()
        from pettingzoo.test import parallel_api_test

        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=COMPLIANCE_AGENT_COUNT)
        try:
            parallel_api_test(env, num_cycles=PARALLEL_API_TEST_CYCLES)
        finally:
            env.close()

    def test_is_a_parallel_env_subclass(self) -> None:
        """Inheritance is what PettingZoo's conversion wrappers dispatch on."""
        pettingzoo = pytest.importorskip("pettingzoo")
        _require_native()

        env = ForgeParallelEnvFactory()
        try:
            assert isinstance(env, pettingzoo.ParallelEnv)
        finally:
            env.close()

    def test_every_agent_action_reaches_the_simulation(self) -> None:
        """Each agent's own non-noop action must change that agent from a noop control."""
        pytest.importorskip("pettingzoo")
        _require_native()

        env = ForgeParallelEnvFactory()
        try:
            agents = list(env.agents)
            assert len(agents) == COMPLIANCE_AGENT_COUNT

            def run_once(actions: dict[str, int]) -> dict[str, Any]:
                env.reset(seed=COMPLIANCE_SEED)
                observations, _rewards, _term, _trunc, _infos = env.step(actions)
                return observations

            def observations_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
                import numpy as np

                return all(np.array_equal(np.asarray(left[k]), np.asarray(right[k])) for k in left)

            noop_actions = dict.fromkeys(agents, _NOOP_ACTION)
            baseline = run_once(noop_actions)
            action_space_n = int(env.action_space(agents[0]).n)

            for target_agent in agents:
                found_in_bounds = False
                for candidate in range(1, action_space_n):
                    actions = dict(noop_actions)
                    actions[target_agent] = candidate
                    shifted = run_once(actions)
                    if not observations_equal(shifted[target_agent], baseline[target_agent]):
                        found_in_bounds = True
                        break
                assert found_in_bounds, (
                    f"{target_agent} action never changed its own observation from noop; "
                    "this indicates per-agent actions are not reaching the simulation"
                )
        finally:
            env.close()

    def test_spaces_are_returned_by_identity(self) -> None:
        """`parallel_api_test` requires the same object, not an equal copy."""
        pytest.importorskip("pettingzoo")
        _require_native()

        env = ForgeParallelEnvFactory()
        try:
            for agent in env.possible_agents:
                assert env.observation_space(agent) is env.observation_space(agent)
                assert env.action_space(agent) is env.action_space(agent)
        finally:
            env.close()


#: ``Action::Noop``; see ``crates/forge-python/src/env.rs::test_action_noop_is_zero``.
_NOOP_ACTION: int = 0


def ForgeParallelEnvFactory() -> Any:
    """Construct the compliance-sized parallel env."""
    from forge_env.pettingzoo_env import ForgeParallelEnv

    return ForgeParallelEnv(n_agents=COMPLIANCE_AGENT_COUNT)
