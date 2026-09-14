"""Byte-level determinism gate across the Python boundary.

Determinism is FORGE's defining guarantee (``docs/CHARTER.md`` Invariant 6:
*same seed + same actions = identical state*) and it is already defended inside
Rust by a property test over serialized world state and by golden state hashes.
What was missing is the same assertion *through the PyO3 boundary* — the surface
every training run actually consumes. A determinism bug introduced in
observation conversion, buffer reuse, or space fitting would not move a single
Rust hash while corrupting every Python rollout.

So this drives two independently constructed environments through an identical
action sequence and asserts their observations are byte-identical and their
rewards exactly equal — not "close", not "statistically similar". The comparison
is on ``ndarray.tobytes()``, which is why a float that differs in its last
mantissa bit fails here rather than being rounded away.

Depth is configurable rather than fixed: ``--determinism-steps N`` (or
``FORGE_DETERMINISM_STEPS``) selects the step count, so PR CI runs a fast gate
and a release soak runs the same code far longer:

    pytest tests/python/test_determinism.py --determinism-steps 1000000 --no-cov
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger(__name__)

#: Seed for both the simulation and the action sequence, so a failure
#: reproduces from the test name alone.
DETERMINISM_SEED: int = 20260912

#: Steps between progress logs on a long soak, so a million-step run shows
#: liveness instead of appearing hung.
PROGRESS_LOG_INTERVAL: int = 50_000


def _skip_unless_runnable() -> None:
    """Skip when gymnasium or the native extension is unavailable."""
    pytest.importorskip("gymnasium")
    forge_env = pytest.importorskip("forge_env")
    if getattr(forge_env, "ForgeEnv", None) is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")


def _make_env() -> Any:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    return ForgeGymnasiumEnv(config={"world": {"seed": DETERMINISM_SEED}})


def _observation_fingerprint(observation: Mapping[str, Any]) -> bytes:
    """Reduce an observation to bytes for exact comparison.

    Keys are sorted so the fingerprint does not depend on dict ordering, and
    each key name is included so a value moving between keys cannot cancel out.
    """
    import numpy as np

    chunks: list[bytes] = []
    for key in sorted(observation):
        value = observation[key]
        chunks.append(key.encode("utf-8"))
        chunks.append(np.asarray(value).tobytes())
    return b"".join(chunks)


def _action_sequence(action_count: int, steps: int) -> Iterator[int]:
    """Yield a reproducible action sequence.

    Drawn from a dedicated NumPy generator rather than from ``action_space``:
    both environments must see the *same* actions, and sampling from each env's
    own space would make the test depend on the very seeding it is verifying.
    """
    import numpy as np

    rng = np.random.default_rng(DETERMINISM_SEED)
    for action in rng.integers(0, action_count, size=steps):
        yield int(action)


class TestObservationAndRewardDeterminism:
    """Two identically seeded environments must not diverge by a single byte."""

    def test_identical_seed_and_actions_produce_identical_trajectories(
        self, determinism_steps: int
    ) -> None:
        _skip_unless_runnable()

        left, right = _make_env(), _make_env()
        try:
            left_obs, _ = left.reset(seed=DETERMINISM_SEED)
            right_obs, _ = right.reset(seed=DETERMINISM_SEED)
            assert _observation_fingerprint(left_obs) == _observation_fingerprint(
                right_obs
            ), "reset() observations diverged before a single step was taken"

            episodes = 0
            for index, action in enumerate(
                _action_sequence(int(left.action_space.n), determinism_steps)
            ):
                left_result = left.step(action)
                right_result = right.step(action)

                left_obs, left_reward, left_term, left_trunc, _ = left_result
                right_obs, right_reward, right_term, right_trunc, _ = right_result

                assert _observation_fingerprint(left_obs) == _observation_fingerprint(
                    right_obs
                ), f"observations diverged at step {index} (action {action})"
                assert left_reward == right_reward, (
                    f"rewards diverged at step {index}: "
                    f"{left_reward!r} != {right_reward!r} (action {action})"
                )
                assert left_term == right_term, f"terminated diverged at step {index}"
                assert left_trunc == right_trunc, f"truncated diverged at step {index}"

                if left_term or left_trunc:
                    episodes += 1
                    # Same derived seed on both sides keeps the run reproducible
                    # across episode boundaries rather than only within one.
                    episode_seed = DETERMINISM_SEED + episodes
                    left_obs, _ = left.reset(seed=episode_seed)
                    right_obs, _ = right.reset(seed=episode_seed)
                    assert _observation_fingerprint(left_obs) == _observation_fingerprint(
                        right_obs
                    ), f"observations diverged after reset at step {index}"

                if index and index % PROGRESS_LOG_INTERVAL == 0:
                    logger.info(
                        "Determinism gate: %d/%d steps, %d episode(s), still identical.",
                        index,
                        determinism_steps,
                        episodes,
                    )

            logger.info(
                "Determinism gate passed: %d steps, %d episode boundary/-ies, 0 diverging bytes.",
                determinism_steps,
                episodes,
            )
        finally:
            left.close()
            right.close()

    def test_reset_with_the_same_seed_is_repeatable(self) -> None:
        """A single env re-reset with one seed must reproduce its observation."""
        _skip_unless_runnable()

        env = _make_env()
        try:
            first, _ = env.reset(seed=DETERMINISM_SEED)
            first_fingerprint = _observation_fingerprint(first)
            # Step away from the reset state so a stale-buffer bug cannot pass.
            env.step(0)
            second, _ = env.reset(seed=DETERMINISM_SEED)
            assert _observation_fingerprint(second) == first_fingerprint
        finally:
            env.close()


class TestTheGateCanActuallyFail:
    """A determinism test that cannot fail proves nothing.

    These pin the *sensitivity* of the comparison, so a future refactor that
    weakens the fingerprint (say, comparing only shapes) is caught here rather
    than silently turning the gate above into a no-op.
    """

    def test_different_seeds_produce_different_observations(self) -> None:
        _skip_unless_runnable()

        left, right = _make_env(), _make_env()
        try:
            left_obs, _ = left.reset(seed=DETERMINISM_SEED)
            right_obs, _ = right.reset(seed=DETERMINISM_SEED + 1)
            assert _observation_fingerprint(left_obs) != _observation_fingerprint(right_obs), (
                "two different seeds produced identical observations, so the "
                "fingerprint is not sensitive to world state and the gate above "
                "would pass even on a broken engine"
            )
        finally:
            left.close()
            right.close()

    def test_fingerprint_detects_a_single_changed_value(self) -> None:
        """The comparison is byte-level, not shape-level."""
        _skip_unless_runnable()
        import numpy as np

        base = {"grid": np.zeros(4, dtype=np.uint8), "scalar": np.float32(1.0)}
        perturbed = {"grid": np.zeros(4, dtype=np.uint8), "scalar": np.float32(1.0)}
        perturbed["grid"][2] = 1

        assert _observation_fingerprint(base) != _observation_fingerprint(perturbed)

    def test_fingerprint_detects_a_value_moving_between_keys(self) -> None:
        """Key names are part of the fingerprint, so swaps do not cancel out."""
        _skip_unless_runnable()
        import numpy as np

        left = {"a": np.uint8(1), "b": np.uint8(2)}
        right = {"a": np.uint8(2), "b": np.uint8(1)}
        assert _observation_fingerprint(left) != _observation_fingerprint(right)
