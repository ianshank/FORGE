"""Shared MuZero training-target helpers.

Trajectory-agnostic value/return computations the MuZero family of
trainers (`muzero_trainer.MuZeroTrainer`, the new
`muzero_mc.trainer.MuzeroMcTrainer`) all consume. Extracted here so the
formula has one canonical source — divergent copies would silently
shift the discounting between trainers.

No hard-coded values; every numeric (discount, td_steps, position)
flows in through arguments. The helper does not own a trajectory type
— it takes plain lists of floats so it can run against
``GameHistory``, ``TrajectoryV2``, or any future format without
adapters.
"""

from __future__ import annotations

__all__ = ["compute_n_step_return"]

from collections.abc import (
    Sequence,  # noqa: TC003 — used in runtime annotations only but kept top-level for documented public-API clarity
)


def compute_n_step_return(
    rewards: Sequence[float],
    values: Sequence[float],
    position: int,
    td_steps: int,
    discount: float,
) -> float:
    """Compute the n-step bootstrapped return from ``position``.

    Mirrors the original
    ``MuZeroReplayBuffer._compute_n_step_return``
    (muzero_buffer.py:315-345) byte-for-byte; this module-level
    function is the canonical home so the buffer + the new
    ``muzero_mc.trainer`` consume the same formula.

    Args:
        rewards: Per-step reward sequence for the trajectory.
        values: Per-step root-value sequence used for the bootstrap.
            Must be aligned with ``rewards`` by index.
        position: Starting step index. The return is computed
            starting from this step.
        td_steps: How many steps to sum before bootstrapping. ``0``
            collapses to a pure bootstrap from ``values[position]``.
        discount: Per-step reward discount factor (typically near 1).

    Returns:
        The n-step return as a Python ``float``.

    Notes:
        - The loop is bounded by ``min(td_steps, len(rewards) -
          position)`` — if the trajectory ends inside the window, the
          tail is treated as zero rewards (no bootstrap from
          out-of-bounds).
        - The bootstrap term is added only when ``position + td_steps
          < len(values)``; otherwise the return is the truncated sum
          alone.
        - Negative ``position`` / ``td_steps`` are caller errors;
          asserted in debug builds via ``assert`` to avoid the cost on
          hot paths.
    """
    assert position >= 0, f"position must be >= 0, got {position}"
    assert td_steps >= 0, f"td_steps must be >= 0, got {td_steps}"

    value = 0.0
    reward_len = len(rewards)
    for i in range(td_steps):
        step = position + i
        if step >= reward_len:
            break
        value += (discount**i) * float(rewards[step])

    bootstrap_pos = position + td_steps
    if bootstrap_pos < len(values):
        value += (discount**td_steps) * float(values[bootstrap_pos])

    return value
