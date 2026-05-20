"""Pin :func:`forge.training._targets.compute_n_step_return`.

The helper is the single source of truth for the n-step MuZero return.
Both ``MuZeroReplayBuffer`` (existing) and the new
``muzero_mc.trainer.MuzeroMcTrainer`` consume it. A regression here is
a silent bug in every downstream value target.
"""

from __future__ import annotations

import math

import pytest

from forge.training._targets import compute_n_step_return


def test_pure_bootstrap_when_td_steps_zero() -> None:
    # td_steps == 0 collapses to value[position] * discount**0 == value[position].
    v = compute_n_step_return(
        rewards=[10.0, 20.0, 30.0],
        values=[0.5, 0.6, 0.7],
        position=1,
        td_steps=0,
        discount=0.99,
    )
    assert math.isclose(v, 0.6, abs_tol=1e-9)


def test_pure_reward_sum_when_bootstrap_past_end() -> None:
    # bootstrap_pos (= position + td_steps) exceeds len(values),
    # so only the discounted reward sum survives.
    rewards = [1.0, 1.0, 1.0]
    values = [0.0]  # length 1, bootstrap_pos=3 is past the end
    v = compute_n_step_return(
        rewards=rewards,
        values=values,
        position=0,
        td_steps=3,
        discount=0.5,
    )
    # 1.0 + 0.5*1.0 + 0.25*1.0 = 1.75
    assert math.isclose(v, 1.75, abs_tol=1e-9)


def test_full_formula_matches_hand_computed() -> None:
    rewards = [1.0, 1.0, 1.0, 0.0]
    values = [10.0, 10.0, 10.0, 10.0, 5.0]  # bootstrap_pos = 0 + 3 = 3 -> value 10.0
    discount = 0.9
    # 1.0 + 0.9*1.0 + 0.81*1.0 + 0.9**3 * 10.0
    expected = 1.0 + 0.9 + 0.81 + (0.9**3) * 10.0
    actual = compute_n_step_return(
        rewards=rewards,
        values=values,
        position=0,
        td_steps=3,
        discount=discount,
    )
    assert math.isclose(actual, expected, abs_tol=1e-9)


def test_truncates_sum_at_end_of_rewards() -> None:
    # Position 2 with td_steps=5 in a 4-step trajectory:
    # only rewards[2], rewards[3] count; rest is silently dropped.
    rewards = [1.0, 1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0, 0.0, 0.0]  # bootstrap_pos = 7 > len(values)
    v = compute_n_step_return(
        rewards=rewards,
        values=values,
        position=2,
        td_steps=5,
        discount=1.0,
    )
    # 1.0 + 1.0 (from steps 2 and 3) + nothing from steps 4/5/6 (OOB) + no bootstrap
    assert math.isclose(v, 2.0, abs_tol=1e-9)


def test_rejects_negative_position() -> None:
    with pytest.raises(AssertionError):
        compute_n_step_return(
            rewards=[1.0], values=[0.0], position=-1, td_steps=1, discount=1.0
        )


def test_rejects_negative_td_steps() -> None:
    with pytest.raises(AssertionError):
        compute_n_step_return(
            rewards=[1.0], values=[0.0], position=0, td_steps=-1, discount=1.0
        )


def test_extraction_matches_original_buffer_method() -> None:
    """Cross-check the extraction against the original
    ``MuZeroReplayBuffer._compute_n_step_return`` invocation shape so
    the wrapper inside the buffer keeps producing identical results.
    """
    from forge.training.muzero_buffer import GameHistory

    # GameHistory.length is len(actions). Populate actions to size 4
    # so length == 4; the buffer slices rewards[:length] before
    # passing to the helper.
    game = GameHistory()
    game.actions = [0, 0, 0, 0]
    game.rewards = [1.0, 0.5, 0.25, 0.125]
    game.root_values = [0.9, 0.8, 0.7, 0.6]
    # Hand-computed for position=1, td_steps=2, discount=0.99:
    #   sum_i rewards[1+i] * 0.99**i for i in 0..1
    #     = 0.5 + 0.99 * 0.25
    #   + bootstrap: 0.99**2 * values[3]
    #     = (0.99**2) * 0.6
    expected = 0.5 + 0.99 * 0.25 + (0.99**2) * 0.6

    helper_value = compute_n_step_return(
        rewards=game.rewards[: game.length],
        values=game.root_values,
        position=1,
        td_steps=2,
        discount=0.99,
    )
    assert math.isclose(helper_value, expected, abs_tol=1e-9)
