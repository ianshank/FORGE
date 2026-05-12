"""Tests for ``BDIPreTrainer.build_dataset(teacher_intentions=...)``."""

from __future__ import annotations

import numpy as np
from forge.mangomas.bdi_trainer import BDIPreTrainer
from forge.mangomas.config import BDITrainerConfig


def _fake_episode(steps: int) -> tuple[np.ndarray, list[str], list[float]]:
    obs = np.random.default_rng(0).normal(size=(steps, 4)).astype(np.float32)
    actions = ["Noop"] * steps
    rewards = [0.0] * steps
    return obs, actions, rewards


def test_default_dataset_unchanged_without_teacher_intentions() -> None:
    trainer = BDIPreTrainer(BDITrainerConfig(num_intentions=8, default_intention=7))
    obs1, act1, rew1 = _fake_episode(4)
    obs2, act2, rew2 = _fake_episode(3)
    dataset = trainer.build_dataset([obs1, obs2], [act1, act2], [rew1, rew2])
    # All actions are "Noop" -> default_intention=7
    assert dataset.intentions.shape == (7,)
    assert (dataset.intentions == 7).all()


def test_teacher_intentions_override_action_map() -> None:
    trainer = BDIPreTrainer(BDITrainerConfig(num_intentions=8, default_intention=7))
    obs1, act1, rew1 = _fake_episode(3)
    obs2, act2, rew2 = _fake_episode(2)
    teacher = [[1, 2, 3], [4, 5]]
    dataset = trainer.build_dataset(
        [obs1, obs2],
        [act1, act2],
        [rew1, rew2],
        teacher_intentions=teacher,
    )
    assert list(dataset.intentions) == [1, 2, 3, 4, 5]


def test_teacher_intentions_negative_falls_back_to_rule_map() -> None:
    """A sentinel of -1 (used when the teacher didn't emit an intention) falls back."""
    trainer = BDIPreTrainer(BDITrainerConfig(num_intentions=8, default_intention=7))
    obs1, act1, rew1 = _fake_episode(3)
    teacher = [[1, -1, 2]]
    dataset = trainer.build_dataset(
        [obs1], [act1], [rew1], teacher_intentions=teacher
    )
    assert list(dataset.intentions) == [1, 7, 2]


def test_teacher_intentions_length_mismatch_raises() -> None:
    trainer = BDIPreTrainer(BDITrainerConfig())
    obs1, act1, rew1 = _fake_episode(2)
    import pytest

    with pytest.raises(ValueError, match="number of episodes"):
        trainer.build_dataset(
            [obs1], [act1], [rew1], teacher_intentions=[[0], [0]]
        )
