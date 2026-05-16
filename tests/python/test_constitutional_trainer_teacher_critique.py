"""Tests for ConstitutionalPreTrainer.build_dataset teacher critique merging."""

from __future__ import annotations

import numpy as np

from forge.mangomas.config import ConstitutionalTrainerConfig
from forge.mangomas.constitutional_trainer import ConstitutionalPreTrainer


def _sample_inputs(n: int = 4) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]
]:
    obs = np.zeros((n, 4), dtype=np.float32)
    actions = np.zeros((n,), dtype=np.int64)
    rewards = np.ones((n,), dtype=np.float32)
    obs_dicts: list[dict[str, float]] = [
        {"battery": 0.9, "altitude": 0.2, "stamina_inverse": 0.0,
         "boundary_distance": 0.5, "threat_proximity": 1.0}
        for _ in range(n)
    ]
    return obs, actions, rewards, obs_dicts


def test_default_violations_unchanged_without_critique() -> None:
    trainer = ConstitutionalPreTrainer(ConstitutionalTrainerConfig())
    obs, actions, rewards, obs_dicts = _sample_inputs(3)
    dataset = trainer.build_dataset(obs, actions, rewards, obs_dicts)
    # battery=0.9 (>=0.2 lower bound -> no violation),
    # altitude=0.2 (<=0.9 ceiling -> no violation),
    # threat_proximity=1.0 (>=0.3 lower bound -> no violation),
    # so violation_matrix should be all zero.
    assert dataset.constraint_violations.shape == (3, len(trainer.constraints))
    assert (dataset.constraint_violations == 0).all()
    assert (dataset.penalties == 0).all()


def test_teacher_critique_or_merged_with_rules() -> None:
    trainer = ConstitutionalPreTrainer(ConstitutionalTrainerConfig())
    obs, actions, rewards, obs_dicts = _sample_inputs(3)
    critiques = [
        {"battery_minimum": True, "altitude_ceiling": False},
        {},
        {"speed_ceiling": True},
    ]
    dataset = trainer.build_dataset(
        obs,
        actions,
        rewards,
        obs_dicts,
        teacher_constraint_critiques=critiques,
    )
    name_index = {c["name"]: i for i, c in enumerate(trainer.constraints)}
    assert dataset.constraint_violations[0, name_index["battery_minimum"]] == 1.0
    assert dataset.constraint_violations[2, name_index["speed_ceiling"]] == 1.0
    assert dataset.constraint_violations[1].sum() == 0.0


def test_penalty_recomputed_with_merged_violations() -> None:
    trainer = ConstitutionalPreTrainer(
        ConstitutionalTrainerConfig(penalty_weight=2.0)
    )
    obs, actions, rewards, obs_dicts = _sample_inputs(2)
    critiques = [{"battery_minimum": True, "speed_ceiling": True}, {}]
    dataset = trainer.build_dataset(
        obs,
        actions,
        rewards,
        obs_dicts,
        teacher_constraint_critiques=critiques,
        teacher_severity_default=1.5,
    )
    # 2 teacher flags x severity 1.5 x penalty_weight 2.0 = 6.0
    assert dataset.penalties[0] == 6.0
    assert dataset.penalties[1] == 0.0


def test_teacher_critique_length_mismatch_raises() -> None:
    trainer = ConstitutionalPreTrainer(ConstitutionalTrainerConfig())
    obs, actions, rewards, obs_dicts = _sample_inputs(3)
    import pytest

    with pytest.raises(ValueError, match="same length"):
        trainer.build_dataset(
            obs,
            actions,
            rewards,
            obs_dicts,
            teacher_constraint_critiques=[{}],
        )
