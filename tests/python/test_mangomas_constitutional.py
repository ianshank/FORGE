"""Tests for MangoMAS constitutional pre-trainer."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forge.mangomas.config import ConstitutionalTrainerConfig
from forge.mangomas.constitutional_trainer import (
    DEFAULT_CONSTRAINTS,
    ConstitutionalDataset,
    ConstitutionalPreTrainer,
    ConstraintViolation,
)


class TestConstraintChecking:
    """Tests for constraint violation checking."""

    def test_no_violations_safe_state(self) -> None:
        trainer = ConstitutionalPreTrainer()
        obs = {
            "battery": 0.8,           # above 0.2 floor
            "altitude": 0.5,          # below 0.9 ceiling
            "stamina_inverse": 0.3,   # below 0.8 ceiling
            "boundary_distance": 0.5, # above 0.1 floor
            "threat_proximity": 0.8,  # above 0.3 floor
        }
        violations = trainer.check_violations(obs)
        assert len(violations) == 0

    def test_battery_violation(self) -> None:
        trainer = ConstitutionalPreTrainer()
        obs = {"battery": 0.1, "altitude": 0.5, "stamina_inverse": 0.3,
               "boundary_distance": 0.5, "threat_proximity": 0.5}
        violations = trainer.check_violations(obs)
        names = [v.constraint_name for v in violations]
        assert "battery_minimum" in names

    def test_altitude_violation(self) -> None:
        trainer = ConstitutionalPreTrainer()
        obs = {"battery": 0.8, "altitude": 0.95, "stamina_inverse": 0.3,
               "boundary_distance": 0.5, "threat_proximity": 0.5}
        violations = trainer.check_violations(obs)
        names = [v.constraint_name for v in violations]
        assert "altitude_ceiling" in names

    def test_multiple_violations(self) -> None:
        trainer = ConstitutionalPreTrainer()
        obs = {"battery": 0.05, "altitude": 0.95, "stamina_inverse": 0.9,
               "boundary_distance": 0.05, "threat_proximity": 0.1}
        violations = trainer.check_violations(obs)
        assert len(violations) == 5  # All 5 constraints violated

    def test_all_5_constraints_covered(self) -> None:
        assert len(DEFAULT_CONSTRAINTS) == 5
        names = {c["name"] for c in DEFAULT_CONSTRAINTS}
        assert names == {"battery_minimum", "altitude_ceiling", "speed_ceiling",
                         "geofence", "threat_exclusion"}


class TestConstitutionalPenalty:
    """Tests for constraint penalty computation."""

    def test_no_penalty_no_violations(self) -> None:
        trainer = ConstitutionalPreTrainer()
        assert trainer.compute_penalty([]) == 0.0

    def test_penalty_scales_with_severity(self) -> None:
        trainer = ConstitutionalPreTrainer(
            config=ConstitutionalTrainerConfig(penalty_weight=10.0)
        )
        v1 = ConstraintViolation("test", 0.0, 0.2, True, 0.2)
        v2 = ConstraintViolation("test", 0.0, 0.2, True, 0.1)
        p1 = trainer.compute_penalty([v1])
        p2 = trainer.compute_penalty([v2])
        assert p1 > p2


class TestConstitutionalDataset:
    """Tests for ConstitutionalDataset."""

    def test_violation_rate(self) -> None:
        ds = ConstitutionalDataset(
            observations=np.zeros((10, 18)),
            actions=np.zeros(10, dtype=np.int64),
            rewards=np.ones(10),
            constraint_violations=np.zeros((10, 5)),
            penalties=np.zeros(10),
        )
        assert ds.violation_rate == 0.0

    def test_violation_rate_partial(self) -> None:
        violations = np.zeros((10, 5))
        violations[0, 0] = 1.0
        violations[5, 2] = 1.0
        ds = ConstitutionalDataset(
            observations=np.zeros((10, 18)),
            actions=np.zeros(10, dtype=np.int64),
            rewards=np.ones(10),
            constraint_violations=violations,
            penalties=np.zeros(10),
        )
        assert ds.violation_rate == pytest.approx(0.2)


class TestConstitutionalPreTrainer:
    """Tests for constitutional RL training."""

    def test_train_uses_config_dimensions_for_empty_dataset(self) -> None:
        config = ConstitutionalTrainerConfig(
            num_epochs=1,
            batch_size=4,
            state_dim=7,
            action_dim=3,
            seed=123,
        )
        trainer = ConstitutionalPreTrainer(config=config)
        ds = ConstitutionalDataset(
            observations=np.zeros((0, 0), dtype=np.float32),
            actions=np.zeros(0, dtype=np.int64),
            rewards=np.zeros(0, dtype=np.float32),
            constraint_violations=np.zeros((0, 5), dtype=np.float32),
            penalties=np.zeros(0, dtype=np.float32),
        )

        trainer.train(ds)

        assert trainer._weights is not None
        assert trainer._weights["policy_w"].shape == (3, 7)
        assert trainer._weights["value_w"].shape == (1, 7)

    def test_train_seed_comes_from_config(self) -> None:
        dataset = ConstitutionalDataset(
            observations=np.zeros((0, 0), dtype=np.float32),
            actions=np.zeros(0, dtype=np.int64),
            rewards=np.zeros(0, dtype=np.float32),
            constraint_violations=np.zeros((0, 5), dtype=np.float32),
            penalties=np.zeros(0, dtype=np.float32),
        )
        config_a = ConstitutionalTrainerConfig(
            num_epochs=1,
            batch_size=4,
            state_dim=6,
            action_dim=4,
            seed=1,
        )
        config_b = ConstitutionalTrainerConfig(
            num_epochs=1,
            batch_size=4,
            state_dim=6,
            action_dim=4,
            seed=2,
        )

        trainer_a = ConstitutionalPreTrainer(config=config_a)
        trainer_b = ConstitutionalPreTrainer(config=config_b)

        trainer_a.train(dataset)
        trainer_b.train(dataset)

        assert trainer_a._weights is not None
        assert trainer_b._weights is not None
        assert not np.allclose(trainer_a._weights["policy_w"], trainer_b._weights["policy_w"])

    def test_build_dataset(self) -> None:
        trainer = ConstitutionalPreTrainer()
        obs = np.random.rand(20, 18).astype(np.float32)
        actions = np.random.randint(0, 10, 20).astype(np.int64)
        rewards = np.ones(20, dtype=np.float32)
        obs_dicts = [{"battery": 0.1, "altitude": 0.5, "stamina_inverse": 0.3,
                       "boundary_distance": 0.5, "threat_proximity": 0.5}] * 20
        ds = trainer.build_dataset(obs, actions, rewards, obs_dicts)
        assert ds.num_samples == 20
        assert ds.constraint_violations.shape == (20, 5)
        # battery_minimum should be violated in all samples
        assert ds.constraint_violations[:, 0].sum() == 20

    def test_train(self) -> None:
        config = ConstitutionalTrainerConfig(num_epochs=3, batch_size=16)
        trainer = ConstitutionalPreTrainer(config=config)
        ds = ConstitutionalDataset(
            observations=np.random.rand(50, 18).astype(np.float32),
            actions=np.random.randint(0, 10, 50).astype(np.int64),
            rewards=np.random.rand(50).astype(np.float32),
            constraint_violations=np.zeros((50, 5)),
            penalties=np.zeros(50, dtype=np.float32),
        )
        result = trainer.train(ds)
        assert result.epochs_run == 3
        assert len(result.loss_history) == 3

    def test_export_weights(self, tmp_path: Any) -> None:
        config = ConstitutionalTrainerConfig(num_epochs=2)
        trainer = ConstitutionalPreTrainer(config=config)
        ds = ConstitutionalDataset(
            observations=np.random.rand(30, 18).astype(np.float32),
            actions=np.random.randint(0, 10, 30).astype(np.int64),
            rewards=np.random.rand(30).astype(np.float32),
            constraint_violations=np.zeros((30, 5)),
            penalties=np.zeros(30, dtype=np.float32),
        )
        trainer.train(ds)
        path = tmp_path / "const_weights.npz"
        trainer.export_weights(path)
        assert path.exists()
        loaded = np.load(str(path))
        assert "policy_w" in loaded
        assert "value_w" in loaded

    def test_export_before_train_raises(self, tmp_path: Any) -> None:
        trainer = ConstitutionalPreTrainer()
        with pytest.raises(RuntimeError, match="No trained weights"):
            trainer.export_weights(tmp_path / "weights.npz")

    def test_empty_dataset_violation_rate(self) -> None:
        ds = ConstitutionalDataset(
            observations=np.zeros((0, 18)),
            actions=np.zeros(0, dtype=np.int64),
            rewards=np.zeros(0),
            constraint_violations=np.zeros((0, 5)),
            penalties=np.zeros(0),
        )
        assert ds.violation_rate == 0.0
