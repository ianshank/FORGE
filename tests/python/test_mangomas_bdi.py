"""Tests for MangoMAS BDI pre-trainer."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forge.mangomas.bdi_trainer import (
    DEFAULT_ACTION_INTENTION_MAP,
    BDIDataset,
    BDIPreTrainer,
)
from forge.mangomas.config import BDITrainerConfig


class TestBDIIntentionMapping:
    """Tests for BDI action → intention mapping."""

    def test_navigate_actions(self) -> None:
        trainer = BDIPreTrainer()
        for action in ["Move", "MoveUp", "MoveDown", "Ascend", "Descend", "TakeOff", "Land"]:
            assert trainer.map_action_to_intention(action) == 0  # Navigate

    def test_gather_actions(self) -> None:
        trainer = BDIPreTrainer()
        for action in ["PickUp", "Drop", "DropPayload"]:
            assert trainer.map_action_to_intention(action) == 1  # Gather

    def test_plan_actions(self) -> None:
        trainer = BDIPreTrainer()
        assert trainer.map_action_to_intention("Craft") == 2

    def test_manipulate_actions(self) -> None:
        trainer = BDIPreTrainer()
        for action in ["Push", "Use", "Interact"]:
            assert trainer.map_action_to_intention(action) == 3

    def test_cooperate_actions(self) -> None:
        trainer = BDIPreTrainer()
        assert trainer.map_action_to_intention("Communicate") == 4

    def test_track_actions(self) -> None:
        trainer = BDIPreTrainer()
        assert trainer.map_action_to_intention("Scan") == 6

    def test_idle_actions(self) -> None:
        trainer = BDIPreTrainer()
        for action in ["Noop", "Hover"]:
            assert trainer.map_action_to_intention(action) == 7

    def test_unknown_action_defaults_to_idle(self) -> None:
        trainer = BDIPreTrainer()
        assert trainer.map_action_to_intention("UnknownAction") == 7

    def test_override(self) -> None:
        trainer = BDIPreTrainer(overrides={39: 4})  # action 39 → Cooperate
        assert trainer.map_action_to_intention("Interact", action_id=39) == 4

    def test_all_8_intentions_reachable(self) -> None:
        intentions = set(DEFAULT_ACTION_INTENTION_MAP.values())
        # All except Evade(5) which is contextual
        assert 0 in intentions  # Navigate
        assert 1 in intentions  # Gather
        assert 2 in intentions  # Plan
        assert 3 in intentions  # Manipulate
        assert 4 in intentions  # Cooperate
        assert 6 in intentions  # Track
        assert 7 in intentions  # Idle


class TestBDIDataset:
    """Tests for BDIDataset."""

    def test_num_samples(self) -> None:
        ds = BDIDataset(
            observations=np.zeros((100, 18)),
            intentions=np.zeros(100, dtype=np.int64),
            rewards=np.zeros(100),
        )
        assert ds.num_samples == 100

    def test_intention_distribution(self) -> None:
        intentions = np.array([0, 0, 0, 1, 1, 2, 7, 7], dtype=np.int64)
        ds = BDIDataset(
            observations=np.zeros((8, 18)),
            intentions=intentions,
            rewards=np.zeros(8),
        )
        dist = ds.intention_distribution()
        assert dist["Navigate"] == pytest.approx(3 / 8)
        assert dist["Idle"] == pytest.approx(2 / 8)
        assert dist["Evade"] == 0.0

    def test_empty_dataset_intention_distribution(self) -> None:
        ds = BDIDataset(
            observations=np.zeros((0, 18)),
            intentions=np.array([], dtype=np.int64),
            rewards=np.array([]),
        )
        dist = ds.intention_distribution()
        assert all(v == 0.0 for v in dist.values())
        assert len(dist) == 8


class TestBDIPreTrainer:
    """Tests for BDI pre-training."""

    def test_build_dataset(self) -> None:
        trainer = BDIPreTrainer()
        obs = [np.random.rand(10, 18).astype(np.float32)]
        actions = [["Move"] * 10]
        rewards = [[1.0] * 10]
        ds = trainer.build_dataset(obs, actions, rewards)
        assert ds.num_samples == 10
        assert np.all(ds.intentions == 0)  # All Navigate

    def test_train(self) -> None:
        config = BDITrainerConfig(num_epochs=5, batch_size=16)
        trainer = BDIPreTrainer(config=config)
        ds = BDIDataset(
            observations=np.random.rand(100, 18).astype(np.float32),
            intentions=np.random.randint(0, 8, 100).astype(np.int64),
            rewards=np.random.rand(100).astype(np.float32),
        )
        result = trainer.train(ds)
        assert result.epochs_run == 5
        assert len(result.loss_history) == 5
        assert len(result.accuracy_history) == 5

    def test_export_weights(self, tmp_path: Any) -> None:
        config = BDITrainerConfig(num_epochs=2)
        trainer = BDIPreTrainer(config=config)
        ds = BDIDataset(
            observations=np.random.rand(50, 18).astype(np.float32),
            intentions=np.random.randint(0, 8, 50).astype(np.int64),
            rewards=np.random.rand(50).astype(np.float32),
        )
        trainer.train(ds)
        path = tmp_path / "bdi_weights.npz"
        trainer.export_weights(path)
        assert path.exists()
        loaded = np.load(str(path))
        assert "gru_w_ih" in loaded
        assert "mlp_w_out" in loaded

    def test_export_before_train_raises(self, tmp_path: Any) -> None:
        trainer = BDIPreTrainer()
        with pytest.raises(RuntimeError, match="No trained weights"):
            trainer.export_weights(tmp_path / "weights.npz")
