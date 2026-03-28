"""Tests for MangoMAS RSSM pre-trainer."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forge.mangomas.config import RSSMPreTrainConfig
from forge.mangomas.rssm_pretrainer import RSSMPreTrainer, SequenceDataset


class TestSequenceDataset:
    """Tests for SequenceDataset."""

    def test_num_sequences(self) -> None:
        ds = SequenceDataset(
            states=np.zeros((10, 50, 22)),
            actions=np.zeros((10, 50), dtype=np.int64),
            next_states=np.zeros((10, 50, 22)),
            rewards=np.zeros((10, 50)),
            dones=np.zeros((10, 50)),
        )
        assert ds.num_sequences == 10
        assert ds.sequence_length == 50


class TestRSSMPreTrainer:
    """Tests for RSSM pre-training."""

    def test_build_sequences(self) -> None:
        config = RSSMPreTrainConfig(sequence_length=10, state_dim=18)
        trainer = RSSMPreTrainer(config=config)

        # 2 episodes of 25 steps each → 2 sequences of length 10 each (floor)
        ep_obs = [np.random.rand(26, 18).astype(np.float32) for _ in range(2)]
        ep_acts = [np.random.randint(0, 75, 25).astype(np.int64) for _ in range(2)]
        ep_rews = [np.random.rand(25).astype(np.float32) for _ in range(2)]
        ep_dones = [np.zeros(25, dtype=np.float32) for _ in range(2)]

        ds = trainer.build_sequences(ep_obs, ep_acts, ep_rews, ep_dones)
        assert ds.num_sequences > 0
        assert ds.sequence_length == 10
        assert ds.states.shape[2] == 18

    def test_build_sequences_short_episodes(self) -> None:
        config = RSSMPreTrainConfig(sequence_length=50)
        trainer = RSSMPreTrainer(config=config)

        # Episode shorter than sequence length → no sequences
        ep_obs = [np.random.rand(10, 22).astype(np.float32)]
        ep_acts = [np.random.randint(0, 75, 9).astype(np.int64)]
        ep_rews = [np.random.rand(9).astype(np.float32)]
        ep_dones = [np.zeros(9, dtype=np.float32)]

        ds = trainer.build_sequences(ep_obs, ep_acts, ep_rews, ep_dones)
        assert ds.num_sequences == 0

    def test_train(self) -> None:
        config = RSSMPreTrainConfig(num_epochs=3, batch_size=4, state_dim=18)
        trainer = RSSMPreTrainer(config=config)

        ds = SequenceDataset(
            states=np.random.rand(8, 10, 18).astype(np.float32),
            actions=np.random.randint(0, 75, (8, 10)).astype(np.int64),
            next_states=np.random.rand(8, 10, 18).astype(np.float32),
            rewards=np.random.rand(8, 10).astype(np.float32),
            dones=np.zeros((8, 10), dtype=np.float32),
        )

        result = trainer.train(ds)
        assert result.epochs_run == 3
        assert result.total_loss >= 0.0
        assert len(result.loss_history) == 3

    def test_train_empty_dataset(self) -> None:
        config = RSSMPreTrainConfig(num_epochs=2, state_dim=18)
        trainer = RSSMPreTrainer(config=config)

        ds = SequenceDataset(
            states=np.zeros((0, 10, 18), dtype=np.float32),
            actions=np.zeros((0, 10), dtype=np.int64),
            next_states=np.zeros((0, 10, 18), dtype=np.float32),
            rewards=np.zeros((0, 10), dtype=np.float32),
            dones=np.zeros((0, 10), dtype=np.float32),
        )

        result = trainer.train(ds)
        assert result.epochs_run == 2
        assert result.total_loss == 0.0

    def test_export_all(self, tmp_path: Any) -> None:
        config = RSSMPreTrainConfig(num_epochs=2, batch_size=4, state_dim=18)
        trainer = RSSMPreTrainer(config=config)

        ds = SequenceDataset(
            states=np.random.rand(4, 10, 18).astype(np.float32),
            actions=np.random.randint(0, 75, (4, 10)).astype(np.int64),
            next_states=np.random.rand(4, 10, 18).astype(np.float32),
            rewards=np.random.rand(4, 10).astype(np.float32),
            dones=np.zeros((4, 10), dtype=np.float32),
        )
        trainer.train(ds)

        path = tmp_path / "rssm_weights.npz"
        trainer.export_all(path)
        assert path.exists()

        loaded = np.load(str(path))
        assert "gru_w_ih" in loaded
        assert "gru_w_hh" in loaded
        assert "prior_mean_w" in loaded
        assert "reward_w0" in loaded
        assert "value_w0" in loaded

    def test_export_before_train_raises(self, tmp_path: Any) -> None:
        trainer = RSSMPreTrainer()
        with pytest.raises(RuntimeError, match="No trained weights"):
            trainer.export_all(tmp_path / "weights.npz")

    def test_weight_shapes(self) -> None:
        config = RSSMPreTrainConfig(hidden_dim=64, latent_dim=16, state_dim=18, action_dim=75, num_epochs=1, batch_size=4)
        trainer = RSSMPreTrainer(config=config)

        ds = SequenceDataset(
            states=np.random.rand(4, 10, 18).astype(np.float32),
            actions=np.random.randint(0, 75, (4, 10)).astype(np.int64),
            next_states=np.random.rand(4, 10, 18).astype(np.float32),
            rewards=np.random.rand(4, 10).astype(np.float32),
            dones=np.zeros((4, 10), dtype=np.float32),
        )
        trainer.train(ds)

        w = trainer._weights
        assert w["gru_w_ih"].shape == (3 * 64, 18 + 75)
        assert w["gru_w_hh"].shape == (3 * 64, 64)
        assert w["prior_mean_w"].shape == (16, 64)
        assert w["prior_logvar_w"].shape == (16, 64)
