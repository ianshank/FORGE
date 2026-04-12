"""End-to-end integration test for MuZero pipeline.

Tests the complete flow: create model -> self-play -> train -> export.
Uses a mock environment to avoid FORGE native library dependency.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.agents.muzero_mcts import MuZeroMCTS, MuZeroMCTSConfig  # noqa: E402
from forge.models.muzero_config import MuZeroConfig  # noqa: E402
from forge.models.muzero_world_model import MuZeroWorldModel  # noqa: E402
from forge.training.muzero_buffer import MuZeroBufferConfig  # noqa: E402
from forge.training.muzero_trainer import MuZeroTrainer, MuZeroTrainerConfig  # noqa: E402

OBS_DIM = 11 * 11 * 7 + 73
ACTION_DIM = 5


def _make_mock_env(max_steps: int = 8):
    env = MagicMock()
    env.reset.return_value = (np.random.randn(OBS_DIM).astype(np.float32), {})

    counter = {"n": 0}

    def step_fn(action):
        counter["n"] += 1
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        reward = 1.0 if counter["n"] % 3 == 0 else 0.0
        terminated = counter["n"] >= max_steps
        return obs, reward, terminated, False, {}

    env.step.side_effect = step_fn
    env.close.return_value = None
    return env


class TestMuZeroEndToEnd:
    """Full pipeline smoke test: model -> self-play -> train -> verify."""

    def test_pipeline(self, tmp_path) -> None:
        # 1. Create model with small dimensions
        config = MuZeroConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            latent_dim=16,
            hidden_dim=16,
            num_blocks=1,
            num_unroll_steps=2,
            td_steps=3,
            reward_support_size=11,
            value_support_size=11,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
        )
        model = MuZeroWorldModel(config)

        # 2. Verify initial inference works
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        output = model.initial_inference(obs)
        assert output.latent_state.shape == (16,)
        assert output.policy_logits.shape == (ACTION_DIM,)

        # 3. Verify recurrent inference works
        rec_out = model.recurrent_inference(output.latent_state, action=0)
        assert rec_out.latent_state.shape == (16,)
        assert np.isfinite(rec_out.reward)

        # 4. Self-play + training
        trainer_config = MuZeroTrainerConfig(
            training_steps_per_iter=2,
            self_play_games_per_iter=2,
            batch_size=4,
            max_episode_steps=10,
            checkpoint_dir=str(tmp_path / "ckpt"),
            checkpoint_interval=0,  # Disable checkpointing
            buffer_config=MuZeroBufferConfig(capacity=50),
            mcts_config=MuZeroMCTSConfig(
                num_simulations=3,
                add_exploration_noise=False,
            ),
        )
        trainer = MuZeroTrainer(trainer_config, model)

        # Run 1 iteration
        history = trainer.train(
            env_factory=lambda: _make_mock_env(max_steps=6),
            num_iterations=1,
        )

        assert trainer.total_games >= 2
        assert trainer.total_train_steps >= 2
        assert len(history["loss"]) > 0

        # 5. Save and reload model
        save_path = str(tmp_path / "muzero.pt")
        model.save(save_path)
        model2 = MuZeroWorldModel(config)
        model2.load(save_path)

        # 6. Verify loaded model produces valid output
        torch.manual_seed(42)
        out1 = model.initial_inference(obs)
        torch.manual_seed(42)
        out2 = model2.initial_inference(obs)
        np.testing.assert_allclose(out1.latent_state, out2.latent_state, atol=1e-5)

    def test_mcts_with_trained_model(self) -> None:
        """Verify MCTS produces valid actions after training."""
        config = MuZeroConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            latent_dim=16,
            hidden_dim=16,
            num_blocks=1,
            num_unroll_steps=2,
            reward_support_size=11,
            value_support_size=11,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
        )
        model = MuZeroWorldModel(config)

        # Quick training
        trainer_config = MuZeroTrainerConfig(
            training_steps_per_iter=1,
            self_play_games_per_iter=1,
            batch_size=2,
            max_episode_steps=5,
            buffer_config=MuZeroBufferConfig(capacity=20),
            mcts_config=MuZeroMCTSConfig(num_simulations=3, add_exploration_noise=False),
        )
        trainer = MuZeroTrainer(trainer_config, model)
        trainer.train(env_factory=lambda: _make_mock_env(5), num_iterations=1)

        # Run MCTS with the trained model
        mcts = MuZeroMCTS(model, MuZeroMCTSConfig(num_simulations=5))
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, info = mcts.search(obs, temperature=0.0)

        assert 0 <= action < ACTION_DIM
        assert info["visit_counts"].sum() > 0
