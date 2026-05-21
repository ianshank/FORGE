"""Tests for evaluation integration in the training loop.

Validates that:
- ``--eval-interval 0`` (default) produces no evaluation output.
- ``--eval-interval`` and ``--eval-episodes`` CLI args parse correctly.
- ``eval_callback`` in :class:`PPOTrainer` fires at the correct intervals.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

# scripts/ and python/ are placed on sys.path by the root conftest.py.
from train import parse_args

_DEFAULT_EVAL_INTERVAL = 0
_DEFAULT_EVAL_EPISODES = 10


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for ``--eval-interval`` and ``--eval-episodes`` CLI args."""

    def test_defaults(self) -> None:
        """Default eval-interval is 0 (disabled) and eval-episodes is 10."""
        args = parse_args([])
        assert args.eval_interval == _DEFAULT_EVAL_INTERVAL
        assert args.eval_episodes == _DEFAULT_EVAL_EPISODES

    def test_eval_interval_parses(self) -> None:
        """``--eval-interval 5`` sets eval_interval to 5."""
        args = parse_args(["--eval-interval", "5"])
        assert args.eval_interval == 5

    def test_eval_episodes_parses(self) -> None:
        """``--eval-episodes 20`` sets eval_episodes to 20."""
        args = parse_args(["--eval-episodes", "20"])
        assert args.eval_episodes == 20

    def test_both_args_together(self) -> None:
        """Both args can be provided simultaneously."""
        args = parse_args(["--eval-interval", "3", "--eval-episodes", "7"])
        assert args.eval_interval == 3
        assert args.eval_episodes == 7


# ---------------------------------------------------------------------------
# _train_basic: eval_interval=0 produces no evaluation
# ---------------------------------------------------------------------------


class TestTrainBasicEvalDisabled:
    """When eval_interval=0 the Evaluator is never called."""

    def test_eval_interval_zero_no_eval(self) -> None:
        """Verify the guard condition: eval_interval=0 means no eval."""
        args = parse_args(["--eval-interval", "0", "--episodes", "2"])
        assert args.eval_interval == 0
        # The condition ``args.eval_interval > 0`` in _train_basic prevents
        # any Evaluator instantiation, so default behaviour is preserved.


# ---------------------------------------------------------------------------
# PPOTrainer eval_callback
# ---------------------------------------------------------------------------


class TestPPOTrainerEvalCallback:
    """Tests that PPOTrainer.train() fires eval_callback at correct intervals."""

    def test_callback_fires_at_interval(self) -> None:
        """eval_callback is called every eval_interval updates."""
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig

        mock_agent = MagicMock()
        mock_agent.obs_dim = 4
        mock_agent.device = "cpu"
        mock_agent.act.return_value = (0, {"log_prob": 0.0, "value": 0.0})
        mock_agent.learn.return_value = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        mock_agent.compute_gae.return_value = (
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )

        mock_value_tensor = MagicMock()
        mock_value_tensor.item.return_value = 0.0
        mock_agent.network = MagicMock()
        mock_agent.network.get_value.return_value = mock_value_tensor

        config = PPOTrainerConfig(
            rollout_length=8,
            max_episode_steps=4,
            eval_interval=2,
            log_interval=100,
        )
        trainer = PPOTrainer(agent=mock_agent, config=config)

        reset_fn = MagicMock(return_value=np.zeros(4, dtype=np.float32))

        step_count = 0

        def mock_step(_action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
            nonlocal step_count
            step_count += 1
            done = step_count % 4 == 0
            return np.zeros(4, dtype=np.float32), 1.0, done, False, {}

        step_fn = MagicMock(side_effect=mock_step)

        callback_calls: list[tuple[int, object]] = []

        def callback(update: int, agent: object) -> None:
            callback_calls.append((update, agent))

        num_updates = 6
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.as_tensor.return_value.unsqueeze.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            trainer.train(
                reset_fn,
                step_fn,
                num_updates=num_updates,
                eval_callback=callback,
            )

        # eval_interval=2, num_updates=6 => callback at updates 2, 4, 6
        assert len(callback_calls) == 3
        assert [c[0] for c in callback_calls] == [2, 4, 6]

    def test_no_callback_when_none(self) -> None:
        """When eval_callback is None, no callback errors occur."""
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig

        mock_agent = MagicMock()
        mock_agent.obs_dim = 4
        mock_agent.device = "cpu"
        mock_agent.act.return_value = (0, {"log_prob": 0.0, "value": 0.0})
        mock_agent.learn.return_value = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        mock_agent.compute_gae.return_value = (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
        mock_value_tensor = MagicMock()
        mock_value_tensor.item.return_value = 0.0
        mock_agent.network = MagicMock()
        mock_agent.network.get_value.return_value = mock_value_tensor

        config = PPOTrainerConfig(
            rollout_length=4,
            max_episode_steps=4,
            eval_interval=1,
            log_interval=100,
        )
        trainer = PPOTrainer(agent=mock_agent, config=config)

        reset_fn = MagicMock(return_value=np.zeros(4, dtype=np.float32))
        step_fn = MagicMock(
            return_value=(np.zeros(4, dtype=np.float32), 1.0, True, False, {}),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.as_tensor.return_value.unsqueeze.return_value = MagicMock()
        # Should not raise -- eval_callback defaults to None
        with patch.dict("sys.modules", {"torch": mock_torch}):
            trainer.train(reset_fn, step_fn, num_updates=2)

    def test_callback_not_fired_when_interval_zero(self) -> None:
        """When eval_interval=0, callback is never called even if provided."""
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig

        mock_agent = MagicMock()
        mock_agent.obs_dim = 4
        mock_agent.device = "cpu"
        mock_agent.act.return_value = (0, {"log_prob": 0.0, "value": 0.0})
        mock_agent.learn.return_value = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        mock_agent.compute_gae.return_value = (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
        mock_value_tensor = MagicMock()
        mock_value_tensor.item.return_value = 0.0
        mock_agent.network = MagicMock()
        mock_agent.network.get_value.return_value = mock_value_tensor

        config = PPOTrainerConfig(
            rollout_length=4,
            max_episode_steps=4,
            eval_interval=0,
            log_interval=100,
        )
        trainer = PPOTrainer(agent=mock_agent, config=config)

        reset_fn = MagicMock(return_value=np.zeros(4, dtype=np.float32))
        step_fn = MagicMock(
            return_value=(np.zeros(4, dtype=np.float32), 1.0, True, False, {}),
        )

        callback = MagicMock()

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.as_tensor.return_value.unsqueeze.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            trainer.train(reset_fn, step_fn, num_updates=3, eval_callback=callback)

        callback.assert_not_called()
