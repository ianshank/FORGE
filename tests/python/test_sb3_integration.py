"""Tests for forge_env.sb3_callbacks — ForgeCurriculumCallback, ForgeMetricsCallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip if SB3 not installed
# ---------------------------------------------------------------------------
_sb3_callbacks = pytest.importorskip("forge_env.sb3_callbacks", reason="SB3 required")

_EPISODE_KEY = _sb3_callbacks._EPISODE_KEY
_TASK_SUCCESS_KEY = _sb3_callbacks._TASK_SUCCESS_KEY
ForgeCurriculumCallback = _sb3_callbacks.ForgeCurriculumCallback
ForgeMetricsCallback = _sb3_callbacks.ForgeMetricsCallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_callback_locals(
    infos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a dict that mimics SB3's ``self.locals`` dict."""
    return {"infos": infos or []}


def _call_on_step(callback: Any, infos: list[dict[str, Any]]) -> bool:
    """Simulate BaseCallback._on_step by injecting locals and calling it."""
    callback.locals = _make_callback_locals(infos)
    callback.num_timesteps = getattr(callback, "num_timesteps", 0) + len(infos)
    return callback._on_step()


# ---------------------------------------------------------------------------
# ForgeCurriculumCallback
# ---------------------------------------------------------------------------


class TestForgeCurriculumCallback:
    """Tests for ForgeCurriculumCallback."""

    def _make(
        self,
        target: float = 0.5,
        window: int = 4,
        rate: int = 1,
    ) -> ForgeCurriculumCallback:
        cb = ForgeCurriculumCallback(
            target_success_rate=target,
            window_size=window,
            adjustment_rate=rate,
        )
        cb.training_env = MagicMock()
        cb.training_env.envs = []
        cb.num_timesteps = 0
        return cb

    # -- construction validation --

    def test_invalid_target_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="target_success_rate"):
            ForgeCurriculumCallback(target_success_rate=0.0)

    def test_invalid_window_size_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            ForgeCurriculumCallback(window_size=0)

    def test_invalid_adjustment_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="adjustment_rate"):
            ForgeCurriculumCallback(adjustment_rate=0)

    # -- tier increments --

    def test_starts_at_tier_one(self) -> None:
        cb = self._make()
        assert cb.current_tier == 1

    def test_success_below_threshold_no_increment(self) -> None:
        cb = self._make(target=1.0, window=4)
        # 3 failures, 1 success (25% < 100%)
        for _ in range(3):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])
        _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        assert cb.current_tier == 1

    def test_success_at_threshold_increments(self) -> None:
        cb = self._make(target=0.5, window=4)
        # 2 success, 2 failure = 50% → threshold met
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])
        assert cb.current_tier == 2
        assert cb.num_tier_increments == 1

    def test_adjustment_rate_applied(self) -> None:
        cb = self._make(target=0.5, window=4, rate=3)
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])
        assert cb.current_tier == 1 + 3

    def test_window_clears_after_increment(self) -> None:
        cb = self._make(target=0.5, window=4, rate=1)
        # First window — triggers increment
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])
        assert cb.current_tier == 2
        # Second window — should not increment again with only 1 success
        _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        assert cb.current_tier == 2

    def test_no_episode_key_ignored(self) -> None:
        cb = self._make(target=0.5, window=4)
        # Steps without episode key do not fill the window
        for _ in range(10):
            _call_on_step(cb, [{"tick": 1}])
        assert cb.current_tier == 1

    # -- set_config on envs --

    def test_calls_set_config_on_wrapped_envs(self) -> None:
        cb = self._make(target=0.5, window=4, rate=1)
        mock_unwrapped = MagicMock()
        mock_unwrapped.set_config = MagicMock()
        mock_env = MagicMock()
        mock_env.unwrapped = mock_unwrapped
        cb.training_env.envs = [mock_env]

        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
        for _ in range(2):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])

        mock_unwrapped.set_config.assert_called_once_with({"task": {"max_tier": 2}})

    def test_missing_envs_attr_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = self._make(target=0.5, window=4)
        del cb.training_env.envs  # Remove .envs attribute
        cb.training_env.__class__.__name__ = "MockVecEnv"

        with caplog.at_level("WARNING"):
            for _ in range(2):
                _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
            for _ in range(2):
                _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: False}])

        assert "does not expose .envs" in caplog.text

    def test_env_without_set_config_is_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = self._make(target=0.5, window=2)

        class BareEnv:
            pass

        cb.training_env.envs = [BareEnv()]

        with caplog.at_level("DEBUG"):
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])
            _call_on_step(cb, [{_EPISODE_KEY: {}, _TASK_SUCCESS_KEY: True}])

        assert cb.current_tier == 2
        assert "does not support set_config" in caplog.text

    def test_on_step_returns_true(self) -> None:
        cb = self._make()
        result = _call_on_step(cb, [])
        assert result is True


# ---------------------------------------------------------------------------
# ForgeMetricsCallback
# ---------------------------------------------------------------------------


class TestForgeMetricsCallback:
    """Tests for ForgeMetricsCallback."""

    def _make(
        self,
        log_freq: int = 1,
        forge_logger: Any = None,
    ) -> ForgeMetricsCallback:
        cb = ForgeMetricsCallback(forge_logger=forge_logger, log_freq=log_freq)
        cb.num_timesteps = 0
        cb._last_log_step = 0
        return cb

    # -- construction --

    def test_invalid_log_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="log_freq"):
            ForgeMetricsCallback(log_freq=0)

    # -- metric recording --

    def test_records_episode_return_and_length(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 5.0, "l": 10}}]
        _call_on_step(cb, infos)
        logged = mock_logger.log.call_args[0][0]
        assert "episode/return" in logged
        assert logged["episode/return"] == pytest.approx(5.0)
        assert "episode/length" in logged
        assert logged["episode/length"] == pytest.approx(10.0)

    def test_records_task_progress_mean(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 0.0, "l": 1}, "task_progress": [0.5, 1.0]}]
        _call_on_step(cb, infos)
        logged = mock_logger.log.call_args[0][0]
        assert "episode/task_progress" in logged
        assert logged["episode/task_progress"] == pytest.approx(0.75)

    def test_records_scalar_task_progress(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 0.0, "l": 1}, "task_progress": 0.5}]
        _call_on_step(cb, infos)
        logged = mock_logger.log.call_args[0][0]
        assert logged["episode/task_progress"] == pytest.approx(0.5)

    def test_records_empty_task_progress_as_zero(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 0.0, "l": 1}, "task_progress": []}]
        _call_on_step(cb, infos)
        logged = mock_logger.log.call_args[0][0]
        assert logged["episode/task_progress"] == pytest.approx(0.0)

    def test_records_task_success(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 1.0, "l": 5}, "task_success": True}]
        _call_on_step(cb, infos)
        logged = mock_logger.log.call_args[0][0]
        assert logged["episode/task_success"] == pytest.approx(1.0)

    def test_skips_non_terminal_steps(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        _call_on_step(cb, [{"tick": 0}])
        # log_freq=1 so flush is triggered, but no metrics buffered
        # logger.log should not have been called with episode metrics
        if mock_logger.log.called:
            logged = mock_logger.log.call_args[0][0]
            assert "episode/return" not in logged

    def test_log_freq_batches_calls(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=5, forge_logger=mock_logger)
        for _ in range(4):
            _call_on_step(cb, [{"episode": {"r": 1.0, "l": 10}}])
        assert mock_logger.log.call_count == 0

        _call_on_step(cb, [{"episode": {"r": 1.0, "l": 10}}])
        assert mock_logger.log.call_count == 1

    def test_no_logger_uses_python_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = self._make(log_freq=1, forge_logger=None)
        cb.verbose = 1
        with caplog.at_level("INFO"):
            _call_on_step(cb, [{"episode": {"r": 2.0, "l": 20}}])
        # Nothing should raise; message may or may not appear depending on logger hierarchy

    def test_on_training_end_closes_logger(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1000, forge_logger=mock_logger)
        cb._on_training_end()
        mock_logger.close.assert_called_once()

    def test_flush_without_pending_metrics_is_noop(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1000, forge_logger=mock_logger)
        cb._flush()
        mock_logger.log.assert_not_called()

    def test_on_step_returns_true(self) -> None:
        cb = self._make()
        result = _call_on_step(cb, [])
        assert result is True

    # -- step id passed to logger --

    def test_step_passed_to_logger(self) -> None:
        mock_logger = MagicMock()
        cb = self._make(log_freq=1, forge_logger=mock_logger)
        infos = [{"episode": {"r": 3.0, "l": 15}}]
        _call_on_step(cb, infos)
        _, kwargs = mock_logger.log.call_args
        assert "step" in kwargs


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


def test_require_sb3_raises_without_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge_env.sb3_callbacks as cb_mod  # noqa: PLC0415

    monkeypatch.setattr(cb_mod, "HAS_SB3", False)
    with pytest.raises(ImportError, match="Stable Baselines"):
        cb_mod._require_sb3()
