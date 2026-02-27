"""test_callbacks.py — Unit tests for forge_env.callbacks.

Tests are pure-Python (no forge_env native module required) and cover:
- EpisodeStats construction and as_dict()
- ConsoleCallback output frequency
- CsvCallback header/row writing and flush behaviour
- WandbCallback graceful no-op when wandb is missing
- MLflowCallback graceful no-op when mlflow is missing
- CompositeCallback fan-out and exception isolation
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make forge_env importable from the repo's python/ directory
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).parent.parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from forge_env.callbacks import (
    CompositeCallback,
    ConsoleCallback,
    CsvCallback,
    EpisodeStats,
    LoggingCallback,
    MLflowCallback,
    WandbCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats(
    episode: int = 0,
    ep_len: int = 100,
    ep_return: float = 1.5,
    total_steps: int = 1000,
    fps: float = 60.0,
    **extra: object,
) -> EpisodeStats:
    return EpisodeStats(
        episode=episode,
        total_steps=total_steps,
        episode_length=ep_len,
        episode_return=ep_return,
        fps=fps,
        extra=dict(extra),
    )


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------


class TestEpisodeStats:
    def test_as_dict_contains_all_fields(self) -> None:
        s = _stats(episode=3, ep_len=50, ep_return=2.0, total_steps=500, fps=30.0)
        d = s.as_dict()
        assert d["episode"] == 3
        assert d["episode_length"] == 50
        assert d["episode_return"] == pytest.approx(2.0)
        assert d["total_steps"] == 500
        assert d["fps"] == pytest.approx(30.0)

    def test_extra_included_in_dict(self) -> None:
        s = _stats(entropy=0.7, loss=0.3)
        d = s.as_dict()
        assert d["entropy"] == pytest.approx(0.7)
        assert d["loss"] == pytest.approx(0.3)

    def test_frozen_immutable(self) -> None:
        s = _stats()
        with pytest.raises((AttributeError, TypeError)):
            s.episode = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConsoleCallback
# ---------------------------------------------------------------------------


class TestConsoleCallback:
    def test_logs_every_episode_by_default(self, capsys: pytest.CaptureFixture) -> None:
        cb = ConsoleCallback(log_every=1)
        for i in range(3):
            cb.on_episode_end(_stats(episode=i))
        out = capsys.readouterr().out
        assert out.count("[ep") == 3

    def test_logs_every_n_episodes(self, capsys: pytest.CaptureFixture) -> None:
        cb = ConsoleCallback(log_every=5)
        for i in range(10):
            cb.on_episode_end(_stats(episode=i))
        out = capsys.readouterr().out
        # Episodes 0 and 5 only
        assert out.count("[ep") == 2

    def test_log_every_zero_treated_as_one(self, capsys: pytest.CaptureFixture) -> None:
        """log_every=0 should not cause ZeroDivisionError."""
        cb = ConsoleCallback(log_every=0)
        cb.on_episode_end(_stats(episode=0))
        out = capsys.readouterr().out
        assert "[ep" in out


# ---------------------------------------------------------------------------
# CsvCallback
# ---------------------------------------------------------------------------


class TestCsvCallback:
    def test_creates_file_on_training_start(self, tmp_path: Path) -> None:
        cb = CsvCallback(output_path=tmp_path / "metrics.csv")
        cb.on_training_start()
        cb.on_episode_end(_stats())
        cb.on_training_end()
        assert (tmp_path / "metrics.csv").exists()

    def test_header_written(self, tmp_path: Path) -> None:
        cb = CsvCallback(output_path=tmp_path / "m.csv")
        cb.on_training_start()
        cb.on_episode_end(_stats())
        cb.on_training_end()
        with (tmp_path / "m.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert "episode" in rows[0]
        assert "episode_return" in rows[0]

    def test_multiple_rows(self, tmp_path: Path) -> None:
        cb = CsvCallback(output_path=tmp_path / "m.csv")
        cb.on_training_start()
        for i in range(5):
            cb.on_episode_end(_stats(episode=i, ep_return=float(i)))
        cb.on_training_end()
        with (tmp_path / "m.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 5
        assert float(rows[4]["episode_return"]) == pytest.approx(4.0)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "metrics.csv"
        cb = CsvCallback(output_path=deep)
        cb.on_training_start()
        cb.on_episode_end(_stats())
        cb.on_training_end()
        assert deep.exists()

    def test_no_training_start_still_works(self, tmp_path: Path) -> None:
        """If on_training_start is not called, file is created lazily."""
        cb = CsvCallback(output_path=tmp_path / "m.csv")
        cb.on_episode_end(_stats())
        cb.on_training_end()
        assert (tmp_path / "m.csv").exists()


# ---------------------------------------------------------------------------
# WandbCallback
# ---------------------------------------------------------------------------


class TestWandbCallback:
    def test_noop_when_wandb_missing(self) -> None:
        """Should warn and not raise when wandb is not installed."""
        with patch("forge_env.callbacks._HAS_WANDB", False):
            cb = WandbCallback()
            with pytest.warns(UserWarning, match="wandb is not installed"):
                cb.on_training_start()
            # Should be a no-op — no exceptions
            cb.on_episode_end(_stats())
            cb.on_training_end()

    def test_noop_when_disabled_env(self) -> None:
        with (
            patch("forge_env.callbacks._HAS_WANDB", True),
            patch.dict("os.environ", {"WANDB_MODE": "disabled"}),
        ):
            cb = WandbCallback()
            mock_wandb = MagicMock()
            with patch("forge_env.callbacks.wandb", mock_wandb):
                cb.on_training_start()
                # wandb.init should NOT have been called
                mock_wandb.init.assert_not_called()

    def test_logs_when_available(self) -> None:
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with (
            patch("forge_env.callbacks._HAS_WANDB", True),
            patch("forge_env.callbacks.wandb", mock_wandb),
            patch.dict("os.environ", {"WANDB_MODE": "online"}),
        ):
            cb = WandbCallback(project="test-proj")
            cb.on_training_start()
            mock_wandb.init.assert_called_once()
            cb.on_episode_end(_stats(episode=0))
            mock_wandb.log.assert_called_once()
            cb.on_training_end()
            mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# MLflowCallback
# ---------------------------------------------------------------------------


class TestMLflowCallback:
    def test_noop_when_mlflow_missing(self) -> None:
        with patch("forge_env.callbacks._HAS_MLFLOW", False):
            cb = MLflowCallback()
            with pytest.warns(UserWarning, match="mlflow is not installed"):
                cb.on_training_start()
            cb.on_episode_end(_stats())
            cb.on_training_end()

    def test_logs_when_available(self) -> None:
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_run.info.run_id = "test-run"

        with (
            patch("forge_env.callbacks._HAS_MLFLOW", True),
            patch("forge_env.callbacks.mlflow", mock_mlflow),
        ):
            cb = MLflowCallback(experiment_name="test-exp")
            cb.on_training_start()
            mock_mlflow.set_experiment.assert_called_with("test-exp")
            mock_mlflow.start_run.assert_called_once()
            cb.on_episode_end(_stats())
            mock_mlflow.log_metrics.assert_called_once()
            cb.on_training_end()
            mock_mlflow.end_run.assert_called_once()


# ---------------------------------------------------------------------------
# CompositeCallback
# ---------------------------------------------------------------------------


class TestCompositeCallback:
    def test_delegates_to_all(self) -> None:
        cb1 = MagicMock(spec=LoggingCallback)
        cb2 = MagicMock(spec=LoggingCallback)
        composite = CompositeCallback([cb1, cb2])
        s = _stats()
        composite.on_episode_end(s)
        cb1.on_episode_end.assert_called_once_with(s)
        cb2.on_episode_end.assert_called_once_with(s)

    def test_on_training_start_delegated(self) -> None:
        cb = MagicMock(spec=LoggingCallback)
        CompositeCallback([cb]).on_training_start()
        cb.on_training_start.assert_called_once()

    def test_on_training_end_delegated(self) -> None:
        cb = MagicMock(spec=LoggingCallback)
        CompositeCallback([cb]).on_training_end()
        cb.on_training_end.assert_called_once()

    def test_exception_isolation(self, capsys: pytest.CaptureFixture) -> None:
        """An exception in one callback should not prevent others from running."""
        bad_cb = MagicMock(spec=LoggingCallback)
        bad_cb.on_episode_end.side_effect = RuntimeError("boom")
        good_cb = MagicMock(spec=LoggingCallback)

        composite = CompositeCallback([bad_cb, good_cb])
        composite.on_episode_end(_stats())  # must not raise
        good_cb.on_episode_end.assert_called_once()

    def test_empty_composite_noop(self) -> None:
        CompositeCallback([]).on_episode_end(_stats())  # must not raise
