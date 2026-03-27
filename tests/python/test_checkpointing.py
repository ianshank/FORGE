"""Tests for checkpoint management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from forge.training.checkpointing import CheckpointManager


@pytest.fixture()
def mock_agent() -> MagicMock:
    """Create a mock agent with save/load methods and step_count."""
    agent = MagicMock()
    agent.step_count = 42

    def _save_side_effect(path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")

    agent.save.side_effect = _save_side_effect
    return agent


@pytest.fixture()
def manager(tmp_path):
    """Create a CheckpointManager with a temporary directory."""
    return CheckpointManager(str(tmp_path / "checkpoints"), max_checkpoints=5)


class TestSaveCreatesFiles:
    """Saving a checkpoint should create both agent state and metadata files."""

    def test_save_creates_files(self, manager, mock_agent) -> None:
        manager.save(mock_agent, episode=1, metrics={"reward": 10.0})

        checkpoints = list(Path(manager.checkpoint_dir).iterdir())
        assert len(checkpoints) == 1

        checkpoint_dir = checkpoints[0]
        assert (checkpoint_dir / "agent.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()


class TestLoadLatest:
    """Loading the latest checkpoint should restore the most recent save."""

    def test_load_latest(self, manager, mock_agent) -> None:
        manager.save(mock_agent, episode=1, metrics={"reward": 1.0})
        manager.save(mock_agent, episode=5, metrics={"reward": 5.0})

        result = manager.load_latest(mock_agent)

        assert isinstance(result, dict)
        assert result["episode"] == 5
        mock_agent.load.assert_called_once()
        load_path = mock_agent.load.call_args[0][0]
        assert "agent.json" in load_path


class TestListCheckpoints:
    """Listing checkpoints should return metadata sorted by episode."""

    def test_list_checkpoints(self, manager, mock_agent) -> None:
        manager.save(mock_agent, episode=3, metrics={"reward": 3.0})
        manager.save(mock_agent, episode=1, metrics={"reward": 1.0})
        manager.save(mock_agent, episode=7, metrics={"reward": 7.0})

        result = manager.list_checkpoints()

        assert len(result) == 3
        episodes = [c["episode"] for c in result]
        assert episodes == [1, 3, 7]


class TestRotateCheckpoints:
    """Old checkpoints should be removed when exceeding max_checkpoints."""

    def test_rotate_checkpoints(self, tmp_path, mock_agent) -> None:
        mgr = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoints=2)

        mgr.save(mock_agent, episode=1, metrics={})
        mgr.save(mock_agent, episode=2, metrics={})
        mgr.save(mock_agent, episode=3, metrics={})

        remaining = mgr.list_checkpoints()
        assert len(remaining) <= 2
        episodes = [c["episode"] for c in remaining]
        assert 1 not in episodes


class TestSaveWithMetrics:
    """Saved metrics should be persisted in metadata."""

    def test_save_with_metrics(self, manager, mock_agent) -> None:
        metrics = {"reward": 42.5, "loss": 0.01, "epsilon": 0.1}
        manager.save(mock_agent, episode=10, metrics=metrics)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        stored_metrics = checkpoints[0]["metrics"]
        assert stored_metrics["reward"] == 42.5
        assert stored_metrics["loss"] == 0.01
        assert stored_metrics["epsilon"] == 0.1


class TestLoadLatestEmptyDir:
    """Loading from an empty directory should return None."""

    def test_load_latest_empty_dir(self, manager, mock_agent) -> None:
        result = manager.load_latest(mock_agent)
        assert result is None
        mock_agent.load.assert_not_called()


class TestMaxCheckpointsRespected:
    """Only max_checkpoints checkpoint directories should be kept."""

    def test_max_checkpoints_respected(self, tmp_path, mock_agent) -> None:
        mgr = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoints=3)

        for ep in range(10):
            mgr.save(mock_agent, episode=ep, metrics={"ep": float(ep)})

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 3
        # The most recent episodes should survive
        episodes = [c["episode"] for c in remaining]
        assert all(ep >= 7 for ep in episodes)
