"""test_replay.py — Unit tests for forge_env.replay and RecordEpisodeWrapper.

Tests use mock environments so they run without the Rust native build.

Coverage targets:
  - RecordEpisodeWrapper: record lifecycle, file output, schema validation
  - load_replay: round-trip, version mismatch warning, missing file, bad JSON
  - play_replay: renders without error, respects start_frame
  - export_gif: raises ImportError when Pillow absent, ValueError when no obs
  - CLI: --help, good path, bad path, --export-gif
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────
_PYTHON_DIR = Path(__file__).parent.parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from forge_env.replay import (  # noqa: E402
    ReplayData,
    _cli,
    export_gif,
    load_replay,
    play_replay,
)
from forge_env.wrappers import RecordEpisodeWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_env(
    obs_shape: tuple[int, ...] = (4,),
    n_actions: int = 4,
    ep_length: int = 5,
) -> MagicMock:
    """Minimal mock Gymnasium-style environment."""
    env = MagicMock()
    obs = np.zeros(obs_shape, dtype=np.float32)
    env.observation_space.shape = obs_shape
    env.action_space.n = n_actions
    env.reset.return_value = (obs, {})
    # Terminate on the last step
    responses: list[tuple[np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, dict[str, Any]]] = [
        (obs, float(i), False, False, {}) for i in range(ep_length - 1)
    ]
    responses.append((obs, 1.0, True, False, {}))  # terminated on last
    env.step.side_effect = responses
    return env


def _make_replay_data(**overrides: Any) -> ReplayData:
    defaults: dict[str, Any] = dict(
        forge_version="0.2.0",
        format_version=1,
        seed=42,
        config={"world": {"width": 4, "height": 4}},
        actions=[0, 1, 2, 3, 0],
        rewards=[0.0, 1.0, -0.5, 0.0, 1.0],
        terminated_at=5,
        observations=[[0.0] * 16] * 6,
        timestamps_ms=[0.0, 16.0, 32.0, 48.0, 64.0],
    )
    defaults.update(overrides)
    return ReplayData(**defaults)


# ---------------------------------------------------------------------------
# RecordEpisodeWrapper
# ---------------------------------------------------------------------------


class TestRecordEpisodeWrapper:
    def test_file_created_on_episode_end(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=5)
        wrapper = RecordEpisodeWrapper(env, out, seed=42)
        wrapper.reset(seed=42)
        done = False
        while not done:
            _, _, terminated, truncated, _ = wrapper.step(0)
            done = terminated or truncated
        assert out.exists(), "replay file should exist after episode ends"

    def test_forge_schema_valid(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out, seed=7, config={"test": True})
        wrapper.reset(seed=7)
        for _ in range(3):
            wrapper.step(1)
        data = json.loads(out.read_text())
        assert "forge_version" in data
        assert data["format_version"] == 1  # noqa: PLR2004
        assert data["seed"] == 7  # noqa: PLR2004
        assert data["config"] == {"test": True}
        assert len(data["actions"]) == 3  # noqa: PLR2004
        assert len(data["rewards"]) == 3  # noqa: PLR2004

    def test_observations_stored_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out)
        wrapper.reset()
        for _ in range(3):
            wrapper.step(0)
        data = json.loads(out.read_text())
        assert "observations" in data
        assert len(data["observations"]) == 4  # reset obs + 3 step obs  # noqa: PLR2004

    def test_observations_omitted_when_disabled(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out, store_observations=False)
        wrapper.reset()
        for _ in range(3):
            wrapper.step(0)
        data = json.loads(out.read_text())
        assert "observations" not in data

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "ep.forge"
        env = _make_mock_env(ep_length=2)
        wrapper = RecordEpisodeWrapper(env, out)
        wrapper.reset()
        for _ in range(2):
            wrapper.step(0)
        assert out.exists()

    def test_reset_clears_accumulators(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out)
        # First episode
        wrapper.reset(seed=1)
        for _ in range(3):
            wrapper.step(0)
        env.step.side_effect = [
            (np.zeros(4, dtype=np.float32), 0.5, False, False, {}),
            (np.zeros(4, dtype=np.float32), 1.0, True, False, {}),
        ]
        # Second episode — accumulators must have been cleared
        wrapper.reset(seed=99)
        for _ in range(2):
            wrapper.step(1)
        data = json.loads(out.read_text())
        assert len(data["actions"]) == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# load_replay
# ---------------------------------------------------------------------------


class TestLoadReplay:
    def test_round_trip(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=5)
        wrapper = RecordEpisodeWrapper(env, out, seed=42)
        wrapper.reset(seed=42)
        for _ in range(5):
            wrapper.step(2)
        data = load_replay(out)
        assert isinstance(data, ReplayData)
        assert data.seed == 42  # noqa: PLR2004
        assert data.num_steps == 5  # noqa: PLR2004

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_replay(tmp_path / "nonexistent.forge")

    def test_bad_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.forge"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_replay(bad)

    def test_unsupported_format_version_raises(self, tmp_path: Path) -> None:
        future = tmp_path / "future.forge"
        future.write_text(json.dumps({"format_version": 999}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            load_replay(future)

    def test_version_mismatch_warns(self, tmp_path: Path) -> None:
        replay = tmp_path / "old.forge"
        replay.write_text(
            json.dumps({
                "forge_version": "0.0.1",
                "format_version": 1,
                "seed": None,
                "config": {},
                "actions": [],
                "rewards": [],
                "terminated_at": 0,
            }),
            encoding="utf-8",
        )
        with patch("forge_env.replay.logger") as mock_log:
            load_replay(replay)
            mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# play_replay
# ---------------------------------------------------------------------------


class TestPlayReplay:
    def test_renders_without_error(self) -> None:
        import io  # noqa: PLC0415
        import re  # noqa: PLC0415
        buf = io.StringIO()
        data = _make_replay_data()
        play_replay(data, fps=1000.0, stream=buf)
        out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", buf.getvalue())
        assert "Replay complete" in out
        assert "Step 1/5" in out

    def test_start_frame_skips_earlier_frames(self) -> None:
        import io  # noqa: PLC0415
        import re  # noqa: PLC0415
        buf = io.StringIO()
        data = _make_replay_data(actions=[0, 1, 2], rewards=[0.0, 0.0, 1.0])
        play_replay(data, fps=1000.0, start_frame=2, stream=buf)
        out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", buf.getvalue())
        assert "Step 3/3" in out
        assert "Step 1" not in out
        assert "Step 2" not in out



# ---------------------------------------------------------------------------
# export_gif
# ---------------------------------------------------------------------------


class TestExportGif:
    def test_raises_importerror_without_pillow(self, tmp_path: Path) -> None:
        data = _make_replay_data()
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ImportError, match="Pillow"):
                export_gif(data, tmp_path / "out.gif")

    def test_raises_valueerror_without_observations(self, tmp_path: Path) -> None:
        data = _make_replay_data(observations=[])
        with pytest.raises(ValueError, match="observations"):
            export_gif(data, tmp_path / "out.gif")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _cli(["--help"])
        assert exc.value.code == 0

    def test_missing_file_returns_1(self) -> None:
        rc = _cli(["/no/such/file.forge"])
        assert rc == 1  # noqa: PLR2004

    def test_good_file_returns_0(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out, seed=0)
        wrapper.reset()
        for _ in range(3):
            wrapper.step(0)
        with patch("forge_env.replay.play_replay"):
            rc = _cli([str(out), "--fps", "1000"])
        assert rc == 0

    def test_export_gif_path_passed_through(self, tmp_path: Path) -> None:
        out = tmp_path / "ep.forge"
        env = _make_mock_env(ep_length=3)
        wrapper = RecordEpisodeWrapper(env, out, seed=0)
        wrapper.reset()
        for _ in range(3):
            wrapper.step(0)
        gif_out = tmp_path / "out.gif"
        with patch("forge_env.replay.export_gif") as mock_gif:
            _cli([str(out), "--export-gif", str(gif_out)])
        mock_gif.assert_called_once()
        # Verify out_path kwarg was passed correctly (Path or str)
        call_kwargs = mock_gif.call_args[1]
        assert Path(call_kwargs["out_path"]) == gif_out
