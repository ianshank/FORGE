"""Always-on unit tests for ``scripts/_e2e_progress.py``.

The orchestrator pulls these helpers off ``sys.path`` at runtime; pinning
their contract here keeps the 85% coverage gate honest and makes the
"checkpoint survives a crash" property machine-checkable rather than a
prayer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

# scripts/ is placed on sys.path by the root conftest.py (_ensure_importable),
# matching how run_e2e_long.py is imported at runtime.
from _e2e_progress import ProgressState, load, save

if TYPE_CHECKING:
    from pathlib import Path


def _state(**overrides: object) -> ProgressState:
    base: dict[str, object] = {
        "run_id": "r-001",
        "episodes_completed": 5,
        "scenario_cursor": 2,
        "last_seed": 42,
    }
    base.update(overrides)
    return ProgressState(
        run_id=str(base["run_id"]),
        episodes_completed=int(cast("int", base["episodes_completed"])),
        scenario_cursor=int(cast("int", base["scenario_cursor"])),
        last_seed=int(cast("int", base["last_seed"])),
    )


def test_progress_load_missing_returns_none(tmp_path: Path) -> None:
    path = tmp_path / ".e2e_progress.json"
    assert load(path) is None, "first run must return None, not raise"


def test_progress_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / ".e2e_progress.json"
    original = _state(run_id="abc-123", episodes_completed=7, scenario_cursor=3, last_seed=99)
    save(path, original)
    assert load(path) == original


def test_progress_save_creates_parent_directory(tmp_path: Path) -> None:
    # Resume support assumes the orchestrator can drop the checkpoint inside
    # an output dir that may not exist yet (clean CI environment).
    path = tmp_path / "nested" / "dir" / ".e2e_progress.json"
    save(path, _state())
    assert path.exists()


def test_progress_save_is_atomic_via_tempfile_replace(tmp_path: Path) -> None:
    # Saving the same checkpoint twice must overwrite cleanly with no orphan
    # tempfile left behind. Anything starting with ``.e2e_progress.`` and
    # ending in ``.tmp`` would be a leaked half-write.
    path = tmp_path / ".e2e_progress.json"
    save(path, _state(episodes_completed=1))
    save(path, _state(episodes_completed=2))
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".e2e_progress.") and p.name.endswith(".tmp")]
    assert leftovers == [], f"orphan tempfile(s) leaked: {leftovers}"
    assert load(path) == _state(episodes_completed=2)


def test_progress_load_raises_on_corrupted_json(tmp_path: Path) -> None:
    # A truncated/corrupted checkpoint must be loud — silently restarting
    # would obscure data-loss bugs.
    path = tmp_path / ".e2e_progress.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load(path)


def test_progress_load_preserves_field_types(tmp_path: Path) -> None:
    # Hand-rolled JSON to exercise the int-coercion paths in load().
    path = tmp_path / ".e2e_progress.json"
    payload = {"run_id": "rA", "episodes_completed": "11", "scenario_cursor": "3", "last_seed": "7"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    state = load(path)
    assert state is not None
    assert state.run_id == "rA"
    assert state.episodes_completed == 11
    assert state.scenario_cursor == 3
    assert state.last_seed == 7
