"""Tests for forge_env.datasets offline data utilities.

Covers: ForgeDataset, ForgeStep, load_forge_jsonl, load_minerl,
load_maze_jsonl, _parse_forge_jsonl_step, _minerl_action_to_discrete,
and the module-level constants.  Does NOT require minari (skips those tests).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from forge_env.datasets import (
    ForgeDataset,
    ForgeStep,
    _FORGE_CRAFT_BASE,
    _MAZE_DIR_TO_ACTION,
    _MAZE_GOAL_REWARD,
    _MAZE_STEP_REWARD,
    _MINERL_BOOL_ACTIONS,
    _MINERL_CRAFT_MAP,
    _parse_forge_jsonl_step,
    _minerl_action_to_discrete,
    load_forge_jsonl,
    load_maze_jsonl,
    load_minerl,
    load_minari,
    _DEFAULT_INV_SLOTS,
    _DEFAULT_VIEW_SIZE,
    _MAX_STACK_SIZE,
    _MINECRAFT_MAX_HEALTH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIEW = _DEFAULT_VIEW_SIZE


def _make_step(
    action: int = 0,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    health: float = 1.0,
) -> ForgeStep:
    return ForgeStep(
        grid_view=np.zeros((_VIEW, _VIEW, 7), dtype=np.uint8),
        inventory=np.zeros((_DEFAULT_INV_SLOTS, 2), dtype=np.uint16),
        health=health,
        stamina=1.0,
        position=(0, 0),
        day_phase=1,
        task_progress=np.zeros(1, dtype=np.float32),
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def _write_forge_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "steps.jsonl"
    with p.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


def _write_minerl_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "minerl.jsonl"
    with p.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


def _write_maze_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "maze.jsonl"
    with p.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


# ---------------------------------------------------------------------------
# Module-level constant sanity checks
# ---------------------------------------------------------------------------


def test_constants_are_positive() -> None:
    assert _DEFAULT_VIEW_SIZE > 0
    assert _DEFAULT_INV_SLOTS > 0
    assert _MAX_STACK_SIZE > 0
    assert _MINECRAFT_MAX_HEALTH > 0
    assert _MAZE_STEP_REWARD > 0
    assert _MAZE_GOAL_REWARD > _MAZE_STEP_REWARD


def test_maze_dir_to_action_completeness() -> None:
    for ch in ("U", "D", "L", "R"):
        assert ch in _MAZE_DIR_TO_ACTION
        assert _MAZE_DIR_TO_ACTION[ch] > 0


def test_craft_map_no_substring_collision() -> None:
    # "axe" must not match before "pickaxe" — longer patterns first
    fragments = [frag for frag, _ in _MINERL_CRAFT_MAP]
    axe_idx = fragments.index("axe")
    pickaxe_idx = fragments.index("pickaxe")
    assert pickaxe_idx < axe_idx, "pickaxe must precede axe to avoid substring collision"


def test_forge_craft_base_positive() -> None:
    assert _FORGE_CRAFT_BASE > 0


# ---------------------------------------------------------------------------
# ForgeStep
# ---------------------------------------------------------------------------


def test_forge_step_fields() -> None:
    s = _make_step(action=5, reward=2.5, terminated=True)
    assert s.action == 5
    assert s.reward == pytest.approx(2.5)
    assert s.terminated is True
    assert s.truncated is False
    assert s.grid_view.shape == (_VIEW, _VIEW, 7)
    assert s.inventory.shape == (_DEFAULT_INV_SLOTS, 2)


# ---------------------------------------------------------------------------
# ForgeDataset
# ---------------------------------------------------------------------------


def test_dataset_empty() -> None:
    ds = ForgeDataset([], source="test")
    assert len(ds) == 0
    obs, acts, rews, dones = ds.to_arrays()
    assert obs.shape == (0, 1)
    assert acts.shape == (0,)
    assert rews.shape == (0,)
    assert dones.shape == (0,)


def test_dataset_len_and_getitem() -> None:
    steps = [_make_step(action=i) for i in range(5)]
    ds = ForgeDataset(steps)
    assert len(ds) == 5
    assert ds[2].action == 2


def test_dataset_iter() -> None:
    steps = [_make_step(action=i) for i in range(3)]
    ds = ForgeDataset(steps)
    actions = [s.action for s in ds]
    assert actions == [0, 1, 2]


def test_dataset_repr() -> None:
    ds = ForgeDataset([_make_step()], source="mysource")
    assert "mysource" in repr(ds)
    assert "1" in repr(ds)


def test_dataset_to_arrays_shape() -> None:
    steps = [_make_step(action=i, reward=float(i), terminated=(i == 2)) for i in range(3)]
    ds = ForgeDataset(steps)
    obs, acts, rews, dones = ds.to_arrays()
    assert obs.shape[0] == 3
    assert acts.shape == (3,)
    assert rews.shape == (3,)
    assert dones.shape == (3,)


def test_dataset_to_arrays_values() -> None:
    steps = [
        _make_step(action=7, reward=3.0, terminated=True),
        _make_step(action=3, reward=0.5, truncated=True),
    ]
    ds = ForgeDataset(steps)
    obs, acts, rews, dones = ds.to_arrays()
    assert acts[0] == 7
    assert acts[1] == 3
    assert rews[0] == pytest.approx(3.0)
    assert rews[1] == pytest.approx(0.5)
    assert dones[0] is np.bool_(True)
    assert dones[1] is np.bool_(True)


def test_dataset_to_arrays_obs_dtype() -> None:
    ds = ForgeDataset([_make_step()])
    obs, acts, rews, dones = ds.to_arrays()
    assert obs.dtype == np.float32
    assert acts.dtype == np.int32
    assert rews.dtype == np.float32


def test_flatten_obs_normalisation() -> None:
    # grid_view all 255 → should produce 1.0 in obs
    s = _make_step()
    s.grid_view[:] = 255
    flat = ForgeDataset._flatten_obs(s)
    grid_part = flat[: _VIEW * _VIEW * 7]
    assert np.allclose(grid_part, 1.0)


def test_flatten_obs_inventory_normalisation() -> None:
    s = _make_step()
    s.inventory[:] = _MAX_STACK_SIZE
    flat = ForgeDataset._flatten_obs(s)
    inv_start = _VIEW * _VIEW * 7
    inv_end = inv_start + _DEFAULT_INV_SLOTS * 2
    assert np.allclose(flat[inv_start:inv_end], 1.0)


# ---------------------------------------------------------------------------
# _parse_forge_jsonl_step
# ---------------------------------------------------------------------------


def test_parse_step_basic() -> None:
    raw = {
        "observations": [{"health": 0.8, "stamina": 0.5, "position": [3, 4], "day_phase": 2}],
        "actions": [5],
        "rewards": [2.0],
        "terminated": True,
        "truncated": False,
    }
    step = _parse_forge_jsonl_step(raw, tick=0)
    assert step is not None
    assert step.health == pytest.approx(0.8)
    assert step.stamina == pytest.approx(0.5)
    assert step.position == (3, 4)
    assert step.day_phase == 2
    assert step.action == 5
    assert step.reward == pytest.approx(2.0)
    assert step.terminated is True


def test_parse_step_defaults() -> None:
    step = _parse_forge_jsonl_step({}, tick=0)
    assert step is not None
    assert step.health == pytest.approx(1.0)
    assert step.action == 0
    assert step.reward == pytest.approx(0.0)
    assert step.terminated is False


def test_parse_step_grid_view_dict_tiles() -> None:
    tiles = [{"terrain": 1, "has_agent": True, "elevation": 2, "object_type": 0,
               "resource_type": 0, "has_object": False, "has_resource": False}
             for _ in range(121)]
    raw = {"observations": [{"grid_view": tiles, "view_width": 11, "view_height": 11}]}
    step = _parse_forge_jsonl_step(raw, tick=0)
    assert step is not None
    assert step.grid_view.shape == (11, 11, 7)
    assert step.grid_view[0, 0, 0] == 1


def test_parse_step_grid_view_list_tiles() -> None:
    # Each tile is a 7-element list (all uint8-safe values 0-6)
    tiles = [[j for j in range(7)] for _ in range(121)]
    raw = {"observations": [{"grid_view": tiles, "view_width": 11, "view_height": 11}]}
    step = _parse_forge_jsonl_step(raw, tick=0)
    assert step is not None
    assert step.grid_view.shape == (11, 11, 7)
    assert step.grid_view[0, 0, 3] == 3


def test_parse_step_inventory() -> None:
    raw = {"observations": [{"inventory": {"slots": [[10, 5], [3, 1]]}}]}
    step = _parse_forge_jsonl_step(raw, tick=0)
    assert step is not None
    assert step.inventory.shape == (_DEFAULT_INV_SLOTS, 2)
    assert step.inventory[0, 0] == 10
    assert step.inventory[0, 1] == 5


def test_parse_step_task_progress() -> None:
    raw = {"observations": [{"task_progress": [0.5, 0.8]}]}
    step = _parse_forge_jsonl_step(raw, tick=0)
    assert step is not None
    assert len(step.task_progress) == 2
    assert step.task_progress[0] == pytest.approx(0.5)


def test_parse_step_malformed_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="forge_env.datasets"):
        step = _parse_forge_jsonl_step({"observations": "not a list"}, tick=7)
    assert step is None


# ---------------------------------------------------------------------------
# load_forge_jsonl
# ---------------------------------------------------------------------------


def test_load_forge_jsonl_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    ds = load_forge_jsonl(str(p))
    assert len(ds) == 0
    assert ds.source.startswith("forge-jsonl:")


def test_load_forge_jsonl_with_steps(tmp_path: Path) -> None:
    rows = [
        {"observations": [{"health": 0.9}], "actions": [1], "rewards": [0.5]},
        {"observations": [{"health": 0.7}], "actions": [2], "rewards": [1.0], "terminated": True},
    ]
    p = _write_forge_jsonl(tmp_path, rows)
    ds = load_forge_jsonl(str(p))
    assert len(ds) == 2
    assert ds[0].action == 1
    assert ds[1].action == 2
    assert ds[1].terminated is True


def test_load_forge_jsonl_max_steps(tmp_path: Path) -> None:
    rows = [{"actions": [i]} for i in range(10)]
    p = _write_forge_jsonl(tmp_path, rows)
    ds = load_forge_jsonl(str(p), max_steps=3)
    assert len(ds) == 3


def test_load_forge_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "steps.jsonl"
    p.write_text('\n{"actions": [1]}\n\n{"actions": [2]}\n')
    ds = load_forge_jsonl(str(p))
    assert len(ds) == 2


def test_load_forge_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "steps.jsonl"
    p.write_text('{"actions": [1]}\nnot-json\n{"actions": [2]}\n')
    # json.loads will raise on malformed line -> propagates as JSONDecodeError
    with pytest.raises(Exception):  # noqa: B017
        load_forge_jsonl(str(p))


# ---------------------------------------------------------------------------
# _minerl_action_to_discrete
# ---------------------------------------------------------------------------


def test_minerl_noop() -> None:
    assert _minerl_action_to_discrete({"no_op": True}) == 0
    assert _minerl_action_to_discrete({}) == 0


def test_minerl_movement_actions() -> None:
    assert _minerl_action_to_discrete({"forward": True}) == 1
    assert _minerl_action_to_discrete({"back": True}) == 2
    assert _minerl_action_to_discrete({"left": True}) == 3
    assert _minerl_action_to_discrete({"right": True}) == 4


def test_minerl_attack_and_interact() -> None:
    assert _minerl_action_to_discrete({"attack": True}) == 16
    assert _minerl_action_to_discrete({"use": True}) == 39


def test_minerl_pickup() -> None:
    assert _minerl_action_to_discrete({"pickup": True}) == 5


def test_minerl_craft_pickaxe_before_axe() -> None:
    # "stone_pickaxe" contains "axe" — must resolve to pickaxe, not axe
    result = _minerl_action_to_discrete({"craft": "stone_pickaxe"})
    assert result == _FORGE_CRAFT_BASE + 1, "pickaxe recipe must be index 1"


def test_minerl_craft_axe() -> None:
    result = _minerl_action_to_discrete({"craft": "wooden_axe"})
    assert result == _FORGE_CRAFT_BASE + 0, "axe recipe must be index 0"


def test_minerl_craft_plank() -> None:
    assert _minerl_action_to_discrete({"craft": "oak_plank"}) == _FORGE_CRAFT_BASE + 2


def test_minerl_craft_unknown() -> None:
    assert _minerl_action_to_discrete({"craft": "unknown_item_xyz"}) == 0


def test_minerl_all_craft_types() -> None:
    # Verify every entry in _MINERL_CRAFT_MAP resolves correctly
    for fragment, recipe_idx in _MINERL_CRAFT_MAP:
        result = _minerl_action_to_discrete({"craft": fragment})
        assert result == _FORGE_CRAFT_BASE + recipe_idx, (
            f"craft '{fragment}' expected offset {recipe_idx}, got {result - _FORGE_CRAFT_BASE}"
        )


def test_minerl_bool_actions_covered() -> None:
    # Every bool action in the lookup table must map correctly
    for key, expected_id in _MINERL_BOOL_ACTIONS:
        result = _minerl_action_to_discrete({key: True})
        assert result == expected_id, f"key '{key}' expected {expected_id}, got {result}"


# ---------------------------------------------------------------------------
# load_minerl
# ---------------------------------------------------------------------------


def test_load_minerl_basic(tmp_path: Path) -> None:
    rows = [
        {"action": {"forward": True}, "reward": 0.5, "terminated": False},
        {"action": {"craft": "stone_pickaxe"}, "reward": 1.0, "terminated": True},
    ]
    p = _write_minerl_jsonl(tmp_path, rows)
    ds = load_minerl(str(p))
    assert len(ds) == 2
    assert ds[0].action == 1  # Move(Up)
    assert ds[1].action == _FORGE_CRAFT_BASE + 1  # Craft(pickaxe)
    assert ds.source.startswith("minerl:")


def test_load_minerl_health_normalisation(tmp_path: Path) -> None:
    rows = [{"obs": {"health": 10.0, "position": [5.0, 0.0, 8.0]}, "reward": 0.0}]
    p = _write_minerl_jsonl(tmp_path, rows)
    ds = load_minerl(str(p))
    assert ds[0].health == pytest.approx(10.0 / _MINECRAFT_MAX_HEALTH)


def test_load_minerl_position_xz(tmp_path: Path) -> None:
    rows = [{"obs": {"position": [3.0, 100.0, 7.0]}, "reward": 0.0}]
    p = _write_minerl_jsonl(tmp_path, rows)
    ds = load_minerl(str(p))
    assert ds[0].position == (3, 7)


def test_load_minerl_max_steps(tmp_path: Path) -> None:
    rows = [{"action": {}, "reward": 0.0} for _ in range(20)]
    p = _write_minerl_jsonl(tmp_path, rows)
    ds = load_minerl(str(p), max_steps=5)
    assert len(ds) == 5


def test_load_minerl_max_episodes(tmp_path: Path) -> None:
    rows = []
    for _ in range(3):
        rows.extend([
            {"action": {}, "reward": 0.0, "terminated": False},
            {"action": {}, "reward": 1.0, "terminated": True},
        ])
    p = _write_minerl_jsonl(tmp_path, rows)
    ds = load_minerl(str(p), max_episodes=2)
    # 2 episodes x 2 steps each
    assert len(ds) == 4


def test_load_minerl_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    ds = load_minerl(str(p))
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# load_maze_jsonl
# ---------------------------------------------------------------------------


def test_load_maze_basic(tmp_path: Path) -> None:
    records = [{"maze": "# #", "solution": "R", "start": [0, 0], "end": [1, 0]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert len(ds) == 1
    assert ds[0].action == _MAZE_DIR_TO_ACTION["R"]
    assert ds[0].terminated is True
    assert ds[0].reward == pytest.approx(_MAZE_GOAL_REWARD)
    assert ds.source.startswith("maze-jsonl:")


def test_load_maze_multi_step(tmp_path: Path) -> None:
    records = [{"solution": "RRDD", "start": [0, 0]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert len(ds) == 4
    # Only the last step is terminal
    assert sum(s.terminated for s in ds) == 1
    assert ds[-1].terminated is True
    # Non-terminal steps get intermediate reward
    assert ds[0].reward == pytest.approx(_MAZE_STEP_REWARD)


def test_load_maze_direction_mapping(tmp_path: Path) -> None:
    records = [{"solution": "UDLR", "start": [2, 2]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert ds[0].action == _MAZE_DIR_TO_ACTION["U"]
    assert ds[1].action == _MAZE_DIR_TO_ACTION["D"]
    assert ds[2].action == _MAZE_DIR_TO_ACTION["L"]
    assert ds[3].action == _MAZE_DIR_TO_ACTION["R"]


def test_load_maze_max_mazes(tmp_path: Path) -> None:
    records = [{"solution": "R", "start": [0, 0]} for _ in range(5)]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p), max_mazes=3)
    assert len(ds) == 3


def test_load_maze_max_solution_length_filter(tmp_path: Path) -> None:
    records = [
        {"solution": "R", "start": [0, 0]},        # passes (len 1)
        {"solution": "RRRRRR", "start": [0, 0]},   # filtered (len 6 > 4)
    ]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p), max_solution_length=4)
    assert len(ds) == 1


def test_load_maze_position_tracking(tmp_path: Path) -> None:
    records = [{"solution": "RRD", "start": [1, 1]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert ds[0].position == (1, 1)
    assert ds[1].position == (2, 1)
    assert ds[2].position == (3, 1)


def test_load_maze_unknown_direction_maps_to_noop(tmp_path: Path) -> None:
    records = [{"solution": "X", "start": [0, 0]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert len(ds) == 1
    assert ds[0].action == 0  # unknown -> noop


def test_load_maze_task_progress_monotone(tmp_path: Path) -> None:
    records = [{"solution": "RRRR", "start": [0, 0]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    progress = [s.task_progress[0] for s in ds]
    assert progress == sorted(progress)
    assert progress[-1] == pytest.approx(1.0)


def test_load_maze_empty_solution_skipped(tmp_path: Path) -> None:
    records = [{"solution": "", "start": [0, 0]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    # Empty solution = 0 steps, maze counted but produces no steps
    assert len(ds) == 0


def test_load_maze_uppercase_normalisation(tmp_path: Path) -> None:
    records = [{"solution": "rdlu", "start": [2, 2]}]
    p = _write_maze_jsonl(tmp_path, records)
    ds = load_maze_jsonl(str(p))
    assert ds[0].action == _MAZE_DIR_TO_ACTION["R"]
    assert ds[1].action == _MAZE_DIR_TO_ACTION["D"]


def test_load_maze_blank_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "maze.jsonl"
    p.write_text('\n{"solution":"R","start":[0,0]}\n\n{"solution":"L","start":[1,0]}\n')
    ds = load_maze_jsonl(str(p))
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# load_minari — skip if package not available
# ---------------------------------------------------------------------------


def test_load_minari_raises_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge_env.datasets as _mod
    original = _mod._minari
    monkeypatch.setattr(_mod, "_minari", None)
    with pytest.raises(ImportError, match="minari"):
        load_minari("fake_dataset_id")
    monkeypatch.setattr(_mod, "_minari", original)
