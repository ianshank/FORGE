"""Tests for MangoMAS FORGE scenario collection."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from forge.mangomas.collector import (
    collect_training_data_from_scenarios,
    decode_action_name,
    resolve_forge_scenarios,
    write_collection_report,
)
from forge.mangomas.config import MangoMASBridgeConfig


class _FakeEnv:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.action_space = SimpleNamespace(n=75)
        self._step = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step = 0
        return self._observation(), {"tick": 0, "tasks_completed": [[]]}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        terminated = self._step >= 3
        info = {"tick": self._step, "tasks_completed": [[1]] if terminated else [[]]}
        return self._observation(), 1.0, terminated, False, info

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        grid = np.zeros((11, 11, 7), dtype=np.uint8)
        grid[5, 5, 1] = 1
        return {
            "grid_view": grid,
            "inventory": np.zeros((10, 2), dtype=np.uint16),
            "health": 0.9,
            "stamina": 0.8,
            "position": (5, 5),
            "messages": np.zeros((0,), dtype=np.uint16),
            "day_phase": 1,
            "altitude": 0.25,
            "battery": 0.75,
            "morphology": 2.0,
            "heading": 0.5,
        }


def test_decode_action_name_covers_base_and_drone_actions() -> None:
    assert decode_action_name(0, 16, True) == "Noop"
    assert decode_action_name(1, 16, True) == "MoveUp"
    assert decode_action_name(39, 16, True) == "Interact"
    assert decode_action_name(44, 16, True) == "Communicate"
    assert decode_action_name(56, 16, True) == "Ascend"
    assert decode_action_name(61, 16, True) == "Scan"
    assert decode_action_name(70, 16, True) == "DropPayload"


def test_decode_action_name_supports_agri_and_hex_layouts() -> None:
    comm_vocab = 8
    drone_base = 40 + comm_vocab

    assert decode_action_name(
        drone_base + 19 + 12,
        comm_vocab,
        True,
        agri_enabled=True,
        hex_enabled=False,
    ) == "RelaySoilData"
    assert decode_action_name(
        drone_base,
        comm_vocab,
        False,
        agri_enabled=False,
        hex_enabled=True,
    ) == "Move"
    assert decode_action_name(
        drone_base + 19 + 14,
        comm_vocab,
        True,
        agri_enabled=True,
        hex_enabled=True,
    ) == "Move"


def test_resolve_forge_scenarios_supports_high_level_and_eval_formats(tmp_path: Path) -> None:
    high_level_path = tmp_path / "patrol_doc.toml"
    high_level_path.write_text(
        "\n".join(
            [
                "[scenario]",
                'name = "Patrol"',
                "min_agents = 1",
                "max_agents = 2",
                "",
                "[scenario.map]",
                "grid_size = 32",
                "",
                "[scenario.objectives]",
                "time_limit = 123",
                "",
                "[scenario.difficulty]",
                "base_tier = 3",
            ]
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "basic_patrol.toml"
    eval_path.write_text(
        "\n".join(
            [
                "[scenario]",
                'id = "basic_patrol"',
                'name = "Basic Patrol"',
                "difficulty_tier = 1",
                "min_agents = 1",
                "max_agents = 1",
                "",
                "[forge.world]",
                "width = 16",
                "height = 16",
                "seed = 7",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_forge_scenarios([high_level_path, eval_path])

    assert resolved[0].scenario_id == "patrol"
    assert resolved[0].forge_config["world"]["width"] == 32
    assert resolved[0].forge_config["task"]["max_episode_length"] == 123
    assert resolved[0].difficulty_tier == 3
    assert resolved[1].scenario_id == "basic_patrol"
    assert resolved[1].forge_config["world"]["width"] == 16


def test_collect_training_data_from_scenarios_builds_pipeline_inputs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "drone_patrol.toml"
    scenario_path.write_text(
        "\n".join(
            [
                "[scenario]",
                'name = "drone_patrol"',
                "min_agents = 1",
                "max_agents = 1",
                "",
                "[scenario.map]",
                "grid_size = 24",
                "",
                "[scenario.objectives]",
                "time_limit = 12",
                "",
                "[scenario.difficulty]",
                "base_tier = 2",
            ]
        ),
        encoding="utf-8",
    )

    bridge_config = MangoMASBridgeConfig()
    bridge_config.platform = "drone"
    bridge_config.batch_collector.max_steps = 5

    result = collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge_config,
        scenario_refs=[scenario_path],
        total_episodes=2,
        base_seed=17,
        policy_name="random",
        env_factory=_FakeEnv,
    )

    assert len(result.training_data.observations) == 2
    assert result.training_data.observations[0].shape[0] == 4
    assert result.training_data.action_ids[0].shape == (3,)
    assert len(result.training_data.raw_observations[0]) == 3
    assert all(outcome is True for outcome in result.curriculum_outcomes)
    assert result.scenario_summaries[0].episodes_collected == 2
    assert result.scenario_summaries[0].mean_reward == pytest.approx(3.0)
    assert result.resolved_scenarios[0].scenario_id == "drone_patrol"


def test_write_collection_report_serializes_collection_summary(tmp_path: Path) -> None:
    scenario_path = tmp_path / "drone_patrol.toml"
    scenario_path.write_text(
        "\n".join(
            [
                "[scenario]",
                'name = "drone_patrol"',
                "min_agents = 1",
                "max_agents = 1",
                "",
                "[scenario.map]",
                "grid_size = 24",
            ]
        ),
        encoding="utf-8",
    )

    result = collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=MangoMASBridgeConfig(),
        scenario_refs=[scenario_path],
        total_episodes=2,
        base_seed=23,
        env_factory=_FakeEnv,
    )

    report_path = write_collection_report(
        result,
        tmp_path / "collection_report.json",
        mode="collect-only",
        platform="drone",
        policy_name="random",
        base_seed=23,
        scenario_refs=[scenario_path],
        config_paths=["configs/mangomas/default.toml"],
        run_name="drone-seed-23",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "collect-only"
    assert payload["totals"]["episodes"] == 2
    assert payload["totals"]["steps"] == 6
    assert payload["totals"]["success_rate"] == pytest.approx(1.0)
    assert payload["scenarios"][0]["scenario_id"] == "drone_patrol"
    assert payload["scenarios"][0]["difficulty_tier"] == 1


def test_collect_training_data_requires_enough_episodes_for_selected_scenarios(tmp_path: Path) -> None:
    first = tmp_path / "one.toml"
    second = tmp_path / "two.toml"
    for path, name in ((first, "one"), (second, "two")):
        path.write_text(
            "\n".join(
                [
                    "[scenario]",
                    f'name = "{name}"',
                    "min_agents = 1",
                    "",
                    "[scenario.map]",
                    "grid_size = 16",
                ]
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Collection episodes must be >= number of scenarios"):
        collect_training_data_from_scenarios(
            base_forge_config={"world": {"width": 16, "height": 16, "seed": 0}},
            mangomas_config=MangoMASBridgeConfig(),
            scenario_refs=[first, second],
            total_episodes=1,
            base_seed=1,
            env_factory=_FakeEnv,
        )
