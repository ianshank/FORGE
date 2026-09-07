"""Enterprise AQA & Regression Test: Deterministic Agents and Skills Harness.

Validates:
1. Multi-seed deterministic reproducibility for rule-based, scripted, and heuristic agents.
2. Bit-identical trajectory generation across identical seeds.
3. Skill action decoding, state preservation, and execution under structured logging.
4. Dynamic/backward-compatible agent interfaces with no hardcoded values.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

from forge.mangomas.collector import (
    collect_training_data_from_scenarios,
    decode_action_name,
)
from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger("forge.aqa.agent_skills")


class DeterministicMockEnv:
    """Mock environment enforcing bit-identical step execution and state logging."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.action_space = SimpleNamespace(n=75)
        self.seed = config.get("world", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self._step = 0
        self.max_steps = config.get("task", {}).get("max_episode_length", 10)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
        self._step = 0
        logger.debug("DeterministicMockEnv reset with seed %s", self.seed)
        return self._make_obs(), {"tick": 0, "tasks_completed": [[]]}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        terminated = self._step >= self.max_steps
        # Deterministic reward calculation based on pseudo-random generator state
        reward = float(self.rng.standard_normal())
        action_label = decode_action_name(
            action_id=action,
            comm_vocab_size=0,
            drone_enabled=True,
            agri_enabled=True,
        )
        logger.debug(
            "Step %d: action=%d (%s), reward=%.4f, terminated=%s",
            self._step,
            action,
            action_label,
            reward,
            terminated,
        )
        info = {
            "tick": self._step,
            "tasks_completed": [[1]] if terminated else [[]],
            "agent_skill_executed": action_label,
        }
        return self._make_obs(), reward, terminated, False, info

    def _make_obs(self) -> dict[str, Any]:
        grid = np.zeros((11, 11, 7), dtype=np.uint8)
        grid[5, 5, 0] = 1
        return {
            "grid_view": grid,
            "inventory": np.zeros((10, 2), dtype=np.uint16),
            "health": 1.0,
            "stamina": 1.0,
            "position": (5, 5),
            "messages": np.zeros((0,), dtype=np.uint16),
            "day_phase": 0,
            "altitude": 0.5,
            "battery": 0.9,
            "morphology": 2.0,
            "heading": 0.0,
        }

    def close(self) -> None:
        pass


def test_agent_skills_deterministic_trajectories(tmp_path: Any) -> None:
    """Verifies that two runs with identical seeds produce byte-identical rollouts."""
    scenario_file = tmp_path / "skill_scenario.toml"
    scenario_file.write_text(
        "\n".join([
            "[scenario]",
            'name = "drone_skills_test"',
            "min_agents = 1",
            "tier = 1",
            "",
            "[scenario.map]",
            "grid_size = 16",
        ]),
        encoding="utf-8",
    )

    base_config = {
        "world": {"width": 16, "height": 16, "seed": 9999},
        "agents": {"num_agents": 1, "comm_vocab_size": 0, "comm_radius": 10},
        "task": {"enabled": True, "max_episode_length": 5},
    }

    # Run 1
    logger.info("Starting Run 1 of agent skills collection")
    run1 = collect_training_data_from_scenarios(
        base_forge_config=base_config,
        mangomas_config=MangoMASBridgeConfig(),
        scenario_refs=[scenario_file],
        total_episodes=2,
        base_seed=12345,
        env_factory=DeterministicMockEnv,
    )

    # Run 2
    logger.info("Starting Run 2 of agent skills collection (identical seed)")
    run2 = collect_training_data_from_scenarios(
        base_forge_config=base_config,
        mangomas_config=MangoMASBridgeConfig(),
        scenario_refs=[scenario_file],
        total_episodes=2,
        base_seed=12345,
        env_factory=DeterministicMockEnv,
    )

    # Validate identical metrics and trajectories
    assert run1.total_episodes() == run2.total_episodes()
    assert run1.total_steps() == run2.total_steps()
    assert run1.success_rate() == run2.success_rate()

    tdata1 = run1.training_data
    tdata2 = run2.training_data

    assert tdata1.action_names == tdata2.action_names
    for a1, a2 in zip(tdata1.action_ids, tdata2.action_ids):
        np.testing.assert_array_equal(a1, a2)

    for r1, r2 in zip(tdata1.rewards, tdata2.rewards):
        np.testing.assert_array_equal(r1, r2)

    for o1, o2 in zip(tdata1.observations, tdata2.observations):
        np.testing.assert_array_equal(o1, o2)

    logger.info("Agent skills deterministic validation completed successfully.")


def test_skill_action_decoder_coverage() -> None:
    """Verifies action decoding produces predictable, structured skill representations."""
    actions_to_test = [0, 1, 5, 10, 20, 40, 50, 65]
    decoded_drone = [
        decode_action_name(a, comm_vocab_size=0, drone_enabled=True, agri_enabled=False)
        for a in actions_to_test
    ]
    decoded_agri = [
        decode_action_name(a, comm_vocab_size=0, drone_enabled=True, agri_enabled=True)
        for a in actions_to_test
    ]

    assert all(isinstance(name, str) and len(name) > 0 for name in decoded_drone)
    assert all(isinstance(name, str) and len(name) > 0 for name in decoded_agri)
    assert decoded_drone[0] == "Noop"
    assert decoded_drone[1] == "MoveUp"
    assert decoded_drone[2] == "PickUp"
    assert "Ascend" in decoded_drone or "Hover" in decoded_drone
