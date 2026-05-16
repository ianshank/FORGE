"""Tests for the collector's ``policy_name='llm'`` path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from forge.cognitive.providers import (
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
)
from forge.mangomas.collector import (
    _build_llm_agent,
    _create_policy_agent,
    collect_training_data_from_scenarios,
)
from forge.mangomas.config import MangoMASBridgeConfig, TeacherConfig


class _FakeEnv:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.action_space = SimpleNamespace(n=4)
        self._step = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step = 0
        return self._observation(), {"tick": 0, "tasks_completed": [[]]}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        terminated = self._step >= 2
        info = {"tick": self._step, "tasks_completed": [[1]] if terminated else [[]]}
        return self._observation(), 1.0, terminated, False, info

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "grid_view": np.zeros((11, 11, 7), dtype=np.uint8),
            "inventory": np.zeros((10, 2), dtype=np.uint16),
            "health": 1.0,
            "stamina": 0.9,
            "position": (5, 5),
            "messages": np.zeros((0,), dtype=np.uint16),
            "day_phase": 1,
            "altitude": 0.3,
            "battery": 0.8,
            "morphology": 2.0,
            "heading": 0.0,
        }


class _DeterministicTeacherProvider(CognitiveProvider):
    """Provider that returns the same structured JSON every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def name(self) -> str:
        return "fake-teacher"

    def complete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        self.calls.append(prompt)
        text = json.dumps(
            {
                "action_id": 1,
                "intention": 3,
                "subgoals": ["explore"],
                "rationale": "scripted teacher",
                "value_hat": 0.42,
                "constraint_critique": {"violates_safe_distance": False},
                "top_k_probs": [{"action_id": 1, "prob": 0.95}],
            }
        )
        return CompletionResponse(
            text=text,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
        )


def _write_scenario(path: Path) -> None:
    path.write_text(
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
                "[scenario.difficulty]",
                "base_tier = 1",
            ]
        ),
        encoding="utf-8",
    )


def _teacher_config(tmp_path: Path) -> TeacherConfig:
    repo_root = Path(__file__).resolve().parent.parent.parent
    cfg = TeacherConfig(
        enabled=True,
        provider="lmstudio",
        model="qwen",
        prompt_template_path=str(
            repo_root / "configs/cognitive/templates/qwen_teacher.txt"
        ),
        response_schema_path=str(
            repo_root / "python/forge/cognitive/schemas/qwen_action.json"
        ),
        few_shot_examples_path=str(
            repo_root / "configs/cognitive/few_shots/qwen_teacher.jsonl"
        ),
        output_root=str(tmp_path / "traces"),
        shard_size=10,
        compress_traces=False,
        concurrency=1,
        validate_action=False,
    )
    return cfg


def test_create_policy_agent_llm_returns_llm_agent(tmp_path: Path) -> None:
    teacher = _teacher_config(tmp_path)
    fake = _DeterministicTeacherProvider()
    agent = _create_policy_agent(
        "llm",
        action_space_size=4,
        seed=7,
        teacher_config=teacher,
        provider_factory=lambda _cfg: fake,
    )
    from forge.cognitive.llm_agent import LLMAgent

    assert isinstance(agent, LLMAgent)
    assert agent.provider is fake


def test_build_llm_agent_uses_provider_factory(tmp_path: Path) -> None:
    teacher = _teacher_config(tmp_path)
    fake = _DeterministicTeacherProvider()
    agent = _build_llm_agent(
        teacher,
        action_space_size=4,
        seed=0,
        provider_factory=lambda _cfg: fake,
    )
    assert agent.provider is fake


def test_collect_rollout_captures_teacher_fields(tmp_path: Path) -> None:
    scenario_path = tmp_path / "drone_patrol.toml"
    _write_scenario(scenario_path)

    teacher = _teacher_config(tmp_path)
    bridge = MangoMASBridgeConfig()
    bridge.platform = "drone"
    bridge.batch_collector.max_steps = 5
    bridge.teacher = teacher

    fake = _DeterministicTeacherProvider()
    result = collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge,
        scenario_refs=[scenario_path],
        total_episodes=1,
        base_seed=11,
        policy_name="llm",
        env_factory=_FakeEnv,
        teacher_config=teacher,
        provider_factory=lambda _cfg: fake,
    )

    td = result.training_data
    assert len(td.teacher_intentions) == 1
    assert td.teacher_intentions[0] == [3, 3]
    assert td.teacher_rationales[0] == ["scripted teacher", "scripted teacher"]
    assert td.teacher_subgoals[0] == [["explore"], ["explore"]]
    assert td.teacher_value_hats[0] == [0.42, 0.42]
    assert td.teacher_constraint_critiques[0] == [
        {"violates_safe_distance": False},
        {"violates_safe_distance": False},
    ]
    assert td.teacher_top_k_probs[0] == [
        [{"action_id": 1, "prob": 0.95}],
        [{"action_id": 1, "prob": 0.95}],
    ]
    assert fake.calls  # provider was actually queried


def test_collect_rollout_writes_teacher_trace_shards(tmp_path: Path) -> None:
    scenario_path = tmp_path / "drone_patrol.toml"
    _write_scenario(scenario_path)

    teacher = _teacher_config(tmp_path)
    bridge = MangoMASBridgeConfig()
    bridge.platform = "drone"
    bridge.batch_collector.max_steps = 5
    bridge.teacher = teacher

    fake = _DeterministicTeacherProvider()
    collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge,
        scenario_refs=[scenario_path],
        total_episodes=1,
        base_seed=11,
        policy_name="llm",
        env_factory=_FakeEnv,
        teacher_config=teacher,
        provider_factory=lambda _cfg: fake,
    )

    shards = sorted((tmp_path / "traces" / "drone_patrol").glob("*.jsonl"))
    assert len(shards) >= 1
    records = [
        json.loads(line)
        for shard in shards
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert records[0]["intention"] == 3
    assert records[0]["rationale"] == "scripted teacher"
    assert records[0]["scenario_id"] == "drone_patrol"
    assert records[0]["episode_index"] == 0


def test_legacy_collection_unchanged_when_policy_random(tmp_path: Path) -> None:
    """Regression: random policy still works with default kwargs."""
    scenario_path = tmp_path / "drone_patrol.toml"
    _write_scenario(scenario_path)

    result = collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=MangoMASBridgeConfig(),
        scenario_refs=[scenario_path],
        total_episodes=1,
        base_seed=17,
        policy_name="random",
        env_factory=_FakeEnv,
    )
    assert len(result.training_data.observations) == 1
    assert result.training_data.teacher_intentions == []


def test_llm_policy_without_teacher_config_raises(tmp_path: Path) -> None:
    """policy='llm' without a TeacherConfig must error clearly."""
    scenario_path = tmp_path / "drone_patrol.toml"
    _write_scenario(scenario_path)

    # _create_policy_agent should reject policy='llm' when no teacher_config
    # is plumbed through, since downstream code unconditionally dereferences
    # it. The error message must mention "TeacherConfig" so callers can
    # diagnose without reading source.
    import pytest

    with pytest.raises(ValueError, match="requires a TeacherConfig"):
        _create_policy_agent("llm", action_space_size=4, seed=0)
