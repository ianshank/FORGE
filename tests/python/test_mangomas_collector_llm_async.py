"""Async / concurrent collection tests for the LLM teacher policy."""

from __future__ import annotations

import asyncio
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
from forge.mangomas.collector import collect_training_data_from_scenarios
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


class _ConcurrentTeacherProvider(CognitiveProvider):
    """Records the maximum number of overlapping ``acomplete`` calls."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()
        self.call_count = 0

    def name(self) -> str:
        return "fake-async"

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        # The sync path uses MockProvider; this lives behind aact.
        return CompletionResponse(text=self._payload())

    async def acomplete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.call_count += 1
        try:
            await asyncio.sleep(0.005)
            return CompletionResponse(text=self._payload(), latency_ms=5.0)
        finally:
            async with self._lock:
                self.in_flight -= 1

    @staticmethod
    def _payload() -> str:
        return json.dumps(
            {
                "action_id": 1,
                "intention": 2,
                "subgoals": ["a"],
                "rationale": "scripted",
                "value_hat": 0.25,
                "constraint_critique": {},
            }
        )


def _scenario(tmp_path: Path) -> Path:
    p = tmp_path / "drone_patrol.toml"
    p.write_text(
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
    return p


def _teacher_config(tmp_path: Path, concurrency: int) -> TeacherConfig:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return TeacherConfig(
        enabled=True,
        provider="lmstudio",
        model="qwen",
        prompt_template_path=str(
            repo_root / "configs/cognitive/templates/qwen_teacher.txt"
        ),
        response_schema_path=str(
            repo_root / "python/forge/cognitive/schemas/qwen_action.json"
        ),
        few_shot_examples_path="",
        output_root=str(tmp_path / "traces"),
        shard_size=10,
        compress_traces=False,
        concurrency=concurrency,
        validate_action=False,
    )


def _run_collection(tmp_path: Path, concurrency: int, *, provider: CognitiveProvider) -> Any:
    scenario_path = _scenario(tmp_path)
    teacher = _teacher_config(tmp_path, concurrency)
    bridge = MangoMASBridgeConfig()
    bridge.platform = "drone"
    bridge.batch_collector.max_steps = 5
    bridge.teacher = teacher
    return collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge,
        scenario_refs=[scenario_path],
        total_episodes=4,
        base_seed=11,
        policy_name="llm",
        env_factory=_FakeEnv,
        teacher_config=teacher,
        provider_factory=lambda _cfg: provider,
    )


def test_concurrent_collection_runs_episodes_in_parallel(tmp_path: Path) -> None:
    provider = _ConcurrentTeacherProvider()
    _run_collection(tmp_path, concurrency=4, provider=provider)
    assert provider.max_in_flight >= 2, (
        f"expected episode-level parallelism, got max_in_flight={provider.max_in_flight}"
    )
    assert provider.max_in_flight <= 4


def test_async_path_writes_shards_in_episode_index_order(tmp_path: Path) -> None:
    provider = _ConcurrentTeacherProvider()
    _run_collection(tmp_path, concurrency=4, provider=provider)
    shards_dir = tmp_path / "traces" / "drone_patrol"
    files = sorted(shards_dir.glob("*.jsonl"))
    # 4 episodes × shard_size=10 → 4 shards, one per episode.
    assert len(files) == 4
    names = [f.name for f in files]
    assert names == [
        "ep000000-0000.jsonl",
        "ep000001-0000.jsonl",
        "ep000002-0000.jsonl",
        "ep000003-0000.jsonl",
    ]


def test_concurrent_collection_byte_identical_to_serial_given_same_seeds(
    tmp_path: Path,
) -> None:
    serial_root = tmp_path / "serial"
    concurrent_root = tmp_path / "concurrent"
    serial_root.mkdir()
    concurrent_root.mkdir()

    provider_serial = _ConcurrentTeacherProvider()
    provider_concurrent = _ConcurrentTeacherProvider()

    # Serial run
    serial_teacher = _teacher_config(serial_root, concurrency=1)
    bridge_serial = MangoMASBridgeConfig()
    bridge_serial.platform = "drone"
    bridge_serial.batch_collector.max_steps = 5
    bridge_serial.teacher = serial_teacher
    collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge_serial,
        scenario_refs=[_scenario(serial_root)],
        total_episodes=4,
        base_seed=11,
        policy_name="llm",
        env_factory=_FakeEnv,
        teacher_config=serial_teacher,
        provider_factory=lambda _cfg: provider_serial,
    )

    # Concurrent run with the same seed
    concurrent_teacher = _teacher_config(concurrent_root, concurrency=4)
    bridge_concurrent = MangoMASBridgeConfig()
    bridge_concurrent.platform = "drone"
    bridge_concurrent.batch_collector.max_steps = 5
    bridge_concurrent.teacher = concurrent_teacher
    collect_training_data_from_scenarios(
        base_forge_config={
            "world": {"width": 16, "height": 16, "seed": 0},
            "agents": {"num_agents": 1, "comm_vocab_size": 16, "comm_radius": 10},
            "task": {"enabled": True, "max_episode_length": 20},
        },
        mangomas_config=bridge_concurrent,
        scenario_refs=[_scenario(concurrent_root)],
        total_episodes=4,
        base_seed=11,
        policy_name="llm",
        env_factory=_FakeEnv,
        teacher_config=concurrent_teacher,
        provider_factory=lambda _cfg: provider_concurrent,
    )

    serial_files = sorted((serial_root / "traces" / "drone_patrol").glob("*.jsonl"))
    concurrent_files = sorted(
        (concurrent_root / "traces" / "drone_patrol").glob("*.jsonl")
    )
    assert [f.name for f in serial_files] == [f.name for f in concurrent_files]
    for s, c in zip(serial_files, concurrent_files, strict=True):
        s_records = [
            json.loads(line)
            for line in s.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        c_records = [
            json.loads(line)
            for line in c.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Strip latency_ms / token counts which legitimately differ in the
        # async path (they are captured at trace write time, not at agent
        # decision time, because the async path emits records after gather).
        for records in (s_records, c_records):
            for r in records:
                r.pop("latency_ms", None)
                r.pop("prompt_tokens", None)
                r.pop("completion_tokens", None)
        assert s_records == c_records, (
            f"shard {s.name}: serial vs concurrent diverge"
        )
