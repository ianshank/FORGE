"""Tests for the structured (JSON-mode) LLMAgent path."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from forge.cognitive.llm_agent import LLMAgent, StructuredLLMAgentConfig
from forge.cognitive.providers import (
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    MockProvider,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _structured_template(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        "tpl.txt",
        "SYS={system_prompt}\nOBS={obs_json}\nACTS={legal_actions}\nSHOTS={few_shots}\n",
    )


def _schema(tmp_path: Path) -> Path:
    return _write(tmp_path, "schema.json", '{"type": "json_schema"}')


def _good_response(action_id: int = 2) -> str:
    return json.dumps(
        {
            "action_id": action_id,
            "intention": 1,
            "subgoals": ["move"],
            "rationale": "ok",
            "value_hat": 0.3,
            "constraint_critique": {"violates_cooldown": False},
            "top_k_probs": [{"action_id": action_id, "prob": 0.9}],
        }
    )


class _RecordingProvider(MockProvider):
    def __init__(self, response_text: str) -> None:
        super().__init__()
        self.response_text = response_text
        self.last_config: CompletionConfig | None = None

    def complete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        self.last_config = config
        return CompletionResponse(
            text=self.response_text,
            input_tokens=11,
            output_tokens=7,
            latency_ms=12.5,
        )

    async def acomplete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        self.last_config = config
        return CompletionResponse(
            text=self.response_text,
            input_tokens=11,
            output_tokens=7,
            latency_ms=12.5,
        )


def test_legacy_path_unchanged_without_structured_config() -> None:
    from forge.cognitive.llm_agent import LLMAgentConfig

    agent = LLMAgent(LLMAgentConfig(name="legacy"))
    obs = np.zeros(4, dtype=np.float32)
    action_id, trace = agent.act(obs)
    assert action_id == 0
    assert set(trace.keys()) == {"provider", "response", "action_id"}


def test_structured_act_returns_rich_trace_info(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        response_schema_path=str(_schema(tmp_path)),
        legal_actions=(0, 1, 2, 3),
    )
    provider = _RecordingProvider(_good_response(action_id=2))
    agent = LLMAgent(cfg, provider=provider)
    obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    action_id, trace = agent.act(obs)
    assert action_id == 2
    assert trace["intention"] == 1
    assert trace["subgoals"] == ["move"]
    assert trace["rationale"] == "ok"
    assert pytest.approx(trace["value_hat"]) == 0.3
    assert trace["constraint_critique"] == {"violates_cooldown": False}
    assert trace["top_k_probs"] == [{"action_id": 2, "prob": 0.9}]
    assert trace["prompt_tokens"] == 11
    assert trace["completion_tokens"] == 7
    assert trace["latency_ms"] == 12.5


def test_structured_completion_config_forwards_schema_and_seed(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        response_schema_path=str(_schema(tmp_path)),
        seed=42,
        top_p=0.9,
        timeout_secs=15.0,
        legal_actions=(0, 1, 2),
    )
    provider = _RecordingProvider(_good_response(action_id=1))
    agent = LLMAgent(cfg, provider=provider)
    agent.act(np.zeros(2, dtype=np.float32))
    assert provider.last_config is not None
    assert provider.last_config.response_format == {"type": "json_schema"}
    assert provider.last_config.seed == 42
    assert provider.last_config.top_p == 0.9
    assert provider.last_config.timeout_secs == 15.0


def test_malformed_json_falls_back_to_legacy_parser(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        legal_actions=(0, 1, 2, 3),
        validate_action=False,
    )
    provider = _RecordingProvider("not json, but action: 3")
    agent = LLMAgent(cfg, provider=provider)
    action_id, trace = agent.act(np.zeros(1, dtype=np.float32))
    assert action_id == 3
    assert trace["intention"] is None


def test_value_hat_clipped(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        value_clip=5.0,
        legal_actions=(0, 1, 2),
    )
    text = json.dumps(
        {
            "action_id": 0,
            "intention": 0,
            "subgoals": [],
            "rationale": "",
            "value_hat": 1e6,
            "constraint_critique": {},
        }
    )
    provider = _RecordingProvider(text)
    agent = LLMAgent(cfg, provider=provider)
    _, trace = agent.act(np.zeros(1, dtype=np.float32))
    assert trace["value_hat"] == 5.0


def test_invalid_action_id_raises_when_validate_action_true(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        validate_action=True,
        legal_actions=(0, 1, 2),
    )
    text = json.dumps(
        {
            "action_id": 99,
            "intention": 0,
            "subgoals": [],
            "rationale": "",
            "value_hat": 0.0,
            "constraint_critique": {},
        }
    )
    provider = _RecordingProvider(text)
    agent = LLMAgent(cfg, provider=provider)
    with pytest.raises(ValueError, match="not in legal_actions"):
        agent.act(np.zeros(1, dtype=np.float32))


def test_missing_action_id_raises_when_validate_action_true(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        validate_action=True,
        legal_actions=(0, 1),
    )
    provider = _RecordingProvider(json.dumps({"intention": 0}))
    agent = LLMAgent(cfg, provider=provider)
    with pytest.raises(ValueError, match="action_id"):
        agent.act(np.zeros(1, dtype=np.float32))


def test_aact_returns_same_shape_as_act(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        legal_actions=(0, 1, 2),
    )
    provider = _RecordingProvider(_good_response(action_id=1))
    agent = LLMAgent(cfg, provider=provider)
    obs = np.array([0.0, 0.1], dtype=np.float32)
    action_id, trace = asyncio.run(agent.aact(obs))
    assert action_id == 1
    assert trace["intention"] == 1
    assert trace["prompt_tokens"] == 11


def test_schema_path_missing_raises(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        response_schema_path=str(tmp_path / "missing.json"),
        legal_actions=(0, 1),
    )
    with pytest.raises(FileNotFoundError):
        LLMAgent(cfg)


def test_structured_is_cognitive_subclass() -> None:
    assert issubclass(MockProvider, CognitiveProvider)


def test_structured_agent_does_not_keep_reasoning_history_by_default(
    tmp_path: Path,
) -> None:
    """Regression: structured runs persist traces to disk; the in-memory
    history would duplicate every response and grow without bound.
    """
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        legal_actions=(0, 1, 2, 3),
    )
    provider = _RecordingProvider(_good_response(action_id=2))
    agent = LLMAgent(cfg, provider=provider)
    for _ in range(5):
        agent.act(np.zeros(2, dtype=np.float32))
    assert agent._reasoning_history == []


def test_structured_agent_keep_history_can_be_enabled(tmp_path: Path) -> None:
    cfg = StructuredLLMAgentConfig(
        name="structured",
        prompt_template_path=str(_structured_template(tmp_path)),
        legal_actions=(0, 1, 2, 3),
        keep_reasoning_history=True,
    )
    provider = _RecordingProvider(_good_response(action_id=2))
    agent = LLMAgent(cfg, provider=provider)
    for _ in range(3):
        agent.act(np.zeros(2, dtype=np.float32))
    assert len(agent._reasoning_history) == 3


def test_reasoning_history_max_caps_legacy_agent() -> None:
    """Legacy free-text agents keep history but can opt-in to a cap."""
    from forge.cognitive.llm_agent import LLMAgentConfig

    agent = LLMAgent(LLMAgentConfig(name="legacy", reasoning_history_max=3))
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(8):
        agent.act(obs)
    assert len(agent._reasoning_history) == 3
