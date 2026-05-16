"""Tests for the LLM agent module."""

from __future__ import annotations

import numpy as np

from forge.cognitive.llm_agent import LLMAgent, LLMAgentConfig
from forge.cognitive.providers import MockProvider


class TestLLMAgentConfig:
    """Tests for LLMAgentConfig dataclass."""

    def test_defaults(self) -> None:
        config = LLMAgentConfig()
        assert config.provider_name == "mock"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.reasoning_steps == 5

    def test_inherits_agent_config(self) -> None:
        config = LLMAgentConfig(name="my_agent", learning_rate=1e-3)
        assert config.name == "my_agent"
        assert config.learning_rate == 1e-3


class TestLLMAgent:
    """Tests for LLMAgent."""

    def test_creation_with_mock(self) -> None:
        config = LLMAgentConfig()
        agent = LLMAgent(config)
        assert agent.provider_name() == "mock"

    def test_creation_with_custom_provider(self) -> None:
        provider = MockProvider(default_response="Action: 3")
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        assert agent.provider_name() == "mock"

    def test_act_returns_action_and_trace(self) -> None:
        provider = MockProvider(default_response="I should move. Action: 2")
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        obs = np.zeros(20, dtype=np.float32)
        action_id, trace_info = agent.act(obs)
        assert action_id == 2
        assert "provider" in trace_info
        assert "response" in trace_info
        assert trace_info["action_id"] == 2

    def test_act_increments_step_count(self) -> None:
        config = LLMAgentConfig()
        agent = LLMAgent(config)
        obs = np.zeros(10, dtype=np.float32)
        agent.act(obs)
        agent.act(obs)
        assert agent.step_count == 2

    def test_parse_action_from_response(self) -> None:
        provider = MockProvider(default_response="Action: 7")
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        obs = np.zeros(10, dtype=np.float32)
        action_id, _ = agent.act(obs)
        assert action_id == 7

    def test_parse_action_fallback_to_zero(self) -> None:
        provider = MockProvider(default_response="no number here at all")
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        obs = np.zeros(10, dtype=np.float32)
        action_id, _ = agent.act(obs)
        assert action_id == 0

    def test_parse_action_last_number_fallback(self) -> None:
        provider = MockProvider(default_response="I think 5 is good")
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        obs = np.zeros(10, dtype=np.float32)
        action_id, _ = agent.act(obs)
        assert action_id == 5

    def test_learn_returns_empty_dict(self) -> None:
        config = LLMAgentConfig()
        agent = LLMAgent(config)
        result = agent.learn({"obs": np.zeros(10)})
        assert result == {}

    def test_reasoning_history_tracked(self) -> None:
        config = LLMAgentConfig()
        agent = LLMAgent(config)
        obs = np.zeros(10, dtype=np.float32)
        agent.act(obs)
        agent.act(obs)
        assert len(agent._reasoning_history) == 2

    def test_provider_name_method(self) -> None:
        provider = MockProvider()
        config = LLMAgentConfig()
        agent = LLMAgent(config, provider=provider)
        assert agent.provider_name() == "mock"
