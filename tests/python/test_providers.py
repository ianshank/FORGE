"""Tests for the cognitive providers module."""
from __future__ import annotations

import pytest
from forge.cognitive.providers import (
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    MockProvider,
    create_provider,
)


class TestCompletionConfig:
    """Tests for CompletionConfig dataclass."""

    def test_defaults(self) -> None:
        config = CompletionConfig()
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 1024


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_defaults(self) -> None:
        resp = CompletionResponse()
        assert resp.text == ""
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0


class TestMockProvider:
    """Tests for MockProvider."""

    def test_default_response(self) -> None:
        provider = MockProvider()
        config = CompletionConfig()
        response = provider.complete("test prompt", config)
        assert response.text == "Action: 0"

    def test_custom_default_response(self) -> None:
        provider = MockProvider(default_response="Action: 5")
        config = CompletionConfig()
        response = provider.complete("anything", config)
        assert response.text == "Action: 5"

    def test_prefix_matching(self) -> None:
        provider = MockProvider()
        provider.add_response("observe:", "I see resources")
        config = CompletionConfig()
        r1 = provider.complete("observe: grid view", config)
        assert r1.text == "I see resources"
        r2 = provider.complete("act: move", config)
        assert r2.text == "Action: 0"

    def test_name(self) -> None:
        provider = MockProvider()
        assert provider.name() == "mock"

    def test_call_count(self) -> None:
        provider = MockProvider()
        config = CompletionConfig()
        provider.complete("a", config)
        provider.complete("b", config)
        assert provider.call_count == 2

    def test_is_cognitive_provider(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, CognitiveProvider)


class TestCreateProvider:
    """Tests for the create_provider factory."""

    def test_create_mock(self) -> None:
        provider = create_provider("mock")
        assert provider.name() == "mock"

    def test_create_mock_with_kwargs(self) -> None:
        provider = create_provider("mock", default_response="test")
        config = CompletionConfig()
        assert provider.complete("x", config).text == "test"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_anthropic_provider_exists(self) -> None:
        provider = create_provider("anthropic")
        assert provider.name() == "anthropic"

    def test_openai_provider_exists(self) -> None:
        provider = create_provider("openai")
        assert provider.name() == "openai"
