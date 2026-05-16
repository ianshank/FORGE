"""Tests for the cognitive providers module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.cognitive.providers import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MAX_RETRIES,
    DEFAULT_LMSTUDIO_TIMEOUT_SECS,
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    LMStudioProvider,
    MockProvider,
    OpenAIProvider,
    _build_openai_kwargs,
    _extract_usage,
    create_provider,
)


class TestCompletionConfig:
    """Tests for CompletionConfig dataclass."""

    def test_defaults(self) -> None:
        config = CompletionConfig()
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_new_optional_fields_default_to_none(self) -> None:
        config = CompletionConfig()
        assert config.response_format is None
        assert config.seed is None
        assert config.top_p is None
        assert config.extra_body is None
        assert config.timeout_secs is None


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_defaults(self) -> None:
        resp = CompletionResponse()
        assert resp.text == ""
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.latency_ms == 0.0
        assert resp.raw == {}


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

    def test_lmstudio_provider_exists(self) -> None:
        provider = create_provider("lmstudio")
        assert provider.name() == "lmstudio"
        assert isinstance(provider, LMStudioProvider)


class TestBuildOpenAIKwargs:
    """Unit tests for the kwarg builder used by both sync and async paths."""

    def test_basic_kwargs(self) -> None:
        cfg = CompletionConfig(model="x", temperature=0.0, max_tokens=10)
        kwargs = _build_openai_kwargs("hi", cfg, default_model="d")
        assert kwargs["model"] == "x"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 10
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert "response_format" not in kwargs
        assert "seed" not in kwargs
        assert "top_p" not in kwargs

    def test_forwards_response_format(self) -> None:
        cfg = CompletionConfig(response_format={"type": "json_object"})
        kwargs = _build_openai_kwargs("hi", cfg, default_model="d")
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_forwards_seed_and_top_p(self) -> None:
        cfg = CompletionConfig(seed=42, top_p=0.95)
        kwargs = _build_openai_kwargs("hi", cfg, default_model="d")
        assert kwargs["seed"] == 42
        assert kwargs["top_p"] == 0.95

    def test_falls_back_to_default_model(self) -> None:
        cfg = CompletionConfig(model="")
        kwargs = _build_openai_kwargs("hi", cfg, default_model="fallback")
        assert kwargs["model"] == "fallback"

    def test_timeout_forwarded(self) -> None:
        cfg = CompletionConfig(timeout_secs=30.0)
        kwargs = _build_openai_kwargs("hi", cfg, default_model="d")
        assert kwargs["timeout"] == 30.0


def _fake_chat_response(text: str, prompt_tokens: int, completion_tokens: int) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestOpenAIProviderTokenCapture:
    """Regression tests for the silent ``input_tokens=0`` bug."""

    def test_extract_usage_reads_prompt_and_completion_tokens(self) -> None:
        response = _fake_chat_response("ok", 17, 23)
        assert _extract_usage(response) == (17, 23)

    def test_extract_usage_handles_missing_usage(self) -> None:
        response = MagicMock()
        response.usage = None
        assert _extract_usage(response) == (0, 0)

    def test_complete_captures_usage_tokens(self) -> None:
        provider = OpenAIProvider(api_key="dummy")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_chat_response(
            "hi", 11, 7
        )
        provider._client = fake_client
        resp = provider.complete("hello", CompletionConfig(model="m"))
        assert resp.text == "hi"
        assert resp.input_tokens == 11
        assert resp.output_tokens == 7
        assert resp.latency_ms >= 0.0

    def test_complete_forwards_optional_fields(self) -> None:
        provider = OpenAIProvider(api_key="dummy")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_chat_response(
            "x", 1, 1
        )
        provider._client = fake_client
        cfg = CompletionConfig(
            model="m",
            response_format={"type": "json_object"},
            seed=99,
            top_p=0.9,
            extra_body={"foo": "bar"},
        )
        provider.complete("hi", cfg)
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["seed"] == 99
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["extra_body"] == {"foo": "bar"}

    def test_complete_does_not_send_optional_when_unset(self) -> None:
        provider = OpenAIProvider(api_key="dummy")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_chat_response(
            "x", 1, 1
        )
        provider._client = fake_client
        provider.complete("hi", CompletionConfig(model="m"))
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert "seed" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert "extra_body" not in call_kwargs


class TestLMStudioProviderSync:
    """Synchronous LMStudioProvider tests with HTTP mocked."""

    def test_defaults(self) -> None:
        provider = LMStudioProvider()
        assert provider.name() == "lmstudio"
        assert provider._base_url == DEFAULT_LMSTUDIO_BASE_URL
        assert provider._timeout_secs == DEFAULT_LMSTUDIO_TIMEOUT_SECS
        assert provider._max_retries == DEFAULT_LMSTUDIO_MAX_RETRIES

    def test_model_kwarg_sets_default_model(self, lmstudio_model_id: str) -> None:
        provider = LMStudioProvider(model=lmstudio_model_id)
        assert provider._default_model == lmstudio_model_id

    def test_complete_uses_lmstudio_provider_name_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = LMStudioProvider()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_chat_response(
            "ok", 4, 5
        )
        provider._client = fake_client
        import logging
        caplog.set_level(logging.INFO, logger="forge.cognitive.providers")
        provider.complete("hi", CompletionConfig(model="qwen"))
        assert any("provider=lmstudio" in r.message for r in caplog.records)

    def test_retry_on_transient_error_then_succeeds(self) -> None:
        provider = LMStudioProvider(max_retries=2, retry_backoff_secs=0.0)
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            RuntimeError("boom"),
            _fake_chat_response("ok", 1, 1),
        ]
        provider._client = fake_client
        resp = provider.complete("hi", CompletionConfig(model="m"))
        assert resp.text == "ok"
        assert fake_client.chat.completions.create.call_count == 2

    def test_giveup_after_max_retries(self) -> None:
        provider = LMStudioProvider(max_retries=1, retry_backoff_secs=0.0)
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        provider._client = fake_client
        with pytest.raises(RuntimeError, match="boom"):
            provider.complete("hi", CompletionConfig(model="m"))
        assert fake_client.chat.completions.create.call_count == 2

    def test_factory_passes_kwargs(self) -> None:
        provider = create_provider(
            "lmstudio",
            base_url="http://example:9/v1",
            model="qwen",
            timeout_secs=5.0,
            max_retries=0,
        )
        assert isinstance(provider, LMStudioProvider)
        assert provider._base_url == "http://example:9/v1"
        assert provider._default_model == "qwen"

    def test_default_api_key_is_module_constant(self) -> None:
        from forge.cognitive import providers as providers_mod

        assert hasattr(providers_mod, "DEFAULT_LMSTUDIO_API_KEY")
        assert providers_mod.DEFAULT_LMSTUDIO_API_KEY == "lm-studio"
        provider = providers_mod.LMStudioProvider()
        assert provider._api_key == providers_mod.DEFAULT_LMSTUDIO_API_KEY

    def test_explicit_none_api_key_falls_back_to_default(self) -> None:
        from forge.cognitive import providers as providers_mod

        provider = providers_mod.LMStudioProvider(api_key=None)
        # Explicit None must collapse to the module constant, not stay None.
        assert provider._api_key == providers_mod.DEFAULT_LMSTUDIO_API_KEY


class TestTruncate:
    """Covers providers._truncate (used in payload-preview logging)."""

    def test_text_shorter_than_limit_passes_through(self) -> None:
        from forge.cognitive.providers import _truncate

        assert _truncate("abc", 10) == "abc"

    def test_text_at_exactly_limit_is_unmodified(self) -> None:
        from forge.cognitive.providers import _truncate

        assert _truncate("abcde", 5) == "abcde"

    def test_text_longer_than_limit_gets_ellipsis_marker(self) -> None:
        from forge.cognitive.providers import _truncate

        # Note the marker is literal: "...<truncated>". The limit applies
        # to the leading slice, not the total returned length.
        result = _truncate("abcdefghij", 3)
        assert result == "abc...<truncated>"

    def test_non_positive_limit_disables_truncation(self) -> None:
        from forge.cognitive.providers import _truncate

        long = "x" * 1000
        assert _truncate(long, 0) == long
        assert _truncate(long, -5) == long
