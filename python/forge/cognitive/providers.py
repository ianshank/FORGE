"""Provider-agnostic LLM providers for the cognitive agent.

Supports multiple backends: mock (for testing), Anthropic, OpenAI, and
OpenAI-compatible local servers such as LM Studio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL: str = "gpt-4"
DEFAULT_LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
DEFAULT_LMSTUDIO_MODEL: str = ""
DEFAULT_LMSTUDIO_TIMEOUT_SECS: float = 120.0
DEFAULT_LMSTUDIO_MAX_RETRIES: int = 2
DEFAULT_LMSTUDIO_RETRY_BACKOFF_SECS: float = 1.0
# LM Studio's local server doesn't authenticate, but the openai SDK rejects
# an empty/None api_key at client construction time, so a placeholder string
# is required. Override only if you've put LM Studio behind a real auth proxy.
DEFAULT_LMSTUDIO_API_KEY: str = "lm-studio"
DEFAULT_PAYLOAD_PREVIEW_CHARS: int = 256


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars with an ellipsis marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


@dataclass
class CompletionConfig:
    """Configuration for a completion request.

    The optional fields default to ``None`` so they are only forwarded to
    upstream APIs when explicitly set, preserving compatibility with
    providers that do not accept them.
    """

    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    top_p: float | None = None
    extra_body: dict[str, Any] | None = None
    timeout_secs: float | None = None


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class CognitiveProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion for the given prompt."""

    async def acomplete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        """Async completion. Default implementation off-loads ``complete``.

        Concrete providers that have a native async client (e.g. OpenAI's
        :class:`AsyncOpenAI`) should override this to avoid the thread hop.
        This is intentionally **not** abstract: existing third-party
        subclasses of :class:`CognitiveProvider` continue to work unchanged.
        """
        return await asyncio.to_thread(self.complete, prompt, config)


class MockProvider(CognitiveProvider):
    """Mock provider for testing with deterministic responses."""

    def __init__(self, default_response: str = "Action: 0") -> None:
        self.default_response = default_response
        self.responses: dict[str, str] = {}
        self.call_count = 0
        logger.info("MockProvider initialized")

    def name(self) -> str:
        """Return provider name."""
        return "mock"

    def add_response(self, prefix: str, response: str) -> None:
        """Add a response for prompts starting with the given prefix."""
        self.responses[prefix] = response

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Return a deterministic response."""
        self.call_count += 1
        for prefix, response in self.responses.items():
            if prompt.startswith(prefix):
                return CompletionResponse(text=response)
        return CompletionResponse(text=self.default_response)


class AnthropicProvider(CognitiveProvider):
    """Anthropic Claude API provider.

    Requires the ``anthropic`` package and an API key in the
    environment variable ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any = None
        logger.info("AnthropicProvider initialized")

    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    def _get_client(self) -> Any:
        """Return a cached Anthropic client instance (lazy initialization)."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                msg = "anthropic package required for AnthropicProvider"
                raise ImportError(msg) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion using the Anthropic API."""
        client = self._get_client()
        response = client.messages.create(
            model=config.model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return CompletionResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _build_openai_kwargs(
    prompt: str, config: CompletionConfig, default_model: str
) -> dict[str, Any]:
    """Construct ``chat.completions.create`` kwargs from a CompletionConfig.

    Optional fields are forwarded only when explicitly set, so servers that
    reject unknown keys (some LM Studio releases) keep working.
    """
    kwargs: dict[str, Any] = {
        "model": config.model or default_model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.response_format is not None:
        kwargs["response_format"] = config.response_format
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.extra_body is not None:
        kwargs["extra_body"] = config.extra_body
    if config.timeout_secs is not None:
        kwargs["timeout"] = config.timeout_secs
    return kwargs


def _extract_usage(response: Any) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from a chat completion."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return (prompt_tokens, completion_tokens)


class OpenAIProvider(CognitiveProvider):
    """OpenAI API provider.

    Requires the ``openai`` package and an API key.
    """

    _provider_name: str = "openai"
    _default_model: str = DEFAULT_OPENAI_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout_secs: float | None = None,
        max_retries: int = 0,
        retry_backoff_secs: float = DEFAULT_LMSTUDIO_RETRY_BACKOFF_SECS,
        payload_preview_chars: int = DEFAULT_PAYLOAD_PREVIEW_CHARS,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_secs = timeout_secs
        self._max_retries = max_retries
        self._retry_backoff_secs = retry_backoff_secs
        self._payload_preview_chars = payload_preview_chars
        self._client: Any = None
        self._aclient: Any = None
        logger.info(
            "%sProvider initialized base_url=%s timeout_secs=%s max_retries=%d",
            self._provider_name,
            self._base_url,
            self._timeout_secs,
            self._max_retries,
        )

    def name(self) -> str:
        """Return provider name."""
        return self._provider_name

    def _get_client(self) -> Any:
        """Return a cached synchronous OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                msg = "openai package required for OpenAIProvider"
                raise ImportError(msg) from exc

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._timeout_secs is not None:
                kwargs["timeout"] = self._timeout_secs
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        """Return a cached asynchronous OpenAI client (lazy initialization)."""
        if self._aclient is None:
            try:
                import openai
            except ImportError as exc:
                msg = "openai package required for OpenAIProvider"
                raise ImportError(msg) from exc

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._timeout_secs is not None:
                kwargs["timeout"] = self._timeout_secs
            self._aclient = openai.AsyncOpenAI(**kwargs)
        return self._aclient

    def _log_payload(self, label: str, text: str) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "provider=%s %s=%s",
                self._provider_name,
                label,
                _truncate(text, self._payload_preview_chars),
            )

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion using the OpenAI-compatible chat API."""
        client = self._get_client()
        kwargs = _build_openai_kwargs(prompt, config, self._default_model)
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._max_retries:
            start = time.perf_counter()
            try:
                self._log_payload("prompt_preview", prompt)
                response = client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000.0
                text = (
                    response.choices[0].message.content or ""
                    if response.choices
                    else ""
                )
                prompt_tokens, completion_tokens = _extract_usage(response)
                logger.info(
                    "provider=%s model=%s tokens_in=%d tokens_out=%d latency_ms=%.1f attempt=%d",
                    self._provider_name,
                    kwargs["model"],
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    attempt,
                )
                self._log_payload("response_preview", text)
                return CompletionResponse(
                    text=text,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 — provider retries any transient
                last_exc = exc
                logger.warning(
                    "provider=%s request failed attempt=%d err=%s",
                    self._provider_name,
                    attempt,
                    exc,
                )
                if attempt >= self._max_retries:
                    break
                time.sleep(self._retry_backoff_secs * (2**attempt))
                attempt += 1
        assert last_exc is not None  # for type narrowing
        raise last_exc

    async def acomplete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        """Async completion using ``openai.AsyncOpenAI``."""
        client = self._get_async_client()
        kwargs = _build_openai_kwargs(prompt, config, self._default_model)
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._max_retries:
            start = time.perf_counter()
            try:
                self._log_payload("prompt_preview", prompt)
                response = await client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000.0
                text = (
                    response.choices[0].message.content or ""
                    if response.choices
                    else ""
                )
                prompt_tokens, completion_tokens = _extract_usage(response)
                logger.info(
                    "provider=%s model=%s tokens_in=%d tokens_out=%d latency_ms=%.1f attempt=%d async=true",
                    self._provider_name,
                    kwargs["model"],
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    attempt,
                )
                self._log_payload("response_preview", text)
                return CompletionResponse(
                    text=text,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "provider=%s async request failed attempt=%d err=%s",
                    self._provider_name,
                    attempt,
                    exc,
                )
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._retry_backoff_secs * (2**attempt))
                attempt += 1
        assert last_exc is not None
        raise last_exc


class LMStudioProvider(OpenAIProvider):
    """LM Studio provider.

    LM Studio exposes an OpenAI-compatible HTTP API. This subclass supplies
    LM-Studio-appropriate defaults (local base URL, longer timeout, light
    retry policy) so callers can construct it with no arguments and get a
    sensible client for a workstation running Qwen 14B locally.
    """

    _provider_name = "lmstudio"
    _default_model = DEFAULT_LMSTUDIO_MODEL

    def __init__(
        self,
        api_key: str | None = DEFAULT_LMSTUDIO_API_KEY,
        base_url: str | None = DEFAULT_LMSTUDIO_BASE_URL,
        *,
        model: str | None = None,
        timeout_secs: float | None = DEFAULT_LMSTUDIO_TIMEOUT_SECS,
        max_retries: int = DEFAULT_LMSTUDIO_MAX_RETRIES,
        retry_backoff_secs: float = DEFAULT_LMSTUDIO_RETRY_BACKOFF_SECS,
        payload_preview_chars: int = DEFAULT_PAYLOAD_PREVIEW_CHARS,
    ) -> None:
        # Explicit None must collapse to the module constant so callers
        # passing api_key=None (e.g. when reading an empty env var) still get
        # a working client.
        if api_key is None:
            api_key = DEFAULT_LMSTUDIO_API_KEY
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout_secs=timeout_secs,
            max_retries=max_retries,
            retry_backoff_secs=retry_backoff_secs,
            payload_preview_chars=payload_preview_chars,
        )
        if model:
            self._default_model = model


def create_provider(provider_name: str, **kwargs: Any) -> CognitiveProvider:
    """Factory function to create a provider by name.

    Args:
        provider_name: One of ``"mock"``, ``"anthropic"``, ``"openai"``,
            ``"lmstudio"``.
        **kwargs: Provider-specific configuration.

    Returns:
        A CognitiveProvider instance.
    """
    providers: dict[str, type[CognitiveProvider]] = {
        "mock": MockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "lmstudio": LMStudioProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        msg = f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        raise ValueError(msg)
    return cls(**kwargs)
