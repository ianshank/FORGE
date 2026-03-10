"""Provider-agnostic LLM providers for the cognitive agent.

Supports multiple backends: mock (for testing), Anthropic, OpenAI, and local.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompletionConfig:
    """Configuration for a completion request."""

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class CognitiveProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion for the given prompt."""


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
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                msg = "anthropic package required for AnthropicProvider"
                raise ImportError(msg) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion using the Anthropic API."""
        client = self._get_client()
        response = client.messages.create(
            model=config.model or "claude-sonnet-4-20250514",
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return CompletionResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIProvider(CognitiveProvider):
    """OpenAI API provider.

    Requires the ``openai`` package and an API key.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None
        logger.info("OpenAIProvider initialized")

    def name(self) -> str:
        """Return provider name."""
        return "openai"

    def _get_client(self) -> Any:
        """Return a cached OpenAI client instance (lazy initialization)."""
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                msg = "openai package required for OpenAIProvider"
                raise ImportError(msg) from exc

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, config: CompletionConfig) -> CompletionResponse:
        """Generate a completion using the OpenAI API."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=config.model or "gpt-4",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or "" if response.choices else ""
        return CompletionResponse(text=text)


def create_provider(provider_name: str, **kwargs: Any) -> CognitiveProvider:
    """Factory function to create a provider by name.

    Args:
        provider_name: One of "mock", "anthropic", "openai".
        **kwargs: Provider-specific configuration.

    Returns:
        A CognitiveProvider instance.
    """
    providers: dict[str, type[CognitiveProvider]] = {
        "mock": MockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        msg = f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        raise ValueError(msg)
    return cls(**kwargs)
