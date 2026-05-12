"""FORGE cognitive module: LLM-backed agent with provider-agnostic architecture."""

from __future__ import annotations

from forge.cognitive.llm_agent import LLMAgent, LLMAgentConfig
from forge.cognitive.providers import (
    AnthropicProvider,
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    LMStudioProvider,
    MockProvider,
    OpenAIProvider,
    create_provider,
)

__all__ = [
    "AnthropicProvider",
    "CognitiveProvider",
    "CompletionConfig",
    "CompletionResponse",
    "LLMAgent",
    "LLMAgentConfig",
    "LMStudioProvider",
    "MockProvider",
    "OpenAIProvider",
    "create_provider",
]
