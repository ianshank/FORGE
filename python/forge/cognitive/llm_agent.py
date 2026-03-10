"""LLM-backed agent for FORGE.

Uses a CognitiveProvider to select actions through structured reasoning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np  # noqa: TC002 (used at runtime in _build_prompt)

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.cognitive.providers import (
    CognitiveProvider,
    CompletionConfig,
    MockProvider,
)

logger = logging.getLogger(__name__)


DEFAULT_OBS_PREVIEW_DIM: int = 10
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_REASONING_STEPS: int = 5
DEFAULT_SYSTEM_PROMPT: str = (
    "You are an intelligent agent in a grid-based simulation. "
    "Reason step by step, then select an action."
)
_PARSE_KEYWORD: str = "action"
_PARSE_STRIP_CHARS: str = ":,. "


@dataclass
class LLMAgentConfig(AgentConfig):
    """Configuration for the LLM agent."""

    provider_name: str = "mock"
    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_steps: int = DEFAULT_REASONING_STEPS
    obs_preview_dim: int = DEFAULT_OBS_PREVIEW_DIM
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class LLMAgent(BaseAgent):
    """Agent that uses an LLM to select actions via structured reasoning."""

    def __init__(
        self,
        config: LLMAgentConfig,
        provider: CognitiveProvider | None = None,
    ) -> None:
        super().__init__(config)
        self.llm_config = config
        self.provider = provider or MockProvider()
        self._reasoning_history: list[dict[str, Any]] = []
        logger.info(
            "LLMAgent initialized with provider=%s", self.provider.name()
        )

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action using the LLM provider.

        Constructs a prompt from the observation, queries the provider,
        and parses the response to extract an action ID.
        """
        prompt = self._build_prompt(observation)
        completion_config = CompletionConfig(
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        response = self.provider.complete(prompt, completion_config)
        action_id = self._parse_action(response.text)
        self._step_count += 1

        trace_info = {
            "provider": self.provider.name(),
            "response": response.text,
            "action_id": action_id,
        }
        self._reasoning_history.append(trace_info)
        return action_id, trace_info

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op learning for LLM agents (learning happens via memory/fine-tuning)."""
        return {}

    def provider_name(self) -> str:
        """Return the name of the cognitive provider."""
        return self.provider.name()

    def _build_prompt(self, observation: np.ndarray) -> str:
        """Build a text prompt from a numerical observation."""
        preview_dim = self.llm_config.obs_preview_dim
        obs_summary = f"Observation vector (dim={observation.shape}): {observation[:preview_dim]}..."
        return f"{self.llm_config.system_prompt}\n{obs_summary}\nSelect an action ID (integer)."

    def _parse_action(self, text: str) -> int:
        """Parse an action ID from the provider's response."""
        lower = text.lower()
        if _PARSE_KEYWORD in lower:
            for word in lower.split(_PARSE_KEYWORD)[-1].split():
                cleaned = word.strip(_PARSE_STRIP_CHARS)
                if cleaned.lstrip("-").isdigit():
                    return int(cleaned)
        # Fallback: find last number
        for word in reversed(text.split()):
            stripped = word.strip(_PARSE_STRIP_CHARS)
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        return 0
