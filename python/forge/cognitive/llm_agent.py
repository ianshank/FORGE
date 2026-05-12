"""LLM-backed agent for FORGE.

Uses a CognitiveProvider to select actions through structured reasoning.

Two agent paths share this module:

* ``LLMAgent`` with ``LLMAgentConfig`` — the original free-text reasoning
  agent, kept for backwards compatibility. It builds a short prompt from
  the raw observation vector and parses the trailing integer.
* ``LLMAgent`` with ``StructuredLLMAgentConfig`` — a JSON-mode teacher
  driven by ``PromptBuilder`` and a JSON Schema. Emits richer
  ``trace_info`` (intention, subgoals, rationale, value_hat,
  constraint_critique, top_k_probs, token counts, latency) used by the
  offline BC / SFT pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Mapping

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.cognitive.prompt_builder import PromptBuilder
from forge.cognitive.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    MockProvider,
)

logger = logging.getLogger(__name__)


DEFAULT_OBS_PREVIEW_DIM: int = 10
DEFAULT_REASONING_STEPS: int = 5
DEFAULT_SYSTEM_PROMPT: str = (
    "You are an intelligent agent in a grid-based simulation. "
    "Reason step by step, then select an action."
)
DEFAULT_PAYLOAD_PREVIEW_CHARS: int = 256
DEFAULT_VALUE_CLIP: float = 100.0
_PARSE_KEYWORD: str = "action"
_PARSE_STRIP_CHARS: str = ":,. "


@dataclass
class LLMAgentConfig(AgentConfig):
    """Configuration for the legacy free-text LLM agent."""

    provider_name: str = "mock"
    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_steps: int = DEFAULT_REASONING_STEPS
    obs_preview_dim: int = DEFAULT_OBS_PREVIEW_DIM
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class StructuredLLMAgentConfig(LLMAgentConfig):
    """Configuration for the JSON-mode teacher agent.

    The new fields are all opt-in: with ``prompt_template_path`` unset
    the agent falls back to the legacy ``_build_prompt`` /
    ``_parse_action`` path.
    """

    prompt_template_path: str = ""
    response_schema_path: str = ""
    few_shot_examples_path: str = ""
    include_legal_actions: bool = True
    log_payloads: bool = False
    validate_action: bool = True
    value_clip: float = DEFAULT_VALUE_CLIP
    seed: int | None = None
    top_p: float | None = None
    timeout_secs: float | None = None
    payload_preview_chars: int = DEFAULT_PAYLOAD_PREVIEW_CHARS
    legal_actions: tuple[int, ...] = field(default_factory=tuple)


class LLMAgent(BaseAgent):
    """Agent that uses an LLM to select actions via structured reasoning."""

    def __init__(
        self,
        config: LLMAgentConfig,
        provider: CognitiveProvider | None = None,
        *,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        super().__init__(config)
        self.llm_config = config
        self.provider = provider or MockProvider()
        self._reasoning_history: list[dict[str, Any]] = []
        self._structured_config: StructuredLLMAgentConfig | None = (
            config if isinstance(config, StructuredLLMAgentConfig) else None
        )
        self._prompt_builder = prompt_builder or self._init_prompt_builder()
        self._response_format = self._init_response_format()
        logger.info(
            "LLMAgent initialized provider=%s structured=%s",
            self.provider.name(),
            self._structured_config is not None
            and bool(self._structured_config.prompt_template_path),
        )

    def _init_prompt_builder(self) -> PromptBuilder | None:
        cfg = self._structured_config
        if cfg is None or not cfg.prompt_template_path:
            return None
        fewshot = cfg.few_shot_examples_path or None
        return PromptBuilder(
            cfg.prompt_template_path,
            few_shot_examples_path=fewshot,
            include_legal_actions=cfg.include_legal_actions,
        )

    def _init_response_format(self) -> dict[str, Any] | None:
        cfg = self._structured_config
        if cfg is None or not cfg.response_schema_path:
            return None
        path = Path(cfg.response_schema_path)
        if not path.exists():
            msg = f"response_schema_path does not exist: {path}"
            raise FileNotFoundError(msg)
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_completion_config(self) -> CompletionConfig:
        cfg = self._structured_config
        if cfg is None:
            return CompletionConfig(
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
        return CompletionConfig(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format=self._response_format,
            seed=cfg.seed,
            top_p=cfg.top_p,
            timeout_secs=cfg.timeout_secs,
        )

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Synchronous action selection."""
        prompt = self._compose_prompt(observation)
        completion_config = self._build_completion_config()
        response = self.provider.complete(prompt, completion_config)
        return self._finalise(prompt, response)

    async def aact(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Async action selection using the provider's ``acomplete``."""
        prompt = self._compose_prompt(observation)
        completion_config = self._build_completion_config()
        response = await self.provider.acomplete(prompt, completion_config)
        return self._finalise(prompt, response)

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op learning for LLM agents (learning happens via memory/fine-tuning)."""
        return {}

    def provider_name(self) -> str:
        """Return the name of the cognitive provider."""
        return self.provider.name()

    def _compose_prompt(self, observation: np.ndarray) -> str:
        if self._prompt_builder is None or self._structured_config is None:
            return self._build_prompt(observation)
        structured = self._structured_observation(observation)
        legal = self._structured_config.legal_actions or None
        return self._prompt_builder.render(
            structured,
            legal_actions=legal,
            system_prompt=self._structured_config.system_prompt,
        )

    def _structured_observation(
        self, observation: np.ndarray
    ) -> Mapping[str, Any]:
        """Convert a numeric observation vector into a JSON-serialisable dict.

        Override this in subclasses to expose richer structured context to
        the teacher.
        """
        return {"observation": [float(x) for x in observation.tolist()]}

    def _finalise(
        self, prompt: str, response: CompletionResponse
    ) -> tuple[int, dict[str, Any]]:
        self._step_count += 1
        if self._structured_config is None or self._prompt_builder is None:
            action_id = self._parse_action(response.text)
            trace_info: dict[str, Any] = {
                "provider": self.provider.name(),
                "response": response.text,
                "action_id": action_id,
            }
            self._reasoning_history.append(trace_info)
            return action_id, trace_info
        parsed, action_id = self._parse_structured_response(response.text)
        trace_info = {
            "provider": self.provider.name(),
            "response": response.text,
            "action_id": action_id,
            "intention": parsed.get("intention"),
            "subgoals": parsed.get("subgoals"),
            "rationale": parsed.get("rationale"),
            "value_hat": self._clip_value(parsed.get("value_hat")),
            "constraint_critique": parsed.get("constraint_critique"),
            "top_k_probs": parsed.get("top_k_probs"),
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }
        if self._structured_config.log_payloads and logger.isEnabledFor(
            logging.DEBUG
        ):
            preview = self._structured_config.payload_preview_chars
            logger.debug(
                "agent=structured prompt_preview=%s response_preview=%s",
                prompt[:preview],
                response.text[:preview],
            )
        self._reasoning_history.append(trace_info)
        return action_id, trace_info

    def _clip_value(self, value: Any) -> float | None:
        if value is None:
            return None
        cfg = self._structured_config
        clip = cfg.value_clip if cfg is not None else DEFAULT_VALUE_CLIP
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return 0.0
        return max(-clip, min(clip, v))

    def _parse_structured_response(
        self, text: str
    ) -> tuple[dict[str, Any], int]:
        """Parse a JSON response and return ``(parsed, action_id)``.

        On malformed JSON falls back to the legacy integer extractor so an
        ill-formed teacher response still produces a usable action.
        """
        cfg = self._structured_config
        assert cfg is not None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "agent=structured failed to parse JSON; falling back to legacy parser"
            )
            return ({}, self._parse_action(text))
        if not isinstance(parsed, dict):
            logger.warning(
                "agent=structured top-level JSON is not an object; falling back"
            )
            return ({}, self._parse_action(text))
        action_raw = parsed.get("action_id")
        try:
            action_id = int(action_raw)
        except (TypeError, ValueError) as exc:
            msg = f"teacher response missing or non-integer action_id: {action_raw!r}"
            if cfg.validate_action:
                raise ValueError(msg) from exc
            logger.warning(msg)
            return (parsed, 0)
        if cfg.validate_action and cfg.legal_actions:
            if action_id not in cfg.legal_actions:
                msg = (
                    f"teacher action_id={action_id} not in legal_actions="
                    f"{cfg.legal_actions}"
                )
                raise ValueError(msg)
        return (parsed, action_id)

    def _build_prompt(self, observation: np.ndarray) -> str:
        """Build a text prompt from a numerical observation (legacy path)."""
        preview_dim = self.llm_config.obs_preview_dim
        obs_summary = (
            f"Observation vector (dim={observation.shape}): "
            f"{observation[:preview_dim]}..."
        )
        return (
            f"{self.llm_config.system_prompt}\n{obs_summary}\n"
            "Select an action ID (integer)."
        )

    def _parse_action(self, text: str) -> int:
        """Parse an action ID from the provider's response (legacy path)."""
        lower = text.lower()
        if _PARSE_KEYWORD in lower:
            for word in lower.split(_PARSE_KEYWORD)[-1].split():
                cleaned = word.strip(_PARSE_STRIP_CHARS)
                if cleaned.lstrip("-").isdigit():
                    return int(cleaned)
        for word in reversed(text.split()):
            stripped = word.strip(_PARSE_STRIP_CHARS)
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        return 0
