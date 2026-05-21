"""Deterministic prompt rendering for the LM Studio / Qwen teacher.

The builder loads a template (and optional few-shot exemplars) once at
construction and renders prompts via ``str.format_map`` so we avoid a
Jinja dependency. Rendering is pure: same inputs produce byte-identical
output, which is required for deterministic teacher traces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)


class _SafeDict(dict[str, Any]):
    """``str.format_map`` substitute that leaves unknown keys untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class FewShotExample:
    """A single few-shot exemplar pair."""

    observation_json: str
    response_json: str


def _load_few_shots(path: Path | None) -> tuple[FewShotExample, ...]:
    if path is None:
        return ()
    if not path.exists():
        msg = f"few_shot_examples_path does not exist: {path}"
        raise FileNotFoundError(msg)
    examples: list[FewShotExample] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"invalid JSON at {path}:{line_no}: {exc}"
                raise ValueError(msg) from exc
            obs = record.get("observation")
            resp = record.get("response")
            if obs is None or resp is None:
                msg = (
                    f"few-shot record at {path}:{line_no} must contain "
                    "'observation' and 'response' keys"
                )
                raise ValueError(msg)
            examples.append(
                FewShotExample(
                    observation_json=json.dumps(obs, sort_keys=True),
                    response_json=json.dumps(resp, sort_keys=True),
                )
            )
    return tuple(examples)


def _render_legal_actions(legal_actions: Iterable[int] | None) -> str:
    if legal_actions is None:
        return ""
    return ", ".join(str(int(a)) for a in legal_actions)


def _render_few_shots(examples: tuple[FewShotExample, ...]) -> str:
    if not examples:
        return ""
    chunks = [
        f"Example {i + 1}:\nObservation: {ex.observation_json}\nResponse: {ex.response_json}"
        for i, ex in enumerate(examples)
    ]
    return "\n\n".join(chunks)


class PromptBuilder:
    """Render teacher prompts from a template + structured observation.

    Template placeholders (all optional, missing ones pass through):
        ``{system_prompt}``, ``{obs_json}``, ``{legal_actions}``,
        ``{few_shots}``.
    """

    def __init__(
        self,
        template_path: str | Path,
        *,
        few_shot_examples_path: str | Path | None = None,
        include_legal_actions: bool = True,
    ) -> None:
        path = Path(template_path)
        if not path.exists():
            msg = f"prompt template does not exist: {path}"
            raise FileNotFoundError(msg)
        self._template = path.read_text(encoding="utf-8")
        fewshot_path = Path(few_shot_examples_path) if few_shot_examples_path else None
        self._few_shots = _load_few_shots(fewshot_path)
        self._include_legal_actions = include_legal_actions
        logger.info(
            "PromptBuilder initialized template=%s few_shots=%d include_legal_actions=%s",
            path,
            len(self._few_shots),
            include_legal_actions,
        )

    def render(
        self,
        structured_obs: Mapping[str, Any],
        *,
        legal_actions: Iterable[int] | None = None,
        system_prompt: str = "",
    ) -> str:
        """Render a prompt string for the given observation."""
        legal_str = _render_legal_actions(legal_actions) if self._include_legal_actions else ""
        obs_json = json.dumps(structured_obs, sort_keys=True)
        rendered = self._template.format_map(
            _SafeDict(
                system_prompt=system_prompt,
                obs_json=obs_json,
                legal_actions=legal_str,
                few_shots=_render_few_shots(self._few_shots),
            )
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("PromptBuilder rendered chars=%d", len(rendered))
        return rendered
