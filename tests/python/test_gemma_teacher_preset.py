"""Smoke + structural tests for the google/gemma-4-e4b LM Studio preset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "python" / "forge" / "cognitive" / "schemas" / "gemma_action.json"
QWEN_SCHEMA_PATH = REPO_ROOT / "python" / "forge" / "cognitive" / "schemas" / "qwen_action.json"


def _inner_schema(envelope: dict) -> dict:
    """Both schema files wrap the action schema in OpenAI's json_schema envelope."""
    return envelope["json_schema"]["schema"]


def test_gemma_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file(), f"missing schema asset: {SCHEMA_PATH}"


def test_gemma_schema_envelope_matches_openai_json_schema_shape() -> None:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["type"] == "json_schema"
    inner = data["json_schema"]
    assert inner["strict"] is True
    assert "name" in inner
    assert "schema" in inner


def test_gemma_schema_required_fields_match_qwen() -> None:
    """Action contract is env-driven, not model-driven; both must agree."""
    with QWEN_SCHEMA_PATH.open(encoding="utf-8") as fh:
        qwen = _inner_schema(json.load(fh))
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        gemma = _inner_schema(json.load(fh))
    assert set(qwen["required"]) == set(gemma["required"])
    assert qwen["properties"].keys() == gemma["properties"].keys()


from forge.cognitive.prompt_builder import PromptBuilder  # noqa: E402

TEMPLATE_PATH = REPO_ROOT / "configs" / "cognitive" / "templates" / "gemma_teacher.txt"


def test_gemma_template_file_exists() -> None:
    assert TEMPLATE_PATH.is_file(), f"missing template asset: {TEMPLATE_PATH}"


def test_gemma_template_uses_canonical_placeholders() -> None:
    body = TEMPLATE_PATH.read_text(encoding="utf-8")
    # PromptBuilder injects exactly these names. Spelling matters.
    for key in ("{system_prompt}", "{obs_json}", "{legal_actions}", "{few_shots}"):
        assert key in body, f"template missing canonical placeholder {key!r}"


def test_gemma_template_renders_with_observation_and_legal_actions() -> None:
    builder = PromptBuilder(
        template_path=str(TEMPLATE_PATH),
        few_shot_examples_path=None,
    )
    obs = {"position": [0, 0], "health": 1.0, "visible_resources": [], "visible_enemies": []}
    rendered_a = builder.render(
        obs, legal_actions=[0, 1, 2, 3], system_prompt="You are a teacher."
    )
    rendered_b = builder.render(
        obs, legal_actions=[0, 1, 2, 3], system_prompt="You are a teacher."
    )
    assert rendered_a == rendered_b, "PromptBuilder must be deterministic"
    # Substitutions ACTUALLY happened (defends against silent _SafeDict pass-through):
    assert "You are a teacher." in rendered_a
    assert '"position"' in rendered_a
    assert "0, 1, 2, 3" in rendered_a
    # Model id stays in TOML, not template.
    assert "google/gemma-4-e4b" not in rendered_a
