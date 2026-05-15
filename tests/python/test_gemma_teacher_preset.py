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
