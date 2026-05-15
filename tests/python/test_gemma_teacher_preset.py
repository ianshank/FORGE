"""Smoke + structural tests for the google/gemma-4-e4b LM Studio preset."""
from __future__ import annotations

import json
from typing import Any, cast

import pytest

from conftest import REPO_ROOT, read_toml  # noqa: E402 - pytest adds tests/python to sys.path

SCHEMA_PATH = REPO_ROOT / "python" / "forge" / "cognitive" / "schemas" / "gemma_action.json"
QWEN_SCHEMA_PATH = REPO_ROOT / "python" / "forge" / "cognitive" / "schemas" / "qwen_action.json"


def _inner_schema(envelope: dict[str, Any]) -> dict[str, Any]:
    """Both schema files wrap the action schema in OpenAI's json_schema envelope."""
    return cast("dict[str, Any]", envelope["json_schema"]["schema"])


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


FEW_SHOTS_PATH = REPO_ROOT / "configs" / "cognitive" / "few_shots" / "gemma_teacher.jsonl"


def _load_action_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return cast("dict[str, Any]", json.load(fh)["json_schema"]["schema"])


def test_gemma_few_shots_file_exists() -> None:
    assert FEW_SHOTS_PATH.is_file()


def test_gemma_few_shots_use_observation_response_envelope() -> None:
    rows = [
        json.loads(ln)
        for ln in FEW_SHOTS_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) >= 3, "expect at least 3 few-shot examples"
    for idx, row in enumerate(rows):
        assert "observation" in row, f"row {idx} missing 'observation'"
        assert "response" in row, f"row {idx} missing 'response'"


def test_gemma_few_shot_responses_validate_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_action_schema()
    rows = [
        json.loads(ln)
        for ln in FEW_SHOTS_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for row in rows:
        jsonschema.validate(instance=row["response"], schema=schema)


def test_gemma_few_shots_loadable_by_prompt_builder() -> None:
    """Constructor proves _load_few_shots accepts the file shape."""
    builder = PromptBuilder(
        template_path=str(TEMPLATE_PATH),
        few_shot_examples_path=str(FEW_SHOTS_PATH),
    )
    obs = {"position": [0, 0]}
    rendered = builder.render(obs, legal_actions=[0, 1])
    assert "Example 1:" in rendered, "few-shot examples must appear in rendered prompt"


from forge.mangomas.config import (  # noqa: E402
    DEFAULT_TEACHER_MAX_RETRIES,
    DEFAULT_TEACHER_MAX_TOKENS,
    DEFAULT_TEACHER_PAYLOAD_PREVIEW_CHARS,
    DEFAULT_TEACHER_RETRY_BACKOFF_SECS,
    DEFAULT_TEACHER_SEED,
    DEFAULT_TEACHER_SHARD_SIZE,
    DEFAULT_TEACHER_TEMPERATURE,
    DEFAULT_TEACHER_TIMEOUT_SECS,
    DEFAULT_TEACHER_TOP_P,
    DEFAULT_TEACHER_TRACE_SCHEMA_VERSION,
    MangoMASBridgeConfig,
)

TOML_PATH = REPO_ROOT / "configs" / "cognitive" / "gemma_e4b_teacher.toml"


def test_gemma_toml_loads_via_mangomas_bridge() -> None:
    bridge = MangoMASBridgeConfig.from_toml(TOML_PATH)
    cfg = bridge.teacher
    # Identity:
    assert cfg.enabled is True
    assert cfg.provider == "lmstudio"
    assert cfg.base_url == "http://localhost:1234/v1"
    assert cfg.model == "google/gemma-4-e4b"
    # Sampling — every field, against the constants:
    assert cfg.temperature == DEFAULT_TEACHER_TEMPERATURE
    assert cfg.top_p == DEFAULT_TEACHER_TOP_P
    assert cfg.max_tokens == DEFAULT_TEACHER_MAX_TOKENS
    assert cfg.seed == DEFAULT_TEACHER_SEED
    # Transport:
    assert cfg.timeout_secs == DEFAULT_TEACHER_TIMEOUT_SECS
    assert cfg.max_retries == DEFAULT_TEACHER_MAX_RETRIES
    assert cfg.retry_backoff_secs == DEFAULT_TEACHER_RETRY_BACKOFF_SECS
    # Concurrency: explicitly bumped to 4 in the preset, matches Qwen.
    assert cfg.concurrency == 4
    # Asset paths — backwards-compat: same filename grammar as the Qwen preset.
    assert cfg.prompt_template_path.endswith("gemma_teacher.txt")
    assert cfg.response_schema_path.endswith("gemma_action.json")
    assert cfg.few_shot_examples_path.endswith("gemma_teacher.jsonl")
    # Trace + flags:
    assert cfg.shard_size == DEFAULT_TEACHER_SHARD_SIZE
    assert cfg.trace_schema_version == DEFAULT_TEACHER_TRACE_SCHEMA_VERSION
    assert cfg.payload_preview_chars == DEFAULT_TEACHER_PAYLOAD_PREVIEW_CHARS
    assert cfg.compress_traces is True
    assert cfg.response_format_enabled is True
    assert cfg.validate_action is True
    assert cfg.include_legal_actions is True
    assert cfg.log_payloads is False
    assert cfg.system_prompt == ""
    assert cfg.output_root == "artifacts/teacher_traces"


DEFAULT_TOML_PATH = REPO_ROOT / "configs" / "cognitive" / "default.toml"


def test_default_cognitive_lmstudio_model_is_gemma() -> None:
    data = read_toml(DEFAULT_TOML_PATH)
    assert data["cognitive"]["lmstudio"]["model"] == "google/gemma-4-e4b"


def test_default_cognitive_structured_paths_point_at_gemma_assets() -> None:
    structured = read_toml(DEFAULT_TOML_PATH)["cognitive"]["structured"]
    assert structured["prompt_template_path"].endswith("gemma_teacher.txt")
    assert structured["response_schema_path"].endswith("gemma_action.json")
    assert structured["few_shot_examples_path"].endswith("gemma_teacher.jsonl")


SNAPSHOT_PATH = REPO_ROOT / "tests" / "python" / "snapshots" / "gemma_prompt_minimal.txt"


def test_gemma_prompt_matches_snapshot() -> None:
    """Byte-for-byte pin on the rendered prompt for a fixed input.

    Defends against silent corruption of the template's prose or
    placeholder ordering — a determinism check alone wouldn't catch
    a template that ignored all inputs.
    """
    builder = PromptBuilder(template_path=str(TEMPLATE_PATH), few_shot_examples_path=None)
    obs = {"position": [0, 0], "health": 1.0, "visible_resources": [], "visible_enemies": []}
    rendered = builder.render(
        obs, legal_actions=[0, 1, 2, 3], system_prompt="You are a teacher."
    )
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Rendered prompt drifted from snapshot.\n"
        f"--- expected ---\n{expected!r}\n--- got ---\n{rendered!r}"
    )
