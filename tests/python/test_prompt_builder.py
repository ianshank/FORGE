"""Tests for ``forge.cognitive.prompt_builder``."""

from __future__ import annotations

from pathlib import Path

import pytest
from forge.cognitive.prompt_builder import PromptBuilder


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_deterministic_rendering(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "OBS={obs_json} ACTS={legal_actions}")
    builder = PromptBuilder(tpl)
    out1 = builder.render({"b": 2, "a": 1}, legal_actions=[1, 2, 3])
    out2 = builder.render({"a": 1, "b": 2}, legal_actions=[1, 2, 3])
    assert out1 == out2
    assert "ACTS=1, 2, 3" in out1
    assert '"a": 1' in out1
    assert '"b": 2' in out1


def test_legal_actions_omitted_when_disabled(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "ACTS=[{legal_actions}]")
    builder = PromptBuilder(tpl, include_legal_actions=False)
    out = builder.render({"x": 1}, legal_actions=[1, 2])
    assert "ACTS=[]" in out


def test_few_shot_inclusion(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "{few_shots}")
    fs = _write(
        tmp_path,
        "fs.jsonl",
        '{"observation": {"x": 1}, "response": {"action_id": 0}}\n'
        '{"observation": {"y": 2}, "response": {"action_id": 1}}\n',
    )
    builder = PromptBuilder(tpl, few_shot_examples_path=fs)
    out = builder.render({"z": 3})
    assert "Example 1" in out
    assert "Example 2" in out
    assert '"action_id": 0' in out
    assert '"action_id": 1' in out


def test_missing_template_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PromptBuilder(tmp_path / "nope.txt")


def test_missing_few_shot_file_raises(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "{obs_json}")
    with pytest.raises(FileNotFoundError):
        PromptBuilder(tpl, few_shot_examples_path=tmp_path / "missing.jsonl")


def test_invalid_few_shot_json_raises(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "{few_shots}")
    fs = _write(tmp_path, "fs.jsonl", "not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        PromptBuilder(tpl, few_shot_examples_path=fs)


def test_few_shot_missing_required_keys(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "{few_shots}")
    fs = _write(tmp_path, "fs.jsonl", '{"observation": {"x": 1}}\n')
    with pytest.raises(ValueError, match="must contain"):
        PromptBuilder(tpl, few_shot_examples_path=fs)


def test_system_prompt_substituted(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "SYS={system_prompt} OBS={obs_json}")
    builder = PromptBuilder(tpl)
    out = builder.render({"a": 1}, system_prompt="be helpful")
    assert "SYS=be helpful" in out


def test_unknown_placeholder_passes_through(tmp_path: Path) -> None:
    tpl = _write(tmp_path, "t.txt", "{obs_json} {unknown_key}")
    builder = PromptBuilder(tpl)
    out = builder.render({"a": 1})
    assert "{unknown_key}" in out
