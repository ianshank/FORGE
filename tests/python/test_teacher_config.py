"""Tests for ``TeacherConfig`` and the shared env-override helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forge.mangomas.config import (
    DEFAULT_TEACHER_BASE_URL,
    DEFAULT_TEACHER_PROVIDER,
    DEFAULT_TEACHER_SHARD_SIZE,
    MangoMASBridgeConfig,
    TeacherConfig,
)
from forge.utils.config_env import apply_env_overrides

if TYPE_CHECKING:
    import pytest


def test_teacher_config_defaults() -> None:
    cfg = TeacherConfig()
    assert cfg.enabled is False
    assert cfg.provider == DEFAULT_TEACHER_PROVIDER
    assert cfg.base_url == DEFAULT_TEACHER_BASE_URL
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0
    assert cfg.concurrency == 1
    assert cfg.shard_size == DEFAULT_TEACHER_SHARD_SIZE
    assert cfg.trace_schema_version == "1.0"


def test_mangomas_bridge_default_teacher() -> None:
    cfg = MangoMASBridgeConfig()
    assert isinstance(cfg.teacher, TeacherConfig)
    assert cfg.teacher.enabled is False


def test_from_toml_loads_teacher_section(tmp_path: Path) -> None:
    toml_path = tmp_path / "t.toml"
    toml_path.write_text(
        """
        [teacher]
        enabled = true
        model = "qwen2.5-14b-instruct"
        concurrency = 4
        seed = 7
        """,
        encoding="utf-8",
    )
    cfg = MangoMASBridgeConfig.from_toml(toml_path)
    assert cfg.teacher.enabled is True
    assert cfg.teacher.model == "qwen2.5-14b-instruct"
    assert cfg.teacher.concurrency == 4
    assert cfg.teacher.seed == 7


def test_env_overrides_applied_on_load(
    tmp_path: Path, monkeypatch: object
) -> None:
    toml_path = tmp_path / "t.toml"
    toml_path.write_text(
        """
        [teacher]
        enabled = true
        concurrency = 2
        seed = 1
        """,
        encoding="utf-8",
    )
    import os
    os.environ["FORGE_TEACHER_CONCURRENCY"] = "8"
    os.environ["FORGE_TEACHER_SEED"] = "99"
    os.environ["FORGE_TEACHER_LOG_PAYLOADS"] = "true"
    try:
        cfg = MangoMASBridgeConfig.from_toml(toml_path)
        assert cfg.teacher.concurrency == 8
        assert cfg.teacher.seed == 99
        assert cfg.teacher.log_payloads is True
    finally:
        for k in (
            "FORGE_TEACHER_CONCURRENCY",
            "FORGE_TEACHER_SEED",
            "FORGE_TEACHER_LOG_PAYLOADS",
        ):
            os.environ.pop(k, None)


def test_apply_env_overrides_unknown_key_ignored() -> None:
    @dataclass
    class _Cfg:
        x: int = 1

    cfg = _Cfg()
    import os
    os.environ["FORGE_SECTION_UNKNOWN"] = "5"
    try:
        apply_env_overrides(cfg, "SECTION")
        assert cfg.x == 1
    finally:
        os.environ.pop("FORGE_SECTION_UNKNOWN", None)


def test_apply_env_overrides_invalid_int_is_logged_and_skipped() -> None:
    @dataclass
    class _Cfg:
        n: int = 1

    cfg = _Cfg()
    import os
    os.environ["FORGE_SECTION_N"] = "not-an-int"
    try:
        apply_env_overrides(cfg, "SECTION")
        assert cfg.n == 1
    finally:
        os.environ.pop("FORGE_SECTION_N", None)


def test_apply_env_overrides_skips_unsupported_field_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Complex annotations (list/dict) used to silently take the raw string.

    Now they're skipped with a WARNING — assigning a string to a list field
    would silently corrupt downstream consumers, so the helper refuses.
    """
    import logging
    import os
    from dataclasses import field

    @dataclass
    class _Cfg:
        items: list = field(default_factory=list)

    cfg = _Cfg()
    os.environ["FORGE_SECTION_ITEMS"] = "[a, b]"
    caplog.set_level(logging.WARNING, logger="forge.utils.config_env")
    try:
        apply_env_overrides(cfg, "SECTION")
    finally:
        os.environ.pop("FORGE_SECTION_ITEMS", None)
    assert cfg.items == []  # untouched
    assert any("unsupported field type" in r.message for r in caplog.records)


def test_apply_env_overrides_bool_parsing() -> None:
    @dataclass
    class _Cfg:
        flag: bool = False

    cfg = _Cfg()
    import os
    for raw, expected in (
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ):
        os.environ["FORGE_SECTION_FLAG"] = raw
        try:
            apply_env_overrides(cfg, "SECTION")
            assert cfg.flag is expected, raw
        finally:
            os.environ.pop("FORGE_SECTION_FLAG", None)


def test_qwen14b_preset_loads(tmp_path: Path) -> None:
    cfg = MangoMASBridgeConfig.from_toml(
        Path(__file__).parent.parent.parent
        / "configs"
        / "cognitive"
        / "qwen14b_teacher.toml"
    )
    assert cfg.teacher.enabled is True
    assert cfg.teacher.provider == "lmstudio"
    assert cfg.teacher.model == "qwen2.5-14b-instruct"
    assert cfg.teacher.concurrency == 4
    assert cfg.teacher.response_format_enabled is True
