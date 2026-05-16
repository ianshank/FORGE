"""Regression: LM Studio base URL must end in /v1, never /api/v1/chat.

The OpenAI SDK appends `/chat/completions` on top of `base_url`. If anyone
"fixes" the URL to `/api/v1/chat` (a common misconception based on older
LM Studio docs / partial curl examples), the SDK will hit
`/api/v1/chat/chat/completions` and silently fail.
"""

from __future__ import annotations

from conftest import REPO_ROOT
from forge.cognitive import LMStudioProvider
from forge.mangomas.config import MangoMASBridgeConfig

GEMMA_PRESET = REPO_ROOT / "configs" / "cognitive" / "gemma_e4b_teacher.toml"
QWEN_PRESET = REPO_ROOT / "configs" / "cognitive" / "qwen14b_teacher.toml"


def test_default_base_url_ends_in_v1() -> None:
    provider = LMStudioProvider()
    base_url = provider._base_url
    assert base_url is not None, "LMStudioProvider must set _base_url to a non-None default"
    assert base_url.endswith("/v1"), base_url
    assert "/api/" not in base_url, base_url


def test_gemma_preset_base_url_ends_in_v1() -> None:
    cfg = MangoMASBridgeConfig.from_toml(GEMMA_PRESET).teacher
    assert cfg.base_url.endswith("/v1"), cfg.base_url
    assert "/api/" not in cfg.base_url, cfg.base_url


def test_qwen_preset_base_url_ends_in_v1() -> None:
    cfg = MangoMASBridgeConfig.from_toml(QWEN_PRESET).teacher
    assert cfg.base_url.endswith("/v1"), cfg.base_url
    assert "/api/" not in cfg.base_url, cfg.base_url
