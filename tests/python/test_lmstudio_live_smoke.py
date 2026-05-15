"""Live smoke test against a running LM Studio instance.

Gated by env var FORGE_LMSTUDIO_LIVE=1. Default-off so CI never invokes it.

Run locally with:
    $env:FORGE_LMSTUDIO_LIVE = "1"
    pytest tests/python/test_lmstudio_live_smoke.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge.cognitive import CompletionConfig, create_provider
from forge.mangomas.config import MangoMASBridgeConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
GEMMA_PRESET = REPO_ROOT / "configs" / "cognitive" / "gemma_e4b_teacher.toml"

pytestmark = pytest.mark.skipif(
    os.environ.get("FORGE_LMSTUDIO_LIVE", "") != "1",
    reason="set FORGE_LMSTUDIO_LIVE=1 to run live LM Studio integration tests",
)


def test_gemma_preset_round_trips_against_local_lmstudio() -> None:
    cfg = MangoMASBridgeConfig.from_toml(GEMMA_PRESET).teacher
    provider = create_provider(
        cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        timeout_secs=cfg.timeout_secs,
        max_retries=cfg.max_retries,
        retry_backoff_secs=cfg.retry_backoff_secs,
    )
    resp = provider.complete(
        'Reply with a JSON object that has one key "ok" set to true. No prose.',
        CompletionConfig(
            model=cfg.model,
            temperature=0.0,
            max_tokens=64,
            seed=cfg.seed,
        ),
    )
    assert resp.text, "empty response from local LM Studio"
    parsed = json.loads(resp.text)
    assert parsed.get("ok") in (True, "true", 1)
    assert resp.latency_ms > 0
