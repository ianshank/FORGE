"""Live smoke test against a running LM Studio instance.

Gated by env var FORGE_LMSTUDIO_LIVE=1. Default-off so CI never invokes it.

Run locally with:
    $env:FORGE_LMSTUDIO_LIVE = "1"
    pytest tests/python/test_lmstudio_live_smoke.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from conftest import REPO_ROOT
from forge.cognitive import CompletionConfig, create_provider
from forge.mangomas.config import MangoMASBridgeConfig

GEMMA_PRESET = REPO_ROOT / "configs" / "cognitive" / "gemma_e4b_teacher.toml"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Import the demo helper's PING_* constants so the live test does not roll
# its own prompt / token budget. Single source of truth across helper + test.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from run_lmstudio_demo import PING_MAX_TOKENS, PING_PROMPT  # noqa: E402

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
        PING_PROMPT,
        CompletionConfig(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=PING_MAX_TOKENS,
            seed=cfg.seed,
        ),
    )
    assert resp.text, "empty response from local LM Studio"
    parsed = json.loads(resp.text)
    assert parsed.get("ok") in (True, "true", 1)
    assert resp.latency_ms > 0
