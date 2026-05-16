"""Live smoke test against a running LM Studio instance.

Gated by two mechanisms (defense in depth):

* `@pytest.mark.lmstudio` marker — controls *discovery*. Default `addopts`
  in `pyproject.toml` includes `-m 'not lmstudio'`, so the suite is
  deselected unless invoked with `pytest -m lmstudio`.
* `FORGE_LMSTUDIO_LIVE=1` env var — *runtime* gate. Even when the marker
  selects, missing env var skips the test rather than failing with a
  connection error.

Run locally with:
    $env:FORGE_LMSTUDIO_LIVE = "1"
    pytest tests/python/test_lmstudio_live_smoke.py -m lmstudio -v --no-cov
"""

from __future__ import annotations

import dataclasses
import json
import logging
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

logger = logging.getLogger(__name__)

# Combine the marker (discovery gate) with the env-var skip (runtime gate).
# A list of marks applies all of them to every test in the module.
pytestmark = [
    pytest.mark.lmstudio,
    pytest.mark.skipif(
        os.environ.get("FORGE_LMSTUDIO_LIVE", "") != "1",
        reason="set FORGE_LMSTUDIO_LIVE=1 to run live LM Studio integration tests",
    ),
]


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


def test_lmstudio_round_trip_parametrised_by_model_id(lmstudio_model_id: str) -> None:
    """Round-trip the same PING_PROMPT against each supported model id.

    Reuses the `lmstudio_model_id` fixture (`tests/python/conftest.py:55-65`)
    parametrised over `LMSTUDIO_SUPPORTED_MODEL_IDS` so adding a new id
    automatically runs the test against it. `TeacherConfig` is a plain
    (non-frozen) `@dataclass` (verified `python/forge/mangomas/config.py:352`),
    so we use `dataclasses.replace` to override the model without inline
    mutation.
    """
    base_cfg = MangoMASBridgeConfig.from_toml(GEMMA_PRESET).teacher
    cfg = dataclasses.replace(base_cfg, model=lmstudio_model_id)
    logger.debug(
        "lmstudio round-trip: model=%s base_url=%s timeout=%ss",
        cfg.model,
        cfg.base_url,
        cfg.timeout_secs,
    )
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
    assert resp.text, f"empty response from local LM Studio for model={cfg.model}"
    # PING_PROMPT requests a strict {"ok": true} response; parsing + field
    # validation proves the model is actually following instructions rather
    # than just returning any non-empty string. Mirrors the assertion in
    # test_gemma_preset_round_trips_against_local_lmstudio for parity.
    parsed = json.loads(resp.text)
    assert parsed.get("ok") in (True, "true", 1), (
        f"unexpected response content for model={cfg.model}: {resp.text!r}"
    )
    assert resp.latency_ms > 0
    logger.debug("lmstudio round-trip ok: latency_ms=%s", resp.latency_ms)
