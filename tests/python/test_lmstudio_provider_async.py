"""Async tests for LMStudioProvider and the CognitiveProvider acomplete fallback.

Uses ``asyncio.run`` inside each test function — no pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from forge.cognitive.providers import (
    CognitiveProvider,
    CompletionConfig,
    CompletionResponse,
    LMStudioProvider,
    MockProvider,
)


def _fake_chat_response(text: str, prompt_tokens: int, completion_tokens: int) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class _RecordingMockProvider(MockProvider):
    """MockProvider that records concurrent in-flight count via acomplete."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def acomplete(
        self, prompt: str, config: CompletionConfig
    ) -> CompletionResponse:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return CompletionResponse(text=self.default_response)
        finally:
            async with self._lock:
                self.in_flight -= 1


def test_acomplete_default_runs_in_thread() -> None:
    """The default ``CognitiveProvider.acomplete`` should call ``complete``."""

    class _Provider(CognitiveProvider):
        def __init__(self) -> None:
            self.called_from_main_thread = False

        def name(self) -> str:
            return "tester"

        def complete(
            self, prompt: str, config: CompletionConfig
        ) -> CompletionResponse:
            return CompletionResponse(text="sync:" + prompt)

    provider = _Provider()
    result = asyncio.run(provider.acomplete("ping", CompletionConfig()))
    assert result.text == "sync:ping"


def test_lmstudio_acomplete_uses_async_client() -> None:
    provider = LMStudioProvider(max_retries=0, retry_backoff_secs=0.0)
    fake_async_client = MagicMock()
    fake_async_client.chat.completions.create = AsyncMock(
        return_value=_fake_chat_response("pong", 3, 4)
    )
    provider._aclient = fake_async_client  # noqa: SLF001

    cfg = CompletionConfig(model="m", seed=1, response_format={"type": "json_object"})
    resp = asyncio.run(provider.acomplete("ping", cfg))
    assert resp.text == "pong"
    assert resp.input_tokens == 3
    assert resp.output_tokens == 4
    call_kwargs = fake_async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "m"
    assert call_kwargs["seed"] == 1
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_semaphore_bounds_concurrency() -> None:
    provider = _RecordingMockProvider()
    sem = asyncio.Semaphore(2)

    async def gated_call() -> CompletionResponse:
        async with sem:
            return await provider.acomplete("p", CompletionConfig())

    async def run_all() -> list[CompletionResponse]:
        return await asyncio.gather(*(gated_call() for _ in range(8)))

    results = asyncio.run(run_all())
    assert len(results) == 8
    assert provider.max_in_flight <= 2
    assert provider.max_in_flight >= 1


def test_async_retry_on_transient_error_then_succeeds() -> None:
    provider = LMStudioProvider(max_retries=2, retry_backoff_secs=0.0)
    fake_async_client = MagicMock()
    fake_async_client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("boom"), _fake_chat_response("ok", 1, 1)]
    )
    provider._aclient = fake_async_client  # noqa: SLF001
    resp = asyncio.run(provider.acomplete("p", CompletionConfig(model="m")))
    assert resp.text == "ok"
    assert fake_async_client.chat.completions.create.call_count == 2


def test_async_giveup_after_max_retries() -> None:
    provider = LMStudioProvider(max_retries=1, retry_backoff_secs=0.0)
    fake_async_client = MagicMock()
    fake_async_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    provider._aclient = fake_async_client  # noqa: SLF001
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(provider.acomplete("p", CompletionConfig(model="m")))
    assert fake_async_client.chat.completions.create.call_count == 2
