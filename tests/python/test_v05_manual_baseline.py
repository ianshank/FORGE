"""Unit tests for scripts/v05_manual_baseline.py.

Fakes `recv_text`/`send_text` at the module level with a pre-scripted
message queue rather than round-tripping real WebSocket frames --
`tests/python/test_ws_client.py` already covers that encoding. This
file tests `drive_episode`/`iter_episodes`'s own outcome classification
and consecutive-failure tracking against them as trusted dependencies.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import pytest

from scripts import v05_manual_baseline as v05mb

if TYPE_CHECKING:
    from pathlib import Path


class _ScriptedTransport:
    """Replays a fixed queue of decoded messages as `recv_text` results
    and records every `send_text` payload, standing in for a real
    socket + `_ws_client` pair."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.sent: list[dict[str, Any]] = []

    def recv_text(self, _sock: object, _buf: object) -> dict[str, Any]:
        if not self._messages:
            msg = "scripted transport exhausted before the code under test stopped asking"
            raise AssertionError(msg)
        return self._messages.pop(0)

    def send_text(self, _sock: object, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


def _drive(monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]) -> _ScriptedTransport:
    transport = _ScriptedTransport(messages)
    monkeypatch.setattr(v05mb, "recv_text", transport.recv_text)
    monkeypatch.setattr(v05mb, "send_text", transport.send_text)
    return transport


def _obs(
    *, reward: float = 0.0, terminated: bool = False, truncated: bool = False, obs_len: int = 3
) -> dict[str, Any]:
    return {
        "type": "observation",
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "tick": 1,
        "obs": [0.0] * obs_len,
    }


def _error(code: str, message: str = "boom") -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}


def _run_drive_episode(monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]) -> dict[str, Any]:
    _drive(monkeypatch, messages)
    return v05mb.drive_episode(
        object(), bytearray(), action_count=4, max_steps=3, seed=1, rng=random.Random(1)
    )


# --- drive_episode: outcome classification -----------------------------


def test_error_frame_is_environment_error_not_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_drive_episode(monkeypatch, [_obs(), _error("INTERNAL")])
    assert result["outcome"] == "environment_error"
    assert result["protocol_errors"] == 1
    assert result["truncated"] is False
    assert result["terminated"] is False


def test_reset_error_is_environment_error_with_no_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_drive_episode(monkeypatch, [_error("BUSY")])
    assert result["outcome"] == "environment_error"
    assert result["protocol_errors"] == 1
    assert result["steps"] == 0
    assert result["obs_dim"] == 0


def test_step_budget_exhausted_without_env_signal_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three steps, none of which the environment marks terminated or
    # truncated: this driver's own max_steps loop exhausts first. That
    # is a truncation at the driver level, not an ambiguous outcome.
    messages = [_obs(), _obs(), _obs(), _obs()]
    result = _run_drive_episode(monkeypatch, messages)
    assert result["outcome"] == "truncated"
    assert result["truncated"] is True
    assert result["terminated"] is False
    assert result["protocol_errors"] == 0
    assert result["steps"] == 3


def test_natural_terminal_is_terminated_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_drive_episode(monkeypatch, [_obs(), _obs(terminated=True, reward=5.0)])
    assert result["outcome"] == "terminated"
    assert result["terminated"] is True
    assert result["truncated"] is False
    assert result["total_reward"] == 5.0
    assert result["steps"] == 1


def test_environment_reported_truncation_is_truncated_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_drive_episode(monkeypatch, [_obs(), _obs(truncated=True)])
    assert result["outcome"] == "truncated"
    assert result["truncated"] is True
    assert result["terminated"] is False


# --- drive_episode: malformed message types ------------------------------


def test_unexpected_message_type_at_reset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="expected observation after reset"):
        _run_drive_episode(monkeypatch, [{"type": "bogus"}])


def test_unexpected_message_type_at_step_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="expected observation after step"):
        _run_drive_episode(monkeypatch, [_obs(), {"type": "bogus"}])


# --- drive_episode: contract violations abort ---------------------------


def test_invalid_action_code_at_step_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(v05mb.ContractViolation):
        _run_drive_episode(
            monkeypatch, [_obs(), _error("INVALID_ACTION", "unknown action_id 99")]
        )


def test_bad_message_code_at_reset_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(v05mb.ContractViolation):
        _run_drive_episode(monkeypatch, [_error("BAD_MESSAGE", "malformed frame")])


def test_unrecognised_code_fails_closed_as_contract_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A code this script has never seen is itself evidence of drift --
    # it must not be silently treated as just another transient fault.
    with pytest.raises(v05mb.ContractViolation):
        _run_drive_episode(monkeypatch, [_obs(), _error("SOMETHING_NEW")])


# --- iter_episodes: consecutive-failure tracking ------------------------


def test_halts_after_consecutive_environment_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    threshold = v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES
    messages = [_error("INTERNAL")] * threshold
    _drive(monkeypatch, messages)
    hello = {"action_count": 4}
    gen = v05mb.iter_episodes(
        object(), bytearray(), hello=hello, episodes=10, max_steps=5, base_seed=0
    )
    records: list[dict[str, Any]] = []
    with pytest.raises(v05mb.SustainedEnvironmentFailure):
        for record in gen:
            # `list(gen)` would discard the already-yielded records once
            # the generator raises on the threshold-th failure; the
            # partial output is exactly what this test asserts on.
            records.append(record)  # noqa: PERF402
    assert len(records) == threshold
    assert all(r["outcome"] == "environment_error" for r in records)


def test_isolated_failure_does_not_halt(monkeypatch: pytest.MonkeyPatch) -> None:
    below_threshold = v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES - 1
    assert below_threshold >= 1
    messages = [_error("INTERNAL")] * below_threshold
    _drive(monkeypatch, messages)
    hello = {"action_count": 4}
    records = list(
        v05mb.iter_episodes(
            object(),
            bytearray(),
            hello=hello,
            episodes=below_threshold,
            max_steps=5,
            base_seed=0,
        )
    )
    assert len(records) == below_threshold


def test_failure_followed_by_success_resets_the_consecutive_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fail, succeed, fail again: the failures are not consecutive, so
    # this must run to completion without halting.
    messages = [
        _error("INTERNAL"),  # episode 1: reset fails
        _obs(),  # episode 2: reset ok
        _obs(terminated=True),  # episode 2: step terminates
        _error("BUSY"),  # episode 3: reset fails
    ]
    _drive(monkeypatch, messages)
    hello = {"action_count": 4}
    records = list(
        v05mb.iter_episodes(
            object(), bytearray(), hello=hello, episodes=3, max_steps=5, base_seed=0
        )
    )
    assert [r["outcome"] for r in records] == [
        "environment_error",
        "terminated",
        "environment_error",
    ]


# --- schema_id verification ----------------------------------------------


def test_schema_id_matching_the_repo_configs_does_not_raise() -> None:
    module = v05mb._load_schema_id_module()
    expected = module.compute_schema_id_from_paths(v05mb.ACTION_MAP_PATH, v05mb.REWARDS_PATH)
    v05mb._assert_schema_id_matches_repo({"schema_id": expected})


def test_schema_id_mismatch_raises() -> None:
    with pytest.raises(v05mb.SchemaIdMismatch):
        v05mb._assert_schema_id_matches_repo({"schema_id": "0" * 64})


def test_schema_id_missing_from_handshake_raises() -> None:
    with pytest.raises(v05mb.SchemaIdMismatch):
        v05mb._assert_schema_id_matches_repo({})


def test_load_schema_id_module_raises_when_spec_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # lru_cache(maxsize=1) means a prior successful call in this process
    # would otherwise short-circuit before ever reaching
    # spec_from_file_location; clear it so the patched failure path is
    # actually exercised, then clear it again so later tests reload the
    # real module instead of reusing this test's empty cache slot.
    v05mb._load_schema_id_module.cache_clear()
    monkeypatch.setattr(
        v05mb.importlib.util, "spec_from_file_location", lambda *a, **k: None
    )
    try:
        with pytest.raises(RuntimeError, match="cannot load schema_id module"):
            v05mb._load_schema_id_module()
    finally:
        v05mb._load_schema_id_module.cache_clear()


# --- pinned constant -----------------------------------------------------


def test_max_consecutive_env_failures_is_pinned() -> None:
    assert v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES == 3, (
        "Lowering this changes how much sustained bot failure the capture "
        "tolerates before giving up. Update this pin deliberately and say "
        "why in the PR description."
    )


# --- main(): end-to-end integration ---------------------------------------


class _FakeSocket:
    """Stand-in for the socket `open_ws` would return. `recv_text`/
    `send_text` are monkeypatched to ignore it entirely, but `main`'s
    `finally` block unconditionally calls `.close()` on it."""

    def close(self) -> None:
        pass


def _real_schema_id() -> str:
    module = v05mb._load_schema_id_module()
    return module.compute_schema_id_from_paths(v05mb.ACTION_MAP_PATH, v05mb.REWARDS_PATH)


def _hello(*, schema_id: str | None = None, action_count: int = 4) -> dict[str, Any]:
    return {
        "type": "hello",
        "obs_dim": 3,
        "action_count": action_count,
        "grid_shape": None,
        "schema_id": schema_id if schema_id is not None else _real_schema_id(),
    }


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    messages: list[dict[str, Any]],
    *,
    args: list[str] | None = None,
) -> tuple[int, Path]:
    _drive(monkeypatch, messages)
    monkeypatch.setattr(v05mb, "open_ws", lambda *_a, **_k: _FakeSocket())
    out_path = tmp_path / "out.json"
    argv = ["--out", str(out_path), *(args or [])]
    rc = v05mb.main(argv)
    return rc, out_path


def test_main_happy_path_writes_output_and_returns_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    messages = [_hello(), _obs(), _obs(terminated=True, reward=5.0)]
    rc, out_path = _run_main(monkeypatch, tmp_path, messages, args=["--episodes", "1"])
    assert rc == v05mb.EXIT_OK
    assert out_path.exists()
    body = json.loads(out_path.read_text())
    assert body["episodes_observed"] == 1
    assert body["per_episode"][0]["outcome"] == "terminated"


def test_main_schema_mismatch_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    messages = [_hello(schema_id="0" * 64)]
    rc, out_path = _run_main(monkeypatch, tmp_path, messages, args=["--episodes", "1"])
    assert rc == v05mb.EXIT_SCHEMA_MISMATCH
    assert not out_path.exists()


def test_main_contract_violation_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    messages = [_hello(), _error("INVALID_ACTION")]
    rc, out_path = _run_main(monkeypatch, tmp_path, messages, args=["--episodes", "1"])
    assert rc == v05mb.EXIT_CONTRACT_VIOLATION
    assert not out_path.exists()


def test_main_sustained_failure_writes_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    threshold = v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES
    messages = [_hello(), *([_error("INTERNAL")] * threshold)]
    rc, out_path = _run_main(monkeypatch, tmp_path, messages, args=["--episodes", "10"])
    assert rc == v05mb.EXIT_SUSTAINED_FAILURE
    assert out_path.exists()
    body = json.loads(out_path.read_text())
    assert body["episodes_observed"] == threshold
    assert body["episodes_target"] == 10
