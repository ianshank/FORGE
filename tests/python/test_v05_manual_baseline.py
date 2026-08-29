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
    from collections.abc import Iterator
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


def _real_schema_id() -> str:
    module = v05mb._load_schema_id_module()
    return module.compute_schema_id_from_paths(v05mb.ACTION_MAP_PATH, v05mb.REWARDS_PATH)


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
    # The failed attempt itself must not count as an executed step.
    assert result["steps"] == 0


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


def test_protocol_error_outranks_a_stale_terminated_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reset observation itself reports terminated=True (a
    # degenerate but not impossible bot response). A transient error
    # on the very next step must still classify as environment_error:
    # protocol_errors has to outrank a *stale* `terminated=True` left
    # over from the reset, not just a `False` one -- this pins the
    # outcome-priority order against being silently reordered.
    result = _run_drive_episode(monkeypatch, [_obs(terminated=True), _error("INTERNAL")])
    assert result["outcome"] == "environment_error"
    assert result["terminated"] is True
    assert result["protocol_errors"] == 1
    assert result["steps"] == 0


def test_total_reward_accumulates_across_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every other reward-bearing test has exactly one nonzero-reward
    # step, so a `+=` -> `=` regression would slip through unnoticed.
    result = _run_drive_episode(
        monkeypatch,
        [_obs(), _obs(reward=2.0), _obs(reward=3.0, terminated=True)],
    )
    assert result["total_reward"] == 5.0
    assert result["steps"] == 2


def test_last_tick_and_obs_dim_reflect_the_final_step_not_the_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_obs = _obs(obs_len=3)
    reset_obs["tick"] = 0
    final_obs = _obs(terminated=True, obs_len=5)
    final_obs["tick"] = 9
    result = _run_drive_episode(monkeypatch, [reset_obs, final_obs])
    assert result["last_tick"] == 9
    assert result["obs_dim"] == 5


# --- drive_episode: outbound payloads -------------------------------------


def test_reset_and_step_payloads_carry_the_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _drive(monkeypatch, [_obs(), _obs(terminated=True)])
    result = v05mb.drive_episode(
        object(), bytearray(), action_count=4, max_steps=3, seed=42, rng=random.Random(42)
    )
    assert transport.sent[0] == {"type": "reset", "seed": 42}
    step_payload = transport.sent[1]
    assert step_payload["type"] == "step"
    assert 0 <= step_payload["action_id"] < 4
    assert result["outcome"] == "terminated"


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


def _run_iter_episodes(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    *,
    episodes: int,
    max_steps: int = 5,
    base_seed: int = 0,
    action_count: int = 4,
    max_consecutive_failures: int = v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES,
) -> Iterator[dict[str, Any]]:
    _drive(monkeypatch, messages)
    hello = {"action_count": action_count}
    return v05mb.iter_episodes(
        object(),
        bytearray(),
        hello=hello,
        episodes=episodes,
        max_steps=max_steps,
        base_seed=base_seed,
        max_consecutive_failures=max_consecutive_failures,
    )


def test_halts_after_consecutive_environment_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    threshold = v05mb.DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES
    messages = [_error("INTERNAL")] * threshold
    gen = _run_iter_episodes(monkeypatch, messages, episodes=10)
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
    records = list(_run_iter_episodes(monkeypatch, messages, episodes=below_threshold))
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
    records = list(_run_iter_episodes(monkeypatch, messages, episodes=3))
    assert [r["outcome"] for r in records] == [
        "environment_error",
        "terminated",
        "environment_error",
    ]


def test_max_consecutive_failures_below_one_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad_value in (0, -1):
        gen = _run_iter_episodes(monkeypatch, [], episodes=1, max_consecutive_failures=bad_value)
        with pytest.raises(ValueError, match="max_consecutive_failures"):
            next(gen)


def test_missing_action_count_is_contract_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    _drive(monkeypatch, [])
    gen = v05mb.iter_episodes(
        object(), bytearray(), hello={}, episodes=1, max_steps=5, base_seed=0
    )
    with pytest.raises(v05mb.ContractViolation, match="action_count"):
        next(gen)


# --- schema_id verification ----------------------------------------------


def test_schema_id_matching_the_repo_configs_does_not_raise() -> None:
    v05mb._assert_schema_id_matches_repo({"schema_id": _real_schema_id()})


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
    `finally` block unconditionally calls `.close()` on it -- tracked
    here so a test can confirm that happens even on a failure path."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


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
    assert body["per_episode"][0]["episode_id"] == "ep-000001"


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


def test_main_contract_violation_on_a_later_episode_still_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A contract violation discards the whole capture, not just the
    # episode it occurred on -- confirm that holds even after a prior
    # episode already succeeded and was appended to `records`.
    messages = [
        _hello(),
        _obs(),
        _obs(terminated=True),  # episode 1 succeeds
        _error("INVALID_ACTION"),  # episode 2's reset is a contract violation
    ]
    rc, out_path = _run_main(monkeypatch, tmp_path, messages, args=["--episodes", "2"])
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


def test_main_propagates_invalid_max_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    messages = [_hello()]
    with pytest.raises(ValueError, match="max_consecutive_failures"):
        _run_main(
            monkeypatch,
            tmp_path,
            messages,
            args=["--episodes", "1", "--max-consecutive-failures", "0"],
        )


def _raising_recv_text(_sock: object, _buf: object) -> dict[str, Any]:
    msg = "connection lost while reading hello"
    raise RuntimeError(msg)


def test_main_closes_the_socket_even_if_the_hello_read_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _drive(monkeypatch, [])
    monkeypatch.setattr(v05mb, "recv_text", _raising_recv_text)
    fake_socket = _FakeSocket()
    monkeypatch.setattr(v05mb, "open_ws", lambda *_a, **_k: fake_socket)
    out_path = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="connection lost"):
        v05mb.main(["--out", str(out_path), "--episodes", "1"])
    assert fake_socket.closed is True
    assert not out_path.exists()
