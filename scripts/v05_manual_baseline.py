#!/usr/bin/env python3
"""v0.5 first-real-run manual baseline driver.

Stand-in for the (currently mc-live-broken) Rust runner. Drives the
bot's WebSocket protocol directly from Python: sends one `reset`
followed by N `step` messages per episode, samples random actions,
and writes the trajectory + per-episode summary to disk.

The output schema mirrors `forge.training.muzero_mc.capture_baseline`'s
`BaselineRecord` so the existing `mc_plot_baseline.py` can consume
it once the operator pivots to the proper runner-driven flow.

Outcome classification (openspec/changes/refuse-non-evidential-aggregates/):
an environment-reported step failure is recorded as a distinct
``"environment_error"`` outcome rather than by setting `truncated`,
so it cannot be mistaken for a real completion. An error code meaning
the action space or wire shape disagrees across languages (Invariant 2
territory) aborts the whole capture immediately rather than being
recorded as one more failed episode. The transient error codes instead
halt the capture once they occur on `DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES`
consecutive episodes, so a wedged bot cannot silently burn through the
entire configured episode budget. Before any episode runs, the bot's
handshake `schema_id` is checked against a recomputation over this
repo's own `configs/minecraft/{action_map,rewards}.toml` -- a mismatch
means the bot's configs have drifted from this repo's and the capture
refuses to proceed.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import logging
import random
import socket  # noqa: TC003 — runtime use in drive_episode signature
import sys
import time
from collections.abc import Iterator  # noqa: TC003 — runtime use in iter_episodes
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from _ws_client import open_ws, recv_text, send_text

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger("v05_manual_baseline")

DEFAULT_HOST: Final[str] = "127.0.0.1"
# Matches `configs/minecraft/env.toml` (local-dev) and
# `docker/compose.minecraft.env.example` (`MC_BOT_WS_PORT=8765`).
DEFAULT_PORT: Final[int] = 8765
DEFAULT_EPISODES: Final[int] = 10
DEFAULT_MAX_STEPS_PER_EPISODE: Final[int] = 100
DEFAULT_BASE_SEED: Final[int] = 0xCAFEF00D
DEFAULT_OUT_PATH: Final[str] = "baseline_random_manual.json"
# Pinned in tests/python/test_v05_manual_baseline.py -- lowering this
# changes how much sustained bot failure this driver tolerates before
# giving up, so a change to it should be visible in review, not silent.
DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES: Final[int] = 3

EXIT_OK: Final[int] = 0
EXIT_SCHEMA_MISMATCH: Final[int] = 3
EXIT_CONTRACT_VIOLATION: Final[int] = 4
EXIT_SUSTAINED_FAILURE: Final[int] = 5

# The bot's own error-code vocabulary (mc-bot/src/index.ts). Only these
# three are transient environment-health signals; everything else --
# including the two the bot documents (BAD_MESSAGE, INVALID_ACTION) and
# any code this script doesn't recognise at all -- means the action
# space or wire shape disagrees across languages, or that the bot's
# error vocabulary has drifted from what this script was written
# against. Either way that is Invariant 2 territory: fail closed on
# the unknown rather than silently treating it as just another
# transient fault.
_TRANSIENT_ENVIRONMENT_CODES: Final[frozenset[str]] = frozenset(
    {"BUSY", "RECONNECTING", "INTERNAL"}
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_SCHEMA_ID_MODULE_PATH: Final[Path] = (
    REPO_ROOT / "python" / "forge" / "training" / "muzero_mc" / "schema_id.py"
)
_REPLAY_MODULE_PATH: Final[Path] = (
    REPO_ROOT / "python" / "forge" / "training" / "muzero_mc" / "replay.py"
)
ACTION_MAP_PATH: Final[Path] = REPO_ROOT / "configs" / "minecraft" / "action_map.toml"
REWARDS_PATH: Final[Path] = REPO_ROOT / "configs" / "minecraft" / "rewards.toml"


class ContractViolation(RuntimeError):
    """The bot reported a non-transient error: the action space or wire
    shape disagrees across languages. Aborts the capture; no output
    file is written, since a partial capture past this point would be
    evidence of a broken configuration, not of the system under test.
    """


class SustainedEnvironmentFailure(RuntimeError):
    """The environment failed on `DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES`
    consecutive episodes. Halts the capture; the episodes already
    captured (including the failing ones) are still written, since
    they are legitimate evidence of what happened.
    """


class SchemaIdMismatch(RuntimeError):
    """The bot's handshake `schema_id` disagrees with a recomputation
    over this repo's own `configs/minecraft/{action_map,rewards}.toml`.
    Raised before any episode runs; no output file is written.
    """


def _load_module_by_path(path: Path, module_name: str, *, what: str) -> ModuleType:
    """Load a module directly by file path, bypassing any parent
    package's `__init__` chain -- both modules loaded through this
    helper live under `forge.training`, whose `__init__` unconditionally
    imports the MLflow/W&B/TensorBoard-backed logger factories this
    stdlib-only driver has no other reason to need.

    Registers the module in `sys.modules` before executing it -- the
    standard pattern for by-path loading, and required here: a
    `@dataclass` in the loaded module (`replay.py`'s `StepBatch`)
    resolves its lazily-stringified annotations via
    `sys.modules[cls.__module__]`, which raises `AttributeError` on a
    module that was `exec_module`-run without ever being registered.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load {what} from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def _load_schema_id_module() -> ModuleType:
    """Load ``schema_id.py`` by path. It only imports `hashlib`, `json`,
    and `tomllib`, so this costs nothing beyond the package-init
    isolation `_load_module_by_path` exists for.
    """
    return _load_module_by_path(
        _SCHEMA_ID_MODULE_PATH, "_forge_schema_id", what="schema_id module"
    )


@functools.lru_cache(maxsize=1)
def _load_replay_module() -> ModuleType:
    """Load ``replay.py`` by path, for its cross-language-pinned
    `format_episode_id` -- the episode-id format this script writes
    must match `forge_mc_runner::format_episode_id` on the Rust side,
    not redefine it locally. `replay.py`'s own heavy dependency
    (`torch`) is deferred past module scope, so loading it is cheap.
    """
    return _load_module_by_path(
        _REPLAY_MODULE_PATH, "_forge_replay", what="replay module"
    )


def _assert_schema_id_matches_repo(hello: dict[str, Any]) -> None:
    reported = hello.get("schema_id")
    module = _load_schema_id_module()
    expected = module.compute_schema_id_from_paths(ACTION_MAP_PATH, REWARDS_PATH)
    if reported != expected:
        msg = (
            f"handshake schema_id={reported!r} does not match a recomputation "
            f"over {ACTION_MAP_PATH} + {REWARDS_PATH} ({expected!r}); the bot's "
            "configs have drifted from this repo's, or the schema changed on "
            "one side without the other -- refusing to capture"
        )
        raise SchemaIdMismatch(msg)


def _raise_if_contract_violation(msg: dict[str, Any], *, context: str) -> None:
    code = msg.get("code")
    if code not in _TRANSIENT_ENVIRONMENT_CODES:
        detail = msg.get("message")
        error_msg = f"{context}: non-transient error code={code!r} message={detail!r}"
        raise ContractViolation(error_msg)


def drive_episode(
    sock: socket.socket,
    buf: bytearray,
    *,
    action_count: int,
    max_steps: int,
    seed: int,
    rng: random.Random,
) -> dict[str, Any]:
    send_text(sock, {"type": "reset", "seed": seed})
    obs_msg = recv_text(sock, buf)
    if obs_msg.get("type") == "error":
        _raise_if_contract_violation(obs_msg, context="reset")
        logger.warning(
            "reset returned protocol error code=%s message=%s; skipping episode",
            obs_msg.get("code"),
            obs_msg.get("message"),
        )
        return {
            "total_reward": 0.0,
            "steps": 0,
            "terminated": False,
            "truncated": False,
            "protocol_errors": 1,
            "outcome": "environment_error",
            "last_tick": 0,
            "obs_dim": 0,
        }
    if obs_msg.get("type") != "observation":
        msg = f"expected observation after reset, got {obs_msg.get('type')}"
        raise RuntimeError(msg)
    total_reward = 0.0
    last_msg = obs_msg
    terminated = bool(obs_msg.get("terminated"))
    truncated = bool(obs_msg.get("truncated"))
    step_count = 0
    protocol_errors = 0
    for tick in range(max_steps):
        action_id = rng.randrange(action_count)
        send_text(sock, {"type": "step", "action_id": action_id})
        last_msg = recv_text(sock, buf)
        msg_type = last_msg.get("type")
        if msg_type == "error":
            _raise_if_contract_violation(last_msg, context=f"step {tick}")
            # Transient environment-health error (BUSY / RECONNECTING /
            # INTERNAL): the environment did not execute this step, so
            # it does not count toward `step_count` -- that keeps
            # whatever value the last successfully-executed step set.
            # Recorded via protocol_errors, NOT by setting `truncated`
            # -- that flag's other meaning is "reached the step
            # budget", and this episode did not.
            protocol_errors += 1
            logger.warning(
                "step %d returned protocol error code=%s message=%s; ending episode",
                tick,
                last_msg.get("code"),
                last_msg.get("message"),
            )
            break
        if msg_type != "observation":
            msg = f"expected observation after step, got {msg_type!r}"
            raise RuntimeError(msg)
        total_reward += float(last_msg.get("reward", 0.0))
        terminated = bool(last_msg.get("terminated"))
        truncated = bool(last_msg.get("truncated"))
        step_count = tick + 1
        if terminated or truncated:
            break
    else:
        # The loop ran to completion without the environment ever
        # reporting terminated or truncated: this driver's own step
        # budget was reached first. That is itself a truncation, at
        # the driver level rather than the environment's.
        truncated = True

    if protocol_errors > 0:
        outcome = "environment_error"
    elif terminated:
        outcome = "terminated"
    else:
        outcome = "truncated"

    return {
        "total_reward": total_reward,
        "steps": step_count,
        "terminated": terminated,
        "truncated": truncated,
        "protocol_errors": protocol_errors,
        "outcome": outcome,
        "last_tick": int(last_msg.get("tick", 0)),
        "obs_dim": len(last_msg.get("obs", [])),
    }


def iter_episodes(
    sock: socket.socket,
    buf: bytearray,
    *,
    hello: dict[str, Any],
    episodes: int,
    max_steps: int,
    base_seed: int,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES,
) -> Iterator[dict[str, Any]]:
    if max_consecutive_failures < 1:
        msg = (
            "max_consecutive_failures must be >= 1, got "
            f"{max_consecutive_failures}"
        )
        raise ValueError(msg)
    try:
        action_count = int(hello["action_count"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"handshake missing a usable action_count: {hello.get('action_count')!r}"
        raise ContractViolation(msg) from exc
    format_episode_id = _load_replay_module().format_episode_id
    consecutive_failures = 0
    for episode_seq in range(1, episodes + 1):
        seed = base_seed + episode_seq
        rng = random.Random(seed)
        result = drive_episode(
            sock,
            buf,
            action_count=action_count,
            max_steps=max_steps,
            seed=seed,
            rng=rng,
        )
        result["episode_id"] = format_episode_id(episode_seq)
        result["seed"] = seed
        yield result
        if result["outcome"] == "environment_error":
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                msg = (
                    f"{consecutive_failures} consecutive environment failures "
                    f"through episode {result['episode_id']}; halting capture"
                )
                raise SustainedEnvironmentFailure(msg)
        else:
            consecutive_failures = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="v05_manual_baseline",
        description="Drive N episodes through the mc-bot WS and dump a v0.5 baseline JSON.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_ENV_FAILURES,
        help="Halt the capture after this many consecutive environment-failure episodes.",
    )
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    started = datetime.now(tz=timezone.utc).isoformat()
    started_mono = time.monotonic()
    sock = open_ws(args.host, args.port)
    buf = bytearray()

    records: list[dict[str, Any]] = []
    result_code = EXIT_OK
    try:
        # Read inside `try` (not before it): a dropped or malformed
        # handshake then still reaches `finally` and closes `sock`,
        # instead of leaking it on an exception raised before the
        # block began.
        hello = recv_text(sock, buf)
        logger.info(
            "Hello: obs_dim=%s action_count=%s grid_shape=%s schema_id=%s",
            hello.get("obs_dim"),
            hello.get("action_count"),
            hello.get("grid_shape"),
            hello.get("schema_id"),
        )
        _assert_schema_id_matches_repo(hello)
        for record in iter_episodes(
            sock,
            buf,
            hello=hello,
            episodes=args.episodes,
            max_steps=args.max_steps_per_episode,
            base_seed=args.base_seed,
            max_consecutive_failures=args.max_consecutive_failures,
        ):
            logger.info(
                "episode %s: steps=%d reward=%.4f outcome=%s",
                record["episode_id"],
                record["steps"],
                record["total_reward"],
                record["outcome"],
            )
            records.append(record)
    except SchemaIdMismatch as exc:
        logger.error("%s", exc)
        return EXIT_SCHEMA_MISMATCH
    except ContractViolation as exc:
        logger.error("%s", exc)
        return EXIT_CONTRACT_VIOLATION
    except SustainedEnvironmentFailure as exc:
        logger.error("%s", exc)
        result_code = EXIT_SUSTAINED_FAILURE
    finally:
        with contextlib.suppress(OSError):
            send_text(sock, {"type": "close"})
        sock.close()

    ended = datetime.now(tz=timezone.utc).isoformat()
    # Schema-compat with `forge.training.muzero_mc.capture_baseline`'s
    # snapshot — empty defaults for the gauge/counter blocks so
    # `scripts/mc_plot_baseline.py` can consume the file without
    # `KeyError`. Reviewer S4.
    snapshot = {
        "variant": "random",
        "source": "v05_manual_baseline.py",
        "started_at": started,
        "ended_at": ended,
        "duration_secs": time.monotonic() - started_mono,
        "hello": hello,
        "episodes_target": args.episodes,
        "episodes_observed": len(records),
        "trajectory_dir": str(args.out.parent),
        "manifest_versions_seen": [],
        "summary_counters": {
            "forge_mc_episode_total": float(len(records)),
            "forge_mc_steps_total": float(sum(r.get("steps", 0) for r in records)),
            "forge_mc_protocol_errors_total": float(
                sum(r.get("protocol_errors", 0) for r in records)
            ),
        },
        "summary_gauges": {"forge_mc_model_version": None},
        "prometheus_snapshot": "",
        "per_episode": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    logger.info("wrote %s (%d episodes, exit=%d)", args.out, len(records), result_code)
    return result_code


if __name__ == "__main__":
    sys.exit(main())
