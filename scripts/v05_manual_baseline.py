#!/usr/bin/env python3
"""v0.5 first-real-run manual baseline driver.

Stand-in for the (currently mc-live-broken) Rust runner. Drives the
bot's WebSocket protocol directly from Python: sends one `reset`
followed by N `step` messages per episode, samples random actions,
and writes the trajectory + per-episode summary to disk.

The output schema mirrors `forge.training.muzero_mc.capture_baseline`'s
`BaselineRecord` so the existing `mc_plot_baseline.py` can consume
it once the operator pivots to the proper runner-driven flow.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import socket  # noqa: TC003 — runtime use in drive_episode signature
import sys
import time
from collections.abc import Iterator  # noqa: TC003 — runtime use in iter_episodes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from _ws_client import open_ws, recv_text, send_text

logger = logging.getLogger("v05_manual_baseline")

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8766
DEFAULT_EPISODES: Final[int] = 10
DEFAULT_MAX_STEPS_PER_EPISODE: Final[int] = 100
DEFAULT_BASE_SEED: Final[int] = 0xCAFEF00D
DEFAULT_OUT_PATH: Final[str] = "baseline_random_manual.json"


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
            # Server-side per-step error (typically INTERNAL when the
            # bot's mineflayer call timed out / a random action hit an
            # invalid entity). Mark the episode truncated and move on
            # — the next reset re-establishes state without crashing
            # the whole capture.
            protocol_errors += 1
            truncated = True
            step_count = tick + 1
            logger.warning(
                "step %d returned protocol error code=%s message=%s; "
                "truncating episode",
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
    return {
        "total_reward": total_reward,
        "steps": step_count,
        "terminated": terminated,
        "truncated": truncated,
        "protocol_errors": protocol_errors,
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
) -> Iterator[dict[str, Any]]:
    action_count = int(hello["action_count"])
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
        result["episode_id"] = f"ep-{episode_seq:06d}"
        result["seed"] = seed
        yield result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="v05_manual_baseline",
        description="Drive N episodes through the mc-bot WS and dump a v0.5 baseline JSON.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    started = datetime.now(tz=timezone.utc).isoformat()
    started_mono = time.monotonic()
    sock = open_ws(args.host, args.port)
    buf = bytearray()
    hello = recv_text(sock, buf)
    logger.info(
        "Hello: obs_dim=%s action_count=%s grid_shape=%s schema_id=%s",
        hello.get("obs_dim"),
        hello.get("action_count"),
        hello.get("grid_shape"),
        hello.get("schema_id"),
    )

    records: list[dict[str, Any]] = []
    try:
        for record in iter_episodes(
            sock,
            buf,
            hello=hello,
            episodes=args.episodes,
            max_steps=args.max_steps_per_episode,
            base_seed=args.base_seed,
        ):
            logger.info(
                "episode %s: steps=%d reward=%.4f term=%s trunc=%s",
                record["episode_id"],
                record["steps"],
                record["total_reward"],
                record["terminated"],
                record["truncated"],
            )
            records.append(record)
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
    logger.info("wrote %s (%d episodes)", args.out, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
