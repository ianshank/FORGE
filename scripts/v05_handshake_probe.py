#!/usr/bin/env python3
"""v0.5 first-real-run handshake probe.

Connects to the running mc-bot WebSocket endpoint, reads the initial
``Hello`` server message, and dumps it to stdout. Validates that the
v0.5 ``grid_shape`` payload + ``obs_dim`` math line up with the
``configs/minecraft/env.toml`` defaults.

Standalone — uses stdlib + `websockets` if available, falls back to a
hand-rolled WebSocket client over raw sockets so the probe works
without `pip install websockets`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Final

from _ws_client import open_ws, recv_text

# Single source-of-truth defaults — kept aligned with
# `configs/minecraft/env.toml` and `docker/compose.minecraft.env.example`
# per the CLAUDE.md "no hard-coded values" rule. Operators override
# via CLI args (`v05_handshake_probe.py <host> <port>`).
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8766
# v0.5 Phase 1 contract: 11x11x1x7 grid (847 floats) + 73 flat = 920.
# Drift here means env.toml's [observation] knobs were changed without
# updating the probe's acceptance gate.
EXPECTED_OBS_DIM: Final[int] = 920
# Exit-code convention matches the rest of the v0.5 CLI surface
# (forge.training.muzero_mc.cli::EXIT_OK / EXIT_VALIDATION / EXIT_IO).
EXIT_OK: Final[int] = 0
EXIT_UNEXPECTED_TYPE: Final[int] = 2
EXIT_GRID_SHAPE_MISSING: Final[int] = 3
EXIT_GRID_SHAPE_MISMATCH: Final[int] = 4


def fetch_hello(host: str, port: int) -> dict[str, Any]:
    """Connect to the bot's WS, read the initial Hello frame."""
    sock = open_ws(host, port)
    try:
        return recv_text(sock, bytearray())
    finally:
        sock.close()


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    print(f"[probe] connecting to ws://{host}:{port}")
    hello = fetch_hello(host, port)
    print(json.dumps(hello, indent=2))

    # v0.5 acceptance checks. Replaces the previous `assert` calls
    # so the gate stays effective under `python -O` (which strips
    # asserts).
    if hello.get("type") != "hello":
        print(f"[probe] FAIL: expected type=hello, got {hello.get('type')!r}")
        return EXIT_UNEXPECTED_TYPE
    obs_dim = hello.get("obs_dim")
    grid_shape = hello.get("grid_shape")
    print(f"\n[probe] obs_dim = {obs_dim}")
    print(f"[probe] grid_shape = {grid_shape}")
    if grid_shape is None:
        print("[probe] WARN: grid_shape is missing (legacy flat-only bot?)")
        return EXIT_GRID_SHAPE_MISSING
    derived = (
        grid_shape["height"]
        * grid_shape["width"]
        * grid_shape["depth"]
        * grid_shape["channels"]
        + grid_shape.get("vector_dim", 0)
    )
    print(f"[probe] derived obs_dim from grid_shape = {derived}")
    if derived != obs_dim:
        print(f"[probe] FAIL: derived {derived} != advertised {obs_dim}")
        return EXIT_GRID_SHAPE_MISMATCH
    if obs_dim == EXPECTED_OBS_DIM:
        print(
            f"[probe] PASS: v0.5 Phase 1 contract "
            f"(grid {grid_shape['height']}*{grid_shape['width']}*"
            f"{grid_shape['depth']}*{grid_shape['channels']} + "
            f"{grid_shape['vector_dim']} = {EXPECTED_OBS_DIM}) "
            f"verified end-to-end"
        )
    else:
        print(
            f"[probe] WARN: obs_dim = {obs_dim}, expected "
            f"{EXPECTED_OBS_DIM} for default v0.5 config"
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
