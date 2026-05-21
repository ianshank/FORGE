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

import base64
import json
import os
import socket
import struct
import sys


def fetch_hello(host: str, port: int, timeout_secs: float = 10.0) -> dict[str, object]:
    """Connect, send a minimal WS upgrade, read one text frame, return it."""
    with socket.create_connection((host, port), timeout=timeout_secs) as sock:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        sock.sendall(request)

        # Drain HTTP response headers up to the blank line.
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                msg = "ws upgrade: server closed before completing handshake"
                raise RuntimeError(msg)
            response += chunk
        header_end = response.index(b"\r\n\r\n") + 4
        body_start = response[header_end:]

        # Read remaining bytes from socket until we have at least one
        # complete frame (text payload). Buffer header bytes already
        # received past the HTTP response.
        buffer = body_start
        deadline_iters = 50
        while len(buffer) < 2 and deadline_iters > 0:
            buffer += sock.recv(4096)
            deadline_iters -= 1
        # WS frame parse: opcode in low 4 bits, masking + length in byte 2.
        # Server frames are NOT masked.
        first, second = buffer[0], buffer[1]
        opcode = first & 0x0F
        if opcode != 0x1:
            msg = f"expected text frame (opcode 0x1), got 0x{opcode:x}"
            raise RuntimeError(msg)
        masked = bool(second & 0x80)
        payload_len = second & 0x7F
        cursor = 2
        if payload_len == 126:
            while len(buffer) < cursor + 2:
                buffer += sock.recv(4096)
            payload_len = struct.unpack(">H", buffer[cursor : cursor + 2])[0]
            cursor += 2
        elif payload_len == 127:
            while len(buffer) < cursor + 8:
                buffer += sock.recv(4096)
            payload_len = struct.unpack(">Q", buffer[cursor : cursor + 8])[0]
            cursor += 8
        if masked:
            mask = buffer[cursor : cursor + 4]
            cursor += 4
        while len(buffer) < cursor + payload_len:
            buffer += sock.recv(4096)
        payload = buffer[cursor : cursor + payload_len]
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return json.loads(payload.decode("utf-8"))


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
    print(f"[probe] connecting to ws://{host}:{port}")
    hello = fetch_hello(host, port)
    print(json.dumps(hello, indent=2))

    # v0.5 acceptance checks.
    assert hello.get("type") == "hello", f"expected type=hello, got {hello.get('type')}"
    obs_dim = hello.get("obs_dim")
    grid_shape = hello.get("grid_shape")
    print(f"\n[probe] obs_dim = {obs_dim}")
    print(f"[probe] grid_shape = {grid_shape}")
    if grid_shape is None:
        print("[probe] WARN: grid_shape is missing (legacy flat-only bot?)")
        return 1
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
        return 2
    if obs_dim == 920:
        print("[probe] PASS: v0.5 Phase 1 contract (11*11*1*7 + 73 = 920) verified end-to-end")
    else:
        print(f"[probe] WARN: obs_dim = {obs_dim}, expected 920 for default v0.5 config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
