"""Minimal stdlib-only WebSocket client.

Extracted from `scripts/v05_handshake_probe.py` and
`scripts/v05_manual_baseline.py` (peer-review S3) so the two
scripts share one frame parser. Implements just enough RFC 6455
to drive the mc-bot WS protocol from a Python subprocess without
adding `websockets` or `httpx` as a hard dependency.

Public surface:
- ``open_ws(host, port, timeout_secs=...) -> socket.socket``
- ``send_text(sock, payload)``
- ``recv_text(sock, buf) -> dict[str, Any]``
- ``DEFAULT_*`` constants for timeout / buffer sizing
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
from typing import Any, Final

#: Default upgrade-handshake timeout. The mc-bot accepts the WS
#: upgrade synchronously so anything beyond a couple of seconds means
#: the bot container is wedged.
DEFAULT_HANDSHAKE_TIMEOUT_SECS: Final[float] = 10.0
#: Socket recv() chunk size for frame-buffer growth. Matches the
#: stdlib default for buffered IO so behaviour is predictable.
DEFAULT_RECV_CHUNK_SIZE: Final[int] = 65536
#: Hard cap on a single inbound frame's payload size (64 MiB). The
#: mc-bot's largest realistic frame is the 920-float observation +
#: a small info-blob (~10-30 KiB); 64 MiB is several orders of
#: magnitude above that and below the host-RAM exhaustion threshold.
#: A hostile bot could otherwise send a u64-typed payload_len header
#: (~9 EiB) and crash this client via the `while len(buf) <
#: cursor + payload_len` allocation loop. Security audit HIGH-1.
DEFAULT_MAX_FRAME_BYTES: Final[int] = 64 * 1024 * 1024
#: Short frame-length sentinel: payload_len in [126, 65535] uses 2
#: extended bytes; payload_len > 65535 uses 8. Pinned per RFC 6455.
EXT_PAYLOAD_LEN_16: Final[int] = 126
EXT_PAYLOAD_LEN_64: Final[int] = 127
#: Frame opcodes we care about. The bot only ever sends text + close.
OPCODE_TEXT: Final[int] = 0x1
OPCODE_CLOSE: Final[int] = 0x8
#: WS spec FIN + masked-from-client bits.
FIN_TEXT_FRAME: Final[int] = 0x81
MASK_BIT: Final[int] = 0x80


def open_ws(
    host: str,
    port: int,
    *,
    timeout_secs: float = DEFAULT_HANDSHAKE_TIMEOUT_SECS,
) -> socket.socket:
    """Connect + perform the RFC 6455 client-side upgrade handshake.

    Drains the HTTP response headers and returns the underlying
    socket ready for `send_text` / `recv_text`.
    """
    sock = socket.create_connection((host, port), timeout=timeout_secs)
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
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            msg = "ws upgrade: server closed before completing handshake"
            raise RuntimeError(msg)
        response += chunk
    return sock


def send_text(sock: socket.socket, payload: dict[str, Any]) -> None:
    """Encode `payload` as a single masked text frame and write it."""
    body = json.dumps(payload).encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(body))
    header = bytearray()
    header.append(FIN_TEXT_FRAME)
    length = len(body)
    if length < EXT_PAYLOAD_LEN_16:
        header.append(MASK_BIT | length)
    elif length < 1 << 16:
        header.append(MASK_BIT | EXT_PAYLOAD_LEN_16)
        header.extend(struct.pack(">H", length))
    else:
        header.append(MASK_BIT | EXT_PAYLOAD_LEN_64)
        header.extend(struct.pack(">Q", length))
    header.extend(mask)
    sock.sendall(bytes(header) + masked)


def recv_text(sock: socket.socket, buf: bytearray) -> dict[str, Any]:  # noqa: PLR0912
    """Read one text frame from `sock`, decoding the JSON payload.

    Mutates `buf` to retain any over-read bytes that belong to the
    next frame, so back-to-back `recv_text` calls don't lose data.

    PLR0912 (too many branches): RFC 6455 frame parsing requires
    distinct branches for opcode classification, mask presence, and
    three extended-payload-length sizes; splitting into smaller
    helpers obscures the spec mapping. Suppression is preferable
    to a forced refactor.
    """
    while len(buf) < 2:
        chunk = sock.recv(DEFAULT_RECV_CHUNK_SIZE)
        if not chunk:
            msg = "ws closed mid-frame"
            raise RuntimeError(msg)
        buf.extend(chunk)
    first, second = buf[0], buf[1]
    opcode = first & 0x0F
    if opcode == OPCODE_CLOSE:
        msg = "server sent close frame"
        raise RuntimeError(msg)
    if opcode != OPCODE_TEXT:
        msg = f"expected text frame (opcode 0x{OPCODE_TEXT:x}), got 0x{opcode:x}"
        raise RuntimeError(msg)
    masked = bool(second & MASK_BIT)
    payload_len = second & 0x7F
    cursor = 2
    if payload_len == EXT_PAYLOAD_LEN_16:
        while len(buf) < cursor + 2:
            buf.extend(sock.recv(DEFAULT_RECV_CHUNK_SIZE))
        payload_len = struct.unpack(">H", bytes(buf[cursor : cursor + 2]))[0]
        cursor += 2
    elif payload_len == EXT_PAYLOAD_LEN_64:
        while len(buf) < cursor + 8:
            buf.extend(sock.recv(DEFAULT_RECV_CHUNK_SIZE))
        payload_len = struct.unpack(">Q", bytes(buf[cursor : cursor + 8]))[0]
        cursor += 8
    # Hostile-server DoS guard (security audit HIGH-1): refuse
    # frames larger than DEFAULT_MAX_FRAME_BYTES BEFORE the
    # recv-loop tries to allocate them.
    if payload_len > DEFAULT_MAX_FRAME_BYTES:
        msg = (
            f"frame payload_len={payload_len} exceeds cap "
            f"{DEFAULT_MAX_FRAME_BYTES}; refusing to allocate"
        )
        raise RuntimeError(msg)
    mask = bytes(buf[cursor : cursor + 4]) if masked else None
    if masked:
        cursor += 4
    while len(buf) < cursor + payload_len:
        chunk = sock.recv(DEFAULT_RECV_CHUNK_SIZE)
        if not chunk:
            msg = "ws closed mid-payload"
            raise RuntimeError(msg)
        buf.extend(chunk)
    payload = bytes(buf[cursor : cursor + payload_len])
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    del buf[: cursor + payload_len]
    decoded: Any = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        msg = f"expected JSON object payload, got {type(decoded).__name__}"
        raise RuntimeError(msg)
    return decoded
