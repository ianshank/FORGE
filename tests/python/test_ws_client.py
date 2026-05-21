"""Tests for `scripts/_ws_client.py` — the stdlib-only RFC 6455 frame
parser shared by `scripts/v05_handshake_probe.py` and
`scripts/v05_manual_baseline.py`.

Focus areas:
- `recv_text` happy-path (text frame round-trip)
- Security HIGH-1: the `DEFAULT_MAX_FRAME_BYTES = 64 MiB` cap fires
  before the recv-loop allocates a u64-max payload_len header
- `recv_text` close-frame propagates as RuntimeError
- `send_text` masking + extended-payload-length encoding
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest


# Make the `scripts/` shim importable by tests without adding it to
# `sys.path` permanently — pytest adds the test file's directory,
# so we extend with the project's `scripts/` once at import time.
def _ensure_scripts_on_path() -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


_ensure_scripts_on_path()


from _ws_client import (  # noqa: E402 — import after sys.path tweak
    DEFAULT_MAX_FRAME_BYTES,
    EXT_PAYLOAD_LEN_16,
    EXT_PAYLOAD_LEN_64,
    FIN_TEXT_FRAME,
    MASK_BIT,
    OPCODE_CLOSE,
    OPCODE_TEXT,
    recv_text,
    send_text,
)


class _FakeSocket:
    """Stub socket whose `recv` drains a pre-loaded byte buffer.

    Captures every `sendall` payload for assertion. Mirrors just
    enough of `socket.socket` for the RFC 6455 client routines.
    """

    def __init__(self, recv_buffer: bytes = b"") -> None:
        self.recv_buffer = bytearray(recv_buffer)
        self.sent: list[bytes] = []
        self.closed = False

    def recv(self, n: int) -> bytes:
        if not self.recv_buffer:
            return b""  # signal EOF
        chunk = bytes(self.recv_buffer[:n])
        del self.recv_buffer[:n]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    def close(self) -> None:
        self.closed = True


def _server_text_frame(payload: bytes) -> bytes:
    """Build an unmasked server-to-client text frame."""
    header = bytearray([FIN_TEXT_FRAME])
    length = len(payload)
    if length < EXT_PAYLOAD_LEN_16:
        header.append(length)
    elif length < 1 << 16:
        header.append(EXT_PAYLOAD_LEN_16)
        header.extend(struct.pack(">H", length))
    else:
        header.append(EXT_PAYLOAD_LEN_64)
        header.extend(struct.pack(">Q", length))
    return bytes(header) + payload


# --- recv_text happy-path -----------------------------------------


def test_recv_text_decodes_short_text_frame() -> None:
    payload = json.dumps({"type": "hello", "obs_dim": 920}).encode("utf-8")
    sock = _FakeSocket(_server_text_frame(payload))
    decoded = recv_text(sock, bytearray())  # type: ignore[arg-type]
    assert decoded == {"type": "hello", "obs_dim": 920}


def test_recv_text_decodes_extended_16bit_payload() -> None:
    big_payload = json.dumps({"data": "x" * 1000}).encode("utf-8")
    assert len(big_payload) >= EXT_PAYLOAD_LEN_16
    sock = _FakeSocket(_server_text_frame(big_payload))
    decoded = recv_text(sock, bytearray())  # type: ignore[arg-type]
    assert decoded["data"] == "x" * 1000


def test_recv_text_back_to_back_frames_share_buffer() -> None:
    """The buffer-overflow invariant: bytes belonging to frame N+1
    that landed during frame N's recv must NOT be discarded."""
    payload_a = json.dumps({"frame": "a"}).encode("utf-8")
    payload_b = json.dumps({"frame": "b"}).encode("utf-8")
    sock = _FakeSocket(_server_text_frame(payload_a) + _server_text_frame(payload_b))
    buf = bytearray()
    assert recv_text(sock, buf) == {"frame": "a"}  # type: ignore[arg-type]
    assert recv_text(sock, buf) == {"frame": "b"}  # type: ignore[arg-type]


# --- recv_text security: HIGH-1 DoS cap ---------------------------


def test_recv_text_rejects_payload_len_exceeding_cap() -> None:
    """Security audit HIGH-1: a hostile server claiming
    `payload_len > DEFAULT_MAX_FRAME_BYTES` must be rejected BEFORE
    the recv-loop attempts to allocate. Crafts a frame whose 8-byte
    extended payload-length header advertises 1 byte over the cap.
    """
    over_cap = DEFAULT_MAX_FRAME_BYTES + 1
    header = bytearray([FIN_TEXT_FRAME, EXT_PAYLOAD_LEN_64])
    header.extend(struct.pack(">Q", over_cap))
    sock = _FakeSocket(bytes(header))
    with pytest.raises(RuntimeError, match="exceeds cap"):
        recv_text(sock, bytearray())  # type: ignore[arg-type]


def test_recv_text_accepts_payload_len_at_cap_boundary() -> None:
    """The cap is an upper bound, not a strict-less-than gate. Frames
    AT the cap are valid; only > cap is refused. Verifies the
    comparison is `> DEFAULT_MAX_FRAME_BYTES`, not `>=`.
    """
    # We can't actually transmit DEFAULT_MAX_FRAME_BYTES of data
    # in this test (64 MiB), but we can craft a frame whose header
    # advertises payload_len = cap and then close the socket so
    # the recv loop exits cleanly.  The cap-check must not fire.
    at_cap = DEFAULT_MAX_FRAME_BYTES
    header = bytearray([FIN_TEXT_FRAME, EXT_PAYLOAD_LEN_64])
    header.extend(struct.pack(">Q", at_cap))
    sock = _FakeSocket(bytes(header))  # no payload follows
    # The cap check passes (at == cap is OK); the recv loop hits EOF
    # while waiting for the payload → RuntimeError("ws closed mid-payload").
    # The KEY assertion is that the error is NOT the cap-rejection.
    with pytest.raises(RuntimeError) as excinfo:
        recv_text(sock, bytearray())  # type: ignore[arg-type]
    assert "exceeds cap" not in str(excinfo.value)
    assert "closed mid-payload" in str(excinfo.value)


# --- recv_text close-frame propagation -----------------------------


def test_recv_text_close_frame_raises_runtime_error() -> None:
    close_frame = bytes([OPCODE_CLOSE | 0x80, 0])  # FIN + close, zero-len payload
    sock = _FakeSocket(close_frame)
    with pytest.raises(RuntimeError, match="server sent close frame"):
        recv_text(sock, bytearray())  # type: ignore[arg-type]


def test_recv_text_unexpected_opcode_raises_runtime_error() -> None:
    # Opcode 0x9 (ping) is not one we expect from the bot
    ping_frame = bytes([0x80 | 0x9, 0])
    sock = _FakeSocket(ping_frame)
    with pytest.raises(RuntimeError, match=f"opcode 0x{OPCODE_TEXT:x}"):
        recv_text(sock, bytearray())  # type: ignore[arg-type]


def test_recv_text_non_object_payload_raises_runtime_error() -> None:
    """JSON arrays / scalars at the payload root are protocol errors
    — the bot always wraps in `{...}`."""
    array_payload = b"[1, 2, 3]"
    sock = _FakeSocket(_server_text_frame(array_payload))
    with pytest.raises(RuntimeError, match="expected JSON object"):
        recv_text(sock, bytearray())  # type: ignore[arg-type]


# --- send_text masking + framing pins ------------------------------


def test_send_text_emits_masked_short_payload() -> None:
    sock = _FakeSocket()
    send_text(sock, {"type": "reset"})  # type: ignore[arg-type]
    assert len(sock.sent) == 1
    frame = sock.sent[0]
    # First byte: FIN + text opcode
    assert frame[0] == FIN_TEXT_FRAME
    # Second byte: MASK_BIT + length
    body_bytes = b'{"type": "reset"}'
    assert frame[1] == (MASK_BIT | len(body_bytes))
    # Bytes 2..5 are the mask; bytes 6.. are masked payload
    mask = frame[2:6]
    masked_payload = frame[6:]
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(masked_payload))
    assert unmasked == body_bytes


def test_send_text_extended_16bit_length_for_large_payloads() -> None:
    sock = _FakeSocket()
    big: dict[str, Any] = {"data": "x" * 1000}
    send_text(sock, big)  # type: ignore[arg-type]
    frame = sock.sent[0]
    # Second byte's low 7 bits == EXT_PAYLOAD_LEN_16 sentinel
    assert (frame[1] & 0x7F) == EXT_PAYLOAD_LEN_16
    declared_len = struct.unpack(">H", frame[2:4])[0]
    # JSON-encoded body length matches declared length
    assert declared_len == len(json.dumps(big).encode("utf-8"))
