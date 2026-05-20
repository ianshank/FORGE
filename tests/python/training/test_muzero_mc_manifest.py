"""Tests for ``forge.training.muzero_mc.manifest``.

Covers:

- Round-trip: save → load gives back an identical dataclass.
- Validation rejects bad schema_version, empty schema_id, empty file
  paths/sha values, zero version.
- ``sha256_file`` matches a known reference hash.
- ``save_manifest`` writes atomically (no orphan ``.tmp`` left behind).
- The JSON written by Python is loadable as a dict whose top-level
  keys match the Rust struct field names exactly (catches drift early
  without spawning a Rust subprocess).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 — used as a runtime fixture type.

import pytest

from forge.training.muzero_mc.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    ModelFileEntry,
    ModelManifest,
    ModelManifestFiles,
    build_manifest,
    load_manifest,
    save_manifest,
    sha256_file,
    utc_now_rfc3339,
)

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "version",
    "schema_id",
    "created_at",
    "files",
}
EXPECTED_FILE_ROLES = {"representation", "dynamics", "prediction"}


def _files_block() -> ModelManifestFiles:
    return ModelManifestFiles(
        representation=ModelFileEntry(path="representation.onnx", sha256="a" * 64),
        dynamics=ModelFileEntry(path="dynamics.onnx", sha256="b" * 64),
        prediction=ModelFileEntry(path="prediction.onnx", sha256="c" * 64),
    )


def _well_formed_manifest(version: int = 1) -> ModelManifest:
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=version,
        schema_id="schema-sha256-abc",
        created_at="2026-05-20T00:00:00Z",
        files=_files_block(),
    )


def test_round_trip_save_then_load(tmp_path: Path) -> None:
    m = _well_formed_manifest(version=7)
    out = save_manifest(m, tmp_path / MANIFEST_FILENAME)
    assert out.exists()
    back = load_manifest(out)
    assert back == m


def test_round_trip_via_build_manifest(tmp_path: Path) -> None:
    # Drop three small files and build a manifest from them.
    for name, content in [
        ("representation.onnx", b"rep-bytes"),
        ("dynamics.onnx", b"dyn-bytes"),
        ("prediction.onnx", b"pred-bytes"),
    ]:
        (tmp_path / name).write_bytes(content)

    m = build_manifest(
        version=3,
        schema_id="abc123",
        files_dir=tmp_path,
    )
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.version == 3
    assert m.files.representation.path == "representation.onnx"
    assert m.files.representation.sha256 == hashlib.sha256(b"rep-bytes").hexdigest()
    assert m.files.dynamics.sha256 == hashlib.sha256(b"dyn-bytes").hexdigest()
    assert m.files.prediction.sha256 == hashlib.sha256(b"pred-bytes").hexdigest()

    saved = save_manifest(m, tmp_path / MANIFEST_FILENAME)
    again = load_manifest(saved)
    assert again == m


def test_validate_rejects_schema_version_drift() -> None:
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION + 1,
        version=1,
        schema_id="sid",
        created_at="t",
        files=_files_block(),
    )
    with pytest.raises(ManifestError, match="schema_version"):
        m.validate()


def test_validate_rejects_zero_version() -> None:
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=0,
        schema_id="sid",
        created_at="t",
        files=_files_block(),
    )
    with pytest.raises(ManifestError, match="version"):
        m.validate()


def test_validate_rejects_empty_schema_id() -> None:
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=1,
        schema_id="",
        created_at="t",
        files=_files_block(),
    )
    with pytest.raises(ManifestError, match="schema_id"):
        m.validate()


@pytest.mark.parametrize("role", sorted(EXPECTED_FILE_ROLES))
def test_validate_rejects_empty_role_path(role: str) -> None:
    files = _files_block()
    bad_entry = ModelFileEntry(path="", sha256="x" * 64)
    kwargs = {r: getattr(files, r) for r in EXPECTED_FILE_ROLES}
    kwargs[role] = bad_entry
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=1,
        schema_id="sid",
        created_at="t",
        files=ModelManifestFiles(**kwargs),
    )
    with pytest.raises(ManifestError, match=f"files.{role}.path"):
        m.validate()


@pytest.mark.parametrize("role", sorted(EXPECTED_FILE_ROLES))
def test_validate_rejects_empty_role_sha(role: str) -> None:
    files = _files_block()
    bad_entry = ModelFileEntry(path="ok.onnx", sha256="")
    kwargs = {r: getattr(files, r) for r in EXPECTED_FILE_ROLES}
    kwargs[role] = bad_entry
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=1,
        schema_id="sid",
        created_at="t",
        files=ModelManifestFiles(**kwargs),
    )
    with pytest.raises(ManifestError, match=f"files.{role}.sha256"):
        m.validate()


def test_save_manifest_refuses_to_persist_invalid_manifest(tmp_path: Path) -> None:
    bad = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=0,  # invalid
        schema_id="x",
        created_at="t",
        files=_files_block(),
    )
    out = tmp_path / MANIFEST_FILENAME
    with pytest.raises(ManifestError):
        save_manifest(bad, out)
    assert not out.exists(), "no file should be written when validation fails"


def test_save_is_atomic_no_orphan_tmp(tmp_path: Path) -> None:
    m = _well_formed_manifest()
    save_manifest(m, tmp_path / MANIFEST_FILENAME)
    siblings = [p.name for p in tmp_path.iterdir()]
    assert all(not n.startswith(".tmp-") for n in siblings), siblings


def test_sha256_file_matches_reference(tmp_path: Path) -> None:
    content = b"the quick brown fox jumps over the lazy dog\n"
    f = tmp_path / "sample.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_json_top_level_keys_match_rust_schema(tmp_path: Path) -> None:
    """Catch field-name drift between Python and Rust without spawning
    a Rust subprocess.

    The Rust struct field names are stable (``schema_version``,
    ``version``, ``schema_id``, ``created_at``, ``files``); if either
    side renames or adds a field this test fails.
    """
    m = _well_formed_manifest()
    out = save_manifest(m, tmp_path / MANIFEST_FILENAME)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS
    assert set(payload["files"].keys()) == EXPECTED_FILE_ROLES
    for role in EXPECTED_FILE_ROLES:
        assert set(payload["files"][role].keys()) == {"path", "sha256"}


def test_load_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_load_invalid_json_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / MANIFEST_FILENAME
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_utc_now_rfc3339_round_trip_parses() -> None:
    from datetime import datetime

    s = utc_now_rfc3339()
    assert s.endswith("Z"), s
    # Confirm Python can parse what Python wrote.
    parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
