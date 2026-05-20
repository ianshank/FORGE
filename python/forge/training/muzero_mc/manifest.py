"""Python mirror of ``forge_mc_runner::ModelManifest``.

This module is the Python side of the model hot-reload contract. The
Rust runner (``crates/forge-mc-runner/src/manifest.rs``) and this
module MUST agree on the JSON shape; any drift breaks the
``schema_version`` cross-check at runtime.

Cross-language constants:

- :data:`MANIFEST_SCHEMA_VERSION` — pinned at ``1``; bumping is a
  breaking change (Rust + Python + any other consumer must bump in
  lockstep).
- :data:`ONNX_OPSET_VERSION` — ONNX opset used by the bootstrap and
  trainer exports. Pinned alongside the Rust ``onnxruntime`` version
  so cross-runtime tests catch opset drift.

Atomic write semantics match the Rust side: writes go to a ``.tmp``
sibling and are then renamed into place. The Rust ``HotReloadWatcher``
relies on this so it never observes a half-written manifest.

No hard-coded values: every dimension, hash, and file role flows
through the :class:`ModelManifest` / :class:`ModelManifestFiles`
dataclasses. Call sites construct manifests via :func:`build_manifest`
or by assembling the dataclasses directly — no raw dict literals.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_BUNDLE_FILENAMES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ONNX_OPSET_VERSION",
    "ModelFileEntry",
    "ModelManifest",
    "ModelManifestFiles",
    "build_manifest",
    "load_manifest",
    "save_manifest",
    "sha256_file",
    "utc_now_rfc3339",
]

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Cross-language constants
# ---------------------------------------------------------------------

#: Pinned manifest schema version. Mirrors the Rust constant
#: ``forge_mc_runner::manifest::MANIFEST_SCHEMA_VERSION``. Bumping is a
#: breaking change.
MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: Default ONNX opset for exports. Pinned alongside ``onnxruntime``
#: pulled in by the ``minecraft`` optional-deps group. A round-trip CI
#: test should fail if this drifts away from what the Rust loader
#: accepts.
ONNX_OPSET_VERSION: Final[int] = 17

#: Canonical filename the Rust runner watches for. Bootstrap and
#: trainer exports should write to this name in the bundle directory.
MANIFEST_FILENAME: Final[str] = "model_manifest.json"

#: Default per-role ONNX filenames inside a bundle directory. Callers
#: can override on a per-role basis via :class:`ModelManifestFiles`,
#: but defaulting through this dict keeps every export consistent.
DEFAULT_BUNDLE_FILENAMES: Final[dict[str, str]] = {
    "representation": "representation.onnx",
    "dynamics": "dynamics.onnx",
    "prediction": "prediction.onnx",
}

# Read-buffer size for SHA-256 file hashing. Tuned for typical SSD
# block sizes; not exposed as a config because it has no behavioural
# implications.
_HASH_CHUNK_BYTES: Final[int] = 64 * 1024


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFileEntry:
    """A single per-role file entry inside the manifest.

    Attributes:
        path: Path relative to the manifest directory (or absolute).
            Stored as a string for cross-language JSON portability —
            the Rust side reads this as `String`, not `PathBuf`.
        sha256: Hex SHA-256 digest of the file's bytes. The Rust runner
            recomputes this on load and rejects the bundle on mismatch.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class ModelManifestFiles:
    """The three per-role file entries a MuZero bundle ships."""

    representation: ModelFileEntry
    dynamics: ModelFileEntry
    prediction: ModelFileEntry


@dataclass(frozen=True)
class ModelManifest:
    """Top-level manifest dataclass — mirrors the Rust struct exactly.

    See ``crates/forge-mc-runner/src/manifest.rs`` for the canonical
    schema. Field order, names, and types must stay in lockstep.
    """

    schema_version: int
    version: int
    schema_id: str
    created_at: str
    files: ModelManifestFiles

    # ---- conversion helpers ----

    def to_json_dict(self) -> dict[str, object]:
        """Render as a JSON-serialisable dict whose shape matches the
        Rust ``serde_json`` output.

        Uses ``dataclasses.asdict`` so any future field additions
        propagate automatically.
        """
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> ModelManifest:
        """Build a manifest from a parsed JSON dict.

        Raises ``ValueError`` if required keys are missing or have the
        wrong type — these become :class:`ManifestError` at the
        ``load_manifest`` boundary.
        """
        try:
            files_dict = data["files"]
            if not isinstance(files_dict, dict):
                raise ValueError("files: expected object")
            files = ModelManifestFiles(
                representation=_entry_from_dict(files_dict.get("representation")),
                dynamics=_entry_from_dict(files_dict.get("dynamics")),
                prediction=_entry_from_dict(files_dict.get("prediction")),
            )
            return cls(
                schema_version=_require_int(data, "schema_version"),
                version=_require_int(data, "version"),
                schema_id=str(data["schema_id"]),
                created_at=str(data["created_at"]),
                files=files,
            )
        except KeyError as e:
            raise ValueError(f"manifest missing key: {e.args[0]}") from e

    # ---- validation ----

    def validate(self) -> None:
        """Lightweight cross-check that this manifest could be loaded
        by the Rust runner.

        Raises :class:`ManifestError` on the same conditions as
        ``forge_mc_runner::ModelManifest::validate`` so client code
        sees identical failure modes regardless of where the manifest
        was constructed.
        """
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"manifest schema_version mismatch: expected "
                f"{MANIFEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.version < 1:
            raise ManifestError("manifest version must be >= 1")
        if not self.schema_id:
            raise ManifestError("schema_id must be non-empty")
        for role in ("representation", "dynamics", "prediction"):
            entry: ModelFileEntry = getattr(self.files, role)
            if not entry.path:
                raise ManifestError(f"files.{role}.path must be non-empty")
            if not entry.sha256:
                raise ManifestError(f"files.{role}.sha256 must be non-empty")


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class ManifestError(Exception):
    """Raised on manifest schema/validation failures."""


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def utc_now_rfc3339() -> str:
    """Return the current UTC time formatted as RFC 3339 with ``Z``
    suffix — the canonical form the Rust ``chrono::Utc::now().to_rfc3339()``
    produces, modulo subsecond precision which both sides tolerate.
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Stream-hash a file with SHA-256 and return its hex digest.

    Streaming (rather than ``hashlib.sha256(open(path, 'rb').read())``)
    keeps memory bounded for large ONNX exports.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _entry_from_dict(d: object) -> ModelFileEntry:
    if not isinstance(d, dict):
        raise ValueError("expected file entry object")
    return ModelFileEntry(path=str(d["path"]), sha256=str(d["sha256"]))


def _require_int(data: dict[str, object], key: str) -> int:
    """Pull an integer field out of an untrusted JSON dict.

    Mypy sees the loaded dict as ``dict[str, object]`` and refuses to
    accept ``int(obj)`` on the bare values; this helper narrows the
    type with an explicit isinstance check, producing the same
    ``ValueError`` shape ``from_json_dict`` already promises.
    """
    v = data[key]
    if isinstance(v, bool):
        # JSON has no separate bool type vs int; reject booleans
        # explicitly so we don't silently accept ``true`` where the
        # Rust side expects a number.
        raise ValueError(f"manifest field {key!r} must be an integer (got bool)")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.lstrip("-").isdigit():
        return int(v)
    raise ValueError(f"manifest field {key!r} must be an integer (got {type(v).__name__})")


def load_manifest(path: str | os.PathLike[str]) -> ModelManifest:
    """Load and validate a manifest from disk.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ManifestError: if the JSON parses but fails validation.
        ValueError: if the JSON is malformed.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be a JSON object, got {type(data).__name__}")
    m = ModelManifest.from_json_dict(data)
    m.validate()
    return m


def save_manifest(manifest: ModelManifest, path: str | os.PathLike[str]) -> Path:
    """Validate and atomically write a manifest to ``path``.

    Atomic semantics: the JSON is written to a ``.tmp`` sibling in the
    same directory, fsync'd, then renamed into place. Matches
    ``forge_mc_runner::ModelManifest::save_json`` so the watcher never
    observes a half-written file.
    """
    manifest.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True)
    # Write to a tmp sibling, fsync, rename. ``delete=False`` because
    # we manage the rename ourselves; ``dir=`` keeps the tmp on the
    # same filesystem so ``os.replace`` is atomic on POSIX and Windows.
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".manifest", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(p)
        logger.debug("manifest saved", extra={"path": str(p), "version": manifest.version})
    except Exception:
        # Best-effort cleanup of the orphan tmp file on failure.
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise
    return p


def build_manifest(
    *,
    version: int,
    schema_id: str,
    files_dir: str | os.PathLike[str],
    representation_filename: str = DEFAULT_BUNDLE_FILENAMES["representation"],
    dynamics_filename: str = DEFAULT_BUNDLE_FILENAMES["dynamics"],
    prediction_filename: str = DEFAULT_BUNDLE_FILENAMES["prediction"],
    created_at: str | None = None,
) -> ModelManifest:
    """Assemble a :class:`ModelManifest` from three on-disk ONNX files.

    Computes each role's SHA-256 from the file at
    ``files_dir / <role>_filename``. Stores the per-role ``path`` as
    relative-to-``files_dir`` so the manifest is portable when the
    bundle is later moved.

    Args:
        version: Monotonic export counter.
        schema_id: sha256 of (env_id, obs_dim, action_count, action_map,
            reward_config). Must match the env's handshake `schema_id`.
        files_dir: Directory containing the three ONNX files.
        representation_filename: Override for the representation file
            name (defaults to ``representation.onnx``).
        dynamics_filename: Override for the dynamics file name.
        prediction_filename: Override for the prediction file name.
        created_at: Optional RFC 3339 timestamp; defaults to
            :func:`utc_now_rfc3339`.

    Returns:
        A validated :class:`ModelManifest`.
    """
    files_dir_p = Path(files_dir)
    rep_p = files_dir_p / representation_filename
    dyn_p = files_dir_p / dynamics_filename
    pred_p = files_dir_p / prediction_filename
    files = ModelManifestFiles(
        representation=ModelFileEntry(path=representation_filename, sha256=sha256_file(rep_p)),
        dynamics=ModelFileEntry(path=dynamics_filename, sha256=sha256_file(dyn_p)),
        prediction=ModelFileEntry(path=prediction_filename, sha256=sha256_file(pred_p)),
    )
    m = ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=version,
        schema_id=schema_id,
        created_at=created_at if created_at is not None else utc_now_rfc3339(),
        files=files,
    )
    m.validate()
    return m
