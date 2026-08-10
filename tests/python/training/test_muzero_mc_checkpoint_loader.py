"""Tests for ``forge.training.muzero_mc.checkpoint_loader``.

The loader warm-starts a MuZero bundle from the HuggingFace Hub. These
tests exercise it end-to-end **without** touching the network by
injecting a fake ``huggingface_hub`` module (the loader imports
``hf_hub_download`` lazily inside the function, so a ``sys.modules``
stand-in is sufficient — no ``huggingface_hub`` install required).

Covers:

- Happy path: three role files downloaded into a ``v{NNNNNNNN}`` subdir,
  a valid ``model_manifest.json`` written with the requested schema_id /
  version, and the manifest path returned.
- ``filename_map`` override and ``subfolder`` pass-through reach
  ``hf_hub_download`` verbatim.
- Download failure propagates and leaves no manifest behind.
- The versioned subdir name matches ``format_bundle_version_dir``.
- Structured logging emits the warm-start / per-file / success lines.

No hard-coded bundle names: role filenames flow through
``DEFAULT_BUNDLE_FILENAMES`` so the assertions track the production
default rather than duplicating literals.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from forge.training.muzero_mc.checkpoint_loader import load_from_hf
from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    load_manifest,
)
from forge.training.muzero_mc.trainer import format_bundle_version_dir

# A syntactically valid canonical schema_id (64 lowercase hex chars).
SCHEMA_ID = "a" * 64


class _FakeHub:
    """Records ``hf_hub_download`` calls and materializes each file.

    Mirrors the huggingface_hub 1.x behaviour the loader relies on:

    - the removed ``local_dir_use_symlinks`` kwarg is NOT accepted, so a
      loader regression re-adding it fails these tests with a
      ``TypeError`` exactly as it would against the locked real client;
    - with a ``subfolder``, the file is materialized under
      ``local_dir/<subfolder>/<filename>`` and that path is returned —
      the loader must normalize from the returned path, not recompute it.

    ``build_manifest`` hashes the downloaded files, so the fake must
    actually write bytes where the real Hub client places them.
    """

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail_on = fail_on

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        filename: str,
        subfolder: str | None,
        local_dir: Path,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "local_dir": str(local_dir),
            }
        )
        if self._fail_on is not None and filename == self._fail_on:
            raise RuntimeError(f"simulated Hub download failure for {filename}")
        dest = Path(local_dir)
        if subfolder is not None:
            dest = dest / subfolder
        dest = dest / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"weights::{repo_id}::{filename}".encode())
        return str(dest)


def _install_hub(
    monkeypatch: pytest.MonkeyPatch, *, fail_on: str | None = None
) -> _FakeHub:
    hub = _FakeHub(fail_on=fail_on)
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hub.hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return hub


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> _FakeHub:
    """Install a fake ``huggingface_hub`` module for the duration of a test."""
    return _install_hub(monkeypatch)


def test_load_from_hf_happy_path(fake_hub: _FakeHub, tmp_path: Path) -> None:
    manifest_path = load_from_hf(
        "user/forge-muzero",
        schema_id=SCHEMA_ID,
        output_dir=tmp_path,
        version=3,
    )

    # Returns the manifest path, and it exists on disk.
    assert manifest_path == (tmp_path / MANIFEST_FILENAME).resolve()
    assert manifest_path.is_file()

    # All three role files landed in the versioned subdir.
    version_dir = tmp_path / format_bundle_version_dir(3)
    for fname in DEFAULT_BUNDLE_FILENAMES.values():
        assert (version_dir / fname).is_file()

    # Manifest round-trips with the requested schema_id / version, and the
    # per-role paths are relative to output_dir (portable bundle).
    manifest = load_manifest(manifest_path)
    assert manifest.schema_id == SCHEMA_ID
    assert manifest.version == 3
    subdir = format_bundle_version_dir(3)
    for role in ("representation", "dynamics", "prediction"):
        expected = f"{subdir}/{DEFAULT_BUNDLE_FILENAMES[role]}"
        assert getattr(manifest.files, role).path == expected

    # Exactly one download per role, no subfolder.
    assert len(fake_hub.calls) == len(DEFAULT_BUNDLE_FILENAMES)
    assert {c["filename"] for c in fake_hub.calls} == set(
        DEFAULT_BUNDLE_FILENAMES.values()
    )
    assert all(c["repo_id"] == "user/forge-muzero" for c in fake_hub.calls)
    assert all(c["subfolder"] is None for c in fake_hub.calls)


def test_default_version_is_one(fake_hub: _FakeHub, tmp_path: Path) -> None:
    manifest_path = load_from_hf(
        "user/repo", schema_id=SCHEMA_ID, output_dir=tmp_path
    )
    manifest = load_manifest(manifest_path)
    assert manifest.version == 1
    assert (tmp_path / format_bundle_version_dir(1)).is_dir()


def test_filename_map_override(fake_hub: _FakeHub, tmp_path: Path) -> None:
    load_from_hf(
        "user/repo",
        schema_id=SCHEMA_ID,
        output_dir=tmp_path,
        filename_map={"representation": "repr_v2.onnx"},
    )
    downloaded = {c["filename"] for c in fake_hub.calls}
    # Overridden role uses the custom name; the others keep the defaults.
    assert "repr_v2.onnx" in downloaded
    assert DEFAULT_BUNDLE_FILENAMES["representation"] not in downloaded
    assert DEFAULT_BUNDLE_FILENAMES["dynamics"] in downloaded
    assert DEFAULT_BUNDLE_FILENAMES["prediction"] in downloaded

    manifest = load_manifest(tmp_path / MANIFEST_FILENAME)
    subdir = format_bundle_version_dir(1)
    assert manifest.files.representation.path == f"{subdir}/repr_v2.onnx"


def test_subfolder_round_trip(fake_hub: _FakeHub, tmp_path: Path) -> None:
    # The Hub client materializes subfolder downloads under
    # ``local_dir/<subfolder>/``; the loader must still produce a flat,
    # hashable versioned bundle plus a valid manifest (regression test for
    # the FileNotFoundError this used to raise in build_manifest).
    manifest_path = load_from_hf(
        "user/repo",
        schema_id=SCHEMA_ID,
        output_dir=tmp_path,
        subfolder="checkpoints/best",
    )
    assert fake_hub.calls
    assert all(c["subfolder"] == "checkpoints/best" for c in fake_hub.calls)

    version_dir = tmp_path / format_bundle_version_dir(1)
    for fname in DEFAULT_BUNDLE_FILENAMES.values():
        assert (version_dir / fname).is_file(), "bundle files must be flat"
    # The Hub client's subfolder scaffolding is cleaned up.
    assert not (version_dir / "checkpoints").exists()
    manifest = load_manifest(manifest_path)
    assert manifest.schema_id == SCHEMA_ID


def test_hostile_subfolder_never_cleans_outside_bundle(
    fake_hub: _FakeHub, tmp_path: Path
) -> None:
    # A subfolder that path-escapes the versioned dir must not let the
    # empty-dir cleanup walk (or delete) anything outside the bundle. The
    # fake hub writes to local_dir/<subfolder>/<file>, which for "../side"
    # lands next to the versioned dir; the loader must still produce a
    # valid flat bundle and leave outside directories alone.
    sentinel = tmp_path / "outside-empty-dir"
    sentinel.mkdir()
    manifest_path = load_from_hf(
        "user/repo",
        schema_id=SCHEMA_ID,
        output_dir=tmp_path / "bundle",
        subfolder="../escape",
    )
    assert manifest_path.is_file()
    version_dir = tmp_path / "bundle" / format_bundle_version_dir(1)
    for fname in DEFAULT_BUNDLE_FILENAMES.values():
        assert (version_dir / fname).is_file()
    assert sentinel.is_dir(), "cleanup escaped the bundle directory"


def test_download_failure_propagates_and_writes_no_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_hub(monkeypatch, fail_on=DEFAULT_BUNDLE_FILENAMES["dynamics"])
    with pytest.raises(RuntimeError, match="simulated Hub download failure"):
        load_from_hf("user/repo", schema_id=SCHEMA_ID, output_dir=tmp_path)
    # The manifest is only written after every file downloads, so a
    # mid-download failure must leave none behind.
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_output_dir_accepts_str(fake_hub: _FakeHub, tmp_path: Path) -> None:
    # output_dir is typed Path but the loader resolves via Path(...), so a
    # str must also work — guards the backward-compatible call surface.
    manifest_path = load_from_hf(
        "user/repo", schema_id=SCHEMA_ID, output_dir=str(tmp_path)
    )
    assert manifest_path.is_file()


def test_logs_warm_start_and_success(
    fake_hub: _FakeHub, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(
        logging.INFO, logger="forge.training.muzero_mc.checkpoint_loader"
    ):
        load_from_hf(
            "user/repo", schema_id=SCHEMA_ID, output_dir=tmp_path, version=7
        )
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "Warm-starting from HF" in messages
    assert "user/repo" in messages
    # One "Downloading network" line per role, plus the success line.
    assert messages.count("Downloading network") == len(DEFAULT_BUNDLE_FILENAMES)
    assert "Successfully loaded HF checkpoint" in messages
