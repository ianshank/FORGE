"""HuggingFace checkpoint warm-start loader for MuZero models.

Downloads pretrained MuZero ONNX weights from the HuggingFace Hub and
builds a local model_manifest.json stamped with the caller-supplied
schema_id (the Hub repo contributes only the ONNX bytes; schema_id
alignment is the caller's responsibility — see ``scripts/hf_publish_model.py``
for the publish-side counterpart that records it in the uploaded manifest).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    build_manifest,
    save_manifest,
    sha256_file,
)
from forge.training.muzero_mc.trainer import format_bundle_version_dir

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "COMMIT_SHA_RE",
    "DEFAULT_HF_REVISION",
    "ChecksumMismatchError",
    "load_from_hf",
]

#: Default Hub revision. ``"main"`` preserves the historical behaviour for
#: existing callers, but it is a *mutable* ref: the same command can fetch
#: different bytes on different days. Pass an immutable commit SHA (or a
#: signed tag) via ``revision=`` for a reproducible warm-start; anything
#: that is not a commit SHA is logged as a WARNING so the non-reproducibility
#: is visible in the bootstrap-container logs rather than silent.
DEFAULT_HF_REVISION: Final[str] = "main"

#: A 40-hex git object id — the only revision form the Hub cannot move.
COMMIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-fA-F]{40}\Z")


class ChecksumMismatchError(ValueError):
    """A downloaded bundle file did not match its expected SHA-256.

    Subclasses :class:`ValueError` so the CLI's existing
    ``except ValueError`` → ``EXIT_USAGE`` path keeps working for callers
    that do not catch this type explicitly.
    """


def _verify_digest(role: str, path: Path, expected: str) -> None:
    """Raise :class:`ChecksumMismatchError` unless ``path`` hashes to ``expected``."""
    actual = sha256_file(path)
    if actual.lower() != expected.strip().lower():
        raise ChecksumMismatchError(
            f"{role}: downloaded {path.name} has sha256 {actual}, "
            f"expected {expected} — refusing to build a manifest over unverified bytes"
        )
    logger.info("verified %s sha256=%s", role, actual)


def load_from_hf(
    repo_id: str,
    *,
    schema_id: str,
    output_dir: Path,
    version: int = 1,
    filename_map: dict[str, str] | None = None,
    subfolder: str | None = None,
    revision: str = DEFAULT_HF_REVISION,
    expected_sha256: Mapping[str, str] | None = None,
) -> Path:
    """Download a MuZero model bundle from HuggingFace Hub.

    Downloads the three network ONNX files (representation, dynamics, prediction),
    writes them into a versioned subdirectory under output_dir, and generates a
    model_manifest.json verifying the specified schema_id.

    Supply-chain notes:

    * ``revision`` pins the Hub ref the bytes come from. The default
      (:data:`DEFAULT_HF_REVISION`) is the repo's mutable ``main`` branch,
      which is what this function has always fetched; pass a 40-hex commit
      SHA to make the warm-start reproducible. A non-SHA revision logs a
      WARNING.
    * ``expected_sha256`` lets the caller pin the bytes themselves. Without
      it the generated manifest is self-certifying: ``build_manifest``
      hashes whatever arrived and records that as the truth, so a swapped
      artefact produces a perfectly valid-looking manifest.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g. "my-user/forge-muzero").
        schema_id: Canonical SHA256 that the environment expects.
        output_dir: Absolute path where the model bundle will be built.
        version: Manifest version to stamp (default: 1).
        filename_map: Dict mapping roles to Hub filenames.
        subfolder: Subfolder within the Hub repo containing the files.
        revision: Hub git revision (commit SHA, tag, or branch) to download
            from. Defaults to :data:`DEFAULT_HF_REVISION`.
        expected_sha256: Optional ``role -> hex digest`` map. Every role
            present is verified against the downloaded file *before* the
            manifest is built; a mismatch raises
            :class:`ChecksumMismatchError` and leaves no manifest behind.
            Roles absent from the map are not verified.

    Returns:
        Absolute Path to the written model_manifest.json.

    Raises:
        ChecksumMismatchError: A downloaded file failed its expected-digest
            check.
        ValueError: ``subfolder`` escapes the bundle directory, or
            ``expected_sha256`` names a role that is not in the bundle.
    """
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir).resolve()
    fnames = dict(DEFAULT_BUNDLE_FILENAMES)
    if filename_map:
        fnames.update(filename_map)

    digests: Mapping[str, str] = expected_sha256 or {}
    unknown_roles = sorted(set(digests) - set(fnames))
    if unknown_roles:
        raise ValueError(
            f"expected_sha256 names roles not in the bundle: {unknown_roles} "
            f"(known roles: {sorted(fnames)})"
        )

    bundle_subdir_name = format_bundle_version_dir(version)
    versioned_dir = output_dir / bundle_subdir_name
    versioned_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Warm-starting from HF: repo_id=%s, revision=%s, version=%d, schema_id=%s",
        repo_id,
        revision,
        version,
        schema_id,
    )
    if not COMMIT_SHA_RE.match(revision):
        logger.warning(
            "HF revision %r is a mutable ref — the same bootstrap can fetch different "
            "bytes later. Pin a 40-hex commit SHA (and/or pass expected_sha256=) for a "
            "reproducible warm-start.",
            revision,
        )
    if not digests:
        logger.warning(
            "no expected_sha256 supplied — the generated manifest will certify whatever "
            "bytes arrived from %s, not bytes you chose in advance",
            repo_id,
        )

    # Reject subfolder values that could escape versioned_dir via path traversal.
    if subfolder is not None:
        # Normalize and verify the joined path stays inside versioned_dir.
        candidate = (versioned_dir / subfolder).resolve()
        if not candidate.is_relative_to(versioned_dir.resolve()):
            raise ValueError(
                f"subfolder {subfolder!r} resolves outside the bundle directory "
                f"({versioned_dir}) — path traversal rejected"
            )

    # Download each network file. The Hub client materializes files under
    # `local_dir/<subfolder>/<fname>` when a subfolder is given, so always
    # trust the *returned* path and normalize into the flat versioned dir
    # the manifest contract expects.
    for role, fname in fnames.items():
        logger.info("Downloading network %s (%s)...", role, fname)
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                subfolder=subfolder,
                revision=revision,
                local_dir=versioned_dir,
            )
        )
        target = versioned_dir / fname
        if downloaded.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), target)
        # Verify BEFORE the manifest is built, so a mismatch never leaves a
        # valid-looking manifest certifying substituted bytes.
        expected = digests.get(role)
        if expected is not None:
            _verify_digest(role, target, expected)

    # Drop any now-empty subfolder chain the Hub client left behind. The
    # walk is constrained to descendants of the versioned dir so a hostile
    # or malformed subfolder ("../..", absolute path) can never delete
    # directories outside the bundle.
    if subfolder:
        root = versioned_dir.resolve()
        leftover = (versioned_dir / subfolder).resolve()
        while (
            leftover != root
            and leftover.is_relative_to(root)
            and leftover.is_dir()
            and not any(leftover.iterdir())
        ):
            leftover.rmdir()
            leftover = leftover.parent

    # Build the manifest referencing these files
    manifest = build_manifest(
        version=version,
        schema_id=schema_id,
        files_dir=output_dir,
        representation_filename=f"{bundle_subdir_name}/{fnames['representation']}",
        dynamics_filename=f"{bundle_subdir_name}/{fnames['dynamics']}",
        prediction_filename=f"{bundle_subdir_name}/{fnames['prediction']}",
    )

    manifest_path = output_dir / MANIFEST_FILENAME
    save_manifest(manifest, manifest_path)
    logger.info("Successfully loaded HF checkpoint into %s", manifest_path)

    return manifest_path
