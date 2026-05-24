"""HuggingFace checkpoint warm-start loader for MuZero models.

Downloads pretrained MuZero ONNX weights from the HuggingFace Hub
and builds a local model_manifest.json with checked schema_id alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path

from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    build_manifest,
    save_manifest,
)
from forge.training.muzero_mc.trainer import format_bundle_version_dir

logger = logging.getLogger(__name__)


def load_from_hf(
    repo_id: str,
    *,
    schema_id: str,
    output_dir: Path,
    version: int = 1,
    filename_map: dict[str, str] | None = None,
    subfolder: str | None = None,
) -> Path:
    """Download a MuZero model bundle from HuggingFace Hub.

    Downloads the three network ONNX files (representation, dynamics, prediction),
    writes them into a versioned subdirectory under output_dir, and generates a
    model_manifest.json verifying the specified schema_id.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g. "my-user/forge-muzero").
        schema_id: Canonical SHA256 that the environment expects.
        output_dir: Absolute path where the model bundle will be built.
        version: Manifest version to stamp (default: 1).
        filename_map: Dict mapping roles to Hub filenames.
        subfolder: Subfolder within the Hub repo containing the files.

    Returns:
        Absolute Path to the written model_manifest.json.
    """
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir).resolve()
    fnames = dict(DEFAULT_BUNDLE_FILENAMES)
    if filename_map:
        fnames.update(filename_map)

    bundle_subdir_name = format_bundle_version_dir(version)
    versioned_dir = output_dir / bundle_subdir_name
    versioned_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Warm-starting from HF: repo_id=%s, version=%d, schema_id=%s",
        repo_id,
        version,
        schema_id,
    )

    # Download each network file
    for role, fname in fnames.items():
        logger.info("Downloading network %s (%s)...", role, fname)
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            subfolder=subfolder,
            local_dir=versioned_dir,
            local_dir_use_symlinks=False,
        )

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
