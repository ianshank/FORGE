"""Publish a MuZero model bundle to the Hugging Face Hub.

Renders a Hub-ready staging directory from a local trainer/bootstrap
bundle (a directory containing ``model_manifest.json`` plus a
``v{NNNNNNNN}/`` subdir with the three ONNX networks) and uploads it via
``HfApi``. The staged layout is exactly what
``forge.training.muzero_mc.checkpoint_loader.load_from_hf`` consumes:

.. code-block:: text

    <repo root>/
      representation.onnx     # flat, DEFAULT_BUNDLE_FILENAMES names
      dynamics.onnx
      prediction.onnx
      model_manifest.json     # re-built for the flat layout, same sha256s
      README.md               # model card rendered from a template

Per-file SHA-256s are re-verified against the source manifest before
anything is staged, so a corrupted bundle can never be published.

Usage::

    python scripts/hf_publish_model.py \
        --bundle-dir models/ --repo-id ianshank/forge-muzero-minecraft \
        --private --obs-dim 920 --action-dim 12 [--trained] [--dry-run]

``huggingface_hub`` is imported lazily (same convention as
``forge.utils.weight_loader``); ``--dry-run`` works without it installed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from forge.training.muzero_mc.manifest import (  # noqa: E402
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ONNX_OPSET_VERSION,
    ModelManifest,
    build_manifest,
    load_manifest,
    save_manifest,
    sha256_file,
)

logger = logging.getLogger("hf_publish_model")

DEFAULT_CARD_TEMPLATE = REPO_ROOT / "docs" / "hf" / "model-card.md"

UNTRAINED_WARNING = (
    "> [!WARNING]\n"
    "> **Random-init bundle — NOT a trained model.** These weights come\n"
    "> from the `bootstrap` random initializer and exist to validate the\n"
    "> publish/warm-start pipeline. Do not expect useful play behaviour.\n"
)


class PublishError(Exception):
    """Raised when the bundle fails verification or staging."""


def _require_huggingface_hub() -> Any:
    """Import huggingface_hub lazily with an actionable error message."""
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - exercised via message test
        raise PublishError(
            "huggingface_hub is required to publish; install it via "
            "`pip install -e \".[minecraft]\"` or `pip install huggingface_hub`"
        ) from exc
    return huggingface_hub


def verify_bundle(bundle_dir: Path) -> ModelManifest:
    """Load the bundle manifest and re-verify every per-file SHA-256.

    Args:
        bundle_dir: Directory containing ``model_manifest.json``.

    Returns:
        The parsed, verified manifest.

    Raises:
        PublishError: If the manifest or any file is missing, or a
            checksum does not match.
    """
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise PublishError(f"no {MANIFEST_FILENAME} in {bundle_dir}")
    manifest = load_manifest(manifest_path)

    for role in ("representation", "dynamics", "prediction"):
        entry = getattr(manifest.files, role)
        file_path = bundle_dir / entry.path
        if not file_path.is_file():
            raise PublishError(f"manifest references missing file: {file_path}")
        actual = sha256_file(file_path)
        if actual != entry.sha256:
            raise PublishError(
                f"sha256 mismatch for {role}: manifest={entry.sha256} actual={actual}"
            )
    logger.info(
        "Bundle verified: version=%d schema_id=%s", manifest.version, manifest.schema_id
    )
    return manifest


def render_card(
    template_path: Path,
    *,
    manifest: ModelManifest,
    repo_id: str,
    obs_dim: str,
    action_dim: str,
    trained: bool,
) -> str:
    """Render the model card template with bundle facts substituted."""
    text = template_path.read_text(encoding="utf-8")
    substitutions = {
        "__TRAINED_WARNING__": "" if trained else UNTRAINED_WARNING,
        "__SCHEMA_ID__": manifest.schema_id,
        "__VERSION__": str(manifest.version),
        "__CREATED_AT__": manifest.created_at,
        "__OBS_DIM__": obs_dim,
        "__ACTION_DIM__": action_dim,
        "__REPO_ID__": repo_id,
        "__ONNX_OPSET__": str(ONNX_OPSET_VERSION),
        "__MANIFEST_SCHEMA_VERSION__": str(MANIFEST_SCHEMA_VERSION),
        "__REPR_SHA256__": manifest.files.representation.sha256,
        "__DYN_SHA256__": manifest.files.dynamics.sha256,
        "__PRED_SHA256__": manifest.files.prediction.sha256,
    }
    for key, value in substitutions.items():
        text = text.replace(key, value)
    return text


def stage_bundle(
    bundle_dir: Path,
    staging_dir: Path,
    *,
    manifest: ModelManifest,
    card_template: Path,
    repo_id: str,
    obs_dim: str,
    action_dim: str,
    trained: bool,
) -> Path:
    """Assemble the Hub-ready directory: flat ONNX + manifest + card.

    The three networks are staged under their canonical
    ``DEFAULT_BUNDLE_FILENAMES`` names at the staging root — the layout
    ``load_from_hf`` downloads — and the manifest is re-built against the
    flat layout (identical bytes, so identical sha256s).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for role in ("representation", "dynamics", "prediction"):
        entry = getattr(manifest.files, role)
        source = bundle_dir / entry.path
        target = staging_dir / DEFAULT_BUNDLE_FILENAMES[role]
        shutil.copy2(source, target)

    hub_manifest = build_manifest(
        version=manifest.version,
        schema_id=manifest.schema_id,
        files_dir=staging_dir,
        created_at=manifest.created_at,
    )
    save_manifest(hub_manifest, staging_dir / MANIFEST_FILENAME)

    card = render_card(
        card_template,
        manifest=manifest,
        repo_id=repo_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        trained=trained,
    )
    (staging_dir / "README.md").write_text(card, encoding="utf-8")
    logger.info("Staged bundle at %s", staging_dir)
    return staging_dir


def publish(
    staging_dir: Path,
    *,
    repo_id: str,
    private: bool,
    commit_message: str,
    api: Any | None = None,
) -> str:
    """Create the model repo (idempotent) and upload the staged folder.

    Args:
        staging_dir: Hub-ready directory from :func:`stage_bundle`.
        repo_id: Target ``user/name`` model repo.
        private: Create the repo private (existing visibility unchanged).
        commit_message: Commit message for the upload.
        api: Injectable ``HfApi``-compatible client (tests); defaults to a
            real ``HfApi`` using ambient auth (``HF_TOKEN`` env var).

    Returns:
        The repo URL.
    """
    if api is None:  # pragma: no cover - network path, covered by fake in tests
        api = _require_huggingface_hub().HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(staging_dir),
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Published %s", url)
    return url


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Local bundle dir containing model_manifest.json + v*/ ONNX files",
    )
    parser.add_argument(
        "--repo-id", required=True, help="Target Hub model repo, e.g. user/name"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo private (default: public)",
    )
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Bundle holds real trained weights; omits the random-init warning",
    )
    parser.add_argument(
        "--card-template",
        type=Path,
        default=DEFAULT_CARD_TEMPLATE,
        help="Model card template path",
    )
    parser.add_argument(
        "--obs-dim", default="unspecified", help="Observation dim recorded in the card"
    )
    parser.add_argument(
        "--action-dim", default="unspecified", help="Action dim recorded in the card"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Staging directory (default: a fresh temp dir)",
    )
    parser.add_argument(
        "--commit-message",
        default="Publish MuZero bundle via scripts/hf_publish_model.py",
        help="Hub commit message",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify + stage only; print the staged layout and skip upload",
    )
    return parser


def main(argv: list[str] | None = None, *, api: Any | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        manifest = verify_bundle(args.bundle_dir)
        staging = args.staging_dir or Path(tempfile.mkdtemp(prefix="hf-model-staging-"))
        stage_bundle(
            args.bundle_dir,
            staging,
            manifest=manifest,
            card_template=args.card_template,
            repo_id=args.repo_id,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            trained=args.trained,
        )
        if args.dry_run:
            print(f"dry-run: staged at {staging}")
            for path in sorted(staging.iterdir()):
                print(f"  {path.name}\t{path.stat().st_size} bytes")
            return 0
        url = publish(
            staging,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            api=api,
        )
        print(url)
        return 0
    except PublishError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
