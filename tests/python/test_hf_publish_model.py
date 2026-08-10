"""Tests for ``scripts/hf_publish_model.py``.

Exercises the publish pipeline end-to-end without touching the network:
bundle verification (sha256 re-check), Hub staging layout (flat ONNX
names + rebuilt manifest + rendered card), and the upload flow via an
injected fake ``HfApi``. The fake mirrors the real client's method
signatures (``create_repo`` / ``upload_folder`` keyword surface) so
signature drift in the script fails here first.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

# scripts/ is placed on sys.path by the root conftest.py (_ensure_importable).
import hf_publish_model as pub
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from forge.training.muzero_mc.manifest import (
    DEFAULT_BUNDLE_FILENAMES,
    MANIFEST_FILENAME,
    build_manifest,
    load_manifest,
    save_manifest,
)
from forge.training.muzero_mc.trainer import format_bundle_version_dir

SCHEMA_ID = "b" * 64


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """A minimal valid bundle: v00000002/ ONNX files + manifest."""
    bundle = tmp_path / "bundle"
    subdir = bundle / format_bundle_version_dir(2)
    subdir.mkdir(parents=True)
    for role, fname in DEFAULT_BUNDLE_FILENAMES.items():
        (subdir / fname).write_bytes(f"onnx::{role}".encode())
    manifest = build_manifest(
        version=2,
        schema_id=SCHEMA_ID,
        files_dir=bundle,
        representation_filename=f"{subdir.name}/{DEFAULT_BUNDLE_FILENAMES['representation']}",
        dynamics_filename=f"{subdir.name}/{DEFAULT_BUNDLE_FILENAMES['dynamics']}",
        prediction_filename=f"{subdir.name}/{DEFAULT_BUNDLE_FILENAMES['prediction']}",
    )
    save_manifest(manifest, bundle / MANIFEST_FILENAME)
    return bundle


class _FakeHfApi:
    """Records create_repo/upload_folder calls with the real keyword surface."""

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.uploaded: list[dict[str, object]] = []

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> None:
        self.created.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }
        )

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        commit_message: str,
    ) -> None:
        self.uploaded.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "folder_path": folder_path,
                "commit_message": commit_message,
            }
        )


def test_verify_bundle_happy_path(bundle_dir: Path) -> None:
    manifest = pub.verify_bundle(bundle_dir)
    assert manifest.version == 2
    assert manifest.schema_id == SCHEMA_ID


def test_verify_bundle_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(pub.PublishError, match=r"no model_manifest\.json"):
        pub.verify_bundle(tmp_path)


def test_verify_bundle_detects_corruption(bundle_dir: Path) -> None:
    corrupted = (
        bundle_dir
        / format_bundle_version_dir(2)
        / DEFAULT_BUNDLE_FILENAMES["dynamics"]
    )
    corrupted.write_bytes(b"tampered")
    with pytest.raises(pub.PublishError, match="sha256 mismatch for dynamics"):
        pub.verify_bundle(bundle_dir)


def test_stage_bundle_layout_and_manifest(bundle_dir: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    manifest = pub.verify_bundle(bundle_dir)
    pub.stage_bundle(
        bundle_dir,
        staging,
        manifest=manifest,
        card_template=pub.DEFAULT_CARD_TEMPLATE,
        repo_id="user/forge-muzero",
        obs_dim="920",
        action_dim="12",
        trained=False,
    )

    # Flat canonical filenames + manifest + card, nothing else.
    names = {p.name for p in staging.iterdir()}
    assert names == set(DEFAULT_BUNDLE_FILENAMES.values()) | {
        MANIFEST_FILENAME,
        "README.md",
    }

    # The hub manifest references the flat layout with unchanged hashes.
    hub_manifest = load_manifest(staging / MANIFEST_FILENAME)
    assert hub_manifest.version == manifest.version
    assert hub_manifest.schema_id == SCHEMA_ID
    for role in ("representation", "dynamics", "prediction"):
        hub_entry = getattr(hub_manifest.files, role)
        src_entry = getattr(manifest.files, role)
        assert hub_entry.path == DEFAULT_BUNDLE_FILENAMES[role]
        assert hub_entry.sha256 == src_entry.sha256


def test_card_rendering_untrained_warning(bundle_dir: Path, tmp_path: Path) -> None:
    manifest = pub.verify_bundle(bundle_dir)
    card = pub.render_card(
        pub.DEFAULT_CARD_TEMPLATE,
        manifest=manifest,
        repo_id="user/forge-muzero",
        obs_dim="920",
        action_dim="12",
        trained=False,
    )
    assert "NOT a trained model" in card
    assert SCHEMA_ID in card
    assert "user/forge-muzero" in card
    assert "__" not in card.replace("__init__", ""), "unsubstituted placeholder left"

    trained_card = pub.render_card(
        pub.DEFAULT_CARD_TEMPLATE,
        manifest=manifest,
        repo_id="user/forge-muzero",
        obs_dim="920",
        action_dim="12",
        trained=True,
    )
    assert "NOT a trained model" not in trained_card


def test_main_dry_run_skips_upload(bundle_dir: Path, tmp_path: Path, capsys) -> None:
    staging = tmp_path / "stage-dry"
    code = pub.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--repo-id",
            "user/forge-muzero",
            "--staging-dir",
            str(staging),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert (staging / "README.md").is_file()


def test_main_publishes_via_injected_api(bundle_dir: Path, tmp_path: Path) -> None:
    api = _FakeHfApi()
    staging = tmp_path / "stage-pub"
    code = pub.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--repo-id",
            "user/forge-muzero",
            "--staging-dir",
            str(staging),
            "--private",
        ],
        api=api,
    )
    assert code == 0
    assert api.created == [
        {
            "repo_id": "user/forge-muzero",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert len(api.uploaded) == 1
    assert api.uploaded[0]["repo_id"] == "user/forge-muzero"
    assert api.uploaded[0]["folder_path"] == str(staging)


def test_main_corrupted_bundle_fails(bundle_dir: Path) -> None:
    (
        bundle_dir
        / format_bundle_version_dir(2)
        / DEFAULT_BUNDLE_FILENAMES["representation"]
    ).write_bytes(b"bad")
    code = pub.main(
        ["--bundle-dir", str(bundle_dir), "--repo-id", "user/x", "--dry-run"]
    )
    assert code == 1


def test_staged_manifest_is_valid_json(bundle_dir: Path, tmp_path: Path) -> None:
    staging = tmp_path / "stage-json"
    manifest = pub.verify_bundle(bundle_dir)
    pub.stage_bundle(
        bundle_dir,
        staging,
        manifest=manifest,
        card_template=pub.DEFAULT_CARD_TEMPLATE,
        repo_id="user/r",
        obs_dim="1",
        action_dim="1",
        trained=True,
    )
    data = json.loads((staging / MANIFEST_FILENAME).read_text())
    assert data["schema_version"] == 1
    assert isinstance(data["version"], int)
