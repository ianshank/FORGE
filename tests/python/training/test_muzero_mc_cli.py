"""Tests for ``forge.training.muzero_mc.cli``.

Only the manifest-touching pieces of the CLI are exercised here —
``validate-manifest`` is fast and torch-free. The ``bootstrap``
subcommand is covered separately in ``test_muzero_mc_bootstrap.py``
which gates on ``torch`` availability via
``pytest.importorskip("torch")``.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used as a runtime fixture type.

from forge.training.muzero_mc.cli import EXIT_IO, EXIT_OK, EXIT_VALIDATION, main
from forge.training.muzero_mc.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ModelFileEntry,
    ModelManifest,
    ModelManifestFiles,
    save_manifest,
)


def _well_formed(version: int = 1) -> ModelManifest:
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        version=version,
        schema_id="sid",
        created_at="t",
        files=ModelManifestFiles(
            representation=ModelFileEntry(path="r.onnx", sha256="a" * 64),
            dynamics=ModelFileEntry(path="d.onnx", sha256="b" * 64),
            prediction=ModelFileEntry(path="p.onnx", sha256="c" * 64),
        ),
    )


def test_validate_manifest_returns_ok_for_well_formed(tmp_path: Path) -> None:
    p = save_manifest(_well_formed(), tmp_path / MANIFEST_FILENAME)
    rc = main(["validate-manifest", str(p)])
    assert rc == EXIT_OK


def test_validate_manifest_accepts_directory_target(tmp_path: Path) -> None:
    save_manifest(_well_formed(), tmp_path / MANIFEST_FILENAME)
    rc = main(["validate-manifest", str(tmp_path)])
    assert rc == EXIT_OK


def test_validate_manifest_missing_file_returns_io_error(tmp_path: Path) -> None:
    rc = main(["validate-manifest", str(tmp_path / "no.json")])
    assert rc == EXIT_IO


def test_validate_manifest_invalid_json_returns_validation_error(tmp_path: Path) -> None:
    p = tmp_path / MANIFEST_FILENAME
    p.write_text("{not valid json", encoding="utf-8")
    rc = main(["validate-manifest", str(p)])
    assert rc == EXIT_VALIDATION


def test_validate_manifest_schema_drift_returns_validation_error(tmp_path: Path) -> None:
    # Write a manifest with wrong schema_version that bypasses
    # ``save_manifest``'s validation gate by writing JSON directly.
    p = tmp_path / MANIFEST_FILENAME
    p.write_text(
        '{"schema_version": 999, "version": 1, "schema_id": "x", '
        '"created_at": "t", "files": {'
        '"representation": {"path": "r.onnx", "sha256": "a"}, '
        '"dynamics": {"path": "d.onnx", "sha256": "b"}, '
        '"prediction": {"path": "p.onnx", "sha256": "c"}}}',
        encoding="utf-8",
    )
    rc = main(["validate-manifest", str(p)])
    assert rc == EXIT_VALIDATION


def test_build_parser_has_two_subcommands() -> None:
    from forge.training.muzero_mc.cli import build_parser

    parser = build_parser()
    # argparse's sub-parser registry lives under _subparsers (private
    # but stable since Python 3.0). Reach into it to assert both
    # subcommands are registered without invoking them.
    sub_actions = [
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "bootstrap" in a.choices
    ]
    assert sub_actions, "no subparser action found"
    raw_choices = sub_actions[0].choices
    assert raw_choices is not None
    choices = set(raw_choices)
    assert {"bootstrap", "validate-manifest"} <= choices


def test_bootstrap_rejects_invalid_obs_dim(tmp_path: Path) -> None:
    rc = main(
        [
            "bootstrap",
            "--obs-dim",
            "0",  # invalid
            "--action-dim",
            "4",
            "--schema-id",
            "x",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc != EXIT_OK
