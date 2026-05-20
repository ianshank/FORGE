"""Tests for ``forge.training.muzero_mc.cli``.

Only the manifest-touching pieces of the CLI are exercised here —
``validate-manifest`` is fast and torch-free. The ``bootstrap``
subcommand is covered separately in ``test_muzero_mc_bootstrap.py``
which gates on ``torch`` availability via
``pytest.importorskip("torch")``.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used as a runtime fixture type.
from typing import TYPE_CHECKING

from forge.training.muzero_mc.cli import EXIT_IO, EXIT_OK, EXIT_VALIDATION, main

if TYPE_CHECKING:
    import pytest
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


def test_build_parser_has_all_subcommands() -> None:
    from forge.training.muzero_mc.cli import build_parser

    parser = build_parser()
    # argparse's sub-parser registry lives under _subparsers (private
    # but stable since Python 3.0). Reach into it to assert every
    # subcommand is registered without invoking them. Uses an equality
    # comparison (not `<=`) so a future deletion of any subcommand fails
    # this test rather than silently passing.
    sub_actions = [
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "bootstrap" in a.choices
    ]
    assert sub_actions, "no subparser action found"
    raw_choices = sub_actions[0].choices
    assert raw_choices is not None
    assert set(raw_choices) == {
        "bootstrap",
        "validate-manifest",
        "train",
        "compute-schema-id",
        "capture-baseline",
    }


def test_compute_schema_id_prints_hash_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: `compute-schema-id --quiet` loads two TOMLs and
    prints ONLY the 64-hex sha256 to stdout (stderr-bound logs).
    """
    am_path = tmp_path / "action_map.toml"
    am_path.write_text(
        '[[action]]\nid = 0\nkind = "noop"\nticks = 1\n', encoding="utf-8"
    )
    rw_path = tmp_path / "rewards.toml"
    rw_path.write_text(
        '[[reward]]\nkind = "survival"\nvalue = 0.01\n', encoding="utf-8"
    )
    rc = main(
        [
            "compute-schema-id",
            "--action-map",
            str(am_path),
            "--rewards",
            str(rw_path),
            "--quiet",
        ]
    )
    assert rc == EXIT_OK
    captured = capsys.readouterr()
    hash_line = captured.out.strip()
    assert len(hash_line) == 64
    assert all(c in "0123456789abcdef" for c in hash_line)


def test_compute_schema_id_missing_action_map_returns_io(tmp_path: Path) -> None:
    rw_path = tmp_path / "rewards.toml"
    rw_path.write_text(
        '[[reward]]\nkind = "survival"\nvalue = 0.01\n', encoding="utf-8"
    )
    rc = main(
        [
            "compute-schema-id",
            "--action-map",
            str(tmp_path / "missing.toml"),
            "--rewards",
            str(rw_path),
        ]
    )
    assert rc == EXIT_IO


def test_compute_schema_id_missing_rewards_returns_io(tmp_path: Path) -> None:
    am_path = tmp_path / "action_map.toml"
    am_path.write_text('[[action]]\nid = 0\nkind = "noop"\n', encoding="utf-8")
    rc = main(
        [
            "compute-schema-id",
            "--action-map",
            str(am_path),
            "--rewards",
            str(tmp_path / "missing.toml"),
        ]
    )
    assert rc == EXIT_IO


def test_train_subcommand_requires_input_dir(tmp_path: Path) -> None:
    """argparse rejects `train` invocations missing the mandatory
    `--input` flag. Confirms the subcommand is registered and exposes
    the documented surface, without running torch."""
    import pytest

    with pytest.raises(SystemExit):
        main(["train", "--out", str(tmp_path), "--manifest", str(tmp_path / "m.json")])


def test_train_subcommand_returns_io_when_input_dir_missing(tmp_path: Path) -> None:
    """`--input` points at a non-existent directory → exit code
    `EXIT_IO`. Doesn't need torch because the dispatch path validates
    the input path before importing the trainer module."""
    missing = tmp_path / "does-not-exist"
    rc = main(
        [
            "train",
            "--input",
            str(missing),
            "--out",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "m.json"),
            "--schema-id",
            "x",
            "--obs-dim",
            "4",
            "--action-dim",
            "3",
            "--iters",
            "1",
        ]
    )
    assert rc == EXIT_IO


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
