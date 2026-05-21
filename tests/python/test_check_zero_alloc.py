"""Tests for ``benchmarks/runner/check_zero_alloc.py``.

Covers every exit path of the zero-allocation audit post-processor so the
helper itself has regression protection independent of the Rust side that
produces the JSON report. Exit paths:

* ``0`` when every variant is clean.
* ``0`` when every violating variant is allowlisted.
* ``1`` when one or more non-allowlisted variants violate the contract.
* ``2`` when the input file is missing or malformed.

These tests import the module by file path because ``benchmarks/runner``
is not a Python package and must not be turned into one (it is build
infrastructure, not product code).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

#: Absolute path to the helper under test. Resolving from ``__file__`` makes
#: the test insensitive to the caller's cwd.
_HELPER: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "runner" / "check_zero_alloc.py"
)


@pytest.fixture(scope="module")
def helper_module() -> ModuleType:
    """Load ``check_zero_alloc.py`` as a module under the name ``check_zero_alloc``."""
    assert _HELPER.is_file(), f"helper not found at {_HELPER}"
    spec = importlib.util.spec_from_file_location("check_zero_alloc", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_zero_alloc"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tmp_report_dir(tmp_path: Path) -> Iterator[Path]:
    """Yield a clean temporary directory for report + summary paths."""
    yield tmp_path


def _write_report(path: Path, variants: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"variants": variants}), encoding="utf-8")


def _run_main(module: ModuleType, args: list[str]) -> int:
    """Invoke the helper's ``main()`` with the given argv tail.

    Patches ``sys.argv`` inline so the module's ``argparse`` picks it up. The
    caller passes arguments without the program name.
    """
    old_argv = sys.argv
    sys.argv = ["check_zero_alloc", *args]
    try:
        return int(module.main())
    finally:
        sys.argv = old_argv


def test_clean_report_exits_zero(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    report = tmp_report_dir / "clean.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
            {
                "variant": "Move",
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
        ],
    )
    summary = tmp_report_dir / "summary.json"
    rc = _run_main(
        helper_module,
        ["--input", str(report), "--json", str(summary), "--log-level", "ERROR"],
    )
    assert rc == 0
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["violations"] == []
    assert body["clean_count"] == 2


def test_violation_exits_one(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    report = tmp_report_dir / "bad.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
            {
                "variant": "Move_Up",
                "iters": 100,
                "total_blocks": 1000,
                "total_bytes": 128,
                "peak_live_bytes": 64,
            },
        ],
    )
    rc = _run_main(helper_module, ["--input", str(report), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_VIOLATION


def test_allowlisted_violation_exits_zero(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    report = tmp_report_dir / "allowed.json"
    _write_report(
        report,
        [
            {
                "variant": "Communicate_0",
                "iters": 100,
                "total_blocks": 10,
                "total_bytes": 64,
                "peak_live_bytes": 32,
            }
        ],
    )
    summary = tmp_report_dir / "summary.json"
    rc = _run_main(
        helper_module,
        [
            "--input",
            str(report),
            "--allow",
            "Communicate_0",
            "--json",
            str(summary),
            "--log-level",
            "ERROR",
        ],
    )
    assert rc == 0
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["violations"] == []
    assert body["allowlisted"][0]["variant"] == "Communicate_0"


def test_missing_input_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    missing = tmp_report_dir / "does_not_exist.json"
    with pytest.raises(SystemExit) as exc:
        _run_main(helper_module, ["--input", str(missing), "--log-level", "ERROR"])
    assert exc.value.code == helper_module.EXIT_INPUT_ERROR


def test_malformed_input_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    bad = tmp_report_dir / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run_main(helper_module, ["--input", str(bad), "--log-level", "ERROR"])
    assert exc.value.code == helper_module.EXIT_INPUT_ERROR


def test_empty_variants_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    empty = tmp_report_dir / "empty.json"
    empty.write_text(json.dumps({"variants": []}), encoding="utf-8")
    rc = _run_main(helper_module, ["--input", str(empty), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_INPUT_ERROR


def test_max_bytes_threshold(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """A variant with ``total_bytes <= max-bytes`` passes when blocks == 0."""
    report = tmp_report_dir / "threshold.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 32,
                "peak_live_bytes": 16,
            },
        ],
    )
    rc = _run_main(
        helper_module,
        ["--input", str(report), "--max-bytes", "64", "--log-level", "ERROR"],
    )
    assert rc == 0


def test_max_bytes_relaxes_blocks_check(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """When `--max-bytes > 0` (regression-investigation mode), non-zero
    `total_blocks` within the byte ceiling is NOT a violation. The strict
    `total_blocks == 0` contract only applies in the default zero-max-bytes
    mode."""
    report = tmp_report_dir / "blocks.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": 3,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
        ],
    )
    rc = _run_main(
        helper_module,
        ["--input", str(report), "--max-bytes", "999", "--log-level", "ERROR"],
    )
    assert rc == 0


def test_non_integer_field_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """A variant row with a non-integer numeric field is malformed input.

    Surfaces as ``EXIT_INPUT_ERROR`` via the ``_MalformedReport`` sentinel
    rather than propagating a bare ``TypeError``/``ValueError`` to the
    shell as an uncaught exception.
    """
    report = tmp_report_dir / "bad_type.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": "not-an-int",
                "total_bytes": 0,
                "peak_live_bytes": 0,
            }
        ],
    )
    rc = _run_main(helper_module, ["--input", str(report), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_INPUT_ERROR


def test_null_field_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """A ``null`` numeric field is also malformed input, not a crash."""
    report = tmp_report_dir / "null_field.json"
    report.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "variant": "Noop",
                        "iters": 100,
                        "total_blocks": None,
                        "total_bytes": 0,
                        "peak_live_bytes": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rc = _run_main(helper_module, ["--input", str(report), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_INPUT_ERROR


def test_non_object_row_exits_two(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """A variant row that is not a JSON object is malformed input."""
    report = tmp_report_dir / "non_object.json"
    report.write_text(json.dumps({"variants": ["not-a-dict"]}), encoding="utf-8")
    rc = _run_main(helper_module, ["--input", str(report), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_INPUT_ERROR


def test_strict_mode_rejects_any_blocks(helper_module: ModuleType, tmp_report_dir: Path) -> None:
    """Default strict mode (`--max-bytes 0`) counts any non-zero
    `total_blocks` as a violation — the original zero-allocation contract."""
    report = tmp_report_dir / "strict_blocks.json"
    _write_report(
        report,
        [
            {
                "variant": "Noop",
                "iters": 100,
                "total_blocks": 1,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
        ],
    )
    rc = _run_main(helper_module, ["--input", str(report), "--log-level", "ERROR"])
    assert rc == helper_module.EXIT_VIOLATION


def test_multi_agent_variant_with_num_agents_field(
    helper_module: ModuleType, tmp_report_dir: Path
) -> None:
    """The validator must accept the multi-agent audit schema unchanged.

    The audit binary emits rows shaped like ``Move_Up@n=8`` with a typed
    ``num_agents`` field alongside the original numeric columns. The
    validator only inspects ``variant``/``total_bytes``/``total_blocks``/
    ``iters``/``peak_live_bytes``, so the extra field must not break
    parsing and the row-level checks must still trip on a real violation.
    """
    report = tmp_report_dir / "multi_agent.json"
    _write_report(
        report,
        [
            {
                "variant": "Move_Up@n=1",
                "num_agents": 1,
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
            {
                "variant": "Move_Up@n=8",
                "num_agents": 8,
                "iters": 100,
                "total_blocks": 0,
                "total_bytes": 0,
                "peak_live_bytes": 0,
            },
        ],
    )
    summary = tmp_report_dir / "multi_agent_summary.json"
    rc = _run_main(
        helper_module,
        ["--input", str(report), "--json", str(summary), "--log-level", "ERROR"],
    )
    assert rc == 0
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["violations"] == []
    assert body["clean_count"] == 2
