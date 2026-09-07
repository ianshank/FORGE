"""Tests for ``benchmarks/runner/export_criterion_scaling.py``.

Imports the helper by file path because ``benchmarks/runner`` is build
infrastructure, not a Python package (same pattern as
``test_check_zero_alloc.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_HELPER: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "runner" / "export_criterion_scaling.py"
)


@pytest.fixture(scope="module")
def helper_module() -> ModuleType:
    assert _HELPER.is_file(), f"helper not found at {_HELPER}"
    spec = importlib.util.spec_from_file_location("export_criterion_scaling", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_criterion_scaling"] = module
    spec.loader.exec_module(module)
    return module


def _estimates(*, mean_ns: float, median_ns: float) -> dict[str, Any]:
    return {
        "mean": {"point_estimate": mean_ns},
        "median": {"point_estimate": median_ns},
    }


def _write_estimates(root: Path, group: str, num_agents: int, payload: dict[str, Any]) -> Path:
    path = root / group / "num_agents" / str(num_agents) / "new" / "estimates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_collect_rows_parses_square_and_hex(helper_module: ModuleType, tmp_path: Path) -> None:
    _write_estimates(
        tmp_path, "multi_agent_scaling_square", 1, _estimates(mean_ns=10_000.0, median_ns=9_500.0)
    )
    _write_estimates(
        tmp_path, "multi_agent_scaling_square", 8, _estimates(mean_ns=20_000.0, median_ns=19_000.0)
    )
    _write_estimates(
        tmp_path, "multi_agent_scaling_hex", 1, _estimates(mean_ns=12_000.0, median_ns=11_000.0)
    )
    rows = helper_module.collect_rows(tmp_path)
    assert [ (r["group"], r["num_agents"]) for r in rows ] == [
        ("hex", 1),
        ("square", 1),
        ("square", 8),
    ]
    square_one = next(r for r in rows if r["group"] == "square" and r["num_agents"] == 1)
    assert square_one["env_steps_per_sec"] == pytest.approx(100_000.0)
    assert square_one["agent_steps_per_sec"] == pytest.approx(100_000.0)
    square_eight = next(r for r in rows if r["group"] == "square" and r["num_agents"] == 8)
    assert square_eight["env_steps_per_sec"] == pytest.approx(50_000.0)
    assert square_eight["agent_steps_per_sec"] == pytest.approx(400_000.0)


def test_collect_rows_ignores_named_baseline_copies(
    helper_module: ModuleType, tmp_path: Path
) -> None:
    _write_estimates(
        tmp_path, "multi_agent_scaling_square", 1, _estimates(mean_ns=10_000.0, median_ns=9_500.0)
    )
    stale = (
        tmp_path
        / "multi_agent_scaling_square"
        / "num_agents"
        / "1"
        / "cloud_agent"
        / "estimates.json"
    )
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps(_estimates(mean_ns=99_999.0, median_ns=99_999.0)), encoding="utf-8")
    rows = helper_module.collect_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["mean_ns"] == pytest.approx(10_000.0)


def test_build_report_missing_tree_raises(helper_module: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        helper_module.build_report(
            criterion_dir=tmp_path / "missing",
            profile="cloud_agent",
            repo_root=tmp_path,
            world_side=128,
            seed=42,
        )


def test_build_report_empty_tree_names_remedy(helper_module: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "other_bench").mkdir()
    with pytest.raises(ValueError, match="Remedy:"):
        helper_module.build_report(
            criterion_dir=tmp_path,
            profile="cloud_agent",
            repo_root=tmp_path,
            world_side=128,
            seed=42,
        )


def test_malformed_estimates_do_not_crash_collect(
    helper_module: ModuleType, tmp_path: Path
) -> None:
    _write_estimates(tmp_path, "multi_agent_scaling_square", 1, {"mean": {}})
    with pytest.raises(ValueError, match="point_estimate"):
        helper_module.collect_rows(tmp_path)


def test_main_writes_sorted_json(helper_module: ModuleType, tmp_path: Path) -> None:
    _write_estimates(
        tmp_path / "criterion",
        "multi_agent_scaling_square",
        1,
        _estimates(mean_ns=10_000.0, median_ns=9_500.0),
    )
    out = tmp_path / "out.json"
    rc = helper_module.main(
        [
            "--criterion-dir",
            str(tmp_path / "criterion"),
            "--out",
            str(out),
            "--profile",
            "cloud_agent",
            "--world-side",
            "128",
            "--seed",
            "42",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["producer"] == "multi_agent_scaling"
    assert report["profile"] == "cloud_agent"
    assert report["world_side"] == 128
    assert report["seed"] == 42
    assert report["agent_counts"] == [1]
    assert report["variants"][0]["group"] == "square"
    assert "hardware" in report
    assert "os" in report["hardware"]
    assert "cpu" in report["hardware"]
    # sort_keys makes the file byte-stable for a given payload
    assert out.read_text(encoding="utf-8") == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_main_exits_2_on_empty_tree(helper_module: ModuleType, tmp_path: Path) -> None:
    rc = helper_module.main(
        [
            "--criterion-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
            "--profile",
            "cloud_agent",
        ]
    )
    assert rc == helper_module.EXIT_INPUT_ERROR
    assert not (tmp_path / "out.json").exists()
