"""Confirm asset paths are listed in pyproject.toml [tool.maturin] include.

Maturin only auto-packages *.py under python-source. Non-Python assets
(JSON schemas, TOML configs, prompt templates, few-shots) need an
explicit include entry or they will not ship in built wheels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def _read_pyproject() -> dict[str, Any]:
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover - py39/py310
        import tomli as _toml
    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as fh:
        return cast("dict[str, Any]", _toml.load(fh))


def _include_paths(maturin_table: dict[str, Any]) -> list[str]:
    raw = maturin_table.get("include", []) or []
    paths: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, dict):
            path = entry.get("path", "")
            if path:
                paths.append(path)
    return paths


def test_maturin_includes_cognitive_schemas() -> None:
    data = _read_pyproject()
    paths = _include_paths(data["tool"]["maturin"])
    assert any("python/forge/cognitive/schemas" in p for p in paths), (
        "[tool.maturin] include must cover python/forge/cognitive/schemas/*.json "
        f"so wheels ship the action schemas; got {paths}"
    )


def test_maturin_includes_cognitive_configs() -> None:
    data = _read_pyproject()
    paths = _include_paths(data["tool"]["maturin"])
    assert any("configs/cognitive" in p for p in paths), (
        "[tool.maturin] include must cover configs/cognitive/** so wheels ship "
        f"preset TOML/templates/few-shots; got {paths}"
    )
