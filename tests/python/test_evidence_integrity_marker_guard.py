"""Marker guard for evidence integrity gate.

Asserts that the test suite's default marker expression in ``pyproject.toml``
does not exclude any marker carried by ``tests/python/test_evidence_integrity.py``,
ensuring the evidence gate cannot be silently bypassed via marker configuration.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
EVIDENCE_GATE_TEST = REPO_ROOT / "tests" / "python" / "test_evidence_integrity.py"


def extract_markers_from_file(path: Path) -> set[str]:
    """Extract all pytest marker names attached to functions, classes, or modules."""
    if not path.is_file():
        return set()

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    markers: set[str] = set()

    for node in ast.walk(tree):
        # 1. Look for @pytest.mark.<name> decorators
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                marker_name = _extract_marker_name(decorator)
                if marker_name:
                    markers.add(marker_name)
        # 2. Look for pytestmark = pytest.mark.<name> or [pytest.mark.<name>]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    markers.update(_extract_markers_from_value(node.value))

    return markers


def _extract_marker_name(node: ast.AST) -> str | None:
    # @pytest.mark.foo or @pytest.mark.foo(...)
    target = node.func if isinstance(node, ast.Call) else node
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
        and target.value.attr == "mark"
    ):
        return target.attr
    return None


def _extract_markers_from_value(node: ast.AST) -> set[str]:
    found: set[str] = set()
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            name = _extract_marker_name(elt)
            if name:
                found.add(name)
    else:
        name = _extract_marker_name(node)
        if name:
            found.add(name)
    return found


def extract_marker_expression(pyproject_path: Path) -> str:
    """Extract the -m marker expression from [tool.pytest.ini_options].addopts."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    match = re.search(r"-m\s+(['\"])(.*?)\1", addopts)
    if match:
        return match.group(2)
    return ""


def check_marker_exclusion(marker_expr: str, gate_markers: set[str]) -> list[str]:
    """Verify that marker_expr does not exclude any of gate_markers or 'evidence_integrity'."""
    errors: list[str] = []
    # Find all negated tokens: 'not <token>'
    negated_tokens = set(re.findall(r"\bnot\s+([a-zA-Z0-9_]+)", marker_expr))

    # 1. Any direct exclusion of 'evidence_integrity'
    if "evidence_integrity" in negated_tokens:
        errors.append(
            f"Pytest addopts marker expression '{marker_expr}' explicitly excludes 'evidence_integrity'. "
            "Remedy: remove 'not evidence_integrity' from tool.pytest.ini_options.addopts in pyproject.toml."
        )

    # 2. Any exclusion of markers carried by the gate test file
    errors.extend(
        f"Pytest addopts marker expression '{marker_expr}' excludes marker '{m}' carried by "
        f"test_evidence_integrity.py. "
        f"Remedy: remove 'not {m}' from tool.pytest.ini_options.addopts in pyproject.toml."
        for m in gate_markers
        if m in negated_tokens
    )

    return errors


def test_evidence_integrity_is_not_excluded_by_markers() -> None:
    """The default pytest marker expression must not exclude test_evidence_integrity.py."""
    assert PYPROJECT_TOML.is_file(), f"pyproject.toml not found at {PYPROJECT_TOML}"
    assert EVIDENCE_GATE_TEST.is_file(), f"gate test not found at {EVIDENCE_GATE_TEST}"

    marker_expr = extract_marker_expression(PYPROJECT_TOML)
    gate_markers = extract_markers_from_file(EVIDENCE_GATE_TEST)

    errors = check_marker_exclusion(marker_expr, gate_markers)
    assert not errors, "\n".join(errors)


# ==============================================================================
# Negative unit tests
# ==============================================================================


def test_negative_direct_marker_exclusion_detected() -> None:
    """If addopts excludes evidence_integrity, check_marker_exclusion fails."""
    expr = "not lmstudio and not evidence_integrity"
    errors = check_marker_exclusion(expr, set())
    assert any("explicitly excludes 'evidence_integrity'" in e for e in errors), errors
    assert any("Remedy: remove 'not evidence_integrity'" in e for e in errors), errors


def test_negative_carried_marker_exclusion_detected() -> None:
    """If addopts excludes a marker carried by the gate test, check_marker_exclusion fails."""
    expr = "not lmstudio and not slow_gate"
    errors = check_marker_exclusion(expr, {"slow_gate"})
    assert any("excludes marker 'slow_gate'" in e for e in errors), errors
    assert any("Remedy: remove 'not slow_gate'" in e for e in errors), errors
