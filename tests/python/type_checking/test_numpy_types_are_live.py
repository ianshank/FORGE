"""Sentinel: prove mypy still type-checks numpy rather than erasing it to ``Any``.

``pyproject.toml``'s ``[[tool.mypy.overrides]]`` block lists ``numpy`` /
``numpy.*`` under ``ignore_missing_imports = true``. That flag is meant to
suppress a *missing-stub* error, and it is a no-op for a package that ships
``py.typed`` — which numpy has since 1.20. But nothing proved that, and the
difference is invisible: if numpy ever does resolve to ``Any`` (a changed
override, a lint job that stops installing numpy, a numpy build without
``py.typed``), every ``np.ndarray`` annotation in this repository silently
stops being checked and mypy keeps exiting 0. The gate would look green while
checking nothing — the exact failure mode this branch exists to close.

**How this file fails.** ``pyproject.toml`` sets ``warn_unused_ignores =
true``. The ``# type: ignore[...]`` comments below sit on expressions that are
genuine type errors *under real numpy stubs*:

* numpy typed → the errors occur, the ignores are **used**, mypy is silent,
  and this file passes;
* numpy erased to ``Any`` → no error occurs, the ignores are **unused**, and
  ``warn_unused_ignores`` fails the ``Mypy type check`` step naming this file.

The assertion is carried by the ignore comments themselves. Nothing to keep in
sync, and no new tooling: it reuses a setting the repo already had on.

**Why it also runs under pytest.** mypy checks ``python/`` and ``scripts/`` in
CI, not ``tests/``, so this module is added to that step's argument list
explicitly (see ``.github/workflows/ci.yml``). The runtime tests below guard
the other direction — that the expressions really are the shapes the ignores
claim, so a future edit cannot leave a comment pinned to an expression that no
longer errors for the stated reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _shape_arg_is_type_checked() -> None:
    """``np.zeros`` takes a shape; a ``str`` is not one.

    Real numpy stubs reject this with ``arg-type``. Under ``Any`` they do not,
    and the ignore below goes unused.
    """
    np.zeros("not-a-shape")  # type: ignore[arg-type]


def _attribute_access_is_type_checked() -> None:
    """``ndarray`` has no such method.

    Real numpy stubs reject this with ``attr-defined``. Under ``Any`` every
    attribute resolves and the ignore below goes unused.
    """
    np.array([1]).definitely_not_an_ndarray_method()  # type: ignore[attr-defined]


def test_the_sentinel_expressions_still_fail_at_runtime() -> None:
    """The ignored expressions must still be genuinely wrong, not merely untyped.

    An ignore pinned to an expression that had quietly become *valid* would
    leave mypy silent for the wrong reason. That direction is already safe —
    the ignore would be unused and the sentinel would fail — but this test says
    so directly and points at the cause instead of at ``warn_unused_ignores``.
    """
    with pytest.raises(TypeError):
        np.zeros("not-a-shape")  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        np.array([1]).definitely_not_an_ndarray_method()  # type: ignore[attr-defined]


def test_numpy_ships_inline_types() -> None:
    """numpy must carry ``py.typed``; without it mypy has nothing to check.

    This is the precondition for the ignore comments above meaning anything.
    Asserting it separately makes a numpy build that dropped the marker report
    *that*, rather than surfacing as a confusing wall of unused-ignore errors.
    """
    marker = Path(np.__file__).parent / "py.typed"
    assert marker.is_file(), (
        f"numpy {np.__version__} at {marker.parent} ships no py.typed marker, so "
        "mypy cannot type-check any numpy usage in this repository"
    )
