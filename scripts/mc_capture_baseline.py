#!/usr/bin/env python3
"""Operator-facing entry point for the v0.5 baseline capture flow.

Thin wrapper around the ``forge.training.muzero_mc.cli`` subcommand
of the same name. Kept as a top-level script so the docs/quickstart
can point operators at one file path while the actual implementation
lives in a packaged + mypy-checked module.

Usage::

    python scripts/mc_capture_baseline.py --variant random --episodes 100

…is equivalent to::

    python -m forge.training.muzero_mc.cli capture-baseline \\
        --variant random --episodes 100

The subcommand path is the canonical surface — every flag this
script forwards is documented there and unit-tested in
``tests/python/training/test_muzero_mc_capture_baseline.py``.
"""

from __future__ import annotations

import sys

from forge.training.muzero_mc.cli import main


def _entrypoint() -> int:
    argv = ["capture-baseline", *sys.argv[1:]]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
