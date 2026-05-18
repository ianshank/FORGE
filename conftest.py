"""conftest.py — Root pytest configuration for FORGE.

For CI or full test runs, install the package first:
    maturin develop          # builds native extension + installs python/ packages
    pip install -e ".[all]"  # if you need optional deps (sb3, torch, etc.)

For local pure-Python development without the Rust toolchain, the
sys.path fallback below ensures that ``forge`` and ``demo_ui`` are
importable directly from source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FORGE_ROOT = Path(__file__).resolve().parent
_PYTHON_DIR = _FORGE_ROOT / "python"
_SCRIPTS_DIR = _FORGE_ROOT / "scripts"

# Public alias for tests that need to anchor paths relative to the repo root
# without re-deriving it from ``__file__`` (single source of truth).
REPO_ROOT = _FORGE_ROOT


def _ensure_importable(directory: Path) -> None:
    """Add *directory* to ``sys.path`` only when it is not already present."""
    path_str = str(directory)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# Make ``forge`` (pure Python) and ``demo_ui`` importable from source.
_ensure_importable(_FORGE_ROOT)
_ensure_importable(_PYTHON_DIR)
# Make the orchestrator scripts (``run_e2e_long``, ``_e2e_progress``, ``train``,
# ``calibrate_agri`` …) importable from tests without each one needing its own
# ``sys.path.insert`` + ``# noqa: E402`` dance.
_ensure_importable(_SCRIPTS_DIR)
