"""conftest.py -- Root pytest configuration for FORGE.

With ``pip install -e .`` the ``forge`` and ``forge_env`` packages are
importable without path hacks.  The only addition here is ``scripts/``
which contains standalone CLI modules that some tests import directly
(e.g. ``from train import parse_args``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ contains standalone CLI modules, not a package.
# Add it so tests can ``import train``, ``import calibrate_agri``, etc.
_SCRIPTS_DIR = str(Path(__file__).parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
