"""conftest.py — Root pytest configuration for FORGE demo_ui tests.

Adds the FORGE root to sys.path so that `demo_ui` is importable
regardless of which directory pytest is invoked from.
"""

import sys
from pathlib import Path

# FORGE root is one level above this file
FORGE_ROOT = Path(__file__).parent
if str(FORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FORGE_ROOT))

# Also add the python/ directory so forge_env is importable without maturin install
PYTHON_DIR = FORGE_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
