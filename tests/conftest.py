"""Pytest configuration: make the repository root importable.

Tests import the package as ``src.backtest.*``. Adding the repository root
(the parent of ``tests/``) to ``sys.path`` lets that work regardless of the
directory pytest is invoked from.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
