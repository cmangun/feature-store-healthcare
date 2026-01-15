"""Basic sanity tests for package structure."""
from __future__ import annotations

import sys
from pathlib import Path


def test_python_version():
    """Verify Python version is 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_src_directory_exists():
    """Verify src directory exists."""
    root = Path(__file__).parent.parent
    src_dir = root / "src"
    assert src_dir.exists()
