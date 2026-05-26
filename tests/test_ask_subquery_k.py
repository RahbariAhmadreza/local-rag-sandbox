"""Offline CLI tests for ask.py --subquery-k validation."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASK_SCRIPT = PROJECT_ROOT / "scripts" / "ask.py"


def test_ask_rejects_subquery_k_with_standard_depth() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ASK_SCRIPT),
            "--depth",
            "standard",
            "--subquery-k",
            "10",
            "test question",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 2
    combined = (result.stderr + result.stdout).lower()
    assert "subquery-k" in combined or "learning" in combined
