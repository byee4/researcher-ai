from __future__ import annotations

import subprocess
from pathlib import Path


def test_model_alias_guard_script_passes_on_repo():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["python3", "scripts/check_model_alias_usage.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
