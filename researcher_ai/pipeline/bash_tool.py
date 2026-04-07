"""Local bash execution helper for pipeline validation commands."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path
from typing import Optional


@dataclass
class BashResult:
    status: str
    cmd: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


class BashTool:
    """Run shell commands with bounded timeout and structured error capture."""

    def __init__(self, *, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

    def run(self, cmd: list[str], *, cwd: str | Path) -> BashResult:
        cmd_text = " ".join(cmd)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return BashResult(
                status="tool_unavailable",
                cmd=cmd_text,
                stderr=f"{cmd[0]} not installed",
            )
        except subprocess.TimeoutExpired as exc:
            return BashResult(
                status="error",
                cmd=cmd_text,
                stderr=f"TimeoutExpired: {exc}",
            )

        return BashResult(
            status="ok" if proc.returncode == 0 else "error",
            cmd=cmd_text,
            returncode=proc.returncode,
            stdout=proc.stdout[-5000:],
            stderr=proc.stderr[-5000:],
        )
