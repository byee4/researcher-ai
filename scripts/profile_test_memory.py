#!/usr/bin/env python3
"""Profile memory usage for pytest tests one node at a time.

This script collects pytest node IDs, runs each test in isolation, and monitors
resident set size (RSS). If memory exceeds a configurable limit (default 4 GiB),
the test process group is terminated immediately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BYTES_PER_GIB = 1024 ** 3


@dataclass
class TestResult:
    """Per-test memory and execution result."""

    nodeid: str
    status: str
    duration_seconds: float
    peak_rss_bytes: int
    kill_reason: str | None = None

    @property
    def peak_rss_gib(self) -> float:
        return self.peak_rss_bytes / BYTES_PER_GIB


def collect_nodeids(pytest_bin: str) -> list[str]:
    """Return collected pytest node IDs from the current working directory."""
    proc = subprocess.run(
        [pytest_bin, "--collect-only", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Pytest collection failed.\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )

    nodeids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if "::" in line and line.startswith("tests/"):
            nodeids.append(line)
    return nodeids


def pid_exists(pid: int) -> bool:
    """Return True if a pid currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def list_descendant_pids(root_pid: int) -> set[int]:
    """Return all descendant pids for root_pid (including root_pid)."""
    seen: set[int] = set()
    queue = [root_pid]

    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)

        proc = subprocess.run(
            ["pgrep", "-P", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            continue

        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                child_pid = int(line)
            except ValueError:
                continue
            queue.append(child_pid)

    return seen


def rss_bytes_for_pid(pid: int) -> int:
    """Return RSS in bytes for pid, or 0 if unavailable."""
    if not pid_exists(pid):
        return 0

    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0

    text = proc.stdout.strip()
    if not text:
        return 0

    try:
        rss_kib = int(text)
    except ValueError:
        return 0

    return rss_kib * 1024


def process_tree_rss_bytes(root_pid: int) -> int:
    """Return combined RSS bytes for root pid and descendants."""
    total = 0
    for pid in list_descendant_pids(root_pid):
        total += rss_bytes_for_pid(pid)
    return total


def kill_process_group(popen_proc: subprocess.Popen[str]) -> None:
    """Terminate the entire process group for a running test command."""
    try:
        os.killpg(os.getpgid(popen_proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_single_test(
    pytest_bin: str,
    nodeid: str,
    threshold_bytes: int,
    poll_seconds: float,
    per_test_timeout_seconds: int,
) -> TestResult:
    """Run one pytest node ID and return memory profile data."""
    start = time.monotonic()
    popen_proc = subprocess.Popen(
        [pytest_bin, "-q", nodeid],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )

    peak_rss = 0
    kill_reason: str | None = None
    status = "passed"

    while True:
        ret = popen_proc.poll()
        if ret is not None:
            if ret != 0:
                status = "failed"
            break

        elapsed = time.monotonic() - start
        if elapsed > per_test_timeout_seconds:
            kill_reason = f"timeout>{per_test_timeout_seconds}s"
            status = "killed"
            kill_process_group(popen_proc)
            popen_proc.wait(timeout=5)
            break

        rss = process_tree_rss_bytes(popen_proc.pid)
        if rss > peak_rss:
            peak_rss = rss

        if rss > threshold_bytes:
            kill_reason = f"rss>{threshold_bytes}"
            status = "killed"
            kill_process_group(popen_proc)
            popen_proc.wait(timeout=5)
            break

        time.sleep(poll_seconds)

    duration = time.monotonic() - start
    return TestResult(
        nodeid=nodeid,
        status=status,
        duration_seconds=duration,
        peak_rss_bytes=peak_rss,
        kill_reason=kill_reason,
    )


def write_csv(results: list[TestResult], path: Path) -> None:
    """Write per-test results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nodeid", "status", "duration_seconds", "peak_rss_bytes", "peak_rss_gib", "kill_reason"])
        for r in results:
            writer.writerow([r.nodeid, r.status, f"{r.duration_seconds:.3f}", r.peak_rss_bytes, f"{r.peak_rss_gib:.3f}", r.kill_reason or ""])


def write_json_summary(results: list[TestResult], path: Path, threshold_bytes: int) -> None:
    """Write aggregate summary data as JSON."""
    total = len(results)
    killed = [r for r in results if r.status == "killed"]
    failed = [r for r in results if r.status == "failed"]
    passed = [r for r in results if r.status == "passed"]

    top = sorted(results, key=lambda r: r.peak_rss_bytes, reverse=True)[:20]

    summary = {
        "threshold_bytes": threshold_bytes,
        "threshold_gib": threshold_bytes / BYTES_PER_GIB,
        "total_tests": total,
        "passed": len(passed),
        "failed": len(failed),
        "killed": len(killed),
        "top_peak_memory_tests": [
            {
                "nodeid": r.nodeid,
                "status": r.status,
                "peak_rss_bytes": r.peak_rss_bytes,
                "peak_rss_gib": r.peak_rss_gib,
                "duration_seconds": r.duration_seconds,
                "kill_reason": r.kill_reason,
            }
            for r in top
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile memory usage for pytest tests.")
    parser.add_argument("--pytest", default=".venv/bin/pytest", help="Path to pytest executable.")
    parser.add_argument("--threshold-gib", type=float, default=4.0, help="Kill threshold in GiB RSS.")
    parser.add_argument("--poll-seconds", type=float, default=0.1, help="Memory poll interval in seconds.")
    parser.add_argument(
        "--per-test-timeout-seconds",
        type=int,
        default=300,
        help="Maximum runtime per test before forced kill.",
    )
    parser.add_argument(
        "--output-prefix",
        default="artifacts/test_memory_profile",
        help="Output prefix. Produces <prefix>.csv and <prefix>_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run memory profiling across all collected pytest node IDs."""
    args = parse_args()
    threshold_bytes = int(args.threshold_gib * BYTES_PER_GIB)
    pytest_bin = args.pytest

    try:
        nodeids = collect_nodeids(pytest_bin)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not nodeids:
        print("No tests collected.", file=sys.stderr)
        return 1

    print(f"Collected {len(nodeids)} tests")
    print(f"Memory threshold: {args.threshold_gib:.2f} GiB")

    results: list[TestResult] = []
    for idx, nodeid in enumerate(nodeids, start=1):
        result = run_single_test(
            pytest_bin=pytest_bin,
            nodeid=nodeid,
            threshold_bytes=threshold_bytes,
            poll_seconds=args.poll_seconds,
            per_test_timeout_seconds=args.per_test_timeout_seconds,
        )
        results.append(result)
        print(
            f"[{idx}/{len(nodeids)}] {result.status:<6} "
            f"peak={result.peak_rss_gib:>6.3f}GiB dur={result.duration_seconds:>6.2f}s {nodeid}"
        )

    prefix = Path(args.output_prefix)
    csv_path = Path(f"{prefix}.csv")
    summary_path = Path(f"{prefix}_summary.json")
    write_csv(results, csv_path)
    write_json_summary(results, summary_path, threshold_bytes)

    killed_count = sum(1 for r in results if r.status == "killed")
    failed_count = sum(1 for r in results if r.status == "failed")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(f"Finished. killed={killed_count} failed={failed_count} total={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
