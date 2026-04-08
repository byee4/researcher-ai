"""Helpers for snapshot fixture refresh workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SnapshotRefreshPlan:
    command: str
    manifest_path: str


def build_snapshot_refresh_plan(manifest_path: str = "tests/snapshots/manifest.yaml") -> SnapshotRefreshPlan:
    command = "pytest tests/ -m live --refresh-snapshots"
    return SnapshotRefreshPlan(command=command, manifest_path=manifest_path)


def update_manifest_created_date(manifest_path: str, *, today: date | None = None) -> str:
    target_day = today or date.today()
    path = Path(manifest_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload["created"] = target_day.isoformat()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target_day.isoformat()

