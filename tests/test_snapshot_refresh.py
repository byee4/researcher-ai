from __future__ import annotations

from datetime import date

import yaml

from researcher_ai.benchmarks.snapshot_refresh import (
    build_snapshot_refresh_plan,
    update_manifest_created_date,
)


def test_build_snapshot_refresh_plan_defaults():
    plan = build_snapshot_refresh_plan()
    assert "pytest tests/ -m live --refresh-snapshots" == plan.command
    assert plan.manifest_path.endswith("tests/snapshots/manifest.yaml")


def test_update_manifest_created_date(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("version: '1.0'\ncreated: '2026-01-01'\n", encoding="utf-8")
    updated = update_manifest_created_date(str(manifest), today=date(2026, 4, 8))
    assert updated == "2026-04-08"
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["created"] == "2026-04-08"

