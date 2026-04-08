#!/usr/bin/env python3
"""Print or apply snapshot refresh workflow steps."""

from __future__ import annotations

import argparse

from researcher_ai.benchmarks.snapshot_refresh import (
    build_snapshot_refresh_plan,
    update_manifest_created_date,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot refresh helper")
    parser.add_argument("--manifest", default="tests/snapshots/manifest.yaml")
    parser.add_argument(
        "--touch-manifest-date",
        action="store_true",
        help="Update manifest.created to today's date",
    )
    args = parser.parse_args(argv)

    plan = build_snapshot_refresh_plan(args.manifest)
    print(f"Run refresh command:\n{plan.command}")
    print(f"Manifest path: {plan.manifest_path}")

    if args.touch_manifest_date:
        updated = update_manifest_created_date(args.manifest)
        print(f"Updated manifest created date to {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

