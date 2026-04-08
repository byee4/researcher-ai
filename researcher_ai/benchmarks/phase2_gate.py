"""Phase 2 benchmark gate utilities.

Compares baseline and candidate recall artifacts and enforces:
- minimum paper-count coverage
- no recall regression beyond configured tolerance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ("overall_recall", "heading_recall", "merge_recall")


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_metrics(payload: dict[str, Any]) -> dict[str, float]:
    source = payload.get("metrics", payload)
    out: dict[str, float] = {}
    for key, value in source.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _paper_count(payload: dict[str, Any]) -> int:
    if "paper_count" in payload:
        try:
            return int(payload["paper_count"])
        except Exception:
            return 0
    papers = payload.get("papers")
    if isinstance(papers, list):
        return len(papers)
    return 0


def evaluate_phase2_gate(
    *,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    min_papers: int = 10,
    max_recall_regression: float = 0.0,
    required_metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> list[str]:
    """Return a list of gate failures (empty list means pass)."""
    failures: list[str] = []
    baseline_count = _paper_count(baseline_payload)
    candidate_count = _paper_count(candidate_payload)

    if baseline_count < min_papers:
        failures.append(
            f"baseline_paper_count_below_minimum: {baseline_count} < {min_papers}"
        )
    if candidate_count < min_papers:
        failures.append(
            f"candidate_paper_count_below_minimum: {candidate_count} < {min_papers}"
        )

    baseline_metrics = _coerce_metrics(baseline_payload)
    candidate_metrics = _coerce_metrics(candidate_payload)
    for metric in required_metrics:
        if metric not in baseline_metrics:
            failures.append(f"baseline_missing_metric: {metric}")
            continue
        if metric not in candidate_metrics:
            failures.append(f"candidate_missing_metric: {metric}")
            continue
        delta = candidate_metrics[metric] - baseline_metrics[metric]
        if delta < -abs(float(max_recall_regression)):
            failures.append(
                f"recall_regression_{metric}: baseline={baseline_metrics[metric]:.4f} "
                f"candidate={candidate_metrics[metric]:.4f} delta={delta:.4f}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 benchmark regression gate")
    parser.add_argument("--baseline", required=True, help="Path to baseline metrics JSON")
    parser.add_argument("--candidate", required=True, help="Path to candidate metrics JSON")
    parser.add_argument(
        "--min-papers",
        type=int,
        default=10,
        help="Minimum number of papers required in both artifacts",
    )
    parser.add_argument(
        "--max-recall-regression",
        type=float,
        default=0.0,
        help="Allowed negative delta before gate fails",
    )
    parser.add_argument(
        "--required-metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated recall metrics to compare",
    )
    args = parser.parse_args(argv)

    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)
    required_metrics = tuple(
        m.strip() for m in str(args.required_metrics).split(",") if m.strip()
    ) or DEFAULT_METRICS
    failures = evaluate_phase2_gate(
        baseline_payload=baseline,
        candidate_payload=candidate,
        min_papers=max(1, int(args.min_papers)),
        max_recall_regression=float(args.max_recall_regression),
        required_metrics=required_metrics,
    )
    if failures:
        print("Phase 2 benchmark gate: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Phase 2 benchmark gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

