"""Phase 4 end-to-end benchmark comparison helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


_METRICS = (
    "recall",
    "grounded_precision",
    "retrieval_rounds",
    "latency_seconds",
    "token_cost",
)


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _paper_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    papers = payload.get("papers", [])
    if not isinstance(papers, list):
        return []
    return [p for p in papers if isinstance(p, dict)]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in _METRICS:
        vals: list[float] = []
        for row in rows:
            try:
                vals.append(float(row.get(metric)))
            except Exception:
                continue
        out[metric] = float(mean(vals)) if vals else 0.0
    return out


def compare_phase4_runs(
    *,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = _paper_rows(baseline_payload)
    candidate_rows = _paper_rows(candidate_payload)
    baseline_agg = _aggregate(baseline_rows)
    candidate_agg = _aggregate(candidate_rows)
    deltas = {k: candidate_agg.get(k, 0.0) - baseline_agg.get(k, 0.0) for k in _METRICS}
    return {
        "baseline": baseline_agg,
        "candidate": candidate_agg,
        "delta": deltas,
        "baseline_paper_count": len(baseline_rows),
        "candidate_paper_count": len(candidate_rows),
    }


def evaluate_phase4_gate(
    *,
    comparison: dict[str, Any],
    min_papers: int = 10,
    allow_recall_regression: float = 0.0,
) -> list[str]:
    failures: list[str] = []
    baseline_count = int(comparison.get("baseline_paper_count", 0))
    candidate_count = int(comparison.get("candidate_paper_count", 0))
    if baseline_count < min_papers:
        failures.append(f"baseline_paper_count_below_minimum: {baseline_count} < {min_papers}")
    if candidate_count < min_papers:
        failures.append(f"candidate_paper_count_below_minimum: {candidate_count} < {min_papers}")

    delta = comparison.get("delta", {})
    recall_delta = float(delta.get("recall", 0.0))
    precision_delta = float(delta.get("grounded_precision", 0.0))
    if recall_delta < -abs(float(allow_recall_regression)):
        failures.append(f"recall_regression: delta={recall_delta:.4f}")
    if precision_delta < 0:
        failures.append(f"grounded_precision_regression: delta={precision_delta:.4f}")
    return failures


def render_markdown_report(comparison: dict[str, Any], failures: list[str]) -> str:
    lines = [
        "# Phase 4 Benchmark Report",
        "",
        f"- Baseline papers: {comparison.get('baseline_paper_count', 0)}",
        f"- Candidate papers: {comparison.get('candidate_paper_count', 0)}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in _METRICS:
        b = float(comparison.get("baseline", {}).get(metric, 0.0))
        c = float(comparison.get("candidate", {}).get(metric, 0.0))
        d = float(comparison.get("delta", {}).get(metric, 0.0))
        lines.append(f"| {metric} | {b:.4f} | {c:.4f} | {d:+.4f} |")
    lines.extend(["", "## Gate", ""])
    if failures:
        lines.append("FAIL")
        lines.extend(f"- {f}" for f in failures)
    else:
        lines.append("PASS")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 benchmark comparison and gate")
    parser.add_argument("--baseline", required=True, help="Baseline JSON artifact path")
    parser.add_argument("--candidate", required=True, help="Candidate JSON artifact path")
    parser.add_argument("--min-papers", type=int, default=10)
    parser.add_argument("--allow-recall-regression", type=float, default=0.0)
    parser.add_argument("--report-out", default="", help="Optional markdown report output path")
    parser.add_argument("--json-out", default="", help="Optional JSON summary output path")
    parser.add_argument("--gate", action="store_true", help="Return non-zero on gate failure")
    args = parser.parse_args(argv)

    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)
    comparison = compare_phase4_runs(
        baseline_payload=baseline,
        candidate_payload=candidate,
    )
    failures = evaluate_phase4_gate(
        comparison=comparison,
        min_papers=max(1, int(args.min_papers)),
        allow_recall_regression=float(args.allow_recall_regression),
    )
    report = render_markdown_report(comparison, failures)
    print(report)
    if args.report_out:
        Path(args.report_out).write_text(report, encoding="utf-8")
    if args.json_out:
        payload = {"comparison": comparison, "failures": failures}
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.gate and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

