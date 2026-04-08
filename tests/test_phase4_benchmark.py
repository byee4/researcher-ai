from __future__ import annotations

from researcher_ai.benchmarks.phase4_benchmark import (
    compare_phase4_runs,
    evaluate_phase4_gate,
    render_markdown_report,
)


def _payload(rows):
    return {"papers": rows}


def test_phase4_compare_aggregates_and_deltas():
    baseline = _payload(
        [
            {"recall": 0.8, "grounded_precision": 0.7, "retrieval_rounds": 1.2, "latency_seconds": 10.0, "token_cost": 0.4},
            {"recall": 0.9, "grounded_precision": 0.8, "retrieval_rounds": 1.4, "latency_seconds": 12.0, "token_cost": 0.5},
        ]
    )
    candidate = _payload(
        [
            {"recall": 0.9, "grounded_precision": 0.85, "retrieval_rounds": 1.6, "latency_seconds": 13.0, "token_cost": 0.55},
            {"recall": 0.92, "grounded_precision": 0.86, "retrieval_rounds": 1.8, "latency_seconds": 11.0, "token_cost": 0.52},
        ]
    )
    out = compare_phase4_runs(baseline_payload=baseline, candidate_payload=candidate)
    assert out["baseline_paper_count"] == 2
    assert out["candidate_paper_count"] == 2
    assert out["delta"]["recall"] > 0
    assert out["delta"]["grounded_precision"] > 0


def test_phase4_gate_fails_on_recall_or_precision_regression():
    comparison = {
        "baseline_paper_count": 10,
        "candidate_paper_count": 10,
        "delta": {"recall": -0.01, "grounded_precision": -0.02},
    }
    failures = evaluate_phase4_gate(comparison=comparison, min_papers=10, allow_recall_regression=0.0)
    assert any("recall_regression" in x for x in failures)
    assert any("grounded_precision_regression" in x for x in failures)


def test_phase4_markdown_report_contains_gate_status():
    comparison = {
        "baseline": {"recall": 0.8, "grounded_precision": 0.7, "retrieval_rounds": 1.0, "latency_seconds": 9.0, "token_cost": 0.4},
        "candidate": {"recall": 0.81, "grounded_precision": 0.71, "retrieval_rounds": 1.2, "latency_seconds": 9.5, "token_cost": 0.42},
        "delta": {"recall": 0.01, "grounded_precision": 0.01, "retrieval_rounds": 0.2, "latency_seconds": 0.5, "token_cost": 0.02},
        "baseline_paper_count": 10,
        "candidate_paper_count": 10,
    }
    report = render_markdown_report(comparison, failures=[])
    assert "Phase 4 Benchmark Report" in report
    assert "PASS" in report

