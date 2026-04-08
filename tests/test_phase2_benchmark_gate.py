from __future__ import annotations

from researcher_ai.benchmarks.phase2_gate import evaluate_phase2_gate


def test_phase2_benchmark_gate_passes_with_no_regression():
    baseline = {
        "paper_count": 10,
        "metrics": {
            "overall_recall": 0.81,
            "heading_recall": 0.72,
            "merge_recall": 0.85,
        },
    }
    candidate = {
        "paper_count": 10,
        "metrics": {
            "overall_recall": 0.82,
            "heading_recall": 0.72,
            "merge_recall": 0.86,
        },
    }
    failures = evaluate_phase2_gate(
        baseline_payload=baseline,
        candidate_payload=candidate,
        min_papers=10,
        max_recall_regression=0.0,
    )
    assert failures == []


def test_phase2_benchmark_gate_fails_on_recall_regression_and_coverage():
    baseline = {
        "paper_count": 10,
        "metrics": {
            "overall_recall": 0.81,
            "heading_recall": 0.72,
            "merge_recall": 0.85,
        },
    }
    candidate = {
        "paper_count": 8,
        "metrics": {
            "overall_recall": 0.79,
            "heading_recall": 0.71,
            "merge_recall": 0.84,
        },
    }
    failures = evaluate_phase2_gate(
        baseline_payload=baseline,
        candidate_payload=candidate,
        min_papers=10,
        max_recall_regression=0.0,
    )
    joined = "\n".join(failures)
    assert "candidate_paper_count_below_minimum" in joined
    assert "recall_regression_overall_recall" in joined
    assert "recall_regression_heading_recall" in joined
    assert "recall_regression_merge_recall" in joined
