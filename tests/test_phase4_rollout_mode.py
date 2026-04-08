from __future__ import annotations

import researcher_ai.pipeline.orchestrator as orch_mod
from researcher_ai.models.method import (
    AssayGraph,
    EvidenceCategory,
    Method,
    ValidationReport,
    ValidationVerdict,
)
from researcher_ai.models.paper import PaperSource
from researcher_ai.pipeline.orchestrator import WorkflowOrchestrator, _normalize_bioworkflow_mode


def _minimal_method() -> Method:
    return Method(assay_graph=AssayGraph(assays=[], dependencies=[]))


def _stub_nodes(orch: WorkflowOrchestrator, monkeypatch) -> None:
    monkeypatch.setattr(orch, "_node_parse_paper", lambda state: {"paper": object()})
    monkeypatch.setattr(orch, "_node_parse_figures", lambda state: {"figures": []})
    monkeypatch.setattr(orch, "_node_parse_methods", lambda state: {"method": _minimal_method()})
    monkeypatch.setattr(orch, "_node_parse_datasets", lambda state: {"datasets": [], "dataset_parse_errors": []})
    monkeypatch.setattr(orch, "_node_parse_software", lambda state: {"software": []})
    monkeypatch.setattr(
        orch,
        "_node_build_workflow_graph",
        lambda state: {"workflow_graph": object(), "workflow_graph_validation_issues": []},
    )


def test_normalize_bioworkflow_mode_aliases():
    assert _normalize_bioworkflow_mode("legacy") == "off"
    assert _normalize_bioworkflow_mode("warn+continue") == "warn"
    assert _normalize_bioworkflow_mode("strict") == "on"
    assert _normalize_bioworkflow_mode("unknown-value") == "warn"


def test_bioworkflow_mode_off_skips_validation(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orch = WorkflowOrchestrator(bioworkflow_mode="off")
    _stub_nodes(orch, monkeypatch)
    monkeypatch.setattr(
        orch.validation_agent,
        "validate",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("validate should be skipped in off mode")),
    )
    monkeypatch.setattr(
        orch.pipeline_builder,
        "build",
        lambda *args, **kwargs: type("P", (), {"validation_report": {"passed": True}})(),
    )
    state = orch.run("dummy", PaperSource.PMID)
    assert state["stage"] == "completed"
    assert "method_validation_report" not in state


def test_bioworkflow_mode_on_blocks_on_ungrounded(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orch = WorkflowOrchestrator(bioworkflow_mode="on")
    _stub_nodes(orch, monkeypatch)
    report = ValidationReport(
        verdicts=[
            ValidationVerdict(
                field="a",
                claimed_value="v",
                evidence_category=EvidenceCategory.ungrounded,
                action="flag_ungrounded",
            )
        ],
        ungrounded_count=1,
        inferred_default_count=0,
        total_fields_checked=1,
        warnings=[],
    )
    monkeypatch.setattr(orch.validation_agent, "validate", lambda **kwargs: report)
    calls = {"n": 0}

    def _build(*args, **kwargs):
        calls["n"] += 1
        return type("P", (), {"validation_report": {"passed": True}})()

    monkeypatch.setattr(orch.pipeline_builder, "build", _build)
    state = orch.run("dummy", PaperSource.PMID)
    assert state["stage"] == "needs_human_review"
    assert state["human_review_required"] is True
    assert state["human_review_summary"]["ungrounded_count"] == 1
    assert state["human_review_summary"]["ungrounded_fields"] == ["a"]
    assert calls["n"] == 0
