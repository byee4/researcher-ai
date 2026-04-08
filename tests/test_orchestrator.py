from __future__ import annotations

import pytest
import researcher_ai.pipeline.orchestrator as orch_mod
from researcher_ai.models.method import Method
from researcher_ai.models.paper import PaperSource
from researcher_ai.models.pipeline import Pipeline, PipelineBackend, PipelineConfig
from researcher_ai.pipeline.orchestrator import WorkflowOrchestrator


def _fake_pipeline(passed: bool) -> Pipeline:
    return Pipeline(
        config=PipelineConfig(
            name="test",
            description="test",
            backend=PipelineBackend.SNAKEMAKE,
            steps=[],
        ),
        validation_report={"passed": passed, "attempts": []},
    )


def test_orchestrator_retries_builder_after_failed_validation(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orchestrator = WorkflowOrchestrator(max_build_attempts=3)

    monkeypatch.setattr(orchestrator, "_node_parse_paper", lambda state: {"paper": object()})
    monkeypatch.setattr(orchestrator, "_node_parse_figures", lambda state: {"figures": []})
    monkeypatch.setattr(orchestrator, "_node_parse_methods", lambda state: {"method": Method()})
    monkeypatch.setattr(orchestrator, "_node_parse_datasets", lambda state: {"datasets": [], "dataset_parse_errors": []})
    monkeypatch.setattr(orchestrator, "_node_parse_software", lambda state: {"software": []})

    calls = {"n": 0}

    def _build(*args, **kwargs):
        calls["n"] += 1
        return _fake_pipeline(passed=calls["n"] >= 2)

    monkeypatch.setattr(orchestrator.pipeline_builder, "build", _build)
    state = orchestrator.run("dummy", PaperSource.PMID)

    assert calls["n"] == 2
    assert state["build_attempts"] == 2
    assert state["stage"] == "completed"
    assert state["pipeline"].validation_report["passed"] is True


def test_orchestrator_stops_after_max_builder_attempts(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orchestrator = WorkflowOrchestrator(max_build_attempts=2)

    monkeypatch.setattr(orchestrator, "_node_parse_paper", lambda state: {"paper": object()})
    monkeypatch.setattr(orchestrator, "_node_parse_figures", lambda state: {"figures": []})
    monkeypatch.setattr(orchestrator, "_node_parse_methods", lambda state: {"method": Method()})
    monkeypatch.setattr(orchestrator, "_node_parse_datasets", lambda state: {"datasets": [], "dataset_parse_errors": []})
    monkeypatch.setattr(orchestrator, "_node_parse_software", lambda state: {"software": []})
    monkeypatch.setattr(orchestrator.pipeline_builder, "build", lambda *args, **kwargs: _fake_pipeline(passed=False))

    state = orchestrator.run("dummy", PaperSource.PMID)
    assert state["build_attempts"] == 2
    assert state["stage"] == "builder_retry"
    assert state["pipeline"].validation_report["passed"] is False


def test_orchestrator_state_validation_raises_on_missing_paper(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orchestrator = WorkflowOrchestrator(max_build_attempts=1)
    with pytest.raises(KeyError):
        orchestrator._node_parse_figures({"source": "x", "source_type": PaperSource.PMID})
