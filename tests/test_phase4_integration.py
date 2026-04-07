"""Phase 4 integration tests: State-Graph Orchestration & Agentic Execution.

Covers gaps identified in architecture review:
- WorkflowState boundary validation at every node
- Sequential fallback correctness (state propagation across nodes)
- Accession regex extraction (GEO/SRA patterns, edge cases)
- Builder retry semantics (stateless retry, max-attempt capping)
- LLM repair fallback chain (LLM → deterministic → no-op)
- BashTool timeout and error capture
- Orchestrator end-to-end (all nodes, sequential path)
- PipelineConfig.execution_order cycle resilience
- _computational_only_method filtering of experimental assays
- _parse_mem_mb edge cases
- Placeholder step generation for assays with no AnalysisSteps
"""

from __future__ import annotations

import subprocess
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import researcher_ai.pipeline.orchestrator as orch_mod
from researcher_ai.models.dataset import Dataset, DataSource
from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayDependency,
    AssayGraph,
    Method,
    MethodCategory,
)
from researcher_ai.models.paper import Paper, PaperSource
from researcher_ai.models.pipeline import (
    Pipeline,
    PipelineBackend,
    PipelineConfig,
    PipelineStep,
)
from researcher_ai.models.software import Software
from researcher_ai.pipeline.bash_tool import BashResult, BashTool
from researcher_ai.pipeline.builder import PipelineBuilder
from researcher_ai.pipeline.orchestrator import (
    WorkflowOrchestrator,
    WorkflowState,
    _collect_accessions,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _step(n: int, sw: str = "star") -> AnalysisStep:
    return AnalysisStep(
        step_number=n,
        description=f"Step {n}",
        input_data=f"in{n}.fastq.gz",
        output_data=f"out{n}.bam",
        software=sw,
        parameters={},
    )


def _assay(name: str = "RNA-seq", n_steps: int = 1, category=MethodCategory.computational) -> Assay:
    return Assay(
        name=name,
        description=f"{name} analysis",
        data_type="sequencing",
        method_category=category,
        steps=[_step(i + 1) for i in range(n_steps)],
    )


def _method(assays=None, dependencies=None) -> Method:
    return Method(
        assay_graph=AssayGraph(
            assays=assays or [_assay()],
            dependencies=dependencies or [],
        )
    )


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


def _stub_orchestrator(monkeypatch, builder_fn=None, max_attempts=2):
    """Return an orchestrator with all parser nodes stubbed out."""
    monkeypatch.setattr(orch_mod, "_HAS_LANGGRAPH", False)
    orch = WorkflowOrchestrator(max_build_attempts=max_attempts)

    fake_paper = MagicMock(spec=Paper)
    fake_paper.raw_text = "GSE12345 SRR9876543"
    fake_paper.sections = []

    fake_method = _method()

    monkeypatch.setattr(orch, "_node_parse_paper", lambda s: {"paper": fake_paper})
    monkeypatch.setattr(orch, "_node_parse_figures", lambda s: {"figures": []})
    monkeypatch.setattr(orch, "_node_parse_methods", lambda s: {"method": fake_method})
    monkeypatch.setattr(orch, "_node_parse_datasets", lambda s: {"datasets": [], "dataset_parse_errors": []})
    monkeypatch.setattr(orch, "_node_parse_software", lambda s: {"software": []})

    if builder_fn:
        monkeypatch.setattr(orch.pipeline_builder, "build", builder_fn)
    else:
        monkeypatch.setattr(
            orch.pipeline_builder,
            "build",
            lambda *a, **kw: _fake_pipeline(passed=True),
        )

    return orch


# ===================================================================
# 1. WorkflowState boundary validation
# ===================================================================


class TestWorkflowStateBoundaryValidation:
    """Every node should reject missing required keys with a clear KeyError."""

    def test_parse_paper_requires_source_keys(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="source"):
            orch._node_parse_paper({})

    def test_parse_figures_requires_paper(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="paper"):
            orch._node_parse_figures({"source": "x"})

    def test_parse_methods_requires_paper(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="paper"):
            orch._node_parse_methods({})

    def test_parse_datasets_requires_paper_and_method(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="paper"):
            orch._node_parse_datasets({})

        with pytest.raises(KeyError, match="method"):
            orch._node_parse_datasets({"paper": MagicMock()})

    def test_parse_software_requires_method(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="method"):
            orch._node_parse_software({})

    def test_build_pipeline_requires_method(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError, match="method"):
            orch._node_build_pipeline({})


# ===================================================================
# 2. Accession regex extraction
# ===================================================================


class TestAccessionExtraction:
    """_collect_accessions should find GEO/SRA accessions and de-duplicate."""

    def test_finds_geo_accessions(self):
        text = "Data deposited in GEO under GSE12345 and GSM9876543."
        result = _collect_accessions(text)
        assert "GSE12345" in result
        assert "GSM9876543" in result

    def test_finds_sra_accessions(self):
        text = "Reads at SRA: SRR1234567, ERP123456, PRJNA123456."
        result = _collect_accessions(text)
        assert "SRR1234567" in result
        assert "ERP123456" in result
        assert "PRJNA123456" in result

    def test_deduplicates_accessions(self):
        text = "GSE12345 ... methods: GSE12345 again."
        result = _collect_accessions(text)
        assert result.count("GSE12345") == 1

    def test_case_insensitive(self):
        text = "gse12345 Gse12345"
        result = _collect_accessions(text)
        assert len(result) == 1
        assert result[0] == "GSE12345"

    def test_no_false_positives_on_short_numbers(self):
        """Accessions with too-few digits should not match."""
        text = "GSE12 SRR12 PRJNA12"
        result = _collect_accessions(text)
        assert result == []

    def test_handles_none_input(self):
        assert _collect_accessions(None) == []

    def test_handles_empty_string(self):
        assert _collect_accessions("") == []

    def test_preserves_order(self):
        text = "PRJNA999999 GSE11111 SRR222222"
        result = _collect_accessions(text)
        assert result == ["PRJNA999999", "GSE11111", "SRR222222"]


# ===================================================================
# 3. Sequential fallback: end-to-end state propagation
# ===================================================================


class TestSequentialFallback:
    """The _run_sequential path must propagate state across all nodes."""

    def test_sequential_run_produces_completed_state(self, monkeypatch):
        orch = _stub_orchestrator(monkeypatch)
        state = orch.run("dummy", PaperSource.PMID)

        assert state["stage"] == "completed"
        assert state["progress"] == 100
        assert state["pipeline"] is not None
        assert state["build_attempts"] == 1

    def test_sequential_state_carries_all_intermediate_keys(self, monkeypatch):
        orch = _stub_orchestrator(monkeypatch)
        state = orch.run("dummy", PaperSource.PMID)

        # All intermediate results should be in final state
        for key in ("paper", "figures", "method", "datasets", "software", "pipeline"):
            assert key in state, f"Expected '{key}' in final state"

    def test_sequential_retry_increments_build_attempts(self, monkeypatch):
        calls = {"n": 0}

        def _build(*a, **kw):
            calls["n"] += 1
            return _fake_pipeline(passed=calls["n"] >= 3)

        orch = _stub_orchestrator(monkeypatch, builder_fn=_build, max_attempts=5)
        state = orch.run("dummy", PaperSource.PMID)

        assert calls["n"] == 3
        assert state["build_attempts"] == 3
        assert state["stage"] == "completed"


class TestLangGraphPath:
    """The LangGraph execution path should run when dependency is available."""

    @pytest.mark.skipif(not orch_mod._HAS_LANGGRAPH, reason="LangGraph not installed")
    def test_run_uses_graph_path_not_sequential(self, monkeypatch):
        orch = WorkflowOrchestrator(max_build_attempts=2)

        monkeypatch.setattr(
            orch,
            "_run_sequential",
            lambda state: (_ for _ in ()).throw(AssertionError("sequential fallback should not run")),
        )
        monkeypatch.setattr(orch, "_node_parse_paper", lambda s: {"paper": MagicMock(spec=Paper)})
        monkeypatch.setattr(orch, "_node_parse_figures", lambda s: {"figures": []})
        monkeypatch.setattr(orch, "_node_parse_methods", lambda s: {"method": _method()})
        monkeypatch.setattr(orch, "_node_parse_datasets", lambda s: {"datasets": [], "dataset_parse_errors": []})
        monkeypatch.setattr(orch, "_node_parse_software", lambda s: {"software": []})
        monkeypatch.setattr(orch.pipeline_builder, "build", lambda *a, **kw: _fake_pipeline(passed=True))

        state = orch.run("dummy", PaperSource.PMID)
        assert state["stage"] == "completed"
        assert state["progress"] == 100
        assert state["pipeline"].validation_report["passed"] is True


# ===================================================================
# 4. Builder retry semantics
# ===================================================================


class TestBuilderRetrySematics:
    """Retry logic edge cases in the orchestrator."""

    def test_max_attempts_of_one_runs_build_once(self, monkeypatch):
        calls = {"n": 0}

        def _build(*a, **kw):
            calls["n"] += 1
            return _fake_pipeline(passed=False)

        orch = _stub_orchestrator(monkeypatch, builder_fn=_build, max_attempts=1)
        state = orch.run("dummy", PaperSource.PMID)

        assert calls["n"] == 1
        assert state["build_attempts"] == 1
        assert state["pipeline"].validation_report["passed"] is False

    def test_pipeline_none_ends_immediately(self, monkeypatch):
        """_next_after_build_pipeline should return 'end' when pipeline is None."""
        orch = WorkflowOrchestrator()
        assert orch._next_after_build_pipeline({}) == "end"

    def test_passed_true_ends_immediately(self, monkeypatch):
        orch = WorkflowOrchestrator()
        state = {"pipeline": _fake_pipeline(passed=True), "build_attempts": 1}
        assert orch._next_after_build_pipeline(state) == "end"

    def test_failed_under_max_retries(self):
        orch = WorkflowOrchestrator(max_build_attempts=3)
        state = {
            "pipeline": _fake_pipeline(passed=False),
            "build_attempts": 1,
            "max_build_attempts": 3,
        }
        assert orch._next_after_build_pipeline(state) == "build_pipeline"

    def test_failed_at_max_retries_ends(self):
        orch = WorkflowOrchestrator(max_build_attempts=2)
        state = {
            "pipeline": _fake_pipeline(passed=False),
            "build_attempts": 2,
            "max_build_attempts": 2,
        }
        assert orch._next_after_build_pipeline(state) == "end"


# ===================================================================
# 5. BashTool edge cases
# ===================================================================


class TestBashToolEdgeCases:
    """BashTool should handle timeout, missing binary, and non-zero exit."""

    def test_timeout_returns_error(self, monkeypatch):
        tool = BashTool(timeout_seconds=1)

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="snakemake", timeout=1)

        monkeypatch.setattr(subprocess, "run", _timeout)
        result = tool.run(["snakemake", "--lint"], cwd="/tmp")
        assert result.status == "error"
        assert "TimeoutExpired" in result.stderr

    def test_nonzero_exit_returns_error(self, monkeypatch):
        tool = BashTool(timeout_seconds=5)

        def _fail(*args, **kwargs):
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="lint error: missing rule all",
            )

        monkeypatch.setattr(subprocess, "run", _fail)
        result = tool.run(["snakemake", "--lint"], cwd="/tmp")
        assert result.status == "error"
        assert result.returncode == 1
        assert "lint error" in result.stderr

    def test_success_returns_ok(self, monkeypatch):
        tool = BashTool(timeout_seconds=5)

        def _ok(*args, **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="all good", stderr="")

        monkeypatch.setattr(subprocess, "run", _ok)
        result = tool.run(["snakemake", "-n"], cwd="/tmp")
        assert result.status == "ok"
        assert result.returncode == 0

    def test_truncates_long_output(self, monkeypatch):
        tool = BashTool(timeout_seconds=5)
        long_text = "x" * 10000

        def _long(*args, **kwargs):
            return types.SimpleNamespace(returncode=0, stdout=long_text, stderr=long_text)

        monkeypatch.setattr(subprocess, "run", _long)
        result = tool.run(["echo", "hi"], cwd="/tmp")
        assert len(result.stdout) <= 5000
        assert len(result.stderr) <= 5000


# ===================================================================
# 6. LLM repair fallback chain
# ===================================================================


class TestRepairSnakefileFallback:
    """_repair_snakefile should try LLM first, then deterministic fixes."""

    def test_llm_repair_used_when_available(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        monkeypatch.setattr(
            builder,
            "repair_snakefile_with_llm",
            lambda sf, err: "rule all:\n    input: 'fixed.txt'\n",
        )
        result = builder._repair_snakefile("broken content", "rule all missing")
        assert "fixed.txt" in result

    def test_deterministic_fallback_when_llm_returns_unchanged(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        # LLM returns content unchanged → triggers deterministic path
        monkeypatch.setattr(
            builder,
            "repair_snakefile_with_llm",
            lambda sf, err: sf,
        )
        result = builder._repair_snakefile(
            "rule x:\n    output: 'x.txt'\n",
            "rule all missing in lint output",
        )
        assert "rule all:" in result

    def test_deterministic_adds_configfile(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        monkeypatch.setattr(builder, "repair_snakefile_with_llm", lambda sf, err: sf)
        result = builder._repair_snakefile_deterministic(
            "rule all:\n    input: []\n",
            "configfile directive not found",
        )
        assert 'configfile: "config.yaml"' in result

    def test_deterministic_adds_min_version_import(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        result = builder._repair_snakefile_deterministic(
            "rule all:\n    input: []\n",
            "min_version not defined",
        )
        assert "from snakemake.utils import min_version" in result

    def test_llm_repair_returns_original_on_empty_error(self):
        builder = PipelineBuilder(validation_max_rounds=1)
        original = "rule all:\n    input: []\n"
        assert builder.repair_snakefile_with_llm(original, "") == original
        assert builder.repair_snakefile_with_llm(original, "   ") == original


# ===================================================================
# 7. _computational_only_method filtering
# ===================================================================


class TestComputationalOnlyFiltering:
    """Builder should filter out experimental assays and prune invalid edges."""

    def test_removes_experimental_assays(self):
        builder = PipelineBuilder(validation_max_rounds=1)
        method = _method(
            assays=[
                _assay("RNA-seq", category=MethodCategory.computational),
                _assay("Western Blot", category=MethodCategory.experimental),
            ]
        )
        filtered = builder._computational_only_method(method)
        names = [a.name for a in filtered.assay_graph.assays]
        assert "RNA-seq" in names
        assert "Western Blot" not in names

    def test_prunes_dangling_dependency_edges(self):
        builder = PipelineBuilder(validation_max_rounds=1)
        method = _method(
            assays=[
                _assay("RNA-seq", category=MethodCategory.computational),
                _assay("Western Blot", category=MethodCategory.experimental),
            ],
            dependencies=[
                AssayDependency(
                    upstream_assay="Western Blot",
                    downstream_assay="RNA-seq",
                    dependency_type="normalization_reference",
                )
            ],
        )
        filtered = builder._computational_only_method(method)
        assert len(filtered.assay_graph.dependencies) == 0

    def test_preserves_valid_computational_edges(self):
        builder = PipelineBuilder(validation_max_rounds=1)
        method = _method(
            assays=[
                _assay("RNA-seq", category=MethodCategory.computational),
                _assay("CLIP-seq", category=MethodCategory.computational),
            ],
            dependencies=[
                AssayDependency(
                    upstream_assay="RNA-seq",
                    downstream_assay="CLIP-seq",
                    dependency_type="normalization_reference",
                )
            ],
        )
        filtered = builder._computational_only_method(method)
        assert len(filtered.assay_graph.dependencies) == 1


# ===================================================================
# 8. _parse_mem_mb edge cases
# ===================================================================


class TestParseMemMb:
    """Memory string parsing should handle varied formats."""

    def test_gigabytes(self):
        builder = PipelineBuilder()
        assert builder._parse_mem_mb("64G") == 65536
        assert builder._parse_mem_mb("8g") == 8192

    def test_megabytes(self):
        builder = PipelineBuilder()
        assert builder._parse_mem_mb("512M") == 512
        assert builder._parse_mem_mb("1024m") == 1024

    def test_raw_number(self):
        builder = PipelineBuilder()
        assert builder._parse_mem_mb("4096") == 4096

    def test_invalid_returns_default(self):
        builder = PipelineBuilder()
        assert builder._parse_mem_mb("lots") == 65536

    def test_fractional_gigabytes(self):
        builder = PipelineBuilder()
        assert builder._parse_mem_mb("1.5G") == 1536


# ===================================================================
# 9. Placeholder step generation
# ===================================================================


class TestPlaceholderSteps:
    """Assays with no AnalysisSteps should produce placeholder steps."""

    def test_empty_steps_produce_placeholders(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        monkeypatch.setattr(
            builder,
            "_validate_and_repair_snakefile",
            lambda *, snakefile_content, profile: (snakefile_content, {"passed": True, "attempts": []}),
        )
        assay = Assay(
            name="RNA-seq",
            description="RNA-seq analysis",
            data_type="sequencing",
            method_category=MethodCategory.computational,
            steps=[],
        )
        method = Method(assay_graph=AssayGraph(assays=[assay], dependencies=[]))
        pipeline = builder.build(method=method, datasets=[], software=[], figures=[])
        assert len(pipeline.config.steps) == 1
        assert "rna_seq_run" == pipeline.config.steps[0].step_id
        assert "TODO" in pipeline.config.steps[0].command

    def test_multiple_placeholder_steps_are_chained(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=1)
        monkeypatch.setattr(
            builder,
            "_validate_and_repair_snakefile",
            lambda *, snakefile_content, profile: (snakefile_content, {"passed": True, "attempts": []}),
        )
        assay_a = Assay(
            name="Alignment",
            description="Read alignment",
            data_type="sequencing",
            method_category=MethodCategory.computational,
            steps=[],
        )
        assay_b = Assay(
            name="Counting",
            description="Feature counting",
            data_type="sequencing",
            method_category=MethodCategory.computational,
            steps=[],
        )
        method = Method(assay_graph=AssayGraph(assays=[assay_a, assay_b], dependencies=[]))
        pipeline = builder.build(method=method, datasets=[], software=[], figures=[])
        steps = pipeline.config.steps
        assert len(steps) == 2
        # Second placeholder depends on the first
        assert steps[1].depends_on == [steps[0].step_id]


# ===================================================================
# 10. PipelineConfig.execution_order
# ===================================================================


class TestExecutionOrder:
    """execution_order should topologically sort step_ids."""

    def test_linear_chain(self):
        steps = [
            PipelineStep(step_id="a", name="A", description="A", software="x", command="x", depends_on=[]),
            PipelineStep(step_id="b", name="B", description="B", software="x", command="x", depends_on=["a"]),
            PipelineStep(step_id="c", name="C", description="C", software="x", command="x", depends_on=["b"]),
        ]
        config = PipelineConfig(name="t", description="t", backend=PipelineBackend.SNAKEMAKE, steps=steps)
        order = config.execution_order()
        assert order == ["a", "b", "c"]

    def test_diamond_dag(self):
        steps = [
            PipelineStep(step_id="root", name="R", description="R", software="x", command="x", depends_on=[]),
            PipelineStep(step_id="left", name="L", description="L", software="x", command="x", depends_on=["root"]),
            PipelineStep(step_id="right", name="Ri", description="Ri", software="x", command="x", depends_on=["root"]),
            PipelineStep(step_id="join", name="J", description="J", software="x", command="x", depends_on=["left", "right"]),
        ]
        config = PipelineConfig(name="t", description="t", backend=PipelineBackend.SNAKEMAKE, steps=steps)
        order = config.execution_order()
        assert order.index("root") < order.index("left")
        assert order.index("root") < order.index("right")
        assert order.index("left") < order.index("join")
        assert order.index("right") < order.index("join")

    def test_no_dependencies(self):
        steps = [
            PipelineStep(step_id="x", name="X", description="X", software="x", command="x"),
            PipelineStep(step_id="y", name="Y", description="Y", software="x", command="x"),
        ]
        config = PipelineConfig(name="t", description="t", backend=PipelineBackend.SNAKEMAKE, steps=steps)
        order = config.execution_order()
        assert set(order) == {"x", "y"}

    def test_missing_dependency_ref_does_not_crash(self):
        """If depends_on references a non-existent step_id, execution_order should not crash."""
        steps = [
            PipelineStep(step_id="a", name="A", description="A", software="x", command="x", depends_on=["ghost"]),
        ]
        config = PipelineConfig(name="t", description="t", backend=PipelineBackend.SNAKEMAKE, steps=steps)
        order = config.execution_order()
        assert "a" in order


# ===================================================================
# 11. Validation loop integration
# ===================================================================


class TestValidationLoopIntegration:
    """Test the full validate → repair → re-validate cycle."""

    def test_lint_passes_then_dry_run_passes(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=2)

        def _fake_run(cmd, *, cwd):
            return BashResult(status="ok", cmd=" ".join(cmd), returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(builder.bash_tool, "run", _fake_run)
        content, report = builder._validate_and_repair_snakefile(
            snakefile_content="rule all:\n    input: []\n",
            profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
        )
        assert report["passed"] is True

    def test_lint_fails_then_repair_fixes_it(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=2)
        call_count = {"n": 0}

        def _fake_run(cmd, *, cwd):
            call_count["n"] += 1
            # First call (lint round 1) fails, subsequent calls pass
            if call_count["n"] == 1:
                return BashResult(status="error", cmd=" ".join(cmd), returncode=1, stderr="rule all missing")
            return BashResult(status="ok", cmd=" ".join(cmd), returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(builder.bash_tool, "run", _fake_run)
        monkeypatch.setattr(builder, "repair_snakefile_with_llm", lambda sf, err: sf)

        content, report = builder._validate_and_repair_snakefile(
            snakefile_content="rule x:\n    output: 'x.txt'\n",
            profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
        )
        # Deterministic repair should have added rule all
        assert "rule all:" in content

    def test_all_rounds_fail_returns_passed_false(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=2)

        def _always_fail(cmd, *, cwd):
            return BashResult(status="error", cmd=" ".join(cmd), returncode=1, stderr="persistent error")

        monkeypatch.setattr(builder.bash_tool, "run", _always_fail)
        monkeypatch.setattr(builder, "repair_snakefile_with_llm", lambda sf, err: sf)

        _, report = builder._validate_and_repair_snakefile(
            snakefile_content="broken",
            profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
        )
        assert report["passed"] is False
        assert len(report["attempts"]) == 2  # One per round

    def test_retry_repairs_receive_cumulative_error_context(self, monkeypatch):
        builder = PipelineBuilder(validation_max_rounds=2)
        captured_contexts: list[str] = []
        errs = iter(["first lint error", "second lint error"])

        def _always_lint_fail(cmd, *, cwd):  # noqa: ARG001
            return BashResult(status="error", cmd=" ".join(cmd), returncode=1, stderr=next(errs))

        def _capture_repair(snakefile_content: str, error_text: str) -> str:
            captured_contexts.append(error_text)
            return snakefile_content

        monkeypatch.setattr(builder.bash_tool, "run", _always_lint_fail)
        monkeypatch.setattr(builder, "_repair_snakefile", _capture_repair)

        _, report = builder._validate_and_repair_snakefile(
            snakefile_content="rule x:\n    output: 'x.txt'\n",
            profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
        )

        assert report["passed"] is False
        assert len(captured_contexts) == 2
        assert "first lint error" in captured_contexts[0]
        assert "first lint error" in captured_contexts[1]
        assert "second lint error" in captured_contexts[1]


# ===================================================================
# 12. Resource inference
# ===================================================================


class TestResourceInference:
    """Builder should infer correct threads/memory per tool category."""

    def test_heavy_tools(self):
        builder = PipelineBuilder()
        for tool in ("star", "hisat2", "bwa", "bowtie2", "cellranger", "bismark"):
            threads, mem = builder._infer_resources(tool)
            assert threads == 16 and mem == 64, f"Failed for {tool}"

    def test_medium_tools(self):
        builder = PipelineBuilder()
        for tool in ("samtools", "picard", "trimgalore", "fastp", "kallisto", "salmon"):
            threads, mem = builder._infer_resources(tool)
            assert threads == 8 and mem == 16, f"Failed for {tool}"

    def test_unknown_tool_gets_defaults(self):
        builder = PipelineBuilder()
        threads, mem = builder._infer_resources("my_custom_tool")
        assert threads == 4 and mem == 8

    def test_case_insensitive(self):
        builder = PipelineBuilder()
        assert builder._infer_resources("STAR") == (16, 64)
        assert builder._infer_resources("Samtools") == (8, 16)


# ===================================================================
# 13. nf-core detection
# ===================================================================


class TestNfCoreDetection:
    """_check_nfcore should use whole-word matching."""

    def test_rnaseq_match(self):
        builder = PipelineBuilder()
        assert builder._check_nfcore("RNA-seq") == "rnaseq"
        assert builder._check_nfcore("Standard RNA-seq analysis") == "rnaseq"

    def test_wes_does_not_match_western(self):
        builder = PipelineBuilder()
        assert builder._check_nfcore("Western blot") is None

    def test_no_match_returns_none(self):
        builder = PipelineBuilder()
        assert builder._check_nfcore("Proteomics") is None

    def test_nfcore_module_mapping(self):
        builder = PipelineBuilder()
        assert builder._check_nfcore_module("star") == "star/align"
        assert builder._check_nfcore_module("SALMON") == "salmon/quant"
        assert builder._check_nfcore_module("unknown_tool") is None


# ===================================================================
# 14. Topo sort cycle detection
# ===================================================================


class TestTopoSortCycleDetection:
    """_topo_sort_assays should fall back to original order on cycles."""

    def test_cycle_falls_back_to_original_order(self):
        builder = PipelineBuilder()
        method = _method(
            assays=[
                _assay("A", category=MethodCategory.computational),
                _assay("B", category=MethodCategory.computational),
            ],
            dependencies=[
                AssayDependency(upstream_assay="A", downstream_assay="B", dependency_type="dep"),
                AssayDependency(upstream_assay="B", downstream_assay="A", dependency_type="dep"),
            ],
        )
        result = builder._topo_sort_assays(method)
        # Falls back to original order
        assert result == ["A", "B"]

    def test_no_dependencies_preserves_order(self):
        builder = PipelineBuilder()
        method = _method(
            assays=[
                _assay("Z", category=MethodCategory.computational),
                _assay("A", category=MethodCategory.computational),
            ],
        )
        result = builder._topo_sort_assays(method)
        assert result == ["Z", "A"]

    def test_correct_ordering_with_dependencies(self):
        builder = PipelineBuilder()
        method = _method(
            assays=[
                _assay("CLIP-seq", category=MethodCategory.computational),
                _assay("RNA-seq", category=MethodCategory.computational),
            ],
            dependencies=[
                AssayDependency(
                    upstream_assay="RNA-seq",
                    downstream_assay="CLIP-seq",
                    dependency_type="normalization",
                )
            ],
        )
        result = builder._topo_sort_assays(method)
        assert result.index("RNA-seq") < result.index("CLIP-seq")


# ===================================================================
# 15. SLURM profile resolution
# ===================================================================


class TestSlurmProfileResolution:
    """_slurm_profile should resolve profile by name."""

    def test_tscc_default(self):
        builder = PipelineBuilder(hpc_profile_name="tscc")
        profile = builder._slurm_profile()
        assert profile["partition"] == "hotel"
        assert profile["account"] == "csd786"

    def test_local_profile(self):
        builder = PipelineBuilder(hpc_profile_name="local")
        profile = builder._slurm_profile()
        assert profile["partition"] == "local"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TSCC_SLURM_PARTITION", "gpu-shared")
        monkeypatch.setenv("TSCC_SLURM_ACCOUNT", "mylab123")
        builder = PipelineBuilder(hpc_profile_name="tscc")
        profile = builder._slurm_profile()
        assert profile["partition"] == "gpu-shared"
        assert profile["account"] == "mylab123"


# ===================================================================
# 16. _sanitize_id
# ===================================================================


class TestSanitizeId:
    """_sanitize_id should produce valid snake_case identifiers."""

    def test_basic(self):
        from researcher_ai.pipeline.builder import _sanitize_id
        assert _sanitize_id("RNA-seq") == "rna_seq"

    def test_special_characters(self):
        from researcher_ai.pipeline.builder import _sanitize_id
        assert _sanitize_id("Hi-C (2.0)") == "hi_c_2_0"

    def test_empty_string(self):
        from researcher_ai.pipeline.builder import _sanitize_id
        assert _sanitize_id("") == "step"

    def test_already_valid(self):
        from researcher_ai.pipeline.builder import _sanitize_id
        assert _sanitize_id("align_reads") == "align_reads"
