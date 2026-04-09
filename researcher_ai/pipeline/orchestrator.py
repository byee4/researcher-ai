"""State-graph workflow orchestrator for researcher-ai.

Phase 4 architecture:
- Uses a LangGraph state machine when available.
- Falls back to deterministic sequential execution if LangGraph is unavailable.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from typing import Any, Optional, TypedDict

from researcher_ai.models.dataset import Dataset
from researcher_ai.models.method import Method
from researcher_ai.models.paper import Paper, PaperSource
from researcher_ai.models.pipeline import Pipeline
from researcher_ai.models.software import Software
from researcher_ai.models.method import ValidationReport
from researcher_ai.models.workflow_graph import GraphValidationIssue, WorkflowGraph
from researcher_ai.parsers.data.geo_parser import GEOParser
from researcher_ai.parsers.data.sra_parser import SRAParser
from researcher_ai.parsers.figure_parser import FigureParser
from researcher_ai.parsers.methods_parser import MethodsParser
from researcher_ai.parsers.paper_parser import PaperParser
from researcher_ai.parsers.validation_agent import ValidationAgent
from researcher_ai.parsers.software_parser import SoftwareParser
from researcher_ai.pipeline.builder import PipelineBuilder
from researcher_ai.pipeline.workflow_graph_mapper import build_workflow_graph

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional runtime dependency
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - optional runtime dependency
    END = "__end__"
    StateGraph = None
    _HAS_LANGGRAPH = False


_ACC_RE = re.compile(
    r"\b("
    r"GSE\d{4,8}|GSM\d{4,8}|GDS\d{3,7}|GPL\d{3,7}|"
    r"SRP\d{4,9}|SRX\d{4,9}|SRR\d{4,9}|ERP\d{4,9}|ERR\d{4,9}|"
    r"PRJNA\d{4,9}|PRJEB\d{4,9}"
    r")\b",
    re.IGNORECASE,
)


class WorkflowState(TypedDict, total=False):
    source: str
    source_type: PaperSource
    progress: int
    stage: str
    paper: Paper
    figures: list[Any]
    method: Method
    datasets: list[Dataset]
    dataset_parse_errors: list[str]
    figure_parse_errors: list[str]
    software: list[Software]
    workflow_graph: WorkflowGraph
    workflow_graph_validation_issues: list[GraphValidationIssue]
    method_validation_report: ValidationReport
    validation_blocked: bool
    human_review_required: bool
    human_review_summary: dict[str, Any]
    pipeline: Pipeline
    build_attempts: int
    max_build_attempts: int


def _collect_accessions(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ACC_RE.finditer(text or ""):
        acc = m.group(1).upper()
        if acc not in seen:
            seen.add(acc)
            out.append(acc)
    return out


def _parse_dataset(accession: str) -> Optional[Dataset]:
    if accession.startswith(("GSE", "GSM", "GDS", "GPL")):
        return GEOParser().parse(accession)
    if accession.startswith(("SRP", "SRX", "SRR", "ERP", "ERR", "PRJNA", "PRJEB")):
        return SRAParser().parse(accession)
    return None


def _normalize_bioworkflow_mode(raw: Optional[str]) -> str:
    """Normalize rollout mode for validation behavior."""
    mode = (raw or "").strip().lower()
    if not mode:
        mode = os.environ.get("RESEARCHER_AI_BIOWORKFLOW_MODE", "warn").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "disabled": "off",
        "legacy": "off",
        "1": "on",
        "true": "on",
        "enabled": "on",
        "strict": "on",
        "warn+continue": "warn",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "warn", "on"}:
        return "warn"
    return mode


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class WorkflowOrchestrator:
    """Agentic/stateful orchestrator for parsing + pipeline generation."""

    def __init__(self, *, max_build_attempts: int = 2, bioworkflow_mode: Optional[str] = None):
        self.paper_parser = PaperParser()
        self.figure_parser = FigureParser()
        self.methods_parser = MethodsParser()
        self.validation_agent = ValidationAgent()
        self.software_parser = SoftwareParser()
        self.pipeline_builder = PipelineBuilder()
        self.max_build_attempts = max_build_attempts
        self.bioworkflow_mode = _normalize_bioworkflow_mode(bioworkflow_mode)
        self.skip_figures = _env_bool("RESEARCHER_AI_SKIP_FIGURES", default=False)
        self.parse_figures_timeout_seconds = float(
            os.environ.get("RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS", "0")
        )

    def run(self, source: str, source_type: PaperSource) -> WorkflowState:
        initial: WorkflowState = {
            "source": source,
            "source_type": source_type,
            "progress": 0,
            "stage": "initialized",
            "build_attempts": 0,
            "max_build_attempts": self.max_build_attempts,
        }
        if _HAS_LANGGRAPH:
            graph = self._build_graph()
            return graph.invoke(initial)
        return self._run_sequential(initial)

    def _build_graph(self):
        assert StateGraph is not None
        graph = StateGraph(WorkflowState)
        graph.add_node("parse_paper", self._node_parse_paper)
        graph.add_node("parse_figures", self._node_parse_figures)
        graph.add_node("parse_methods", self._node_parse_methods)
        graph.add_node("parse_datasets", self._node_parse_datasets)
        graph.add_node("parse_software", self._node_parse_software)
        graph.add_node("build_workflow_graph", self._node_build_workflow_graph)
        graph.add_node("validate_method", self._node_validate_method)
        graph.add_node("build_pipeline", self._node_build_pipeline)

        graph.set_entry_point("parse_paper")
        graph.add_edge("parse_paper", "parse_figures")
        graph.add_edge("parse_figures", "parse_methods")
        graph.add_edge("parse_methods", "parse_datasets")
        graph.add_edge("parse_datasets", "parse_software")
        graph.add_edge("parse_software", "build_workflow_graph")
        if self.bioworkflow_mode == "off":
            graph.add_edge("build_workflow_graph", "build_pipeline")
        else:
            graph.add_edge("build_workflow_graph", "validate_method")
            graph.add_edge("validate_method", "build_pipeline")
        graph.add_conditional_edges(
            "build_pipeline",
            self._next_after_build_pipeline,
            {
                "build_pipeline": "build_pipeline",
                "end": END,
            },
        )
        return graph.compile()

    def _run_sequential(self, state: WorkflowState) -> WorkflowState:
        nodes = [
            self._node_parse_paper,
            self._node_parse_figures,
            self._node_parse_methods,
            self._node_parse_datasets,
            self._node_parse_software,
            self._node_build_workflow_graph,
        ]
        if self.bioworkflow_mode != "off":
            nodes.append(self._node_validate_method)
        for fn in nodes:
            state.update(fn(state))
        while True:
            state.update(self._node_build_pipeline(state))
            if self._next_after_build_pipeline(state) == "end":
                break
        return state

    def _require_state_keys(self, state: WorkflowState, keys: list[str], *, node: str) -> None:
        missing = [k for k in keys if k not in state]
        if missing:
            raise KeyError(f"{node} missing required state keys: {', '.join(missing)}")

    def _node_parse_paper(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["source", "source_type"], node="parse_paper")
        paper = self.paper_parser.parse(state["source"], source_type=state["source_type"])
        return {"paper": paper, "progress": 15, "stage": "parsed_paper"}

    def _node_parse_figures(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["paper"], node="parse_figures")
        if self.skip_figures:
            logger.warning("Skipping figure parsing because RESEARCHER_AI_SKIP_FIGURES is enabled.")
            return {
                "figures": [],
                "figure_parse_errors": ["figure_parsing_skipped_by_flag"],
                "progress": 35,
                "stage": "parsed_figures_skipped",
            }

        timeout_s = self.parse_figures_timeout_seconds
        try:
            if timeout_s > 0:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    fut = executor.submit(self.figure_parser.parse_all_figures, state["paper"])
                    try:
                        figures = fut.result(timeout=timeout_s)
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "parse_figures timed out after %.1fs; degrading to empty figures.",
                            timeout_s,
                        )
                        fut.cancel()
                        return {
                            "figures": [],
                            "figure_parse_errors": [
                                f"parse_figures_timeout:{timeout_s:.1f}s"
                            ],
                            "progress": 35,
                            "stage": "parsed_figures_timeout",
                        }
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                figures = self.figure_parser.parse_all_figures(state["paper"])
        except Exception as exc:
            logger.warning(
                "parse_figures failed (%s): %s; degrading to empty figures.",
                exc.__class__.__name__,
                exc,
            )
            return {
                "figures": [],
                "figure_parse_errors": [f"parse_figures_error:{exc.__class__.__name__}:{exc}"],
                "progress": 35,
                "stage": "parsed_figures_degraded",
            }
        return {"figures": figures, "progress": 35, "stage": "parsed_figures"}

    def _node_parse_methods(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["paper"], node="parse_methods")
        method = self.methods_parser.parse(
            state["paper"],
            figures=state.get("figures", []),
            computational_only=True,
        )
        return {"method": method, "progress": 55, "stage": "parsed_methods"}

    def _node_parse_datasets(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["paper", "method"], node="parse_datasets")
        paper = state["paper"]
        method = state["method"]
        figures = state.get("figures", [])
        section_text = "\n".join((getattr(sec, "text", "") or "") for sec in (paper.sections or []))
        combined_text = "\n".join(
            [
                paper.raw_text or "",
                section_text,
                method.data_availability or "",
                method.code_availability or "",
                "\n".join(getattr(fig, "caption", "") or "" for fig in figures),
            ]
        )
        accessions = _collect_accessions(combined_text)
        datasets: list[Dataset] = []
        dataset_errors: list[str] = []
        for acc in accessions[:25]:
            try:
                ds = _parse_dataset(acc)
                if ds is not None:
                    datasets.append(ds)
            except Exception as exc:  # pragma: no cover - best effort behavior
                dataset_errors.append(f"{acc}: {type(exc).__name__}: {exc}")
        return {
            "datasets": datasets,
            "dataset_parse_errors": dataset_errors,
            "progress": 70,
            "stage": "parsed_datasets",
        }

    def _node_parse_software(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["method"], node="parse_software")
        software = self.software_parser.parse_from_method(state["method"])
        return {"software": software, "progress": 80, "stage": "parsed_software"}

    def _node_build_workflow_graph(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["method"], node="build_workflow_graph")
        graph = build_workflow_graph(
            method=state["method"],
            datasets=state.get("datasets", []),
            software=state.get("software", []),
            targets=[getattr(fig, "figure_id", "") for fig in state.get("figures", []) if getattr(fig, "figure_id", "")],
        )
        issues = graph.validation_issues()
        return {
            "workflow_graph": graph,
            "workflow_graph_validation_issues": issues,
            "progress": 86,
            "stage": "built_workflow_graph",
        }

    def _node_validate_method(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["method"], node="validate_method")
        if self.bioworkflow_mode == "off":
            return {
                "progress": 90,
                "stage": "validation_skipped",
                "validation_blocked": False,
            }
        method = state["method"]
        report = self.validation_agent.validate(
            method=method,
            paper_rag=getattr(self.methods_parser, "paper_rag", None),
            protocol_rag=getattr(self.methods_parser, "protocol_rag", None),
        )
        warnings = list(method.parse_warnings)
        warnings.extend(report.warnings)
        updated_method = method.model_copy(
            update={
                "validation_report": report,
                "parse_warnings": warnings,
            }
        )
        validation_blocked = self.bioworkflow_mode == "on" and report.ungrounded_count > 0
        if validation_blocked:
            warnings.append(
                f"bioworkflow_blocked: ungrounded_fields={report.ungrounded_count} mode=on"
            )
            updated_method = updated_method.model_copy(update={"parse_warnings": warnings})
        return {
            "method": updated_method,
            "method_validation_report": report,
            "validation_blocked": validation_blocked,
            "progress": 90,
            "stage": "validated_method",
        }

    def _node_build_pipeline(self, state: WorkflowState) -> WorkflowState:
        self._require_state_keys(state, ["method"], node="build_pipeline")
        if state.get("validation_blocked", False):
            report = state.get("method_validation_report")
            summary = self._build_human_review_summary(report)
            return {
                "build_attempts": int(state.get("build_attempts", 0)),
                "human_review_required": True,
                "human_review_summary": summary,
                "progress": 100,
                "stage": "needs_human_review",
            }
        attempts = int(state.get("build_attempts", 0)) + 1
        pipeline = self.pipeline_builder.build(
            state["method"],
            state.get("datasets", []),
            state.get("software", []),
            state.get("figures", []),
        )
        passed = bool((pipeline.validation_report or {}).get("passed", True))
        if passed:
            return {
                "pipeline": pipeline,
                "build_attempts": attempts,
                "progress": 100,
                "stage": "completed",
            }
        return {
            "pipeline": pipeline,
            "build_attempts": attempts,
            "progress": 92,
            "stage": "builder_retry",
        }

    def _next_after_build_pipeline(self, state: WorkflowState) -> str:
        if state.get("validation_blocked", False):
            return "end"
        pipeline = state.get("pipeline")
        if pipeline is None:
            return "end"
        passed = bool((pipeline.validation_report or {}).get("passed", True))
        if passed:
            return "end"
        attempts = int(state.get("build_attempts", 0))
        max_attempts = int(state.get("max_build_attempts", self.max_build_attempts))
        if attempts < max_attempts:
            return "build_pipeline"
        return "end"

    def _build_human_review_summary(self, report: Optional[ValidationReport]) -> dict[str, Any]:
        """Summarize blocked validation findings for explicit manual triage."""
        if report is None:
            return {
                "reason": "validation_blocked",
                "ungrounded_count": 0,
                "ungrounded_fields": [],
                "recommended_action": "Review extraction output manually before pipeline build.",
            }
        ungrounded_fields = [
            verdict.field
            for verdict in report.verdicts
            if str(
                getattr(
                    getattr(verdict, "evidence_category", ""),
                    "value",
                    getattr(verdict, "evidence_category", ""),
                )
            )
            == "ungrounded"
        ]
        return {
            "reason": "validation_blocked",
            "ungrounded_count": int(report.ungrounded_count),
            "ungrounded_fields": ungrounded_fields,
            "recommended_action": (
                "Provide missing parameters manually or switch RESEARCHER_AI_BIOWORKFLOW_MODE=warn "
                "to continue with flagged defaults."
            ),
        }
