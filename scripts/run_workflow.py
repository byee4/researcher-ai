#!/usr/bin/env python3
"""Run researcher-ai workflow through the Phase 4 state-graph orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from researcher_ai.models.paper import PaperSource
from researcher_ai.pipeline.orchestrator import WorkflowOrchestrator


def _emit(progress: int, stage: str) -> None:
    print(f"PROGRESS|{progress}|{stage}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="PMID or PDF path")
    parser.add_argument("--source-type", choices=["pmid", "pdf"], required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    source = args.source
    source_type = PaperSource.PMID if args.source_type == "pmid" else PaperSource.PDF
    output_path = Path(args.output)

    _emit(5, "Initializing state-graph orchestrator")
    orchestrator = WorkflowOrchestrator()
    state = orchestrator.run(source, source_type)
    _emit(int(state.get("progress", 98)), str(state.get("stage", "serializing_output")))

    paper = state["paper"]
    figures = state.get("figures", [])
    method = state["method"]
    datasets = state.get("datasets", [])
    software = state.get("software", [])
    workflow_graph = state.get("workflow_graph")
    workflow_graph_issues = state.get("workflow_graph_validation_issues", [])
    pipeline = state["pipeline"]

    output = {
        "paper": paper.model_dump(mode="json"),
        "figures": [f.model_dump(mode="json") for f in figures],
        "method": method.model_dump(mode="json"),
        "datasets": [d.model_dump(mode="json") for d in datasets],
        "software": [s.model_dump(mode="json") for s in software],
        "workflow_graph": workflow_graph.model_dump(mode="json") if workflow_graph is not None else None,
        "workflow_graph_validation_issues": [
            issue.model_dump(mode="json") for issue in workflow_graph_issues
        ],
        "pipeline": pipeline.model_dump(mode="json"),
        "dataset_parse_errors": state.get("dataset_parse_errors", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    _emit(100, "Completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
