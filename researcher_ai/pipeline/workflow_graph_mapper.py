"""Map parsed method/data/software artifacts into WorkflowGraph IR."""

from __future__ import annotations

import re
from typing import Optional

from researcher_ai.models.dataset import Dataset
from researcher_ai.models.method import Assay, Method
from researcher_ai.models.software import Software
from researcher_ai.models.workflow_graph import (
    GraphEdge,
    GraphNode,
    GraphPort,
    NodeKind,
    PortDirection,
    PortType,
    WorkflowGraph,
)


def _sanitize_id(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", (text or "").lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def _infer_port_type(text: str) -> PortType:
    t = (text or "").lower()
    if "fastq" in t:
        return PortType.fastq
    if "bam" in t:
        return PortType.bam
    if "peak" in t:
        return PortType.peaks
    if "matrix" in t or "count" in t or "expression" in t:
        return PortType.matrix
    if "report" in t or "qc" in t:
        return PortType.report
    if "table" in t or "tsv" in t or "csv" in t:
        return PortType.table
    return PortType.generic


def _software_command_map(software: list[Software]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sw in software:
        if not sw.name:
            continue
        for cmd in sw.commands:
            if cmd.command_template:
                out[sw.name.lower()] = cmd.command_template
                break
    return out


def _assay_step_nodes(
    assay: Assay,
    command_map: dict[str, str],
) -> tuple[list[GraphNode], list[GraphEdge], Optional[str], Optional[str]]:
    """Map one assay into graph nodes/edges and return first/last node IDs."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    steps = sorted(assay.steps, key=lambda s: s.step_number)
    if not steps:
        return [], [], None, None

    assay_id = _sanitize_id(assay.name)
    prev_node_id: Optional[str] = None
    first_node_id: Optional[str] = None

    for idx, step in enumerate(steps, start=1):
        node_id = f"{assay_id}_step{idx}"
        if first_node_id is None:
            first_node_id = node_id

        sw_name = step.software or ""
        command_template = command_map.get(sw_name.lower()) if sw_name else None

        input_required = idx > 1
        # First-step inputs often represent assay-level external data sources;
        # keep them generic until higher-confidence typing is available.
        input_type = PortType.generic if idx == 1 else _infer_port_type(step.input_data)
        output_type = _infer_port_type(step.output_data)
        node = GraphNode(
            node_id=node_id,
            kind=NodeKind.analysis,
            label=f"{assay.name}: step {idx}",
            tool_name=step.software,
            tool_version=step.software_version,
            command_template=command_template,
            parameters=step.parameters,
            metadata={
                "assay_name": assay.name,
                "step_number": step.step_number,
                "input_data": step.input_data,
                "output_data": step.output_data,
            },
            ports=[
                GraphPort(
                    port_id="in",
                    name="input",
                    direction=PortDirection.input,
                    port_type=input_type,
                    required=input_required,
                ),
                GraphPort(
                    port_id="out",
                    name="output",
                    direction=PortDirection.output,
                    port_type=output_type,
                    required=False,
                ),
            ],
        )
        nodes.append(node)

        if prev_node_id is not None:
            edges.append(
                GraphEdge(
                    edge_id=f"{assay_id}_chain_{idx-1}_{idx}",
                    from_node_id=prev_node_id,
                    from_port_id="out",
                    to_node_id=node_id,
                    to_port_id="in",
                )
            )
        prev_node_id = node_id

    return nodes, edges, first_node_id, prev_node_id


def build_workflow_graph(
    *,
    method: Method,
    datasets: list[Dataset],
    software: list[Software],
    graph_id: Optional[str] = None,
    name: Optional[str] = None,
    targets: Optional[list[str]] = None,
) -> WorkflowGraph:
    """Build canonical WorkflowGraph IR from parser outputs."""
    command_map = _software_command_map(software)
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    first_step_by_assay: dict[str, str] = {}
    last_step_by_assay: dict[str, str] = {}

    # Dataset source nodes
    dataset_node_ids: list[str] = []
    for ds in datasets:
        ds_id = _sanitize_id(ds.accession)
        node_id = f"dataset_{ds_id}"
        dataset_node_ids.append(node_id)
        nodes.append(
            GraphNode(
                node_id=node_id,
                kind=NodeKind.source,
                label=ds.accession,
                metadata={
                    "source": str(ds.source.value) if hasattr(ds.source, "value") else str(ds.source),
                    "title": ds.title,
                },
                ports=[
                    GraphPort(
                        port_id="out_dataset",
                        name="dataset",
                        direction=PortDirection.output,
                        # Dataset accessions are generic upstream anchors; downstream
                        # parsers/compilers can specialize this based on assay context.
                        port_type=PortType.generic,
                        required=False,
                    )
                ],
            )
        )

    # Assay step nodes
    for assay in method.assay_graph.assays:
        assay_nodes, assay_edges, first_node, last_node = _assay_step_nodes(assay, command_map)
        nodes.extend(assay_nodes)
        edges.extend(assay_edges)
        if first_node:
            first_step_by_assay[assay.name] = first_node
        if last_node:
            last_step_by_assay[assay.name] = last_node

    # Assay-level dependencies
    dep_counter = 1
    for dep in method.assay_graph.dependencies:
        src = last_step_by_assay.get(dep.upstream_assay)
        dst = first_step_by_assay.get(dep.downstream_assay)
        if src is None or dst is None:
            continue
        edges.append(
            GraphEdge(
                edge_id=f"assay_dep_{dep_counter}",
                from_node_id=src,
                from_port_id="out",
                to_node_id=dst,
                to_port_id="in",
            )
        )
        dep_counter += 1

    # Best-effort dataset wiring: connect dataset sources to first assay nodes
    input_counter = 1
    first_assay_steps = list(first_step_by_assay.values())
    for ds_node_id in dataset_node_ids:
        for dst in first_assay_steps:
            edges.append(
                GraphEdge(
                    edge_id=f"dataset_input_{input_counter}",
                    from_node_id=ds_node_id,
                    from_port_id="out_dataset",
                    to_node_id=dst,
                    to_port_id="in",
                )
            )
            input_counter += 1

    resolved_graph_id = graph_id or _sanitize_id(method.paper_doi or "workflow_graph")
    resolved_name = name or resolved_graph_id
    return WorkflowGraph(
        graph_id=resolved_graph_id,
        name=resolved_name,
        nodes=nodes,
        edges=edges,
        targets=targets or [],
        provenance={"paper_doi": method.paper_doi or ""},
    )
