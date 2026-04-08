"""WorkflowGraph intermediate representation (IR) models.

These contracts define a backend-agnostic workflow graph used to decouple:
- extraction/parsing (`Paper`/`Method`/`Dataset`/`Software`),
- from execution/compilation (Snakemake/Nextflow/Slurm/Airflow/Prefect).

The model intentionally separates semantic graph state from UI concerns.
Node coordinates, viewport zoom, and layout metadata should live outside this
module in frontend-specific contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PortDirection(str, Enum):
    """Port direction in a dataflow graph."""

    input = "input"
    output = "output"


class PortMultiplicity(str, Enum):
    """How many edges can target or originate from a port."""

    one = "one"
    many = "many"


class PortType(str, Enum):
    """Semantic data category carried by a port."""

    dataset = "dataset"
    table = "table"
    bam = "bam"
    fastq = "fastq"
    peaks = "peaks"
    matrix = "matrix"
    report = "report"
    generic = "generic"


class NodeKind(str, Enum):
    """High-level role of a node in the workflow."""

    source = "source"
    transform = "transform"
    analysis = "analysis"
    qc = "qc"
    visualization = "visualization"
    sink = "sink"


class ExecutionBackend(str, Enum):
    """Execution target used during compilation."""

    snakemake = "snakemake"
    nextflow = "nextflow"
    slurm = "slurm"
    airflow = "airflow"
    prefect = "prefect"


class ValidationSeverity(str, Enum):
    """Severity level for graph diagnostics."""

    error = "error"
    warning = "warning"


class GraphValidationIssue(BaseModel):
    """Machine-readable validation issue emitted by WorkflowGraph checks."""

    severity: ValidationSeverity
    code: str
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    port_id: Optional[str] = None


class GraphPort(BaseModel):
    """Typed node port used for graph edge wiring."""

    port_id: str
    name: str
    direction: PortDirection
    port_type: PortType = PortType.generic
    required: bool = True
    multiplicity: PortMultiplicity = PortMultiplicity.one


class NodeResources(BaseModel):
    """Execution resource hints for a graph node."""

    threads: int = 1
    memory_gb: int = 4
    walltime: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None


class GraphNode(BaseModel):
    """Executable or logical graph node."""

    node_id: str
    kind: NodeKind
    label: str
    ports: list[GraphPort] = Field(default_factory=list)
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None
    command_template: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    resources: NodeResources = Field(default_factory=NodeResources)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Directed edge from an output port to an input port."""

    edge_id: str
    from_node_id: str
    from_port_id: str
    to_node_id: str
    to_port_id: str
    condition: Optional[str] = None


class WorkflowGraph(BaseModel):
    """Canonical backend-agnostic workflow graph."""

    graph_id: str
    name: str
    description: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def validate_unique_ids(self) -> list[GraphValidationIssue]:
        """Validate uniqueness of node, port, and edge IDs."""
        issues: list[GraphValidationIssue] = []

        node_ids: set[str] = set()
        for node in self.nodes:
            if node.node_id in node_ids:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="duplicate_node_id",
                        message=f"Duplicate node_id: {node.node_id}",
                        node_id=node.node_id,
                    )
                )
            node_ids.add(node.node_id)

            port_ids: set[str] = set()
            for port in node.ports:
                if port.port_id in port_ids:
                    issues.append(
                        GraphValidationIssue(
                            severity=ValidationSeverity.error,
                            code="duplicate_port_id_within_node",
                            message=f"Duplicate port_id '{port.port_id}' within node '{node.node_id}'",
                            node_id=node.node_id,
                            port_id=port.port_id,
                        )
                    )
                port_ids.add(port.port_id)

        edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.edge_id in edge_ids:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="duplicate_edge_id",
                        message=f"Duplicate edge_id: {edge.edge_id}",
                        edge_id=edge.edge_id,
                    )
                )
            edge_ids.add(edge.edge_id)
        return issues

    def validate_references_exist(self) -> list[GraphValidationIssue]:
        """Validate that all edge endpoints refer to existing nodes and ports."""
        issues: list[GraphValidationIssue] = []
        node_map = {n.node_id: n for n in self.nodes}

        for edge in self.edges:
            src_node = node_map.get(edge.from_node_id)
            dst_node = node_map.get(edge.to_node_id)

            if src_node is None:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="missing_from_node",
                        message=f"Edge '{edge.edge_id}' references missing from_node_id '{edge.from_node_id}'",
                        edge_id=edge.edge_id,
                        node_id=edge.from_node_id,
                    )
                )
                continue
            if dst_node is None:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="missing_to_node",
                        message=f"Edge '{edge.edge_id}' references missing to_node_id '{edge.to_node_id}'",
                        edge_id=edge.edge_id,
                        node_id=edge.to_node_id,
                    )
                )
                continue

            src_ports = {p.port_id: p for p in src_node.ports}
            dst_ports = {p.port_id: p for p in dst_node.ports}

            if edge.from_port_id not in src_ports:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="missing_from_port",
                        message=(
                            f"Edge '{edge.edge_id}' references missing from_port_id "
                            f"'{edge.from_port_id}' on node '{edge.from_node_id}'"
                        ),
                        edge_id=edge.edge_id,
                        node_id=edge.from_node_id,
                        port_id=edge.from_port_id,
                    )
                )
            if edge.to_port_id not in dst_ports:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="missing_to_port",
                        message=(
                            f"Edge '{edge.edge_id}' references missing to_port_id "
                            f"'{edge.to_port_id}' on node '{edge.to_node_id}'"
                        ),
                        edge_id=edge.edge_id,
                        node_id=edge.to_node_id,
                        port_id=edge.to_port_id,
                    )
                )
        return issues

    def validate_port_direction_compatibility(self) -> list[GraphValidationIssue]:
        """Validate edge wiring direction: output -> input only."""
        issues: list[GraphValidationIssue] = []
        node_map = {n.node_id: n for n in self.nodes}
        for edge in self.edges:
            src = node_map.get(edge.from_node_id)
            dst = node_map.get(edge.to_node_id)
            if src is None or dst is None:
                continue

            src_ports = {p.port_id: p for p in src.ports}
            dst_ports = {p.port_id: p for p in dst.ports}
            src_port = src_ports.get(edge.from_port_id)
            dst_port = dst_ports.get(edge.to_port_id)
            if src_port is None or dst_port is None:
                continue

            if src_port.direction != PortDirection.output or dst_port.direction != PortDirection.input:
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="incompatible_port_direction",
                        message=(
                            f"Edge '{edge.edge_id}' must connect output->input, got "
                            f"{src_port.direction.value}->{dst_port.direction.value}"
                        ),
                        edge_id=edge.edge_id,
                    )
                )
        return issues

    def validate_type_compatibility(self) -> list[GraphValidationIssue]:
        """Validate edge port-type compatibility."""
        issues: list[GraphValidationIssue] = []
        node_map = {n.node_id: n for n in self.nodes}
        for edge in self.edges:
            src = node_map.get(edge.from_node_id)
            dst = node_map.get(edge.to_node_id)
            if src is None or dst is None:
                continue

            src_ports = {p.port_id: p for p in src.ports}
            dst_ports = {p.port_id: p for p in dst.ports}
            src_port = src_ports.get(edge.from_port_id)
            dst_port = dst_ports.get(edge.to_port_id)
            if src_port is None or dst_port is None:
                continue

            if (
                src_port.port_type != dst_port.port_type
                and src_port.port_type != PortType.generic
                and dst_port.port_type != PortType.generic
            ):
                issues.append(
                    GraphValidationIssue(
                        severity=ValidationSeverity.error,
                        code="incompatible_port_type",
                        message=(
                            f"Edge '{edge.edge_id}' connects incompatible port types "
                            f"{src_port.port_type.value}->{dst_port.port_type.value}"
                        ),
                        edge_id=edge.edge_id,
                    )
                )
        return issues

    def validate_acyclic(self) -> list[GraphValidationIssue]:
        """Validate that the node-level dependency graph is acyclic."""
        issues: list[GraphValidationIssue] = []
        node_ids = {n.node_id for n in self.nodes}
        adjacency: dict[str, set[str]] = {nid: set() for nid in node_ids}
        indegree: dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in self.edges:
            if edge.from_node_id not in node_ids or edge.to_node_id not in node_ids:
                continue
            if edge.to_node_id in adjacency[edge.from_node_id]:
                continue
            adjacency[edge.from_node_id].add(edge.to_node_id)
            indegree[edge.to_node_id] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            curr = queue.pop()
            visited += 1
            for nxt in adjacency[curr]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(node_ids):
            issues.append(
                GraphValidationIssue(
                    severity=ValidationSeverity.error,
                    code="cycle_detected",
                    message="WorkflowGraph contains at least one directed cycle.",
                )
            )
        return issues

    def validate_required_inputs_satisfied(self) -> list[GraphValidationIssue]:
        """Validate that required input ports have incoming edges."""
        issues: list[GraphValidationIssue] = []
        incoming: set[tuple[str, str]] = {
            (e.to_node_id, e.to_port_id)
            for e in self.edges
        }

        for node in self.nodes:
            for port in node.ports:
                if port.direction != PortDirection.input or not port.required:
                    continue
                if (node.node_id, port.port_id) not in incoming:
                    issues.append(
                        GraphValidationIssue(
                            severity=ValidationSeverity.error,
                            code="missing_required_input",
                            message=(
                                f"Required input port '{port.port_id}' on node "
                                f"'{node.node_id}' has no incoming edge."
                            ),
                            node_id=node.node_id,
                            port_id=port.port_id,
                        )
                    )
        return issues

    def validation_issues(self) -> list[GraphValidationIssue]:
        """Run all validation checks and return combined diagnostics."""
        checks = (
            self.validate_unique_ids,
            self.validate_references_exist,
            self.validate_port_direction_compatibility,
            self.validate_type_compatibility,
            self.validate_acyclic,
            self.validate_required_inputs_satisfied,
        )
        out: list[GraphValidationIssue] = []
        for fn in checks:
            out.extend(fn())
        return out

    def is_valid(self) -> bool:
        """Return True when no error-severity validation issues are present."""
        return not any(i.severity == ValidationSeverity.error for i in self.validation_issues())
