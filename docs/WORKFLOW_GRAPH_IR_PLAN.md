# WorkflowGraph IR Plan

**Status:** Draft  
**Owner:** researcher-ai core maintainers  
**Last Updated:** 2026-04-07

## Objectives

- Establish a canonical, backend-agnostic intermediate representation (IR) for computational workflows.
- Separate extraction concerns (`Paper`/`Method` parsing) from execution concerns (Snakemake, Nextflow, Slurm, Airflow, Prefect).
- Define validation rules that make graph-level errors explicit before code generation.
- Create a clear implementation checklist for v0.2-v0.3 hardening.

## Non-Goals

- Replacing existing parser contracts (`Method`, `Dataset`, `Software`) in v0.2.
- Shipping a visual editor in the same phase as IR contract introduction.
- Enabling all execution backends in the first compiler release.

## Canonical IR Schema Contract

### `WorkflowGraph`

- `graph_id: str`
- `name: str`
- `description: str = ""`
- `nodes: list[GraphNode]`
- `edges: list[GraphEdge]`
- `targets: list[str] = []`
- `provenance: dict[str, str] = {}`

### `GraphNode`

- `node_id: str`
- `kind: str` (source, transform, analysis, qc, visualization, sink)
- `label: str`
- `ports: list[GraphPort]`
- `tool_name: str | None = None`
- `tool_version: str | None = None`
- `command_template: str | None = None`
- `parameters: dict[str, str] = {}`
- `resources: dict[str, str | int] = {}`
- `metadata: dict[str, str] = {}`

### `GraphPort`

- `port_id: str`
- `name: str`
- `direction: str` (input, output)
- `port_type: str` (fastq, bam, peaks, matrix, report, generic, ...)
- `required: bool = True`
- `multiplicity: str = "one"` (one, many)

### `GraphEdge`

- `edge_id: str`
- `from_node_id: str`
- `from_port_id: str`
- `to_node_id: str`
- `to_port_id: str`
- `condition: str | None = None`

### `GraphValidationIssue`

- `severity: str` (error, warning)
- `code: str`
- `message: str`
- `node_id: str | None = None`
- `edge_id: str | None = None`
- `port_id: str | None = None`

## Parser-to-IR Field Mapping Matrix

| Source model | Key fields | IR target fields | Notes |
|---|---|---|---|
| `Method.assay_graph.assays[]` | assay name, steps, method_category | `GraphNode.kind`, `GraphNode.label`, `GraphNode.command_template`, `GraphNode.parameters` | Computational assays map to executable nodes; experimental assays can be retained as provenance-only nodes in mixed pipelines. |
| `Method.assay_graph.dependencies[]` | upstream/downstream assays | `GraphEdge.from_node_id`, `GraphEdge.to_node_id` | Dependency edges become graph dataflow edges with typed ports. |
| `Dataset` | accession, source, sample metadata | source `GraphNode` + dataset output `GraphPort` | Accessions become graph-level input anchors. |
| `Software` | name, version, commands, env | `GraphNode.tool_name`, `tool_version`, `command_template`, `resources` | Tool metadata enriches command templates and runtime hints. |
| `Figure` (optional for targeting) | figure_id | `WorkflowGraph.targets` | Used for prioritizing terminal outputs and validation goals. |

## Gap Checklist (Current Extraction vs Required IR)

- [ ] Stable `node_id` strategy for assay/step-level nodes.
- [ ] Typed input/output port extraction from `AnalysisStep.input_data` and `output_data`.
- [ ] Explicit edge-port binding (current dependencies are assay-level only).
- [ ] Resource hints per node (threads, memory, walltime, partition/account where available).
- [ ] Provenance payload linking source statements to generated nodes/edges.
- [ ] Structured diagnostics for missing command templates and incompatible types.

## Validation Rules

`WorkflowGraph` validation must enforce:

1. Unique IDs for graphs, nodes, ports, and edges.
2. Edge endpoint existence (`from_*` and `to_*` references must resolve).
3. Port direction compatibility (output -> input only).
4. Port type compatibility (or explicit `generic` fallback with warning).
5. Acyclicity for execution graph components.
6. Required inputs satisfied for executable nodes.

Validation output contract: list of `GraphValidationIssue` items; any `error` blocks compilation.

## Compiler Pipeline

1. `Method` + `Dataset` + `Software` -> inferred `WorkflowGraph`
2. Validate `WorkflowGraph` and emit `GraphValidationIssue` diagnostics
3. `WorkflowGraph` -> `PipelineConfig`
4. `PipelineConfig` -> backend adapter output
   - v0.3: Snakemake, Nextflow
   - v0.4a+: Slurm/TSCC adapter
   - later: Airflow, Prefect

## Phased Delivery

### v0.2 (Hardening)

- Finalize IR contract and validation API surface.
- Add parser diagnostics needed to populate IR safely.
- Preserve backward compatibility: existing CLI output remains unchanged except additive metadata.

### v0.3 (Headless Compiler)

- Implement `WorkflowGraph` model module and compiler bridge.
- Add deterministic graph validation and compile diagnostics.
- Compile inferred graph to existing Snakemake/Nextflow generators.

### v0.4a (Service Layer)

- Add API endpoints for parse/graph/compile/execute using semantic graph payloads.
- Support JSON-first graph editing (no visual editor requirement).
- Add initial Slurm/TSCC execution adapter path.

### v0.4b (Visual Editor)

- Introduce React Flow editor backed by separate UI layout schema.
- Keep semantic IR payload isolated from visual state payload.
- Add server-backed validation feedback for graph edits.

### v1.0 (Evidence + Ensemble)

- Add evidence mapping for method -> code -> figure traceability.
- Add extraction-only ensemble confidence loops and tie-break evaluation.

## Acceptance Criteria and Metrics

- >= 95% graph validation pass rate on curated parser fixtures.
- >= 95% successful compilation of validated graphs to `PipelineConfig`.
- >= 90% Snakemake/Nextflow generation success on validated `PipelineConfig`.
- 100% of blocked compilations include machine-readable diagnostics.
- Zero breaking changes to existing CLI output schema in v0.x (additive only).

## Risks and Mitigations

- **Risk:** Overfitting IR to one backend.  
  **Mitigation:** Keep backend-specific fields in adapter metadata, not core node/edge schema.

- **Risk:** Ambiguous parser text creates unstable ports and IDs.  
  **Mitigation:** deterministic ID normalization and warning-first fallback rules.

- **Risk:** UI state leaking into semantic graph.  
  **Mitigation:** strict contract split between semantic IR and UI layout schema.

- **Risk:** Migration churn for current pipeline builder internals.  
  **Mitigation:** adapter layer that preserves `PipelineConfig` and existing generators during transition.

## Appendix: Migration Compatibility Notes

- `PipelineBuilder` remains callable during transition, with an internal path that can consume inferred `WorkflowGraph`.
- Existing orchestrator flow remains default in v0.2/v0.3; IR path is additive.
- New IR validation failures must surface as non-silent diagnostics and must not silently drop steps.
- Documentation authority for IR planning is this file; operational architecture remains documented in `docs/ARCHITECTURE.md`.
