from __future__ import annotations

from researcher_ai.models.dataset import DataSource, Dataset
from researcher_ai.models.method import AnalysisStep, Assay, AssayDependency, AssayGraph, Method
from researcher_ai.models.software import Command, Software
from researcher_ai.models.workflow_graph import NodeKind
from researcher_ai.pipeline.workflow_graph_mapper import build_workflow_graph


def _method() -> Method:
    rna = Assay(
        name="RNA-seq",
        description="RNA workflow",
        data_type="sequencing",
        steps=[
            AnalysisStep(
                step_number=1,
                description="Align reads",
                input_data="FASTQ reads",
                output_data="BAM file",
                software="STAR",
                parameters={"threads": "8"},
            ),
            AnalysisStep(
                step_number=2,
                description="Call peaks",
                input_data="BAM file",
                output_data="peaks bed",
                software="CLIPper",
            ),
        ],
    )
    de = Assay(
        name="DESeq2 analysis",
        description="DE workflow",
        data_type="sequencing",
        steps=[
            AnalysisStep(
                step_number=1,
                description="Build count matrix",
                input_data="BAM file",
                output_data="count matrix",
                software="DESeq2",
            )
        ],
    )
    deps = [AssayDependency(upstream_assay="RNA-seq", downstream_assay="DESeq2 analysis", dependency_type="data")]
    return Method(paper_doi="10.1000/test", assay_graph=AssayGraph(assays=[rna, de], dependencies=deps))


def _datasets() -> list[Dataset]:
    return [
        Dataset(
            accession="GSE12345",
            source=DataSource.GEO,
            title="RNA dataset",
        )
    ]


def _software() -> list[Software]:
    return [
        Software(
            name="STAR",
            commands=[Command(command_template="STAR --runThreadN 8", description="align")],
        ),
        Software(
            name="DESeq2",
            commands=[Command(command_template="Rscript deseq2.R", description="de")],
        ),
    ]


def test_build_workflow_graph_emits_nodes_and_edges():
    graph = build_workflow_graph(method=_method(), datasets=_datasets(), software=_software())
    assert graph.nodes
    assert graph.edges
    assert any(n.kind == NodeKind.source for n in graph.nodes)
    assert any("rna_seq_step1" == n.node_id for n in graph.nodes)


def test_build_workflow_graph_wires_assay_dependency():
    graph = build_workflow_graph(method=_method(), datasets=_datasets(), software=_software())
    # RNA-seq last step should feed DESeq2 first step through an assay dependency edge.
    assert any(
        e.from_node_id == "rna_seq_step2" and e.to_node_id == "deseq2_analysis_step1"
        for e in graph.edges
    )


def test_build_workflow_graph_uses_software_command_templates():
    graph = build_workflow_graph(method=_method(), datasets=_datasets(), software=_software())
    star = next(n for n in graph.nodes if n.node_id == "rna_seq_step1")
    assert star.command_template == "STAR --runThreadN 8"


def test_build_workflow_graph_validation_runs():
    graph = build_workflow_graph(method=_method(), datasets=_datasets(), software=_software())
    issues = graph.validation_issues()
    # Mapper should produce a structurally coherent graph without hard errors.
    assert not [i for i in issues if i.severity.value == "error"]
