"""Unit tests for Phase 7: Pipeline Builder and code generators.

Testing strategy:
- SnakemakeGenerator: rule generation for a simple two-step pipeline (no LLM, no network).
- NextflowGenerator: nf-core config generation for RNA-seq; custom DSL2 for multi-step.
- JupyterGenerator: notebook generation with mock Figure objects; fallback when nbformat absent.
- PipelineBuilder.build(): full build with mock Method, Dataset, Software, Figure objects.
- PipelineBuilder._topo_sort_assays(): correct ordering with and without dependencies.
- PipelineBuilder._generate_conda_env(): bioconda + pip + fallback entries.
"""

from __future__ import annotations

import json
import textwrap
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from researcher_ai.models.dataset import Dataset, DataSource
from researcher_ai.models.figure import (
    Axis,
    AxisScale,
    ColorMapping,
    ColormapType,
    Figure,
    PanelLayout,
    PlotCategory,
    PlotLayer,
    PlotType,
    SubFigure,
)
from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayDependency,
    AssayGraph,
    Method,
    MethodCategory,
)
from researcher_ai.models.pipeline import (
    Pipeline,
    PipelineBackend,
    PipelineConfig,
    PipelineStep,
)
from researcher_ai.models.software import Command, Environment, LicenseType, Software
from researcher_ai.pipeline.builder import PipelineBuilder, _sanitize_id
from researcher_ai.pipeline.jupyter_gen import JupyterGenerator
from researcher_ai.pipeline.nextflow_gen import NextflowGenerator
from researcher_ai.pipeline.snakemake_gen import SnakemakeGenerator


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_analysis_step(n: int, sw: str = "star", **kwargs) -> AnalysisStep:
    return AnalysisStep(
        step_number=n,
        description=f"Step {n}: {sw} processing",
        input_data=f"step{n}_input.fastq.gz",
        output_data=f"step{n}_aligned.bam",
        software=sw,
        software_version="2.7.3a",
        parameters={"threads": "8", "genome": "hg38"},
        **kwargs,
    )


def _make_assay(
    name: str = "RNA-seq",
    n_steps: int = 2,
    method_category: MethodCategory = MethodCategory.computational,
) -> Assay:
    return Assay(
        name=name,
        description=f"Standard {name} assay",
        data_type="sequencing",
        method_category=method_category,
        raw_data_source="GEO: GSE99999",
        steps=[_make_analysis_step(i + 1) for i in range(n_steps)],
        figures_produced=["Figure 1", "Figure 2"],
    )


def _make_method(assay_names: list[str] | None = None) -> Method:
    names = assay_names or ["RNA-seq"]
    assays = [_make_assay(n) for n in names]
    return Method(
        paper_doi="10.1000/test",
        assay_graph=AssayGraph(assays=assays),
    )


def _make_dataset(accession: str = "GSE99999") -> Dataset:
    return Dataset(
        accession=accession,
        source=DataSource.GEO,
        title="Test dataset",
        organism="Homo sapiens",
        experiment_type="RNA-seq",
    )


def _make_software(name: str = "STAR", bioconda: str = "star") -> Software:
    return Software(
        name=name,
        version="2.7.3a",
        bioconda_package=bioconda,
        license_type=LicenseType.OPEN_SOURCE,
        description=f"{name} aligner",
        commands=[
            Command(
                command_template=f"{name.lower()} --runMode alignReads --genomeDir {{genome_dir}} --readFilesIn {{input}} --outSAMtype BAM",
                description="Align reads",
                required_inputs=["fastq"],
                outputs=["bam"],
            )
        ],
    )


def _make_subfigure(label: str = "a", plot_type: PlotType = PlotType.VOLCANO) -> SubFigure:
    return SubFigure(
        label=label,
        description=f"Volcano plot for subfigure {label}",
        plot_category=PlotCategory.GENOMIC,
        plot_type=plot_type,
        x_axis=Axis(label="log2FC", scale=AxisScale.LINEAR),
        y_axis=Axis(label="-log10(p)", scale=AxisScale.LOG10, is_inverted=True),
        color_mapping=ColorMapping(
            variable="significance", colormap_type=ColormapType.BINARY
        ),
    )


def _make_figure(fig_id: str = "Figure 1") -> Figure:
    return Figure(
        figure_id=fig_id,
        title=f"Test figure {fig_id}",
        caption=f"This is the caption for {fig_id}.",
        purpose="Show differential expression results.",
        subfigures=[_make_subfigure("a"), _make_subfigure("b", PlotType.HEATMAP)],
        layout=PanelLayout(n_rows=1, n_cols=2),
    )


def _make_config(n_steps: int = 2, backend: PipelineBackend = PipelineBackend.SNAKEMAKE) -> PipelineConfig:
    steps = [
        PipelineStep(
            step_id=f"align_step{i + 1}",
            name=f"Alignment step {i + 1}",
            description=f"Align reads step {i + 1}",
            software="STAR",
            command="star --input {input} --output {output}",
            inputs=[f"sample_R1.fastq.gz"],
            outputs=[f"aligned_step{i + 1}.bam"],
            threads=8,
            memory_gb=32,
            depends_on=[f"align_step{i}"] if i > 0 else [],
        )
        for i in range(n_steps)
    ]
    return PipelineConfig(
        name="test_pipeline",
        description="A two-step test pipeline.",
        backend=backend,
        steps=steps,
        datasets=["GSE99999"],
        figure_targets=["Figure 1"],
    )


# ---------------------------------------------------------------------------
# Tests: _sanitize_id helper
# ---------------------------------------------------------------------------


def test_sanitize_id_basic():
    assert _sanitize_id("RNA-seq") == "rna_seq"


def test_sanitize_id_spaces():
    assert _sanitize_id("Hi-C analysis") == "hi_c_analysis"


def test_sanitize_id_empty():
    assert _sanitize_id("") == "step"


def test_sanitize_id_no_special():
    assert _sanitize_id("rnaseq") == "rnaseq"


# ---------------------------------------------------------------------------
# Tests: SnakemakeGenerator
# ---------------------------------------------------------------------------


class TestSnakemakeGenerator:
    def setup_method(self):
        self.gen = SnakemakeGenerator()

    def test_generate_returns_string(self):
        config = _make_config(2)
        result = self.gen.generate(config)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_rule_all(self):
        config = _make_config(2)
        result = self.gen.generate(config)
        assert "rule all:" in result

    def test_contains_rules_for_each_step(self):
        config = _make_config(2)
        result = self.gen.generate(config)
        assert "rule align_step1:" in result
        assert "rule align_step2:" in result

    def test_threads_and_resources_present(self):
        config = _make_config(1)
        result = self.gen.generate(config)
        assert "threads: 8" in result
        assert "mem_mb=" in result

    def test_configfile_block_present(self):
        config = _make_config(1)
        result = self.gen.generate(config)
        assert 'configfile:' in result

    def test_rule_with_conda_env(self):
        config = _make_config(1)
        config.steps[0].conda_env = "bioconda::star"
        result = self.gen.generate(config)
        assert 'conda:' in result

    def test_rule_with_container(self):
        config = _make_config(1)
        config.steps[0].container = "biocontainers/star:2.7.3a"
        result = self.gen.generate(config)
        assert 'singularity:' in result

    def test_single_step_pipeline(self):
        config = _make_config(1)
        result = self.gen.generate(config)
        assert "rule align_step1:" in result
        assert "rule all:" in result

    def test_dataset_in_configfile(self):
        config = _make_config(1)
        config.datasets = ["GSE12345", "SRP99999"]
        result = self.gen.generate(config)
        assert "GSE12345" in result

    def test_log_directive_present(self):
        config = _make_config(1)
        result = self.gen.generate(config)
        assert "log:" in result

    def test_execution_order_respected(self):
        """Step 2 depends on step 1: rule align_step1 should appear before rule align_step2."""
        config = _make_config(2)
        result = self.gen.generate(config)
        pos1 = result.index("rule align_step1:")
        pos2 = result.index("rule align_step2:")
        assert pos1 < pos2


# ---------------------------------------------------------------------------
# Tests: NextflowGenerator — nf-core mode
# ---------------------------------------------------------------------------


class TestNextflowGeneratorNfCore:
    def setup_method(self):
        self.gen = NextflowGenerator()

    def _rnaseq_config(self) -> PipelineConfig:
        config = _make_config(2, PipelineBackend.NEXTFLOW)
        config.nf_core_pipeline = "rnaseq"
        config.nf_core_version = "3.14.0"
        return config

    def test_nfcore_generates_string(self):
        result = self.gen.generate(self._rnaseq_config())
        assert isinstance(result, str) and len(result) > 0

    def test_nfcore_contains_params_key(self):
        result = self.gen.generate(self._rnaseq_config())
        assert "input:" in result or "input: " in result

    def test_nfcore_contains_pipeline_name(self):
        result = self.gen.generate(self._rnaseq_config())
        assert "rnaseq" in result

    def test_nfcore_samplesheet_csv_in_comment(self):
        result = self.gen.generate(self._rnaseq_config())
        assert "sample,fastq_1" in result

    def test_nfcore_dataset_accession_in_samplesheet(self):
        config = self._rnaseq_config()
        config.datasets = ["GSE12345"]
        result = self.gen.generate(config)
        assert "GSE12345" in result

    def test_nfcore_version_in_output(self):
        result = self.gen.generate(self._rnaseq_config())
        assert "3.14.0" in result

    def test_atacseq_samplesheet_has_correct_headers(self):
        gen = NextflowGenerator()
        config = _make_config(1, PipelineBackend.NEXTFLOW)
        config.nf_core_pipeline = "atacseq"
        config.datasets = ["GSE77777"]
        sheet = gen._generate_samplesheet(config)
        assert sheet.startswith("sample,fastq_1,fastq_2")

    def test_sarek_samplesheet_has_patient_column(self):
        gen = NextflowGenerator()
        config = _make_config(1, PipelineBackend.NEXTFLOW)
        config.nf_core_pipeline = "sarek"
        config.datasets = ["SRP1234"]
        sheet = gen._generate_samplesheet(config)
        assert sheet.startswith("patient,")


# ---------------------------------------------------------------------------
# Tests: NextflowGenerator — custom DSL2 mode
# ---------------------------------------------------------------------------


class TestNextflowGeneratorCustom:
    def setup_method(self):
        self.gen = NextflowGenerator()
        self.config = _make_config(2, PipelineBackend.NEXTFLOW)

    def test_custom_mode_returns_string(self):
        result = self.gen.generate(self.config)
        assert isinstance(result, str) and len(result) > 0

    def test_custom_mode_has_dsl2(self):
        result = self.gen.generate(self.config)
        assert "dsl = 2" in result.lower() or "nextflow.enable.dsl" in result

    def test_custom_mode_process_blocks(self):
        result = self.gen.generate(self.config)
        assert "process ALIGN_STEP1" in result
        assert "process ALIGN_STEP2" in result

    def test_custom_mode_workflow_block(self):
        result = self.gen.generate(self.config)
        assert "workflow {" in result

    def test_custom_mode_cpus_declared(self):
        result = self.gen.generate(self.config)
        assert "cpus" in result

    def test_custom_mode_params_block(self):
        result = self.gen.generate(self.config)
        assert "params {" in result

    def test_custom_mode_container_directive(self):
        self.config.steps[0].container = "biocontainers/star:2.7.3a"
        result = self.gen.generate(self.config)
        assert "container" in result


# ---------------------------------------------------------------------------
# Tests: JupyterGenerator
# ---------------------------------------------------------------------------


class TestJupyterGenerator:
    def setup_method(self):
        self.gen = JupyterGenerator()
        self.config = _make_config(2)

    def test_generate_returns_json_string(self):
        figures = [_make_figure()]
        result = self.gen.generate(self.config, figures)
        assert isinstance(result, str)
        nb = json.loads(result)
        assert nb["nbformat"] == 4

    def test_notebook_has_cells(self):
        figures = [_make_figure()]
        result = self.gen.generate(self.config, figures)
        nb = json.loads(result)
        assert len(nb["cells"]) > 0

    def _src(self, cell: dict) -> str:
        """Return cell source as a single string regardless of nbformat storage format."""
        src = cell["source"]
        return "".join(src) if isinstance(src, list) else src

    def test_setup_cell_is_first(self):
        figures = [_make_figure()]
        result = self.gen.generate(self.config, figures)
        nb = json.loads(result)
        first_cell = nb["cells"][0]
        assert first_cell["cell_type"] == "code"
        assert "import" in self._src(first_cell)

    def test_figure_markdown_cell_present(self):
        fig = _make_figure("Figure 1")
        result = self.gen.generate(self.config, [fig])
        nb = json.loads(result)
        md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
        assert any("Figure 1" in self._src(c) for c in md_cells)

    def test_subfigure_cells_generated(self):
        fig = _make_figure()
        result = self.gen.generate(self.config, [fig])
        nb = json.loads(result)
        # Should have: setup + fig header + 2 subfigs × 3 cells each = 1 + 1 + 6 = 8+
        assert len(nb["cells"]) >= 5

    def test_volcano_plot_code_generated(self):
        fig = _make_figure()
        result = self.gen.generate(self.config, [fig])
        nb = json.loads(result)
        code_sources = " ".join(
            self._src(c) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "scatter" in code_sources.lower() or "volcano" in code_sources.lower()

    def test_heatmap_plot_code_generated(self):
        fig = _make_figure()
        result = self.gen.generate(self.config, [fig])
        nb = json.loads(result)
        code_sources = " ".join(
            self._src(c) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "heatmap" in code_sources.lower()

    def test_empty_figures_list(self):
        result = self.gen.generate(self.config, [])
        nb = json.loads(result)
        assert len(nb["cells"]) >= 1  # at least the setup cell

    def test_pipeline_name_in_setup_cell(self):
        result = self.gen.generate(self.config, [])
        nb = json.loads(result)
        assert "test_pipeline" in self._src(nb["cells"][0])

    def test_multiple_figures(self):
        figs = [_make_figure("Figure 1"), _make_figure("Figure 2")]
        result = self.gen.generate(self.config, figs)
        nb = json.loads(result)
        sources = " ".join(self._src(c) for c in nb["cells"])
        assert "Figure 1" in sources
        assert "Figure 2" in sources


# ---------------------------------------------------------------------------
# Tests: PipelineBuilder — full build
# ---------------------------------------------------------------------------


class TestPipelineBuilder:
    def setup_method(self):
        self.builder = PipelineBuilder(llm_model="test-model")

    def _build_simple(
        self,
        backend: PipelineBackend = PipelineBackend.SNAKEMAKE,
    ) -> Pipeline:
        method = _make_method(["RNA-seq"])
        datasets = [_make_dataset("GSE99999")]
        software = [_make_software("STAR", "star")]
        figures = [_make_figure()]
        return self.builder.build(method, datasets, software, figures, backend)

    # --- Pipeline object structure ---

    def test_build_returns_pipeline(self):
        pipeline = self._build_simple()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_config(self):
        pipeline = self._build_simple()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)

    def test_pipeline_config_has_steps(self):
        pipeline = self._build_simple()
        assert len(pipeline.config.steps) > 0

    def test_pipeline_snakefile_generated(self):
        pipeline = self._build_simple(PipelineBackend.SNAKEMAKE)
        assert pipeline.snakefile_content is not None
        assert "rule all:" in pipeline.snakefile_content

    def test_pipeline_nextflow_generated_when_requested(self):
        pipeline = self._build_simple(PipelineBackend.NEXTFLOW)
        assert pipeline.nextflow_content is not None

    def test_pipeline_jupyter_always_generated(self):
        pipeline = self._build_simple()
        assert pipeline.jupyter_content is not None
        nb = json.loads(pipeline.jupyter_content)
        assert nb["nbformat"] == 4

    def test_pipeline_conda_env_generated(self):
        pipeline = self._build_simple()
        assert pipeline.conda_env_yaml is not None
        assert "conda-forge" in pipeline.conda_env_yaml

    def test_conda_env_contains_bioconda_package(self):
        pipeline = self._build_simple()
        assert "star" in pipeline.conda_env_yaml

    def test_config_datasets_populated(self):
        pipeline = self._build_simple()
        assert "GSE99999" in pipeline.config.datasets

    def test_config_figure_targets_populated(self):
        pipeline = self._build_simple()
        assert "Figure 1" in pipeline.config.figure_targets

    def test_build_filters_out_non_computational_assays(self):
        graph = AssayGraph(
            assays=[
                _make_assay("eCLIP library prep", method_category=MethodCategory.experimental),
                _make_assay("Read alignment", method_category=MethodCategory.computational),
            ],
            dependencies=[],
        )
        method = Method(paper_doi="10.1000/test", assay_graph=graph)
        pipeline = self.builder.build(method, [_make_dataset()], [_make_software()], [_make_figure()])
        step_ids = [s.step_id for s in pipeline.config.steps]
        assert any("read_alignment" in sid for sid in step_ids)
        assert not any("eclip_library_prep" in sid for sid in step_ids)

    def test_build_prunes_dependencies_to_filtered_assays(self):
        graph = AssayGraph(
            assays=[
                _make_assay("Read alignment", method_category=MethodCategory.computational),
                _make_assay("DESeq2", method_category=MethodCategory.computational),
                _make_assay("Western blot", method_category=MethodCategory.experimental),
            ],
            dependencies=[
                AssayDependency(upstream_assay="Read alignment", downstream_assay="DESeq2", dependency_type="data"),
                AssayDependency(upstream_assay="Western blot", downstream_assay="DESeq2", dependency_type="data"),
            ],
        )
        method = Method(paper_doi="10.1000/test", assay_graph=graph)
        filtered = self.builder._computational_only_method(method)
        assert {a.name for a in filtered.assay_graph.assays} == {"Read alignment", "DESeq2"}
        assert len(filtered.assay_graph.dependencies) == 1
        assert filtered.assay_graph.dependencies[0].upstream_assay == "Read alignment"
        assert filtered.assay_graph.dependencies[0].downstream_assay == "DESeq2"

    # --- nf-core detection ---

    def test_rnaseq_maps_to_nfcore_rnaseq(self):
        result = self.builder._check_nfcore("RNA-seq alignment")
        assert result == "rnaseq"

    def test_atacseq_maps_to_nfcore_atacseq(self):
        result = self.builder._check_nfcore("ATAC-seq peak calling")
        assert result == "atacseq"

    def test_unknown_assay_returns_none(self):
        result = self.builder._check_nfcore("Western blot quantification")
        assert result is None

    def test_nfcore_pipeline_set_in_config_for_rnaseq(self):
        method = _make_method(["RNA-seq"])
        config = self.builder._build_config(
            method, [_make_dataset()], [_make_figure()], PipelineBackend.NEXTFLOW
        )
        assert config.nf_core_pipeline == "rnaseq"

    # --- DAG / step ordering ---

    def test_step_depends_on_previous_step_in_assay(self):
        method = _make_method(["RNA-seq"])
        config = self.builder._build_config(
            method, [], [], PipelineBackend.SNAKEMAKE
        )
        # Step 2 should depend on step 1
        step2 = next(s for s in config.steps if "step2" in s.step_id)
        assert len(step2.depends_on) > 0
        assert "step1" in step2.depends_on[0]

    def test_multi_assay_cross_dependencies(self):
        """Downstream assay's first step depends on terminal step of upstream assay."""
        clip_assay = _make_assay("CLIP-seq", n_steps=2)
        rnaseq_assay = _make_assay("RNA-seq", n_steps=2)
        graph = AssayGraph(
            assays=[rnaseq_assay, clip_assay],
            dependencies=[
                AssayDependency(
                    upstream_assay="RNA-seq",
                    downstream_assay="CLIP-seq",
                    dependency_type="normalization_reference",
                    description="CLIP-seq binding uses RNA-seq expression for normalization",
                )
            ],
        )
        method = Method(paper_doi="10.1000/multiomics", assay_graph=graph)
        config = self.builder._build_config(
            method, [], [], PipelineBackend.SNAKEMAKE
        )
        # First step of CLIP-seq should depend on last step of RNA-seq
        clip_steps = [s for s in config.steps if "clip_seq" in s.step_id]
        rna_steps = [s for s in config.steps if "rna_seq" in s.step_id]
        assert len(clip_steps) > 0 and len(rna_steps) > 0
        clip_first = clip_steps[0]
        rna_last_id = rna_steps[-1].step_id
        assert rna_last_id in clip_first.depends_on

    # --- Conda env generation ---

    def test_conda_env_channels_present(self):
        sw = [_make_software("STAR", "star"), _make_software("DESeq2", "bioconductor-deseq2")]
        env = self.builder._generate_conda_env(sw)
        assert "bioconda" in env
        assert "conda-forge" in env

    def test_conda_env_deduplicates_packages(self):
        sw = [_make_software("STAR", "star"), _make_software("STAR2", "star")]
        env = self.builder._generate_conda_env(sw)
        # Only one entry for the "star" bioconda package; STAR2 should be suppressed
        star_lines = [ln for ln in env.splitlines() if "star" in ln.lower()]
        assert len(star_lines) == 1

    def test_conda_env_pip_section_for_pypi(self):
        sw_pypi = Software(
            name="scanpy",
            pypi_package="scanpy",
            version="1.9.0",
            license_type=LicenseType.OPEN_SOURCE,
            description="Single-cell analysis",
        )
        env = self.builder._generate_conda_env([sw_pypi])
        assert "scanpy" in env


# ---------------------------------------------------------------------------
# Tests: PipelineBuilder — topo sort
# ---------------------------------------------------------------------------


class TestTopoSort:
    def setup_method(self):
        self.builder = PipelineBuilder(llm_model="test-model")

    def test_linear_chain_sorted_correctly(self):
        assays = [_make_assay("step_A"), _make_assay("step_B"), _make_assay("step_C")]
        graph = AssayGraph(
            assays=assays,
            dependencies=[
                AssayDependency(upstream_assay="step_A", downstream_assay="step_B", dependency_type="data"),
                AssayDependency(upstream_assay="step_B", downstream_assay="step_C", dependency_type="data"),
            ],
        )
        method = Method(assay_graph=graph)
        order = self.builder._topo_sort_assays(method)
        assert order.index("step_A") < order.index("step_B")
        assert order.index("step_B") < order.index("step_C")

    def test_no_dependencies_preserves_order(self):
        assays = [_make_assay("A"), _make_assay("B")]
        graph = AssayGraph(assays=assays, dependencies=[])
        method = Method(assay_graph=graph)
        order = self.builder._topo_sort_assays(method)
        assert set(order) == {"A", "B"}

    def test_single_assay(self):
        method = _make_method(["RNA-seq"])
        order = self.builder._topo_sort_assays(method)
        assert order == ["RNA-seq"]

    def test_cycle_fallback_returns_all_assays(self):
        """If a cycle exists, should return all assay names (not raise)."""
        assays = [_make_assay("X"), _make_assay("Y")]
        graph = AssayGraph(
            assays=assays,
            dependencies=[
                AssayDependency(upstream_assay="X", downstream_assay="Y", dependency_type="data"),
                AssayDependency(upstream_assay="Y", downstream_assay="X", dependency_type="data"),
            ],
        )
        method = Method(assay_graph=graph)
        order = self.builder._topo_sort_assays(method)
        assert set(order) == {"X", "Y"}


# ---------------------------------------------------------------------------
# Tests: DAG wiring — would fail if generators linearise the dependency graph
# ---------------------------------------------------------------------------


def _make_branching_config() -> PipelineConfig:
    """Two independent root steps (A, B) feeding into one join step (C).

    A ──┐
        ├──> C
    B ──┘

    step_C.depends_on = ["step_A", "step_B"]
    """
    step_a = PipelineStep(
        step_id="step_a",
        name="Step A",
        description="Root step A",
        software="toolA",
        command="toolA {input} > {output}",
        inputs=["raw_a.fastq.gz"],
        outputs=["aligned_a.bam"],
        threads=4, memory_gb=8,
    )
    step_b = PipelineStep(
        step_id="step_b",
        name="Step B",
        description="Root step B",
        software="toolB",
        command="toolB {input} > {output}",
        inputs=["raw_b.fastq.gz"],
        outputs=["aligned_b.bam"],
        threads=4, memory_gb=8,
    )
    step_c = PipelineStep(
        step_id="step_c",
        name="Step C — join",
        description="Join step consuming outputs of A and B",
        software="toolC",
        command="toolC {input} > {output}",
        inputs=[],
        outputs=["integrated_results/"],
        threads=8, memory_gb=16,
        depends_on=["step_a", "step_b"],
    )
    return PipelineConfig(
        name="branching_pipeline",
        description="A→C, B→C branching test pipeline.",
        backend=PipelineBackend.SNAKEMAKE,
        steps=[step_a, step_b, step_c],
        datasets=[],
        figure_targets=[],
    )


class TestSnakemakeDAGWiring:
    """Tests that would fail if Snakemake generator linearises the depends_on graph."""

    def setup_method(self):
        self.gen = SnakemakeGenerator()

    def test_step_with_inputs_and_deps_includes_both(self):
        """Step with explicit inputs AND depends_on must emit both in input block."""
        config = _make_config(2)
        # Give step 2 both explicit inputs and a depends_on
        config.steps[1].inputs = ["explicit_input.txt"]
        config.steps[1].depends_on = ["align_step1"]
        result = self.gen.generate(config)
        rule_start = result.index("rule align_step2:")
        rule_end = result.index("\nrule ", rule_start + 1) if "\nrule " in result[rule_start + 1:] else len(result)
        rule_block = result[rule_start:rule_end]
        assert '"explicit_input.txt"' in rule_block, "Explicit input must appear in rule"
        assert "rules.align_step1.output" in rule_block, "depends_on reference must appear in rule"

    def test_join_step_references_both_upstream_rules(self):
        """Fan-in step (depends on A and B) must reference both in its input block."""
        config = _make_branching_config()
        result = self.gen.generate(config)
        rule_start = result.index("rule step_c:")
        rule_end = result.find("\nrule ", rule_start + 1)
        rule_block = result[rule_start:] if rule_end == -1 else result[rule_start:rule_end]
        assert "rules.step_a.output" in rule_block
        assert "rules.step_b.output" in rule_block

    def test_root_steps_have_no_dep_references(self):
        """Root steps (no depends_on) must not reference rules.*.output."""
        config = _make_branching_config()
        result = self.gen.generate(config)
        for step_id in ("step_a", "step_b"):
            rule_start = result.index(f"rule {step_id}:")
            rule_end = result.find("\nrule ", rule_start + 1)
            block = result[rule_start:] if rule_end == -1 else result[rule_start:rule_end]
            assert "rules." not in block, f"{step_id} should have no rule references"

    def test_rule_all_uses_expand_only_for_sample_outputs(self):
        """expand() must only wrap outputs containing {sample}."""
        config = _make_config(1)
        config.steps[0].outputs = ["{sample}_results.bam", "summary_report.html"]
        # Make step_0 a terminal step
        result = self.gen.generate(config)
        assert 'expand("{sample}_results.bam"' in result
        assert '"summary_report.html",' in result
        assert 'expand("summary_report.html"' not in result

    def test_execution_order_covers_all_steps(self):
        """All steps must appear in the generated Snakefile, not just terminal ones."""
        config = _make_branching_config()
        result = self.gen.generate(config)
        assert "rule step_a:" in result
        assert "rule step_b:" in result
        assert "rule step_c:" in result


class TestNextflowDAGWiring:
    """Tests that would fail if Nextflow generator linearises the depends_on graph."""

    def setup_method(self):
        self.gen = NextflowGenerator()

    def test_root_steps_consume_ch_input(self):
        """Steps with no depends_on should consume ch_input, not a prior process output."""
        config = _make_branching_config()
        config.backend = PipelineBackend.NEXTFLOW
        result = self.gen.generate(config)
        # STEP_A and STEP_B are roots — they must take ch_input
        assert "STEP_A(ch_input)" in result
        assert "STEP_B(ch_input)" in result

    def test_downstream_step_does_not_consume_ch_input(self):
        """Join step C must NOT consume ch_input; it must consume upstream process output."""
        config = _make_branching_config()
        config.backend = PipelineBackend.NEXTFLOW
        result = self.gen.generate(config)
        assert "STEP_C(ch_input)" not in result

    def test_join_step_references_multiple_upstream_channels(self):
        """Fan-in step (depends on A and B) must reference both upstream output channels."""
        config = _make_branching_config()
        config.backend = PipelineBackend.NEXTFLOW
        result = self.gen.generate(config)
        # Both upstream channels must be present in the STEP_C() call line
        step_c_line = next(
            (ln for ln in result.splitlines() if "STEP_C(" in ln), ""
        )
        assert "STEP_A.out" in step_c_line or "STEP_A.out" in result.split("STEP_C(")[1][:100]
        assert "STEP_B.out" in step_c_line or "STEP_B.out" in result.split("STEP_C(")[1][:100]

    def test_linear_pipeline_wires_sequentially(self):
        """For a purely linear pipeline (A→B→C), each step takes previous output."""
        steps = [
            PipelineStep(
                step_id=f"step_{x}",
                name=f"Step {x}",
                description=f"Step {x}",
                software="tool",
                command="tool {input}",
                inputs=[f"{x}.txt"],
                outputs=[f"{x}_out.txt"],
                threads=4, memory_gb=8,
                depends_on=[f"step_{prev}"] if prev else [],
            )
            for x, prev in [("a", None), ("b", "a"), ("c", "b")]
        ]
        config = PipelineConfig(
            name="linear",
            description="Linear A→B→C",
            backend=PipelineBackend.NEXTFLOW,
            steps=steps,
        )
        result = self.gen.generate(config)
        assert "STEP_A(ch_input)" in result
        assert "STEP_B(STEP_A.out.out)" in result
        assert "STEP_C(STEP_B.out.out)" in result
