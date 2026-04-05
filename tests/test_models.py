"""Unit tests for all Pydantic data models.

Tests validate:
1. Minimal instantiation — every model can be created with minimum required fields
2. JSON round-trip — model → JSON → model produces equal objects
3. Pydantic validation — bad enum values and wrong types are rejected
4. Model-specific invariants (helper methods, property accessors, etc.)
"""

import json
import pytest
from pydantic import ValidationError

from researcher_ai.models import (
    # paper
    Paper, PaperSource, PaperType, Reference, Section, SupplementaryItem,
    # figure
    Axis, AxisScale, ColorMapping, ColormapType, ErrorBarType,
    Figure, PanelLayout, PlotCategory, PlotLayer, PlotType,
    RenderingSpec, StatisticalAnnotation, SubFigure,
    # method
    AnalysisStep, Assay, AssayDependency, AssayGraph, Method,
    # dataset
    DataSource, Dataset, GEODataset, ProteomicsDataset, SampleMetadata, SRADataset,
    # software
    Command, Environment, LicenseType, Software,
    # pipeline
    Pipeline, PipelineBackend, PipelineConfig, PipelineStep,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def roundtrip(model_instance):
    """Serialize a model to JSON and deserialize back."""
    cls = type(model_instance)
    json_str = model_instance.model_dump_json()
    data = json.loads(json_str)
    return cls.model_validate(data)


# ── Paper models ──────────────────────────────────────────────────────────────

class TestSection:
    def test_minimal(self):
        s = Section(title="Methods", text="We did stuff.")
        assert s.title == "Methods"
        assert s.subsections == []

    def test_nested(self):
        child = Section(title="Statistics", text="Mann-Whitney U test.")
        parent = Section(title="Methods", text="", subsections=[child])
        assert parent.subsections[0].title == "Statistics"

    def test_roundtrip(self):
        s = Section(title="Abstract", text="This study shows...")
        assert roundtrip(s) == s


class TestReference:
    def test_minimal(self):
        r = Reference(ref_id="[1]")
        assert r.ref_id == "[1]"
        assert r.authors == []

    def test_full(self):
        r = Reference(
            ref_id="Smith2020",
            title="A great paper",
            authors=["Smith, J.", "Doe, A."],
            journal="Nature",
            year=2020,
            doi="10.1038/test",
            pmid="12345678",
        )
        assert r.year == 2020
        assert roundtrip(r) == r


class TestSupplementaryItem:
    def test_minimal(self):
        item = SupplementaryItem(item_id="Table S1", label="DEG list")
        assert item.item_id == "Table S1"

    def test_roundtrip(self):
        item = SupplementaryItem(
            item_id="Data S2",
            label="Peak calls",
            url="https://example.com/s2.bed.gz",
            file_type="bed.gz",
            data_content="peak_list",
        )
        assert roundtrip(item) == item


class TestPaper:
    def test_minimal(self):
        p = Paper(
            title="Test Paper",
            source=PaperSource.PMID,
            source_path="12345678",
        )
        assert p.title == "Test Paper"
        assert p.paper_type == PaperType.EXPERIMENTAL

    def test_get_section(self):
        methods = Section(title="Methods and Materials", text="We did stuff.")
        results = Section(title="Results", text="We found stuff.")
        p = Paper(
            title="Test",
            source=PaperSource.PDF,
            source_path="/tmp/paper.pdf",
            sections=[methods, results],
        )
        assert p.methods_section is not None
        assert p.methods_section.title == "Methods and Materials"
        assert p.results_section is not None
        assert p.get_section("nonexistent") is None

    def test_paper_type_enum(self):
        p = Paper(
            title="A review",
            source=PaperSource.DOI,
            source_path="10.1000/test",
            paper_type=PaperType.REVIEW,
        )
        assert p.paper_type == PaperType.REVIEW

    def test_roundtrip(self):
        p = Paper(
            title="CLIP-seq reveals...",
            authors=["Yeo, G.W.", "Van Nostrand, E.L."],
            doi="10.1016/j.cell.2020.01.001",
            pmid="31983180",
            source=PaperSource.PMID,
            source_path="31983180",
            paper_type=PaperType.MULTI_OMIC,
        )
        assert roundtrip(p) == p

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            Paper(title="T", source="ftp", source_path="x")


# ── Figure models ─────────────────────────────────────────────────────────────

class TestAxis:
    def test_minimal(self):
        a = Axis(label="log2FC")
        assert a.scale == AxisScale.LINEAR
        assert a.is_inverted is False

    def test_log_scale(self):
        a = Axis(label="-log10(p-value)", scale=AxisScale.LOG10, is_inverted=False)
        assert a.scale == AxisScale.LOG10

    def test_roundtrip(self):
        a = Axis(label="Expression", scale=AxisScale.LOG2, units="TPM", limits=(0.0, 10.0))
        assert roundtrip(a) == a


class TestColorMapping:
    def test_defaults(self):
        cm = ColorMapping()
        assert cm.colormap_type == ColormapType.QUALITATIVE
        assert cm.is_colorblind_safe is True

    def test_diverging(self):
        cm = ColorMapping(
            variable="log2FC",
            colormap_type=ColormapType.DIVERGING,
            colormap_name="RdBu_r",
            center_value=0.0,
        )
        assert cm.center_value == 0.0
        assert roundtrip(cm) == cm


class TestStatisticalAnnotation:
    def test_defaults(self):
        sa = StatisticalAnnotation()
        assert "***" in sa.significance_levels

    def test_with_comparisons(self):
        sa = StatisticalAnnotation(
            test_name="Wilcoxon",
            comparisons=[("WT", "KO")],
            p_values={"WT_vs_KO": 0.003},
        )
        assert sa.comparisons[0] == ("WT", "KO")
        assert roundtrip(sa) == sa


class TestPlotLayer:
    def test_primary(self):
        layer = PlotLayer(plot_type=PlotType.BOX, is_primary=True)
        assert layer.is_primary is True

    def test_overlay(self):
        layer = PlotLayer(
            plot_type=PlotType.STRIP,
            is_primary=False,
            library_hint="seaborn",
            function_hint="sns.stripplot",
            parameters={"alpha": "0.4", "jitter": "True"},
        )
        assert layer.parameters["alpha"] == "0.4"
        assert roundtrip(layer) == layer


class TestSubFigure:
    def _minimal(self) -> SubFigure:
        return SubFigure(
            label="a",
            description="Volcano plot of DEGs",
            plot_category=PlotCategory.GENOMIC,
            plot_type=PlotType.VOLCANO,
        )

    def test_minimal(self):
        sf = self._minimal()
        assert sf.error_bars == ErrorBarType.NONE
        assert sf.shows_individual_points is False

    def test_multi_layer(self):
        sf = SubFigure(
            label="b",
            description="Box + strip",
            plot_category=PlotCategory.CATEGORICAL,
            plot_type=PlotType.BOX,
            layers=[
                PlotLayer(plot_type=PlotType.BOX, is_primary=True),
                PlotLayer(plot_type=PlotType.STRIP, is_primary=False),
            ],
        )
        assert len(sf.layers) == 2

    def test_roundtrip(self):
        sf = SubFigure(
            label="c",
            description="UMAP coloured by cluster",
            plot_category=PlotCategory.DIMENSIONALITY,
            plot_type=PlotType.UMAP,
            x_axis=Axis(label="UMAP1"),
            y_axis=Axis(label="UMAP2"),
            color_mapping=ColorMapping(
                variable="cluster",
                colormap_type=ColormapType.QUALITATIVE,
                colormap_name="Okabe-Ito",
            ),
            sample_size="n=12,345 cells",
        )
        assert roundtrip(sf) == sf

    def test_invalid_plot_type(self):
        with pytest.raises(ValidationError):
            SubFigure(
                label="a",
                description="test",
                plot_category=PlotCategory.RELATIONAL,
                plot_type="not_a_real_type",
            )


class TestFigure:
    def test_minimal(self):
        f = Figure(
            figure_id="Figure 1",
            title="Overview",
            caption="Fig 1. Summary of results.",
            purpose="Introduce the main experimental system.",
        )
        assert f.layout.n_rows == 1
        assert f.rendering is None

    def test_full(self):
        sf = SubFigure(
            label="a",
            description="Volcano",
            plot_category=PlotCategory.GENOMIC,
            plot_type=PlotType.VOLCANO,
        )
        f = Figure(
            figure_id="Figure 2",
            title="DEG Analysis",
            caption="Full caption text.",
            purpose="Show differential expression.",
            subfigures=[sf],
            layout=PanelLayout(n_rows=1, n_cols=2),
            rendering=RenderingSpec(target_journal="nature", dpi=300),
        )
        assert len(f.subfigures) == 1
        assert f.layout.n_cols == 2
        assert roundtrip(f) == f


# ── Method models ─────────────────────────────────────────────────────────────

class TestAnalysisStep:
    def test_minimal(self):
        s = AnalysisStep(
            step_number=1,
            description="Align reads with STAR",
            input_data="raw FASTQ files",
            output_data="BAM files",
        )
        assert s.parameters == {}

    def test_roundtrip(self):
        s = AnalysisStep(
            step_number=2,
            description="Count reads",
            input_data="BAM",
            output_data="count matrix",
            software="featureCounts",
            software_version="2.0.3",
            parameters={"strandedness": "reverse"},
        )
        assert roundtrip(s) == s


class TestAssayGraph:
    def test_empty(self):
        ag = AssayGraph()
        assert ag.assays == []
        assert ag.dependencies == []

    def test_get_assay(self):
        a1 = Assay(name="RNA-seq", description="mRNA", data_type="sequencing")
        ag = AssayGraph(assays=[a1])
        assert ag.get_assay("RNA-seq") is a1
        assert ag.get_assay("rna-seq") is a1
        assert ag.get_assay("missing") is None

    def test_dependency_traversal(self):
        a1 = Assay(name="RNA-seq", description="mRNA", data_type="sequencing")
        a2 = Assay(name="CLIP-seq", description="binding", data_type="sequencing")
        dep = AssayDependency(
            upstream_assay="RNA-seq",
            downstream_assay="CLIP-seq",
            dependency_type="normalization_reference",
            description="CLIP signal normalized to RNA-seq expression",
        )
        ag = AssayGraph(assays=[a1, a2], dependencies=[dep])
        assert ag.upstream_of("CLIP-seq") == ["RNA-seq"]
        assert ag.downstream_of("RNA-seq") == ["CLIP-seq"]
        assert ag.upstream_of("RNA-seq") == []


class TestMethod:
    def test_minimal(self):
        m = Method()
        assert m.assays == []

    def test_with_assays(self):
        a = Assay(name="ChIP-seq", description="Histone marks", data_type="sequencing")
        m = Method(assay_graph=AssayGraph(assays=[a]))
        assert len(m.assays) == 1
        assert m.assays[0].name == "ChIP-seq"

    def test_roundtrip(self):
        m = Method(
            paper_doi="10.1000/test",
            data_availability="GEO: GSE12345",
            code_availability="github.com/yeolab/test",
            raw_methods_text="Cells were grown...",
        )
        assert roundtrip(m) == m


# ── Dataset models ────────────────────────────────────────────────────────────

class TestDataset:
    def test_minimal(self):
        d = Dataset(accession="GSE12345", source=DataSource.GEO)
        assert d.total_samples == 0

    def test_geo_dataset(self):
        g = GEODataset(
            accession="GSE99999",
            title="Yeo Lab CLIP-seq",
            organism="Homo sapiens",
            experiment_type="CLIP-seq",
            child_series=["GSE99997", "GSE99998"],
        )
        assert g.source == DataSource.GEO
        assert len(g.child_series) == 2
        assert roundtrip(g) == g

    def test_sra_dataset(self):
        s = SRADataset(
            accession="SRP12345",
            srp="SRP12345",
            srr_list=["SRR123", "SRR124"],
        )
        assert s.source == DataSource.SRA
        assert roundtrip(s) == s

    def test_proteomics_dataset(self):
        p = ProteomicsDataset(
            accession="PXD012345",
            pride_accession="PXD012345",
            instrument="Orbitrap Fusion Lumos",
            quantification_method="TMT",
        )
        assert p.pride_accession == "PXD012345"
        assert roundtrip(p) == p


# ── Software models ───────────────────────────────────────────────────────────

class TestSoftware:
    def test_minimal(self):
        s = Software(name="STAR")
        assert s.license_type == LicenseType.UNKNOWN
        assert s.commands == []

    def test_full(self):
        cmd = Command(
            command_template="STAR --runMode alignReads --readFilesIn {r1} {r2}",
            description="Align paired-end reads",
            required_inputs=["R1.fastq.gz", "R2.fastq.gz"],
            outputs=["Aligned.out.bam"],
        )
        env = Environment(
            conda_yaml="name: star_env\ndeps:\n  - star=2.7.10a",
            docker_image="quay.io/biocontainers/star:2.7.10a",
        )
        s = Software(
            name="STAR",
            version="2.7.10a",
            language="C++",
            license_type=LicenseType.OPEN_SOURCE,
            commands=[cmd],
            environment=env,
            bioconda_package="star",
            github_repo="alexdobin/STAR",
        )
        assert roundtrip(s) == s


# ── Pipeline models ───────────────────────────────────────────────────────────

class TestPipelineStep:
    def test_minimal(self):
        step = PipelineStep(
            step_id="align",
            name="Align reads",
            description="Align FASTQ to genome",
            software="STAR",
            command="STAR --runMode alignReads",
        )
        assert step.depends_on == []
        assert step.threads == 1

    def test_with_dependencies(self):
        step = PipelineStep(
            step_id="count",
            name="Count reads",
            description="Count aligned reads per gene",
            software="featureCounts",
            command="featureCounts -a {gtf} -o {out} {bam}",
            depends_on=["align"],
        )
        assert "align" in step.depends_on
        assert roundtrip(step) == step


class TestPipelineConfig:
    def test_execution_order_linear(self):
        s1 = PipelineStep(step_id="a", name="A", description="", software="x", command="x")
        s2 = PipelineStep(step_id="b", name="B", description="", software="x", command="x", depends_on=["a"])
        s3 = PipelineStep(step_id="c", name="C", description="", software="x", command="x", depends_on=["b"])
        cfg = PipelineConfig(
            name="test",
            description="",
            backend=PipelineBackend.SNAKEMAKE,
            steps=[s1, s2, s3],
        )
        order = cfg.execution_order()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_execution_order_dag(self):
        """Two independent upstream steps both feeding one downstream step."""
        rna = PipelineStep(step_id="rna_align", name="RNA align", description="", software="STAR", command="")
        clip = PipelineStep(step_id="clip_align", name="CLIP align", description="", software="STAR", command="")
        peaks = PipelineStep(
            step_id="call_peaks",
            name="Call peaks",
            description="",
            software="CLIPper",
            command="",
            depends_on=["rna_align", "clip_align"],
        )
        cfg = PipelineConfig(
            name="clip_rna",
            description="",
            backend=PipelineBackend.NEXTFLOW,
            steps=[rna, clip, peaks],
        )
        order = cfg.execution_order()
        assert order.index("rna_align") < order.index("call_peaks")
        assert order.index("clip_align") < order.index("call_peaks")

    def test_roundtrip(self):
        step = PipelineStep(step_id="s1", name="S1", description="", software="X", command="x")
        cfg = PipelineConfig(
            name="pipeline",
            description="test pipeline",
            backend=PipelineBackend.SNAKEMAKE,
            steps=[step],
            datasets=["GSE12345"],
            figure_targets=["Figure 1"],
        )
        assert roundtrip(cfg) == cfg


class TestPipeline:
    def test_minimal(self):
        cfg = PipelineConfig(name="p", description="", backend=PipelineBackend.SNAKEMAKE)
        p = Pipeline(config=cfg)
        assert p.snakefile_content is None
        assert p.jupyter_content is None

    def test_with_content(self):
        cfg = PipelineConfig(name="p", description="", backend=PipelineBackend.NEXTFLOW)
        p = Pipeline(
            config=cfg,
            nextflow_content="nextflow.enable.dsl=2\n...",
            conda_env_yaml="name: env\ndeps:\n  - star",
        )
        assert roundtrip(p) == p


# ── Cross-model integration ──────────────────────────────────────────────────
# Added per EVALUATION_PHASE_1.md Finding 2 (accepted): compose Paper +
# Figure + Method + Dataset + Pipeline into a coherent object graph and
# verify the full structure round-trips through JSON serialization.


class TestCrossModelComposition:
    """End-to-end integration test: compose a realistic eCLIP paper object
    graph spanning all six model modules and verify JSON round-trip."""

    def _build_paper(self) -> Paper:
        methods = Section(
            title="Methods",
            text="eCLIP was performed as described...",
            figures_referenced=["Figure 1", "Figure 2"],
        )
        results = Section(
            title="Results",
            text="We identified 356 significant binding peaks...",
            figures_referenced=["Figure 2"],
        )
        return Paper(
            title="Robust transcriptome-wide discovery of RNA-binding protein binding sites",
            authors=["Van Nostrand, E.L.", "Pratt, G.A.", "Yeo, G.W."],
            doi="10.1038/nmeth.3810",
            pmid="26971820",
            source=PaperSource.PMID,
            source_path="26971820",
            paper_type=PaperType.EXPERIMENTAL,
            sections=[methods, results],
            references=[
                Reference(ref_id="[1]", title="A prior CLIP method", year=2012),
            ],
            supplementary_items=[
                SupplementaryItem(
                    item_id="Table S1",
                    label="eCLIP peaks BED file",
                    file_type="bed.gz",
                    data_content="peak_list",
                ),
            ],
            figure_ids=["Figure 1", "Figure 2"],
        )

    def _build_figure(self) -> Figure:
        volcano = SubFigure(
            label="a",
            description="Volcano plot of RBFOX2 eCLIP enrichment",
            plot_category=PlotCategory.GENOMIC,
            plot_type=PlotType.VOLCANO,
            layers=[
                PlotLayer(plot_type=PlotType.SCATTER, is_primary=True),
            ],
            x_axis=Axis(label="log2 Fold Change", scale=AxisScale.LINEAR),
            y_axis=Axis(label="-log10(p-value)", scale=AxisScale.LOG10),
            color_mapping=ColorMapping(
                variable="significance",
                colormap_type=ColormapType.BINARY,
            ),
            error_bars=ErrorBarType.NONE,
            sample_size="n=2 biological replicates",
            data_source="GSE78509",
            assays=["eCLIP"],
        )
        heatmap = SubFigure(
            label="b",
            description="Heatmap of top enriched peaks across RBPs",
            plot_category=PlotCategory.MATRIX,
            plot_type=PlotType.CLUSTERMAP,
            x_axis=Axis(label="RBP", scale=AxisScale.CATEGORICAL),
            y_axis=Axis(label="Peak region", scale=AxisScale.CATEGORICAL),
            color_mapping=ColorMapping(
                variable="log2FC",
                colormap_type=ColormapType.DIVERGING,
                colormap_name="RdBu_r",
                center_value=0.0,
            ),
        )
        return Figure(
            figure_id="Figure 2",
            title="eCLIP identifies reproducible RBP binding sites",
            caption="(a) Volcano plot... (b) Heatmap...",
            purpose="Demonstrate specificity of eCLIP signal over SMInput",
            subfigures=[volcano, heatmap],
            layout=PanelLayout(n_rows=1, n_cols=2, width_ratios=[1.0, 1.5]),
            rendering=RenderingSpec(target_journal="nature_methods", dpi=300),
            datasets_used=["GSE78509"],
            methods_used=["eCLIP", "Peak Calling"],
        )

    def _build_method(self) -> Method:
        eclip = Assay(
            name="eCLIP",
            description="Enhanced CLIP for RBP binding",
            data_type="sequencing",
            raw_data_source="GEO: GSE78509",
            steps=[
                AnalysisStep(
                    step_number=1,
                    description="Align reads with STAR",
                    input_data="FASTQ",
                    output_data="BAM",
                    software="STAR",
                    software_version="2.7.10a",
                ),
                AnalysisStep(
                    step_number=2,
                    description="UMI-based deduplication",
                    input_data="BAM",
                    output_data="dedup BAM",
                    software="umi_tools",
                ),
            ],
            figures_produced=["Figure 2"],
        )
        sminput = Assay(
            name="SMInput",
            description="Size-matched input control",
            data_type="sequencing",
        )
        peak_calling = Assay(
            name="Peak Calling",
            description="CLIPper peak calling vs. SMInput",
            data_type="analysis",
            figures_produced=["Figure 2"],
        )
        return Method(
            paper_doi="10.1038/nmeth.3810",
            assay_graph=AssayGraph(
                assays=[eclip, sminput, peak_calling],
                dependencies=[
                    AssayDependency(
                        upstream_assay="eCLIP",
                        downstream_assay="Peak Calling",
                        dependency_type="normalization_reference",
                    ),
                    AssayDependency(
                        upstream_assay="SMInput",
                        downstream_assay="Peak Calling",
                        dependency_type="normalization_reference",
                    ),
                ],
            ),
            data_availability="GEO: GSE78509",
            code_availability="https://github.com/YeoLab/eclip",
        )

    def _build_dataset(self) -> GEODataset:
        return GEODataset(
            accession="GSE78509",
            title="eCLIP of 150 RBPs in K562 and HepG2",
            organism="Homo sapiens",
            experiment_type="eCLIP",
            series_type="SuperSeries",
            child_series=["GSE78506", "GSE78507"],
            total_samples=1060,
            samples=[
                SampleMetadata(
                    sample_id="GSM2071727",
                    title="RBFOX2 eCLIP rep1 K562",
                    organism="Homo sapiens",
                    layout="SINGLE",
                    platform="Illumina HiSeq 4000",
                ),
            ],
        )

    def _build_software(self) -> list[Software]:
        return [
            Software(
                name="STAR",
                version="2.7.10a",
                license_type=LicenseType.OPEN_SOURCE,
                language="C++",
                bioconda_package="star",
                github_repo="alexdobin/STAR",
                commands=[
                    Command(
                        command_template="STAR --runMode alignReads --readFilesIn {r1}",
                        description="Align single-end eCLIP reads",
                    ),
                ],
                environment=Environment(docker_image="quay.io/biocontainers/star:2.7.10a"),
            ),
            Software(
                name="CLIPper",
                version="2.0.0",
                license_type=LicenseType.OPEN_SOURCE,
                language="Python",
                github_repo="YeoLab/clipper",
            ),
        ]

    def _build_pipeline(self) -> Pipeline:
        steps = [
            PipelineStep(
                step_id="download",
                name="Download FASTQ",
                description="Fetch from SRA",
                software="fasterq-dump",
                command="fasterq-dump {srr}",
            ),
            PipelineStep(
                step_id="align_eclip",
                name="Align eCLIP",
                description="STAR alignment of eCLIP reads",
                software="STAR",
                command="STAR --readFilesIn {fastq}",
                depends_on=["download"],
                threads=8,
                memory_gb=32,
            ),
            PipelineStep(
                step_id="align_input",
                name="Align SMInput",
                description="STAR alignment of SMInput reads",
                software="STAR",
                command="STAR --readFilesIn {fastq}",
                depends_on=["download"],
                threads=8,
                memory_gb=32,
            ),
            PipelineStep(
                step_id="call_peaks",
                name="Call peaks",
                description="CLIPper vs. SMInput",
                software="CLIPper",
                command="clipper -b {ip} -s hg38",
                depends_on=["align_eclip", "align_input"],
            ),
        ]
        config = PipelineConfig(
            name="eclip_reproduction",
            description="Reproduce eCLIP analysis from Van Nostrand 2016",
            backend=PipelineBackend.SNAKEMAKE,
            steps=steps,
            datasets=["GSE78509"],
            figure_targets=["Figure 2"],
        )
        return Pipeline(config=config)

    def test_full_composition(self):
        """Build the complete object graph and verify all cross-references are consistent."""
        paper = self._build_paper()
        figure = self._build_figure()
        method = self._build_method()
        dataset = self._build_dataset()
        software = self._build_software()
        pipeline = self._build_pipeline()

        # Cross-reference consistency checks
        # Figure references a dataset that exists
        assert figure.datasets_used[0] == dataset.accession
        # Figure references methods that exist in the Method model
        for m in figure.methods_used:
            assert method.assay_graph.get_assay(m) is not None, f"Method '{m}' not in assay graph"
        # Pipeline targets figures that exist
        assert pipeline.config.figure_targets[0] == figure.figure_id
        # Pipeline datasets match the dataset model
        assert pipeline.config.datasets[0] == dataset.accession
        # AssayGraph DAG: Peak Calling depends on eCLIP and SMInput
        assert set(method.assay_graph.upstream_of("Peak Calling")) == {"eCLIP", "SMInput"}
        # Pipeline DAG: call_peaks depends on both alignment steps
        peaks_step = next(s for s in pipeline.config.steps if s.step_id == "call_peaks")
        assert set(peaks_step.depends_on) == {"align_eclip", "align_input"}
        # Pipeline topological order respects dependencies
        order = pipeline.config.execution_order()
        assert order.index("download") < order.index("align_eclip")
        assert order.index("download") < order.index("align_input")
        assert order.index("align_eclip") < order.index("call_peaks")
        # Paper supplementary items exist
        assert paper.supplementary_items[0].data_content == "peak_list"

    def test_full_composition_roundtrip(self):
        """Every component of the object graph survives JSON serialization."""
        paper = self._build_paper()
        figure = self._build_figure()
        method = self._build_method()
        dataset = self._build_dataset()
        pipeline = self._build_pipeline()

        assert roundtrip(paper) == paper
        assert roundtrip(figure) == figure
        assert roundtrip(method) == method
        assert roundtrip(dataset) == dataset
        assert roundtrip(pipeline) == pipeline

        for sw in self._build_software():
            assert roundtrip(sw) == sw
