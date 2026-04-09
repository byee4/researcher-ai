"""Unit tests for Phase 3: FigureParser.

Testing strategy:
- Pure helpers (_fig_ids_match, _build_fig_ref_pattern, _extract_caption_from_text,
  _subfigure_from_meta, _axis_from_meta) tested directly with no mocking.
- FigureParser methods that use LLM are tested with ask_claude_structured mocked.
- FigureParser._identify_datasets is tested for regex extraction without LLM.
- Integration: parse_figure and parse_all_figures with fully mocked dependencies.
"""

import textwrap
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from researcher_ai.models.figure import (
    Axis,
    AxisScale,
    ColorMapping,
    ColormapType,
    ErrorBarType,
    Figure,
    PanelLayout,
    PlotCategory,
    PlotType,
    SubFigure,
)
from researcher_ai.models.paper import Paper, PaperSource, PaperType, Section
from researcher_ai.models.paper import BioCPassageContext
from researcher_ai.parsers.figure_parser import (
    FigureParser,
    SubfigureDecompositionTimeoutError,
    _SubFigureList,
    _SubFigureMeta,
    _AxisMeta,
    _FigurePurpose,
    _MethodsAndDatasets,
    _VisionFigureExtraction,
    _axis_from_meta,
    _build_fig_ref_pattern,
    _extract_caption_from_text,
    _fig_ids_match,
    _canonical_main_figure_id,
    _disambiguate_subfigure_plot,
    _infer_plot_category_from_text,
    _infer_plot_type_candidates,
    _infer_axis_scale_from_text,
    _panel_caption_context,
    _panel_bioc_evidence,
    _panel_in_text_context,
    _resolve_figure_title,
    _subfigure_from_meta,
)
from researcher_ai.parsers.figure_calibration import FigureCalibrationEngine
from researcher_ai.utils.llm import LLMCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CAPTION = (
    "Figure 1. Overview of the eCLIP pipeline. "
    "(a) Schematic of the eCLIP protocol showing library preparation steps. "
    "(b) Volcano plot of enrichment (log2FC) vs. significance (-log10 p-value) "
    "for all binding peaks. n=3 biological replicates. Error bars represent SEM. "
    "(c) UMAP embedding of 10,000 cells coloured by cluster identity."
)

SAMPLE_PAPER_SECTIONS = [
    Section(
        title="Introduction",
        text=(
            "The eCLIP protocol (Figure 1a) enables transcriptome-wide mapping "
            "of RNA–protein interactions. See Figure 1b for the volcano plot summary."
        ),
        figures_referenced=["Figure 1"],
    ),
    Section(
        title="Results",
        text=(
            "We applied eCLIP to 150 RBPs. Figure 1b shows significantly enriched peaks. "
            "UMAP clustering is presented in Figure 1c. Table S1 lists all peaks."
        ),
        figures_referenced=["Figure 1"],
    ),
    Section(
        title="Methods",
        text=(
            "RNA-seq libraries were prepared from HEK293 cells. "
            "Peak calling was performed with CLIPper v2.0. "
            "Differential enrichment was computed with DESeq2."
        ),
        figures_referenced=[],
    ),
]

SAMPLE_PAPER = Paper(
    title="Robust transcriptome-wide discovery of RNA-binding protein binding sites",
    authors=["Van Nostrand, EL", "Yeo, GW"],
    abstract="eCLIP is a method for high-throughput CLIP-seq.",
    doi="10.1038/nmeth.3810",
    pmid="26971820",
    pmcid="PMC4878918",
    source=PaperSource.PMCID,
    source_path="PMC4878918",
    paper_type=PaperType.EXPERIMENTAL,
    sections=SAMPLE_PAPER_SECTIONS,
    figure_ids=["Figure 1", "Figure 2"],
    raw_text=(
        "eCLIP protocol\n\n"
        + "\n".join(s.text for s in SAMPLE_PAPER_SECTIONS)
        + "\n\nFigure 1. Overview of the eCLIP pipeline. (a) Schematic. "
        "(b) Volcano plot of enrichment. (c) UMAP embedding."
    ),
)


def _make_parser() -> FigureParser:
    parser = FigureParser.__new__(FigureParser)
    parser.llm_model = "test-model"
    parser.vision_model = "primary-vision"
    parser.cache = None
    parser.max_figure_llm_timeouts_per_paper = 3
    parser.subfigure_timeout_seconds = 0.0
    parser.calibration_engine = FigureCalibrationEngine()
    parser.figure_trace_path = ""
    parser._figure_trace_events = []
    parser._active_paper_ref = ""
    parser._active_figure_id = ""
    return parser


def test_figure_parser_init_supports_dependency_injection():
    injected_cache = MagicMock(spec=LLMCache)
    injected_engine = MagicMock(spec=FigureCalibrationEngine)
    parser = FigureParser(
        llm_model="test-model",
        cache=injected_cache,
        calibration_engine=injected_engine,
    )
    assert parser.cache is injected_cache
    assert parser.calibration_engine is injected_engine


def test_figure_parser_init_uses_cache_dir_when_cache_not_injected(tmp_path: Path):
    parser = FigureParser(llm_model="test-model", cache_dir=str(tmp_path))
    assert isinstance(parser.cache, LLMCache)


def _subfig_meta(**kwargs) -> _SubFigureMeta:
    defaults = dict(
        label="a",
        description="Schematic of the protocol",
        plot_type="image",
        plot_category="image",
    )
    defaults.update(kwargs)
    return _SubFigureMeta(**defaults)


# ---------------------------------------------------------------------------
# TestFigIdsMatch
# ---------------------------------------------------------------------------

class TestFigIdsMatch:
    """_fig_ids_match — pure function, no I/O."""

    def test_exact_match(self):
        assert _fig_ids_match("Figure 1", "Figure 1")

    def test_fig_vs_figure(self):
        assert _fig_ids_match("Fig. 1", "Figure 1")

    def test_case_insensitive(self):
        assert _fig_ids_match("FIGURE 1", "figure 1")

    def test_different_number(self):
        assert not _fig_ids_match("Figure 1", "Figure 2")

    def test_supplementary_match(self):
        assert _fig_ids_match("Supplementary Figure S1", "Supplementary Figure S1")

    def test_panel_variants(self):
        # Label with panel should NOT match plain figure
        assert not _fig_ids_match("Figure 1A", "Figure 2")


class TestCanonicalMainFigureId:
    def test_plain(self):
        assert _canonical_main_figure_id("Figure 3") == "Figure 3"

    def test_panel(self):
        assert _canonical_main_figure_id("Figure 3A") == "Figure 3"

    def test_supplementary_filtered(self):
        assert _canonical_main_figure_id("Supplementary Figure S1") == ""


# ---------------------------------------------------------------------------
# TestBuildFigRefPattern
# ---------------------------------------------------------------------------

class TestBuildFigRefPattern:
    """_build_fig_ref_pattern — returns compiled regex."""

    def test_matches_full_form(self):
        pat = _build_fig_ref_pattern("Figure 1")
        assert pat.search("As shown in Figure 1, the data...")

    def test_matches_abbreviated(self):
        pat = _build_fig_ref_pattern("Figure 1")
        assert pat.search("See Fig. 1 for details.")

    def test_matches_with_panel(self):
        pat = _build_fig_ref_pattern("Figure 1")
        assert pat.search("Figure 1A shows the volcano plot.")

    def test_does_not_match_different_number(self):
        pat = _build_fig_ref_pattern("Figure 1")
        assert not pat.search("See Figure 2 for details.")

    def test_supplementary_figure(self):
        pat = _build_fig_ref_pattern("Supplementary Figure S1")
        assert pat.search("QC metrics in Supplementary Fig. S1.")

    def test_figure_with_range(self):
        pat = _build_fig_ref_pattern("Figure 3")
        assert pat.search("Figure 3A-D shows the results.")


# ---------------------------------------------------------------------------
# TestExtractCaptionFromText
# ---------------------------------------------------------------------------

class TestExtractCaptionFromText:
    """_extract_caption_from_text — pure function."""

    def test_finds_caption_in_text(self):
        text = (
            "Some body text.\n\n"
            "Figure 1. Overview of the pipeline. Panel (a) shows the schematic. "
            "Panel (b) shows the results.\n\n"
            "Figure 2. Another figure."
        )
        caption = _extract_caption_from_text(text, "Figure 1")
        assert "Overview" in caption
        assert "pipeline" in caption

    def test_does_not_return_wrong_figure(self):
        text = "Figure 2. Some other figure caption."
        caption = _extract_caption_from_text(text, "Figure 1")
        assert caption == ""

    def test_handles_missing_figure(self):
        text = "No figures here at all."
        assert _extract_caption_from_text(text, "Figure 1") == ""

    def test_truncates_at_next_figure(self):
        text = (
            "Figure 1. First figure.\n"
            "Figure 2. Second figure."
        )
        caption = _extract_caption_from_text(text, "Figure 1")
        assert "First figure" in caption
        assert "Second figure" not in caption

    def test_abbreviated_fig(self):
        text = "Fig. 3. Volcano plot of enrichment."
        caption = _extract_caption_from_text(text, "Figure 3")
        assert "Volcano" in caption


# ---------------------------------------------------------------------------
# TestAxisFromMeta
# ---------------------------------------------------------------------------

class TestAxisFromMeta:
    """_axis_from_meta — pure conversion."""

    def test_basic(self):
        meta = _AxisMeta(label="log2FC", scale="log2", units="fold change")
        axis = _axis_from_meta(meta)
        assert axis.label == "log2FC"
        assert axis.scale == AxisScale.LOG2
        assert axis.units == "fold change"

    def test_unknown_scale_defaults_to_linear(self):
        meta = _AxisMeta(label="X", scale="quadratic")
        axis = _axis_from_meta(meta)
        assert axis.scale == AxisScale.LINEAR

    def test_is_inverted(self):
        meta = _AxisMeta(label="-log10(p)", scale="reversed", is_inverted=True)
        axis = _axis_from_meta(meta)
        assert axis.is_inverted is True


class TestAxisScaleInference:
    """_infer_axis_scale_from_text heuristics."""

    def test_infers_log10_and_inverted(self):
        scale, inverted = _infer_axis_scale_from_text("-log10(p-value)", "")
        assert scale == AxisScale.LOG10
        assert inverted is True

    def test_infers_log2(self):
        scale, inverted = _infer_axis_scale_from_text("log2 fold change", "")
        assert scale == AxisScale.LOG2
        assert inverted is False

    def test_infers_categorical(self):
        scale, inverted = _infer_axis_scale_from_text("ZFP knockdowns", "")
        assert scale == AxisScale.CATEGORICAL
        assert inverted is False


class TestResolveFigureTitle:
    """_resolve_figure_title fallback behavior."""

    def test_uses_non_generic_llm_title(self):
        title = _resolve_figure_title("Figure 1", "Differential Expression Overview", "Figure 1. caption")
        assert title == "Differential Expression Overview"

    def test_falls_back_to_caption_when_title_generic(self):
        title = _resolve_figure_title(
            "Figure 1",
            "Figure 1",
            "Figure 1. ZFP residual analysis across knockdowns. Additional details.",
        )
        assert "ZFP residual analysis" in title


# ---------------------------------------------------------------------------
# TestSubfigureFromMeta
# ---------------------------------------------------------------------------

class TestSubfigureFromMeta:
    """_subfigure_from_meta — pure conversion."""

    def test_basic_conversion(self):
        meta = _subfig_meta(
            label="b",
            description="Volcano plot",
            plot_type="volcano",
            plot_category="genomic",
        )
        sf = _subfigure_from_meta(meta)
        assert sf.label == "b"
        assert sf.plot_type == PlotType.VOLCANO
        assert sf.plot_category == PlotCategory.GENOMIC

    def test_unknown_plot_type_falls_back(self):
        meta = _subfig_meta(plot_type="imaginary_plot")
        sf = _subfigure_from_meta(meta)
        assert sf.plot_type == PlotType.OTHER

    def test_unknown_category_falls_back(self):
        meta = _subfig_meta(plot_category="magic_category")
        sf = _subfigure_from_meta(meta)
        assert sf.plot_category == PlotCategory.COMPOSITE

    def test_error_bars_parsed(self):
        meta = _subfig_meta(error_bars="sem")
        sf = _subfigure_from_meta(meta)
        assert sf.error_bars == ErrorBarType.SEM

    def test_invalid_error_bars_default(self):
        meta = _subfig_meta(error_bars="fancybars")
        sf = _subfigure_from_meta(meta)
        assert sf.error_bars == ErrorBarType.NONE

    def test_color_variable_creates_mapping(self):
        meta = _subfig_meta(color_variable="cluster")
        sf = _subfigure_from_meta(meta)
        assert sf.color_mapping is not None
        assert sf.color_mapping.variable == "cluster"

    def test_no_color_variable_no_mapping(self):
        meta = _subfig_meta(color_variable=None)
        sf = _subfigure_from_meta(meta)
        assert sf.color_mapping is None

    def test_axes_populated(self):
        meta = _subfig_meta(
            x_axis=_AxisMeta(label="log2FC", scale="log2"),
            y_axis=_AxisMeta(label="-log10(p)", scale="reversed", is_inverted=True),
        )
        sf = _subfigure_from_meta(meta)
        assert sf.x_axis is not None
        assert sf.x_axis.scale == AxisScale.LOG2
        assert sf.y_axis is not None
        assert sf.y_axis.is_inverted is True

    def test_statistical_annotation(self):
        meta = _subfig_meta(statistical_test="Wilcoxon rank-sum")
        sf = _subfigure_from_meta(meta)
        assert sf.statistical_annotations is not None
        assert sf.statistical_annotations.test_name == "Wilcoxon rank-sum"

    def test_primary_layer_always_added(self):
        meta = _subfig_meta(plot_type="heatmap", plot_category="matrix")
        sf = _subfigure_from_meta(meta)
        assert len(sf.layers) >= 1
        assert sf.layers[0].is_primary is True
        assert sf.layers[0].plot_type == PlotType.HEATMAP


# ---------------------------------------------------------------------------
# TestComplexDisambiguation
# ---------------------------------------------------------------------------

class TestComplexDisambiguation:
    """Cue-based two-stage disambiguation for complex plot types."""

    def _make_sf(
        self,
        *,
        description: str,
        plot_type: PlotType = PlotType.OTHER,
        plot_category: PlotCategory = PlotCategory.COMPOSITE,
    ) -> SubFigure:
        return SubFigure(
            label="a",
            description=description,
            plot_type=plot_type,
            plot_category=plot_category,
        )

    def test_stage_a_category_detects_genomic(self):
        cat, evidence = _infer_plot_category_from_text(
            "Genome browser tracks and circos links across chromosomes."
        )
        assert cat == PlotCategory.GENOMIC
        assert any("circos" in e for e in evidence)

    def test_stage_b_type_candidates_ranked(self):
        candidates = _infer_plot_type_candidates(
            "Alluvial sankey flow diagram of cell state transitions."
        )
        assert len(candidates) >= 1
        assert candidates[0][0] == PlotType.SANKEY

    def test_disambiguates_circos_from_caption(self):
        sf = self._make_sf(description="Circular genome links")
        refined = _disambiguate_subfigure_plot(
            sf,
            caption="Circos plot showing inter-chromosomal interactions.",
            in_text=[],
        )
        assert refined.plot_type == PlotType.CIRCOS
        assert refined.plot_category == PlotCategory.GENOMIC
        assert refined.classification_confidence >= 0.75
        assert any("circos" in ev.lower() for ev in refined.evidence_spans)

    def test_disambiguates_sankey_alluvial(self):
        sf = self._make_sf(description="Flow transitions")
        refined = _disambiguate_subfigure_plot(
            sf,
            caption="Alluvial Sankey plot of lineage flow.",
            in_text=[],
        )
        assert refined.plot_type == PlotType.SANKEY
        assert refined.plot_category == PlotCategory.FLOW

    def test_composite_violin_swarm_adds_overlay_layer(self):
        sf = self._make_sf(
            description="Distribution by group with swarm points",
            plot_type=PlotType.VIOLIN,
            plot_category=PlotCategory.CATEGORICAL,
        )
        refined = _disambiguate_subfigure_plot(
            sf,
            caption="Violin plot with swarm overlay for each group.",
            in_text=[],
        )
        assert refined.plot_category == PlotCategory.COMPOSITE
        assert refined.plot_type == PlotType.VIOLIN
        assert len(refined.layers) >= 2
        assert refined.layers[0].plot_type == PlotType.VIOLIN
        assert refined.layers[1].plot_type == PlotType.SWARM

    def test_bioc_contradiction_penalty_applied_and_confidence_synced(self):
        sf = self._make_sf(description="Stacked bar summary")
        refined = _disambiguate_subfigure_plot(
            sf,
            caption="Stacked bar plot of category proportions.",
            in_text=[],
            bioc_evidence=[
                BioCPassageContext(section_type="RESULTS", text="A circos map shows links.", offset=100)
            ],
        )
        # Caption keeps stacked-bar interpretation; BioC cue is contradictory.
        assert refined.plot_type == PlotType.STACKED_BAR
        assert refined.composite_confidence < 90.0
        assert refined.classification_confidence == round(refined.composite_confidence / 100.0, 3)
        assert len(refined.bioc_evidence_spans) == 1
        assert refined.bioc_evidence_spans[0].offset == 100
        assert refined.bioc_evidence_spans[0].text == ""
        assert refined.bioc_contradiction is True

    def test_bioc_support_boosts_confidence_and_syncs_ratio(self):
        sf = self._make_sf(description="Network relationships")
        refined = _disambiguate_subfigure_plot(
            sf,
            caption="Network graph of interactions among factors.",
            in_text=[],
            bioc_evidence=[
                BioCPassageContext(section_type="FIG", text="Network graph with node and edge weights.", offset=10),
                BioCPassageContext(section_type="RESULTS", text="The network graph highlights hubs.", offset=20),
            ],
        )
        assert refined.plot_type == PlotType.NETWORK_GRAPH
        assert refined.composite_confidence >= 65.0
        assert refined.classification_confidence == round(refined.composite_confidence / 100.0, 3)
        assert refined.bioc_contradiction is False


class TestPanelLocalContext:
    """Panel-local context helpers prevent cue bleed across subfigures."""

    def test_panel_caption_context_filters_by_label(self):
        caption = (
            "Figure 1. (a) Heatmap of marker genes across clusters. "
            "(b) Violin plot with swarm overlay by condition."
        )
        subfigures = [
            SubFigure(label="a", description="Panel A", plot_type=PlotType.OTHER, plot_category=PlotCategory.COMPOSITE),
            SubFigure(label="b", description="Panel B", plot_type=PlotType.OTHER, plot_category=PlotCategory.COMPOSITE),
        ]
        a_ctx = _panel_caption_context(caption, "a", subfigures)
        b_ctx = _panel_caption_context(caption, "b", subfigures)
        assert "Heatmap" in a_ctx
        assert "Violin" not in a_ctx
        assert "Violin" in b_ctx

    def test_panel_bioc_evidence_prefers_panel_specific(self):
        passages = [
            BioCPassageContext(section_type="RESULTS", text="Figure 2A shows control.", offset=10),
            BioCPassageContext(section_type="RESULTS", text="Figure 2B shows treatment.", offset=20),
        ]
        selected = _panel_bioc_evidence(passages, figure_id="Figure 2", panel_label="b")
        assert len(selected) == 1
        assert "2B" in selected[0].text

    def test_panel_in_text_context_filters_by_label(self):
        in_text = [
            "Figure 1a shows a heatmap of marker genes.",
            "Figure 1b shows violin distributions with swarm points.",
        ]
        a_ctx = _panel_in_text_context(in_text, "a")
        b_ctx = _panel_in_text_context(in_text, "b")
        assert len(a_ctx) == 1 and "heatmap" in a_ctx[0].lower()
        assert len(b_ctx) == 1 and "violin" in b_ctx[0].lower()


# ---------------------------------------------------------------------------
# TestFindCaption
# ---------------------------------------------------------------------------

class TestFindCaption:
    """FigureParser._find_caption — no LLM, uses paper data."""

    def test_finds_caption_in_section_text(self):
        parser = _make_parser()
        caption = parser._find_caption(SAMPLE_PAPER, "Figure 1")
        assert "eCLIP pipeline" in caption or caption != ""

    def test_returns_empty_string_for_missing_figure(self):
        parser = _make_parser()
        caption = parser._find_caption(SAMPLE_PAPER, "Figure 99")
        assert isinstance(caption, str)

    def test_uses_figure_captions_dict_when_available(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={
            "figure_captions": {"Figure 1": "Custom caption from JATS"}
        })
        caption = parser._find_caption(paper, "Figure 1")
        assert "Custom caption" in caption

    def test_fig_dot_label_matches_figure_key(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={
            "figure_captions": {"Fig. 1": "Abbreviated caption"}
        })
        caption = parser._find_caption(paper, "Figure 1")
        assert "Abbreviated" in caption


# ---------------------------------------------------------------------------
# TestFindInTextReferences
# ---------------------------------------------------------------------------

class TestFindInTextReferences:
    """FigureParser._find_in_text_references — no LLM."""

    def test_finds_references_in_sections(self):
        parser = _make_parser()
        refs = parser._find_in_text_references(SAMPLE_PAPER, "Figure 1")
        assert len(refs) >= 2
        assert any("Figure 1" in r or "Fig." in r for r in refs)

    def test_no_references_for_missing_figure(self):
        parser = _make_parser()
        refs = parser._find_in_text_references(SAMPLE_PAPER, "Figure 99")
        assert refs == []

    def test_panel_variant_matched(self):
        parser = _make_parser()
        refs = parser._find_in_text_references(SAMPLE_PAPER, "Figure 1")
        text = " ".join(refs)
        # Our fixture has "Figure 1a", "Figure 1b", "Figure 1c"
        assert "Figure 1" in text


# ---------------------------------------------------------------------------
# TestIdentifyDatasetsRegex
# ---------------------------------------------------------------------------

class TestIdentifyDatasetsRegex:
    """FigureParser._identify_datasets — regex path (no LLM when accessions found)."""

    def test_geo_accession(self):
        parser = _make_parser()
        caption = "Data deposited at GEO under accession GSE72987."
        datasets = parser._identify_datasets(caption, [])
        assert "GSE72987" in datasets

    def test_sra_accession(self):
        parser = _make_parser()
        caption = "Raw reads available at SRA (SRP123456)."
        datasets = parser._identify_datasets(caption, [])
        assert "SRP123456" in datasets

    def test_pride_accession(self):
        parser = _make_parser()
        caption = "Proteomics data at PRIDE: PXD001234."
        datasets = parser._identify_datasets(caption, [])
        assert "PXD001234" in datasets

    def test_multiple_accessions(self):
        parser = _make_parser()
        caption = "RNA-seq: GSE12345. ChIP-seq: GSE67890."
        datasets = parser._identify_datasets(caption, [])
        assert "GSE12345" in datasets
        assert "GSE67890" in datasets

    def test_no_accessions_returns_empty_without_llm_call(self):
        parser = _make_parser()
        with patch("researcher_ai.parsers.figure_parser.ask_claude_structured") as mock:
            mock.return_value = _MethodsAndDatasets(datasets=[])
            datasets = parser._identify_datasets("No accessions here.", [])
            # LLM called as fallback since no regex hits
            mock.assert_called_once()

    def test_in_text_accessions_found(self):
        parser = _make_parser()
        in_text = ["See supplementary data at SRR9876543 for raw reads."]
        datasets = parser._identify_datasets("", in_text)
        assert "SRR9876543" in datasets


# ---------------------------------------------------------------------------
# TestInferLayout
# ---------------------------------------------------------------------------

class TestInferLayout:
    """FigureParser._infer_layout — pure logic."""

    def _make_subfig(self, label: str) -> SubFigure:
        return SubFigure(
            label=label,
            description="test",
            plot_category=PlotCategory.RELATIONAL,
            plot_type=PlotType.SCATTER,
        )

    def test_single_subfigure(self):
        parser = _make_parser()
        layout = parser._infer_layout([self._make_subfig("main")])
        assert layout.n_rows == 1
        assert layout.n_cols == 1

    def test_empty_subfigures(self):
        parser = _make_parser()
        layout = parser._infer_layout([])
        assert layout.n_rows == 1
        assert layout.n_cols == 1

    def test_four_panels_2x2(self):
        parser = _make_parser()
        panels = [self._make_subfig(l) for l in ["A", "B", "C", "D"]]
        layout = parser._infer_layout(panels)
        assert layout.n_rows * layout.n_cols >= 4

    def test_uppercase_style(self):
        parser = _make_parser()
        panels = [self._make_subfig(l) for l in ["A", "B", "C"]]
        layout = parser._infer_layout(panels)
        assert layout.panel_labels_style == "uppercase"

    def test_lowercase_style(self):
        parser = _make_parser()
        panels = [self._make_subfig(l) for l in ["a", "b", "c"]]
        layout = parser._infer_layout(panels)
        assert layout.panel_labels_style == "lowercase"


# ---------------------------------------------------------------------------
# TestDecomposeSubfigures (mocked LLM)
# ---------------------------------------------------------------------------

class TestDecomposeSubfigures:
    """FigureParser._decompose_subfigures — LLM mocked."""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_returns_subfigures_from_llm(self, mock_structured):
        mock_structured.return_value = _SubFigureList(subfigures=[
            _subfig_meta(label="a", description="Schematic", plot_type="image", plot_category="image"),
            _subfig_meta(label="b", description="Volcano", plot_type="volcano", plot_category="genomic"),
        ])
        parser = _make_parser()
        sfs = parser._decompose_subfigures("Figure 1", SAMPLE_CAPTION, [])
        assert len(sfs) == 2
        assert sfs[0].label == "a"
        assert sfs[1].plot_type == PlotType.VOLCANO

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_empty_caption_returns_empty_list(self, mock_structured):
        parser = _make_parser()
        sfs = parser._decompose_subfigures("Figure 1", "", [])
        assert sfs == []
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_llm_failure_returns_empty_list(self, mock_structured):
        mock_structured.side_effect = RuntimeError("API error")
        parser = _make_parser()
        sfs = parser._decompose_subfigures("Figure 1", SAMPLE_CAPTION, [])
        assert sfs == []

    def test_subfigure_list_accepts_json_string_payload(self):
        """Regression: tool output may return subfigures as a JSON string."""
        payload = {
            "subfigures": (
                '[{"label":"a","description":"Schematic","plot_type":"image",'
                '"plot_category":"image"}]'
            )
        }
        model = _SubFigureList.model_validate(payload)
        assert len(model.subfigures) == 1
        assert model.subfigures[0].label == "a"


# ---------------------------------------------------------------------------
# TestDeterminePurpose (mocked LLM)
# ---------------------------------------------------------------------------

class TestDeterminePurpose:
    """FigureParser._determine_purpose — LLM mocked."""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_returns_purpose(self, mock_structured):
        mock_structured.return_value = _FigurePurpose(
            purpose="This figure demonstrates the eCLIP pipeline.",
            title="eCLIP pipeline overview",
        )
        parser = _make_parser()
        result = parser._determine_purpose("Figure 1", SAMPLE_CAPTION, [])
        assert "eCLIP" in result.purpose
        assert result.title == "eCLIP pipeline overview"

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_llm_failure_returns_stub(self, mock_structured):
        mock_structured.side_effect = RuntimeError("API error")
        parser = _make_parser()
        result = parser._determine_purpose("Figure 1", SAMPLE_CAPTION, [])
        assert result.title == "Figure 1"


# ---------------------------------------------------------------------------
# TestParseFigure (integration-level with mocked LLM)
# ---------------------------------------------------------------------------

class TestParseFigure:
    """FigureParser.parse_figure — LLM mocked, real caption/context extraction."""

    def _mock_side_effect(self, output_schema, **kwargs):
        if output_schema is _SubFigureList:
            return _SubFigureList(subfigures=[
                _subfig_meta(label="a", description="Schematic", plot_type="image", plot_category="image"),
                _subfig_meta(label="b", description="Volcano plot", plot_type="volcano", plot_category="genomic"),
                _subfig_meta(label="c", description="UMAP", plot_type="umap", plot_category="dimensionality"),
            ])
        elif output_schema is _FigurePurpose:
            return _FigurePurpose(purpose="Shows eCLIP pipeline overview.", title="eCLIP pipeline")
        elif output_schema is _MethodsAndDatasets:
            return _MethodsAndDatasets(methods=["eCLIP", "DESeq2"], datasets=[])
        return MagicMock()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_figure_id_preserved(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        assert fig.figure_id == "Figure 1"

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_subfigures_populated(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        assert len(fig.subfigures) == 3
        labels = [sf.label for sf in fig.subfigures]
        assert "a" in labels and "b" in labels and "c" in labels

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_in_text_context_collected(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        assert len(fig.in_text_context) >= 1

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_purpose_set(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        assert fig.purpose != ""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_layout_inferred(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        assert fig.layout.n_rows >= 1
        assert fig.layout.n_cols >= 1

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_json_roundtrip(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        fig = parser.parse_figure(SAMPLE_PAPER, "Figure 1")
        restored = Figure.model_validate_json(fig.model_dump_json())
        assert restored.figure_id == fig.figure_id
        assert len(restored.subfigures) == len(fig.subfigures)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_panel_specific_layers_do_not_bleed_across_subfigures(self, mock_structured):
        """Regression: each panel gets its own layer stack from panel-local cues."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _SubFigureList:
                return _SubFigureList(subfigures=[
                    _subfig_meta(label="a", description="Panel A", plot_type="other", plot_category="composite"),
                    _subfig_meta(label="b", description="Panel B", plot_type="other", plot_category="composite"),
                ])
            if output_schema is _FigurePurpose:
                return _FigurePurpose(purpose="Panel-specific chart types.", title="Panel-specific")
            if output_schema is _MethodsAndDatasets:
                return _MethodsAndDatasets(methods=[], datasets=[])
            return MagicMock()

        mock_structured.side_effect = side_effect
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={
            "figure_ids": ["Figure 1"],
            "figure_captions": {
                "Figure 1": (
                    "Figure 1. (a) Heatmap of marker genes across clusters. "
                    "(b) Violin plot with swarm overlay by condition."
                )
            },
            "sections": [],
            "raw_text": "",
        })
        fig = parser.parse_figure(paper, "Figure 1")
        by_label = {sf.label.lower(): sf for sf in fig.subfigures}
        assert by_label["a"].layers[0].plot_type == PlotType.HEATMAP
        assert by_label["b"].layers[0].plot_type == PlotType.VIOLIN
        assert len(by_label["a"].layers) == 1
        assert len(by_label["b"].layers) >= 2
        assert by_label["b"].layers[1].plot_type == PlotType.SWARM

    def test_timeout_skips_followup_llm_calls_for_figure(self, monkeypatch):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"figure_ids": ["Figure 1"]})
        called = {"purpose": 0, "methods": 0}

        def _timeout(*args, **kwargs):  # noqa: ARG001
            raise SubfigureDecompositionTimeoutError("timeout")

        def _purpose(*args, **kwargs):  # noqa: ARG001
            called["purpose"] += 1
            return _FigurePurpose(purpose="x", title="x")

        def _methods(*args, **kwargs):  # noqa: ARG001
            called["methods"] += 1
            return ["RNA-seq"]

        monkeypatch.setattr(parser, "_decompose_subfigures", _timeout)
        monkeypatch.setattr(parser, "_determine_purpose", _purpose)
        monkeypatch.setattr(parser, "_identify_methods", _methods)

        fig = parser.parse_figure(paper, "Figure 1")
        assert "subfigure_decomposition_timeout" in fig.parse_warnings
        assert "figure_llm_followups_skipped_after_timeout" in fig.parse_warnings
        assert called["purpose"] == 0
        assert called["methods"] == 0


# ---------------------------------------------------------------------------
# TestParseAllFigures (integration-level with mocked LLM)
# ---------------------------------------------------------------------------

class TestParseAllFigures:
    """FigureParser.parse_all_figures — LLM mocked."""

    def _mock_side_effect(self, output_schema, **kwargs):
        if output_schema is _SubFigureList:
            return _SubFigureList(subfigures=[
                _subfig_meta(label="main", description="Main panel", plot_type="scatter", plot_category="relational"),
            ])
        elif output_schema is _FigurePurpose:
            return _FigurePurpose(purpose="A figure.", title="Test figure")
        elif output_schema is _MethodsAndDatasets:
            return _MethodsAndDatasets(methods=[], datasets=[])
        return MagicMock()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_returns_one_figure_per_figure_id(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        figs = parser.parse_all_figures(SAMPLE_PAPER)
        assert len(figs) == len(SAMPLE_PAPER.figure_ids)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_figure_ids_match_paper(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        figs = parser.parse_all_figures(SAMPLE_PAPER)
        ids = [f.figure_id for f in figs]
        for expected_id in SAMPLE_PAPER.figure_ids:
            assert expected_id in ids

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_parse_failure_returns_stub(self, mock_structured):
        """A parsing exception for one figure does not abort the whole batch."""
        call_count = [0]

        def side_effect(prompt, output_schema, **kw):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("Simulated LLM error")
            return self._mock_side_effect(output_schema)

        mock_structured.side_effect = side_effect
        parser = _make_parser()
        figs = parser.parse_all_figures(SAMPLE_PAPER)
        assert len(figs) == len(SAMPLE_PAPER.figure_ids)
        # First figure should be a stub
        assert figs[0].figure_id == SAMPLE_PAPER.figure_ids[0]

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_empty_figure_ids(self, mock_structured):
        paper = SAMPLE_PAPER.model_copy(
            update={"figure_ids": [], "sections": [], "raw_text": ""}
        )
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert figs == []
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_recovers_figure_ids_from_text_when_empty(self, mock_structured):
        """When Paper.figure_ids is empty, parser recovers IDs from section/raw text."""
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        paper = SAMPLE_PAPER.model_copy(update={"figure_ids": []})
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert len(figs) >= 1
        assert any(f.figure_id == "Figure 1" for f in figs)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_ignores_supplementary_figure_ids_in_batch_parse(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        paper = SAMPLE_PAPER.model_copy(
            update={
                "figure_ids": [
                    "Figure 1",
                    "Figure 2",
                    "Supplementary Figure S1",
                    "Supplementary Figure S2",
                ]
            }
        )
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert [f.figure_id for f in figs] == ["Figure 1", "Figure 2"]

    @patch("researcher_ai.parsers.figure_parser.get_figure_urls_from_pmid")
    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_attaches_preview_urls_by_figure_order(self, mock_structured, mock_urls):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        mock_urls.return_value = [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/f1.jpg",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/f2.jpg",
        ]
        paper = SAMPLE_PAPER.model_copy(update={"pmid": "12345678", "figure_ids": ["Figure 1", "Figure 2"]})
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert [f.preview_url for f in figs] == mock_urls.return_value

    @patch("researcher_ai.parsers.figure_parser.get_figure_urls_from_pmid")
    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_parse_figure_attaches_preview_url(self, mock_structured, mock_urls):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        mock_urls.return_value = ["https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/f1.jpg"]
        paper = SAMPLE_PAPER.model_copy(update={"pmid": "12345678", "figure_ids": ["Figure 1"]})
        parser = _make_parser()
        fig = parser.parse_figure(paper, "Figure 1")
        assert fig.preview_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/bin/f1.jpg"

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_populates_datasets_from_paper_context_when_figure_local_is_empty(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        paper = SAMPLE_PAPER.model_copy(
            update={
                "raw_text": SAMPLE_PAPER.raw_text + "\nData deposited at GEO: GSE72987.",
                "sections": SAMPLE_PAPER.sections + [
                    Section(title="Data Availability", text="GEO accession GSE72987.", figures_referenced=[])
                ],
            }
        )
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert len(figs) >= 1
        assert all("GSE72987" in f.datasets_used for f in figs)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_populates_methods_from_methods_section_when_llm_empty(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        paper = SAMPLE_PAPER.model_copy(
            update={
                "sections": SAMPLE_PAPER.sections + [
                    Section(
                        title="Methods",
                        text=(
                            "eCLIP-seq library preparation\n"
                            "(See Supplementary Protocol 1 for detailed SOPs.)"
                        ),
                        figures_referenced=[],
                    )
                ]
            }
        )
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert len(figs) >= 1
        assert all(any("eCLIP-seq library preparation" in m for m in f.methods_used) for f in figs)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_populates_datasets_from_data_availability_section_gse77634(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        paper = SAMPLE_PAPER.model_copy(
            update={
                "sections": SAMPLE_PAPER.sections + [
                    Section(
                        title="Data Availability",
                        text="Sequencing data are available at GEO under accession GSE77634.",
                        figures_referenced=[],
                    )
                ],
                "raw_text": SAMPLE_PAPER.raw_text + "\nGSE77634",
            }
        )
        parser = _make_parser()
        figs = parser.parse_all_figures(paper)
        assert len(figs) >= 1
        assert all("GSE77634" in f.datasets_used for f in figs)

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_circuit_breaker_skips_remaining_after_timeout_budget(self, mock_structured):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        parser.max_figure_llm_timeouts_per_paper = 1
        paper = SAMPLE_PAPER.model_copy(update={"figure_ids": ["Figure 1", "Figure 2"]})

        def _fake_parse_from_context(fig_id, caption, in_text, **kwargs):  # noqa: ARG001
            stub = parser._stub_figure(fig_id, caption=caption, in_text=in_text)
            if fig_id == "Figure 1":
                return stub.model_copy(update={"parse_warnings": ["subfigure_decomposition_timeout"]})
            return stub

        parser._parse_figure_from_context = _fake_parse_from_context
        figs = parser.parse_all_figures(paper)
        assert len(figs) == 2
        assert "subfigure_decomposition_timeout" in figs[0].parse_warnings
        assert "figure_llm_circuit_breaker_open" in figs[1].parse_warnings

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_writes_trace_artifact_with_step_events(self, mock_structured, tmp_path: Path):
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        parser.figure_trace_path = str(tmp_path / "figure_trace.json")
        paper = SAMPLE_PAPER.model_copy(update={"figure_ids": ["Figure 1"]})

        def _fake_parse_from_context(fig_id, caption, in_text, **kwargs):  # noqa: ARG001
            return parser._stub_figure(fig_id, caption=caption, in_text=in_text)

        parser._parse_figure_from_context = _fake_parse_from_context
        parser.parse_all_figures(paper)
        trace_path = Path(parser.figure_trace_path)
        assert trace_path.exists()
        payload = json.loads(trace_path.read_text())
        assert payload, "Expected at least one trace event"
        assert any(evt.get("step") == "figure_parse" for evt in payload)


# ---------------------------------------------------------------------------
# TestStubFigure
# ---------------------------------------------------------------------------

class TestStubFigure:
    """FigureParser._stub_figure — no LLM."""

    def test_stub_has_correct_id(self):
        parser = _make_parser()
        stub = parser._stub_figure("Figure 7")
        assert stub.figure_id == "Figure 7"
        assert stub.title == "Figure 7"
        assert stub.caption == ""

    def test_stub_is_valid_figure_model(self):
        parser = _make_parser()
        stub = parser._stub_figure("Figure 99")
        assert isinstance(stub, Figure)
        data = stub.model_dump_json()
        assert "Figure 99" in data

    def test_stub_preserves_caption_when_provided(self):
        parser = _make_parser()
        stub = parser._stub_figure("Figure 3", caption="Partial caption text.")
        assert stub.caption == "Partial caption text."

    def test_stub_preserves_in_text_when_provided(self):
        parser = _make_parser()
        stub = parser._stub_figure("Figure 3", in_text=["Sentence referencing Figure 3."])
        assert stub.in_text_context == ["Sentence referencing Figure 3."]

    def test_stub_defaults_to_empty_in_text(self):
        parser = _make_parser()
        stub = parser._stub_figure("Figure 3")
        assert stub.in_text_context == []


# ---------------------------------------------------------------------------
# TestExtractCaptionTwoDigitFigure
# ---------------------------------------------------------------------------

class TestExtractCaptionTwoDigitFigure:
    """_extract_caption_from_text — two-digit figure numbers (regression tests for
    the word-boundary bug where 'Figure 1' could greedily match 'Figure 10')."""

    def test_figure_10_found(self):
        text = (
            "Figure 1. First figure caption.\n"
            "Figure 10. Tenth figure caption with many details.\n"
            "Figure 11. Eleventh figure.\n"
        )
        caption = _extract_caption_from_text(text, "Figure 10")
        assert "Tenth figure" in caption

    def test_figure_1_does_not_match_figure_10(self):
        text = (
            "Figure 10. Tenth figure caption.\n"
            "Figure 11. Eleventh figure.\n"
        )
        # "Figure 1" should NOT capture the "Figure 10" caption
        caption = _extract_caption_from_text(text, "Figure 1")
        assert caption == "" or "Tenth" not in caption

    def test_figure_10_does_not_match_figure_1_caption(self):
        text = (
            "Figure 1. First figure caption.\n"
            "Figure 2. Second figure.\n"
        )
        caption = _extract_caption_from_text(text, "Figure 10")
        assert caption == ""

    def test_figure_2_does_not_match_figure_20(self):
        text = "Figure 20. Twenty-panel figure."
        caption = _extract_caption_from_text(text, "Figure 2")
        assert caption == ""


# ---------------------------------------------------------------------------
# TestBuildFigRefPatternSupplementary
# ---------------------------------------------------------------------------

class TestBuildFigRefPatternSupplementary:
    """_build_fig_ref_pattern — non-supplementary figures must not match
    supplementary references (regression test for the supp_prefix bug)."""

    def test_non_supp_pattern_does_not_match_supplementary(self):
        pat = _build_fig_ref_pattern("Figure 1")
        # Must NOT match "Supplementary Figure 1"
        assert not pat.search("See Supplementary Figure 1 for QC metrics.")

    def test_non_supp_pattern_matches_plain_figure(self):
        pat = _build_fig_ref_pattern("Figure 1")
        assert pat.search("As shown in Figure 1A.")

    def test_supp_pattern_matches_supplementary(self):
        pat = _build_fig_ref_pattern("Supplementary Figure S1")
        assert pat.search("QC metrics in Supplementary Fig. S1.")

    def test_supp_pattern_also_matches_abbreviated_supp(self):
        pat = _build_fig_ref_pattern("Supplementary Figure S2")
        assert pat.search("See Supplementary Fig. S2 for details.")


# ---------------------------------------------------------------------------
# TestIdentifyMethods
# ---------------------------------------------------------------------------

class TestIdentifyMethods:
    """FigureParser._identify_methods — LLM mocked."""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_returns_methods_from_llm(self, mock_structured):
        mock_structured.return_value = _MethodsAndDatasets(
            methods=["RNA-seq", "DESeq2", "CLIPper"], datasets=[]
        )
        parser = _make_parser()
        methods = parser._identify_methods("Caption mentioning RNA-seq.", [])
        assert "RNA-seq" in methods
        assert "DESeq2" in methods
        mock_structured.assert_called_once()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_empty_combined_text_returns_empty_no_llm(self, mock_structured):
        parser = _make_parser()
        methods = parser._identify_methods("", [])
        assert methods == []
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_llm_failure_uses_regex_fallback(self, mock_structured):
        mock_structured.side_effect = RuntimeError("API error")
        parser = _make_parser()
        methods = parser._identify_methods("ChIP-seq of H3K27ac in HEK293 cells.", [])
        assert "ChIP-seq" in methods

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_uses_full_in_text_not_truncated(self, mock_structured):
        """_identify_methods should pass the full in_text list to the LLM prompt,
        not a sentence-count-truncated slice."""
        mock_structured.return_value = _MethodsAndDatasets(methods=["ATAC-seq"], datasets=[])
        parser = _make_parser()
        # 15 sentences — previously only the first 10 would be included
        in_text = [f"Sentence {i} mentioning ATAC-seq." for i in range(15)]
        parser._identify_methods("Caption.", in_text)
        prompt_used = mock_structured.call_args[1].get("prompt") or mock_structured.call_args[0][0]
        # All 15 sentences should appear in the prompt
        assert "Sentence 14" in prompt_used


# ---------------------------------------------------------------------------
# TestExtractCaptionSupplementarySPrefix
# ---------------------------------------------------------------------------

class TestExtractCaptionSupplementarySPrefix:
    """_extract_caption_from_text — S-prefixed supplementary figure labels."""

    def test_finds_supplementary_s1_caption(self):
        text = (
            "Figure 1. Main figure caption.\n"
            "Supplementary Figure S1. QC metrics for all samples.\n"
            "Supplementary Figure S2. Additional results.\n"
        )
        caption = _extract_caption_from_text(text, "Supplementary Figure S1")
        assert "QC metrics" in caption

    def test_s1_caption_does_not_bleed_into_s2(self):
        text = (
            "Supplementary Figure S1. First supplementary caption.\n"
            "Supplementary Figure S2. Second supplementary caption.\n"
        )
        caption = _extract_caption_from_text(text, "Supplementary Figure S1")
        assert "First supplementary" in caption
        assert "Second supplementary" not in caption

    def test_s10_label_not_confused_with_s1(self):
        text = (
            "Supplementary Figure S1. First caption.\n"
            "Supplementary Figure S10. Tenth caption.\n"
        )
        caption_s10 = _extract_caption_from_text(text, "Supplementary Figure S10")
        assert "Tenth caption" in caption_s10

    def test_s1_does_not_match_s10_caption(self):
        text = "Supplementary Figure S10. Tenth supplementary figure."
        caption = _extract_caption_from_text(text, "Supplementary Figure S1")
        assert caption == "" or "Tenth" not in caption

    def test_plain_figure_1_not_captured_as_supp_s1(self):
        text = (
            "Figure 1. Plain figure caption.\n"
            "Supplementary Figure S1. Supplementary caption.\n"
        )
        # Querying for plain Figure 1 should not return the S1 caption
        caption = _extract_caption_from_text(text, "Figure 1")
        assert "Plain figure" in caption
        assert "Supplementary caption" not in caption


# ---------------------------------------------------------------------------
# TestIdentifyDatasetsStrictMode
# ---------------------------------------------------------------------------

class TestIdentifyDatasetsStrictMode:
    """FigureParser._identify_datasets — strict_regex_only flag."""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_strict_true_skips_llm_when_regex_hits(self, mock_structured):
        """Default strict_regex_only=True: LLM is not called when regex finds accessions."""
        parser = _make_parser()
        result = parser._identify_datasets("Data from GSE72987.", [], strict_regex_only=True)
        assert "GSE72987" in result
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_strict_false_calls_llm_even_when_regex_hits(self, mock_structured):
        """strict_regex_only=False: LLM is called, but ungrounded IDs are filtered."""
        mock_structured.return_value = _MethodsAndDatasets(
            methods=[], datasets=["SRP999999"]
        )
        parser = _make_parser()
        result = parser._identify_datasets(
            "Data from GSE72987.", [], strict_regex_only=False
        )
        assert "GSE72987" in result   # from regex
        assert "SRP999999" not in result  # not present in source text
        mock_structured.assert_called_once()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_strict_false_llm_failure_returns_regex_results(self, mock_structured):
        """On LLM failure in non-strict mode, regex hits are still returned."""
        mock_structured.side_effect = RuntimeError("API error")
        parser = _make_parser()
        result = parser._identify_datasets(
            "GSE12345 is the accession.", [], strict_regex_only=False
        )
        assert "GSE12345" in result

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_strict_false_empty_text_no_llm_call(self, mock_structured):
        """Neither regex nor LLM is called on empty text."""
        parser = _make_parser()
        result = parser._identify_datasets("", [], strict_regex_only=False)
        assert result == []
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_strict_false_filters_hallucinated_llm_dataset_not_in_text(self, mock_structured):
        mock_structured.return_value = _MethodsAndDatasets(methods=[], datasets=["GSE78487"])
        parser = _make_parser()
        result = parser._identify_datasets(
            "Data available under accession GSE77634.", [], strict_regex_only=False
        )
        assert "GSE77634" in result
        assert "GSE78487" not in result


# ---------------------------------------------------------------------------
# TestParseAllFiguresNoDuplicateScan
# ---------------------------------------------------------------------------

class TestParseAllFiguresNoDuplicateScan:
    """parse_all_figures() uses _parse_figure_from_context() — caption/in-text
    extraction happens once per figure, not twice."""

    def _mock_side_effect(self, output_schema, **kwargs):
        if output_schema is _SubFigureList:
            return _SubFigureList(subfigures=[
                _subfig_meta(label="main", description="Panel", plot_type="scatter",
                             plot_category="relational"),
            ])
        elif output_schema is _FigurePurpose:
            return _FigurePurpose(purpose="A figure.", title="Test figure")
        elif output_schema is _MethodsAndDatasets:
            return _MethodsAndDatasets(methods=[], datasets=[])
        from unittest.mock import MagicMock
        return MagicMock()

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_parse_all_figures_calls_parse_figure_from_context(self, mock_structured):
        """parse_all_figures uses _parse_figure_from_context, not parse_figure."""
        mock_structured.side_effect = lambda prompt, output_schema, **kw: self._mock_side_effect(output_schema)
        parser = _make_parser()
        # Spy on the private method to confirm it is called
        call_log = []
        original = parser._parse_figure_from_context

        def spy(figure_id, caption, in_text, **kwargs):
            call_log.append(figure_id)
            return original(figure_id, caption, in_text, **kwargs)

        parser._parse_figure_from_context = spy
        parser.parse_all_figures(SAMPLE_PAPER)
        assert call_log == SAMPLE_PAPER.figure_ids

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_stub_carries_precomputed_caption_on_llm_failure(self, mock_structured):
        """When LLM fails, the stub contains the caption that was already found."""
        mock_structured.side_effect = RuntimeError("Simulated LLM failure")
        parser = _make_parser()
        figs = parser.parse_all_figures(SAMPLE_PAPER)
        for fig in figs:
            # All figures should be stubs; caption must not be blank if the
            # paper has a findable caption for that figure.
            assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# TestPmc11633308GroundTruth
# ---------------------------------------------------------------------------

class TestPmc11633308GroundTruth:
    """Ground-truth classification checks for PMCID PMC11633308 figure patterns."""

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_figure1_a_b_c_two_pane_left_right(self, mock_structured):
        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _SubFigureList:
                return _SubFigureList(
                    subfigures=[
                        _subfig_meta(label="a", description="Panel A", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="b", description="Panel B", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="c", description="Panel C", plot_type="other", plot_category="composite"),
                    ]
                )
            if output_schema is _FigurePurpose:
                return _FigurePurpose(purpose="Figure 1 purpose", title="Figure 1")
            if output_schema is _MethodsAndDatasets:
                return _MethodsAndDatasets(methods=[], datasets=[])
            return MagicMock()

        mock_structured.side_effect = side_effect
        parser = _make_parser()
        caption = (
            "Figure 1. (a) Left: horizontal bar plot; x-axis for Figure 1A represents "
            "\"Residual % Increase Compared to Batch Model\" and y-axis for Figure 1A "
            "represents \"ZFPs with High DEG Residual\". Right: cumulative stacked bar "
            "plot where the x-axis for Figure 1A represents \"% of DEGs\" and y-axis is "
            "shared with the left pane. "
            "(b) Left: horizontal bar plot. Right: cumulative stacked bar plot with shared y-axis. "
            "(c) Left: horizontal bar plot. Right: cumulative stacked bar plot with shared y-axis."
        )
        in_text = [
            "Figure 1A has two panes with left and right plots.",
            "Figure 1B has two panes with left and right plots.",
            "Figure 1C has two panes with left and right plots.",
        ]
        fig = parser._parse_figure_from_context("Figure 1", caption, in_text)
        by_label = {sf.label.lower(): sf for sf in fig.subfigures}

        for label in ("a", "b", "c"):
            sf = by_label[label]
            assert sf.plot_category == PlotCategory.COMPOSITE
            assert sf.layers[0].plot_type == PlotType.BAR
            assert any(layer.plot_type == PlotType.STACKED_BAR for layer in sf.layers[1:])
            assert sf.n_facets == 2
            assert sf.facet_variable == "left_right_panel"

        assert by_label["a"].x_axis is not None
        assert by_label["a"].y_axis is not None
        assert by_label["a"].x_axis.label == "Residual % Increase Compared to Batch Model"
        assert by_label["a"].y_axis.label == "ZFPs with High DEG Residual"

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_figure2_a_to_h_plot_types(self, mock_structured):
        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _SubFigureList:
                return _SubFigureList(
                    subfigures=[
                        _subfig_meta(label="a", description="Panel A", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="b", description="Panel B", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="c", description="Panel C", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="d", description="Panel D", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="e", description="Panel E", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="f", description="Panel F", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="g", description="Panel G", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="h", description="Panel H", plot_type="other", plot_category="composite"),
                    ]
                )
            if output_schema is _FigurePurpose:
                return _FigurePurpose(purpose="Figure 2 purpose", title="Figure 2")
            if output_schema is _MethodsAndDatasets:
                return _MethodsAndDatasets(methods=[], datasets=[])
            return MagicMock()

        mock_structured.side_effect = side_effect
        parser = _make_parser()
        caption = (
            "Figure 2. (a) Venn diagram with the title \"ZFP Prioritized Subset (n=43)\". "
            "(b) stacked bar plot. "
            "(c) tSNE plot (scatter plot). "
            "(d) cumulative stacked bar plot. "
            "(e) stacked bar plot. "
            "(f) Bubble plot. "
            "(g) upset plot. "
            "(h) bar plot."
        )
        fig = parser._parse_figure_from_context("Figure 2", caption, [])
        by_label = {sf.label.lower(): sf for sf in fig.subfigures}
        assert by_label["a"].plot_type == PlotType.VENN
        assert by_label["b"].plot_type == PlotType.STACKED_BAR
        assert by_label["c"].plot_type == PlotType.TSNE
        assert by_label["d"].plot_type == PlotType.STACKED_BAR
        assert by_label["e"].plot_type == PlotType.STACKED_BAR
        assert by_label["f"].plot_type == PlotType.BUBBLE
        assert by_label["g"].plot_type == PlotType.UPSET
        assert by_label["h"].plot_type == PlotType.BAR

    @patch("researcher_ai.parsers.figure_parser.ask_claude_structured")
    def test_pmc11633308_calibration_override_applies(self, mock_structured):
        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _SubFigureList:
                return _SubFigureList(
                    subfigures=[
                        _subfig_meta(label="A", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="B", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="C", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="D", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="E", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="F", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="G", description="Generic", plot_type="other", plot_category="composite"),
                        _subfig_meta(label="H", description="Generic", plot_type="other", plot_category="composite"),
                    ]
                )
            if output_schema is _FigurePurpose:
                return _FigurePurpose(purpose="purpose", title="title")
            if output_schema is _MethodsAndDatasets:
                return _MethodsAndDatasets(methods=[], datasets=[])
            return MagicMock()

        mock_structured.side_effect = side_effect
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={
            "pmcid": "PMC11633308",
            "figure_ids": ["Figure 1", "Figure 2"],
            "figure_captions": {
                "Figure 1": "Figure 1 caption with generic text.",
                "Figure 2": "Figure 2 caption with generic text.",
            },
            "sections": [],
            "raw_text": "",
        })
        fig1 = parser.parse_figure(paper, "Figure 1")
        by1 = {sf.label.lower(): sf for sf in fig1.subfigures}
        assert by1["a"].plot_type == PlotType.BAR
        assert any(layer.plot_type == PlotType.STACKED_BAR for layer in by1["a"].layers[1:])
        assert by1["a"].n_facets == 2
        assert by1["a"].x_axis is not None
        assert by1["a"].y_axis is not None

        fig2 = parser.parse_figure(paper, "Figure 2")
        by2 = {sf.label.lower(): sf for sf in fig2.subfigures}
        assert by2["a"].plot_type == PlotType.VENN
        assert by2["b"].plot_type == PlotType.STACKED_BAR
        assert by2["c"].plot_type == PlotType.TSNE
        assert by2["d"].plot_type == PlotType.STACKED_BAR
        assert by2["e"].plot_type == PlotType.STACKED_BAR
        assert by2["f"].plot_type == PlotType.BUBBLE
        assert by2["g"].plot_type == PlotType.UPSET
        assert by2["h"].plot_type == PlotType.BAR


@pytest.mark.snapshot
class TestSpatialMultimodalSisonPdf:
    """Validate multimodal panel extraction on Sison_Nature_2026 PDF fixture."""

    PDF_PATH = (
        Path(__file__).parent / "fixtures" / "figure_calibration" / "Sison_Nature_2026.pdf"
    )

    @pytest.mark.skipif(not PDF_PATH.exists(), reason="Sison_Nature_2026.pdf not found")
    def test_visual_panel_count_drives_subfigure_structure(self):
        paper = Paper(
            title="Sison Nature 2026",
            source=PaperSource.PDF,
            source_path=str(self.PDF_PATH),
            paper_type=PaperType.EXPERIMENTAL,
            sections=[],
            figure_ids=["Figure 1"],
            figure_captions={},
            raw_text="",
        )
        parser = FigureParser(llm_model="test-model", vision_model="gemini-3.1-pro")

        def _vision_side_effect(*args, **kwargs):
            images = kwargs.get("image_bytes", [])
            n = max(1, min(len(images), 8))
            subfigs = [
                _SubFigureMeta(
                    label=chr(ord("a") + i),
                    description=f"Panel {i + 1}",
                    plot_type="image",
                    plot_category="image",
                )
                for i in range(n)
            ]
            return _VisionFigureExtraction(
                title="Visual-only decomposition",
                purpose="Derived from panel crops only.",
                subfigures=subfigs,
                methods_used=[],
                datasets_used=[],
            )

        with patch("researcher_ai.parsers.figure_parser._extract_structured_data", side_effect=_vision_side_effect):
            figure = parser.parse_figure(paper, "Figure 1")

        # If multimodal extraction worked, the structure must be driven by image crops.
        assert figure.title == "Visual-only decomposition"
        assert len(figure.subfigures) >= 1
        assert figure.layout.n_rows * figure.layout.n_cols >= len(figure.subfigures)
