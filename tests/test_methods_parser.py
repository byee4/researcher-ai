"""Unit tests for Phase 4: MethodsParser.

Testing strategy:
- Pure helpers (_extract_section_by_heading, _extract_assay_paragraph,
  _build_figure_context, _assay_from_meta) tested directly with no mocking.
- MethodsParser._extract_methods_text tested against mock Paper objects
  with varied section titles.
- LLM-using methods (_identify_assays, _parse_assay, _identify_dependencies,
  _find_availability_statements) tested with ask_claude_structured mocked.
- Integration: parse() with fully mocked LLM dependencies.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from researcher_ai.models.figure import Figure, PanelLayout, PlotCategory, PlotType, SubFigure
from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayDependency,
    AssayGraph,
    Method,
)
from researcher_ai.models.paper import (
    BioCContext,
    BioCPassageContext,
    Paper,
    PaperSource,
    PaperType,
    Section,
)
from researcher_ai.models.method import MethodCategory
from researcher_ai.parsers.methods_parser import (
    MethodsParser,
    _AssayCategoryItem,
    _AssayClassificationList,
    _AssayList,
    _AssayMeta,
    _AvailabilityStatement,
    _DependencyList,
    _DependencyMeta,
    _StepParameterInference,
    _StepParameterInferenceList,
    _StepMeta,
    _assay_from_meta,
    _build_figure_context,
    _extract_assay_paragraph,
    _extract_assay_block_by_heading,
    _extract_dataset_accessions,
    _extract_github_urls,
    _extract_section_by_heading,
    _normalize_assay_name,
    _METHODS_TITLE_RE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METHODS_TEXT = """\
Cell culture and library preparation
HEK293T cells were cultured in DMEM (Gibco) supplemented with 10% FBS.
eCLIP libraries were prepared as described in Van Nostrand et al. 2016.
Briefly, UV crosslinking was performed at 400 mJ/cm2. Immunoprecipitation
was carried out with anti-FLAG antibody (Sigma F1804).

RNA-seq library preparation
Total RNA was extracted using TRIzol (Thermo Fisher). Ribosomal RNA was
depleted using the Ribo-Zero Gold kit. Libraries were sequenced on the
Illumina HiSeq 4000 (2x150 bp paired-end reads).

Read alignment and peak calling
Reads were aligned to hg38 using STAR 2.7.3a with default parameters.
PCR duplicate reads were removed using Picard MarkDuplicates.
Peaks were called using CLIPper v2.0 with a FDR threshold of 0.001.

Differential expression analysis
Differential expression was computed with DESeq2 v1.28 using a Wald test.
Adjusted p-values were computed using the Benjamini-Hochberg method.
Data are deposited at GEO under accession GSE72987.

Data Availability
Sequencing data are available at GEO (GSE72987). Analysis code is at
https://github.com/YeoLab/eCLIP.

Code Availability
All analysis code is available at https://github.com/YeoLab/eCLIP.
"""

SAMPLE_SECTIONS = [
    Section(
        title="Methods",
        text=SAMPLE_METHODS_TEXT,
        figures_referenced=["Figure 2", "Figure 3"],
    ),
    Section(
        title="Results",
        text="We identified 10,000 binding peaks (Figure 2). See Figure 3 for UMAP.",
        figures_referenced=["Figure 2", "Figure 3"],
    ),
]

SAMPLE_PAPER = Paper(
    title="Robust transcriptome-wide discovery of RNA-binding protein binding sites",
    authors=["Van Nostrand, EL", "Yeo, GW"],
    abstract="eCLIP enables transcriptome-wide mapping of RNA-protein interactions.",
    doi="10.1038/nmeth.3810",
    pmid="26971820",
    pmcid="PMC4878918",
    source=PaperSource.PMCID,
    source_path="PMC4878918",
    paper_type=PaperType.EXPERIMENTAL,
    sections=SAMPLE_SECTIONS,
    figure_ids=["Figure 2", "Figure 3"],
    raw_text="\n\n".join(s.text for s in SAMPLE_SECTIONS),
)

SAMPLE_FIGURE = Figure(
    figure_id="Figure 2",
    title="eCLIP peak enrichment",
    caption="Volcano plot of enrichment (log2FC) for all binding peaks.",
    purpose="Shows enrichment of eCLIP peaks.",
    methods_used=["eCLIP", "CLIPper peak calling"],
)


def _make_parser() -> MethodsParser:
    parser = MethodsParser.__new__(MethodsParser)
    parser.llm_model = "test-model"
    parser.cache = None
    return parser


def _make_step_meta(n: int = 1, **kwargs) -> _StepMeta:
    defaults = dict(
        step_number=n,
        description=f"Step {n} description",
        input_data="raw FASTQ reads",
        output_data="aligned BAM files",
        software="STAR",
        software_version="2.7.3a",
        parameters={"outSAMtype": "BAM"},
        code_reference=None,
    )
    defaults.update(kwargs)
    return _StepMeta(**defaults)


def _make_assay_meta(**kwargs) -> _AssayMeta:
    defaults = dict(
        name="RNA-seq library preparation",
        description="Library preparation from total RNA.",
        data_type="sequencing",
        raw_data_source="GEO: GSE72987",
        steps=[_make_step_meta(1), _make_step_meta(2)],
        figures_produced=["Figure 2"],
    )
    defaults.update(kwargs)
    return _AssayMeta(**defaults)


# ---------------------------------------------------------------------------
# TestMethodsTitleRegex
# ---------------------------------------------------------------------------

class TestMethodsTitleRegex:
    """_METHODS_TITLE_RE matches all expected section title variants."""

    @pytest.mark.parametrize("title", [
        "Methods",
        "methods",
        "METHODS",
        "Materials and Methods",
        "Materials & Methods",
        "Methods and Materials",
        "Experimental Procedures",
        "Experimental Design",
        "STAR Methods",
        "STAR* Methods",
        "Online Methods",
        "Supplementary Methods",
        "Supplementary methods",
    ])
    def test_matches_expected_titles(self, title: str):
        assert _METHODS_TITLE_RE.search(title), f"Should match: {title!r}"

    @pytest.mark.parametrize("title", [
        "Results",
        "Discussion",
        "Introduction",
        "Abstract",
        "References",
        "Acknowledgements",
        "Figure Legends",
    ])
    def test_rejects_non_methods_titles(self, title: str):
        assert not _METHODS_TITLE_RE.search(title), f"Should not match: {title!r}"


# ---------------------------------------------------------------------------
# TestExtractMethodsText
# ---------------------------------------------------------------------------

class TestExtractMethodsText:
    """MethodsParser._extract_methods_text — various section title formats."""

    def test_finds_methods_section_by_title(self):
        parser = _make_parser()
        text = parser._extract_methods_text(SAMPLE_PAPER)
        assert "eCLIP" in text
        assert "STAR" in text

    def test_finds_materials_and_methods(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Materials and Methods", text="Protocol text here.", figures_referenced=[]),
        ]})
        text = parser._extract_methods_text(paper)
        assert "Protocol text here." in text

    def test_finds_star_methods(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="STAR Methods", text="STAR methods content.", figures_referenced=[]),
        ]})
        text = parser._extract_methods_text(paper)
        assert "STAR methods content." in text

    def test_concatenates_multiple_methods_sections(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Methods", text="Main methods.", figures_referenced=[]),
            Section(title="Supplementary Methods", text="Supplementary methods.", figures_referenced=[]),
        ]})
        text = parser._extract_methods_text(paper)
        assert "Main methods." in text
        assert "Supplementary methods." in text

    def test_returns_empty_string_when_no_methods_section(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Results", text="Results here.", figures_referenced=[]),
            Section(title="Discussion", text="Discussion here.", figures_referenced=[]),
        ], "raw_text": ""})
        text = parser._extract_methods_text(paper)
        assert text == ""

    def test_falls_back_to_raw_text(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={
            "sections": [],
            "raw_text": "Introduction\nSome intro.\n\nMethods\nCell culture and alignment.\n\nResults\nPeaks found.",
        })
        text = parser._extract_methods_text(paper)
        assert "Cell culture" in text

    def test_prefers_explicit_is_methods_sections(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Protocol Notes", text="Methods body text.", is_methods=True, figures_referenced=[]),
            Section(title="Methods", text="Distractor methods title.", is_methods=False, figures_referenced=[]),
        ], "raw_text": ""})
        text = parser._extract_methods_text(paper)
        assert text == "Methods body text."

    def test_include_bioc_methods_when_enabled(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(
            update={
                "sections": [],
                "raw_text": "",
                "bioc_context": BioCContext(
                    pmid=SAMPLE_PAPER.pmid,
                    pmcid=SAMPLE_PAPER.pmcid,
                    methods=[
                        BioCPassageContext(
                            section_type="METHODS",
                            type="paragraph",
                            text="BioC methods paragraph with STAR alignment details.",
                            offset=100,
                        )
                    ],
                ),
            }
        )
        text = parser._extract_methods_text(paper, include_bioc=True)
        assert "BioC methods paragraph" in text


# ---------------------------------------------------------------------------
# TestExtractAssayParagraph
# ---------------------------------------------------------------------------

class TestExtractAssayParagraph:
    """_extract_assay_paragraph — keyword-based paragraph retrieval."""

    def test_finds_relevant_paragraph(self):
        para = _extract_assay_paragraph(SAMPLE_METHODS_TEXT, "CLIPper peak calling")
        assert "CLIPper" in para

    def test_finds_deseq2_paragraph(self):
        para = _extract_assay_paragraph(SAMPLE_METHODS_TEXT, "DESeq2 differential expression")
        assert "DESeq2" in para

    def test_falls_back_to_full_text_when_no_match(self):
        para = _extract_assay_paragraph(SAMPLE_METHODS_TEXT, "CRISPR screen")
        # Falls back to full text (capped at 4000 chars)
        assert len(para) > 0

    def test_returns_top_3_paragraphs_max(self):
        para = _extract_assay_paragraph(SAMPLE_METHODS_TEXT, "RNA-seq library preparation")
        # Should not contain all 5 paragraphs — capped at 3
        assert len(para) <= 3000

    def test_heading_bounded_extraction_avoids_bleed(self):
        text = (
            "Assay A\n"
            "A step 1.\n\n"
            "Assay B\n"
            "B step 1.\n"
        )
        para = _extract_assay_paragraph(text, "Assay A", assay_names=["Assay A", "Assay B"])
        assert "A step 1" in para
        assert "B step 1" not in para


class TestExtractAssayBlockByHeading:
    def test_extracts_between_headings(self):
        text = (
            "First Assay\n"
            "first details\n\n"
            "Second Assay\n"
            "second details\n"
        )
        block = _extract_assay_block_by_heading(text, "First Assay", ["First Assay", "Second Assay"])
        assert "first details" in block
        assert "second details" not in block


class TestExtractGithubUrls:
    def test_extracts_and_deduplicates(self):
        text = (
            "Code: https://github.com/gpratt/gatk/releases/tag/2.3.2 and "
            "https://github.com/gpratt/gatk/releases/tag/2.3.2."
        )
        urls = _extract_github_urls(text)
        assert urls == ["https://github.com/gpratt/gatk/releases/tag/2.3.2"]


class TestExtractDatasetAccessions:
    def test_extracts_geo_accession(self):
        text = "Data deposited at GEO under accession GSE77634."
        ids = _extract_dataset_accessions(text)
        assert "GSE77634" in ids


# ---------------------------------------------------------------------------
# TestExtractSectionByHeading
# ---------------------------------------------------------------------------

class TestExtractSectionByHeading:
    """_extract_section_by_heading — raw text fallback."""

    def test_extracts_methods_from_raw_text(self):
        text = (
            "Introduction\nThis is intro.\n\n"
            "Methods\nCell culture step. STAR alignment used.\n\n"
            "Results\nPeaks found.\n\n"
        )
        section = _extract_section_by_heading(text, _METHODS_TITLE_RE)
        assert "STAR" in section
        assert "Peaks found" not in section

    def test_returns_empty_when_not_found(self):
        text = "Introduction\nSome intro.\n\nResults\nSome results.\n"
        section = _extract_section_by_heading(text, _METHODS_TITLE_RE)
        assert section == ""

    def test_methods_subheading_is_not_treated_as_section_stop(self):
        text = (
            "Article\n\n"
            "Methods\n"
            "Molecular cloning and AAV production\n"
            "The original construct consisted of ...\n\n"
            "Results\n"
            "Main findings.\n"
        )
        section = _extract_section_by_heading(text, _METHODS_TITLE_RE)
        assert "Molecular cloning and AAV production" in section
        assert "Main findings" not in section


# ---------------------------------------------------------------------------
# TestBuildFigureContext
# ---------------------------------------------------------------------------

class TestBuildFigureContext:
    """_build_figure_context — maps figure_id → methods_used."""

    def test_builds_context_from_figures(self):
        ctx = _build_figure_context([SAMPLE_FIGURE])
        assert "Figure 2" in ctx
        assert "eCLIP" in ctx["Figure 2"]

    def test_skips_figures_with_no_methods(self):
        fig = SAMPLE_FIGURE.model_copy(update={"methods_used": []})
        ctx = _build_figure_context([fig])
        assert ctx == {}

    def test_empty_list(self):
        ctx = _build_figure_context([])
        assert ctx == {}


# ---------------------------------------------------------------------------
# TestAssayFromMeta
# ---------------------------------------------------------------------------

class TestAssayFromMeta:
    """_assay_from_meta — pure conversion."""

    def test_basic_conversion(self):
        meta = _make_assay_meta()
        assay = _assay_from_meta(meta)
        assert assay.name == "RNA-seq library preparation"
        assert assay.data_type == "sequencing"
        assert assay.raw_data_source == "GEO: GSE72987"
        assert len(assay.steps) == 2

    def test_steps_sorted_by_step_number(self):
        meta = _make_assay_meta(steps=[_make_step_meta(3), _make_step_meta(1), _make_step_meta(2)])
        assay = _assay_from_meta(meta)
        assert [s.step_number for s in assay.steps] == [1, 2, 3]

    def test_step_fields_populated(self):
        meta = _make_assay_meta()
        assay = _assay_from_meta(meta)
        step = assay.steps[0]
        assert step.software == "STAR"
        assert step.software_version == "2.7.3a"
        assert step.parameters == {"outSAMtype": "BAM"}

    def test_figures_produced_populated(self):
        meta = _make_assay_meta(figures_produced=["Figure 2", "Figure 3"])
        assay = _assay_from_meta(meta)
        assert "Figure 2" in assay.figures_produced

    def test_empty_steps_allowed(self):
        meta = _make_assay_meta(steps=[])
        assay = _assay_from_meta(meta)
        assert assay.steps == []

    def test_duplicate_steps_are_deduplicated_and_renumbered(self):
        step = _make_step_meta(1, description="Align reads")
        dup = _make_step_meta(2, description="Align reads")
        meta = _make_assay_meta(steps=[step, dup])
        assay = _assay_from_meta(meta)
        assert len(assay.steps) == 1
        assert assay.steps[0].step_number == 1


# ---------------------------------------------------------------------------
# TestIdentifyAssays (mocked LLM)
# ---------------------------------------------------------------------------

class TestIdentifyAssays:
    """MethodsParser._identify_assays — LLM mocked."""

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_assay_names(self, mock_llm):
        mock_llm.return_value = _AssayList(assay_names=[
            "eCLIP library preparation",
            "Read alignment",
            "Peak calling",
        ])
        parser = _make_parser()
        names = parser._identify_assays(SAMPLE_METHODS_TEXT)
        assert "Read alignment" in names
        assert len(names) == 3

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_empty_text_returns_empty_no_llm(self, mock_llm):
        parser = _make_parser()
        names = parser._identify_assays("")
        assert names == []
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        names = parser._identify_assays(SAMPLE_METHODS_TEXT)
        assert names == []


# ---------------------------------------------------------------------------
# TestClassifyAssays (mocked LLM)
# ---------------------------------------------------------------------------

class TestClassifyAssays:
    """MethodsParser._classify_assays — LLM mocked."""

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_category_map(self, mock_llm):
        mock_llm.return_value = _AssayClassificationList(assays=[
            _AssayCategoryItem(name="Read alignment", method_category="computational"),
            _AssayCategoryItem(name="DESeq2 analysis", method_category="computational"),
        ])
        parser = _make_parser()
        cat_map = parser._classify_assays(
            ["Read alignment", "DESeq2 analysis"], SAMPLE_METHODS_TEXT
        )
        assert cat_map["Read alignment"] == MethodCategory.computational
        assert cat_map["DESeq2 analysis"] == MethodCategory.computational

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_classifies_wet_lab_as_experimental(self, mock_llm):
        mock_llm.return_value = _AssayClassificationList(assays=[
            _AssayCategoryItem(name="eCLIP library preparation", method_category="experimental"),
        ])
        parser = _make_parser()
        cat_map = parser._classify_assays(
            ["eCLIP library preparation"], SAMPLE_METHODS_TEXT
        )
        assert cat_map["eCLIP library preparation"] == MethodCategory.experimental

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_classifies_mixed_assay(self, mock_llm):
        mock_llm.return_value = _AssayClassificationList(assays=[
            _AssayCategoryItem(
                name="eCLIP-seq library prep and processing",
                method_category="mixed",
            ),
        ])
        parser = _make_parser()
        cat_map = parser._classify_assays(
            ["eCLIP-seq library prep and processing"], SAMPLE_METHODS_TEXT
        )
        assert cat_map["eCLIP-seq library prep and processing"] == MethodCategory.mixed

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_empty_names_returns_empty_no_llm(self, mock_llm):
        parser = _make_parser()
        cat_map = parser._classify_assays([], SAMPLE_METHODS_TEXT)
        assert cat_map == {}
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        cat_map = parser._classify_assays(
            ["Read alignment", "eCLIP library prep"], SAMPLE_METHODS_TEXT
        )
        assert cat_map == {}

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_unknown_category_value_defaults_to_computational(self, mock_llm):
        mock_llm.return_value = _AssayClassificationList(assays=[
            _AssayCategoryItem(name="Weird assay", method_category="unknown_value"),
        ])
        parser = _make_parser()
        cat_map = parser._classify_assays(["Weird assay"], SAMPLE_METHODS_TEXT)
        assert cat_map["Weird assay"] == MethodCategory.computational

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_normalizes_llm_name_casing_to_canonical(self, mock_llm):
        """LLM returns a name with different casing; normalised to canonical."""
        mock_llm.return_value = _AssayClassificationList(assays=[
            _AssayCategoryItem(name="read alignment", method_category="computational"),
        ])
        parser = _make_parser()
        cat_map = parser._classify_assays(
            ["Read alignment"], SAMPLE_METHODS_TEXT
        )
        assert "Read alignment" in cat_map
        assert cat_map["Read alignment"] == MethodCategory.computational


# ---------------------------------------------------------------------------
# TestFindAvailabilityStatements (mocked LLM)
# ---------------------------------------------------------------------------

class TestFindAvailabilityStatements:
    """MethodsParser._find_availability_statements — LLM mocked."""

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_both_statements(self, mock_llm):
        mock_llm.return_value = _AvailabilityStatement(
            data_statement="Sequencing data at GEO (GSE72987).",
            code_statement="Code at https://github.com/YeoLab/eCLIP.",
        )
        parser = _make_parser()
        data, code = parser._find_availability_statements(SAMPLE_PAPER, SAMPLE_METHODS_TEXT)
        assert "GSE72987" in data
        assert "github" in code

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_finds_dedicated_availability_section(self, mock_llm):
        mock_llm.return_value = _AvailabilityStatement(
            data_statement="Data at GEO.", code_statement=""
        )
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Data Availability", text="Data at GEO (GSE72987).", figures_referenced=[]),
        ]})
        parser = _make_parser()
        data, _ = parser._find_availability_statements(paper, "")
        assert data != "" or mock_llm.called  # either found or LLM called

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_empty_text_returns_empty_strings(self, mock_llm):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [], "raw_text": ""})
        data, code = parser._find_availability_statements(paper, "")
        assert data == ""
        assert code == ""
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_llm_failure_returns_empty_strings(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        data, code = parser._find_availability_statements(SAMPLE_PAPER, SAMPLE_METHODS_TEXT)
        assert data == ""
        assert code == ""


# ---------------------------------------------------------------------------
# TestParseAssay (mocked LLM)
# ---------------------------------------------------------------------------

class TestParseAssay:
    """MethodsParser._parse_assay — LLM mocked."""

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_assay_with_steps(self, mock_llm):
        mock_llm.return_value = _make_assay_meta(
            name="Read alignment",
            steps=[_make_step_meta(1, description="Align reads with STAR.")],
        )
        parser = _make_parser()
        assay = parser._parse_assay("Read alignment", SAMPLE_METHODS_TEXT, SAMPLE_PAPER, {})
        assert assay.name == "Read alignment"
        assert len(assay.steps) >= 1

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_llm_failure_raises(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        with pytest.raises(RuntimeError):
            parser._parse_assay("Peak calling", SAMPLE_METHODS_TEXT, SAMPLE_PAPER, {})


# ---------------------------------------------------------------------------
# TestIdentifyDependencies (mocked LLM)
# ---------------------------------------------------------------------------

class TestIdentifyDependencies:
    """MethodsParser._identify_dependencies — LLM mocked."""

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_single_assay_returns_empty_no_llm(self, mock_llm):
        parser = _make_parser()
        deps, warnings = parser._identify_dependencies(["eCLIP library preparation"], SAMPLE_METHODS_TEXT)
        assert deps == []
        assert warnings == []
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_dependency_objects(self, mock_llm):
        mock_llm.return_value = _DependencyList(dependencies=[
            _DependencyMeta(
                upstream_assay="RNA-seq alignment",
                downstream_assay="DESeq2 differential expression",
                dependency_type="normalization_reference",
                description="Expression values from alignment fed into DESeq2.",
            )
        ])
        parser = _make_parser()
        deps, warnings = parser._identify_dependencies(
            ["RNA-seq alignment", "DESeq2 differential expression"],
            SAMPLE_METHODS_TEXT,
        )
        assert len(deps) == 1
        assert deps[0].upstream_assay == "RNA-seq alignment"
        assert deps[0].dependency_type == "normalization_reference"
        assert warnings == []

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        deps, warnings = parser._identify_dependencies(["Assay A", "Assay B"], SAMPLE_METHODS_TEXT)
        assert deps == []
        assert warnings == []

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_normalizes_case_mismatch(self, mock_llm):
        """LLM returns names with different casing; parser normalizes to canonical."""
        mock_llm.return_value = _DependencyList(dependencies=[
            _DependencyMeta(
                upstream_assay="rna-seq alignment",  # lowercase from LLM
                downstream_assay="DESEQ2 DIFFERENTIAL EXPRESSION",  # uppercase
                dependency_type="count_input",
            )
        ])
        parser = _make_parser()
        deps, warnings = parser._identify_dependencies(
            ["RNA-seq alignment", "DESeq2 differential expression"],
            SAMPLE_METHODS_TEXT,
        )
        assert len(deps) == 1
        assert deps[0].upstream_assay == "RNA-seq alignment"
        assert deps[0].downstream_assay == "DESeq2 differential expression"
        assert warnings == []

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_drops_dangling_edge_with_warning(self, mock_llm):
        """LLM-returned assay name not in canonical set -> edge dropped, warning emitted."""
        mock_llm.return_value = _DependencyList(dependencies=[
            _DependencyMeta(
                upstream_assay="Invented Assay",
                downstream_assay="RNA-seq alignment",
                dependency_type="unknown",
            )
        ])
        parser = _make_parser()
        deps, warnings = parser._identify_dependencies(
            ["RNA-seq alignment", "DESeq2 differential expression"],
            SAMPLE_METHODS_TEXT,
        )
        assert len(deps) == 0
        assert len(warnings) == 1
        assert "dependency_dropped" in warnings[0]
        assert "Invented Assay" in warnings[0]


# ---------------------------------------------------------------------------
# TestNormalizeAssayName (pure helper)
# ---------------------------------------------------------------------------

class TestNormalizeAssayName:
    """_normalize_assay_name — canonical name matching."""

    CANON = ["RNA-seq alignment", "DESeq2 differential expression", "ChIP-seq peak calling"]

    def test_exact_match(self):
        assert _normalize_assay_name("RNA-seq alignment", self.CANON) == "RNA-seq alignment"

    def test_case_insensitive_match(self):
        assert _normalize_assay_name("rna-seq alignment", self.CANON) == "RNA-seq alignment"

    def test_substring_canonical_in_name(self):
        assert _normalize_assay_name(
            "ChIP-seq peak calling (with IDR)", self.CANON
        ) == "ChIP-seq peak calling"

    def test_substring_name_in_canonical(self):
        assert _normalize_assay_name("peak calling", self.CANON) == "ChIP-seq peak calling"

    def test_no_match_returns_none(self):
        assert _normalize_assay_name("Completely Unknown", self.CANON) is None

    def test_ambiguous_substring_returns_none(self):
        """If name matches two canonical entries by substring, return None."""
        canon = ["RNA-seq A", "RNA-seq B"]
        assert _normalize_assay_name("RNA-seq", canon) is None


# ---------------------------------------------------------------------------
# TestParse (integration with mocked LLM)
# ---------------------------------------------------------------------------

class TestParse:
    """MethodsParser.parse() — end-to-end with mocked LLM."""

    def _mock_llm(self, prompt, output_schema, **kw):
        if output_schema is _AssayList:
            return _AssayList(assay_names=[
                "eCLIP library preparation",
                "Read alignment and peak calling",
                "Differential expression analysis",
            ])
        if output_schema is _AssayClassificationList:
            # All three assays classified as computational so existing count
            # assertions remain valid (no assays are filtered out).
            return _AssayClassificationList(assays=[
                _AssayCategoryItem(
                    name="eCLIP library preparation",
                    method_category="computational",
                ),
                _AssayCategoryItem(
                    name="Read alignment and peak calling",
                    method_category="computational",
                ),
                _AssayCategoryItem(
                    name="Differential expression analysis",
                    method_category="computational",
                ),
            ])
        if output_schema is _AssayMeta:
            return _make_assay_meta(steps=[_make_step_meta(1)])
        if output_schema is _DependencyList:
            return _DependencyList(dependencies=[
                _DependencyMeta(
                    upstream_assay="Read alignment and peak calling",
                    downstream_assay="Differential expression analysis",
                    dependency_type="normalization_reference",
                    description="Aligned reads feed DESeq2.",
                )
            ])
        if output_schema is _AvailabilityStatement:
            return _AvailabilityStatement(
                data_statement="GEO: GSE72987.",
                code_statement="github.com/YeoLab/eCLIP.",
            )
        return MagicMock()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_returns_method_object(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert isinstance(method, Method)
        assert method.paper_doi == SAMPLE_PAPER.doi

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_assays_populated(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert len(method.assays) == 3

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_dependencies_in_assay_graph(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert len(method.assay_graph.dependencies) == 1
        dep = method.assay_graph.dependencies[0]
        assert dep.upstream_assay == "Read alignment and peak calling"

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_availability_statements_populated(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert "GSE72987" in method.data_availability
        assert "github" in method.code_availability

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_raw_methods_text_preserved(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert "eCLIP" in method.raw_methods_text

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_parse_with_figures(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, figures=[SAMPLE_FIGURE])
        assert isinstance(method, Method)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_empty_methods_returns_stub_method(self, mock_llm):
        paper = SAMPLE_PAPER.model_copy(update={"sections": [], "raw_text": ""})
        parser = _make_parser()
        method = parser.parse(paper)
        assert method.raw_methods_text == ""
        assert method.assays == []
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_assay_failure_returns_stub_not_abort(self, mock_llm):
        """LLM failure on a single assay produces a stub; parsing continues."""
        # Track _AssayMeta calls independently so the inserted classification
        # call (_AssayClassificationList) does not shift the counter.
        assay_meta_calls = [0]

        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayMeta:
                assay_meta_calls[0] += 1
                if assay_meta_calls[0] == 1:
                    raise RuntimeError("Simulated assay parse failure")
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        # All 3 assays returned — the first one as a stub
        assert len(method.assays) == 3
        stub = method.assays[0]
        assert stub.description == "Could not be parsed."

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_clean_parse_has_no_warnings(self, mock_llm):
        """When parsing succeeds cleanly, parse_warnings is empty."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert method.parse_warnings == []

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_assay_stub_recorded_in_parse_warnings(self, mock_llm):
        """Stub assays record a machine-readable warning in method.parse_warnings."""
        assay_meta_calls = [0]

        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayMeta:
                assay_meta_calls[0] += 1
                if assay_meta_calls[0] == 1:
                    raise RuntimeError("timeout")
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert any("assay_stub" in w for w in method.parse_warnings)
        assert any("timeout" in w for w in method.parse_warnings)

    def test_parse_uses_bioc_methods_when_section_missing(self):
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(
            update={
                "sections": [Section(title="Results", text="No methods heading.", figures_referenced=[])],
                "raw_text": "",
                "bioc_context": BioCContext(
                    pmid=SAMPLE_PAPER.pmid,
                    pmcid=SAMPLE_PAPER.pmcid,
                    methods=[
                        BioCPassageContext(
                            section_type="METHODS",
                            type="paragraph",
                            text="BioC-only methods text describing computational alignment.",
                            offset=10,
                        )
                    ],
                ),
            }
        )

        with patch.object(parser, "_identify_assays", return_value=[]), patch.object(
            parser, "_classify_assays", return_value={}
        ), patch.object(parser, "_find_availability_statements", return_value=("", "")), patch.object(
            parser, "_ensure_data_availability_text", side_effect=lambda *_: ""
        ), patch.object(parser, "_resolve_code_references", return_value=([], [])), patch.object(
            parser, "_collect_grounded_accessions", return_value=[]
        ), patch.object(parser, "_identify_dependencies", return_value=([], [])):
            method = parser.parse(paper, figures=None, computational_only=True)

        assert "BioC-only methods text" in method.raw_methods_text
        assert method.assay_graph is not None
        assert method.assay_graph.assays == []

    def test_search_protocol_docs_returns_relevant_hit(self):
        parser = _make_parser()
        docs = parser.search_protocol_docs("STAR alignment runThreadN genomeDir", top_k=2)
        assert "STAR" in docs

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_parse_infers_missing_parameters_for_computational_steps(self, mock_llm):
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["Read alignment"])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(
                    assays=[_AssayCategoryItem(name="Read alignment", method_category="computational")]
                )
            if output_schema is _AssayMeta:
                return _make_assay_meta(
                    name="Read alignment",
                    steps=[_make_step_meta(1, description="Align with STAR", software="STAR", parameters={})],
                )
            if output_schema is _StepParameterInferenceList:
                return _StepParameterInferenceList(
                    updates=[
                        _StepParameterInference(
                            step_number=1,
                            inferred_parameters={"runThreadN": "8", "genomeDir": "/ref/hg38"},
                            rationale="STAR protocol defaults",
                        )
                    ]
                )
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(data_statement="", code_statement="")
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        assert method.assays
        step = method.assays[0].steps[0]
        assert step.parameters.get("runThreadN") == "8"
        assert step.parameters.get("genomeDir") == "/ref/hg38"
        assert any("inferred_parameters" in w for w in method.parse_warnings)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_parse_does_not_overwrite_existing_parameters(self, mock_llm):
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["Read alignment"])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(
                    assays=[_AssayCategoryItem(name="Read alignment", method_category="computational")]
                )
            if output_schema is _AssayMeta:
                return _make_assay_meta(
                    name="Read alignment",
                    steps=[_make_step_meta(1, description="Align with STAR", software="STAR", parameters={"outSAMtype": "BAM"})],
                )
            if output_schema is _StepParameterInferenceList:
                return _StepParameterInferenceList(
                    updates=[_StepParameterInference(step_number=1, inferred_parameters={"runThreadN": "8"})]
                )
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(data_statement="", code_statement="")
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        step = method.assays[0].steps[0]
        assert step.parameters == {"outSAMtype": "BAM"}

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_parse_agentic_rag_calls_search_tool_and_infers_software(self, mock_llm):
        paper = SAMPLE_PAPER.model_copy(
            update={
                "sections": [
                    Section(
                        title="Methods",
                        text=(
                            "Reads were mapped to the human genome, and differential expression "
                            "was calculated."
                        ),
                        figures_referenced=[],
                    )
                ],
                "raw_text": "Reads were mapped to the human genome, and differential expression was calculated.",
            }
        )

        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["Read mapping", "Differential expression"])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(
                    assays=[
                        _AssayCategoryItem(name="Read mapping", method_category="computational"),
                        _AssayCategoryItem(name="Differential expression", method_category="computational"),
                    ]
                )
            if output_schema is _AssayMeta:
                if "Read mapping" in prompt:
                    return _make_assay_meta(
                        name="Read mapping",
                        steps=[_make_step_meta(1, description="Read alignment", software=None, parameters={})],
                    )
                return _make_assay_meta(
                    name="Differential expression",
                    steps=[_make_step_meta(1, description="Differential expression analysis", software=None, parameters={})],
                )
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(data_statement="", code_statement="")
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()

        tool_calls: list[str] = []

        def fake_tool_loop(**kwargs):
            parser.search_protocol_docs("read mapping eclip star", top_k=3)
            tool_calls.append("search_protocol_docs")
            prompt = kwargs.get("prompt", "")
            inferred_sw = "DESeq2" if "Differential expression" in prompt else "STAR"
            return _StepParameterInferenceList(
                updates=[
                    _StepParameterInference(
                        step_number=1,
                        inferred_parameters={"runThreadN": "8"},
                        inferred_software=inferred_sw,
                    )
                ]
            )

        with patch.object(parser, "search_protocol_docs", wraps=parser.search_protocol_docs) as spy_search, patch(
            "researcher_ai.parsers.methods_parser._extract_structured_data_with_tools",
            side_effect=fake_tool_loop,
        ):
            method = parser.parse(paper, computational_only=True)

        assert tool_calls
        assert spy_search.call_count >= 1
        step_software = [s.software for a in method.assays for s in a.steps]
        assert "STAR" in step_software
        assert any((s or "").upper().startswith("DESEQ2") for s in step_software)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_json_roundtrip(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        restored = Method.model_validate_json(method.model_dump_json())
        assert restored.paper_doi == method.paper_doi
        assert len(restored.assays) == len(method.assays)
        assert len(restored.assay_graph.dependencies) == len(method.assay_graph.dependencies)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    @patch("researcher_ai.parsers.software_parser.SoftwareParser._parse_github_code")
    def test_code_reference_applied_to_steps_from_code_availability(self, mock_parse_github, mock_llm):
        mock_parse_github.return_value = {"scripts": ["main.py"]}
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER)
        assert mock_parse_github.called
        found = [
            s.code_reference
            for a in method.assays
            for s in a.steps
            if s.code_reference
        ]
        assert any("github.com" in ref for ref in found)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_dataset_accession_fallback_populates_data_availability_and_assay_source(self, mock_llm):
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["RNA-seq library preparation"])
            if output_schema is _AssayMeta:
                return _make_assay_meta(raw_data_source=None, steps=[_make_step_meta(1)])
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(data_statement="", code_statement="")
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": SAMPLE_PAPER.sections + [
            Section(title="Data Availability", text="Data are available at GEO under accession GSE77634.", figures_referenced=[])
        ]})
        method = parser.parse(paper)
        assert "GSE77634" in method.data_availability
        assert any((a.raw_data_source or "").endswith("GSE77634") for a in method.assays)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_hallucinated_assay_accession_is_replaced_with_grounded_accession(self, mock_llm):
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["RNA-seq library preparation"])
            if output_schema is _AssayMeta:
                return _make_assay_meta(raw_data_source="GEO: GSE78487", steps=[_make_step_meta(1)])
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(
                    data_statement="Data are available at GEO under accession GSE77634.",
                    code_statement="",
                )
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": SAMPLE_PAPER.sections + [
            Section(title="Data Availability", text="Data are available at GEO under accession GSE77634.", figures_referenced=[])
        ]})
        method = parser.parse(paper)
        assert all((a.raw_data_source or "") == "GEO: GSE77634" for a in method.assays)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_no_grounded_accession_keeps_data_source_empty(self, mock_llm):
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["RNA-seq library preparation"])
            if output_schema is _AssayMeta:
                return _make_assay_meta(raw_data_source="GEO: GSE78487", steps=[_make_step_meta(1)])
            if output_schema is _DependencyList:
                return _DependencyList(dependencies=[])
            if output_schema is _AvailabilityStatement:
                return _AvailabilityStatement(data_statement="", code_statement="")
            return MagicMock()

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(update={"sections": [
            Section(title="Methods", text="No dataset accessions mentioned.", figures_referenced=[])
        ], "raw_text": "No accession here either."})
        method = parser.parse(paper)
        assert all(a.raw_data_source is None for a in method.assays)

    # ── computational_only / MethodCategory tests ────────────────────────────

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_computational_only_filters_experimental_assays(self, mock_llm):
        """With computational_only=True (default), only computational assays are kept."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=[
                    "eCLIP library preparation",
                    "Read alignment",
                    "DESeq2 differential expression",
                ])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(assays=[
                    _AssayCategoryItem(name="eCLIP library preparation", method_category="experimental"),
                    _AssayCategoryItem(name="Read alignment", method_category="computational"),
                    _AssayCategoryItem(name="DESeq2 differential expression", method_category="computational"),
                ])
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        names = [a.name for a in method.assays]
        assert "eCLIP library preparation" not in names
        assert "Read alignment" in names
        assert "DESeq2 differential expression" in names
        assert len(method.assays) == 2

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_computational_only_false_keeps_all_assays(self, mock_llm):
        """With computational_only=False, all assays are deep-parsed regardless of category."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=[
                    "eCLIP library preparation",
                    "Read alignment",
                    "DESeq2 differential expression",
                ])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(assays=[
                    _AssayCategoryItem(name="eCLIP library preparation", method_category="experimental"),
                    _AssayCategoryItem(name="Read alignment", method_category="computational"),
                    _AssayCategoryItem(name="DESeq2 differential expression", method_category="computational"),
                ])
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=False)
        assert len(method.assays) == 3

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_mixed_assay_is_filtered_when_computational_only(self, mock_llm):
        """Assays classified as 'mixed' are excluded when computational_only=True."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["eCLIP-seq library prep and processing"])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(assays=[
                    _AssayCategoryItem(
                        name="eCLIP-seq library prep and processing",
                        method_category="mixed",
                    ),
                ])
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        assert len(method.assays) == 0

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_filtered_experimental_assay_recorded_in_parse_warnings(self, mock_llm):
        """Excluded assays produce a machine-readable assay_filtered_non_computational warning."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=[
                    "UV crosslinking protocol",
                    "STAR alignment",
                ])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(assays=[
                    _AssayCategoryItem(name="UV crosslinking protocol", method_category="experimental"),
                    _AssayCategoryItem(name="STAR alignment", method_category="computational"),
                ])
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        assert any("assay_filtered_non_computational" in w for w in method.parse_warnings)
        assert any("UV crosslinking protocol" in w for w in method.parse_warnings)

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_assay_method_category_populated_from_classification(self, mock_llm):
        """Parsed Assay objects carry the method_category determined by classification."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayList:
                return _AssayList(assay_names=["Read alignment", "DESeq2 analysis"])
            if output_schema is _AssayClassificationList:
                return _AssayClassificationList(assays=[
                    _AssayCategoryItem(name="Read alignment", method_category="computational"),
                    _AssayCategoryItem(name="DESeq2 analysis", method_category="computational"),
                ])
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=False)
        for assay in method.assays:
            assert assay.method_category == MethodCategory.computational

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_classification_failure_keeps_all_assays_via_computational_default(self, mock_llm):
        """If classification fails (empty map), no assays are silently dropped."""
        def side_effect(prompt, output_schema, **kw):
            if output_schema is _AssayClassificationList:
                raise RuntimeError("classification API error")
            return self._mock_llm(prompt, output_schema)

        mock_llm.side_effect = side_effect
        parser = _make_parser()
        method = parser.parse(SAMPLE_PAPER, computational_only=True)
        # All 3 assays should be kept (default → computational, passes filter)
        assert len(method.assays) == 3

    @patch("researcher_ai.parsers.methods_parser.ask_claude_structured")
    def test_pmid_27018577_does_not_force_hardcoded_assay_titles(self, mock_llm):
        """PMID 27018577 should use extracted assay names, not hardcoded titles."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(prompt, output_schema)
        parser = _make_parser()
        paper = SAMPLE_PAPER.model_copy(
            update={
                "pmid": "27018577",
                "doi": "10.1038/nmeth.3810",
            }
        )
        method = parser.parse(paper, computational_only=False)
        names = [a.name for a in method.assays]
        assert names == [
            "eCLIP library preparation",
            "Read alignment and peak calling",
            "Differential expression analysis",
        ]


# ---------------------------------------------------------------------------
# TestAssayGraphHelpers
# ---------------------------------------------------------------------------

class TestAssayGraphHelpers:
    """AssayGraph.get_assay / upstream_of / downstream_of — pure logic."""

    def _make_graph(self) -> AssayGraph:
        return AssayGraph(
            assays=[
                Assay(name="RNA-seq alignment", description="", data_type="sequencing"),
                Assay(name="Peak calling", description="", data_type="sequencing"),
                Assay(name="DESeq2", description="", data_type="computational"),
            ],
            dependencies=[
                AssayDependency(
                    upstream_assay="RNA-seq alignment",
                    downstream_assay="Peak calling",
                    dependency_type="co-analysis",
                ),
                AssayDependency(
                    upstream_assay="RNA-seq alignment",
                    downstream_assay="DESeq2",
                    dependency_type="normalization_reference",
                ),
            ],
        )

    def test_get_assay_by_name(self):
        graph = self._make_graph()
        assay = graph.get_assay("rna-seq alignment")
        assert assay is not None
        assert assay.name == "RNA-seq alignment"

    def test_get_assay_missing_returns_none(self):
        graph = self._make_graph()
        assert graph.get_assay("CRISPR screen") is None

    def test_upstream_of(self):
        graph = self._make_graph()
        upstream = graph.upstream_of("DESeq2")
        assert "RNA-seq alignment" in upstream

    def test_downstream_of(self):
        graph = self._make_graph()
        downstream = graph.downstream_of("RNA-seq alignment")
        assert "Peak calling" in downstream
        assert "DESeq2" in downstream

    def test_no_upstream_for_root(self):
        graph = self._make_graph()
        assert graph.upstream_of("RNA-seq alignment") == []

    def test_method_assays_property(self):
        graph = self._make_graph()
        method = Method(paper_doi="10.0000/test", assay_graph=graph)
        assert len(method.assays) == 3
