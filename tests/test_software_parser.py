"""Unit tests for Phase 6: SoftwareParser.

Testing strategy:
- KNOWN_TOOLS registry lookup tested directly (no mocking needed).
- Environment generation (_build_environment) tested with hand-crafted Software
  objects — no LLM, no network.
- License classification (_check_open_source) tested against registry entries
  and with mocked LLM for unknown tools.
- Open-source alternative (_find_alternative) tested for known closed-source
  tools and with mocked LLM fallback.
- parse_from_method() tested with a mock Method object carrying AnalysisStep
  records, verifying deduplication and full Software output.
- parse_from_text() tested with mocked _extract_mentions_from_text.
- _identify_tool() tested with both registry and LLM paths (LLM mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from researcher_ai.models.method import AnalysisStep, Assay, AssayGraph, Method
from researcher_ai.models.software import Command, Environment, LicenseType, Software
from researcher_ai.parsers.software_parser import (
    SoftwareParser,
    _AlternativeDecision,
    _CommandExtraction,
    _CommandMeta,
    _LicenseDecision,
    _SoftwareMention,
    _SoftwareMentionList,
    _ToolMeta,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_METHODS_TEXT = """\
Reads were aligned to hg38 using STAR 2.7.3a with default parameters.
PCR duplicates were removed using Picard MarkDuplicates v2.27.
Differential expression was computed with DESeq2 v1.28 using a Wald test.
Single-cell data were processed with Cell Ranger 6.0 (10x Genomics).
Data were visualised with ggplot2 (R package).
"""


def _make_parser() -> SoftwareParser:
    parser = SoftwareParser.__new__(SoftwareParser)
    parser.llm_model = "test-model"
    parser.cache = None
    return parser


def _make_tool_meta(**kwargs) -> _ToolMeta:
    defaults = dict(
        name="MyTool",
        description="A useful analysis tool.",
        language="Python",
        github_repo="author/my-tool",
        bioconda_package="my-tool",
        pypi_package=None,
        cran_package=None,
        license_type="open_source",
        open_source_alternative=None,
        docker_image=None,
    )
    defaults.update(kwargs)
    return _ToolMeta(**defaults)


def _make_method_with_steps(steps_per_assay: list[list[dict]]) -> Method:
    """Build a Method with the given analysis steps."""
    assays = []
    for i, step_dicts in enumerate(steps_per_assay):
        steps = [AnalysisStep(**s) for s in step_dicts]
        assays.append(Assay(
            name=f"Assay {i + 1}",
            description="Test assay.",
            data_type="sequencing",
            steps=steps,
        ))
    graph = AssayGraph(assays=assays)
    return Method(paper_doi="10.0000/test", assay_graph=graph)


# ---------------------------------------------------------------------------
# TestRegistryLookup
# ---------------------------------------------------------------------------

class TestRegistryLookup:
    """SoftwareParser._lookup_registry — direct KNOWN_TOOLS access."""

    def test_exact_match_star(self):
        parser = _make_parser()
        entry = parser._lookup_registry("STAR")
        assert entry is not None
        assert entry["bioconda"] == "star"
        assert entry["license"] == "open_source"

    def test_exact_match_deseq2(self):
        parser = _make_parser()
        entry = parser._lookup_registry("DESeq2")
        assert entry is not None
        assert entry["language"] == "R"

    def test_case_insensitive_match(self):
        parser = _make_parser()
        entry = parser._lookup_registry("deseq2")
        assert entry is not None
        assert "bioconda" in entry

    def test_cellranger_is_closed_source(self):
        parser = _make_parser()
        entry = parser._lookup_registry("cellranger")
        assert entry is not None
        assert entry["license"] == "closed_source"
        assert "alternative" in entry

    def test_cell_ranger_space_variant(self):
        """'Cell Ranger' (with space) should also resolve."""
        parser = _make_parser()
        entry = parser._lookup_registry("Cell Ranger")
        assert entry is not None
        assert entry["license"] == "closed_source"

    def test_unknown_tool_returns_none(self):
        parser = _make_parser()
        assert parser._lookup_registry("NonExistentTool9999") is None

    def test_samtools_has_github(self):
        parser = _make_parser()
        entry = parser._lookup_registry("samtools")
        assert entry["github"] == "samtools/samtools"


# ---------------------------------------------------------------------------
# TestBuildFromRegistry
# ---------------------------------------------------------------------------

class TestBuildFromRegistry:
    """SoftwareParser._build_from_registry — no LLM needed."""

    def test_builds_software_from_star_entry(self):
        parser = _make_parser()
        entry = parser.KNOWN_TOOLS["STAR"]
        sw = parser._build_from_registry("STAR", "2.7.3a", "Align with STAR", entry)
        assert sw.name == "STAR"
        assert sw.version == "2.7.3a"
        assert sw.license_type == LicenseType.OPEN_SOURCE
        assert sw.bioconda_package == "star"
        assert sw.github_repo == "alexdobin/STAR"
        assert "github.com/alexdobin/STAR" in sw.source_url

    def test_closed_source_sets_alternative(self):
        parser = _make_parser()
        entry = parser.KNOWN_TOOLS["cellranger"]
        sw = parser._build_from_registry("cellranger", "6.0", "", entry)
        assert sw.license_type == LicenseType.CLOSED_SOURCE
        assert sw.open_source_alternative is not None
        assert "STARsolo" in sw.open_source_alternative or "alevin" in sw.open_source_alternative

    def test_no_version_none(self):
        parser = _make_parser()
        entry = parser.KNOWN_TOOLS["samtools"]
        sw = parser._build_from_registry("samtools", None, "", entry)
        assert sw.version is None

    def test_r_package_no_source_url_when_no_github(self):
        parser = _make_parser()
        entry = parser.KNOWN_TOOLS["DESeq2"]
        sw = parser._build_from_registry("DESeq2", "1.28", "", entry)
        assert sw.source_url is None  # DESeq2 has no github in registry
        assert sw.bioconda_package == "bioconductor-deseq2"


# ---------------------------------------------------------------------------
# TestBuildEnvironment
# ---------------------------------------------------------------------------

class TestBuildEnvironment:
    """SoftwareParser._build_environment — no LLM, pure logic."""

    def test_bioconda_generates_conda_yaml(self):
        parser = _make_parser()
        sw = Software(name="STAR", bioconda_package="star", version="2.7.3a")
        env = parser._build_environment(sw)
        assert env.conda_yaml is not None
        assert "star=2.7.3a" in env.conda_yaml
        assert "bioconda" in env.conda_yaml

    def test_bioconda_no_version_yaml_no_pin(self):
        parser = _make_parser()
        sw = Software(name="samtools", bioconda_package="samtools")
        env = parser._build_environment(sw)
        assert env.conda_yaml is not None
        assert "samtools\n" in env.conda_yaml or "samtools" in env.conda_yaml
        # No version pin when version is None
        assert "=None" not in env.conda_yaml

    def test_pypi_generates_pip_requirement(self):
        parser = _make_parser()
        sw = Software(name="scanpy", pypi_package="scanpy", version="1.9.0")
        env = parser._build_environment(sw)
        assert "scanpy==1.9.0" in env.pip_requirements

    def test_pypi_no_version_no_pin(self):
        parser = _make_parser()
        sw = Software(name="multiqc", pypi_package="multiqc")
        env = parser._build_environment(sw)
        assert "multiqc" in env.pip_requirements
        assert "==" not in env.pip_requirements[0]

    def test_cran_package_sets_setup_instructions(self):
        parser = _make_parser()
        sw = Software(name="Seurat", cran_package="Seurat")
        env = parser._build_environment(sw)
        assert "install.packages" in env.setup_instructions
        assert "Seurat" in env.setup_instructions

    def test_closed_source_setup_instructions(self):
        parser = _make_parser()
        sw = Software(
            name="cellranger",
            license_type=LicenseType.CLOSED_SOURCE,
            open_source_alternative="STARsolo",
        )
        env = parser._build_environment(sw)
        assert "closed-source" in env.setup_instructions
        assert "STARsolo" in env.setup_instructions

    def test_github_only_tool_setup_instructions(self):
        parser = _make_parser()
        sw = Software(name="MyTool", github_repo="author/my-tool")
        env = parser._build_environment(sw)
        assert "github.com/author/my-tool" in env.setup_instructions or \
               "git clone" in env.setup_instructions

    def test_docker_image_preserved(self):
        parser = _make_parser()
        sw = Software(
            name="STAR",
            environment=Environment(docker_image="quay.io/biocontainers/star"),
        )
        env = parser._build_environment(sw)
        assert env.docker_image == "quay.io/biocontainers/star"


# ---------------------------------------------------------------------------
# TestCheckOpenSource
# ---------------------------------------------------------------------------

class TestCheckOpenSource:
    """SoftwareParser._check_open_source — registry + mocked LLM."""

    def test_registry_open_source(self):
        parser = _make_parser()
        sw = Software(name="STAR", description="RNA-seq aligner")
        result = parser._check_open_source(sw)
        assert result == LicenseType.OPEN_SOURCE

    def test_registry_closed_source(self):
        parser = _make_parser()
        sw = Software(name="cellranger", description="10x pipeline")
        result = parser._check_open_source(sw)
        assert result == LicenseType.CLOSED_SOURCE

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_unknown_tool_uses_llm(self, mock_llm):
        mock_llm.return_value = _make_tool_meta(name="UnknownTool", license_type="open_source")
        parser = _make_parser()
        sw = Software(name="UnknownTool", description="Some obscure tool")
        result = parser._check_open_source(sw)
        assert result == LicenseType.OPEN_SOURCE
        mock_llm.assert_called_once()

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_llm_failure_returns_unknown(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        sw = Software(name="MyObscureTool")
        result = parser._check_open_source(sw)
        assert result == LicenseType.UNKNOWN


# ---------------------------------------------------------------------------
# TestFindAlternative
# ---------------------------------------------------------------------------

class TestFindAlternative:
    """SoftwareParser._find_alternative — closed-source tools."""

    def test_returns_none_for_open_source(self):
        parser = _make_parser()
        sw = Software(name="STAR", license_type=LicenseType.OPEN_SOURCE)
        assert parser._find_alternative(sw) is None

    def test_cellranger_returns_registry_alternative(self):
        parser = _make_parser()
        sw = Software(name="cellranger", license_type=LicenseType.CLOSED_SOURCE)
        result = parser._find_alternative(sw)
        assert result is not None
        assert "STARsolo" in result or "alevin" in result

    def test_cell_ranger_returns_registry_alternative(self):
        parser = _make_parser()
        sw = Software(name="Cell Ranger", license_type=LicenseType.CLOSED_SOURCE)
        result = parser._find_alternative(sw)
        assert result is not None

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_unknown_closed_source_uses_llm(self, mock_llm):
        mock_llm.return_value = _make_tool_meta(
            name="Ingenuity",
            license_type="closed_source",
            open_source_alternative="GSEA or ReactomePA",
        )
        parser = _make_parser()
        sw = Software(
            name="Ingenuity Pathway Analysis",
            license_type=LicenseType.CLOSED_SOURCE,
            description="Pathway analysis tool",
        )
        result = parser._find_alternative(sw)
        assert result is not None
        mock_llm.assert_called_once()

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_llm_failure_returns_none(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        sw = Software(name="SomeTool", license_type=LicenseType.CLOSED_SOURCE)
        assert parser._find_alternative(sw) is None


# ---------------------------------------------------------------------------
# TestIdentifyTool
# ---------------------------------------------------------------------------

class TestIdentifyTool:
    """SoftwareParser._identify_tool — registry path and LLM path."""

    def test_known_tool_uses_registry(self):
        parser = _make_parser()
        sw = parser._identify_tool("STAR", "2.7.3a", "Aligned with STAR.")
        assert sw.name == "STAR"
        assert sw.version == "2.7.3a"
        assert sw.bioconda_package == "star"
        assert sw.environment is not None
        assert sw.environment.conda_yaml is not None

    def test_known_tool_case_insensitive(self):
        parser = _make_parser()
        sw = parser._identify_tool("deseq2", "1.28", "DESeq2 analysis")
        assert sw.bioconda_package is not None

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_unknown_tool_uses_llm(self, mock_llm):
        # Two LLM calls: (1) _build_from_llm for tool metadata,
        # (2) _extract_commands for command extraction from context.
        mock_llm.side_effect = [
            _make_tool_meta(name="SpecialTool", bioconda_package="special-tool"),
            _CommandExtraction(commands=[]),  # no commands found
        ]
        parser = _make_parser()
        sw = parser._identify_tool("SpecialTool", "1.0", "Ran SpecialTool for analysis.")
        assert sw.name == "SpecialTool"
        assert sw.environment is not None
        assert mock_llm.call_count == 2

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_llm_failure_returns_minimal_stub(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        sw = parser._identify_tool("ObscureTool", "0.5", "some context")
        assert sw.name == "ObscureTool"
        assert sw.version == "0.5"

    def test_environment_built_for_registry_tool(self):
        parser = _make_parser()
        sw = parser._identify_tool("scanpy", "1.9.0", "Used scanpy for clustering.")
        assert sw.environment is not None
        # scanpy has both bioconda and pypi
        assert sw.environment.conda_yaml is not None or sw.environment.pip_requirements


# ---------------------------------------------------------------------------
# TestParseFromMethod
# ---------------------------------------------------------------------------

class TestParseFromMethod:
    """SoftwareParser.parse_from_method — integration with Method objects."""

    def test_extracts_tools_from_steps(self):
        method = _make_method_with_steps([[
            dict(step_number=1, description="Align reads", input_data="FASTQ",
                 output_data="BAM", software="STAR", software_version="2.7.3a"),
            dict(step_number=2, description="Count features", input_data="BAM",
                 output_data="counts", software="featureCounts", software_version=None),
        ]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        names = {t.name for t in tools}
        assert "STAR" in names
        assert "featureCounts" in names

    def test_deduplicates_same_tool_across_assays(self):
        """Same tool in two assays should appear only once."""
        method = _make_method_with_steps([
            [dict(step_number=1, description="Align", input_data="FASTQ",
                  output_data="BAM", software="samtools", software_version=None)],
            [dict(step_number=1, description="Sort", input_data="BAM",
                  output_data="sorted BAM", software="samtools", software_version="1.17")],
        ])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        samtools_tools = [t for t in tools if t.name == "samtools"]
        assert len(samtools_tools) == 1
        # Should keep the entry with version
        assert samtools_tools[0].version == "1.17"

    def test_empty_method_returns_empty_list(self):
        method = _make_method_with_steps([[]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        assert tools == []

    def test_steps_without_software_ignored(self):
        method = _make_method_with_steps([[
            dict(step_number=1, description="Manual QC", input_data="gel",
                 output_data="pass/fail", software=None, software_version=None),
        ]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        assert tools == []

    def test_tools_have_environments(self):
        method = _make_method_with_steps([[
            dict(step_number=1, description="DESeq2 analysis", input_data="counts",
                 output_data="results", software="DESeq2", software_version="1.28"),
        ]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        deseq2 = next(t for t in tools if t.name == "DESeq2")
        assert deseq2.environment is not None
        assert deseq2.environment.conda_yaml is not None

    def test_closed_source_has_alternative(self):
        method = _make_method_with_steps([[
            dict(step_number=1, description="Cell Ranger preprocessing",
                 input_data="FASTQ", output_data="feature matrix",
                 software="cellranger", software_version="6.0"),
        ]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        cr = next(t for t in tools if t.name == "cellranger")
        assert cr.license_type == LicenseType.CLOSED_SOURCE
        assert cr.open_source_alternative is not None


# ---------------------------------------------------------------------------
# TestPhase5BiocRegression
# ---------------------------------------------------------------------------

class TestPhase5BiocRegression:
    """Phase 5 regression: enriched AnalysisStep text improves Software parsing."""

    def test_39303722_style_enrichment_improves_software_grounding(self):
        parser = _make_parser()

        baseline_method = _make_method_with_steps([[
            dict(
                step_number=1,
                description="Reads were aligned to hg38.",
                input_data="FASTQ",
                output_data="BAM",
                software="STAR",
                software_version=None,
            ),
            dict(
                step_number=2,
                description="Counts were generated from aligned reads.",
                input_data="BAM",
                output_data="gene counts",
                software="samtools",
                software_version=None,
            ),
        ]])

        enriched_method = _make_method_with_steps([[
            dict(
                step_number=1,
                description="Reads were aligned to hg38 with STAR 2.7.11a.",
                input_data="FASTQ",
                output_data="BAM",
                software="STAR",
                software_version="2.7.11a",
            ),
            dict(
                step_number=2,
                description="Gene-level counts were generated with featureCounts 2.0.1.",
                input_data="BAM",
                output_data="gene counts",
                software="featureCounts",
                software_version="2.0.1",
            ),
            dict(
                step_number=3,
                description="Alignment files were sorted/indexed with samtools 1.17.",
                input_data="BAM",
                output_data="sorted BAM + index",
                software="samtools",
                software_version="1.17",
            ),
        ]])

        baseline_tools = parser.parse_from_method(baseline_method)
        enriched_tools = parser.parse_from_method(enriched_method)

        baseline_by_name = {t.name: t for t in baseline_tools}
        enriched_by_name = {t.name: t for t in enriched_tools}

        new_tools = set(enriched_by_name) - set(baseline_by_name)

        upgraded_existing = []
        for name in set(enriched_by_name) & set(baseline_by_name):
            before = baseline_by_name[name]
            after = enriched_by_name[name]
            version_gained = (before.version in (None, "")) and (after.version not in (None, ""))
            url_gained = (before.source_url in (None, "")) and (after.source_url not in (None, ""))
            if version_gained or url_gained:
                upgraded_existing.append(name)

        # Phase-5 gate:
        # at least one new Software item OR an existing item gains grounding metadata.
        assert new_tools or upgraded_existing

        # Make regression intent explicit for this fixture:
        assert "featureCounts" in new_tools
        assert "STAR" in upgraded_existing or "samtools" in upgraded_existing


# ---------------------------------------------------------------------------
# TestParseFromText
# ---------------------------------------------------------------------------

class TestParseFromText:
    """SoftwareParser.parse_from_text — LLM extraction mocked."""

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_empty_text_returns_empty_no_llm(self, mock_llm):
        parser = _make_parser()
        tools = parser.parse_from_text("")
        assert tools == []
        mock_llm.assert_not_called()

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_extracts_known_tools(self, mock_llm):
        mock_llm.return_value = _SoftwareMentionList(mentions=[
            _SoftwareMention(name="STAR", version="2.7.3a",
                             context="Reads aligned with STAR 2.7.3a"),
            _SoftwareMention(name="DESeq2", version="1.28",
                             context="Differential expression with DESeq2"),
        ])
        parser = _make_parser()
        tools = parser.parse_from_text(SAMPLE_METHODS_TEXT)
        names = {t.name for t in tools}
        assert "STAR" in names
        assert "DESeq2" in names

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_url_enrichment_from_mentions(self, mock_llm):
        # First call: _extract_mentions_from_text
        # Second call: _build_from_llm (returns no github_repo so source_url stays None)
        mock_llm.side_effect = [
            _SoftwareMentionList(mentions=[
                _SoftwareMention(
                    name="MyCustomTool",
                    version="2.0",
                    context="Processed with MyCustomTool",
                    url="https://github.com/lab/mycustomtool",
                )
            ]),
            # _build_from_llm — no github_repo so source_url is not set by LLM path
            _make_tool_meta(name="MyCustomTool", github_repo=None),
        ]
        parser = _make_parser()
        tools = parser.parse_from_text("Processed with MyCustomTool 2.0 (github.com/lab/mycustomtool).")
        assert len(tools) == 1
        # URL from the mention is applied because LLM returned no github_repo
        assert tools[0].source_url == "https://github.com/lab/mycustomtool"

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        parser = _make_parser()
        tools = parser.parse_from_text(SAMPLE_METHODS_TEXT)
        assert tools == []

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_deduplicates_within_text(self, mock_llm):
        """Same tool mentioned twice should appear once."""
        mock_llm.return_value = _SoftwareMentionList(mentions=[
            _SoftwareMention(name="samtools", version=None, context="samtools view"),
            _SoftwareMention(name="samtools", version="1.17", context="samtools sort v1.17"),
        ])
        parser = _make_parser()
        tools = parser.parse_from_text("samtools view; samtools sort (v1.17)")
        samtools_tools = [t for t in tools if t.name == "samtools"]
        assert len(samtools_tools) == 1
        assert samtools_tools[0].version == "1.17"


# ---------------------------------------------------------------------------
# TestResolveMentions
# ---------------------------------------------------------------------------

class TestResolveMentions:
    """SoftwareParser._resolve_mentions — deduplication logic."""

    def test_dedup_prefers_versioned_entry(self):
        parser = _make_parser()
        # 4-tuples: (name, version, context, step)
        mentions = [
            ("STAR", None, "first mention", None),
            ("STAR", "2.7.3a", "second mention with version", None),
        ]
        tools = parser._resolve_mentions(mentions)
        star = next(t for t in tools if t.name == "STAR")
        assert star.version == "2.7.3a"

    def test_case_insensitive_dedup(self):
        parser = _make_parser()
        mentions = [
            ("samtools", "1.17", "samtools sort", None),
            ("Samtools", None, "Samtools view", None),
        ]
        tools = parser._resolve_mentions(mentions)
        assert len([t for t in tools if t.name.lower() == "samtools"]) == 1

    def test_stub_on_resolution_failure(self):
        """Even if _identify_tool fails, a minimal stub is returned."""
        parser = _make_parser()
        with patch.object(parser, "_identify_tool", side_effect=RuntimeError("fail")):
            tools = parser._resolve_mentions([("FailTool", "1.0", "context", None)])
        assert len(tools) == 1
        assert tools[0].name == "FailTool"
        assert tools[0].version == "1.0"


# ---------------------------------------------------------------------------
# TestSoftwareModelIntegration
# ---------------------------------------------------------------------------

class TestSoftwareModelIntegration:
    """Verify Software model fields are correctly populated end-to-end."""

    def test_star_full_round_trip(self):
        """Full end-to-end for STAR: registry → Software → Environment → JSON round-trip."""
        parser = _make_parser()
        sw = parser._identify_tool("STAR", "2.7.3a", "Reads aligned with STAR 2.7.3a")

        assert sw.name == "STAR"
        assert sw.version == "2.7.3a"
        assert sw.language == "C++"
        assert sw.license_type == LicenseType.OPEN_SOURCE
        assert sw.bioconda_package == "star"
        assert sw.github_repo == "alexdobin/STAR"
        assert sw.environment is not None
        assert sw.environment.conda_yaml is not None
        assert "star=2.7.3a" in sw.environment.conda_yaml

        # JSON round-trip
        restored = Software.model_validate_json(sw.model_dump_json())
        assert restored.name == sw.name
        assert restored.version == sw.version
        assert restored.environment.conda_yaml == sw.environment.conda_yaml

    def test_deseq2_r_package(self):
        parser = _make_parser()
        sw = parser._identify_tool("DESeq2", "1.28", "")
        assert sw.language == "R"
        assert sw.bioconda_package == "bioconductor-deseq2"
        assert sw.environment.conda_yaml is not None
        assert "bioconductor-deseq2" in sw.environment.conda_yaml

    def test_cellranger_closed_source_env(self):
        parser = _make_parser()
        sw = parser._identify_tool("cellranger", "6.0", "")
        assert sw.license_type == LicenseType.CLOSED_SOURCE
        assert sw.open_source_alternative is not None
        assert sw.environment is not None
        assert "closed-source" in sw.environment.setup_instructions


# ---------------------------------------------------------------------------
# TestNarrowLLMSchemas  (Evaluation finding #1)
# ---------------------------------------------------------------------------

class TestNarrowLLMSchemas:
    """_LicenseDecision and _AlternativeDecision are narrow single-field schemas.

    These replace the full _ToolMeta schema in _check_open_source and
    _find_alternative, preventing validation failures when the LLM returns
    a minimal response with only the requested field.
    """

    def test_license_decision_minimal_payload(self):
        """_LicenseDecision can be constructed with only license_type."""
        dec = _LicenseDecision(license_type="open_source")
        assert dec.license_type == "open_source"

    def test_license_decision_default_unknown(self):
        dec = _LicenseDecision()
        assert dec.license_type == "unknown"

    def test_alternative_decision_minimal_payload(self):
        """_AlternativeDecision can be constructed with only the alternative field."""
        dec = _AlternativeDecision(open_source_alternative="salmon")
        assert dec.open_source_alternative == "salmon"

    def test_alternative_decision_null_payload(self):
        dec = _AlternativeDecision()
        assert dec.open_source_alternative is None

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_check_open_source_uses_narrow_schema(self, mock_llm):
        """_check_open_source sends _LicenseDecision, not full _ToolMeta."""
        mock_llm.return_value = _LicenseDecision(license_type="open_source")
        parser = _make_parser()
        sw = Software(name="ObscureTool")
        result = parser._check_open_source(sw)
        assert result == LicenseType.OPEN_SOURCE
        # Confirm schema used is _LicenseDecision
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["output_schema"] is _LicenseDecision

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_find_alternative_uses_narrow_schema(self, mock_llm):
        """_find_alternative sends _AlternativeDecision, not full _ToolMeta."""
        mock_llm.return_value = _AlternativeDecision(open_source_alternative="GSEA")
        parser = _make_parser()
        sw = Software(name="IPA", license_type=LicenseType.CLOSED_SOURCE)
        result = parser._find_alternative(sw)
        assert result == "GSEA"
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["output_schema"] is _AlternativeDecision

    @patch("researcher_ai.parsers.software_parser.ask_claude_structured")
    def test_check_open_source_logs_unrecognised_value(self, mock_llm):
        """An unrecognised license_type value degrades to UNKNOWN (not an exception)."""
        mock_llm.return_value = _LicenseDecision(license_type="proprietary")
        parser = _make_parser()
        sw = Software(name="SomeTool")
        result = parser._check_open_source(sw)
        assert result == LicenseType.UNKNOWN


# ---------------------------------------------------------------------------
# TestExtractCommands  (Evaluation finding #2)
# ---------------------------------------------------------------------------

class TestExtractCommands:
    """SoftwareParser._extract_commands — CLI command population."""

    def test_builds_command_from_step_parameters(self):
        """Structured AnalysisStep parameters produce a Command without LLM."""
        step = AnalysisStep(
            step_number=1,
            description="Align reads with STAR",
            input_data="FASTQ reads",
            output_data="BAM files",
            software="STAR",
            software_version="2.7.3a",
            parameters={"runThreadN": "8", "outSAMtype": "BAM SortedByCoordinate"},
        )
        parser = _make_parser()
        commands = parser._extract_commands("STAR", "Aligned with STAR", step=step)
        assert len(commands) == 1
        cmd = commands[0]
        assert "STAR" in cmd.command_template
        assert "runThreadN" in cmd.command_template
        assert cmd.parameters == {"runThreadN": "8", "outSAMtype": "BAM SortedByCoordinate"}
        assert "FASTQ reads" in cmd.required_inputs
        assert "BAM files" in cmd.outputs

    def test_no_step_parameters_tries_llm(self):
        """No AnalysisStep → falls through to LLM extraction path."""
        with patch("researcher_ai.parsers.software_parser.ask_claude_structured") as mock_llm:
            mock_llm.return_value = _CommandExtraction(commands=[
                _CommandMeta(
                    command_template="samtools sort -@ 8 input.bam -o sorted.bam",
                    description="Sort BAM file",
                    required_inputs=["input.bam"],
                    outputs=["sorted.bam"],
                    parameters={"-@": "8"},
                )
            ])
            parser = _make_parser()
            commands = parser._extract_commands(
                "samtools",
                "samtools sort -@ 8 input.bam -o sorted.bam",
                step=None,
            )
        assert len(commands) == 1
        assert "sort" in commands[0].command_template

    def test_empty_context_skips_llm(self):
        """Empty context string → no LLM call, empty command list."""
        with patch("researcher_ai.parsers.software_parser.ask_claude_structured") as mock_llm:
            parser = _make_parser()
            commands = parser._extract_commands("STAR", "", step=None)
        assert commands == []
        mock_llm.assert_not_called()

    def test_llm_failure_returns_empty_commands(self):
        """LLM failure is caught; empty list returned, no exception."""
        with patch("researcher_ai.parsers.software_parser.ask_claude_structured") as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            parser = _make_parser()
            commands = parser._extract_commands("SomeTool", "some context", step=None)
        assert commands == []

    def test_parse_from_method_populates_commands(self):
        """Commands are populated on Software objects from parse_from_method."""
        step = dict(
            step_number=1,
            description="Align reads",
            input_data="FASTQ",
            output_data="BAM",
            software="STAR",
            software_version="2.7.3a",
            parameters={"runThreadN": "8"},
        )
        method = _make_method_with_steps([[step]])
        parser = _make_parser()
        tools = parser.parse_from_method(method)
        star = next(t for t in tools if t.name == "STAR")
        assert len(star.commands) == 1
        assert "runThreadN" in star.commands[0].command_template


# ---------------------------------------------------------------------------
# TestCanonicalToolName  (Evaluation finding #4)
# ---------------------------------------------------------------------------

class TestCanonicalToolName:
    """SoftwareParser._canonical_tool_name — alias normalisation."""

    @pytest.mark.parametrize("name,expected", [
        ("STAR", "star"),
        ("star", "star"),
        ("Cell Ranger", "cellranger"),
        ("cell ranger", "cellranger"),
        ("cell-ranger", "cellranger"),
        ("CellRanger", "cellranger"),
        ("deepTools", "deeptools"),
        ("deep-tools", "deeptools"),
        ("deep_tools", "deeptools"),
        ("featureCounts", "featurecounts"),
        ("DESeq2", "deseq2"),
    ])
    def test_normalises_to_canonical_form(self, name: str, expected: str):
        assert SoftwareParser._canonical_tool_name(name) == expected

    def test_resolve_mentions_deduplicates_spaced_variant(self):
        """'Cell Ranger' and 'cellranger' collapse to a single entry."""
        parser = _make_parser()
        mentions = [
            ("Cell Ranger", "6.0", "Cell Ranger 6.0 used.", None),
            ("cellranger", None, "cellranger preprocessing.", None),
        ]
        tools = parser._resolve_mentions(mentions)
        cr_tools = [t for t in tools if "ranger" in t.name.lower().replace(" ", "").replace("-", "")]
        assert len(cr_tools) == 1
        # Should keep the versioned entry
        assert cr_tools[0].version == "6.0"

    def test_lookup_registry_with_canonical_match(self):
        """Registry lookup succeeds for 'cell-ranger' (not an explicit registry key)."""
        parser = _make_parser()
        entry = parser._lookup_registry("cell-ranger")
        assert entry is not None
        assert entry["license"] == "closed_source"


# ---------------------------------------------------------------------------
# TestNoSpeculativeDockerInference  (Evaluation finding #6)
# ---------------------------------------------------------------------------

class TestNoSpeculativeDockerInference:
    """Docker images are only set when explicitly known (finding #6)."""

    def test_github_only_tool_has_no_docker_image(self):
        """A tool known only by GitHub repo should not get a fabricated docker image."""
        parser = _make_parser()
        sw = Software(name="MyTool", github_repo="author/my-tool")
        env = parser._build_environment(sw)
        assert env.docker_image is None

    def test_registry_tool_with_docker_keeps_image(self):
        """Tools with an explicit docker field in the registry keep their image."""
        parser = _make_parser()
        sw = Software(
            name="STAR",
            environment=Environment(docker_image="quay.io/biocontainers/star"),
        )
        env = parser._build_environment(sw)
        assert env.docker_image == "quay.io/biocontainers/star"

    def test_no_docker_image_for_bioconda_only_tool(self):
        """A tool with bioconda but no explicit docker gets no docker image."""
        parser = _make_parser()
        sw = Software(name="DESeq2", bioconda_package="bioconductor-deseq2", language="R")
        env = parser._build_environment(sw)
        assert env.docker_image is None
        assert env.conda_yaml is not None  # conda is still populated


# ---------------------------------------------------------------------------
# TestSoftwareExtractionSnapshot  (Evaluation finding — snapshot fixture)
# ---------------------------------------------------------------------------

# Frozen methods text for snapshot testing — no LLM needed for known tools
SNAPSHOT_METHODS_TEXT = """\
Reads were aligned to the human genome (hg38) using STAR v2.7.3a with the
flags --outSAMtype BAM SortedByCoordinate --runThreadN 16. PCR duplicates
were removed using Picard MarkDuplicates v2.27.0. Read counts per gene were
obtained using featureCounts (Subread v2.0). Differential expression was
computed with DESeq2 v1.28 (Love et al., 2014) using a Wald test; adjusted
p-values were calculated by the Benjamini-Hochberg method. Single-cell
libraries were processed with Cell Ranger 6.0 (10x Genomics); STARsolo
was used as an open-source alternative for validation.
"""

# Expected tools that should be resolved purely from the KNOWN_TOOLS registry
_SNAPSHOT_KNOWN_TOOLS = {"STAR", "Picard", "featureCounts", "DESeq2"}


@patch("researcher_ai.parsers.software_parser.ask_claude_structured")
class TestSoftwareExtractionSnapshot:
    """Frozen snapshot tests for software extraction.

    Use a fixed methods text and a mocked LLM that returns a pre-defined
    mention list. Validates that the core resolution pipeline (registry path,
    environment generation, deduplication) is stable across code changes.

    Evaluation recommendation: snapshot fixture to reduce LLM oracle drift.
    """

    def _mock_llm(self, prompt, output_schema, **kw):
        """Deterministic LLM mock — returns snapshot fixture data."""
        if output_schema is _SoftwareMentionList:
            return _SoftwareMentionList(mentions=[
                _SoftwareMention(name="STAR", version="2.7.3a",
                                 context="aligned using STAR v2.7.3a",
                                 parameters_raw="--outSAMtype BAM --runThreadN 16"),
                _SoftwareMention(name="Picard", version="2.27.0",
                                 context="Picard MarkDuplicates v2.27.0"),
                _SoftwareMention(name="featureCounts", version="2.0",
                                 context="featureCounts (Subread v2.0)"),
                _SoftwareMention(name="DESeq2", version="1.28",
                                 context="DESeq2 v1.28"),
                _SoftwareMention(name="Cell Ranger", version="6.0",
                                 context="Cell Ranger 6.0 (10x Genomics)"),
            ])
        # Command extraction — return empty for snapshot stability
        if output_schema is _CommandExtraction:
            return _CommandExtraction(commands=[])
        return MagicMock()

    def test_snapshot_known_tools_resolved_from_registry(self, mock_llm):
        """All tools in SNAPSHOT_KNOWN_TOOLS resolve via registry (no LLM for metadata)."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        names = {t.name for t in tools}
        for expected in _SNAPSHOT_KNOWN_TOOLS:
            assert expected in names, f"Expected {expected!r} in parsed tools"

    def test_snapshot_versions_preserved(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        by_name = {t.name: t for t in tools}
        assert by_name["STAR"].version == "2.7.3a"
        assert by_name["DESeq2"].version == "1.28"

    def test_snapshot_environments_generated(self, mock_llm):
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        by_name = {t.name: t for t in tools}
        assert by_name["STAR"].environment.conda_yaml is not None
        assert by_name["DESeq2"].environment.conda_yaml is not None
        assert "bioconductor-deseq2" in by_name["DESeq2"].environment.conda_yaml

    def test_snapshot_cell_ranger_closed_source(self, mock_llm):
        """Cell Ranger (spaced variant) resolves as closed-source with alternative."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        # canonical dedup collapses 'Cell Ranger' → cellranger registry entry
        cr = next((t for t in tools if "ranger" in t.name.lower().replace(" ", "")), None)
        assert cr is not None
        assert cr.license_type == LicenseType.CLOSED_SOURCE
        assert cr.open_source_alternative is not None

    def test_snapshot_dedup_no_duplicates(self, mock_llm):
        """No tool appears twice in the output despite alias variants."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        names = [t.name for t in tools]
        assert len(names) == len(set(t.name.lower() for t in tools)), \
            f"Duplicate tools detected: {names}"

    def test_snapshot_json_round_trip(self, mock_llm):
        """All Software objects survive a JSON round-trip (model stability check)."""
        mock_llm.side_effect = lambda prompt, output_schema, **kw: self._mock_llm(
            prompt, output_schema
        )
        parser = _make_parser()
        tools = parser.parse_from_text(SNAPSHOT_METHODS_TEXT)
        for sw in tools:
            restored = Software.model_validate_json(sw.model_dump_json())
            assert restored.name == sw.name
            assert restored.version == sw.version
