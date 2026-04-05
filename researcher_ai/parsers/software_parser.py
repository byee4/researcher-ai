"""Software parser: identify and structure software tools from methods sections.

Strategy:

- Accept either a ``Method`` object (via ``parse_from_method``) or raw methods
  text (via ``parse_from_text``).
- For each ``AnalysisStep`` in a method, collect ``(name, version, context)``
  tuples.
- For direct text parsing, use LLM to extract all tool mentions at once.
- Resolve each tool via ``KNOWN_TOOLS`` registry first (fast path), then LLM
  (slow path for novel or niche tools).
- Build ``Environment`` spec with conda/pip/docker info.
- Classify license and suggest open-source alternatives for closed-source tools.
- Extract CLI commands and populate ``Software.commands`` from step parameters.

Deferred (planned for Phase 7 / post-Phase 6):

- GitHub license API and web-search enrichment for niche tools (Finding #3).
  The spec says "GitHub license API, known tool registry, OR LLM inference" —
  the registry + LLM path satisfies the OR; external HTTP calls are deferred
  to avoid non-offline tests and rate-limit concerns during pipeline generation.
- _parse_github_code / _parse_notebook full implementations (Finding #5).
  These are intentional thin wrappers; the Pipeline Builder (Phase 7) will
  trigger real GitHub/notebook parsing when code_reference URLs are present.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel, Field

from researcher_ai.models.method import AnalysisStep, Method
from researcher_ai.models.software import Command, Environment, LicenseType, Software
from researcher_ai.utils import llm as llm_utils
from researcher_ai.utils.llm import extract_structured_data, SYSTEM_METHODS_PARSER

logger = logging.getLogger(__name__)

# Deprecated compatibility alias for legacy tests/mocks.
ask_claude_structured = extract_structured_data


def _extract_structured_data(*args, **kwargs):
    return ask_claude_structured(*args, **kwargs)


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------

class _SoftwareMention(BaseModel):
    """A single software mention extracted by the LLM."""

    name: str = Field(description="Tool name (e.g., 'STAR', 'DESeq2', 'samtools')")
    version: Optional[str] = Field(
        default=None,
        description="Version string if stated (e.g., '2.7.3a', 'v1.28')",
    )
    context: str = Field(
        default="",
        description="Verbatim sentence(s) mentioning this tool",
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to tool homepage or GitHub if mentioned",
    )
    parameters_raw: Optional[str] = Field(
        default=None,
        description="Parameters or flags mentioned in the text (raw string)",
    )


class _SoftwareMentionList(BaseModel):
    """LLM-extracted list of software mentions."""
    mentions: list[_SoftwareMention] = Field(default_factory=list)


class _ToolMeta(BaseModel):
    """LLM-enriched metadata for an unrecognised tool."""

    name: str
    description: str = Field(description="One-sentence description of what the tool does")
    language: Optional[str] = Field(
        default=None,
        description="Primary implementation language (e.g., 'Python', 'R', 'C++')",
    )
    github_repo: Optional[str] = Field(
        default=None,
        description="GitHub owner/repo slug if known (e.g., 'alexdobin/STAR')",
    )
    bioconda_package: Optional[str] = Field(
        default=None,
        description="Bioconda package name if the tool is in bioconda",
    )
    pypi_package: Optional[str] = Field(
        default=None,
        description="PyPI package name if the tool is a Python package",
    )
    cran_package: Optional[str] = Field(
        default=None,
        description="CRAN package name if the tool is an R package",
    )
    license_type: str = Field(
        default="unknown",
        description="One of: 'open_source', 'closed_source', 'freemium', 'unknown'",
    )
    open_source_alternative: Optional[str] = Field(
        default=None,
        description="Best open-source alternative if the tool is closed-source",
    )
    docker_image: Optional[str] = Field(
        default=None,
        description="Biocontainers or quay.io Docker image if known",
    )


class _LicenseDecision(BaseModel):
    """Narrow schema for license classification calls.

    Using a full _ToolMeta schema here caused required fields (name, description)
    to fail validation when the LLM returned a minimal answer, silently degrading
    to UNKNOWN (Evaluation finding #1).
    """

    license_type: str = Field(
        default="unknown",
        description=(
            "License classification for the software tool. "
            "Must be one of: 'open_source', 'closed_source', 'freemium', 'unknown'."
        ),
    )


class _AlternativeDecision(BaseModel):
    """Narrow schema for open-source alternative recommendation calls.

    Using a full _ToolMeta schema caused schema mismatch failures for
    single-field queries (Evaluation finding #1).
    """

    open_source_alternative: Optional[str] = Field(
        default=None,
        description=(
            "Name of the best open-source alternative to the closed-source tool. "
            "Null if none is known."
        ),
    )


class _CommandMeta(BaseModel):
    """Metadata for a single CLI command or function call."""

    command_template: str = Field(
        description=(
            "Full command template, e.g. "
            "'STAR --runMode alignReads --genomeDir {genome} --readFilesIn {fastq}'"
        )
    )
    description: str = Field(
        default="",
        description="One-sentence description of what this command does",
    )
    required_inputs: list[str] = Field(
        default_factory=list,
        description="Input files or data required (e.g., ['FASTQ', 'genome index'])",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Output files or data produced (e.g., ['BAM', 'log file'])",
    )
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Key parameter flags and their values",
    )


class _CommandExtraction(BaseModel):
    """LLM-extracted CLI command(s) for a specific software tool."""

    commands: list[_CommandMeta] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SoftwareParser
# ---------------------------------------------------------------------------

class SoftwareParser:
    """Identify and structure software tools from parsed methods.

    For each software tool:
    - Identify version and source
    - Determine environment requirements (conda, pip, system)
    - Parse CLI commands or code references
    - Classify as open/closed source
    - Suggest open-source alternatives for closed-source tools

    Args:
        llm_model: Claude model identifier.
        cache_dir: Optional directory for caching LLM responses.
    """

    # Known bioinformatics tool registry for fast lookup and version resolution
    KNOWN_TOOLS: dict[str, dict] = {
        "STAR": {
            "language": "C++",
            "bioconda": "star",
            "github": "alexdobin/STAR",
            "description": "Splice-aware RNA-seq aligner",
            "license": "open_source",
            "docker": "quay.io/biocontainers/star",
        },
        "HISAT2": {
            "language": "C++",
            "bioconda": "hisat2",
            "github": "DaehwanKimLab/hisat2",
            "description": "Graph-based RNA-seq aligner",
            "license": "open_source",
            "docker": "quay.io/biocontainers/hisat2",
        },
        "samtools": {
            "language": "C",
            "bioconda": "samtools",
            "github": "samtools/samtools",
            "description": "Utilities for manipulating alignments in SAM/BAM format",
            "license": "open_source",
            "docker": "quay.io/biocontainers/samtools",
        },
        "DESeq2": {
            "language": "R",
            "bioconda": "bioconductor-deseq2",
            "description": "Differential expression analysis based on negative binomial model",
            "license": "open_source",
        },
        "edgeR": {
            "language": "R",
            "bioconda": "bioconductor-edger",
            "description": "Differential expression analysis using empirical Bayes",
            "license": "open_source",
        },
        "Seurat": {
            "language": "R",
            "cran": "Seurat",
            "github": "satijalab/seurat",
            "description": "Single-cell RNA-seq analysis toolkit",
            "license": "open_source",
        },
        "scanpy": {
            "language": "Python",
            "pypi": "scanpy",
            "github": "scverse/scanpy",
            "description": "Single-cell analysis in Python",
            "license": "open_source",
            "docker": "quay.io/biocontainers/scanpy",
        },
        "cellranger": {
            "language": "C++/Python",
            "description": "10x Genomics pipeline for single-cell RNA-seq preprocessing",
            "license": "closed_source",
            "alternative": "STARsolo or alevin-fry",
        },
        "Cell Ranger": {
            "language": "C++/Python",
            "description": "10x Genomics pipeline for single-cell RNA-seq preprocessing",
            "license": "closed_source",
            "alternative": "STARsolo or alevin-fry",
        },
        "MACS2": {
            "language": "Python",
            "bioconda": "macs2",
            "github": "macs3-project/MACS",
            "description": "Model-based Analysis of ChIP-Seq",
            "license": "open_source",
        },
        "MACS3": {
            "language": "Python",
            "bioconda": "macs3",
            "github": "macs3-project/MACS",
            "description": "Model-based Analysis of ChIP-Seq (v3)",
            "license": "open_source",
        },
        "bowtie2": {
            "language": "C++",
            "bioconda": "bowtie2",
            "github": "BenLangmead/bowtie2",
            "description": "Fast and sensitive short-read aligner",
            "license": "open_source",
            "docker": "quay.io/biocontainers/bowtie2",
        },
        "FastQC": {
            "language": "Java",
            "bioconda": "fastqc",
            "description": "Quality control for high-throughput sequencing data",
            "license": "open_source",
        },
        "MultiQC": {
            "language": "Python",
            "bioconda": "multiqc",
            "pypi": "multiqc",
            "github": "MultiQC/MultiQC",
            "description": "Aggregate bioinformatics QC reports",
            "license": "open_source",
        },
        "trimmomatic": {
            "language": "Java",
            "bioconda": "trimmomatic",
            "description": "Flexible read trimming for Illumina NGS data",
            "license": "open_source",
        },
        "Trimmomatic": {
            "language": "Java",
            "bioconda": "trimmomatic",
            "description": "Flexible read trimming for Illumina NGS data",
            "license": "open_source",
        },
        "cutadapt": {
            "language": "Python",
            "bioconda": "cutadapt",
            "pypi": "cutadapt",
            "github": "marcelm/cutadapt",
            "description": "Trim adapter sequences from reads",
            "license": "open_source",
        },
        "featureCounts": {
            "language": "C",
            "bioconda": "subread",
            "description": "Read summarization for genes, exons, and other features",
            "license": "open_source",
        },
        "salmon": {
            "language": "C++",
            "bioconda": "salmon",
            "github": "COMBINE-lab/salmon",
            "description": "Wicked-fast transcript-level quantification",
            "license": "open_source",
            "docker": "quay.io/biocontainers/salmon",
        },
        "kallisto": {
            "language": "C++",
            "bioconda": "kallisto",
            "github": "pachterlab/kallisto",
            "description": "Near-optimal probabilistic RNA-seq quantification",
            "license": "open_source",
        },
        "bedtools": {
            "language": "C++",
            "bioconda": "bedtools",
            "github": "arq5x/bedtools2",
            "description": "Swiss-army knife for genome arithmetic",
            "license": "open_source",
            "docker": "quay.io/biocontainers/bedtools",
        },
        "picard": {
            "language": "Java",
            "bioconda": "picard",
            "github": "broadinstitute/picard",
            "description": "Java tools for manipulating high-throughput sequencing data",
            "license": "open_source",
        },
        "Picard": {
            "language": "Java",
            "bioconda": "picard",
            "github": "broadinstitute/picard",
            "description": "Java tools for manipulating high-throughput sequencing data",
            "license": "open_source",
        },
        "GATK": {
            "language": "Java",
            "bioconda": "gatk4",
            "github": "broadinstitute/gatk",
            "description": "Genome Analysis Toolkit for variant discovery",
            "license": "open_source",
            "docker": "broadinstitute/gatk",
        },
        "BWA": {
            "language": "C",
            "bioconda": "bwa",
            "github": "lh3/bwa",
            "description": "Burrows-Wheeler Aligner for short read mapping",
            "license": "open_source",
        },
        "bwa": {
            "language": "C",
            "bioconda": "bwa",
            "github": "lh3/bwa",
            "description": "Burrows-Wheeler Aligner for short read mapping",
            "license": "open_source",
        },
        "minimap2": {
            "language": "C",
            "bioconda": "minimap2",
            "github": "lh3/minimap2",
            "description": "Versatile pairwise aligner for long reads and assemblies",
            "license": "open_source",
        },
        "deeptools": {
            "language": "Python",
            "bioconda": "deeptools",
            "github": "deeptools/deeptools",
            "description": "Tools for exploring NGS data",
            "license": "open_source",
        },
        "deepTools": {
            "language": "Python",
            "bioconda": "deeptools",
            "github": "deeptools/deeptools",
            "description": "Tools for exploring NGS data",
            "license": "open_source",
        },
        "CLIPper": {
            "language": "Python",
            "bioconda": "clipper",
            "github": "YeoLab/clipper",
            "description": "CLIP-seq peak caller",
            "license": "open_source",
        },
        "RSEM": {
            "language": "C++",
            "bioconda": "rsem",
            "github": "deweylab/RSEM",
            "description": "RNA-seq transcript quantification and differential expression",
            "license": "open_source",
        },
        "HTSeq": {
            "language": "Python",
            "bioconda": "htseq",
            "pypi": "HTSeq",
            "github": "htseq/htseq",
            "description": "Python framework for processing high-throughput sequencing data",
            "license": "open_source",
        },
        "SPAdes": {
            "language": "C++",
            "bioconda": "spades",
            "github": "ablab/spades",
            "description": "St. Petersburg genome assembler",
            "license": "open_source",
        },
        "Snakemake": {
            "language": "Python",
            "pypi": "snakemake",
            "bioconda": "snakemake",
            "github": "snakemake/snakemake",
            "description": "Workflow management system",
            "license": "open_source",
        },
        "Nextflow": {
            "language": "Groovy",
            "bioconda": "nextflow",
            "github": "nextflow-io/nextflow",
            "description": "Data-driven computational pipeline tool",
            "license": "open_source",
        },
        "R": {
            "language": "R",
            "description": "Statistical computing language",
            "license": "open_source",
        },
        "Python": {
            "language": "Python",
            "description": "General-purpose programming language",
            "license": "open_source",
        },
    }

    def __init__(
        self,
        llm_model: str = llm_utils.DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
    ):
        """Initialize SoftwareParser with model and optional LLM cache directory."""
        self.llm_model = llm_model
        from researcher_ai.utils.llm import LLMCache
        self.cache = LLMCache(cache_dir) if cache_dir else None

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_from_method(self, method: Method) -> list[Software]:
        """Extract all software tools from a parsed Method object.

        Strategy:

        1. Collect ``(name, version, context, step)`` tuples from every
           ``AnalysisStep`` across all assays.
        2. Deduplicate by canonical name (keeping richest context/version).
        3. Resolve each tool via ``KNOWN_TOOLS`` then LLM.
        4. Build environment specs, classify licenses, and extract CLI commands.

        Args:
            method: A fully populated Method object from MethodsParser.

        Returns:
            Deduplicated list of Software objects.
        """
        # (name, version, context, step_or_None)
        mentions: list[tuple[str, Optional[str], str, Optional[object]]] = []

        for assay in method.assays:
            for step in assay.steps:
                if step.software:
                    ctx = f"{step.description} {step.input_data} → {step.output_data}"
                    mentions.append((step.software, step.software_version, ctx, step))

        return self._resolve_mentions(mentions)  # type: ignore[arg-type]

    def parse_from_text(self, methods_text: str) -> list[Software]:
        """Extract software tools directly from methods text.

        Uses LLM to identify tool mentions with version numbers and URLs.

        Args:
            methods_text: Raw methods section text from a paper.

        Returns:
            List of Software objects with environment specs.
        """
        if not methods_text.strip():
            return []

        mentions_raw = self._extract_mentions_from_text(methods_text)
        mentions = [
            (m.name, m.version, m.context or "", None)
            for m in mentions_raw
        ]
        tools = self._resolve_mentions(mentions)

        # Enrich with URL info from LLM extraction
        url_map = {m.name: m.url for m in mentions_raw if m.url}
        for tool in tools:
            if tool.name in url_map and not tool.source_url:
                tool.source_url = url_map[tool.name]

        return tools

    # ── Private: LLM text extraction ─────────────────────────────────────────

    def _extract_mentions_from_text(self, methods_text: str) -> list[_SoftwareMention]:
        """Use LLM to extract all software mentions from a methods text block.

        Returns a list of raw _SoftwareMention objects (not yet enriched).
        """
        try:
            result = _extract_structured_data(
                prompt=(
                    "Extract all software tools, packages, and pipelines mentioned in this "
                    "methods text. For each tool, capture: name, version (if stated), "
                    "the verbatim sentence mentioning it, any URL mentioned, and any "
                    "parameters or flags mentioned.\n\n"
                    "Include standalone analysis tools (STAR, DESeq2, etc.), R/Python "
                    "packages (ggplot2, pandas, etc.), and custom scripts if referenced.\n\n"
                    f"METHODS TEXT:\n{methods_text[:4000]}"
                ),
                output_schema=_SoftwareMentionList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return result.mentions
        except Exception as exc:
            logger.warning("Software mention extraction failed: %s", exc)
            return []

    # ── Private: mention deduplication & resolution ───────────────────────────

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        """Normalise a tool name for deduplication.

        Strips spaces, hyphens, and underscores, then lowercases.
        This collapses common alias variants:
          'cell ranger', 'Cell Ranger', 'cell-ranger', 'CellRanger' → 'cellranger'
          'deepTools', 'deeptools', 'deep-tools'                    → 'deeptools'

        Evaluation finding #4: lower-only dedup missed spaced/hyphenated variants.
        """
        import re as _re
        return _re.sub(r"[\s\-_]+", "", name).lower()

    def _resolve_mentions(
        self,
        mentions: list[tuple[str, Optional[str], str, Optional[object]]],
    ) -> list[Software]:
        """Deduplicate mentions by canonical name and resolve each to a Software object.

        Uses _canonical_tool_name so that 'Cell Ranger', 'cell-ranger', and
        'cellranger' all collapse to a single entry (Evaluation finding #4).
        Keeps the versioned entry when duplicates exist.
        """
        # canonical_key → (original_name, version, context, step)
        deduped: dict[str, tuple[str, Optional[str], str, Optional[object]]] = {}
        for name, version, context, step in mentions:
            key = self._canonical_tool_name(name)
            if key not in deduped or (version and not deduped[key][1]):
                deduped[key] = (name, version, context, step)

        tools: list[Software] = []
        for _key, (original_name, version, context, step) in deduped.items():
            try:
                tool = self._identify_tool(original_name, version, context, step=step)
                tools.append(tool)
            except Exception as exc:
                logger.warning(
                    "Failed to resolve tool %r: %s — creating minimal stub", original_name, exc
                )
                tools.append(Software(name=original_name, version=version))

        return tools

    # ── Private: single-tool identification ──────────────────────────────────

    def _identify_tool(
        self,
        name: str,
        version: Optional[str],
        context: str,
        step: Optional[object] = None,
    ) -> Software:
        """Identify a single software tool and build a full Software object.

        Priority:
        1. Check KNOWN_TOOLS registry (exact match, then case-insensitive).
        2. If not found, use LLM to enrich metadata.
        3. Build Environment from package info.
        4. Extract CLI commands from step parameters + context text.

        Args:
            name: Tool name as found in the text.
            version: Version string if extracted.
            context: Surrounding text for additional context.
            step: Optional AnalysisStep for command/parameter extraction.

        Returns:
            Fully populated Software object.
        """
        registry_entry = self._lookup_registry(name)

        if registry_entry is not None:
            sw = self._build_from_registry(name, version, context, registry_entry)
        else:
            sw = self._build_from_llm(name, version, context)

        sw.environment = self._build_environment(sw)
        sw.commands = self._extract_commands(name, context, step=step)
        return sw

    def _lookup_registry(self, name: str) -> Optional[dict]:
        """Return registry entry for name.

        Match order:
        1. Exact match (fast path, preserves intentional casing like 'DESeq2').
        2. Case-insensitive exact match.
        3. Canonical alias match via _canonical_tool_name (collapses spaces/hyphens).

        Evaluation finding #4: canonical matching catches 'Cell Ranger' → cellranger
        even without a duplicate registry entry.
        """
        if name in self.KNOWN_TOOLS:
            return self.KNOWN_TOOLS[name]
        name_lower = name.lower()
        name_canonical = self._canonical_tool_name(name)
        for key, entry in self.KNOWN_TOOLS.items():
            if key.lower() == name_lower:
                return entry
        for key, entry in self.KNOWN_TOOLS.items():
            if self._canonical_tool_name(key) == name_canonical:
                return entry
        return None

    def _build_from_registry(
        self,
        name: str,
        version: Optional[str],
        context: str,
        entry: dict,
    ) -> Software:
        """Build a Software object from a KNOWN_TOOLS registry entry."""
        license_type = LicenseType(entry.get("license", "unknown"))

        github_repo = entry.get("github")
        source_url = (
            f"https://github.com/{github_repo}" if github_repo else None
        )

        sw = Software(
            name=name,
            version=version,
            description=entry.get("description", ""),
            language=entry.get("language"),
            license_type=license_type,
            source_url=source_url,
            bioconda_package=entry.get("bioconda"),
            cran_package=entry.get("cran"),
            pypi_package=entry.get("pypi"),
            github_repo=github_repo,
        )

        if license_type == LicenseType.CLOSED_SOURCE:
            sw.open_source_alternative = entry.get("alternative")

        docker = entry.get("docker")
        if docker:
            sw.environment = Environment(docker_image=docker)

        return sw

    def _build_from_llm(
        self,
        name: str,
        version: Optional[str],
        context: str,
    ) -> Software:
        """Use LLM to enrich metadata for an unrecognised tool."""
        try:
            meta = _extract_structured_data(
                prompt=(
                    f"Provide metadata for the bioinformatics/scientific software tool "
                    f"named '{name}'. Context from the paper: '{context[:500]}'\n\n"
                    "Return: description, language, github_repo (owner/repo), "
                    "bioconda_package, pypi_package, cran_package, license_type "
                    "('open_source'/'closed_source'/'freemium'/'unknown'), "
                    "open_source_alternative (only if closed-source), docker_image."
                ),
                output_schema=_ToolMeta,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
        except Exception as exc:
            logger.warning("LLM tool enrichment failed for %r: %s", name, exc)
            return Software(name=name, version=version)

        try:
            license_type = LicenseType(meta.license_type)
        except ValueError:
            license_type = LicenseType.UNKNOWN

        github_repo = meta.github_repo
        source_url = (
            f"https://github.com/{github_repo}" if github_repo else None
        )

        return Software(
            name=meta.name,
            version=version,
            description=meta.description,
            language=meta.language,
            license_type=license_type,
            source_url=source_url,
            bioconda_package=meta.bioconda_package,
            cran_package=meta.cran_package,
            pypi_package=meta.pypi_package,
            github_repo=github_repo,
            open_source_alternative=meta.open_source_alternative,
        )

    # ── Private: environment building ────────────────────────────────────────

    def _build_environment(self, software: Software) -> Environment:
        """Generate environment specification for a tool.

        Priority order:
        1. Bioconda package  → conda env YAML
        2. PyPI package      → pip requirements line
        3. CRAN package      → R install.packages() instruction
        4. Docker image      → docker pull instruction
        5. GitHub repo       → generic source instructions
        6. Closed source     → note with alternative
        """
        conda_yaml: Optional[str] = None
        pip_requirements: list[str] = []
        system_deps: list[str] = []
        docker_image: Optional[str] = None
        setup_instructions: str = ""

        pkg_version = f"={software.version}" if software.version else ""

        if software.bioconda_package:
            channel = (
                "bioconda" if software.bioconda_package.startswith("bioconductor-")
                or software.language == "R"
                else "bioconda"
            )
            conda_yaml = (
                f"name: {software.name.lower()}-env\n"
                f"channels:\n"
                f"  - {channel}\n"
                f"  - conda-forge\n"
                f"  - defaults\n"
                f"dependencies:\n"
                f"  - {software.bioconda_package}{pkg_version}\n"
            )

        if software.pypi_package:
            version_spec = f"=={software.version}" if software.version else ""
            pip_requirements.append(f"{software.pypi_package}{version_spec}")

        # Only set docker_image when explicitly known from the registry entry.
        # Evaluation finding #6: heuristic quay.io inference fabricated non-existent
        # image paths and could break generated pipeline environments. We now rely
        # solely on verified images carried in the registry entry (via software.environment)
        # or from the LLM-returned docker_image field.
        if software.environment and software.environment.docker_image:
            docker_image = software.environment.docker_image

        if software.cran_package:
            setup_instructions = (
                f"# Install in R:\n"
                f"install.packages('{software.cran_package}')\n"
            )
        elif software.license_type == LicenseType.CLOSED_SOURCE:
            alt = software.open_source_alternative or "an open-source alternative"
            setup_instructions = (
                f"# {software.name} is closed-source. "
                f"Consider using {alt} instead.\n"
                f"# Download from the vendor's website.\n"
            )
        elif software.github_repo and not conda_yaml and not pip_requirements:
            setup_instructions = (
                f"# Install from source:\n"
                f"git clone https://github.com/{software.github_repo}\n"
                f"# Follow repository README for build instructions.\n"
            )

        return Environment(
            conda_yaml=conda_yaml,
            docker_image=docker_image,
            pip_requirements=pip_requirements,
            system_dependencies=system_deps,
            setup_instructions=setup_instructions,
        )

    # ── Private: license and alternative ─────────────────────────────────────

    def _check_open_source(self, software: Software) -> LicenseType:
        """Determine if software is open source.

        Check KNOWN_TOOLS registry first, then fall back to LLM inference.

        Evaluation finding #1: previously used full _ToolMeta schema which has
        required fields (name, description) that caused validation failures when
        the LLM returned a minimal response focused only on license_type.  Now
        uses the narrow _LicenseDecision schema whose only field is license_type.
        """
        entry = self._lookup_registry(software.name)
        if entry:
            try:
                return LicenseType(entry.get("license", "unknown"))
            except ValueError:
                return LicenseType.UNKNOWN

        # LLM fallback — narrow schema avoids required-field validation failures
        try:
            result = _extract_structured_data(
                prompt=(
                    f"Is '{software.name}' open-source, closed-source, freemium, or unknown? "
                    f"Context: {software.description}. "
                    "Return license_type as exactly one of: "
                    "'open_source', 'closed_source', 'freemium', 'unknown'."
                ),
                output_schema=_LicenseDecision,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            try:
                return LicenseType(result.license_type)
            except ValueError:
                logger.warning(
                    "LLM returned unrecognised license_type %r for %r",
                    result.license_type, software.name,
                )
                return LicenseType.UNKNOWN
        except Exception as exc:
            logger.warning("License classification failed for %r: %s", software.name, exc)
            return LicenseType.UNKNOWN

    def _find_alternative(self, software: Software) -> Optional[str]:
        """If closed source, find the best open-source alternative.

        Uses KNOWN_TOOLS registry first, then LLM with bioinformatics ecosystem knowledge.

        Evaluation finding #1: previously used full _ToolMeta schema which caused
        validation failures for minimal responses.  Now uses the narrow
        _AlternativeDecision schema whose only field is open_source_alternative.
        """
        if software.license_type != LicenseType.CLOSED_SOURCE:
            return None

        entry = self._lookup_registry(software.name)
        if entry and entry.get("alternative"):
            return entry["alternative"]

        try:
            result = _extract_structured_data(
                prompt=(
                    f"'{software.name}' is closed-source bioinformatics software. "
                    f"Description: {software.description}. "
                    "What is the best open-source alternative? "
                    "Return open_source_alternative as a short tool name or null."
                ),
                output_schema=_AlternativeDecision,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return result.open_source_alternative
        except Exception as exc:
            logger.warning("Alternative lookup failed for %r: %s", software.name, exc)
            return None

    # ── Private: CLI command extraction ──────────────────────────────────────

    def _extract_commands(
        self,
        software_name: str,
        context: str,
        step: Optional[object] = None,
    ) -> list[Command]:
        """Extract CLI commands for a tool from step parameters and context text.

        Strategy:
        1. If the AnalysisStep has structured parameters, build a template from them.
        2. Supplement with LLM extraction from the surrounding context sentence.

        Evaluation finding #2: Software.commands was never populated despite
        being a first-class model field and an explicit spec goal.

        Args:
            software_name: Name of the tool (used for template and LLM prompt).
            context: Surrounding methods text sentence(s).
            step: Optional AnalysisStep with structured parameters dict.

        Returns:
            List of Command objects (may be empty for tools with no parseable commands).
        """
        commands: list[Command] = []

        # --- Path 1: build from AnalysisStep.parameters (structured, no LLM) ---
        if step is not None:
            from researcher_ai.models.method import AnalysisStep as _Step
            if isinstance(step, _Step) and step.parameters:
                param_str = " ".join(
                    f"--{k} {v}" for k, v in step.parameters.items()
                )
                template = f"{software_name} {param_str}".strip()
                required_inputs = [step.input_data] if step.input_data else []
                outputs = [step.output_data] if step.output_data else []
                commands.append(Command(
                    command_template=template,
                    description=step.description or f"Run {software_name}",
                    required_inputs=required_inputs,
                    outputs=outputs,
                    parameters=step.parameters,
                ))

        # --- Path 2: LLM extraction from context text ---
        # Only invoke if there is meaningful context and no structured command yet.
        if not commands and context.strip():
            try:
                result = _extract_structured_data(
                    prompt=(
                        f"Extract the CLI command or function call for '{software_name}' "
                        f"described in this text. Include flag names and values as given.\n\n"
                        f"TEXT: {context[:800]}"
                    ),
                    output_schema=_CommandExtraction,
                    system=SYSTEM_METHODS_PARSER,
                    model=self.llm_model,
                    cache=self.cache,
                )
                for cm in result.commands:
                    commands.append(Command(
                        command_template=cm.command_template,
                        description=cm.description,
                        required_inputs=cm.required_inputs,
                        outputs=cm.outputs,
                        parameters=cm.parameters,
                    ))
            except Exception as exc:
                logger.debug(
                    "Command extraction skipped for %r: %s", software_name, exc
                )

        return commands

    # ── Private: GitHub / notebook parsing ───────────────────────────────────
    # NOTE: These are intentional deferred stubs (Evaluation finding #5).
    # The Pipeline Builder (Phase 7) will trigger real GitHub/notebook parsing
    # when code_reference URLs are present in AnalysisStep objects.

    def _parse_github_code(self, github_url: str) -> dict:
        """Parse a GitHub repo for relevant analysis scripts.

        Returns dict with keys: language, scripts, notebooks, readme_summary.

        Note: This method is intentionally a thin wrapper for now; a full
        implementation would use the GitHub API or web fetching to retrieve
        the repo structure.
        """
        return {
            "language": None,
            "scripts": [],
            "notebooks": [],
            "readme_summary": f"See {github_url} for details.",
        }

    def _parse_notebook(self, notebook_url: str) -> dict:
        """Parse a publicly available Jupyter notebook.

        Returns dict with keys: cells_summary, packages_used, code_snippets.

        Note: This method is intentionally a thin wrapper for now; a full
        implementation would fetch the notebook JSON and parse cells.
        """
        return {
            "cells_summary": f"Notebook at {notebook_url}",
            "packages_used": [],
            "code_snippets": [],
        }
