"""Methods parser: convert prose methods sections to structured analysis pipelines.

Strategy:

- Extract the methods section text from the ``Paper`` object by searching
  section titles for common variants (Methods, Materials and Methods,
  STAR Methods, and related forms), with fallback to supplementary methods.
- Identify all distinct assays via LLM.
- For each assay, locate relevant paragraph(s) and decompose them into ordered
  ``AnalysisStep`` objects via LLM.
- Identify assay dependencies (DAG edges) for multi-omic papers.
- Extract data-availability and code-availability statements.
- Return a fully populated ``Method`` object with an ``AssayGraph``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from researcher_ai.models.figure import Figure
from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayDependency,
    AssayGraph,
    Method,
    MethodCategory,
)
from researcher_ai.models.paper import Paper
from researcher_ai.utils.rag import ProtocolRAGStore, search_protocol_docs as rag_search_protocol_docs
from researcher_ai.utils import llm as llm_utils
from researcher_ai.utils.llm import (
    LLMCache,
    extract_structured_data,
    extract_structured_data_with_tools as llm_extract_structured_data_with_tools,
    SYSTEM_METHODS_PARSER,
)

logger = logging.getLogger(__name__)

# Deprecated compatibility alias for legacy tests/mocks.
ask_claude_structured = extract_structured_data


def _extract_structured_data(*args, **kwargs):
    return ask_claude_structured(*args, **kwargs)


def _extract_structured_data_with_tools(**kwargs):
    """Tool-calling extractor with offline/mock-friendly fallback."""
    try:
        return llm_extract_structured_data_with_tools(**kwargs)
    except Exception:
        # Fallback keeps tests deterministic when litellm/tool calling is unavailable.
        return _extract_structured_data(
            prompt=kwargs.get("prompt", ""),
            output_schema=kwargs["schema"],
            system=kwargs.get("system", ""),
            model=kwargs.get("model_router"),
            cache=None,
        )

# ---------------------------------------------------------------------------
# Section-title keywords for methods detection (case-insensitive)
# ---------------------------------------------------------------------------
_METHODS_TITLE_RE = re.compile(
    r"(?:^|\b)(?:"
    r"methods?(?:\s+and\s+materials?)?|"
    r"materials?\s+and\s+methods?|"
    r"experimental\s+(?:procedures?|design|methods?)|"
    r"star\s*[\u2605\*]?\s*methods?|"
    r"online\s+methods?|"
    r"supplementary\s+methods?"
    r")(?:\b|$)",
    re.IGNORECASE,
)

_METHODS_HEADING_LINE_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*(?:"
    r"methods?(?:\s+and\s+materials?)?|"
    r"materials?\s+and\s+methods?|"
    r"experimental\s+(?:procedures?|design|methods?)|"
    r"star\s*[\u2605\*]?\s*methods?|"
    r"online\s+methods?"
    r")\s*[:.]?\s*$",
    re.IGNORECASE,
)

_TOP_LEVEL_SECTION_STOP_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*(?:"
    r"abstract|introduction|results?|discussion|conclusions?|references|"
    r"acknowledg(?:e)?ments?|funding|author\s+contributions?|"
    r"competing\s+interests?|declarations?|ethics?|supplementary(?:\s+information)?|"
    r"appendix|data\s+availability|code\s+availability"
    r")\s*[:.]?\s*$",
    re.IGNORECASE,
)

_AVAIL_TITLE_RE = re.compile(
    r"(?:data|code|software|resource)\s+availability",
    re.IGNORECASE,
)
_DATASET_ACCESSION_RE = re.compile(
    r"\b("
    r"GSE\d{4,8}"
    r"|GSM\d{4,8}"
    r"|GDS\d{3,7}"
    r"|SRP\d{4,9}"
    r"|SRX\d{4,9}"
    r"|SRR\d{4,9}"
    r"|ERP\d{4,9}"
    r"|ERR\d{4,9}"
    r"|PRJNA\d{4,9}"
    r"|PRJEB\d{4,9}"
    r"|PXD\d{4,9}"
    r"|MSV\d{4,9}"
    r"|E-MTAB-\d{4,9}"
    r"|EGAS\d{4,9}"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------

class _AssayList(BaseModel):
    """LLM-extracted list of assay names from a methods section."""
    assay_names: list[str] = Field(
        default_factory=list,
        description=(
            "List of distinct experimental assays or analysis procedures described. "
            "E.g., ['RNA-seq library preparation', 'STAR alignment', "
            "'DESeq2 differential expression', 'ChIP-seq peak calling']"
        ),
    )


class _StepMeta(BaseModel):
    """LLM-extracted metadata for a single analysis step."""
    step_number: int
    description: str = Field(description="What this step does (1–2 sentences)")
    input_data: str = Field(description="Data consumed by this step (e.g., 'raw FASTQ reads')")
    output_data: str = Field(description="Data produced by this step (e.g., 'aligned BAM files')")
    software: Optional[str] = Field(default=None, description="Tool name (e.g., 'STAR', 'DESeq2')")
    software_version: Optional[str] = Field(default=None, description="Version if stated")
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Key parameters (e.g., {'min_mapq': '255', 'threads': '8'})",
    )
    code_reference: Optional[str] = Field(
        default=None,
        description="URL or DOI if a specific code repository is referenced",
    )


class _AssayMeta(BaseModel):
    """LLM-extracted metadata for a single assay."""
    name: str
    description: str = Field(description="One-sentence description of this assay")
    data_type: str = Field(
        description=(
            "What kind of data this assay operates on: 'sequencing', 'imaging', "
            "'proteomics', 'mass_spec', 'flow_cytometry', or 'other'. "
            "Do NOT use 'computational' here — use the separate method_category "
            "field to indicate whether the assay is experimental vs computational."
        )
    )
    raw_data_source: Optional[str] = Field(
        default=None,
        description="Repository accession or supplementary location (e.g., 'GEO: GSE72987')",
    )
    steps: list[_StepMeta] = Field(default_factory=list)
    figures_produced: list[str] = Field(
        default_factory=list,
        description="Figure IDs this assay contributes to (e.g., ['Figure 2', 'Figure 3b'])",
    )


class _DependencyMeta(BaseModel):
    upstream_assay: str = Field(description="Assay that must run first")
    downstream_assay: str = Field(description="Assay that depends on the upstream")
    dependency_type: str = Field(
        description=(
            "How the upstream output feeds the downstream assay. "
            "E.g., 'normalization_reference', 'peak_filter', "
            "'co-analysis', 'integration', 'quality_control'"
        )
    )
    description: str = Field(default="", description="Human-readable explanation")


class _DependencyList(BaseModel):
    """LLM-extracted assay dependency edges."""
    dependencies: list[_DependencyMeta] = Field(default_factory=list)


class _AvailabilityStatement(BaseModel):
    data_statement: str = Field(
        default="",
        description="Data availability statement verbatim from the paper",
    )
    code_statement: str = Field(
        default="",
        description="Code availability statement verbatim from the paper",
    )


class _AssayCategoryItem(BaseModel):
    """Classification of a single assay as experimental, computational, or mixed."""

    name: str = Field(description="Exact assay name from the provided list")
    method_category: str = Field(
        description=(
            "Classify as:\n"
            "- 'experimental': wet-lab protocol, sample preparation, UV crosslinking, "
            "immunoprecipitation, library prep, instrument run, cell culture, Western blot, "
            "FACS, microscopy, or any hands-on bench work.\n"
            "- 'computational': bioinformatics or statistical analysis on a computer — "
            "read alignment, peak calling, differential expression, dimensionality reduction, "
            "normalization, variant calling, motif analysis, clustering.\n"
            "- 'mixed': entry explicitly combines both wet-lab steps and computational "
            "analysis (e.g., 'eCLIP-seq library preparation and read processing')."
        )
    )


class _AssayClassificationList(BaseModel):
    """LLM-returned classification for a list of assay names."""

    assays: list[_AssayCategoryItem] = Field(default_factory=list)


class _StepParameterInference(BaseModel):
    """LLM-inferred parameter payload for one analysis step."""

    step_number: int
    inferred_parameters: dict[str, str] = Field(default_factory=dict)
    inferred_software: Optional[str] = None
    inferred_software_version: Optional[str] = None
    rationale: str = ""


class _StepParameterInferenceList(BaseModel):
    """LLM response container for inferred step parameters."""

    updates: list[_StepParameterInference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MethodsParser
# ---------------------------------------------------------------------------

class MethodsParser:
    """Parse a paper's methods section into structured Method / AssayGraph objects.

    For each assay identified:
    - Locates the relevant paragraph(s) in the methods text.
    - Decomposes it into ordered AnalysisStep objects via LLM.
    - Links steps to figures when figure context is provided.
    - Identifies assay dependencies to build the AssayGraph DAG.

    Args:
        llm_model: Claude model identifier.
        cache_dir: Optional directory for caching LLM responses.
    """

    def __init__(
        self,
        llm_model: str = llm_utils.DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
    ):
        """Initialize MethodsParser with optional on-disk LLM cache."""
        self.llm_model = llm_model
        self.cache = LLMCache(cache_dir) if cache_dir else None
        self.protocol_rag = ProtocolRAGStore()

    # ── Public API ───────────────────────────────────────────────────────────

    def parse(
        self,
        paper: Paper,
        figures: Optional[list[Figure]] = None,
        computational_only: bool = True,
    ) -> Method:
        """Parse the methods section of a paper into a structured Method object.

        Args:
            paper: A Paper object (from PaperParser).
            figures: Optional list of parsed Figure objects for cross-referencing
                     which assay produces which figure.
            computational_only: When True (default), classify all identified assays
                as experimental / computational / mixed and then deep-parse only
                assays classified as strictly ``computational``. Excluded assays
                (experimental or mixed) are recorded as
                ``assay_filtered_non_computational:`` entries in
                ``Method.parse_warnings`` so callers can audit what was skipped.
                Set to False to parse every assay regardless of category (useful
                for full-pipeline audits or when the classification step should be
                a separate concern).

        Returns:
            Method with populated AssayGraph (assays + dependency edges).
        """
        methods_text = self._extract_methods_text(paper, include_bioc=True)
        data_avail, code_avail = self._find_availability_statements(paper, methods_text)
        data_avail = self._ensure_data_availability_text(paper, methods_text, data_avail)

        if not methods_text.strip():
            logger.warning("No methods section found in paper %s", paper.doi or paper.pmid)
            return Method(
                paper_doi=paper.doi,
                data_availability=data_avail,
                code_availability=code_avail,
                raw_methods_text="",
            )

        assay_names = self._identify_assays(methods_text)
        assay_names = self._canonicalize_assay_names(paper, methods_text, assay_names)
        assay_names = _deduplicate_ordered_casefold(assay_names)

        # Classify assays and optionally filter to strictly computational assays only.
        # _classify_assays returns {} on failure; the fallback in the filter loop
        # defaults unknown entries to MethodCategory.computational (inclusive).
        category_map = self._classify_assays(assay_names, methods_text)

        figure_context = _build_figure_context(figures) if figures else {}
        code_refs, code_ref_warnings = self._resolve_code_references(code_avail, methods_text)
        grounded_accessions = self._collect_grounded_accessions(
            paper=paper,
            methods_text=methods_text,
            data_avail=data_avail,
        )

        assays: list[Assay] = []
        warnings: list[str] = list(code_ref_warnings)

        if computational_only:
            filtered_names: list[str] = []
            for n in assay_names:
                # Default to computational when classification is unavailable so
                # that classification failures never silently discard assays.
                cat = category_map.get(n, MethodCategory.computational)
                if cat == MethodCategory.computational:
                    filtered_names.append(n)
                else:
                    msg = (
                        "assay_filtered_non_computational: "
                        f"{n!r} excluded (category={cat.value}, computational_only=True)"
                    )
                    logger.info(msg)
                    warnings.append(msg)
            assay_names = filtered_names

        for name in assay_names:
            method_cat = category_map.get(name, MethodCategory.computational)
            try:
                assay = self._parse_assay(
                    name,
                    methods_text,
                    paper,
                    figure_context,
                    assay_names=assay_names,
                    method_category=method_cat,
                )
                assay = self._apply_code_references(assay, code_refs)
                assay = self._sanitize_assay_data_source(assay, grounded_accessions)
                assays.append(assay)
            except Exception as exc:
                msg = f"assay_stub: {name!r} could not be parsed ({type(exc).__name__}: {exc})"
                logger.warning(msg)
                warnings.append(msg)
                paragraph = _extract_assay_paragraph(
                    methods_text,
                    name,
                    assay_names=assay_names,
                )
                fallback = _fallback_assay_from_text(
                    assay_name=name,
                    paragraph=paragraph,
                    method_category=method_cat,
                )
                assays.append(
                    fallback.model_copy(update={"description": "Could not be parsed."})
                )

        dependencies, dep_warnings = self._identify_dependencies(assay_names, methods_text)
        warnings.extend(dep_warnings)
        if grounded_accessions:
            assays = [self._apply_dataset_sources(a, grounded_accessions) for a in assays]
        assays, rag_warnings = self._infer_missing_computational_parameters(
            assays=assays,
            methods_text=methods_text,
        )
        warnings.extend(rag_warnings)

        return Method(
            paper_doi=paper.doi,
            assay_graph=AssayGraph(assays=assays, dependencies=dependencies),
            data_availability=data_avail,
            code_availability=code_avail,
            raw_methods_text=methods_text,
            parse_warnings=warnings,
        )

    def _ensure_data_availability_text(
        self,
        paper: Paper,
        methods_text: str,
        current_data_avail: str,
    ) -> str:
        """Ensure data availability captures explicit dataset IDs when LLM misses."""
        if _extract_dataset_accessions(current_data_avail):
            return current_data_avail

        candidates: list[str] = []
        for section in paper.sections:
            title = (section.title or "").lower()
            if any(k in title for k in ("data availability", "availability", "accession", "deposition", "data")):
                candidates.append(section.text)
        candidates.append(methods_text)
        if paper.raw_text:
            candidates.append(paper.raw_text)

        for text in candidates:
            accessions = _extract_dataset_accessions(text)
            if accessions:
                joined = ", ".join(accessions)
                return f"Data availability (regex): {joined}."
        return current_data_avail

    def _collect_grounded_accessions(
        self,
        paper: Paper,
        methods_text: str,
        data_avail: str,
    ) -> list[str]:
        """Collect grounded dataset accessions with availability-first priority."""
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            for acc in _extract_dataset_accessions(text):
                key = acc.upper()
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)

        # Priority 1: explicit data-availability extraction
        _add(data_avail)
        # Priority 2: dedicated availability sections in paper
        for section in paper.sections:
            title = (section.title or "").lower()
            if "availability" in title or "accession" in title or "deposition" in title:
                _add(section.text)
        # Priority 3: methods + remaining paper text
        _add(methods_text)
        for section in paper.sections:
            _add(section.text)
        _add(paper.raw_text or "")
        return ordered

    # ── Methods text extraction ──────────────────────────────────────────────

    def _extract_methods_text(self, paper: Paper, include_bioc: bool = False) -> str:
        """Find and return the methods section text.

        Search order:
        1. Sections explicitly flagged as methods by the source parser.
        2. Raw-text extraction bounded by a Methods heading and next top-level heading.
        3. Title-matched sections as a final fallback.

        Returns concatenated text of all matching sections, preserving order.
        """
        # 1: explicit source flags (JATS sec-type or parser metadata)
        matched: list[str] = []
        for section in paper.sections:
            section_type = (getattr(section, "section_type", "") or "").lower().replace("-", " ")
            if getattr(section, "is_methods", False) or "method" in section_type:
                matched.append(section.text)

        if matched:
            result = "\n\n".join(matched)
            return self._merge_bioc_methods_text(result, paper) if include_bioc else result

        # 2: raw-text heading scan (PDF/OCR path)
        if paper.raw_text:
            extracted = _extract_section_by_heading(paper.raw_text, _METHODS_HEADING_LINE_RE)
            if extracted:
                return self._merge_bioc_methods_text(extracted, paper) if include_bioc else extracted

        # 3: title-based fallback when structured flags and raw boundaries are unavailable
        for section in paper.sections:
            if _METHODS_TITLE_RE.search(section.title or ""):
                matched.append(section.text)
        result = "\n\n".join(matched) if matched else ""
        if include_bioc:
            return self._merge_bioc_methods_text(result, paper)
        return result

    def _merge_bioc_methods_text(self, methods_text: str, paper: Paper) -> str:
        """Prepend BioC METHODS passages to prioritize grounded method paragraphs."""
        bioc = getattr(paper, "bioc_context", None)
        if bioc is None:
            return methods_text
        passages = getattr(bioc, "methods", None) or []
        if not passages:
            return methods_text

        seen: set[str] = set()
        bioc_lines: list[str] = []
        for p in passages:
            text = (getattr(p, "text", "") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            bioc_lines.append(text)

        if not bioc_lines:
            return methods_text
        bioc_block = "\n\n".join(bioc_lines)
        if methods_text.strip():
            return f"{bioc_block}\n\n{methods_text}".strip()
        return bioc_block

    # ── Availability statements ──────────────────────────────────────────────

    def _find_availability_statements(
        self, paper: Paper, methods_text: str
    ) -> tuple[str, str]:
        """Extract data-availability and code-availability statements.

        Searches dedicated availability sections first, then falls back to LLM
        extraction from the methods text or abstract.

        Returns:
            (data_statement, code_statement) — either or both may be empty.
        """
        # 1. Look for a dedicated availability section
        avail_texts: list[str] = []
        for section in paper.sections:
            if _AVAIL_TITLE_RE.search(section.title or ""):
                avail_texts.append(section.text)

        combined = "\n\n".join(avail_texts)

        # 2. Also scan methods text for inline availability sentences
        combined = combined or methods_text

        if not combined.strip():
            return "", ""

        try:
            result = _extract_structured_data(
                prompt=(
                    "Extract the data availability statement and the code availability "
                    "statement from this text. If either is absent, return an empty string.\n\n"
                    f"TEXT:\n{combined[:3000]}"
                ),
                output_schema=_AvailabilityStatement,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return result.data_statement, result.code_statement
        except Exception as exc:
            logger.warning("Availability extraction failed: %s", exc)
            return "", ""

    # ── Assay identification ─────────────────────────────────────────────────

    def _identify_assays(self, methods_text: str) -> list[str]:
        """Use LLM to identify all distinct assays/experiments described.

        Returns a flat list of assay name strings, ordered roughly as they
        appear in the methods text.
        """
        if not methods_text.strip():
            return []
        try:
            result = _extract_structured_data(
                prompt=(
                    "List each distinct experimental assay or analysis procedure described "
                    "in this methods section. Be specific but not over-granular: treat "
                    "'RNA-seq library preparation' and 'RNA-seq read alignment' as separate "
                    "entries, but do not split a single method into sub-sentences. "
                    "Include both wet-lab protocols and computational steps.\n\n"
                    f"METHODS TEXT:\n{methods_text[:4000]}"
                ),
                output_schema=_AssayList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return result.assay_names
        except Exception as exc:
            logger.warning("Assay identification failed: %s", exc)
            return []

    def _canonicalize_assay_names(
        self,
        paper: Paper,
        methods_text: str,
        assay_names: list[str],
    ) -> list[str]:
        """Normalize assay naming to stable, section-aware labels."""
        # Generic fallback: if methods text has recognizable heading-like lines,
        # prefer those as assay titles over fuzzy LLM labels.
        headings = _extract_heading_like_lines(methods_text)
        if headings and not assay_names:
            return headings
        return assay_names

    def _classify_assays(
        self,
        assay_names: list[str],
        methods_text: str,
    ) -> dict[str, MethodCategory]:
        """Classify each assay as experimental, computational, or mixed.

        Uses a single LLM call for the entire list (cheaper than per-assay).
        Returns a dict mapping canonical assay name → MethodCategory.

        On any failure (LLM error, schema validation error, empty list) an
        empty dict is returned so that callers can apply a safe fallback.
        LLM-returned names are normalised against the canonical list via the
        same case-insensitive substring matching used elsewhere in the parser.
        """
        if not assay_names:
            return {}

        assay_list = "\n".join(f"- {n}" for n in assay_names)
        try:
            result = _extract_structured_data(
                prompt=(
                    "For each assay in the list below, classify it as "
                    "'experimental', 'computational', or 'mixed'.\n\n"
                    "- experimental: wet-lab protocol or instrument-based measurement "
                    "(cell culture, library prep, immunoprecipitation, UV crosslinking, "
                    "Western blot, FACS, microscopy, etc.).\n"
                    "- computational: bioinformatics or statistical analysis on a computer "
                    "(read alignment, peak calling, differential expression, clustering, "
                    "normalization, variant calling, motif analysis, etc.).\n"
                    "- mixed: the assay entry explicitly spans both wet-lab and computational "
                    "steps.\n\n"
                    "Use the EXACT assay names from the list.\n\n"
                    f"ASSAYS:\n{assay_list}\n\n"
                    f"METHODS TEXT (for context):\n{methods_text[:2000]}"
                ),
                output_schema=_AssayClassificationList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            out: dict[str, MethodCategory] = {}
            for item in result.assays:
                canonical = _normalize_assay_name(item.name, assay_names)
                if canonical is None:
                    canonical = item.name  # best-effort: keep as-is
                try:
                    out[canonical] = MethodCategory(item.method_category)
                except ValueError:
                    logger.warning(
                        "Unknown method_category %r for assay %r — defaulting to 'computational'",
                        item.method_category, item.name,
                    )
                    out[canonical] = MethodCategory.computational
            return out
        except Exception as exc:
            logger.warning("Assay classification failed: %s", exc)
            return {}

    # ── Per-assay parsing ────────────────────────────────────────────────────

    def _parse_assay(
        self,
        assay_name: str,
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        assay_names: Optional[list[str]] = None,
        method_category: MethodCategory = MethodCategory.computational,
    ) -> Assay:
        """Decompose a single assay into ordered AnalysisSteps via LLM.

        Strategy:
        1. Extract the paragraph(s) in methods_text most relevant to assay_name.
        2. Prompt LLM for structured step decomposition with figure links.

        Args:
            method_category: Pre-computed category from _classify_assays.
                Stamped onto the returned Assay so downstream code can filter
                without re-running classification.
        """
        paragraph = _extract_assay_paragraph(
            methods_text,
            assay_name,
            assay_names=assay_names,
        )
        fig_hint = (
            f"\n\nKnown figures: {figure_context}" if figure_context else ""
        )

        try:
            result = _extract_structured_data(
                prompt=(
                    f"Parse the following methods text for the assay '{assay_name}' into "
                    "a structured description with ordered analysis steps.\n\n"
                    "For each step, identify: step_number, description, input_data, "
                    "output_data, software, software_version, parameters, code_reference.\n\n"
                    "Also provide: name, description, data_type "
                    "('sequencing'/'imaging'/'proteomics'/'mass_spec'/'flow_cytometry'/"
                    "'computational'/'other'), raw_data_source (GEO/SRA accession if present), "
                    f"and figures_produced (figure IDs this assay contributes to).{fig_hint}\n\n"
                    f"METHODS TEXT:\n{paragraph[:3000]}"
                ),
                output_schema=_AssayMeta,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            assay = _assay_from_meta(result)
            # Keep the canonical assay title chosen upstream instead of
            # allowing per-assay LLM calls to rename the assay.
            # Also stamp the pre-computed method_category.
            return assay.model_copy(update={
                "name": assay_name,
                "method_category": method_category,
            })
        except Exception as exc:
            logger.warning("Assay parsing failed for %r: %s", assay_name, exc)
            raise

    def _resolve_code_references(
        self,
        code_availability: str,
        methods_text: str,
    ) -> tuple[list[str], list[str]]:
        """Extract and validate code URLs via SoftwareParser._parse_github_code."""
        from researcher_ai.parsers.software_parser import SoftwareParser

        urls = _extract_github_urls(code_availability + "\n" + methods_text)
        if not urls:
            return [], []

        sp = SoftwareParser(llm_model=self.llm_model, cache_dir=None)
        warnings: list[str] = []
        resolved: list[str] = []
        for url in urls:
            try:
                sp._parse_github_code(url)
                resolved.append(url)
            except Exception as exc:
                msg = f"code_reference_unresolved: {url} ({type(exc).__name__}: {exc})"
                logger.warning(msg)
                warnings.append(msg)
        return resolved, warnings

    def _apply_code_references(self, assay: Assay, code_refs: list[str]) -> Assay:
        """Attach code reference URLs to assay steps lacking explicit references."""
        if not code_refs or not assay.steps:
            return assay
        primary = code_refs[0]
        updated_steps: list[AnalysisStep] = []
        for step in assay.steps:
            if step.code_reference:
                updated_steps.append(step)
                continue
            updated_steps.append(step.model_copy(update={"code_reference": primary}))
        return assay.model_copy(update={"steps": updated_steps})

    def _apply_dataset_sources(self, assay: Assay, accessions: list[str]) -> Assay:
        """Attach dataset accession fallback to assays missing raw_data_source."""
        if not accessions:
            return assay
        if assay.raw_data_source and any(acc in assay.raw_data_source for acc in accessions):
            return assay
        if assay.raw_data_source:
            return assay
        primary = accessions[0]
        prefix = "GEO" if primary.startswith(("GSE", "GSM", "GDS")) else "SRA"
        return assay.model_copy(update={"raw_data_source": f"{prefix}: {primary}"})

    def _sanitize_assay_data_source(self, assay: Assay, grounded_accessions: list[str]) -> Assay:
        """Drop or replace hallucinated accession values in assay.raw_data_source."""
        raw = assay.raw_data_source or ""
        if not raw.strip():
            return assay

        reported = _extract_dataset_accessions(raw)
        if not reported:
            return assay

        grounded_set = {a.upper() for a in grounded_accessions}
        valid = [a for a in reported if a.upper() in grounded_set]
        if valid:
            primary = valid[0]
            prefix = "GEO" if primary.startswith(("GSE", "GSM", "GDS")) else "SRA"
            return assay.model_copy(update={"raw_data_source": f"{prefix}: {primary}"})

        # Reported accession(s) are not grounded in source text: remove.
        return assay.model_copy(update={"raw_data_source": None})

    # ── Dependency identification ─────────────────────────────────────────────

    def _identify_dependencies(
        self, assay_names: list[str], methods_text: str
    ) -> tuple[list[AssayDependency], list[str]]:
        """Identify directed dependencies between assays.

        Only meaningful for multi-assay papers. Returns an empty list for
        single-assay papers or when the LLM finds no dependencies.

        Dependency assay names returned by the LLM are normalized against the
        canonical ``assay_names`` list using case-insensitive matching.  Edges
        that reference an unknown assay name are dropped and recorded as
        parse_warnings on the returned Method rather than silently discarded.

        Returns:
            (accepted_deps, warnings) where warnings is a list of human-readable
            strings describing any dropped or degraded edges.
        """
        if len(assay_names) < 2:
            return [], []

        assay_list = "\n".join(f"- {n}" for n in assay_names)
        try:
            result = _extract_structured_data(
                prompt=(
                    "Given the list of assays below and the methods text, identify any "
                    "directed dependencies between assays — i.e., cases where the output "
                    "of one assay is required input for another. "
                    "Return only real data-flow dependencies, not temporal sequencing. "
                    "IMPORTANT: use the exact assay names from the list below.\n\n"
                    f"ASSAYS:\n{assay_list}\n\n"
                    f"METHODS TEXT:\n{methods_text[:3000]}"
                ),
                output_schema=_DependencyList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            deps: list[AssayDependency] = []
            warnings: list[str] = []
            for d in result.dependencies:
                up = _normalize_assay_name(d.upstream_assay, assay_names)
                down = _normalize_assay_name(d.downstream_assay, assay_names)
                if up is None or down is None:
                    msg = (
                        f"dependency_dropped: edge {d.upstream_assay!r} → "
                        f"{d.downstream_assay!r} could not be resolved to canonical assay names"
                    )
                    logger.warning(msg)
                    warnings.append(msg)
                    continue
                deps.append(AssayDependency(
                    upstream_assay=up,
                    downstream_assay=down,
                    dependency_type=d.dependency_type,
                    description=d.description,
                ))
            return deps, warnings
        except Exception as exc:
            logger.warning("Dependency identification failed: %s", exc)
            return [], []

    # ── Phase 3: RAG-style parameter inference ──────────────────────────────

    def search_protocol_docs(self, query: str, top_k: int = 3) -> str:
        """Search local vector/lexical protocol index and return top chunks."""
        store = getattr(self, "protocol_rag", None)
        if store is None:
            store = ProtocolRAGStore()
            self.protocol_rag = store
        return rag_search_protocol_docs(query, top_k=top_k, store=store)

    def _infer_missing_computational_parameters(
        self,
        assays: list[Assay],
        methods_text: str,
    ) -> tuple[list[Assay], list[str]]:
        """Fill missing computational-step parameters using retrieved protocol docs."""
        out: list[Assay] = []
        warnings: list[str] = []
        for assay in assays:
            if assay.method_category != MethodCategory.computational:
                out.append(assay)
                continue
            missing = [s for s in assay.steps if _step_needs_parameter_inference(s)]
            if not missing:
                out.append(assay)
                continue

            steps_block = "\n".join(
                f"- step {s.step_number}: {s.description}; software={s.software or 'unknown'}"
                for s in missing
            )
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_protocol_docs",
                        "description": (
                            "Search local protocol/SOP docs and return relevant excerpts for "
                            "missing computational-method parameters."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
            handlers = {
                "search_protocol_docs": lambda args: self.search_protocol_docs(
                    str(args.get("query", "")),
                    int(args.get("top_k", 3)),
                )
            }
            try:
                inferred = _extract_structured_data_with_tools(
                    model_router=self.llm_model,
                    prompt=(
                        "You are filling missing computational-step parameters for reproducibility.\n"
                        "First identify what is missing. Use the search_protocol_docs tool to retrieve "
                        "relevant SOP/manual context. Then return only grounded inferred parameters.\n\n"
                        f"ASSAY: {assay.name}\n"
                        f"MISSING STEPS:\n{steps_block}\n\n"
                        f"METHODS TEXT:\n{methods_text[:1800]}\n\n"
                        "Call search_protocol_docs before finalizing when any key setting is missing."
                    ),
                    schema=_StepParameterInferenceList,
                    tools=tools,
                    tool_handlers=handlers,
                    system=SYSTEM_METHODS_PARSER,
                )
            except Exception as exc:
                logger.warning("RAG parameter inference failed for assay %r: %s", assay.name, exc)
                out.append(assay)
                continue

            by_step = {u.step_number: u for u in inferred.updates}
            updated_steps: list[AnalysisStep] = []
            applied = 0
            for step in assay.steps:
                if step.parameters and step.software:
                    updated_steps.append(step)
                    continue
                match = by_step.get(step.step_number)
                if match:
                    update_payload: dict[str, Any] = {}
                    if not step.parameters and match.inferred_parameters:
                        clean = {
                            str(k): str(v)
                            for k, v in match.inferred_parameters.items()
                            if str(k).strip()
                        }
                        if clean:
                            update_payload["parameters"] = clean
                    if not step.software and match.inferred_software:
                        update_payload["software"] = str(match.inferred_software)
                    if not step.software_version and match.inferred_software_version:
                        update_payload["software_version"] = str(match.inferred_software_version)
                    if update_payload:
                        updated_steps.append(step.model_copy(update=update_payload))
                        applied += 1
                        continue
                updated_steps.append(step)
            if applied > 0:
                warnings.append(
                    f"inferred_parameters: assay={assay.name!r} updated_steps={applied}"
                )
            out.append(assay.model_copy(update={"steps": updated_steps}))
        return out, warnings


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_assay_name(
    name: str, canonical_names: list[str]
) -> Optional[str]:
    """Match an LLM-returned assay name to the canonical list.

    Returns the canonical name on match, or ``None`` if no match is found.
    Match strategy (in order):
    1. Exact match.
    2. Case-insensitive match.
    3. Case-insensitive substring containment (canonical in name, or name in canonical).
    """
    # 1. Exact
    if name in canonical_names:
        return name
    # 2. Case-insensitive exact
    name_lower = name.lower()
    for cn in canonical_names:
        if cn.lower() == name_lower:
            return cn
    # 3. Substring containment (prefer shortest canonical that contains the name)
    candidates: list[str] = []
    for cn in canonical_names:
        cn_lower = cn.lower()
        if name_lower in cn_lower or cn_lower in name_lower:
            candidates.append(cn)
    if len(candidates) == 1:
        return candidates[0]
    # Ambiguous or no match
    return None


def _extract_section_by_heading(text: str, heading_re: re.Pattern) -> str:
    """Extract text after a matching heading until the next top-level section heading.

    This helper is intentionally conservative about termination so subsection
    titles inside Methods (for example, "Molecular cloning and AAV production")
    do not prematurely end extraction.
    """
    lines = text.split("\n")
    start_idx: Optional[int] = None
    collected: list[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if start_idx is None:
            if heading_re.match(stripped):
                start_idx = idx + 1
            continue

        if stripped and _TOP_LEVEL_SECTION_STOP_RE.match(stripped):
            break
        collected.append(line)

    return "\n".join(collected).strip()


def _extract_assay_paragraph(
    methods_text: str,
    assay_name: str,
    assay_names: Optional[list[str]] = None,
) -> str:
    """Return the paragraph(s) in methods_text most relevant to assay_name.

    Strategy: find paragraphs that contain any keyword token from assay_name.
    Falls back to the full methods text if no paragraph matches.
    """
    # Prefer heading-bounded extraction when the assay appears as a section title.
    heading_block = _extract_assay_block_by_heading(
        methods_text, assay_name, assay_names or []
    )
    if heading_block:
        return heading_block[:3000]

    # Tokenise assay name to meaningful keywords (drop common stop words)
    _STOP = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "by",
              "with", "from", "at", "on", "using", "based", "via"}
    keywords = [
        w.lower() for w in re.split(r"[\s\-/]+", assay_name)
        if len(w) > 2 and w.lower() not in _STOP
    ]

    paragraphs = re.split(r"\n{2,}", methods_text)
    scored: list[tuple[int, str]] = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for kw in keywords if kw in para_lower)
        if score > 0:
            scored.append((score, para))

    if not scored:
        return methods_text[:4000]

    scored.sort(key=lambda x: x[0], reverse=True)
    # Return top 3 paragraphs, capped at 3000 chars
    top = "\n\n".join(p for _, p in scored[:3])
    return top[:3000]


def _build_figure_context(figures: list[Figure]) -> dict[str, str]:
    """Build a compact {figure_id: methods_used} map from parsed figures."""
    ctx: dict[str, str] = {}
    for fig in figures:
        if fig.methods_used:
            ctx[fig.figure_id] = ", ".join(fig.methods_used)
    return ctx


_SOFTWARE_HINT_RE = re.compile(
    r"\b("
    r"STAR|HISAT2|Bowtie2|BWA|minimap2|Cutadapt|Trimmomatic|FastQC|MultiQC|"
    r"samtools|bedtools|featureCounts|RSEM|HTSeq|DESeq2|edgeR|Seurat|scanpy|"
    r"CLIPper|MACS2|MACS3|GATK|Picard|Snakemake|Nextflow|R\b|Python\b"
    r")\b",
    re.IGNORECASE,
)


def _fallback_assay_from_text(
    assay_name: str,
    paragraph: str,
    method_category: MethodCategory,
) -> Assay:
    """Build a deterministic assay fallback when LLM parsing is unavailable."""
    text = (paragraph or "").strip()
    if not text:
        return Assay(
            name=assay_name,
            description="Could not be parsed.",
            data_type="other",
            method_category=method_category,
        )

    software_hint = _infer_software_from_text(text)
    first_sentence = _first_sentence(text)
    output_hint = _infer_output_from_text(text)
    step = AnalysisStep(
        step_number=1,
        description=first_sentence or f"Run assay workflow for {assay_name}.",
        input_data="raw data",
        output_data=output_hint,
        software=software_hint,
        software_version=None,
        parameters={},
        code_reference=None,
    )
    data_type = _infer_data_type_from_text(text)
    return Assay(
        name=assay_name,
        description=first_sentence or f"Heuristic fallback for {assay_name}.",
        data_type=data_type,
        method_category=method_category,
        steps=[step],
    )


def _assay_from_meta(meta: _AssayMeta) -> Assay:
    """Convert LLM-extracted _AssayMeta into a typed Assay model instance.

    Steps are sorted by ``step_number`` to enforce deterministic ordering
    regardless of LLM output order.
    """
    raw_steps = sorted(
        [
            AnalysisStep(
                step_number=s.step_number,
                description=s.description,
                input_data=s.input_data,
                output_data=s.output_data,
                software=s.software,
                software_version=s.software_version,
                parameters=s.parameters,
                code_reference=s.code_reference,
            )
            for s in meta.steps
        ],
        key=lambda s: s.step_number,
    )
    # Deduplicate near-identical steps that can appear from overlapping prompts.
    steps: list[AnalysisStep] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for step in raw_steps:
        sig = (
            step.description.strip().lower(),
            step.input_data.strip().lower(),
            step.output_data.strip().lower(),
            (step.software or "").strip().lower(),
            (step.code_reference or "").strip().lower(),
        )
        if sig in seen:
            continue
        seen.add(sig)
        steps.append(step)

    # Renumber to maintain contiguous ordering after deduplication.
    renumbered = [
        s.model_copy(update={"step_number": idx})
        for idx, s in enumerate(steps, start=1)
    ]
    return Assay(
        name=meta.name,
        description=meta.description,
        data_type=meta.data_type,
        raw_data_source=meta.raw_data_source,
        steps=renumbered,
        figures_produced=meta.figures_produced,
    )


def _extract_heading_like_lines(text: str) -> list[str]:
    """Extract likely subsection headings from methods text."""
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if not line or len(line) > 120:
            continue
        if line.endswith("."):
            continue
        if re.search(r"[;:!?]$", line):
            continue
        if re.search(r"^(data|code)\s+availability$", line, re.IGNORECASE):
            continue
        # Keep title-like lines with at least one alphabetic token.
        if re.search(r"[A-Za-z]", line) and len(line.split()) >= 2:
            key = line.casefold()
            if key not in seen:
                seen.add(key)
                out.append(line)
    return out


def _first_sentence(text: str) -> str:
    """Return the first sentence-like chunk, capped for concise summaries."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    m = re.search(r"(.{1,220}?[.!?])(?:\s|$)", cleaned)
    if m:
        return m.group(1).strip()
    return cleaned[:220]


def _infer_software_from_text(text: str) -> Optional[str]:
    """Heuristically detect a known software/tool mention in free text."""
    m = _SOFTWARE_HINT_RE.search(text or "")
    if not m:
        return None
    value = m.group(1)
    if value.lower() == "bowtie2":
        return "bowtie2"
    return value


def _infer_output_from_text(text: str) -> str:
    """Infer a coarse output artifact label from method paragraph text."""
    lower = (text or "").lower()
    if "bam" in lower or "align" in lower:
        return "aligned BAM files"
    if "peak" in lower:
        return "peak calls"
    if "count" in lower or "expression" in lower:
        return "expression matrix"
    if "qc" in lower:
        return "QC metrics"
    return "processed results"


def _infer_data_type_from_text(text: str) -> str:
    """Infer a best-effort assay data modality from method paragraph text."""
    lower = (text or "").lower()
    if any(k in lower for k in ("rna-seq", "clip", "chip-seq", "atac-seq", "sequencing")):
        return "sequencing"
    if any(k in lower for k in ("image", "microscop", "imaging")):
        return "imaging"
    if any(k in lower for k in ("mass spec", "lc-ms", "proteom")):
        return "mass_spec"
    if any(k in lower for k in ("flow cytometry", "facs")):
        return "flow_cytometry"
    return "computational"


def _extract_assay_block_by_heading(
    methods_text: str,
    assay_name: str,
    assay_names: list[str],
) -> str:
    """Extract text between assay heading and the next assay heading."""
    if not methods_text.strip():
        return ""
    names = _deduplicate_ordered_casefold([assay_name] + list(assay_names))
    # Find heading line start for requested assay.
    target_re = re.compile(
        rf"(?im)^\s*{re.escape(assay_name)}\s*$"
    )
    m = target_re.search(methods_text)
    if not m:
        return ""
    start = m.end()
    # Next heading among known assays
    next_pos = len(methods_text)
    for name in names:
        if name.casefold() == assay_name.casefold():
            continue
        re_next = re.compile(rf"(?im)^\s*{re.escape(name)}\s*$")
        nm = re_next.search(methods_text, pos=start)
        if nm and nm.start() < next_pos:
            next_pos = nm.start()
    block = methods_text[start:next_pos].strip()
    return block


def _extract_github_urls(text: str) -> list[str]:
    """Extract GitHub URLs from free text in first-seen order."""
    urls = re.findall(r"https?://github\.com/[^\s)\]]+", text or "", flags=re.IGNORECASE)
    cleaned: list[str] = []
    seen: set[str] = set()
    for u in urls:
        url = u.rstrip(".,;")
        key = url.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(url)
    return cleaned


def _deduplicate_ordered_casefold(items: list[str]) -> list[str]:
    """Deduplicate strings case-insensitively while preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _extract_dataset_accessions(text: str) -> list[str]:
    """Extract unique dataset accessions in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _DATASET_ACCESSION_RE.finditer(text or ""):
        acc = m.group(1).upper()
        if acc not in seen:
            seen.add(acc)
            out.append(acc)
    return out


def _step_needs_parameter_inference(step: AnalysisStep) -> bool:
    if step.parameters:
        return False
    # Focus on computational tools where defaults materially impact reproducibility.
    return bool(step.software or re.search(r"align|count|differential|cluster|peak", step.description, re.IGNORECASE))
