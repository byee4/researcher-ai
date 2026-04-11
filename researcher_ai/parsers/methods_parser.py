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

import asyncio
import logging
import os
import re
from typing import Any, Optional

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
from researcher_ai.utils.paper_indexer import PaperRAGStore, merge_retrieval_results
from researcher_ai.utils.rag import ProtocolRAGStore
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


class _AssaySkeletonItem(BaseModel):
    """Workflow skeleton stages for one assay."""

    name: str
    stages: list[str] = Field(default_factory=list)


class _AssaySkeletonList(BaseModel):
    """Container for assay skeleton decomposition output."""

    assays: list[_AssaySkeletonItem] = Field(default_factory=list)


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
        protocol_rag: Optional[ProtocolRAGStore] = None,
        paper_rag: Optional[PaperRAGStore] = None,
        rag_docs_dir: Optional[str] = None,
        rag_persist_dir: Optional[str] = None,
        rag_embedding_model: str = "all-MiniLM-L6-v2",
        rag_chunk_size: int = 900,
        rag_chunk_overlap: int = 120,
        rag_lexical_min_token_len: int = 2,
        paper_rag_embedding_model: str = "all-MiniLM-L6-v2",
        paper_rag_chunk_size: int = 900,
        paper_rag_chunk_overlap: int = 120,
        paper_rag_lexical_min_token_len: int = 2,
        assay_parse_concurrency: int = 1,
        assay_parse_base_timeout_seconds: float = llm_utils.DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retrieval_refinement_rounds: int = 2,
    ):
        """Initialize MethodsParser with optional on-disk LLM cache."""
        self.llm_model = llm_model
        self.cache = LLMCache(cache_dir) if cache_dir else None
        self.assay_parse_concurrency = max(1, int(assay_parse_concurrency))
        self.assay_parse_base_timeout_seconds = float(
            max(1.0, float(assay_parse_base_timeout_seconds))
        )
        env_round_cap = os.environ.get("RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS")
        if env_round_cap is not None and str(env_round_cap).strip():
            try:
                max_retrieval_refinement_rounds = int(env_round_cap)
            except ValueError:
                logger.warning(
                    "Invalid RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS=%r, keeping %s",
                    env_round_cap,
                    max_retrieval_refinement_rounds,
                )
        self.max_retrieval_refinement_rounds = max(0, int(max_retrieval_refinement_rounds))
        self._last_skeleton_warnings: list[str] = []
        self.protocol_rag = protocol_rag or ProtocolRAGStore(
            docs_dir=rag_docs_dir,
            persist_dir=rag_persist_dir,
            embedding_model=rag_embedding_model,
            chunk_size=rag_chunk_size,
            chunk_overlap=rag_chunk_overlap,
            lexical_min_token_len=rag_lexical_min_token_len,
        )
        self.paper_rag = paper_rag or PaperRAGStore(
            llm_model=llm_model,
            cache=self.cache,
            embedding_model=paper_rag_embedding_model,
            chunk_size=paper_rag_chunk_size,
            chunk_overlap=paper_rag_chunk_overlap,
            lexical_min_token_len=paper_rag_lexical_min_token_len,
        )

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
        assay_skeletons = self._build_assay_skeletons(assay_names, methods_text)

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
        warnings: list[str] = [
            *getattr(self, "_last_skeleton_warnings", []),
            *code_ref_warnings,
        ]
        try:
            paper_store = getattr(self, "paper_rag", None)
            if paper_store is not None:
                paper_store.build_from(paper=paper, figures=figures or [])
                if getattr(paper_store, "vision_fallback_count", 0) > 0:
                    warnings.append(
                        "paper_rag_vision_fallback: "
                        f"count={paper_store.vision_fallback_count} "
                        f"latency_seconds={paper_store.vision_fallback_latency_seconds:.3f}"
                    )
        except Exception as exc:
            msg = f"paper_index_build_failed: {type(exc).__name__}: {exc}"
            logger.warning(msg)
            warnings.append(msg)

        if computational_only:
            filtered_names: list[str] = []
            _INCLUDE_CATS = {MethodCategory.computational, MethodCategory.mixed}
            for n in assay_names:
                # Default to computational when classification is unavailable so
                # that classification failures never silently discard assays.
                cat = category_map.get(n, MethodCategory.computational)
                if cat in _INCLUDE_CATS:
                    filtered_names.append(n)
                else:
                    msg = (
                        "assay_filtered_non_computational: "
                        f"{n!r} excluded (category={cat.value}, computational_only=True)"
                    )
                    logger.info(msg)
                    warnings.append(msg)
            assay_names = filtered_names

        adaptive_timeout = self._adaptive_assay_timeout_seconds(len(assay_names))
        with llm_utils.temporary_request_timeout(adaptive_timeout):
            parsed_assays, assay_warnings = self._parse_assays(
                assay_names=assay_names,
                category_map=category_map,
                assay_skeletons=assay_skeletons,
                methods_text=methods_text,
                paper=paper,
                figure_context=figure_context,
                code_refs=code_refs,
                grounded_accessions=grounded_accessions,
            )
        assays.extend(parsed_assays)
        warnings.extend(assay_warnings)

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

    def _adaptive_assay_timeout_seconds(self, assay_count: int) -> float:
        """Scale LLM timeout for multi-assay parses to reduce false timeouts."""
        base = float(getattr(self, "assay_parse_base_timeout_seconds", llm_utils.DEFAULT_REQUEST_TIMEOUT_SECONDS))
        count = max(0, int(assay_count))
        if count <= 3:
            return base
        scaled = base * (1.0 + 0.30 * float(count - 3))
        return min(scaled, base * 3.0)

    def _parse_assays(
        self,
        *,
        assay_names: list[str],
        category_map: dict[str, MethodCategory],
        assay_skeletons: dict[str, list[str]],
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        code_refs: list[str],
        grounded_accessions: list[str],
    ) -> tuple[list[Assay], list[str]]:
        assay_parse_concurrency = max(1, int(getattr(self, "assay_parse_concurrency", 1)))
        if assay_parse_concurrency <= 1:
            return self._parse_assays_sequential(
                assay_names=assay_names,
                category_map=category_map,
                assay_skeletons=assay_skeletons,
                methods_text=methods_text,
                paper=paper,
                figure_context=figure_context,
                code_refs=code_refs,
                grounded_accessions=grounded_accessions,
            )

        try:
            asyncio.get_running_loop()
            msg = "async_assay_parse_disabled_running_loop: falling back to sequential assay parsing"
            logger.warning(msg)
            assays, warnings = self._parse_assays_sequential(
                assay_names=assay_names,
                category_map=category_map,
                assay_skeletons=assay_skeletons,
                methods_text=methods_text,
                paper=paper,
                figure_context=figure_context,
                code_refs=code_refs,
                grounded_accessions=grounded_accessions,
            )
            return assays, [msg, *warnings]
        except RuntimeError:
            return asyncio.run(
                self._parse_assays_async(
                    assay_names=assay_names,
                    category_map=category_map,
                    assay_skeletons=assay_skeletons,
                    methods_text=methods_text,
                    paper=paper,
                    figure_context=figure_context,
                    code_refs=code_refs,
                    grounded_accessions=grounded_accessions,
                )
            )

    def _parse_assays_sequential(
        self,
        *,
        assay_names: list[str],
        category_map: dict[str, MethodCategory],
        assay_skeletons: dict[str, list[str]],
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        code_refs: list[str],
        grounded_accessions: list[str],
    ) -> tuple[list[Assay], list[str]]:
        assays: list[Assay] = []
        warnings: list[str] = []
        for idx, name in enumerate(assay_names):
            assay, assay_warnings = self._parse_single_assay(
                assay_name=name,
                assay_names=assay_names,
                category_map=category_map,
                assay_skeletons=assay_skeletons,
                methods_text=methods_text,
                paper=paper,
                figure_context=figure_context,
                code_refs=code_refs,
                grounded_accessions=grounded_accessions,
            )
            assays.append(assay)
            warnings.extend(assay_warnings)
            if _warnings_indicate_rate_limit_or_quota(assay_warnings):
                remaining = assay_names[idx + 1 :]
                if remaining:
                    warnings.append(
                        "assay_parse_circuit_opened: reason=rate_limit_or_quota; "
                        f"remaining_assays={len(remaining)} parsed via text fallback without LLM"
                    )
                for remaining_name in remaining:
                    method_cat = category_map.get(remaining_name, MethodCategory.computational)
                    fallback_assay = self._build_fallback_assay(
                        assay_name=remaining_name,
                        assay_names=assay_names,
                        methods_text=methods_text,
                        method_category=method_cat,
                        code_refs=code_refs,
                        grounded_accessions=grounded_accessions,
                    )
                    assays.append(fallback_assay)
                    warnings.append(
                        "assay_fallback_no_llm_after_circuit: "
                        f"{remaining_name!r} parsed via text fallback"
                    )
                break
        return assays, warnings

    async def _parse_assays_async(
        self,
        *,
        assay_names: list[str],
        category_map: dict[str, MethodCategory],
        assay_skeletons: dict[str, list[str]],
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        code_refs: list[str],
        grounded_accessions: list[str],
    ) -> tuple[list[Assay], list[str]]:
        assay_parse_concurrency = max(1, int(getattr(self, "assay_parse_concurrency", 1)))
        semaphore = asyncio.Semaphore(assay_parse_concurrency)

        async def _worker(idx: int, assay_name: str):
            async with semaphore:
                assay, warnings = await asyncio.to_thread(
                    self._parse_single_assay,
                    assay_name=assay_name,
                    assay_names=assay_names,
                    category_map=category_map,
                    assay_skeletons=assay_skeletons,
                    methods_text=methods_text,
                    paper=paper,
                    figure_context=figure_context,
                    code_refs=code_refs,
                    grounded_accessions=grounded_accessions,
                )
                return idx, assay, warnings

        tasks = [_worker(i, name) for i, name in enumerate(assay_names)]
        resolved = await asyncio.gather(*tasks)
        resolved.sort(key=lambda item: item[0])
        assays = [item[1] for item in resolved]
        warnings: list[str] = []
        for _, _, item_warnings in resolved:
            warnings.extend(item_warnings)
        return assays, warnings

    def _parse_single_assay(
        self,
        *,
        assay_name: str,
        assay_names: list[str],
        category_map: dict[str, MethodCategory],
        assay_skeletons: dict[str, list[str]],
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        code_refs: list[str],
        grounded_accessions: list[str],
    ) -> tuple[Assay, list[str]]:
        method_cat = category_map.get(assay_name, MethodCategory.computational)
        try:
            retrieval_hits, retrieval_warnings = self._iterative_retrieval_loop(
                assay_name=assay_name,
                skeleton_stages=assay_skeletons.get(assay_name) or self._default_stages_for_assay(assay_name),
                max_refinement_rounds=self.max_retrieval_refinement_rounds,
            )
            assay_context = self._render_retrieved_context(retrieval_hits, max_chars=5000)
            assay = self._parse_assay(
                assay_name,
                methods_text,
                paper,
                figure_context,
                assay_names=assay_names,
                method_category=method_cat,
                assay_context=assay_context or None,
            )
            assay = self._apply_code_references(assay, code_refs)
            assay = self._sanitize_assay_data_source(assay, grounded_accessions)
            return assay, retrieval_warnings
        except Exception as exc:
            msg = f"assay_stub: {assay_name!r} could not be parsed ({type(exc).__name__}: {exc})"
            logger.warning(msg)
            fallback = self._build_fallback_assay(
                assay_name=assay_name,
                assay_names=assay_names,
                methods_text=methods_text,
                method_category=method_cat,
                code_refs=code_refs,
                grounded_accessions=grounded_accessions,
            )
            return fallback, [msg]

    def _build_fallback_assay(
        self,
        *,
        assay_name: str,
        assay_names: list[str],
        methods_text: str,
        method_category: MethodCategory,
        code_refs: list[str],
        grounded_accessions: list[str],
    ) -> Assay:
        paragraph = _extract_assay_paragraph(
            methods_text,
            assay_name,
            assay_names=assay_names,
        )
        fallback = _fallback_assay_from_text(
            assay_name=assay_name,
            paragraph=paragraph,
            method_category=method_category,
        )
        fallback = self._apply_code_references(fallback, code_refs)
        fallback = self._sanitize_assay_data_source(fallback, grounded_accessions)
        return fallback

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

        For long methods sections (>4 000 chars), uses a compressed summary
        that preserves all subsection headings plus opening sentences, so that
        assays beyond the old 4 000-char cutoff are still visible to the LLM.
        Heading-extracted names are merged in as a safety net so that clearly
        labelled assays are never silently dropped.
        """
        if not methods_text.strip():
            return []

        # Build a compressed view that fits within the LLM context budget
        # while covering the full breadth of the methods section.
        compressed = _compress_methods_for_identification(methods_text, char_budget=6000)

        try:
            result = _extract_structured_data(
                prompt=(
                    "List each distinct experimental assay or analysis procedure described "
                    "in this methods section. Be specific but not over-granular: treat "
                    "'RNA-seq library preparation' and 'RNA-seq read alignment' as separate "
                    "entries, but do not split a single method into sub-sentences. "
                    "Include both wet-lab protocols and computational steps.\n\n"
                    f"METHODS TEXT:\n{compressed}"
                ),
                output_schema=_AssayList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            llm_names = result.assay_names
        except Exception as exc:
            logger.warning("Assay identification failed: %s", exc)
            llm_names = []

        # Merge heading-extracted assay names that the LLM may have missed.
        heading_names = _extract_heading_like_lines(methods_text)
        return _merge_heading_and_llm_assays(heading_names, llm_names)

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

        # Build per-assay context snippets so the LLM can see computational
        # steps even when they appear late in a long methods section.
        context_parts: list[str] = []
        for name in assay_names:
            block = _extract_assay_block_by_heading(methods_text, name, assay_names)
            if block:
                snippet = _first_n_sentences(block, n=3, max_chars=250)
                context_parts.append(f"[{name}]: {snippet}")
        assay_context = "\n".join(context_parts) if context_parts else methods_text[:3000]

        try:
            result = _extract_structured_data(
                prompt=(
                    "For each assay in the list below, classify it as "
                    "'experimental', 'computational', or 'mixed'.\n\n"
                    "- experimental: purely wet-lab protocol or instrument-based measurement "
                    "with NO computational analysis component "
                    "(cell culture, library prep, immunoprecipitation, UV crosslinking, "
                    "Western blot, FACS, etc.).\n"
                    "- computational: primarily bioinformatics or statistical analysis on a "
                    "computer (read alignment, peak calling, differential expression, "
                    "clustering, normalization, variant calling, motif analysis, etc.).\n"
                    "- mixed: the assay involves BOTH wet-lab/sample-preparation steps AND "
                    "computational analysis (e.g., RNA-seq includes library prep + read "
                    "alignment + differential expression; proteomics includes sample "
                    "digestion + LC-MS/MS + database search; eCLIP includes "
                    "immunoprecipitation + read processing + peak calling). Most "
                    "high-throughput assays (sequencing, mass-spec, imaging with "
                    "quantification) should be classified as 'mixed' because they "
                    "inherently require computational processing of raw instrument data.\n\n"
                    "Use the EXACT assay names from the list.\n\n"
                    f"ASSAYS:\n{assay_list}\n\n"
                    f"ASSAY CONTEXT (opening sentences of each section):\n{assay_context}"
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

    def _build_assay_skeletons(
        self,
        assay_names: list[str],
        methods_text: str,
    ) -> dict[str, list[str]]:
        """Build high-level per-assay stage skeletons for iterative retrieval."""
        self._last_skeleton_warnings = []
        if not assay_names:
            return {}
        assay_list = "\n".join(f"- {name}" for name in assay_names)
        raw_out: dict[str, list[str]] = {}
        try:
            result = _extract_structured_data(
                prompt=(
                    "For each assay below, list concise workflow stages in execution order. "
                    "Use short stage names like qc, trim, align, quantify, differential, peak_call, motif, "
                    "variant_call, filter, annotate. Return 3-8 stages per assay.\n\n"
                    f"ASSAYS:\n{assay_list}\n\n"
                    f"METHODS TEXT:\n{methods_text[:5000]}"
                ),
                output_schema=_AssaySkeletonList,
                system=SYSTEM_METHODS_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            for item in result.assays:
                canonical = _normalize_assay_name(item.name, assay_names) or item.name
                stages = [s.strip().lower() for s in item.stages if str(s).strip()]
                if stages:
                    raw_out[canonical] = _deduplicate_ordered_casefold(stages)
        except Exception as exc:
            logger.warning("Assay skeleton decomposition failed: %s", exc)

        out: dict[str, list[str]] = {}
        for name in assay_names:
            repaired, repair_warnings = self._repair_skeleton_stages(
                assay_name=name,
                llm_stages=raw_out.get(name, []),
            )
            out[name] = repaired
            self._last_skeleton_warnings.extend(repair_warnings)
        return out

    def _assay_template_for_name(self, assay_name: str) -> str:
        """Return template key used for default stage coverage."""
        text = (assay_name or "").casefold()
        if "rbns" in text or "rna bind" in text:
            return "rbns"
        if "ribo" in text:
            return "ribo_seq"
        if "rna" in text:
            return "rna_seq"
        if "clip" in text:
            return "clip_seq"
        if "chip" in text:
            return "chip_seq"
        if "atac" in text:
            return "atac_seq"
        if "variant" in text or "wgs" in text or "wes" in text:
            return "variant"
        return "generic"

    def _default_stages_for_assay(self, assay_name: str) -> list[str]:
        template = self._assay_template_for_name(assay_name)
        if template == "rbns":
            return ["qc", "trim", "align", "quantify", "binding_enrichment", "motif"]
        if template == "ribo_seq":
            return ["qc", "trim", "align", "quantify", "translation_efficiency", "differential"]
        if template == "rna_seq":
            return ["qc", "trim", "align", "quantify", "differential"]
        if template == "chip_seq":
            return ["qc", "trim", "align", "peak_call", "motif"]
        if template == "clip_seq":
            return ["qc", "trim", "align", "analyze"]
        if template == "atac_seq":
            return ["qc", "trim", "align", "peak_call", "differential"]
        if template == "variant":
            return ["qc", "trim", "align", "variant_call", "filter", "annotate"]
        return ["qc", "align", "analyze"]

    def _normalize_stage_name(self, stage: str) -> str:
        """Normalize stage labels to stable tokens for template matching."""
        token = "_".join(re.findall(r"[a-z0-9]+", (stage or "").casefold()))
        if not token:
            return ""
        return _STAGE_ALIAS_MAP.get(token, token)

    def _repair_skeleton_stages(
        self,
        *,
        assay_name: str,
        llm_stages: list[str],
    ) -> tuple[list[str], list[str]]:
        """Repair sparse LLM skeletons to ensure canonical template coverage."""
        normalized_llm = _deduplicate_ordered_casefold(
            [self._normalize_stage_name(s) for s in llm_stages if str(s).strip()]
        )
        defaults = self._default_stages_for_assay(assay_name)

        if not normalized_llm:
            return defaults, []

        missing = [stage for stage in defaults if stage not in normalized_llm]
        if not missing:
            return normalized_llm, []

        repaired = [*normalized_llm, *missing]
        warning = (
            "template_missing_stages: "
            f"assay={assay_name!r} template={self._assay_template_for_name(assay_name)} "
            f"missing={','.join(missing)} source=partial_skeleton"
        )
        return repaired, [warning]

    def _iterative_retrieval_loop(
        self,
        *,
        assay_name: str,
        skeleton_stages: list[str],
        max_refinement_rounds: int = 2,
    ) -> tuple[list[dict[str, object]], list[str]]:
        """Multi-round retrieval with deterministic circuit breakers."""
        collected: list[dict[str, object]] = []
        warnings: list[str] = []
        for stage in skeleton_stages:
            query = f"{assay_name} {stage} software version parameters"
            stage_hits = self._query_evidence_hits(query, top_k=3)
            collected.extend(stage_hits)
            if self._stage_fields_complete(stage_hits, stage):
                continue
            missing = self._detect_missing_fields(stage_hits, stage)
            rounds = 0
            stage_seen: set[str] = set()
            for hit in stage_hits:
                key = str(hit.get("text", "")).strip().lower()[:220]
                if key:
                    stage_seen.add(key)
            for _ in range(max_refinement_rounds):
                if not missing:
                    break
                refined_query = f"{assay_name} {stage} {' '.join(missing)} settings arguments"
                refinement_hits = self._query_evidence_hits(refined_query, top_k=2)
                novel_hits: list[dict[str, object]] = []
                for hit in refinement_hits:
                    key = str(hit.get("text", "")).strip().lower()[:220]
                    if not key or key in stage_seen:
                        continue
                    stage_seen.add(key)
                    novel_hits.append(hit)
                if not novel_hits:
                    warnings.append(
                        "retrieval_refinement_stalled: "
                        f"assay={assay_name!r} stage={stage!r} rounds={rounds} unresolved={','.join(missing)}"
                    )
                    break
                stage_hits.extend(novel_hits)
                collected.extend(novel_hits)
                missing = self._detect_missing_fields(stage_hits, stage)
                rounds += 1
            if missing:
                unresolved = ",".join(missing)
                if missing == ["parameters"]:
                    warnings.append(
                        "retrieval_parameter_gap: "
                        f"assay={assay_name!r} stage={stage!r} rounds={rounds} unresolved={unresolved}"
                    )
                else:
                    warnings.append(
                        "retrieval_circuit_breaker: "
                        f"assay={assay_name!r} stage={stage!r} rounds={rounds} unresolved={unresolved}"
                    )
        deduped: list[dict[str, object]] = []
        seen: set[str] = set()
        for hit in collected:
            text = str(hit.get("text", "")).strip()
            if not text:
                continue
            key = text[:220].lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped, warnings

    def _query_evidence_hits(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        store = getattr(self, "protocol_rag", None)
        if store is None:
            store = ProtocolRAGStore()
            self.protocol_rag = store
        try:
            protocol_raw = store.query(query, top_k=max(top_k, 3))
        except Exception:
            protocol_raw = []
        protocol_hits = protocol_raw if isinstance(protocol_raw, list) else []
        paper_store = getattr(self, "paper_rag", None)
        if paper_store is not None:
            try:
                paper_raw = paper_store.query(query, top_k=max(top_k, 3))
            except Exception:
                paper_raw = []
            paper_hits = paper_raw if isinstance(paper_raw, list) else []
        else:
            paper_hits = []
        return merge_retrieval_results(
            paper_hits=paper_hits,
            protocol_hits=protocol_hits,
            top_k=top_k,
            paper_bias=0.10,
            protocol_bias=0.0,
        )

    def _stage_required_fields(self, stage: str) -> list[str]:
        lower = (stage or "").lower()
        if any(k in lower for k in ("align", "quant", "peak", "call", "variant", "differential", "annotate")):
            return ["software", "parameters"]
        return ["software"]

    def _detect_missing_fields(
        self,
        hits: list[dict[str, object]],
        stage: str,
    ) -> list[str]:
        required = self._stage_required_fields(stage)
        text = " ".join(str(h.get("text", "")) for h in hits)
        missing: list[str] = []
        if "software" in required and _SOFTWARE_HINT_RE.search(text) is None:
            missing.append("software")
        if "parameters" in required:
            has_param = bool(re.search(r"--[A-Za-z0-9_\-]+", text))
            if not has_param:
                has_param = bool(re.search(r"\b[A-Za-z][A-Za-z0-9_]+\s*=\s*[^,\s;]+", text))
            if not has_param:
                has_param = bool(_PARAMETER_HINT_RE.search(text))
            if not has_param:
                missing.append("parameters")
        return missing

    def _stage_fields_complete(
        self,
        hits: list[dict[str, object]],
        stage: str,
    ) -> bool:
        return len(self._detect_missing_fields(hits, stage)) == 0

    def _render_retrieved_context(
        self,
        hits: list[dict[str, object]],
        *,
        max_chars: int = 5000,
    ) -> str:
        if not hits:
            return ""
        parts: list[str] = []
        for hit in hits:
            source = str(hit.get("source", "unknown"))
            source_type = str(hit.get("source_type", "unknown"))
            chunk_type = hit.get("chunk_type")
            chunk_tag = f" chunk_type={chunk_type}" if chunk_type else ""
            txt = str(hit.get("text", "")).strip()
            if not txt:
                continue
            parts.append(f"[{source_type}:{source}{chunk_tag}] {txt}")
            if sum(len(p) for p in parts) >= max_chars:
                break
        combined = "\n\n".join(parts)
        return combined[:max_chars]

    # ── Per-assay parsing ────────────────────────────────────────────────────

    def _parse_assay(
        self,
        assay_name: str,
        methods_text: str,
        paper: Paper,
        figure_context: dict[str, str],
        assay_names: Optional[list[str]] = None,
        method_category: MethodCategory = MethodCategory.computational,
        assay_context: Optional[str] = None,
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
        paragraph = assay_context or _extract_assay_paragraph(
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
        """Search merged paper+protocol evidence with paper-first ranking."""
        merged = self._query_evidence_hits(query, top_k=top_k)
        if not merged:
            return "No protocol documents matched the query."
        lines: list[str] = []
        for i, h in enumerate(merged, start=1):
            source = str(h.get("source", "unknown"))
            source_type = str(h.get("source_type", "protocol"))
            chunk_type = h.get("chunk_type")
            extra = f" chunk_type={chunk_type}" if chunk_type else ""
            lines.append(
                f"[{i}] source={source} source_type={source_type}{extra}\n{h.get('text', '')}"
            )
        return "\n\n".join(lines)

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
            missing = self._collect_missing_parameter_steps(assay)
            if not missing:
                out.append(assay)
                continue

            steps_block = "\n".join(
                f"- step {s.step_number}: {s.description}; software={s.software or 'unknown'}"
                for s in missing
            )
            rag_query = self._build_rag_query_for_steps(assay.name, missing)
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
                            "required": ["query", "top_k"],
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
                        f"SUGGESTED RAG QUERY: {rag_query}\n"
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
                logger.warning(
                    "Tool-calling RAG parameter inference failed for assay %r: %s. "
                    "Falling back to non-tool RAG prompt.",
                    assay.name,
                    exc,
                )
                try:
                    rag_context = self.search_protocol_docs(rag_query, top_k=3)
                    inferred = _extract_structured_data(
                        prompt=(
                            "Infer missing computational-step parameters from grounded protocol context. "
                            "Only include parameters/software supported by the retrieved context.\n\n"
                            f"ASSAY: {assay.name}\n"
                            f"MISSING STEPS:\n{steps_block}\n\n"
                            f"METHODS TEXT:\n{methods_text[:1800]}\n\n"
                            f"PROTOCOL CONTEXT:\n{rag_context}"
                        ),
                        output_schema=_StepParameterInferenceList,
                        system=SYSTEM_METHODS_PARSER,
                        model=self.llm_model,
                        cache=self.cache,
                    )
                    warnings.append(
                        f"inferred_parameters_fallback_mode: assay={assay.name!r} mode=non_tool_rag"
                    )
                except Exception as fallback_exc:
                    logger.warning(
                        "RAG parameter inference failed for assay %r even after non-tool fallback: %s",
                        assay.name,
                        fallback_exc,
                    )
                    out.append(assay)
                    continue

            updates = getattr(inferred, "updates", None)
            if not isinstance(updates, list):
                updates = []
                warnings.append(
                    f"inferred_parameters_invalid_schema: assay={assay.name!r}"
                )
            by_step = {
                u.step_number: u
                for u in updates
                if hasattr(u, "step_number")
            }
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

    def _collect_missing_parameter_steps(self, assay: Assay) -> list[AnalysisStep]:
        """Return computational steps that are missing critical reproducibility fields."""
        return [s for s in assay.steps if _step_needs_parameter_inference(s)]

    def _build_rag_query_for_steps(self, assay_name: str, steps: list[AnalysisStep]) -> str:
        software_names = [
            s.software.strip()
            for s in steps
            if s.software and s.software.strip()
        ]
        if software_names:
            return f"{assay_name} {' '.join(software_names)} default parameters reproducibility"
        return f"{assay_name} computational workflow default parameters reproducibility"


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


def _is_rate_limit_or_quota_error_text(text: str) -> bool:
    message = (text or "").lower()
    indicators = (
        "ratelimiterror",
        "rate limit",
        "too many requests",
        "429",
        "quota",
        "insufficient_quota",
        "exceeded your current quota",
    )
    return any(token in message for token in indicators)


def _warnings_indicate_rate_limit_or_quota(warnings: list[str]) -> bool:
    return any(_is_rate_limit_or_quota_error_text(str(w)) for w in warnings)


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

_PARAMETER_HINT_RE = re.compile(
    r"(?:"
    r"\b(?:fdr|p(?:-?value)?|q(?:-?value)?|alpha|beta|threshold|cutoff|"
    r"minimum|min|max(?:imum)?|window|bin|threads?|runthreadn|mapq|quality|score|"
    r"reads?|depth|coverage|fold(?:\s*change)?|rpm|rpkm|tpm)\b"
    r"[^.;,\n]{0,24}?"
    r"(?:<=|>=|=|<|>|of\s+)?\s*"
    r"\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    r"(?:\s*(?:bp|nt|kb|mb|gb|%))?"
    r")"
    r"|(?:\bn\s*=\s*\d+\b)"
    r"|(?:\bp\s*[<=>]\s*\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)",
    re.IGNORECASE,
)

_STAGE_ALIAS_MAP: dict[str, str] = {
    "quality_control": "qc",
    "quality_check": "qc",
    "qc": "qc",
    "trim": "trim",
    "trimming": "trim",
    "adapter_trim": "trim",
    "adapter_trimming": "trim",
    "align": "align",
    "alignment": "align",
    "map": "align",
    "mapping": "align",
    "quantification": "quantify",
    "quantify": "quantify",
    "count": "quantify",
    "counts": "quantify",
    "abundance": "quantify",
    "differential_expression": "differential",
    "differential": "differential",
    "de": "differential",
    "peak_calling": "peak_call",
    "peak_call": "peak_call",
    "peak": "peak_call",
    "motif_enrichment": "motif",
    "motif_analysis": "motif",
    "motif": "motif",
    "variant_calling": "variant_call",
    "variant_call": "variant_call",
    "annotate": "annotate",
    "annotation": "annotate",
    "filtering": "filter",
    "filter": "filter",
    "dedup": "deduplicate",
    "deduplicate": "deduplicate",
    "normalization": "normalize",
    "normalize": "normalize",
    "translation_efficiency": "translation_efficiency",
    "te": "translation_efficiency",
    "binding_enrichment": "binding_enrichment",
}


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
    """Extract likely subsection headings from methods text.

    Headings are short lines (≤120 chars) that lack sentence-ending
    punctuation, contain at least one alphabetic character, and are preceded
    by a blank line (or appear at the start of the text).  Both multi-word
    headings ("Bisulfite sequencing") and single-word headings
    ("Proteomics", "eCLIP", "MEA", "RNA-seq") are accepted when they look
    like standalone titles (≤40 chars, no lower-case-only common words).
    """
    raw_lines = text.splitlines()
    out: list[str] = []
    seen: set[str] = set()

    # Common English words that should NOT be treated as headings when alone.
    _SINGLE_WORD_BLOCKLIST = {
        "methods", "results", "discussion", "conclusions", "abstract",
        "introduction", "references", "acknowledgements", "funding",
        "the", "and", "for", "with", "from", "that", "this", "were",
        "was", "are", "been", "also", "then", "next", "after", "before",
    }

    for idx, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line or len(line) > 120:
            continue
        if line.endswith("."):
            continue
        if re.search(r"[;:!?]$", line):
            continue
        if re.search(r"^(data|code)\s+availability$", line, re.IGNORECASE):
            continue
        if re.search(r"^(statistics|reporting|online\s+content)", line, re.IGNORECASE):
            continue
        if not re.search(r"[A-Za-z]", line):
            continue

        # Headings should be preceded by a blank line (or be at the start).
        # This prevents wrapped sentence fragments from being treated as headings.
        prev_blank = (idx == 0) or (raw_lines[idx - 1].strip() == "")
        if not prev_blank:
            continue

        # Reject lines with sentence-internal periods (e.g., "performed at 400 mJ/cm2. Immunoprecipitation")
        # but allow periods in section numbers (e.g., "3.5. Data analysis") and
        # abbreviations (e.g., "v.2.1", "AP-MS").
        # The pattern requires at least 3 chars before the ". X" to avoid
        # matching section-number prefixes like "3.5. ".
        if re.search(r"[a-z]{3,}\.\s+[A-Z]", line):
            continue

        words = line.split()
        is_heading = False

        if len(words) >= 2:
            is_heading = True
        elif len(words) == 1 and len(line) <= 40:
            word_lower = line.lower()
            if word_lower not in _SINGLE_WORD_BLOCKLIST:
                is_heading = True

        if is_heading:
            key = line.casefold()
            if key not in seen:
                seen.add(key)
                out.append(line)

    return out


def _compress_methods_for_identification(methods_text: str, char_budget: int = 6000) -> str:
    """Build a compressed summary of the methods section for assay identification.

    For long methods sections (common in multi-omic papers), naive truncation
    misses assays described beyond the cutoff.  This function extracts every
    subsection heading together with the first 1–2 sentences of each subsection
    so the LLM can see the *full breadth* of assays within the char budget.

    Strategy:
    1. Split text into heading + body blocks using ``_extract_heading_like_lines``
       heuristics.
    2. For each block, keep the heading and the opening sentence(s).
    3. If the result still exceeds *char_budget*, trim body snippets
       proportionally while always preserving all headings.

    Falls back to the raw text (truncated) when no heading structure is detected.
    """
    if len(methods_text) <= char_budget:
        return methods_text

    lines = methods_text.splitlines()
    headings = _extract_heading_like_lines(methods_text)

    if not headings:
        # No recognisable heading structure — fall back to truncation.
        return methods_text[:char_budget]

    heading_set = {h.casefold() for h in headings}

    # Group lines into (heading, body_lines) blocks.
    blocks: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.casefold() in heading_set:
            if current_heading or current_body:
                blocks.append((current_heading, current_body))
            current_heading = stripped
            current_body = []
        else:
            if stripped:
                current_body.append(stripped)
    if current_heading or current_body:
        blocks.append((current_heading, current_body))

    # Build compressed version: heading + first 2 sentences of body.
    compressed_parts: list[str] = []
    for heading, body in blocks:
        if heading:
            compressed_parts.append(f"\n{heading}")
        # Take opening sentences (up to ~300 chars) for context.
        body_text = " ".join(body)
        snippet = _first_n_sentences(body_text, n=2, max_chars=300)
        if snippet:
            compressed_parts.append(snippet)

    result = "\n".join(compressed_parts).strip()

    if len(result) <= char_budget:
        return result

    # Over budget — progressively trim body snippets.
    for max_chars in (200, 120, 60, 0):
        trimmed: list[str] = []
        for heading, body in blocks:
            if heading:
                trimmed.append(f"\n{heading}")
            if max_chars > 0:
                body_text = " ".join(body)
                snippet = _first_n_sentences(body_text, n=1, max_chars=max_chars)
                if snippet:
                    trimmed.append(snippet)
        result = "\n".join(trimmed).strip()
        if len(result) <= char_budget:
            return result

    # Last resort: headings only.
    return "\n".join(headings)[:char_budget]


def _first_n_sentences(text: str, n: int = 2, max_chars: int = 300) -> str:
    """Extract the first *n* sentences from *text*, capped at *max_chars*."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    parts: list[str] = []
    remaining = cleaned
    for _ in range(n):
        m = re.search(r"(.+?[.!?])(?:\s|$)", remaining)
        if not m:
            break
        parts.append(m.group(1).strip())
        remaining = remaining[m.end():]
    result = " ".join(parts) if parts else cleaned[:max_chars]
    return result[:max_chars]


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


def _merge_heading_and_llm_assays(
    heading_names: list[str],
    llm_names: list[str],
) -> list[str]:
    """Merge heading-extracted assay names with LLM-identified assay names.

    The LLM list is authoritative for ordering and naming, but headings act as
    a safety net: any heading-derived name not already covered by the LLM list
    (via case-insensitive substring matching or high token overlap) is appended
    at the end.

    This ensures that assays with clear section headings in the methods text
    are never silently dropped due to LLM context-window limitations.
    """
    if not heading_names:
        return llm_names
    if not llm_names:
        return heading_names

    def _tokens(name: str) -> set[str]:
        stop_tokens = {
            "and", "or", "the", "of", "for", "to", "in", "on", "with", "by",
            "analysis", "assay", "assays", "method", "methods", "section",
        }
        return {
            t for t in re.findall(r"[a-z0-9]+", (name or "").casefold())
            if t not in stop_tokens
        }

    def _is_covered(heading: str, llm_name: str) -> bool:
        h_lower = heading.casefold()
        l_lower = llm_name.casefold()
        if h_lower in l_lower or l_lower in h_lower:
            return True

        h_tokens = _tokens(heading)
        l_tokens = _tokens(llm_name)
        if not h_tokens or not l_tokens:
            return False

        overlap = len(h_tokens & l_tokens)
        if len(h_tokens) <= 2:
            return overlap == len(h_tokens)
        return overlap >= 2 and (overlap / len(h_tokens)) >= 0.5

    merged = list(llm_names)
    for heading in heading_names:
        # Check if any LLM name already covers this heading.
        covered = any(_is_covered(heading, llm_name) for llm_name in merged)
        if not covered:
            merged.append(heading)
    return merged


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
    has_parameters = bool(step.parameters)
    has_software = bool(step.software and step.software.strip())
    if has_parameters:
        return False
    if has_software:
        # Missing parameters with known software should trigger inference.
        return True
    # Focus on computational tools where defaults materially impact reproducibility.
    return bool(step.software or re.search(r"align|count|differential|cluster|peak", step.description, re.IGNORECASE))
