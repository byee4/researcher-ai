"""Paper parser: load a publication and produce a structured Paper object.

Supports loading from: PDF file, URL, PMID, PMCID, DOI.

Parsing strategy (ordered by data richness):

- If source is a PMCID or resolves to one, fetch JATS full-text XML.
  Parse sections, figures, and references directly from structured XML.
- If source is a PMID, fetch PubMed XML for metadata, then resolve to PMCID
  for full text. Fall back to LLM parsing of abstract-only if no PMCID exists.
- If source is a DOI, resolve to PMID, then follow the PMID path.
- If source is a PDF, extract text with pdfplumber, then use LLM parsing.
- If source is a URL, fetch HTML text, then use LLM parsing.

LLM parsing is used for:

- Section segmentation and text extraction (PDF / URL sources)
- Paper metadata extraction when JATS XML is unavailable
- Supplementary item detection
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from researcher_ai.models.paper import (
    BioCContext,
    BioCPassageContext,
    Paper,
    PaperSource,
    PaperType,
    Reference,
    Section,
    SupplementaryItem,
)
from researcher_ai.utils import llm as llm_utils
from researcher_ai.utils.llm import LLMCache, ask_claude_structured, SYSTEM_PAPER_PARSER
from researcher_ai.utils.pdf import (
    extract_text_from_pdf,
    extract_figure_ids_from_text,
    split_text_into_sections,
)
from researcher_ai.utils.pubmed import (
    fetch_article_xml,
    fetch_bioc_json_for_paper,
    extract_bioc_passages,
    make_bioc_section_selector,
    bioc_methods_section_selector,
    bioc_results_section_selector,
    map_bioc_figure_id,
    fetch_pmc_fulltext,
    parse_pubmed_xml,
    parse_jats_xml,
    resolve_doi_to_pmid,
    resolve_pmid_to_pmcid,
    resolve_pmcid_to_pmid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured extraction schemas (Pydantic, used with ask_claude_structured)
# ---------------------------------------------------------------------------

class _HeaderMeta(BaseModel):
    """Extracted paper header metadata."""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None


class _ExtractedSection(BaseModel):
    title: str
    text: str
    figures_referenced: list[str] = Field(default_factory=list)


class _ExtractedSections(BaseModel):
    """List of sections extracted from paper text."""
    sections: list[_ExtractedSection] = Field(default_factory=list)
    supplementary_urls: list[str] = Field(default_factory=list)


class _ExtractedReference(BaseModel):
    ref_id: str
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None


class _ExtractedReferences(BaseModel):
    references: list[_ExtractedReference] = Field(default_factory=list)


class _PaperTypeClassification(BaseModel):
    paper_type: str = Field(
        description=(
            "One of: experimental, computational, multi_omic, review, "
            "clinical, reanalysis"
        )
    )
    reasoning: str = ""


class _SupplementaryItems(BaseModel):
    items: list[_SupplementaryItemMeta] = Field(default_factory=list)


class _SupplementaryItemMeta(BaseModel):
    item_id: str
    label: str
    url: Optional[str] = None
    file_type: Optional[str] = None
    data_content: Optional[str] = None


# ---------------------------------------------------------------------------
# PaperParser
# ---------------------------------------------------------------------------

class PaperParser:
    """Parse a scientific paper into a structured Paper model.

    Supports: PDF file, URL, PMID, PMCID, DOI.
    Uses a hybrid approach: structured XML parsing when available,
    LLM-assisted parsing for PDF and URL sources.

    Args:
        llm_model: Claude model identifier for LLM-assisted parsing.
        cache_dir: Optional directory for caching LLM responses. Recommended
                   during development to avoid re-running API calls.
    """

    def __init__(
        self,
        llm_model: str = llm_utils.DEFAULT_MODEL,
        cache_dir: Optional[str | Path] = None,
    ):
        """Initialize PaperParser with optional response caching."""
        self.llm_model = llm_model
        self.cache = LLMCache(cache_dir) if cache_dir else None

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(
        self,
        source: str,
        source_type: Optional[PaperSource] = None,
    ) -> Paper:
        """Parse a paper from any supported source.

        Args:
            source: File path, URL, PMID, PMCID, or DOI string.
            source_type: Explicit source type. Auto-detected if None.

        Returns:
            Structured Paper object.
        """
        if source_type is None:
            source_type = self._detect_source_type(source)

        logger.info("Parsing paper from %s (type: %s)", source, source_type.value)

        if source_type == PaperSource.PMCID:
            return self._parse_from_pmcid(source, source)
        elif source_type == PaperSource.PMID:
            return self._parse_from_pmid(source)
        elif source_type == PaperSource.DOI:
            return self._parse_from_doi(source)
        elif source_type == PaperSource.PDF:
            return self._parse_from_pdf(source)
        elif source_type == PaperSource.URL:
            return self._parse_from_url(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    # ── Source type detection ───────────────────────────────────────────────

    def _detect_source_type(self, source: str) -> PaperSource:
        """Auto-detect source type from the source string.

        Detection order:
        1. PMC prefix → PMCID
        2. Starts with 10. (DOI prefix) → DOI
        3. http/https URL → URL
        4. .pdf file extension → PDF
        5. Exists as a file → PDF
        6. All digits → PMID
        7. Default → PMID (attempt)
        """
        s = source.strip()

        if re.match(r"^PMC\d+$", s, re.IGNORECASE):
            return PaperSource.PMCID

        if s.startswith("10.") or re.match(r"^https?://(?:dx\.)?doi\.org/", s):
            return PaperSource.DOI

        if s.startswith("http://") or s.startswith("https://"):
            return PaperSource.URL

        if s.lower().endswith(".pdf"):
            return PaperSource.PDF

        if Path(s).is_file():
            raise ValueError(
                f"File exists but is not a PDF: '{s}'. "
                "Only .pdf files are supported for file-based parsing."
            )

        if re.match(r"^\d{7,9}$", s):
            return PaperSource.PMID

        # Last resort: treat as PMID
        logger.warning("Could not auto-detect source type for '%s', assuming PMID", s)
        return PaperSource.PMID

    # ── Source-specific loaders ─────────────────────────────────────────────

    def _parse_from_pmcid(self, pmcid: str, source_path: str) -> Paper:
        """Fetch and parse JATS full-text XML from PMC.

        Fallback: if PMC fetch fails, resolve PMCID → PMID via elink and
        delegate to the PMID parsing path before returning a stub.
        """
        logger.debug("Fetching PMC full-text for %s", pmcid)
        try:
            xml_text = fetch_pmc_fulltext(pmcid)
            parsed = parse_jats_xml(xml_text)
            paper = self._build_paper_from_jats(parsed, source_path, PaperSource.PMCID)
            return self._attach_bioc_context(paper)
        except Exception as exc:
            logger.warning("PMC full-text fetch failed for %s: %s. Trying PMID fallback.", pmcid, exc)

        # Attempt PMCID → PMID resolution via elink, then full PMID path
        try:
            pmid = resolve_pmcid_to_pmid(pmcid)
            if pmid:
                logger.info("Resolved %s → PMID %s, falling back to PMID path", pmcid, pmid)
                return self._parse_from_pmid(pmid)
        except Exception as exc:
            logger.warning("PMCID→PMID resolution failed for %s: %s", pmcid, exc)

        return self._build_paper_stub(
            title=pmcid,
            source=PaperSource.PMCID,
            source_path=source_path,
            raw_text="",
        )

    def _parse_from_pmid(self, pmid: str) -> Paper:
        """Fetch PubMed metadata and try to resolve to full-text."""
        logger.debug("Fetching PubMed metadata for PMID %s", pmid)
        try:
            xml_text = fetch_article_xml(pmid)
            meta = parse_pubmed_xml(xml_text)
        except Exception as exc:
            logger.warning("PubMed fetch failed for %s: %s", pmid, exc)
            meta = {"pmid": pmid}

        # Try to get full text via PMCID.
        # If PubMed metadata PMCID fails, re-resolve via elink and retry once.
        pmcid_meta = meta.get("pmcid")
        pmcid_candidates: list[str] = []
        if pmcid_meta:
            pmcid_candidates.append(str(pmcid_meta))

        try:
            pmcid_resolved = resolve_pmid_to_pmcid(pmid)
            if pmcid_resolved and pmcid_resolved not in pmcid_candidates:
                pmcid_candidates.append(pmcid_resolved)
        except Exception as exc:
            logger.warning("PMID→PMCID re-resolution failed for %s: %s", pmid, exc)

        for pmcid in pmcid_candidates:
            logger.debug("Resolved PMID %s → %s, fetching full text", pmid, pmcid)
            try:
                xml_text = fetch_pmc_fulltext(pmcid)
                jats = parse_jats_xml(xml_text)
                # Merge PubMed metadata (more reliable for authors/abstract)
                if meta.get("authors"):
                    jats["authors"] = meta["authors"]
                if meta.get("abstract"):
                    jats["abstract"] = meta["abstract"]
                jats.setdefault("pmid", pmid)
                paper = self._build_paper_from_jats(jats, pmid, PaperSource.PMID)
                return self._attach_bioc_context(paper)
            except Exception as exc:
                logger.warning(
                    "PMC full-text failed for PMID %s via %s: %s", pmid, pmcid, exc
                )

        # No full text available — build from PubMed metadata only
        paper = self._build_paper_from_pubmed_meta(meta, pmid)
        return self._attach_bioc_context(paper)

    def _parse_from_doi(self, doi: str) -> Paper:
        """Resolve DOI → PMID → full parse."""
        logger.debug("Resolving DOI %s to PMID", doi)
        pmid = resolve_doi_to_pmid(doi)
        if pmid:
            paper = self._parse_from_pmid(pmid)
            # Ensure DOI is set even if PubMed didn't return it
            if not paper.doi:
                paper = paper.model_copy(update={"doi": doi})
            return paper
        else:
            logger.warning("Could not resolve DOI %s to PMID", doi)
            return self._build_paper_stub(
                title=doi,
                source=PaperSource.DOI,
                source_path=doi,
                raw_text="",
            )

    def _parse_from_pdf(self, pdf_path: str) -> Paper:
        """Extract text from PDF and use LLM to parse."""
        logger.debug("Parsing PDF: %s", pdf_path)
        raw_text = extract_text_from_pdf(pdf_path)
        return self._parse_raw_text(raw_text, pdf_path, PaperSource.PDF)

    def _parse_from_url(self, url: str) -> Paper:
        """Fetch URL content and use LLM to parse."""
        import httpx
        logger.debug("Fetching URL: %s", url)
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            # Strip HTML tags if HTML response
            content_type = response.headers.get("content-type", "")
            if "html" in content_type:
                raw_text = _strip_html(response.text)
            else:
                raw_text = response.text
        except Exception as exc:
            logger.error("Failed to fetch URL %s: %s", url, exc)
            raw_text = ""

        return self._parse_raw_text(raw_text, url, PaperSource.URL)

    # ── Core LLM-based text parser ──────────────────────────────────────────

    def _parse_raw_text(
        self,
        raw_text: str,
        source_path: str,
        source_type: PaperSource,
    ) -> Paper:
        """Use LLM to extract structured Paper from raw text.

        Three-call strategy:
        1. Extract header metadata (title, authors, abstract, DOI) from the
           first 3000 chars where the header information is densest.
        2. Extract sections from full text.
        3. Extract references from the end of the document.
        """
        if not raw_text.strip():
            return self._build_paper_stub(
                title=source_path, source=source_type,
                source_path=source_path, raw_text=raw_text,
            )

        # 1. Header metadata
        header_text = raw_text[:3000]
        meta = self._extract_header_meta(header_text)

        # 2. Sections (regex-first, LLM fallback)
        sections = self._extract_sections(raw_text)

        # 3. Figure IDs (regex — reliable without LLM)
        figure_ids = _main_figure_ids_only(extract_figure_ids_from_text(raw_text))

        # 4. References — find the reference section boundary first,
        #    fall back to last 3000 chars if no boundary found
        ref_text = self._extract_reference_section_text(raw_text)
        references = self._extract_references_llm(ref_text)

        # 5. Paper type classification
        paper_type = self._classify_paper_type(meta.abstract or raw_text[:500])

        # 6. Supplementary items — two-pass approach:
        #    a) LLM extraction from supplement/data-availability sections
        #    b) Regex scan over full text for inline Table S1, Figure S2 etc.
        supp_section_text = ""
        for sec in sections:
            if any(k in sec.title.lower() for k in
                   ["supplement", "data avail", "code avail"]):
                supp_section_text += sec.text + "\n"
        supp_items = self._extract_supplementary_items(supp_section_text)

        # Merge regex-detected supplementary refs from full text
        regex_supp = self._detect_supplementary_refs_regex(raw_text)
        existing_ids = {item.item_id for item in supp_items}
        for item in regex_supp:
            if item.item_id not in existing_ids:
                supp_items.append(item)
                existing_ids.add(item.item_id)

        supp_urls = [item.url for item in supp_items if item.url]

        return Paper(
            title=meta.title or source_path,
            authors=meta.authors,
            abstract=meta.abstract,
            doi=meta.doi,
            source=source_type,
            source_path=source_path,
            paper_type=paper_type,
            sections=sections,
            references=references,
            raw_text=raw_text,
            figure_ids=figure_ids,
            supplementary_urls=supp_urls,
            supplementary_items=supp_items,
        )

    # ── Build from structured data ──────────────────────────────────────────

    def _build_paper_from_jats(
        self,
        jats: dict,
        source_path: str,
        source_type: PaperSource,
    ) -> Paper:
        """Convert a parsed JATS dict into a Paper model."""
        sections = [
            Section(
                title=s["title"],
                text=s["text"],
                section_type=s.get("section_type"),
                is_methods=bool(s.get("is_methods", False)),
                figures_referenced=_normalize_figure_ids(
                    extract_figure_ids_from_text(s["text"])
                ),
            )
            for s in jats.get("sections", [])
        ]

        references = [
            Reference(
                ref_id=r.get("ref_id", ""),
                title=r.get("title"),
                authors=r.get("authors", []),
                journal=r.get("journal"),
                year=r.get("year"),
                doi=r.get("doi"),
            )
            for r in jats.get("references", [])
        ]

        # Figure captions from JATS <fig> elements (label → caption text)
        figure_captions: dict[str, str] = jats.get("figure_captions", {})

        # Figure IDs from captions dict and in-text references
        caption_ids = _main_figure_ids_only(_normalize_figure_ids(list(figure_captions.keys())))
        full_text = " ".join(s["text"] for s in jats.get("sections", []))
        intext_ids = _main_figure_ids_only(_normalize_figure_ids(extract_figure_ids_from_text(full_text)))
        figure_ids = _deduplicate_ordered(caption_ids + intext_ids)

        # Paper type from abstract
        paper_type = self._classify_paper_type(jats.get("abstract", ""))

        return Paper(
            title=jats.get("title", ""),
            authors=jats.get("authors", []),
            abstract=jats.get("abstract", ""),
            doi=jats.get("doi"),
            pmid=jats.get("pmid"),
            pmcid=jats.get("pmcid"),
            source=source_type,
            source_path=source_path,
            paper_type=paper_type,
            sections=sections,
            references=references,
            figure_ids=figure_ids,
            figure_captions=figure_captions,
        )

    def _build_paper_from_pubmed_meta(self, meta: dict, source_path: str) -> Paper:
        """Build a minimal Paper from PubMed metadata (no full text)."""
        paper_type = self._classify_paper_type(meta.get("abstract", ""))
        return Paper(
            title=meta.get("title", ""),
            authors=meta.get("authors", []),
            abstract=meta.get("abstract", ""),
            doi=meta.get("doi"),
            pmid=meta.get("pmid"),
            pmcid=meta.get("pmcid"),
            source=PaperSource.PMID,
            source_path=source_path,
            paper_type=paper_type,
        )

    def _build_paper_stub(
        self,
        title: str,
        source: PaperSource,
        source_path: str,
        raw_text: str,
    ) -> Paper:
        """Return a minimal stub Paper when all parsing paths fail."""
        return Paper(
            title=title,
            source=source,
            source_path=source_path,
            raw_text=raw_text,
        )

    def _attach_bioc_context(self, paper: Paper) -> Paper:
        """Attach normalized BioC context when available; keep parsing resilient."""
        try:
            collection = fetch_bioc_json_for_paper(
                pmid=paper.pmid,
                pmcid=paper.pmcid,
                encoding="unicode",
            )
            if not collection:
                return paper
            context = _build_bioc_context_from_collection(
                collection=collection,
                pmid=paper.pmid,
                pmcid=paper.pmcid,
                max_passages=200,
            )
            if context is None:
                return paper
            return paper.model_copy(update={"bioc_context": context})
        except Exception as exc:
            logger.debug("BioC context attachment failed for %s: %s", paper.pmid or paper.pmcid or paper.source_path, exc)
            return paper

    # ── LLM extraction helpers ──────────────────────────────────────────────

    def _extract_header_meta(self, text: str) -> _HeaderMeta:
        """Extract title, authors, abstract, and DOI from paper header text."""
        prompt = (
            "Extract the metadata from the beginning of this scientific paper.\n\n"
            f"TEXT:\n{text}\n\n"
            "Extract: title, authors (full name list), abstract (complete text), "
            "DOI (if present), journal name, and publication year."
        )
        try:
            return ask_claude_structured(
                prompt=prompt,
                output_schema=_HeaderMeta,
                system=SYSTEM_PAPER_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
        except Exception as exc:
            logger.warning("Header metadata extraction failed: %s", exc)
            return _HeaderMeta()

    def _extract_sections(self, raw_text: str) -> list[Section]:
        """Extract sections using regex first, then LLM for difficult text."""
        # Try regex-based splitting first (fast, no API cost)
        section_dict = split_text_into_sections(raw_text)
        if len(section_dict) >= 3:
            # Regex found meaningful structure
            return [
                Section(
                    title=title.title(),
                    text=text,
                    figures_referenced=_normalize_figure_ids(
                        extract_figure_ids_from_text(text)
                    ),
                )
                for title, text in section_dict.items()
                if text.strip()
            ]

        # Regex found nothing useful — ask LLM to segment
        # Use a sliding window of 8000 chars if text is very long
        excerpt = raw_text[:8000] if len(raw_text) > 8000 else raw_text
        prompt = (
            "Segment this scientific paper text into its major sections.\n\n"
            f"TEXT:\n{excerpt}\n\n"
            "For each section, identify: the section title, and the full text body. "
            "Also list any URLs found that point to supplementary materials."
        )
        try:
            result = ask_claude_structured(
                prompt=prompt,
                output_schema=_ExtractedSections,
                system=SYSTEM_PAPER_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return [
                Section(
                    title=s.title,
                    text=s.text,
                    figures_referenced=_normalize_figure_ids(
                        s.figures_referenced or extract_figure_ids_from_text(s.text)
                    ),
                )
                for s in result.sections
            ]
        except Exception as exc:
            logger.warning("Section extraction failed: %s", exc)
            return [Section(title="Full Text", text=raw_text)]

    _REFERENCE_HEADER_RE = re.compile(
        r"\n\s*(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*\n",
        re.IGNORECASE,
    )

    def _extract_reference_section_text(self, raw_text: str) -> str:
        """Locate the references section and return its text.

        Looks for a 'References' / 'Bibliography' header to find the start
        of the section, then returns everything from that point to the end.
        Falls back to the last 3000 characters if no boundary is found.
        """
        match = self._REFERENCE_HEADER_RE.search(raw_text)
        if match:
            return raw_text[match.start():]
        return raw_text[-3000:]

    def _extract_references_llm(self, text: str) -> list[Reference]:
        """Extract bibliography references from the end of the paper."""
        # Quick check: does this text look like a reference list?
        ref_indicators = re.findall(r"\[\d+\]|\d+\.\s+[A-Z]", text)
        if not ref_indicators:
            return []

        prompt = (
            "Extract the bibliography references from this text. "
            "Each reference should have: a ref_id (e.g. '[1]' or '1.'), "
            "title, authors (list), journal, year, and DOI if present.\n\n"
            f"TEXT:\n{text}"
        )
        try:
            result = ask_claude_structured(
                prompt=prompt,
                output_schema=_ExtractedReferences,
                system=SYSTEM_PAPER_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return [
                Reference(
                    ref_id=r.ref_id,
                    title=r.title,
                    authors=r.authors,
                    journal=r.journal,
                    year=r.year,
                    doi=r.doi,
                )
                for r in result.references
            ]
        except Exception as exc:
            logger.warning("Reference extraction failed: %s", exc)
            return []

    def _classify_paper_type(self, abstract_or_text: str) -> PaperType:
        """Classify paper type from abstract text."""
        if not abstract_or_text.strip():
            return PaperType.EXPERIMENTAL

        prompt = (
            "Classify this scientific paper's type based on its abstract/text.\n\n"
            f"TEXT:\n{abstract_or_text[:1500]}\n\n"
            "Choose ONE of: experimental, computational, multi_omic, review, "
            "clinical, reanalysis. Provide brief reasoning."
        )
        try:
            result = ask_claude_structured(
                prompt=prompt,
                output_schema=_PaperTypeClassification,
                system=SYSTEM_PAPER_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return PaperType(result.paper_type)
        except Exception as exc:
            logger.warning("Paper type classification failed: %s. Defaulting to EXPERIMENTAL.", exc)
            return PaperType.EXPERIMENTAL

    def _extract_supplementary_items(self, text: str) -> list[SupplementaryItem]:
        """Extract supplementary file references from text."""
        if not text.strip():
            return []

        prompt = (
            "Extract references to supplementary files, tables, or data from this text. "
            "For each: item_id (e.g. 'Table S1'), label/description, URL (if present), "
            "file type (xlsx, pdf, bed, etc.), and data content type "
            "(count_matrix, peak_list, deg_table, etc.).\n\n"
            f"TEXT:\n{text[:2000]}"
        )
        try:
            result = ask_claude_structured(
                prompt=prompt,
                output_schema=_SupplementaryItems,
                system=SYSTEM_PAPER_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return [
                SupplementaryItem(
                    item_id=item.item_id,
                    label=item.label,
                    url=item.url,
                    file_type=item.file_type,
                    data_content=item.data_content,
                )
                for item in result.items
            ]
        except Exception as exc:
            logger.warning("Supplementary item extraction failed: %s", exc)
            return []

    # ── Regex-based supplementary detection ──────────────────────────────────

    _SUPP_REF_RE = re.compile(
        r"\b(?:Supplementary|Supp\.?|Supporting)\s+"
        r"(Table|Figure|Fig\.?|Data|File|Movie|Video|Note)\s+"
        r"S?(\d+[A-Za-z]?)",
        re.IGNORECASE,
    )

    _TABLE_S_RE = re.compile(
        r"\bTable\s+S(\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )

    _DATA_S_RE = re.compile(
        r"\bData\s+S(\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )

    def _detect_supplementary_refs_regex(self, text: str) -> list[SupplementaryItem]:
        """Detect supplementary items via regex scan over full text.

        Catches inline references like 'Table S1', 'Supplementary Figure 2',
        'Data S3' etc. that may appear outside dedicated supp sections.
        """
        found: dict[str, SupplementaryItem] = {}

        for match in self._SUPP_REF_RE.finditer(text):
            item_type = match.group(1).rstrip(".")
            num = match.group(2)
            item_id = f"Supplementary {item_type.title()} {num}"
            if item_id not in found:
                found[item_id] = SupplementaryItem(item_id=item_id, label=item_id)

        for match in self._TABLE_S_RE.finditer(text):
            item_id = f"Table S{match.group(1)}"
            if item_id not in found:
                found[item_id] = SupplementaryItem(item_id=item_id, label=item_id)

        for match in self._DATA_S_RE.finditer(text):
            item_id = f"Data S{match.group(1)}"
            if item_id not in found:
                found[item_id] = SupplementaryItem(item_id=item_id, label=item_id)

        return list(found.values())

    # ── Utility methods ─────────────────────────────────────────────────────

    def _extract_figure_ids(self, sections: list[Section]) -> list[str]:
        """Collect all unique figure IDs referenced across sections."""
        all_ids: list[str] = []
        for section in sections:
            all_ids.extend(section.figures_referenced)
        return _deduplicate_ordered(all_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bioc_context_from_collection(
    collection: dict,
    pmid: Optional[str],
    pmcid: Optional[str],
    max_passages: int = 200,
) -> Optional[BioCContext]:
    """Convert a raw BioC collection into grouped BioCContext fields."""
    fig_selector = make_bioc_section_selector(("FIG",))
    fig_passages = extract_bioc_passages(collection, section_selector=fig_selector)
    results_passages = extract_bioc_passages(collection, section_selector=bioc_results_section_selector)
    methods_passages = extract_bioc_passages(collection, section_selector=bioc_methods_section_selector)

    raw_fig_ids: list[int] = []
    for passage in fig_passages:
        infons = passage.get("infons") if isinstance(passage, dict) else {}
        if not isinstance(infons, dict):
            continue
        raw_id = str(infons.get("id", "")).strip()
        m = re.match(r"(?i)^F(\d+)$", raw_id)
        if m:
            raw_fig_ids.append(int(m.group(1)))
    irregular_f_sequence = bool(raw_fig_ids) and sorted(set(raw_fig_ids)) != list(range(min(raw_fig_ids), max(raw_fig_ids) + 1))

    def _convert(passages: list[dict], with_fig_id: bool = False) -> list[BioCPassageContext]:
        out: list[BioCPassageContext] = []
        for idx, passage in enumerate(passages, start=1):
            infons = passage.get("infons") if isinstance(passage, dict) else {}
            if not isinstance(infons, dict):
                infons = {}
            sec_type = str(infons.get("section_type", "")).upper()
            p_type = str(infons.get("type", "")).strip() or None
            text = str(passage.get("text", "") if isinstance(passage, dict) else "")
            try:
                offset = int(passage.get("offset", 0) if isinstance(passage, dict) else 0)
            except (TypeError, ValueError):
                offset = 0
            figure_id = None
            if with_fig_id:
                figure_id = map_bioc_figure_id(
                    infon_id=str(infons.get("id", "")),
                    text=text,
                    fallback_index=idx,
                    irregular_f_sequence=irregular_f_sequence,
                )
            out.append(
                BioCPassageContext(
                    section_type=sec_type,
                    type=p_type,
                    text=text,
                    offset=offset,
                    figure_id=figure_id,
                    file=str(infons.get("file", "")).strip() or None,
                )
            )
        return out

    fig_ctx = _convert(fig_passages, with_fig_id=True)
    results_ctx = _convert(results_passages, with_fig_id=False)
    methods_ctx = _convert(methods_passages, with_fig_id=False)

    total = len(fig_ctx) + len(results_ctx) + len(methods_ctx)
    if total > max_passages:
        # Priority: FIG > METHODS > RESULTS.
        keep_fig = min(len(fig_ctx), max_passages)
        remaining = max_passages - keep_fig
        keep_methods = min(len(methods_ctx), remaining)
        remaining -= keep_methods
        keep_results = min(len(results_ctx), remaining)
        fig_ctx = fig_ctx[:keep_fig]
        methods_ctx = methods_ctx[:keep_methods]
        results_ctx = results_ctx[:keep_results]

    if not fig_ctx and not results_ctx and not methods_ctx:
        return None

    source_date = str(collection.get("date", "")).strip() or None
    return BioCContext(
        pmid=pmid,
        pmcid=pmcid,
        source_date=source_date,
        fig=fig_ctx,
        results=results_ctx,
        methods=methods_ctx,
    )


def _deduplicate_ordered(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_figure_ids(figure_ids: list[str]) -> list[str]:
    """Normalize figure IDs to main figure granularity.

    Examples:
    - Figure 1A -> Figure 1
    - Figure 2b-c -> Figure 2
    - Supplementary Figure S3d -> Supplementary Figure S3
    """
    out: list[str] = []
    seen: set[str] = set()
    for fid in figure_ids:
        norm = _normalize_single_figure_id(fid)
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def _main_figure_ids_only(figure_ids: list[str]) -> list[str]:
    """Keep only non-supplementary figure IDs, preserving order."""
    return [fid for fid in figure_ids if not _is_supplementary_figure_id(fid)]


def _is_supplementary_figure_id(fid: str) -> bool:
    return bool(re.match(r"(?i)^supplementary\s+figure\b", (fid or "").strip()))


def _normalize_single_figure_id(fid: str) -> str:
    """Normalize one raw figure reference to a canonical main-figure label."""
    text = (fid or "").strip()
    m = re.match(
        r"(?i)^(supplementary\s+)?figure\s+([sS]?\d+)",
        text,
    )
    if not m:
        return text
    is_supp = bool(m.group(1))
    num = m.group(2).upper()
    if is_supp:
        return f"Supplementary Figure {num}"
    return f"Figure {num}"


def _strip_html(html: str) -> str:
    """Very simple HTML tag stripper. For production use, prefer BeautifulSoup."""
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html)
    return html.strip()
