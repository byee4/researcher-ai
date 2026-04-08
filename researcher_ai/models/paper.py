"""Paper-level data models.

Represents a parsed scientific publication with its sections,
references, and figure inventory.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    """How the paper was loaded into the system."""

    PDF = "pdf"
    URL = "url"
    PMID = "pmid"
    PMCID = "pmcid"
    DOI = "doi"


class PaperType(str, Enum):
    """Coarse classification of paper type — drives pipeline routing.

    Added after Ouroboros evaluation revealed that review articles and
    computational-methods papers cannot be handled by the standard
    experimental pipeline.
    """

    EXPERIMENTAL = "experimental"        # Standard wet-lab + analysis paper
    COMPUTATIONAL = "computational"      # Methods/algorithm paper (e.g., new tool)
    MULTI_OMIC = "multi_omic"           # Multiple assay types, often complex DAGs
    REVIEW = "review"                   # Systematic review / meta-analysis
    CLINICAL = "clinical"               # Clinical trial / patient cohort study
    REANALYSIS = "reanalysis"           # Re-analysis of existing public datasets


class Section(BaseModel):
    """A section of a paper (e.g., Abstract, Methods, Results)."""

    title: str
    text: str
    section_type: Optional[str] = Field(
        default=None,
        description=(
            "Structured section type if available from source metadata "
            "(e.g., JATS sec-type='methods')."
        ),
    )
    is_methods: bool = Field(
        default=False,
        description=(
            "True when this section is explicitly marked as methods by the "
            "source parser (preferred over title heuristics)."
        ),
    )
    subsections: list[Section] = Field(default_factory=list)
    figures_referenced: list[str] = Field(
        default_factory=list,
        description="Figure IDs referenced in this section (e.g., ['Fig. 1a', 'Fig. 2'])",
    )


class ChunkType(str, Enum):
    """Modality-aware chunk classification for per-paper retrieval."""

    PROSE = "prose"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    SUPPLEMENTARY = "supplementary"


class AnnotatedChunk(BaseModel):
    """Chunk with provenance metadata used by the PaperRAGStore."""

    chunk_id: str
    text: str
    chunk_type: ChunkType
    source_section: str = ""
    summary: str = ""
    page_number: Optional[int] = None
    figure_id: Optional[str] = None
    panel_id: Optional[str] = None


class BioCPassageContext(BaseModel):
    """Normalized BioC passage payload used for parser enrichment."""

    section_type: str
    type: Optional[str] = None
    text: str = ""
    offset: int = 0
    figure_id: Optional[str] = None
    file: Optional[str] = None


class BioCContext(BaseModel):
    """Grouped BioC context attached to a parsed Paper."""

    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    source_date: Optional[str] = None
    fig: list[BioCPassageContext] = Field(default_factory=list)
    results: list[BioCPassageContext] = Field(default_factory=list)
    methods: list[BioCPassageContext] = Field(default_factory=list)


class Reference(BaseModel):
    """A bibliographic reference cited in the paper."""

    ref_id: str  # In-text citation key (e.g., "[1]" or "Smith et al., 2020")
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None


class SupplementaryItem(BaseModel):
    """A supplementary file or table referenced by the paper.

    Added (Phase 4.5) after Ouroboros evaluation found that many Yeo Lab
    papers store critical processed data (peak lists, DEG tables) in
    supplementary files rather than or in addition to GEO deposits.
    """

    item_id: str                        # e.g., "Table S1", "Figure S3", "Data S2"
    label: str                          # Short description from caption
    url: Optional[str] = None           # Direct download URL if extractable
    file_type: Optional[str] = None     # e.g., "xlsx", "csv", "pdf", "zip"
    description: str = ""              # Full caption / description
    data_content: Optional[str] = Field(
        default=None,
        description="Type of data contained: 'count_matrix', 'peak_list', 'deg_table', etc.",
    )


class Paper(BaseModel):
    """Top-level parsed paper object."""

    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    source: PaperSource
    source_path: str                    # Original path/URL/ID used to load
    paper_type: PaperType = PaperType.EXPERIMENTAL
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    raw_text: str = ""                  # Full raw text fallback
    figure_ids: list[str] = Field(
        default_factory=list,
        description="All figure IDs found in the paper",
    )
    figure_captions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of figure label (e.g. 'Figure 1') to full caption text. "
            "Populated from JATS XML <fig> elements when full-text is available."
        ),
    )
    supplementary_urls: list[str] = Field(default_factory=list)
    supplementary_items: list[SupplementaryItem] = Field(
        default_factory=list,
        description="Structured supplementary items (tables, figures, data files)",
    )
    bioc_context: Optional[BioCContext] = Field(
        default=None,
        description="Optional normalized BioC context grouped by FIG/RESULTS/METHODS passages.",
    )

    # Convenience accessors
    def get_section(self, title_fragment: str) -> Optional[Section]:
        """Return first section whose title contains title_fragment (case-insensitive)."""
        fragment = title_fragment.lower()
        for section in self.sections:
            if fragment in section.title.lower():
                return section
        return None

    @property
    def methods_section(self) -> Optional[Section]:
        """Convenience accessor for the first section whose title mentions methods."""
        return self.get_section("method")

    @property
    def results_section(self) -> Optional[Section]:
        """Convenience accessor for the first section whose title mentions results."""
        return self.get_section("result")
