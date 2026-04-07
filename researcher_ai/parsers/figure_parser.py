"""Figure parser: extract structured Figure objects from a parsed Paper.

Strategy (per figure):

- PDF source: multimodal-first flow that crops panel images from the PDF and
  sends raw image bytes + caption/context to a vision-capable model, then
  instantiates ``Figure``/``SubFigure`` directly from structured output.
- Non-PDF source: caption/context-assisted structured extraction path with
  text/BioC helpers retained for backwards compatibility.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from researcher_ai.models.figure import (
    Axis,
    AxisScale,
    ConfidenceScores,
    ColorMapping,
    ColormapType,
    ErrorBarType,
    Figure,
    PanelLayout,
    PanelBoundingBox,
    PlotCategory,
    PlotLayer,
    PlotType,
    StatisticalAnnotation,
    SubFigure,
)
from researcher_ai.models.paper import Paper, PaperSource
from researcher_ai.models.paper import BioCPassageContext
from researcher_ai.parsers.figure_calibration import FigureCalibrationEngine
from researcher_ai.utils import llm as llm_utils
from researcher_ai.utils.llm import (
    LLMCache,
    extract_structured_data,
    SYSTEM_FIGURE_PARSER,
)
from researcher_ai.utils.pdf import extract_figure_ids_from_text, extract_figure_panel_images_from_pdf
from researcher_ai.utils.pubmed import get_figure_urls_from_pmid, get_figure_urls_from_pmcid

logger = logging.getLogger(__name__)

# Deprecated compatibility alias for legacy tests/mocks.
ask_claude_structured = extract_structured_data


def _extract_structured_data(*args, **kwargs):
    return ask_claude_structured(*args, **kwargs)


# ---------------------------------------------------------------------------
# Structured extraction schemas (Pydantic, used with extract_structured_data)
# ---------------------------------------------------------------------------

class _AxisMeta(BaseModel):
    label: str = Field(default="", description="Axis label text as shown in the panel.")
    scale: str = Field(
        default="linear",
        description="Axis scale type, e.g. linear, log, symlog, or other.",
    )
    units: Optional[str] = Field(
        default=None,
        description="Measurement units shown on the axis, if present.",
    )
    data_type: Optional[str] = Field(
        default=None,
        description="Data type represented on the axis, e.g. count, intensity, percentage.",
    )
    is_inverted: bool = Field(
        default=False,
        description="Whether the axis appears visually inverted.",
    )


class _ConfidenceScoresMeta(BaseModel):
    """LLM-estimated confidence scores for panel fields (0-100)."""

    label: Optional[float] = None
    description: Optional[float] = None
    plot_type: Optional[float] = None
    plot_category: Optional[float] = None
    x_axis: Optional[float] = None
    y_axis: Optional[float] = None
    color_variable: Optional[float] = None
    error_bars: Optional[float] = None
    sample_size: Optional[float] = None
    data_source: Optional[float] = None
    assays: Optional[float] = None
    statistical_test: Optional[float] = None
    facet_variable: Optional[float] = None


class _SubFigureMeta(BaseModel):
    """LLM-extracted metadata for a single panel/subfigure."""
    label: str = Field(description="Panel label, e.g. 'a', 'b', 'A', '1'")
    description: str = Field(description="What this panel shows (1–2 sentences)")
    plot_type: str = Field(
        description=(
            "One of: scatter, line, bubble, step, histogram, kde, ecdf, rug, "
            "density_2d, hexbin, bar, grouped_bar, stacked_bar, box, violin, "
            "strip, swarm, boxen, point, count, heatmap, clustermap, dotplot, "
            "contour, filled_contour, regression, residual, volcano, ma_plot, "
            "manhattan, genome_browser, circos, ideogram, coverage_track, "
            "umap, tsne, pca, network_graph, sankey, venn, upset, dendrogram, "
            "treemap, sunburst, spatial_scatter, tissue_image, image, "
            "flow_cytometry, pairplot, jointplot, pie, area, waterfall, "
            "surface_3d, scatter_3d, other"
        )
    )
    plot_category: str = Field(
        description=(
            "One of: relational, distribution, categorical, matrix, regression, "
            "genomic, dimensionality, network, spatial, image, flow, hierarchical, composite"
        )
    )
    x_axis: Optional[_AxisMeta] = None
    y_axis: Optional[_AxisMeta] = None
    color_variable: Optional[str] = Field(
        default=None,
        description="Data variable mapped to color (e.g., 'cluster', 'log2FC')"
    )
    error_bars: str = Field(
        default="none",
        description="One of: sd, sem, ci_95, ci_99, iqr, min_max, none"
    )
    sample_size: Optional[str] = None
    shows_individual_points: bool = False
    data_source: Optional[str] = None
    assays: list[str] = Field(default_factory=list)
    supplementary_tables: list[str] = Field(default_factory=list)
    statistical_test: Optional[str] = None
    facet_variable: Optional[str] = None
    confidence_scores: Optional[_ConfidenceScoresMeta] = Field(
        default=None,
        description=(
            "Per-field confidence scores from 0-100 for label, description, "
            "plot_type, plot_category, x_axis, y_axis, color_variable, "
            "error_bars, sample_size, data_source, assays, statistical_test, "
            "and facet_variable."
        ),
    )
    composite_confidence: float = Field(
        default=50.0,
        description="Overall confidence from 0-100 for this panel interpretation.",
    )


class _SubFigureList(BaseModel):
    subfigures: list[_SubFigureMeta] = Field(default_factory=list)

    @field_validator("subfigures", mode="before")
    @classmethod
    def _coerce_stringified_subfigures(cls, value):
        """Accept LLM tool outputs where subfigures is a JSON string."""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, dict) and "subfigures" in parsed:
                return parsed["subfigures"]
            return parsed
        return value


class _FigurePurpose(BaseModel):
    purpose: str = Field(description="One paragraph describing what this figure shows and why")
    title: str = Field(description="A short title for the figure (max 12 words)")


class _MethodsAndDatasets(BaseModel):
    methods: list[str] = Field(
        default_factory=list,
        description="Method or assay names referenced (e.g., 'RNA-seq', 'CLIP-seq peak calling')"
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Dataset identifiers mentioned (e.g., 'GSE72987', 'SRP123456')"
    )


class _VisionFigureExtraction(BaseModel):
    """Multimodal extraction payload used for PDF figure parsing."""

    title: str = Field(
        default="",
        description="Short figure title inferred from the caption and visual panels.",
    )
    purpose: str = Field(
        default="",
        description="High-level summary of the scientific purpose and key takeaway of the figure.",
    )
    subfigures: list[_SubFigureMeta] = Field(
        default_factory=list,
        description="All visually distinct panels in this figure, each with panel-level metadata.",
    )
    methods_used: list[str] = Field(
        default_factory=list,
        description="Methods or assays explicitly evidenced by the figure/caption context.",
    )
    datasets_used: list[str] = Field(
        default_factory=list,
        description="Dataset identifiers referenced in the figure/caption context.",
    )


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches all common public dataset accession formats
_ACCESSION_RE = re.compile(
    r"\b("
    r"GSE\d{4,8}"            # GEO Series
    r"|GSM\d{4,8}"           # GEO Sample
    r"|GDS\d{3,7}"           # GEO DataSet
    r"|SRP\d{4,9}"           # SRA Project
    r"|SRX\d{4,9}"           # SRA Experiment
    r"|SRR\d{4,9}"           # SRA Run
    r"|ERP\d{4,9}"           # ENA Project
    r"|ERR\d{4,9}"           # ENA Run
    r"|PRJNA\d{4,9}"         # BioProject (NCBI)
    r"|PRJEB\d{4,9}"         # BioProject (EBI)
    r"|PXD\d{4,9}"           # PRIDE (proteomics)
    r"|MSV\d{4,9}"           # MassIVE (proteomics)
    r"|E-MTAB-\d{4,9}"       # ArrayExpress
    r"|EGAS\d{4,9}"          # EGA (sensitive data)
    r")",
    re.IGNORECASE,
)

# Sentence splitter — split on '. ' followed by capital letter
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Figure reference patterns for caption-finding
_CAPTION_START_RE = re.compile(
    r"(?:^|\n)\s*(Fig(?:ure)?\.?\s*\d+[A-Za-z\-–]*)[\s.:–-]",
    re.IGNORECASE,
)

# Panel label pattern: a, b, c or A, B, C or (a) (b) etc.
_PANEL_LABEL_RE = re.compile(r"\b([A-H])\b|\(([a-h])\)")

_CATEGORY_CUES: dict[PlotCategory, tuple[str, ...]] = {
    PlotCategory.GENOMIC: (
        "volcano", "manhattan", "ma plot", "genome browser", "coverage track", "circos", "ideogram",
    ),
    PlotCategory.FLOW: ("sankey", "alluvial", "flow diagram", "river plot", "venn", "upset"),
    PlotCategory.MATRIX: ("heatmap", "clustermap", "clustered heatmap", "dotplot", "contour"),
    PlotCategory.DIMENSIONALITY: ("umap", "t-sne", "tsne", "pca"),
    PlotCategory.NETWORK: ("network", "graph", "force-directed", "node", "edge"),
    PlotCategory.DISTRIBUTION: ("histogram", "kde", "density", "ecdf", "rug"),
    PlotCategory.CATEGORICAL: ("violin", "boxplot", "box", "bar plot", "grouped bar", "stacked bar", "swarm", "strip"),
    PlotCategory.IMAGE: ("microscopy", "western blot", "gel image", "fluorescence image", "image"),
    PlotCategory.SPATIAL: ("spatial", "tissue image"),
}

_PLOT_CUES: dict[PlotType, tuple[str, ...]] = {
    PlotType.STACKED_BAR: ("stacked bar", "cumulative stacked", "stacked"),
    PlotType.GROUPED_BAR: ("grouped bar", "side-by-side bar", "clustered bar"),
    PlotType.BAR: ("horizontal bar", "vertical bar", "bar plot", "bar chart"),
    PlotType.BUBBLE: ("bubble plot", "bubble chart", "sized points", "point size"),
    PlotType.SCATTER: ("scatter plot", "scatter", "point cloud"),
    PlotType.CIRCOS: ("circos", "chord diagram", "circular genome", "circular plot"),
    PlotType.SANKEY: ("sankey", "alluvial", "river plot", "flow diagram"),
    PlotType.UPSET: ("upset plot", "upset"),
    PlotType.CLUSTERMAP: ("clustermap", "clustered heatmap", "hierarchical clustered heatmap"),
    PlotType.HEATMAP: ("heatmap",),
    PlotType.VIOLIN: ("violin", "violin plot"),
    PlotType.SWARM: ("swarm", "beeswarm"),
    PlotType.STRIP: ("strip plot", "stripplot", "jittered points"),
    PlotType.VOLCANO: ("volcano plot", "log2fc", "-log10", "log fold change"),
    PlotType.MANHATTAN: ("manhattan plot", "genome-wide significance"),
    PlotType.GENOME_BROWSER: ("genome browser", "igv", "track view"),
    PlotType.COVERAGE_TRACK: ("coverage track", "read coverage", "bigwig"),
    PlotType.UMAP: ("umap",),
    PlotType.TSNE: ("t-sne", "tsne"),
    PlotType.PCA: ("pca", "principal component"),
    PlotType.NETWORK_GRAPH: ("network graph", "network", "node", "edge"),
    PlotType.VENN: ("venn", "euler"),
    PlotType.PAIRPLOT: ("pairplot", "pair plot", "scatter matrix"),
    PlotType.JOINTPLOT: ("jointplot", "joint plot"),
}

_PLOT_TYPE_TO_CATEGORY: dict[PlotType, PlotCategory] = {
    PlotType.STACKED_BAR: PlotCategory.CATEGORICAL,
    PlotType.GROUPED_BAR: PlotCategory.CATEGORICAL,
    PlotType.BAR: PlotCategory.CATEGORICAL,
    PlotType.BUBBLE: PlotCategory.RELATIONAL,
    PlotType.SCATTER: PlotCategory.RELATIONAL,
    PlotType.CIRCOS: PlotCategory.GENOMIC,
    PlotType.SANKEY: PlotCategory.FLOW,
    PlotType.UPSET: PlotCategory.FLOW,
    PlotType.CLUSTERMAP: PlotCategory.MATRIX,
    PlotType.HEATMAP: PlotCategory.MATRIX,
    PlotType.VIOLIN: PlotCategory.CATEGORICAL,
    PlotType.SWARM: PlotCategory.CATEGORICAL,
    PlotType.STRIP: PlotCategory.CATEGORICAL,
    PlotType.VOLCANO: PlotCategory.GENOMIC,
    PlotType.MANHATTAN: PlotCategory.GENOMIC,
    PlotType.GENOME_BROWSER: PlotCategory.GENOMIC,
    PlotType.COVERAGE_TRACK: PlotCategory.GENOMIC,
    PlotType.UMAP: PlotCategory.DIMENSIONALITY,
    PlotType.TSNE: PlotCategory.DIMENSIONALITY,
    PlotType.PCA: PlotCategory.DIMENSIONALITY,
    PlotType.NETWORK_GRAPH: PlotCategory.NETWORK,
    PlotType.VENN: PlotCategory.FLOW,
    PlotType.PAIRPLOT: PlotCategory.RELATIONAL,
    PlotType.JOINTPLOT: PlotCategory.RELATIONAL,
}


# ---------------------------------------------------------------------------
# FigureParser
# ---------------------------------------------------------------------------

class FigureParser:
    """Extract and contextualize figures from a parsed Paper.

    For each figure_id in paper.figure_ids, produces a fully structured
    Figure object with subfigures, axes, data sources, and purpose.

    Args:
        llm_model: Claude model identifier.
        cache_dir: Optional directory for caching LLM responses.
    """

    def __init__(
        self,
        llm_model: str = llm_utils.DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        vision_model: str = "gemini-3.1-pro",
    ):
        """Initialize FigureParser with model and optional structured-response cache."""
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.cache = LLMCache(cache_dir) if cache_dir else None
        self.calibration_engine = FigureCalibrationEngine()

    # ── Public API ───────────────────────────────────────────────────────────

    def parse_all_figures(self, paper: Paper) -> list[Figure]:
        """Extract all figures from a parsed paper.

        Args:
            paper: A Paper object (from PaperParser) with populated figure_ids.

        Returns:
            List of Figure objects ordered by figure_id.
        """
        figures: list[Figure] = []

        # Primary source is Paper.figure_ids from PaperParser. If empty, recover
        # figure IDs from section/raw text so FigureParser can still run when
        # upstream parsing falls back to metadata-only paths.
        figure_ids = list(paper.figure_ids)
        if not figure_ids:
            recovered = self._recover_figure_ids(paper)
            if recovered:
                logger.warning(
                    "Paper.figure_ids is empty; recovered %d figure IDs from text",
                    len(recovered),
                )
                figure_ids = recovered
        figure_ids = [
            fid for fid in figure_ids
            if not re.match(r"(?i)^supplementary\s+figure\b", (fid or "").strip())
        ]
        preview_map = self._resolve_preview_urls(paper, figure_ids)

        if paper.source == PaperSource.PDF:
            paper_level_datasets = []
            paper_level_methods = []
        else:
            paper_level_datasets = self._extract_dataset_ids_from_paper(paper)
            paper_level_methods = self._extract_methods_from_paper(paper)

        for fig_id in figure_ids:
            logger.info("Parsing %s", fig_id)
            parse_warnings: list[str] = []
            if paper.source == PaperSource.PDF:
                # PDF path is multimodal-first: avoid legacy text-regex routing.
                caption, in_text = self._pdf_figure_context(paper, fig_id)
            else:
                caption = self._find_caption(paper, fig_id)
                in_text = self._find_in_text_references(paper, fig_id)
            multimodal_figure = self._parse_figure_with_multimodal_pdf(
                paper=paper,
                figure_id=fig_id,
                caption=caption,
                in_text=in_text,
                parse_warnings=parse_warnings,
            )
            if multimodal_figure is not None:
                multimodal_figure = self._apply_paper_specific_overrides(paper, multimodal_figure)
                multimodal_figure = multimodal_figure.model_copy(update={
                    "datasets_used": _merge_ordered_unique(
                        multimodal_figure.datasets_used, paper_level_datasets
                    ),
                    "methods_used": _merge_ordered_unique(
                        multimodal_figure.methods_used, paper_level_methods
                    ),
                    "preview_url": preview_map.get(fig_id),
                    "parse_warnings": _merge_ordered_unique(
                        multimodal_figure.parse_warnings, parse_warnings
                    ),
                })
                figures.append(multimodal_figure)
                continue
            if paper.source == PaperSource.PDF:
                stub = self._stub_figure(
                    fig_id,
                    caption=caption,
                    in_text=in_text,
                    parse_warnings=parse_warnings,
                )
                stub = stub.model_copy(update={
                    "datasets_used": paper_level_datasets,
                    "methods_used": paper_level_methods,
                    "preview_url": preview_map.get(fig_id),
                })
                figures.append(stub)
                continue
            bioc_fig_passages, bioc_results_passages = self._get_bioc_context_for_figure(
                paper,
                fig_id,
                max_results=8,
            )
            try:
                figure = self._parse_figure_from_context(
                    fig_id,
                    caption,
                    in_text,
                    bioc_fig_passages=bioc_fig_passages,
                    bioc_results_passages=bioc_results_passages,
                )
                figure = self._apply_paper_specific_overrides(paper, figure)
                figure = figure.model_copy(update={
                    "datasets_used": _merge_ordered_unique(
                        figure.datasets_used, paper_level_datasets
                    ),
                    "methods_used": _merge_ordered_unique(
                        figure.methods_used, paper_level_methods
                    ),
                    "preview_url": preview_map.get(fig_id),
                })
                figures.append(figure)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", fig_id, exc)
                stub = self._stub_figure(fig_id, caption=caption, in_text=in_text)
                stub = stub.model_copy(update={
                    "datasets_used": paper_level_datasets,
                    "methods_used": paper_level_methods,
                    "preview_url": preview_map.get(fig_id),
                })
                figures.append(stub)
        return figures

    def _recover_figure_ids(self, paper: Paper) -> list[str]:
        """Recover figure IDs from section and raw text when Paper.figure_ids is empty."""
        found: list[str] = []
        for section in paper.sections:
            found.extend(extract_figure_ids_from_text(section.text))
        if paper.raw_text:
            found.extend(extract_figure_ids_from_text(paper.raw_text))

        deduped: list[str] = []
        seen: set[str] = set()
        for fid in found:
            if re.match(r"(?i)^supplementary\s+figure\b", (fid or "").strip()):
                continue
            key = fid.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(fid)
        return deduped

    def parse_figure(self, paper: Paper, figure_id: str) -> Figure:
        """Parse a single figure with full context.

        Strategy:
        1. Find the caption text for figure_id.
        2. Find all in-text references across all sections.
        3. LLM: decompose into subfigures.
        4. LLM: determine purpose and short title.
        5. Regex + LLM: identify datasets and methods.
        6. Infer panel layout from subfigure labels.

        Args:
            paper: Parsed Paper object.
            figure_id: Figure identifier (e.g., "Figure 1").

        Returns:
            Populated Figure object.
        """
        if paper.source == PaperSource.PDF:
            caption, in_text = self._pdf_figure_context(paper, figure_id)
            parse_warnings: list[str] = []
        else:
            caption = self._find_caption(paper, figure_id)
            in_text = self._find_in_text_references(paper, figure_id)
            parse_warnings = []
        multimodal_figure = self._parse_figure_with_multimodal_pdf(
            paper=paper,
            figure_id=figure_id,
            caption=caption,
            in_text=in_text,
            parse_warnings=parse_warnings,
        )
        if multimodal_figure is not None:
            figure = self._apply_paper_specific_overrides(paper, multimodal_figure)
            preview_map = self._resolve_preview_urls(paper, [figure_id])
            if paper.source == PaperSource.PDF:
                paper_level_datasets = []
                paper_level_methods = []
            else:
                paper_level_datasets = self._extract_dataset_ids_from_paper(paper)
                paper_level_methods = self._extract_methods_from_paper(paper)
            return figure.model_copy(update={
                "datasets_used": _merge_ordered_unique(
                    figure.datasets_used, paper_level_datasets
                ),
                "methods_used": _merge_ordered_unique(
                    figure.methods_used, paper_level_methods
                ),
                "preview_url": preview_map.get(figure_id),
                "parse_warnings": _merge_ordered_unique(
                    figure.parse_warnings, parse_warnings
                ),
            })
        if paper.source == PaperSource.PDF:
            return self._stub_figure(
                figure_id,
                caption=caption,
                in_text=in_text,
                parse_warnings=parse_warnings,
            )

        bioc_fig_passages, bioc_results_passages = self._get_bioc_context_for_figure(
            paper,
            figure_id,
            max_results=8,
        )
        figure = self._parse_figure_from_context(
            figure_id,
            caption,
            in_text,
            bioc_fig_passages=bioc_fig_passages,
            bioc_results_passages=bioc_results_passages,
        )
        figure = self._apply_paper_specific_overrides(paper, figure)
        paper_level_datasets = self._extract_dataset_ids_from_paper(paper)
        paper_level_methods = self._extract_methods_from_paper(paper)
        preview_map = self._resolve_preview_urls(paper, [figure_id])
        figure = figure.model_copy(update={
            "datasets_used": _merge_ordered_unique(
                figure.datasets_used, paper_level_datasets
            ),
            "methods_used": _merge_ordered_unique(
                figure.methods_used, paper_level_methods
            ),
            "preview_url": preview_map.get(figure_id),
        })
        return figure

    def _pdf_figure_context(self, paper: Paper, figure_id: str) -> tuple[str, list[str]]:
        """Return lightweight caption/context for PDF multimodal parsing only."""
        if paper.figure_captions:
            for key, caption in paper.figure_captions.items():
                if _fig_ids_match(key, figure_id):
                    return caption, []
        return "", []

    def _parse_figure_with_multimodal_pdf(
        self,
        *,
        paper: Paper,
        figure_id: str,
        caption: str,
        in_text: list[str],
        parse_warnings: Optional[list[str]] = None,
    ) -> Optional[Figure]:
        """Parse figure directly from PDF panel crops and caption context."""
        parse_warnings = parse_warnings if parse_warnings is not None else []
        if paper.source != PaperSource.PDF:
            return None
        pdf_path = Path(paper.source_path)
        if not pdf_path.exists() or not pdf_path.is_file():
            warning = "multimodal_pdf_unavailable:source_pdf_missing"
            parse_warnings.append(warning)
            logger.warning("Multimodal PDF figure extraction unavailable for %s: source PDF missing", figure_id)
            return None

        panel_images = extract_figure_panel_images_from_pdf(
            pdf_path,
            figure_id=figure_id,
            caption=caption,
        )
        if not panel_images:
            warning = "multimodal_pdf_unavailable:no_panel_images_extracted"
            parse_warnings.append(warning)
            logger.warning("Multimodal PDF figure extraction unavailable for %s: no panel images extracted", figure_id)
            return None

        prompt = (
            f"Analyse scientific figure {figure_id} from the provided panel images.\n\n"
            f"CAPTION:\n{caption or '(not found)'}\n\n"
            f"IN-TEXT REFERENCES:\n{chr(10).join(in_text[:10]) or '(none)'}\n\n"
            "Instantiate complete figure metadata with robust panel-level "
            "SubFigure objects."
        )
        try:
            extracted = _extract_structured_data(
                model_router=self.vision_model,
                prompt=prompt,
                schema=_VisionFigureExtraction,
                system=SYSTEM_FIGURE_PARSER,
                cache=self.cache,
                image_bytes=panel_images,
            )
        except Exception as exc:
            logger.warning("Multimodal PDF figure extraction failed for %s: %s", figure_id, exc)
            parse_warnings.append(
                f"multimodal_pdf_unavailable:vision_extraction_failed:{exc.__class__.__name__}"
            )
            return None

        subfigures = [_subfigure_from_meta(m) for m in extracted.subfigures]
        if not subfigures and caption.strip():
            subfigures = _fallback_subfigures_from_caption(caption)
        layout = self._infer_layout(subfigures)
        subfigures = _assign_subfigure_boundary_boxes(subfigures, layout)
        title = _resolve_figure_title(figure_id, extracted.title, caption)
        purpose = extracted.purpose or _fallback_purpose_from_caption(caption, in_text, figure_id)
        return Figure(
            figure_id=figure_id,
            title=title,
            caption=caption,
            purpose=purpose,
            subfigures=subfigures,
            layout=layout,
            in_text_context=in_text,
            datasets_used=[d for d in extracted.datasets_used if isinstance(d, str) and d.strip()],
            methods_used=[m for m in extracted.methods_used if isinstance(m, str) and m.strip()],
            parse_warnings=parse_warnings,
        )

    def _resolve_preview_urls(self, paper: Paper, figure_ids: list[str]) -> dict[str, str]:
        """Map canonical main figure IDs (Figure 1, Figure 2, ...) to preview URLs."""
        if not figure_ids:
            return {}
        try:
            urls: list[str] = []
            if paper.pmid:
                urls = get_figure_urls_from_pmid(paper.pmid)
            elif paper.pmcid:
                urls = get_figure_urls_from_pmcid(paper.pmcid)
            if not urls:
                return {}

            ordered_main: list[str] = []
            seen: set[str] = set()
            for fid in figure_ids:
                norm = _canonical_main_figure_id(fid)
                if not norm:
                    continue
                if norm in seen:
                    continue
                seen.add(norm)
                ordered_main.append(norm)

            return {fid: url for fid, url in zip(ordered_main, urls)}
        except Exception as exc:
            logger.debug("Preview URL resolution failed: %s", exc)
            return {}

    def _parse_figure_from_context(
        self,
        figure_id: str,
        caption: str,
        in_text: list[str],
        *,
        bioc_fig_passages: Optional[list[BioCPassageContext]] = None,
        bioc_results_passages: Optional[list[BioCPassageContext]] = None,
    ) -> Figure:
        """Run LLM extraction given pre-collected caption and in-text context.

        Separating this from parse_figure() allows parse_all_figures() to supply
        already-computed context without a redundant second scan of all sections.
        """
        bioc_fig_passages = bioc_fig_passages or []
        bioc_results_passages = bioc_results_passages or []

        bioc_caption_lines = [p.text.strip() for p in bioc_fig_passages if p.text and p.text.strip()]
        bioc_results_lines = [p.text.strip() for p in bioc_results_passages if p.text and p.text.strip()]
        enriched_caption = caption
        if bioc_caption_lines:
            enriched_caption = (
                caption
                + "\n\nBioC FIG context:\n"
                + "\n".join(bioc_caption_lines)
            ).strip()
        enriched_in_text = list(in_text)
        enriched_in_text.extend(line for line in bioc_results_lines if line not in enriched_in_text)

        subfigures = self._decompose_subfigures(figure_id, enriched_caption, enriched_in_text)
        if not subfigures and caption.strip():
            subfigures = _fallback_subfigures_from_caption(caption)
        layout = self._infer_layout(subfigures)
        subfigures = _assign_subfigure_boundary_boxes(subfigures, layout)
        subfigures = [
            _disambiguate_subfigure_plot(
                sf,
                caption=_panel_caption_context(enriched_caption, sf.label, subfigures),
                in_text=_panel_in_text_context(enriched_in_text, sf.label),
                bioc_evidence=_panel_bioc_evidence(
                    bioc_fig_passages + bioc_results_passages,
                    figure_id=figure_id,
                    panel_label=sf.label,
                ),
            )
            for sf in subfigures
        ]
        purpose_meta = self._determine_purpose(figure_id, enriched_caption, enriched_in_text)
        datasets = self._identify_datasets(enriched_caption, enriched_in_text)
        methods = self._identify_methods(
            enriched_caption,
            enriched_in_text,
            bioc_passages=bioc_fig_passages + bioc_results_passages,
        )
        resolved_title = _resolve_figure_title(figure_id, purpose_meta.title, caption)

        return Figure(
            figure_id=figure_id,
            title=resolved_title,
            caption=caption,
            purpose=purpose_meta.purpose,
            subfigures=subfigures,
            layout=layout,
            in_text_context=in_text,
            datasets_used=datasets,
            methods_used=methods,
        )

    def _get_bioc_context_for_figure(
        self,
        paper: Paper,
        figure_id: str,
        *,
        max_results: int = 8,
    ) -> tuple[list[BioCPassageContext], list[BioCPassageContext]]:
        """Return BioC FIG passages and top-ranked RESULTS mentions for a figure."""
        if not getattr(paper, "bioc_context", None):
            return [], []
        bioc = paper.bioc_context
        fig_passages = [
            p for p in (bioc.fig or [])
            if p.figure_id and _fig_ids_match(p.figure_id, figure_id)
        ]
        pattern = _build_fig_ref_pattern(figure_id)
        anchor = min((p.offset for p in fig_passages), default=None)

        scored: list[tuple[int, int, BioCPassageContext]] = []
        for p in (bioc.results or []):
            text = (p.text or "").strip()
            if not text:
                continue
            mentions = sum(1 for _ in pattern.finditer(text))
            if mentions <= 0:
                continue
            distance = abs(p.offset - anchor) if anchor is not None else p.offset
            scored.append((mentions, distance, p))

        scored.sort(key=lambda x: (-x[0], x[1]))
        selected_results = [item[2] for item in scored[:max_results]]
        return fig_passages, selected_results

    def _apply_paper_specific_overrides(self, paper: Paper, figure: Figure) -> Figure:
        """Apply registry-driven calibration rules."""
        try:
            engine = getattr(self, "calibration_engine", None)
            if engine is None:
                engine = FigureCalibrationEngine()
                self.calibration_engine = engine
            return engine.apply(paper, figure)
        except Exception as exc:
            logger.warning("Figure calibration failed for %s: %s", figure.figure_id, exc)
            return figure

    # ── Caption and context extraction ───────────────────────────────────────

    def _find_caption(self, paper: Paper, figure_id: str) -> str:
        """Locate caption text for a figure.

        Search order:
        1. paper.figure_captions dict (populated from JATS XML) — exact match.
        2. paper.figure_captions dict — parent figure fallback for sub-panel IDs
           (e.g. "Figure 1a" → "Figure 1" caption so subfigure decomposition
           receives the multi-panel caption and can identify panel 'a').
        3. Scan section text for a paragraph starting with the figure label.
        4. Empty string if not found.
        """
        # 1. JATS captions dict — exact match
        if paper.figure_captions:
            for key, caption in paper.figure_captions.items():
                if _fig_ids_match(key, figure_id):
                    return caption

            # 2. Parent figure fallback: strip trailing panel label(s) and retry.
            #    "Figure 1a" → try "Figure 1"; "Figure 3a–b" → try "Figure 3".
            parent_id = re.sub(
                r"(?i)(fig(?:ure)?\.?\s*\d+)[A-Za-z\-–,\s]+$",
                r"\1",
                figure_id,
            ).strip()
            if parent_id and parent_id.lower() != figure_id.lower():
                for key, caption in paper.figure_captions.items():
                    if _fig_ids_match(key, parent_id):
                        logger.debug(
                            "Caption not found for %s; using parent caption (%s)",
                            figure_id,
                            parent_id,
                        )
                        return caption

        # 3. Pattern scan across section text
        for section in paper.sections:
            caption = _extract_caption_from_text(section.text, figure_id)
            if caption:
                return caption

        # 4. Check raw_text directly
        if paper.raw_text:
            caption = _extract_caption_from_text(paper.raw_text, figure_id)
            if caption:
                return caption

        logger.debug("Caption not found for %s", figure_id)
        return ""

    def _find_in_text_references(self, paper: Paper, figure_id: str) -> list[str]:
        """Collect all sentences across sections that reference this figure.

        Matches the full figure ID and sub-panel variants (e.g., 'Figure 1a',
        'Fig. 1B-D', 'Figure 1').
        """
        pattern = _build_fig_ref_pattern(figure_id)
        sentences: list[str] = []

        for section in paper.sections:
            for sentence in _SENTENCE_RE.split(section.text):
                sentence = sentence.strip()
                if sentence and pattern.search(sentence):
                    sentences.append(sentence)

        return sentences

    # ── LLM extraction helpers ───────────────────────────────────────────────

    def _decompose_subfigures(
        self, figure_id: str, caption: str, in_text: list[str]
    ) -> list[SubFigure]:
        """Use LLM to decompose a figure caption into panel-level SubFigure objects."""
        if not caption.strip():
            return []

        context_snippet = "\n".join(in_text[:5])  # top 5 in-text sentences
        prompt = (
            f"Analyse this figure caption and in-text references for {figure_id}.\n\n"
            f"CAPTION:\n{caption}\n\n"
            f"IN-TEXT REFERENCES:\n{context_snippet}\n\n"
            "Decompose into individual panels/subfigures. For each panel identify: "
            "label (a, b, c…), description (what it shows), plot_type, plot_category, "
            "x_axis, y_axis, color_variable, error_bars, sample_size, "
            "shows_individual_points, assays, supplementary_tables, statistical_test, "
            "data_source, facet_variable, confidence_scores (0-100 for each field), "
            "and composite_confidence (0-100 overall). "
            "Score 100 only when unambiguous from evidence; score 0 when unknown. "
            "If the figure appears to be a single panel with no sub-labels, return one "
            "entry with label='main'."
        )
        try:
            result = _extract_structured_data(
                prompt=prompt,
                output_schema=_SubFigureList,
                system=SYSTEM_FIGURE_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            return [_subfigure_from_meta(m) for m in result.subfigures]
        except Exception as exc:
            logger.warning("Subfigure decomposition failed for %s: %s", figure_id, exc)
            return []

    def _determine_purpose(
        self, figure_id: str, caption: str, in_text: list[str]
    ) -> _FigurePurpose:
        """LLM: write a one-paragraph purpose and a short title for the figure."""
        context_snippet = "\n".join(in_text[:8])
        prompt = (
            f"Given the caption and selected in-text references for {figure_id}, "
            "write: (1) a 'purpose' paragraph (2-4 sentences) answering "
            "'What is this figure trying to show and why does it matter?', "
            "and (2) a 'title' of at most 12 words summarising the figure.\n\n"
            f"CAPTION:\n{caption or '(not found)'}\n\n"
            f"IN-TEXT CONTEXT:\n{context_snippet or '(none)'}"
        )
        try:
            return _extract_structured_data(
                prompt=prompt,
                output_schema=_FigurePurpose,
                system=SYSTEM_FIGURE_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
        except Exception as exc:
            logger.warning("Purpose extraction failed for %s: %s", figure_id, exc)
            return _FigurePurpose(
                purpose=_fallback_purpose_from_caption(caption, in_text, figure_id),
                title=figure_id,
            )

    def _identify_datasets(
        self,
        caption: str,
        in_text: list[str],
        *,
        strict_regex_only: bool = True,
    ) -> list[str]:
        """Extract dataset accession IDs via regex, with optional LLM supplement.

        Args:
            caption: Figure caption text.
            in_text: In-text sentences referencing the figure.
            strict_regex_only: When True (default), the LLM is called only as a
                fallback when regex finds no accessions. When False, the LLM is
                always called and its results are merged with the regex hits —
                useful for papers that embed non-standard identifiers in prose.
        """
        combined = caption + " " + " ".join(in_text)
        grounded = {m.group(1).upper() for m in _ACCESSION_RE.finditer(combined)}

        # Regex pass — reliable for well-formatted accessions
        accessions = sorted(grounded)

        if accessions and strict_regex_only:
            return accessions

        # LLM pass — either as fallback (no regex hits) or to supplement regex hits
        if not combined.strip():
            return accessions  # already empty
        try:
            result = _extract_structured_data(
                prompt=(
                    "Extract any dataset or repository accession identifiers from this text. "
                    "Include GEO (GSE/GSM), SRA (SRP/SRR/SRX), PRIDE (PXD), "
                    "BioProject (PRJNA), ArrayExpress (E-MTAB), and similar IDs.\n\n"
                    f"TEXT:\n{combined[:2000]}"
                ),
                output_schema=_MethodsAndDatasets,
                system=SYSTEM_FIGURE_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            llm_candidates = [
                d.upper().strip()
                for d in result.datasets
                if isinstance(d, str) and d.strip()
            ]
            # Accept only IDs grounded in source text to prevent hallucinations.
            llm_grounded = [d for d in llm_candidates if d in grounded]
            return sorted(set(accessions) | set(llm_grounded))
        except Exception as exc:
            logger.warning("Dataset extraction (LLM) failed: %s", exc)
            return accessions

    def _identify_methods(
        self,
        caption: str,
        in_text: list[str],
        *,
        bioc_passages: Optional[list[BioCPassageContext]] = None,
    ) -> list[str]:
        """Extract assay/method names from figure context using LLM."""
        # Use full in-text list but cap total input by characters, not sentence count,
        # so long papers don't silently drop method mentions from later sentences.
        bioc_lines = [
            p.text.strip() for p in (bioc_passages or [])
            if p.text and p.text.strip()
        ]
        combined = (caption + " " + " ".join(in_text) + " " + " ".join(bioc_lines)).strip()
        if not combined:
            return []
        try:
            result = _extract_structured_data(
                prompt=(
                    "List the distinct assay or experimental method names referenced "
                    "in this figure caption and context text. Include sequencing assays "
                    "(RNA-seq, ChIP-seq, ATAC-seq, eCLIP…), computational methods "
                    "(peak calling, differential expression, GSEA…), and imaging assays.\n\n"
                    f"TEXT:\n{combined[:2000]}"
                ),
                output_schema=_MethodsAndDatasets,
                system=SYSTEM_FIGURE_PARSER,
                model=self.llm_model,
                cache=self.cache,
            )
            methods = [m for m in result.methods if m and m.strip()]
            if methods:
                return methods
            return self._identify_methods_regex(combined)
        except Exception as exc:
            logger.warning("Method extraction failed: %s", exc)
            return self._identify_methods_regex(combined)

    def _extract_dataset_ids_from_paper(self, paper: Paper) -> list[str]:
        """Extract dataset IDs from full-paper context for figure fallback."""
        text_parts: list[str] = []
        text_parts.extend(paper.figure_captions.values())
        text_parts.extend(sec.text for sec in paper.sections if sec.text)
        if paper.raw_text:
            text_parts.append(paper.raw_text)
        combined = "\n".join(text_parts)
        if not combined.strip():
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for match in _ACCESSION_RE.finditer(combined):
            accession = match.group(1).upper()
            if accession not in seen:
                seen.add(accession)
                ordered.append(accession)
        return ordered

    def _extract_methods_from_paper(self, paper: Paper) -> list[str]:
        """Extract methods from methods-like sections for figure fallback."""
        methods_texts: list[str] = []
        for section in paper.sections:
            title = (section.title or "").lower()
            if any(k in title for k in ("method", "protocol", "experimental", "materials")):
                methods_texts.append(f"{section.title}\n{section.text}")
        if not methods_texts and paper.raw_text:
            methods_texts.append(paper.raw_text)
        combined = "\n\n".join(t for t in methods_texts if t.strip())
        if not combined:
            return []
        return self._identify_methods_regex(combined)

    def _identify_methods_regex(self, text: str) -> list[str]:
        """Heuristic method extraction when LLM output is empty/unavailable."""
        if not text.strip():
            return []
        found: list[str] = []
        seen: set[str] = set()

        # Preserve explicit protocol-style headings in Methods text.
        for line in text.splitlines():
            candidate = line.strip(" :\t")
            if not candidate:
                continue
            if len(candidate) > 90:
                continue
            if any(
                key in candidate.lower()
                for key in ("library preparation", "crosslink", "immunoprecip", "peak calling", "sequencing")
            ):
                key = candidate.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(candidate)

        # Canonical assay/tool names.
        method_patterns = [
            r"\beCLIP(?:-seq)?\b",
            r"\biCLIP\b",
            r"\bCLIP\b",
            r"\bPAR-CLIP\b",
            r"\bHITS-CLIP\b",
            r"\bRNA-seq\b",
            r"\bChIP-seq\b",
            r"\bATAC-seq\b",
            r"\bCLIPper\b",
            r"\bSMInput\b",
            r"\bDESeq2\b",
            r"\bCutadapt\b",
            r"\bSTAR\b",
            r"\bUV crosslink(?:ing)?\b",
            r"\bRNase I digestion\b",
        ]
        for pattern in method_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                method = match.group(0)
                key = method.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(method)

        return found

    # ── Layout inference ─────────────────────────────────────────────────────

    def _infer_layout(self, subfigures: list[SubFigure]) -> PanelLayout:
        """Infer a PanelLayout from the number and labels of subfigures.

        Uses a simple heuristic: try to detect row×col layout from labels.
        Falls back to a single-row layout.
        """
        n = len(subfigures)
        if n <= 1:
            return PanelLayout(n_rows=1, n_cols=1)

        # Check if labels are uppercase (A, B, C…) or lowercase (a, b, c…)
        labels = [sf.label.strip("()") for sf in subfigures]
        uppercase = all(l.isupper() for l in labels if l.isalpha())
        style = "uppercase" if uppercase else "lowercase"

        # Simple heuristic: use sqrt-like layout
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return PanelLayout(
            n_rows=rows,
            n_cols=cols,
            panel_labels_style=style,
        )

    # ── Stub helpers ─────────────────────────────────────────────────────────

    def _stub_figure(
        self,
        figure_id: str,
        caption: str = "",
        in_text: list[str] | None = None,
        parse_warnings: list[str] | None = None,
    ) -> Figure:
        """Return a minimal Figure when parsing fails.

        Preserves any caption and in-text context that were already collected
        before the LLM calls failed, to aid debugging.
        """
        return Figure(
            figure_id=figure_id,
            title=figure_id,
            caption=caption,
            purpose="Could not be parsed.",
            in_text_context=in_text or [],
            parse_warnings=parse_warnings or [],
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fig_ids_match(key: str, figure_id: str) -> bool:
    """Check if a caption dict key matches a figure_id (case-insensitive, flexible).

    Normalises both sides by stripping non-alphanumeric chars and comparing.
    E.g., "Fig. 1" == "Figure 1" == "figure1".
    """
    def _norm(s: str) -> str:
        s = re.sub(r"fig(?:ure)?\.?\s*", "figure", s.lower())
        return re.sub(r"[^a-z0-9]", "", s)
    return _norm(key) == _norm(figure_id)


def _canonical_main_figure_id(figure_id: str) -> str:
    """Normalize figure references to canonical main figure IDs (e.g., Figure 3)."""
    text = (figure_id or "").strip()
    if re.match(r"(?i)^supplementary\s+figure\b", text):
        return ""
    m = re.match(r"(?i)^fig(?:ure)?\.?\s*(\d+)", text)
    if not m:
        m = re.search(r"(?i)\bfigure\s+(\d+)\b", text)
    if not m:
        return ""
    return f"Figure {int(m.group(1))}"


def _build_fig_ref_pattern(figure_id: str) -> re.Pattern:
    """Build a regex pattern that matches figure_id and panel variants.

    "Figure 1"              → "Figure 1", "Fig. 1", "Fig. 1A", "Fig. 1a-c"
    "Supplementary Figure S1" → "Supplementary Fig. S1", "Supplementary Figure S1"
    """
    is_supp = bool(re.search(r"Supplementary", figure_id, re.IGNORECASE))

    # Extract the figure label: optional S-prefix + digits + optional letter suffix
    # Handles: "1", "1A", "S1", "S1A"
    num_match = re.search(r"([A-Za-z]?\d+[A-Za-z]?)\s*$", figure_id.split()[-1])
    if not num_match:
        num_match = re.search(r"([A-Za-z]?\d+[A-Za-z]?)", figure_id)
    if not num_match:
        return re.compile(re.escape(figure_id), re.IGNORECASE)

    num = num_match.group(1)
    if is_supp:
        # For supplementary figures, allow the optional "Supplementary" prefix
        # so "Fig. S1" and "Supplementary Fig. S1" both match.
        supp_prefix = r"(?:Supplementary\s+)?"
        return re.compile(
            rf"\b{supp_prefix}Fig(?:ure)?\.?\s*{re.escape(num)}(?:[A-Za-z\-–,\s]*)?",
            re.IGNORECASE,
        )
    else:
        # For standard figures, use a negative lookbehind to prevent matching
        # supplementary references like "Supplementary Figure 1" or "Supp. Fig. 1".
        # Both lookbehinds are fixed-width so Python's re engine accepts them.
        return re.compile(
            rf"(?<!supplementary )(?<!supp\. )"
            rf"\bFig(?:ure)?\.?\s*{re.escape(num)}(?:[A-Za-z\-–,\s]*)?",
            re.IGNORECASE,
        )


def _extract_caption_from_text(text: str, figure_id: str) -> str:
    """Scan text for a paragraph that starts with the figure label.

    Returns the paragraph text (up to ~1000 chars) if found.

    Handles both plain figure numbers ("Figure 1") and S-prefixed supplementary
    labels ("Supplementary Figure S1", "Supplementary Figure S10").
    """
    # Capture optional S-prefix + digits so "S1" is preserved for supplementary
    # figures instead of degenerating to just "1".
    num_match = re.search(r"([Ss]?\d+)", figure_id)
    if not num_match:
        return ""
    num = num_match.group(1)

    supp_prefix = r"(?:Supplementary\s+)?" if "Supplementary" in figure_id else ""
    pattern = re.compile(
        rf"(?:^|\n)({supp_prefix}Fig(?:ure)?\.?\s*{re.escape(num)}\b[^\n]{{0,200}})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        # Grab from match start up to ~1000 chars or next figure label
        start = match.start()
        snippet = text[start: start + 1000]
        # Trim at next figure heading (newline + "Figure" or "Fig."), including
        # S-prefixed supplementary figure numbers.
        next_fig = re.search(
            r"\n\s*(?:Supplementary\s+)?Fig(?:ure)?\.?\s*[Ss]?\d+", snippet[20:]
        )
        if next_fig:
            snippet = snippet[: 20 + next_fig.start()]
        return snippet.strip()
    return ""


def _fallback_subfigures_from_caption(caption: str) -> list[SubFigure]:
    """Heuristic subfigure extraction when LLM parsing is unavailable."""
    clean = re.sub(r"\s+", " ", (caption or "")).strip()
    if not clean:
        return []

    # Detect panel labels like "(a)", "(b)", "A", "B" in caption text.
    labels_raw: list[str] = []
    for m in re.finditer(r"\(([a-hA-H])\)", clean):
        labels_raw.append(m.group(1).lower())
    if not labels_raw:
        for m in re.finditer(r"\b([A-H])\b", clean):
            labels_raw.append(m.group(1).lower())

    labels: list[str] = []
    seen: set[str] = set()
    for lb in labels_raw:
        if lb not in seen:
            seen.add(lb)
            labels.append(lb)

    if not labels:
        return [
            SubFigure(
                label="main",
                description=_truncate(clean, 180),
                plot_category=PlotCategory.COMPOSITE,
                plot_type=PlotType.OTHER,
            )
        ]

    out: list[SubFigure] = []
    for lb in labels:
        out.append(
            SubFigure(
                label=lb,
                description=f"Panel {lb}: {_truncate(clean, 140)}",
                plot_category=PlotCategory.COMPOSITE,
                plot_type=PlotType.OTHER,
            )
        )
    return out


def _fallback_purpose_from_caption(caption: str, in_text: list[str], figure_id: str) -> str:
    """Heuristic purpose text when LLM purpose extraction is unavailable."""
    cap = _truncate(re.sub(r"\s+", " ", (caption or "")).strip(), 260)
    ctx = _truncate(re.sub(r"\s+", " ", " ".join(in_text[:2])).strip(), 220)
    if cap and ctx:
        return f"{figure_id} summarizes the reported result: {cap} Context: {ctx}"
    if cap:
        return f"{figure_id} summarizes the reported result: {cap}"
    if ctx:
        return f"{figure_id} is referenced in context: {ctx}"
    return f"{figure_id} could not be fully parsed from available text."


def _truncate(text: str, n: int) -> str:
    """Trim text to length ``n`` with ellipsis when truncation is required."""
    if len(text) <= n:
        return text
    return text[: max(0, n - 3)].rstrip() + "..."


def _merge_ordered_unique(primary: list[str], fallback: list[str]) -> list[str]:
    """Merge two ordered lists without duplicates (case-insensitive key)."""
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(primary) + list(fallback):
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _normalize_panel_label(label: str) -> str:
    """Normalize a panel label to a lowercase token (e.g., '(A)' -> 'a')."""
    raw = (label or "").strip().strip("()").lower()
    return raw[:1] if raw else ""


def _panel_label_pattern(label: str) -> re.Pattern:
    """Build a regex matching common panel label mentions for one label."""
    lb = re.escape(_normalize_panel_label(label))
    if not lb:
        return re.compile(r"$^")
    return re.compile(
        rf"(?i)(?:\({lb}\)|\bpanel\s*{lb}\b|\b{lb}[):.]|\bfig(?:ure)?\.?\s*\d+\s*{lb}\b)"
    )


def _panel_caption_context(caption: str, label: str, all_subfigures: list[SubFigure]) -> str:
    """Return panel-local caption context for one subfigure.

    Uses label-aware sentence filtering so cues from one subplot do not bleed
    into others during disambiguation.
    """
    clean = (caption or "").strip()
    if not clean:
        return clean
    norm_label = _normalize_panel_label(label)
    if not norm_label:
        return clean

    # Single-panel figures should keep full context.
    labels = {_normalize_panel_label(sf.label) for sf in all_subfigures}
    labels.discard("")
    if len(labels) <= 1:
        return clean

    panel_pat = _panel_label_pattern(norm_label)

    # Preferred: explicit span extraction from "a, ... b, ... c, ..." or "(a) ... (b) ...".
    # This avoids cue bleed between neighboring panels in dense captions.
    span = _extract_panel_label_span(clean, norm_label)
    if span:
        return span

    segments = re.split(r"(?<=[.;])\s+", clean)
    selected = [seg for seg in segments if panel_pat.search(seg)]
    if selected:
        return " ".join(selected)
    # Fallback: if the caption uses no explicit panel labels, keep full caption.
    return clean if not re.search(r"\([a-z]\)|\bpanel\s+[a-z]\b", clean, flags=re.IGNORECASE) else ""


def _panel_in_text_context(in_text: list[str], label: str) -> list[str]:
    """Return in-text sentences relevant to one panel label."""
    norm_label = _normalize_panel_label(label)
    if not norm_label:
        return in_text
    pat = _panel_label_pattern(norm_label)
    matched = [s for s in (in_text or []) if pat.search(s or "")]
    return matched or in_text


def _panel_bioc_evidence(
    passages: list[BioCPassageContext],
    *,
    figure_id: str,
    panel_label: str,
) -> list[BioCPassageContext]:
    """Select BioC passages relevant to a panel, falling back to parent figure."""
    if not passages:
        return []
    fig_pattern = _build_fig_ref_pattern(figure_id)
    panel = (panel_label or "").strip().strip("()")
    panel_pat = None
    if panel and panel.lower() != "main":
        panel_pat = _panel_label_pattern(panel)

    selected: list[BioCPassageContext] = []
    for p in passages:
        text = (p.text or "").strip()
        if not text:
            continue
        if panel_pat is not None and panel_pat.search(text):
            selected.append(p)

    if selected:
        return selected

    parent_selected: list[BioCPassageContext] = []
    for p in passages:
        text = (p.text or "").strip()
        if text and (fig_pattern.search(text) or p.section_type.upper() == "FIG"):
            parent_selected.append(p)
    return parent_selected


def _bioc_reference(passage: BioCPassageContext) -> BioCPassageContext:
    """Store BioC evidence as lightweight references keyed by section_type+offset."""
    return BioCPassageContext(
        section_type=passage.section_type,
        type=passage.type,
        text="",
        offset=passage.offset,
        figure_id=passage.figure_id,
        file=passage.file,
    )


def _extract_panel_label_span(caption: str, label: str) -> str:
    """Extract caption span for one panel label from multi-panel captions.

    Supports common formats such as "(a) ... (b) ...", "a, ... b, ...", and
    "panel a ... panel b ...".
    """
    text = (caption or "").strip()
    norm = _normalize_panel_label(label)
    if not text or not norm:
        return ""

    current_pat = re.compile(
        rf"(?i)(?:\({norm}\)|\bpanel\s*{norm}\b|\b{norm}[,.:)])"
    )
    all_panel_pat = re.compile(
        r"(?i)(?:\([a-h]\)|\bpanel\s*[a-h]\b|\b[a-h][,.:)])"
    )
    start_match = current_pat.search(text)
    if not start_match:
        return ""
    start = start_match.start()
    after_current = start_match.end()
    tail = text[after_current:]
    next_match = all_panel_pat.search(tail)
    if next_match:
        end = after_current + next_match.start()
        return text[start:end].strip()
    return text[start:].strip()


def _extract_axis_label_from_text(text: str, axis: str) -> Optional[str]:
    """Extract an axis label from free text using simple lexical patterns."""
    axis_key = axis.lower().strip()
    if axis_key not in {"x", "y"}:
        return None
    source = text or ""
    quoted_pattern = re.compile(
        rf"(?i)\b{axis_key}-?axis\b(?:\s+for\s+[^.;:\n]*?)?\s+"
        r"(?:represents|is|=|denotes)\s*['\"“]([^\"”']+)['\"”]"
    )
    match = quoted_pattern.search(source)
    if match:
        label = match.group(1).strip()
        return label or None

    unquoted_pattern = re.compile(
        rf"(?i)\b{axis_key}-?axis\b(?:\s+for\s+[^.;:\n]*?)?\s+"
        r"(?:represents|is|=|denotes)\s*([^.;\n]+?)(?=\s+\band\s+[xy]-?axis\b|[.;\n]|$)"
    )
    match = unquoted_pattern.search(source)
    if not match:
        return None
    label = match.group(1).strip().strip("'\"“”")
    return label or None


def _infer_axis_scale_from_text(axis_label: str, context_text: str) -> tuple[AxisScale, bool]:
    """Infer axis scale/is_inverted using axis label and nearby context cues."""
    label = (axis_label or "").strip()
    text = f"{label} {context_text or ''}".lower()

    if "symlog" in text:
        return AxisScale.SYMLOG, False
    if "-log10" in text or "−log10" in text:
        return AxisScale.LOG10, True
    if "log10" in text:
        return AxisScale.LOG10, False
    if "log2" in text:
        return AxisScale.LOG2, False
    if re.search(r"\bln\b|natural log", text):
        return AxisScale.LN, False
    if "reversed" in text or "inverted" in text:
        return AxisScale.REVERSED, True

    label_low = label.lower()
    categorical_cues = (
        "cluster", "cell type", "condition", "group", "sample", "dataset",
        "protein", "gene", "zfp", "motif", "category", "class", "assay",
    )
    numeric_cues = ("%", "fold", "ratio", "count", "expression", "score", "p-value")
    has_digits = bool(re.search(r"\d", label_low))
    looks_numeric = has_digits or any(cue in label_low for cue in numeric_cues)
    if any(cue in label_low for cue in categorical_cues) and not looks_numeric:
        return AxisScale.CATEGORICAL, False
    if re.fullmatch(r"[a-zA-Z][a-zA-Z\s\-/()]+", label) and not looks_numeric:
        return AxisScale.CATEGORICAL, False

    return AxisScale.LINEAR, False


def _resolve_figure_title(figure_id: str, purpose_title: str, caption: str) -> str:
    """Choose a robust figure title from purpose title and caption text."""
    title = (purpose_title or "").strip()
    fig_norm = re.sub(r"\s+", " ", (figure_id or "").strip()).lower()
    if title:
        title_norm = re.sub(r"\s+", " ", title).lower()
        generic = (
            title_norm == fig_norm
            or bool(re.fullmatch(r"(?:supplementary\s+)?fig(?:ure)?\.?\s*[Ss]?\d+[a-z]?", title_norm))
            or title_norm in {"figure", "fig", "figure title"}
        )
        if not generic:
            return title

    clean = re.sub(r"\s+", " ", (caption or "")).strip()
    if clean:
        clean = re.sub(
            r"^(?:supplementary\s+)?fig(?:ure)?\.?\s*[Ss]?\d+[A-Za-z\-–]*\s*[:.]\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        if clean:
            sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].strip()
            if sentence:
                return _truncate(sentence, 120)
            return _truncate(clean, 120)
    return figure_id


def _subfigure_from_meta(meta: _SubFigureMeta) -> SubFigure:
    """Convert LLM-extracted _SubFigureMeta into a SubFigure model instance."""
    # Safely resolve PlotType (fall back to OTHER on unknown)
    try:
        plot_type = PlotType(meta.plot_type.lower())
    except ValueError:
        logger.debug("Unknown plot_type '%s', using OTHER", meta.plot_type)
        plot_type = PlotType.OTHER

    # Safely resolve PlotCategory
    try:
        plot_category = PlotCategory(meta.plot_category.lower())
    except ValueError:
        logger.debug("Unknown plot_category '%s', using COMPOSITE", meta.plot_category)
        plot_category = PlotCategory.COMPOSITE

    # Safely resolve ErrorBarType
    try:
        error_bars = ErrorBarType(meta.error_bars.lower())
    except ValueError:
        error_bars = ErrorBarType.NONE

    # Build Axis objects
    x_axis = _axis_from_meta(meta.x_axis) if meta.x_axis else None
    y_axis = _axis_from_meta(meta.y_axis) if meta.y_axis else None

    # Build ColorMapping if a color variable was identified
    color_mapping = None
    if meta.color_variable:
        color_mapping = ColorMapping(
            variable=meta.color_variable,
            colormap_type=ColormapType.QUALITATIVE,
            is_colorblind_safe=True,
        )

    # Build StatisticalAnnotation if a test was named
    stat_ann = None
    if meta.statistical_test:
        stat_ann = StatisticalAnnotation(test_name=meta.statistical_test)

    # Primary layer matching the plot_type
    layers = [PlotLayer(plot_type=plot_type, is_primary=True)]
    confidence_scores = _confidence_scores_from_meta(meta.confidence_scores)
    composite_conf = _clamp_100(meta.composite_confidence)
    if composite_conf <= 0:
        composite_conf = _composite_from_scores(confidence_scores)

    return SubFigure(
        label=meta.label,
        description=meta.description,
        plot_category=plot_category,
        plot_type=plot_type,
        layers=layers,
        x_axis=x_axis,
        y_axis=y_axis,
        color_mapping=color_mapping,
        error_bars=error_bars,
        sample_size=meta.sample_size,
        shows_individual_points=meta.shows_individual_points,
        statistical_annotations=stat_ann,
        data_source=meta.data_source,
        assays=meta.assays,
        supplementary_tables=meta.supplementary_tables,
        facet_variable=meta.facet_variable,
        confidence_scores=confidence_scores,
        composite_confidence=composite_conf,
        classification_confidence=round(composite_conf / 100.0, 3),
    )


def _disambiguate_subfigure_plot(
    subfigure: SubFigure,
    *,
    caption: str,
    in_text: list[str],
    bioc_evidence: Optional[list[BioCPassageContext]] = None,
) -> SubFigure:
    """Two-stage plot disambiguation using deterministic textual evidence.

    Stage A:
      Infer a broad PlotCategory from caption/context/description cues.
    Stage B:
      Infer a concrete PlotType constrained by Stage A, then attach confidence,
      alternatives, and cue evidence for downstream quality checks.
    """
    bioc_evidence = bioc_evidence or []
    bioc_text = " ".join(p.text for p in bioc_evidence if p.text)
    text_raw = " ".join([subfigure.description or "", caption or "", " ".join(in_text or []), bioc_text])
    text = text_raw.lower()
    bioc_text_lower = bioc_text.lower()

    stage_a_cat, stage_a_evidence = _infer_plot_category_from_text(text)
    candidates = _infer_plot_type_candidates(text)
    if stage_a_cat is not None:
        filtered = [
            (plot_type, evidence)
            for plot_type, evidence in candidates
            if _PLOT_TYPE_TO_CATEGORY.get(plot_type) == stage_a_cat
        ]
        if filtered:
            candidates = filtered

    chosen_type = subfigure.plot_type
    chosen_category = subfigure.plot_category
    confidence = 0.5
    evidence_spans: list[str] = list(stage_a_evidence)
    alternatives: list[PlotType] = []
    facet_variable: Optional[str] = subfigure.facet_variable
    n_facets: Optional[int] = subfigure.n_facets
    x_axis = subfigure.x_axis
    y_axis = subfigure.y_axis
    field_scores = subfigure.confidence_scores.model_copy()

    # High-priority explicit rules before general cue ranking.
    if "upset" in text:
        chosen_type = PlotType.UPSET
        chosen_category = PlotCategory.FLOW
        confidence = 0.95
        evidence_spans.append("upset")
        candidates = []
    elif "venn" in text:
        chosen_type = PlotType.VENN
        chosen_category = PlotCategory.FLOW
        confidence = 0.95
        evidence_spans.append("venn")
        candidates = []
    elif "bubble" in text:
        chosen_type = PlotType.BUBBLE
        chosen_category = PlotCategory.RELATIONAL
        confidence = 0.9
        evidence_spans.append("bubble")
        candidates = []
    elif "t-sne" in text or "tsne" in text:
        chosen_type = PlotType.TSNE
        chosen_category = PlotCategory.DIMENSIONALITY
        confidence = 0.9
        evidence_spans.append("tsne")
        candidates = []
    elif "stacked bar" in text or ("stacked" in text and "bar" in text):
        chosen_type = PlotType.STACKED_BAR
        chosen_category = PlotCategory.CATEGORICAL
        confidence = 0.9
        evidence_spans.append("stacked bar")
        candidates = []
    elif "bar plot" in text or "bar chart" in text or "horizontal bar" in text:
        chosen_type = PlotType.BAR
        chosen_category = PlotCategory.CATEGORICAL
        confidence = 0.85
        evidence_spans.append("bar plot")
        candidates = []

    if candidates:
        chosen_type = candidates[0][0]
        chosen_category = _PLOT_TYPE_TO_CATEGORY.get(chosen_type, chosen_category)
        evidence_spans.extend(candidates[0][1])
        alternatives = [pt for pt, _ in candidates[1:4] if pt != chosen_type]
        confidence = 0.9 if len(candidates[0][1]) >= 2 else 0.75
    elif stage_a_cat is not None and stage_a_cat != subfigure.plot_category:
        chosen_category = stage_a_cat
        confidence = 0.65

    bioc_candidates = _infer_plot_type_candidates(bioc_text_lower) if bioc_text_lower else []
    distinct_plot_cues: list[str] = []
    if bioc_candidates:
        cue_set: set[str] = set()
        for _, hits in bioc_candidates:
            for hit in hits:
                cue_set.add(hit)
        distinct_plot_cues = sorted(cue_set)

    # Composite overlay cues: treat violin+swarm/strip as layered composite.
    overlay_types: list[PlotType] = []
    if "violin" in text and "swarm" in text:
        chosen_type = PlotType.VIOLIN
        chosen_category = PlotCategory.COMPOSITE
        overlay_types.append(PlotType.SWARM)
        confidence = max(confidence, 0.9)
    elif "violin" in text and ("strip" in text or "jitter" in text):
        chosen_type = PlotType.VIOLIN
        chosen_category = PlotCategory.COMPOSITE
        overlay_types.append(PlotType.STRIP)
        confidence = max(confidence, 0.9)

    # Two-pane (left/right) composite panels: keep distinct pane plot types.
    # This is common in figure panels where left and right panes use different
    # chart types but share y-axis concepts.
    if (
        ("bar" in text)
        and ("stacked" in text or "cumulative" in text)
        and ("left" in text or "right" in text or "shared y-axis" in text)
        and ("bar plot" in text and "stacked bar" in text)
    ):
        chosen_type = PlotType.BAR
        chosen_category = PlotCategory.COMPOSITE
        overlay_types = [PlotType.STACKED_BAR]
        facet_variable = "left_right_panel"
        n_facets = 2
        confidence = max(confidence, 0.92)
        evidence_spans.append("left/right two-pane")
        evidence_spans.append("bar + stacked bar")

    # Axis labels (when explicitly stated in caption/in-text).
    x_axis_label = _extract_axis_label_from_text(text_raw, "x")
    y_axis_label = _extract_axis_label_from_text(text_raw, "y")
    if x_axis_label and x_axis is None:
        x_scale, x_inv = _infer_axis_scale_from_text(x_axis_label, text_raw)
        x_axis = Axis(label=x_axis_label, scale=x_scale, is_inverted=x_inv)
    elif x_axis is not None:
        inferred_scale, inferred_inverted = _infer_axis_scale_from_text(x_axis.label, text_raw)
        updates: dict[str, object] = {}
        if x_axis.scale == AxisScale.LINEAR and inferred_scale != AxisScale.LINEAR:
            updates["scale"] = inferred_scale
        if not x_axis.is_inverted and inferred_inverted:
            updates["is_inverted"] = True
        if updates:
            x_axis = x_axis.model_copy(update=updates)
    if y_axis_label and y_axis is None:
        y_scale, y_inv = _infer_axis_scale_from_text(y_axis_label, text_raw)
        y_axis = Axis(label=y_axis_label, scale=y_scale, is_inverted=y_inv)
    elif y_axis is not None:
        inferred_scale, inferred_inverted = _infer_axis_scale_from_text(y_axis.label, text_raw)
        updates: dict[str, object] = {}
        if y_axis.scale == AxisScale.LINEAR and inferred_scale != AxisScale.LINEAR:
            updates["scale"] = inferred_scale
        if not y_axis.is_inverted and inferred_inverted:
            updates["is_inverted"] = True
        if updates:
            y_axis = y_axis.model_copy(update=updates)

    # Confidence update to 0-100 scale while preserving legacy 0-1 score.
    # Use rule/disambiguation confidence as a floor for key interpretation fields.
    evidence_passage_hits = len([p for p in bioc_evidence if (p.text or "").strip()])
    base = max(_clamp_100(subfigure.composite_confidence), _composite_from_scores(field_scores))
    raw_bonus = min(15.0, (2.0 * evidence_passage_hits) + (3.0 * len(distinct_plot_cues)))
    bonus = min(raw_bonus, 100.0 - base)
    contradiction = bool(
        bioc_candidates
        and bioc_candidates[0][1]
        and bioc_candidates[0][0] != chosen_type
    )
    contradiction_factor = 0.85 if contradiction else 1.0
    adjusted_conf = _clamp_100((base + bonus) * contradiction_factor)
    floor_score = max(_clamp_100(confidence * 100.0), adjusted_conf)
    field_scores.plot_type = max(field_scores.plot_type, floor_score)
    field_scores.plot_category = max(field_scores.plot_category, floor_score)
    if x_axis is not None:
        field_scores.x_axis = max(field_scores.x_axis, floor_score - 5.0)
    if y_axis is not None:
        field_scores.y_axis = max(field_scores.y_axis, floor_score - 5.0)
    if subfigure.color_mapping is not None:
        field_scores.color_variable = max(field_scores.color_variable, floor_score - 10.0)
    if subfigure.statistical_annotations is not None:
        field_scores.statistical_test = max(field_scores.statistical_test, floor_score - 10.0)
    if subfigure.sample_size:
        field_scores.sample_size = max(field_scores.sample_size, floor_score - 15.0)
    if subfigure.data_source:
        field_scores.data_source = max(field_scores.data_source, floor_score - 15.0)
    if subfigure.assays:
        field_scores.assays = max(field_scores.assays, floor_score - 15.0)
    if subfigure.facet_variable:
        field_scores.facet_variable = max(field_scores.facet_variable, floor_score - 15.0)

    composite_conf = max(
        _clamp_100((base + bonus) * contradiction_factor),
        _clamp_100(confidence * 100.0),
    )

    layers = [PlotLayer(plot_type=chosen_type, is_primary=True)]
    for ov in overlay_types:
        layers.append(PlotLayer(plot_type=ov, is_primary=False))

    # De-duplicate evidence while preserving order.
    dedup_evidence: list[str] = []
    seen_evidence: set[str] = set()
    for ev in evidence_spans:
        key = ev.strip().lower()
        if key and key not in seen_evidence:
            seen_evidence.add(key)
            dedup_evidence.append(ev)

    return subfigure.model_copy(
        update={
            "plot_type": chosen_type,
            "plot_category": chosen_category,
            "layers": layers,
            "alternative_plot_types": alternatives,
            "evidence_spans": dedup_evidence,
            "bioc_evidence_spans": [_bioc_reference(p) for p in bioc_evidence],
            "bioc_contradiction": contradiction,
            "facet_variable": facet_variable,
            "n_facets": n_facets,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "confidence_scores": field_scores,
            "composite_confidence": composite_conf,
            "classification_confidence": round(composite_conf / 100.0, 3),
        }
    )


def _infer_plot_category_from_text(text: str) -> tuple[Optional[PlotCategory], list[str]]:
    """Infer broad plot category via cue matching."""
    best_category: Optional[PlotCategory] = None
    best_hits: list[str] = []
    for category, cues in _CATEGORY_CUES.items():
        hits = [cue for cue in cues if cue in text]
        if len(hits) > len(best_hits):
            best_category = category
            best_hits = hits
    return best_category, best_hits


def _infer_plot_type_candidates(text: str) -> list[tuple[PlotType, list[str]]]:
    """Rank plausible plot types from lexical cues in descending confidence."""
    scored: list[tuple[PlotType, list[str]]] = []
    for plot_type, cues in _PLOT_CUES.items():
        hits = [cue for cue in cues if cue in text]
        if hits:
            scored.append((plot_type, hits))
    scored.sort(key=lambda x: len(x[1]), reverse=True)
    return scored


def _clamp_100(value: Optional[float]) -> float:
    """Clamp a score to [0, 100]."""
    try:
        score = float(value if value is not None else 50.0)
    except (TypeError, ValueError):
        score = 50.0
    return max(0.0, min(100.0, score))


def _confidence_scores_from_meta(meta: Optional[_ConfidenceScoresMeta]) -> ConfidenceScores:
    """Convert optional LLM confidence payload into normalized model scores."""
    base = ConfidenceScores()
    if meta is None:
        return base
    data = meta.model_dump(exclude_none=True)
    for key, value in data.items():
        if hasattr(base, key):
            setattr(base, key, _clamp_100(value))
    return base


def _composite_from_scores(scores: ConfidenceScores) -> float:
    """Compute a weighted composite confidence (0-100)."""
    critical = [scores.plot_type, scores.plot_category, scores.description]
    structural = [scores.label, scores.x_axis, scores.y_axis]
    contextual = [
        scores.color_variable,
        scores.error_bars,
        scores.sample_size,
        scores.data_source,
        scores.assays,
        scores.statistical_test,
        scores.facet_variable,
    ]
    crit_avg = sum(critical) / len(critical)
    struct_avg = sum(structural) / len(structural)
    ctx_avg = sum(contextual) / len(contextual)
    return round((0.5 * crit_avg) + (0.3 * struct_avg) + (0.2 * ctx_avg), 2)


def _assign_subfigure_boundary_boxes(
    subfigures: list[SubFigure],
    layout: PanelLayout,
) -> list[SubFigure]:
    """Assign coarse panel boundary boxes using inferred layout grid."""
    n = len(subfigures)
    if n == 0:
        return subfigures
    rows = max(1, int(layout.n_rows or 1))
    cols = max(1, int(layout.n_cols or 1))
    out: list[SubFigure] = []
    for idx, sf in enumerate(subfigures):
        row = idx // cols
        col = idx % cols
        x0 = col / cols
        y0 = row / rows
        x1 = min(1.0, (col + 1) / cols)
        y1 = min(1.0, (row + 1) / rows)
        out.append(
            sf.model_copy(
                update={
                    "boundary_box": PanelBoundingBox(
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        coordinate_space="normalized",
                        detection_method="layout_heuristic",
                    )
                }
            )
        )
    return out


def _axis_from_meta(meta: _AxisMeta) -> Axis:
    """Convert _AxisMeta into an Axis model instance."""
    try:
        scale = AxisScale(meta.scale.lower())
    except ValueError:
        scale = AxisScale.LINEAR
    return Axis(
        label=meta.label,
        scale=scale,
        units=meta.units,
        data_type=meta.data_type,
        is_inverted=meta.is_inverted,
    )
