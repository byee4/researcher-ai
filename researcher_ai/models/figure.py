"""Figure and subfigure data models.

Expanded based on evaluation against matplotlib, seaborn, plotly, and the
scientific-visualization skill from K-Dense-AI/claude-scientific-skills.

Key additions over the original model:
- PlotType enum (50+ types) replaces bare plot_type: str
- Multi-layer support (layers: list[PlotLayer]) for overlaid traces
- ColorMapping separates data dimension from visual encoding
- StatisticalAnnotation captures error bars, significance, p-values
- AxisScale enum with log2/log10/symlog/reversed variants
- PanelLayout for GridSpec-level figure composition
- RenderingSpec for journal-specific export requirements
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from researcher_ai.models.paper import BioCPassageContext


# ── Plot Type Taxonomy ────────────────────────────────────────────────────────
# Derived from: matplotlib (plot_types.md), seaborn (function_reference),
# plotly (chart-types.md), and scientific-visualization skill.
# Grouped by the plotting function family they map to.


class PlotCategory(str, Enum):
    """High-level plot category — determines which library/function family to use."""

    RELATIONAL = "relational"           # scatter, line, bubble
    DISTRIBUTION = "distribution"       # histogram, kde, ecdf, rug
    CATEGORICAL = "categorical"         # bar, box, violin, strip, swarm, point
    MATRIX = "matrix"                   # heatmap, clustermap, dotplot
    REGRESSION = "regression"           # regplot, lmplot, residplot
    GENOMIC = "genomic"                 # volcano, MA, Manhattan, circos, genome browser
    DIMENSIONALITY = "dimensionality"   # UMAP, t-SNE, PCA scatter
    NETWORK = "network"                 # graph, force-directed, arc
    SPATIAL = "spatial"                 # tissue image, spatial scatter
    IMAGE = "image"                     # microscopy, gel, Western blot scan
    FLOW = "flow"                       # sankey, alluvial, Venn/Euler
    HIERARCHICAL = "hierarchical"       # dendrogram, treemap, sunburst
    COMPOSITE = "composite"             # multi-layer overlays (box + strip, scatter + regression)


class PlotType(str, Enum):
    """Specific plot type — maps 1:1 to a library function call."""

    # Relational
    SCATTER = "scatter"                 # mpl: scatter / sns: scatterplot / plotly: px.scatter
    LINE = "line"                       # mpl: plot / sns: lineplot / plotly: px.line
    BUBBLE = "bubble"                   # scatter with size encoding
    STEP = "step"                       # mpl: step

    # Distribution
    HISTOGRAM = "histogram"             # mpl: hist / sns: histplot / plotly: px.histogram
    KDE = "kde"                         # sns: kdeplot
    ECDF = "ecdf"                       # sns: ecdfplot / plotly: px.ecdf
    RUG = "rug"                         # sns: rugplot
    DENSITY_2D = "density_2d"           # sns: kdeplot(x,y) / plotly: px.density_contour
    HEXBIN = "hexbin"                   # mpl: hexbin / plotly: px.density_heatmap

    # Categorical
    BAR = "bar"                         # mpl: bar / sns: barplot / plotly: px.bar
    GROUPED_BAR = "grouped_bar"         # barmode='group'
    STACKED_BAR = "stacked_bar"         # barmode='stack'
    BOX = "box"                         # mpl: boxplot / sns: boxplot / plotly: px.box
    VIOLIN = "violin"                   # mpl: violinplot / sns: violinplot / plotly: px.violin
    STRIP = "strip"                     # sns: stripplot / plotly: px.strip
    SWARM = "swarm"                     # sns: swarmplot (beeswarm)
    BOXEN = "boxen"                     # sns: boxenplot (letter-value)
    POINT = "point"                     # sns: pointplot (mean + CI)
    COUNT = "count"                     # sns: countplot

    # Matrix / Heatmap
    HEATMAP = "heatmap"                 # mpl: imshow / sns: heatmap / plotly: px.imshow
    CLUSTERMAP = "clustermap"           # sns: clustermap (hierarchical clustering + heatmap)
    DOTPLOT = "dotplot"                 # sized dot matrix (common in GO enrichment)
    CONTOUR = "contour"                 # mpl: contour / plotly: go.Contour
    FILLED_CONTOUR = "filled_contour"   # mpl: contourf

    # Regression
    REGRESSION = "regression"           # sns: regplot / lmplot
    RESIDUAL = "residual"               # sns: residplot

    # Genomics-specific
    VOLCANO = "volcano"                 # -log10(p) vs log2FC scatter
    MA_PLOT = "ma_plot"                 # M (log ratio) vs A (mean expression)
    MANHATTAN = "manhattan"             # GWAS significance across chromosomes
    GENOME_BROWSER = "genome_browser"   # IGV-style track view
    CIRCOS = "circos"                   # circular genome plot
    IDEOGRAM = "ideogram"               # chromosome visualization
    COVERAGE_TRACK = "coverage_track"   # bigWig signal track

    # Dimensionality reduction
    UMAP = "umap"                       # UMAP scatter (2D/3D)
    TSNE = "tsne"                       # t-SNE scatter
    PCA = "pca"                         # PCA biplot / scatter

    # Network / Relationship
    NETWORK_GRAPH = "network_graph"     # networkx / plotly network
    SANKEY = "sankey"                   # plotly: go.Sankey
    VENN = "venn"                       # matplotlib_venn
    UPSET = "upset"                     # UpSet plot (set intersections)

    # Hierarchical
    DENDROGRAM = "dendrogram"           # scipy/matplotlib dendrogram
    TREEMAP = "treemap"                 # plotly: px.treemap
    SUNBURST = "sunburst"               # plotly: px.sunburst

    # Spatial
    SPATIAL_SCATTER = "spatial_scatter" # scanpy: pl.spatial
    TISSUE_IMAGE = "tissue_image"       # H&E or immunofluorescence overlay

    # Images
    IMAGE = "image"                     # microscopy, gel, blot scan
    FLOW_CYTOMETRY = "flow_cytometry"   # FACS dot/density plot

    # Pair/Joint (multi-variable)
    PAIRPLOT = "pairplot"               # sns: pairplot / PairGrid
    JOINTPLOT = "jointplot"             # sns: jointplot / JointGrid

    # Other
    PIE = "pie"                         # plotly: px.pie
    AREA = "area"                       # plotly: px.area
    WATERFALL = "waterfall"             # plotly: go.Waterfall
    SURFACE_3D = "surface_3d"           # mpl/plotly 3D surface
    SCATTER_3D = "scatter_3d"           # mpl/plotly 3D scatter
    OTHER = "other"                     # Catch-all for unrecognized types


# ── Axis Model ────────────────────────────────────────────────────────────────


class AxisScale(str, Enum):
    """Axis scale transform — determines matplotlib set_xscale / plotly log_x."""

    LINEAR = "linear"
    LOG2 = "log2"
    LOG10 = "log10"
    LN = "ln"
    SYMLOG = "symlog"                   # Symmetric log (handles negative values)
    CATEGORICAL = "categorical"
    REVERSED = "reversed"               # Inverted axis (common for p-values)


class Axis(BaseModel):
    """Axis metadata for a plot. Maps to matplotlib Axis / plotly xaxis config."""

    label: str
    scale: AxisScale = AxisScale.LINEAR
    units: Optional[str] = None
    data_type: Optional[str] = None     # e.g., "gene expression", "p-value"
    limits: Optional[tuple[float, float]] = None  # (min, max) explicit range
    is_inverted: bool = False           # True for -log10(p-value) axes
    tick_values: Optional[list[str]] = Field(
        default=None,
        description="Explicit tick labels (for categorical axes, chromosome names, etc.)",
    )


# ── Color Mapping ─────────────────────────────────────────────────────────────
# Derived from scientific-visualization skill (color_palettes.md):
# Okabe-Ito, Paul Tol, viridis family, plus seaborn palette types.


class ColormapType(str, Enum):
    """Colormap family — determines palette selection strategy."""

    SEQUENTIAL = "sequential"           # Low → high (viridis, plasma, rocket, mako)
    DIVERGING = "diverging"             # Centered (RdBu, coolwarm, vlag, PuOr)
    QUALITATIVE = "qualitative"         # Categorical (Okabe-Ito, Set2, tab10, colorblind)
    BINARY = "binary"                   # Two-class (e.g., up/down regulation)


class ColorMapping(BaseModel):
    """How data values are mapped to colors. Replaces the old color_axis."""

    variable: Optional[str] = None      # Data column or dimension being encoded
    colormap_type: ColormapType = ColormapType.QUALITATIVE
    colormap_name: Optional[str] = None  # e.g., "viridis", "RdBu_r", "Okabe-Ito"
    center_value: Optional[float] = None  # For diverging colormaps (e.g., 0 for log2FC)
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    n_colors: Optional[int] = None      # Number of discrete categories
    is_colorblind_safe: bool = True     # From scientific-visualization best practices
    labels: Optional[dict[str, str]] = Field(
        default=None,
        description="Category → color label mapping (e.g., {'cluster_1': 'CD8+ T cells'})",
    )


# ── Statistical Annotations ───────────────────────────────────────────────────
# From scientific-visualization skill: "Always include error bars, sample
# size, significance markers, individual data points when possible."


class ErrorBarType(str, Enum):
    """Error bar representation — must be specified in figure caption."""

    SD = "sd"                           # Standard deviation
    SEM = "sem"                         # Standard error of the mean
    CI_95 = "ci_95"                     # 95% confidence interval
    CI_99 = "ci_99"                     # 99% confidence interval
    IQR = "iqr"                         # Interquartile range
    MIN_MAX = "min_max"                 # Range
    NONE = "none"


class StatisticalAnnotation(BaseModel):
    """Statistical markers on a subplot (significance brackets, p-values, etc.)."""

    test_name: Optional[str] = None     # e.g., "t-test", "Wilcoxon", "ANOVA"
    comparisons: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of groups being compared (e.g., [('WT', 'KO')])",
    )
    significance_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "*": "p<0.05",
            "**": "p<0.01",
            "***": "p<0.001",
            "ns": "not significant",
        },
        description="Symbol → threshold mapping",
    )
    p_values: Optional[dict[str, float]] = Field(
        default=None,
        description="Comparison label → p-value (e.g., {'WT_vs_KO': 0.003})",
    )
    multiple_testing_correction: Optional[str] = None  # "Bonferroni", "BH", "FDR"


# ── Plot Layer (multi-trace support) ─────────────────────────────────────────
# Papers routinely overlay multiple plot types on one panel:
# boxplot + stripplot, scatter + regression line, histogram + KDE, etc.


class PlotLayer(BaseModel):
    """A single visual layer within a SubFigure. Multiple layers stack."""

    plot_type: PlotType
    is_primary: bool = True             # Primary layer vs. overlay
    library_hint: Optional[str] = Field(
        default=None,
        description=(
            "Preferred library: 'matplotlib', 'seaborn', 'plotly'. "
            "Resolved by Jupyter Generator if None."
        ),
    )
    function_hint: Optional[str] = Field(
        default=None,
        description=(
            "Specific function name, e.g., 'sns.stripplot', 'px.scatter'. "
            "Resolved by Jupyter Generator if None."
        ),
    )
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key visual parameters extracted from methods/caption "
            "(e.g., {'alpha': '0.3', 'jitter': 'True', 'order': '2'})"
        ),
    )


# ── Confidence and Geometry ──────────────────────────────────────────────────


class ConfidenceScores(BaseModel):
    """0-100 confidence estimates for subfigure interpretation fields."""

    label: float = 50.0
    description: float = 50.0
    plot_type: float = 50.0
    plot_category: float = 50.0
    x_axis: float = 50.0
    y_axis: float = 50.0
    color_variable: float = 50.0
    error_bars: float = 50.0
    sample_size: float = 50.0
    data_source: float = 50.0
    assays: float = 50.0
    statistical_test: float = 50.0
    facet_variable: float = 50.0


class PanelBoundingBox(BaseModel):
    """Subfigure/panel boundary in normalized figure coordinates."""

    x0: float = Field(default=0.0, ge=0.0, le=1.0)
    y0: float = Field(default=0.0, ge=0.0, le=1.0)
    x1: float = Field(default=1.0, ge=0.0, le=1.0)
    y1: float = Field(default=1.0, ge=0.0, le=1.0)
    coordinate_space: str = "normalized"
    detection_method: str = "layout_heuristic"

    @model_validator(mode="after")
    def _validate_box_bounds(self) -> "PanelBoundingBox":
        if self.x0 >= self.x1:
            raise ValueError("PanelBoundingBox requires x0 < x1")
        if self.y0 >= self.y1:
            raise ValueError("PanelBoundingBox requires y0 < y1")
        return self


# ── SubFigure ─────────────────────────────────────────────────────────────────


class SubFigure(BaseModel):
    """A panel within a composite figure (e.g., Fig. 1a).

    Expanded from original model based on evaluation against matplotlib,
    seaborn, plotly, and scientific-visualization skills. Key additions:
    - PlotType enum (50+ types) replaces freeform string
    - Multi-layer support (layers field) for overlaid plot types
    - ColorMapping separates data dimension from visual encoding
    - StatisticalAnnotation captures error bars, significance, p-values
    - z_axis for 3D plots, size_encoding for bubble charts
    """

    label: str                          # e.g., "a", "b", "1A"
    description: str                    # What this subfigure shows

    # ── Plot specification ──
    plot_category: PlotCategory         # High-level category for routing
    plot_type: PlotType                 # Primary plot type (controlled vocabulary)
    layers: list[PlotLayer] = Field(
        default_factory=list,
        description=(
            "Ordered list of visual layers. First = primary, rest = overlays. "
            "E.g., [PlotLayer(BOX, primary=True), PlotLayer(STRIP, primary=False)]"
        ),
    )

    # ── Axes ──
    x_axis: Optional[Axis] = None
    y_axis: Optional[Axis] = None
    z_axis: Optional[Axis] = None       # For 3D plots (surface, scatter_3d)
    color_mapping: Optional[ColorMapping] = None  # Replaces old color_axis
    size_encoding: Optional[str] = Field(
        default=None,
        description="Variable mapped to point/marker size (bubble charts, sized dot plots)",
    )

    # ── Statistical rigor ──
    error_bars: ErrorBarType = ErrorBarType.NONE
    sample_size: Optional[str] = None  # "n=3 per group", "n=1,234 cells"
    statistical_annotations: Optional[StatisticalAnnotation] = None
    shows_individual_points: bool = False  # Best practice: show raw data when possible

    # ── Data lineage ──
    data_source: Optional[str] = None  # Dataset reference, e.g., "GSE12345"
    assays: list[str] = Field(
        default_factory=list,
        description="Assay types shown (e.g., 'RNA-seq', 'ChIP-seq')",
    )
    methods_referenced: list[str] = Field(
        default_factory=list,
        description="Methods section references relevant to this subfigure",
    )
    supplementary_tables: list[str] = Field(
        default_factory=list,
        description="Associated supplementary table IDs (e.g., 'Table S1')",
    )

    # ── Faceting (for small multiples within a single panel) ──
    facet_variable: Optional[str] = Field(
        default=None,
        description=(
            "Variable used for faceting/small multiples within this panel "
            "(e.g., 'cell_type', 'timepoint'). "
            "Maps to seaborn col=/row= or plotly facet_col="
        ),
    )
    n_facets: Optional[int] = None      # Number of facet panels

    # ── Classification diagnostics (for complex-plot disambiguation) ──
    classification_confidence: float = Field(
        default=0.5,
        description=(
            "Confidence score (0-1) for the resolved plot_type after cue-based "
            "and/or LLM classification."
        ),
    )
    alternative_plot_types: list[PlotType] = Field(
        default_factory=list,
        description=(
            "Ranked fallback plot types considered plausible for this panel "
            "when classification is ambiguous."
        ),
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description=(
            "Short textual cue snippets (caption/in-text/description) that "
            "supported the selected plot type."
        ),
    )
    bioc_evidence_spans: list[BioCPassageContext] = Field(
        default_factory=list,
        description=(
            "Structured BioC evidence references supporting panel classification "
            "and confidence adjustments."
        ),
    )
    bioc_contradiction: bool = Field(
        default=False,
        description=(
            "True when the top BioC cue candidate contradicts the selected plot_type "
            "for this subfigure."
        ),
    )
    confidence_scores: ConfidenceScores = Field(
        default_factory=ConfidenceScores,
        description="Per-field confidence scores in the range 0-100.",
    )
    composite_confidence: float = Field(
        default=50.0,
        description="Composite confidence score in the range 0-100.",
    )
    boundary_box: Optional[PanelBoundingBox] = Field(
        default=None,
        description="Estimated panel boundary used to isolate subfigure interpretation.",
    )


# ── Figure-level layout and rendering ────────────────────────────────────────


class PanelLayout(BaseModel):
    """Layout specification for a multi-panel figure.
    Maps to matplotlib GridSpec / plotly make_subplots."""

    n_rows: int = 1
    n_cols: int = 1
    width_ratios: Optional[list[float]] = None   # GridSpec width_ratios
    height_ratios: Optional[list[float]] = None  # GridSpec height_ratios
    shared_x: bool = False
    shared_y: bool = False
    panel_labels_style: str = "uppercase"  # "uppercase" (A,B,C) or "lowercase" (a,b,c)


class RenderingSpec(BaseModel):
    """Export and rendering configuration.
    From scientific-visualization skill: journal-specific requirements."""

    target_journal: Optional[str] = None  # "nature", "science", "cell", "plos", etc.
    figure_width_mm: Optional[float] = None  # Column width (e.g., 89mm for Nature single)
    figure_height_mm: Optional[float] = None
    dpi: int = 300                      # 300 for raster, 600+ for line art
    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "png"],
        description=(
            "Export formats. Vector (pdf, svg, eps) preferred for plots; "
            "TIFF/PNG for images. Never JPEG for scientific data."
        ),
    )
    colorblind_tested: bool = False
    grayscale_compatible: bool = False


class Figure(BaseModel):
    """A complete figure from a paper, with all subfigures, layout, and rendering spec."""

    figure_id: str                      # e.g., "Figure 1", "Fig. 3"
    title: str                          # Figure title / first line of caption
    caption: str                        # Full caption text
    purpose: str = Field(
        description="High-level answer to: What is this figure trying to show?",
    )
    subfigures: list[SubFigure] = Field(default_factory=list)
    layout: PanelLayout = Field(
        default_factory=PanelLayout,
        description="Grid layout of panels. Defaults to single panel.",
    )
    rendering: Optional[RenderingSpec] = None
    in_text_context: list[str] = Field(
        default_factory=list,
        description="Sentences from the paper body that reference this figure",
    )
    datasets_used: list[str] = Field(
        default_factory=list,
        description="Dataset identifiers referenced by this figure",
    )
    methods_used: list[str] = Field(
        default_factory=list,
        description="Method/assay names used to generate this figure",
    )
    preview_url: Optional[str] = Field(
        default=None,
        description="Direct HTTPS URL to the primary figure image when resolvable from PMC.",
    )
    parse_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable parse degradation warnings. "
            "Empty list means no known fallbacks were triggered."
        ),
    )
