"""Jupyter notebook generator for figure reproduction.

Builds an ``nbformat`` notebook with setup imports, figure-level markdown, and
plotting cells mapped from subfigure plot metadata.
"""

from __future__ import annotations

import json
from typing import Optional

from researcher_ai.models.figure import Figure, PlotCategory, PlotType, SubFigure
from researcher_ai.models.pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# Plot type → (library, function) mapping
# ---------------------------------------------------------------------------

# Maps PlotType → (import_alias, function_call_template)
# {data} and {kwargs} are substitution markers.
_PLOT_DISPATCH: dict[PlotType, tuple[str, str]] = {
    # Relational
    PlotType.SCATTER:        ("sns", "sns.scatterplot(data=df, {kwargs})"),
    PlotType.LINE:           ("sns", "sns.lineplot(data=df, {kwargs})"),
    PlotType.BUBBLE:         ("plt", "plt.scatter({kwargs}, s=df['size'])"),
    PlotType.STEP:           ("plt", "plt.step(df['x'], df['y'], {kwargs})"),
    # Distribution
    PlotType.HISTOGRAM:      ("sns", "sns.histplot(data=df, {kwargs})"),
    PlotType.KDE:            ("sns", "sns.kdeplot(data=df, {kwargs})"),
    PlotType.ECDF:           ("sns", "sns.ecdfplot(data=df, {kwargs})"),
    PlotType.RUG:            ("sns", "sns.rugplot(data=df, {kwargs})"),
    PlotType.DENSITY_2D:     ("sns", "sns.kdeplot(data=df, {kwargs})"),
    PlotType.HEXBIN:         ("plt", "plt.hexbin(df['x'], df['y'], {kwargs})"),
    # Categorical
    PlotType.BAR:            ("sns", "sns.barplot(data=df, {kwargs})"),
    PlotType.GROUPED_BAR:    ("sns", "sns.barplot(data=df, hue='group', {kwargs})"),
    PlotType.STACKED_BAR:    ("plt", "df.plot(kind='bar', stacked=True, {kwargs})"),
    PlotType.BOX:            ("sns", "sns.boxplot(data=df, {kwargs})"),
    PlotType.VIOLIN:         ("sns", "sns.violinplot(data=df, {kwargs})"),
    PlotType.STRIP:          ("sns", "sns.stripplot(data=df, {kwargs})"),
    PlotType.SWARM:          ("sns", "sns.swarmplot(data=df, {kwargs})"),
    PlotType.BOXEN:          ("sns", "sns.boxenplot(data=df, {kwargs})"),
    PlotType.POINT:          ("sns", "sns.pointplot(data=df, {kwargs})"),
    PlotType.COUNT:          ("sns", "sns.countplot(data=df, {kwargs})"),
    # Matrix
    PlotType.HEATMAP:        ("sns", "sns.heatmap(df, {kwargs})"),
    PlotType.CLUSTERMAP:     ("sns", "sns.clustermap(df, {kwargs})"),
    PlotType.DOTPLOT:        ("plt", "plt.scatter({kwargs})  # dotplot: size encodes value"),
    PlotType.CONTOUR:        ("plt", "plt.contour(df['x'], df['y'], df['z'], {kwargs})"),
    PlotType.FILLED_CONTOUR: ("plt", "plt.contourf(df['x'], df['y'], df['z'], {kwargs})"),
    # Regression
    PlotType.REGRESSION:     ("sns", "sns.regplot(data=df, {kwargs})"),
    PlotType.RESIDUAL:       ("sns", "sns.residplot(data=df, {kwargs})"),
    # Genomics
    PlotType.VOLCANO: (
        "plt",
        "plt.scatter(df['log2FC'], df['-log10p'], c=df['color'], {kwargs})\n"
        "plt.axhline(-np.log10(0.05), ls='--', color='gray')\n"
        "plt.axvline(-1, ls='--', color='gray'); plt.axvline(1, ls='--', color='gray')",
    ),
    PlotType.MA_PLOT: (
        "plt",
        "plt.scatter(df['A'], df['M'], c=df['color'], alpha=0.3, {kwargs})\n"
        "plt.axhline(0, ls='--', color='gray')",
    ),
    PlotType.MANHATTAN: (
        "plt",
        "# Manhattan plot — use manhattan_plot() or custom scatter per chromosome\n"
        "plt.scatter(df['pos'], -np.log10(df['pval']), {kwargs})",
    ),
    PlotType.GENOME_BROWSER: ("plt", "# Genome browser track — use pyGenomeTracks or IGV\n# ax.fill_between(positions, coverage, {kwargs})"),
    PlotType.CIRCOS:          ("plt", "# Circos plot — use pycircos or pyCirclize\n# circos.add_track({kwargs})"),
    PlotType.COVERAGE_TRACK:  ("plt", "ax.fill_between(positions, coverage, {kwargs})"),
    # Dimensionality
    PlotType.UMAP:   ("plt", "plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, {kwargs})"),
    PlotType.TSNE:   ("plt", "plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, {kwargs})"),
    PlotType.PCA:    ("plt", "plt.scatter(pca_df['PC1'], pca_df['PC2'], {kwargs})"),
    # Network
    PlotType.NETWORK_GRAPH: ("nx", "nx.draw_networkx(G, pos=nx.spring_layout(G), {kwargs})"),
    PlotType.SANKEY:        ("px", "px.sankey({kwargs})"),
    PlotType.VENN:          ("venn", "venn2([set1, set2], set_labels=('A', 'B'))"),
    PlotType.UPSET:         ("upsetplot", "upsetplot.UpSet(df).plot()"),
    # Hierarchical
    PlotType.DENDROGRAM:    ("scipy", "scipy.cluster.hierarchy.dendrogram(Z, {kwargs})"),
    PlotType.TREEMAP:       ("px", "px.treemap(df, {kwargs})"),
    PlotType.SUNBURST:      ("px", "px.sunburst(df, {kwargs})"),
    # Spatial / Image
    PlotType.SPATIAL_SCATTER: ("sc", "sc.pl.spatial(adata, {kwargs})"),
    PlotType.TISSUE_IMAGE:    ("plt", "plt.imshow(image, {kwargs})"),
    PlotType.IMAGE:           ("plt", "plt.imshow(image, {kwargs})"),
    PlotType.FLOW_CYTOMETRY:  ("plt", "plt.scatter(df['FSC'], df['SSC'], alpha=0.1, {kwargs})"),
    # Pair / Joint
    PlotType.PAIRPLOT:  ("sns", "sns.pairplot(df, {kwargs})"),
    PlotType.JOINTPLOT: ("sns", "sns.jointplot(data=df, {kwargs})"),
    # Misc
    PlotType.PIE:        ("plt", "plt.pie(df['values'], labels=df['labels'], {kwargs})"),
    PlotType.AREA:       ("plt", "df.plot.area({kwargs})"),
    PlotType.WATERFALL:  ("plt", "plt.bar(df['x'], df['value'], bottom=df['bottom'], {kwargs})"),
    PlotType.SURFACE_3D: ("plt", "ax.plot_surface(X, Y, Z, {kwargs})"),
    PlotType.SCATTER_3D: ("plt", "ax.scatter(df['x'], df['y'], df['z'], {kwargs})"),
    PlotType.IDEOGRAM:   ("plt", "# Ideogram — use matplotlib_scalebar or custom code"),
    PlotType.OTHER:      ("plt", "# TODO: implement plot for this subfigure"),
}

# Required imports per library alias
_LIBRARY_IMPORTS: dict[str, str] = {
    "sns": "import seaborn as sns",
    "plt": "import matplotlib.pyplot as plt",
    "px":  "import plotly.express as px",
    "nx":  "import networkx as nx",
    "venn": "from matplotlib_venn import venn2, venn3",
    "upsetplot": "from upsetplot import UpSet",
    "scipy": "import scipy.cluster.hierarchy",
    "sc": "import scanpy as sc",
    "np": "import numpy as np",
    "pd": "import pandas as pd",
}


class JupyterGenerator:
    """Generate a Jupyter notebook that reproduces parsed paper figures."""

    def generate(self, config: PipelineConfig, figures: list[Figure]) -> str:
        """Generate a Jupyter notebook as a JSON string (nbformat v4).

        Returns:
            nbformat-compatible JSON string, ready to write to .ipynb.
        """
        try:
            import nbformat  # type: ignore[import]
            nb = nbformat.v4.new_notebook()
            nb.cells = [self._setup_cell(config)]
            for figure in figures:
                nb.cells.extend(self._figure_cells(figure, config))
            return nbformat.writes(nb)
        except ImportError:
            # nbformat not installed — return a minimal valid notebook JSON
            return self._fallback_notebook(config, figures)

    # ------------------------------------------------------------------
    # Cell generators
    # ------------------------------------------------------------------

    def _setup_cell(self, config: "PipelineConfig") -> object:
        """Generate the setup/import cell."""
        # Collect all library imports needed across all figures handled
        code_lines = [
            "# researcher-ai generated notebook",
            f"# Pipeline: {config.name}",
            "#",
            "import os",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from pathlib import Path",
            "",
            "# Results directory (adjust as needed)",
            f"RESULTS_DIR = Path('results/')",
            "",
        ]
        if config.datasets:
            code_lines += [
                "# Datasets used in this pipeline",
                "DATASETS = " + repr(config.datasets),
                "",
            ]
        if config.figure_targets:
            code_lines += [
                "# Figures to reproduce",
                "FIGURE_TARGETS = " + repr(config.figure_targets),
                "",
            ]
        code_lines += [
            "# Plotting defaults",
            "sns.set_theme(style='whitegrid', font_scale=1.2)",
            "plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300})",
        ]

        return self._make_code_cell("\n".join(code_lines))

    def _figure_cells(
        self, figure: Figure, config: PipelineConfig
    ) -> list[object]:
        """Generate cells to reproduce a single figure.

        For each subfigure:
        1. Markdown cell describing the figure and subfigure.
        2. Code cell to load the processed data.
        3. Code cell to generate the plot matching the subfigure type.
        """
        cells: list[object] = []

        # Figure-level markdown header
        cells.append(
            self._make_markdown_cell(
                f"## {figure.figure_id}\n\n"
                f"**{figure.title}**\n\n"
                f"{figure.purpose}\n\n"
                f"> *{figure.caption[:300]}{'...' if len(figure.caption) > 300 else ''}*"
            )
        )

        if not figure.subfigures:
            cells.append(
                self._make_markdown_cell(
                    f"*No subfigures parsed for {figure.figure_id}.*"
                )
            )
            return cells

        for subfig in figure.subfigures:
            # Markdown: subfigure description
            cells.append(
                self._make_markdown_cell(
                    f"### {figure.figure_id}{subfig.label}\n\n"
                    f"{subfig.description}"
                    + (
                        f"\n\n- Plot type: `{subfig.plot_type.value}`"
                        f"\n- Category: `{subfig.plot_category.value}`"
                    )
                )
            )

            # Code: data loading
            data_source = subfig.data_source or (
                config.datasets[0] if config.datasets else "dataset"
            )
            load_lines = [
                f"# Load data for {figure.figure_id}{subfig.label}",
                f"# Source: {data_source}",
                f"data_path = RESULTS_DIR / '{_safe_stem(data_source)}_processed.csv'",
                f"# df = pd.read_csv(data_path)  # uncomment when data is available",
                f"df = pd.DataFrame()  # placeholder — replace with actual data",
            ]
            cells.append(self._make_code_cell("\n".join(load_lines)))

            # Code: plot generation
            cells.append(
                self._make_code_cell(self._plot_code(subfig, figure))
            )

        return cells

    def _plot_code(self, subfig: SubFigure, figure: Figure) -> str:
        """Generate plotting code for a subfigure.

        Uses the PlotType → library mapping, with layer-level overrides.
        """
        # Determine the effective plot type (primary layer overrides subfig.plot_type)
        primary_layer = next(
            (lay for lay in subfig.layers if lay.is_primary), None
        )
        plot_type = subfig.plot_type
        if primary_layer and primary_layer.plot_type:
            plot_type = primary_layer.plot_type

        # Get dispatch entry
        _lib, plot_expr = _PLOT_DISPATCH.get(
            plot_type, ("plt", "plt.plot()  # TODO: implement")
        )

        # Resolve library hint override
        if primary_layer and primary_layer.function_hint:
            plot_expr = f"{primary_layer.function_hint}(data=df)"
        elif primary_layer and primary_layer.library_hint:
            # Keep plot_expr but note library preference
            pass

        # Build kwargs from axis info
        kwargs_parts: list[str] = []
        if subfig.x_axis:
            kwargs_parts.append(f"x='{subfig.x_axis.label}'")
        if subfig.y_axis:
            kwargs_parts.append(f"y='{subfig.y_axis.label}'")
        if subfig.color_mapping and subfig.color_mapping.variable:
            kwargs_parts.append(f"hue='{subfig.color_mapping.variable}'")
        kwargs_str = ", ".join(kwargs_parts)

        plot_expr = plot_expr.replace("{kwargs}", kwargs_str)

        # Build extra overlays from non-primary layers
        overlay_lines: list[str] = []
        for layer in subfig.layers:
            if not layer.is_primary:
                _ol_lib, ol_expr = _PLOT_DISPATCH.get(
                    layer.plot_type, ("plt", "# overlay layer")
                )
                overlay_lines.append(f"# Overlay: {layer.plot_type.value}")
                overlay_lines.append(ol_expr.replace("{kwargs}", ""))

        # Axis labels and scale
        axis_lines: list[str] = []
        if subfig.x_axis:
            axis_lines.append(f"plt.xlabel('{subfig.x_axis.label}')")
            if subfig.x_axis.scale.value not in ("linear", "categorical"):
                axis_lines.append(
                    f"plt.xscale('{subfig.x_axis.scale.value}')"
                )
        if subfig.y_axis:
            axis_lines.append(f"plt.ylabel('{subfig.y_axis.label}')")
            if subfig.y_axis.scale.value not in ("linear", "categorical"):
                axis_lines.append(
                    f"plt.yscale('{subfig.y_axis.scale.value}')"
                )

        # Statistical annotations
        stat_lines: list[str] = []
        if subfig.statistical_annotations and subfig.statistical_annotations.comparisons:
            stat_lines.append(
                "# Statistical annotations (use statannotations or manual lines)"
            )
            for grp1, grp2 in subfig.statistical_annotations.comparisons[:3]:
                stat_lines.append(f"# Comparison: {grp1} vs {grp2}")

        # Assemble code
        lines = [
            f"# {figure.figure_id}{subfig.label}: {subfig.description[:80]}",
            "fig, ax = plt.subplots(figsize=(6, 4))",
            plot_expr,
        ]
        lines += overlay_lines
        lines += axis_lines
        lines += stat_lines
        lines += [
            f"plt.title('{figure.figure_id}{subfig.label}')",
            "plt.tight_layout()",
            f"plt.savefig(RESULTS_DIR / '{figure.figure_id.replace(' ', '_')}{subfig.label}.pdf', bbox_inches='tight')",
            "plt.show()",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # nbformat helpers
    # ------------------------------------------------------------------

    def _make_code_cell(self, source: str) -> object:
        """Create an nbformat v4 code cell (or a dict if nbformat unavailable)."""
        try:
            import nbformat
            return nbformat.v4.new_code_cell(source)
        except ImportError:
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }

    def _make_markdown_cell(self, source: str) -> object:
        """Create an nbformat v4 markdown cell."""
        try:
            import nbformat
            return nbformat.v4.new_markdown_cell(source)
        except ImportError:
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            }

    def _fallback_notebook(
        self, config: PipelineConfig, figures: list[Figure]
    ) -> str:
        """Return minimal notebook JSON when nbformat is unavailable."""
        cells = []

        # Setup cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                "import numpy as np\nimport pandas as pd\n"
                "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
                f"\n# Pipeline: {config.name}\n"
            ),
        })

        for figure in figures:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": f"## {figure.figure_id}\n\n{figure.title}",
            })
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": f"# Reproduce {figure.figure_id}\n# TODO: load data and plot",
            })

        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": cells,
        }
        return json.dumps(nb, indent=2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_stem(text: str) -> str:
    """Convert a string (e.g., dataset accession) to a safe filename stem."""
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_") or "data"
