"""Methods-level data models.

Represents the structured methodology of a paper, including individual
assays, analysis steps, and the DAG of dependencies between them.

The AssayGraph model was added after Ouroboros evaluation revealed that
multi-omic papers (CLIP-seq + RNA-seq, ChIP-seq + ATAC-seq) have non-linear
assay dependencies that a flat list cannot represent.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MethodCategory(str, Enum):
    """Whether an assay is a wet-lab protocol, a computational analysis, or both.

    Used by MethodsParser to distinguish experimental from computational steps
    so that downstream pipeline generation can focus on reproducible
    computational workflows.
    """

    experimental = "experimental"   # Wet-lab / instrument-based protocol
    computational = "computational"  # Bioinformatics / statistical analysis
    mixed = "mixed"                 # Contains both wet-lab and computational steps


class AnalysisStep(BaseModel):
    """One concrete step in an analysis pipeline."""

    step_number: int
    description: str
    input_data: str                     # What data this step consumes
    output_data: str                    # What data this step produces
    software: Optional[str] = None      # Software name (links to Software model)
    software_version: Optional[str] = None
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Key parameters used (e.g., {'min_quality': '30', 'aligner': 'STAR'})",
    )
    code_reference: Optional[str] = None  # URL to code if available


class Assay(BaseModel):
    """A single assay or experiment described in the methods."""

    name: str                           # e.g., "RNA-seq", "ATAC-seq", "Western Blot"
    description: str                    # Prose description from the paper
    data_type: str = Field(
        description=(
            "What kind of data the assay operates on — e.g., 'sequencing', "
            "'imaging', 'proteomics', 'mass_spec', 'flow_cytometry', 'other'. "
            "This describes the *data modality*, not whether the assay is "
            "wet-lab vs computational (use method_category for that)."
        ),
    )
    method_category: MethodCategory = Field(
        default=MethodCategory.experimental,
        description=(
            "Whether this assay is a wet-lab experimental protocol ('experimental'), "
            "a computational analysis step ('computational'), or contains both ('mixed'). "
            "Set by MethodsParser._classify_assays; defaults to 'experimental' when "
            "classification is unavailable."
        ),
    )
    raw_data_source: Optional[str] = None  # Where raw data lives (e.g., "GEO: GSE12345")
    steps: list[AnalysisStep] = Field(default_factory=list)
    figures_produced: list[str] = Field(
        default_factory=list,
        description="Figure IDs this assay contributes to",
    )


class AssayDependency(BaseModel):
    """A directed edge in the assay dependency graph.

    Example: ChIP-seq peaks depend on ATAC-seq open chromatin regions;
    CLIP-seq differential binding depends on RNA-seq expression values.
    """

    upstream_assay: str                 # Name of the prerequisite assay
    downstream_assay: str               # Name of the dependent assay
    dependency_type: str = Field(
        description=(
            "How the upstream output feeds the downstream assay. "
            "E.g., 'peak_filter', 'normalization_reference', 'co-analysis', 'integration'"
        ),
    )
    description: str = ""              # Human-readable explanation of the dependency


class AssayGraph(BaseModel):
    """Directed acyclic graph of assay dependencies for a paper.

    Replaces the flat assay list from the original Method model to properly
    represent multi-omic pipelines where order and data flow matter.
    The DAG determines PipelineStep.depends_on ordering in the generated
    Snakemake/Nextflow workflow.
    """

    assays: list[Assay] = Field(default_factory=list)
    dependencies: list[AssayDependency] = Field(default_factory=list)

    def get_assay(self, name: str) -> Optional[Assay]:
        """Return assay by name (case-insensitive)."""
        name_lower = name.lower()
        for assay in self.assays:
            if assay.name.lower() == name_lower:
                return assay
        return None

    def upstream_of(self, assay_name: str) -> list[str]:
        """Return names of all assays that must run before assay_name."""
        return [
            dep.upstream_assay
            for dep in self.dependencies
            if dep.downstream_assay == assay_name
        ]

    def downstream_of(self, assay_name: str) -> list[str]:
        """Return names of all assays that depend on assay_name."""
        return [
            dep.downstream_assay
            for dep in self.dependencies
            if dep.upstream_assay == assay_name
        ]


class Method(BaseModel):
    """Complete parsed methodology for a paper."""

    paper_doi: Optional[str] = None
    assay_graph: AssayGraph = Field(
        default_factory=AssayGraph,
        description=(
            "DAG of assays and their dependencies. Use assay_graph.assays "
            "for the flat list and assay_graph.dependencies for ordering."
        ),
    )
    data_availability: str = ""         # Data availability statement
    code_availability: str = ""         # Code availability statement
    raw_methods_text: str = ""          # Original methods section text
    parse_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal diagnostics emitted during parsing — e.g., assay stub "
            "reasons, dropped dependency edges, and unresolved assay names. "
            "An empty list means the parse completed without degraded results."
        ),
    )

    # Convenience accessor kept for backward compatibility
    @property
    def assays(self) -> list[Assay]:
        """Backward-compatible alias for ``assay_graph.assays``."""
        return self.assay_graph.assays
