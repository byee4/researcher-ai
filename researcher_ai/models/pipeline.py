"""Pipeline data models.

Represents the generated analysis pipeline — either Snakemake or Nextflow —
ready for execution. Each PipelineStep maps to one rule (Snakemake) or
one process (Nextflow).

DAG support (depends_on field) was added after Ouroboros evaluation found
that multi-omic papers require non-linear step ordering.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineBackend(str, Enum):
    SNAKEMAKE = "snakemake"
    NEXTFLOW = "nextflow"


class PipelineStep(BaseModel):
    """A single step in the generated pipeline.

    Maps to one Snakemake rule or one Nextflow process.
    The depends_on field enables non-linear DAG execution ordering.
    """

    step_id: str                        # Unique ID, e.g., "align_reads"
    name: str
    description: str
    software: str                       # Software name (matches Software.name)
    software_version: Optional[str] = None
    command: str                        # Templated command string
    inputs: list[str] = Field(default_factory=list)   # File patterns or placeholders
    outputs: list[str] = Field(default_factory=list)  # File patterns
    parameters: dict[str, Any] = Field(default_factory=dict)
    threads: int = 1
    memory_gb: int = 4
    container: Optional[str] = None     # Docker/Singularity image
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "step_ids that must complete before this step runs. "
            "Enables non-linear DAG ordering (e.g., CLIP-seq peak calling "
            "depends on both the CLIP-seq alignment AND the matched RNA-seq "
            "expression values for normalization)."
        ),
    )
    conda_env: Optional[str] = None     # Per-step conda env name or YAML path
    nf_core_module: Optional[str] = Field(
        default=None,
        description="nf-core module name if this step maps to one (e.g., 'star/align')",
    )


class PipelineConfig(BaseModel):
    """Configuration for the full pipeline."""

    name: str
    description: str
    backend: PipelineBackend
    steps: list[PipelineStep] = Field(default_factory=list)
    datasets: list[str] = Field(
        default_factory=list,
        description="Dataset accession IDs to process",
    )
    figure_targets: list[str] = Field(
        default_factory=list,
        description="Figure IDs this pipeline aims to reproduce",
    )
    environment: Optional[str] = None  # Global env spec (conda YAML or Dockerfile)
    nf_core_pipeline: Optional[str] = Field(
        default=None,
        description="nf-core pipeline name if applicable (e.g., 'rnaseq', 'atacseq')",
    )
    nf_core_version: Optional[str] = None

    def execution_order(self) -> list[str]:
        """Topological sort of step_ids based on depends_on edges.

        Returns step_ids in an order where all dependencies of a step
        appear before the step itself.
        """
        step_map = {s.step_id: s for s in self.steps}
        visited: set[str] = set()
        order: list[str] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on:
                    visit(dep)
            order.append(step_id)

        for step in self.steps:
            visit(step.step_id)

        return order


class Pipeline(BaseModel):
    """Complete pipeline output, ready for code generation."""

    config: PipelineConfig
    snakefile_content: Optional[str] = None
    nextflow_content: Optional[str] = None
    jupyter_content: Optional[str] = None  # JSON string of .ipynb
    conda_env_yaml: Optional[str] = None
    dockerfile: Optional[str] = None
