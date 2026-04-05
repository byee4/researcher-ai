"""Pipeline builder: orchestrate data + software + methods into executable pipeline.

Takes parsed paper components (Method, Dataset, Software, Figure) and produces
a complete Pipeline with Snakemake and/or Nextflow workflow code plus a Jupyter
notebook for figure reproduction.

DAG-aware step ordering uses AssayGraph.dependencies so multi-omic pipelines
(e.g., CLIP-seq dependent on RNA-seq expression) produce correct depends_on chains.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Optional

from researcher_ai.models.dataset import Dataset
from researcher_ai.models.figure import Figure
from researcher_ai.models.method import (
    Assay,
    AnalysisStep,
    AssayDependency,
    AssayGraph,
    Method,
    MethodCategory,
)
from researcher_ai.models.pipeline import (
    Pipeline,
    PipelineBackend,
    PipelineConfig,
    PipelineStep,
)
from researcher_ai.models.software import Software
from researcher_ai.pipeline.jupyter_gen import JupyterGenerator
from researcher_ai.pipeline.nextflow_gen import NextflowGenerator
from researcher_ai.pipeline.snakemake_gen import SnakemakeGenerator


# ---------------------------------------------------------------------------
# nf-core pipeline mapping
# ---------------------------------------------------------------------------

NFCORE_MAPPING: dict[str, str] = {
    "RNA-seq": "rnaseq",
    "RNAseq": "rnaseq",
    "ATAC-seq": "atacseq",
    "ATACseq": "atacseq",
    "ChIP-seq": "chipseq",
    "ChIPseq": "chipseq",
    "WGS": "sarek",
    "WES": "sarek",
    "Hi-C": "hic",
    "HiC": "hic",
    "methylation": "methylseq",
    "ampliseq": "ampliseq",
    "16S": "ampliseq",
    "metagenomics": "mag",
}


def _sanitize_id(text: str) -> str:
    """Convert arbitrary text to a valid snake_case identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", text.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "step"


class PipelineBuilder:
    """Build an executable pipeline from parsed paper components.

    Takes:
    - Method object (assays with analysis steps and dependency graph)
    - List of Dataset objects (metadata, sample info)
    - List of Software objects (tools with environments)
    - List of Figure objects (target endpoints)

    Produces:
    - PipelineConfig with ordered PipelineSteps (DAG-aware)
    - Choice of Snakemake or Nextflow backend (or both)
    - Identifies nf-core pipelines when applicable
    - Jupyter notebook for figure reproduction
    - Conda environment YAML
    """

    def __init__(
        self,
        llm_model: str = os.environ.get("RESEARCHER_AI_MODEL", "gpt-5.4"),
    ):
        """Initialize PipelineBuilder.

        Args:
            llm_model: Reserved model identifier for future LLM-assisted planning.
        """
        self.llm_model = llm_model
        self._software_index: dict[str, Software] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        method: Method,
        datasets: list[Dataset],
        software: list[Software],
        figures: list[Figure],
        backend: PipelineBackend = PipelineBackend.SNAKEMAKE,
    ) -> Pipeline:
        """Build a complete pipeline.

        Strategy:
        1. Index software by name for fast lookup.
        2. Keep only assays classified as computational for downstream pipelining.
        3. For each remaining assay (topo-sorted), check if nf-core applies.
        4. Build PipelineConfig with ordered, dependency-linked PipelineSteps.
        5. Generate backend-specific workflow code (Snakemake and/or Nextflow).
        6. Generate Jupyter notebook for figure reproduction.
        7. Generate unified conda environment YAML.
        """
        # Build software lookup
        self._software_index = {s.name.lower(): s for s in software}

        method_for_pipeline = self._computational_only_method(method)
        config = self._build_config(method_for_pipeline, datasets, figures, backend)
        pipeline = Pipeline(config=config)

        # Always generate Snakemake for custom steps
        if backend == PipelineBackend.SNAKEMAKE or self._needs_custom_steps(config):
            pipeline.snakefile_content = SnakemakeGenerator().generate(config)

        # Generate Nextflow when requested or nf-core steps exist
        if backend == PipelineBackend.NEXTFLOW or self._has_nfcore_steps(config):
            pipeline.nextflow_content = NextflowGenerator().generate(config)

        pipeline.jupyter_content = JupyterGenerator().generate(config, figures)
        pipeline.conda_env_yaml = self._generate_conda_env(software)

        return pipeline

    def _computational_only_method(self, method: Method) -> Method:
        """Return a Method containing only computational assays and valid edges."""
        assays = [
            assay
            for assay in method.assay_graph.assays
            if assay.method_category == MethodCategory.computational
        ]
        assay_names = {assay.name for assay in assays}
        dependencies: list[AssayDependency] = [
            dep
            for dep in method.assay_graph.dependencies
            if dep.upstream_assay in assay_names and dep.downstream_assay in assay_names
        ]
        return method.model_copy(update={"assay_graph": AssayGraph(assays=assays, dependencies=dependencies)})

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------

    def _build_config(
        self,
        method: Method,
        datasets: list[Dataset],
        figures: list[Figure],
        backend: PipelineBackend,
    ) -> PipelineConfig:
        """Build pipeline configuration from parsed components.

        1. Topologically sort assays using AssayGraph dependency edges.
        2. Detect if any assay maps to an nf-core pipeline.
        3. Convert each AnalysisStep to a PipelineStep with proper depends_on.
        4. Resolve dataset accessions and figure targets.
        """
        graph = method.assay_graph
        sorted_assay_names = self._topo_sort_assays(method)

        # Detect nf-core pipeline (first match wins for primary backend)
        nf_core_pipeline: Optional[str] = None
        for name in sorted_assay_names:
            match = self._check_nfcore(name)
            if match and nf_core_pipeline is None:
                nf_core_pipeline = match

        # Build PipelineSteps, tracking step_ids per assay for depends_on
        all_steps: list[PipelineStep] = []
        # Maps assay name → list of its step_ids (to chain inter-assay deps)
        assay_step_ids: dict[str, list[str]] = {}

        for assay_name in sorted_assay_names:
            assay = graph.get_assay(assay_name)
            if assay is None:
                continue

            # Determine which upstream assays this assay depends on
            upstream_assay_names = graph.upstream_of(assay_name)
            # Last step_id(s) of each upstream assay form cross-assay depends_on
            upstream_terminal_ids: list[str] = []
            for up_name in upstream_assay_names:
                up_ids = assay_step_ids.get(up_name, [])
                if up_ids:
                    upstream_terminal_ids.append(up_ids[-1])

            step_ids_this_assay: list[str] = []
            prev_step_id: Optional[str] = None

            for analysis_step in assay.steps:
                step = self._analysis_step_to_pipeline_step(
                    analysis_step=analysis_step,
                    assay=assay,
                    prev_step_id=prev_step_id,
                    upstream_terminal_ids=upstream_terminal_ids if prev_step_id is None else [],
                )
                all_steps.append(step)
                step_ids_this_assay.append(step.step_id)
                prev_step_id = step.step_id

            assay_step_ids[assay_name] = step_ids_this_assay

        # If no steps were generated (e.g., assays with no AnalysisSteps),
        # create placeholder steps from assay names.
        if not all_steps:
            all_steps = self._placeholder_steps(method, sorted_assay_names)

        return PipelineConfig(
            name=_sanitize_id(method.paper_doi or "pipeline"),
            description=(
                f"Reproducible pipeline generated from paper "
                f"({method.paper_doi or 'unknown DOI'}). "
                f"Assays: {', '.join(sorted_assay_names)}."
            ),
            backend=backend,
            steps=all_steps,
            datasets=[d.accession for d in datasets],
            figure_targets=[f.figure_id for f in figures],
            nf_core_pipeline=nf_core_pipeline,
        )

    def _analysis_step_to_pipeline_step(
        self,
        analysis_step: AnalysisStep,
        assay: Assay,
        prev_step_id: Optional[str],
        upstream_terminal_ids: list[str],
    ) -> PipelineStep:
        """Convert one AnalysisStep to a PipelineStep.

        - step_id: <assay_name>_step<N>
        - depends_on: previous step within assay + terminal steps of upstream assays
        - inputs/outputs: taken from AnalysisStep.input_data / output_data
        - command: from matching Software.commands[0], else a templated placeholder
        - threads/memory: inferred from software category or default
        """
        assay_id = _sanitize_id(assay.name)
        step_id = f"{assay_id}_step{analysis_step.step_number}"

        depends_on: list[str] = []
        if prev_step_id:
            depends_on.append(prev_step_id)
        depends_on.extend(upstream_terminal_ids)

        # Look up matching Software object for command and container info
        sw_name = (analysis_step.software or "").lower()
        sw_obj = self._software_index.get(sw_name)

        command = self._resolve_command(analysis_step, sw_obj)
        threads, memory_gb = self._infer_resources(sw_name)

        container: Optional[str] = None
        conda_env: Optional[str] = None
        nf_core_module: Optional[str] = None

        if sw_obj:
            if sw_obj.environment and sw_obj.environment.docker_image:
                container = sw_obj.environment.docker_image
            if sw_obj.bioconda_package:
                conda_env = f"bioconda::{sw_obj.bioconda_package}"
            # Map well-known tools to nf-core modules
            nf_core_module = self._check_nfcore_module(sw_name)

        return PipelineStep(
            step_id=step_id,
            name=f"{assay.name} — {analysis_step.description[:60]}",
            description=analysis_step.description,
            software=analysis_step.software or "unknown",
            software_version=analysis_step.software_version,
            command=command,
            inputs=[analysis_step.input_data] if analysis_step.input_data else [],
            outputs=[analysis_step.output_data] if analysis_step.output_data else [],
            parameters=analysis_step.parameters,
            threads=threads,
            memory_gb=memory_gb,
            container=container,
            depends_on=depends_on,
            conda_env=conda_env,
            nf_core_module=nf_core_module,
        )

    def _placeholder_steps(
        self, method: Method, sorted_assay_names: list[str]
    ) -> list[PipelineStep]:
        """Generate placeholder steps for assays that have no AnalysisSteps."""
        steps = []
        prev_id: Optional[str] = None
        for assay_name in sorted_assay_names:
            assay = method.assay_graph.get_assay(assay_name)
            if assay is None:
                continue
            step_id = f"{_sanitize_id(assay_name)}_run"
            steps.append(
                PipelineStep(
                    step_id=step_id,
                    name=f"Run {assay.name}",
                    description=assay.description or f"Execute {assay.name} analysis.",
                    software=assay.name,
                    command=f"# TODO: add {assay.name} command",
                    inputs=[assay.raw_data_source or "input_data"] if assay.raw_data_source else ["input_data"],
                    outputs=[f"{_sanitize_id(assay_name)}_results/"],
                    depends_on=[prev_id] if prev_id else [],
                )
            )
            prev_id = step_id
        return steps

    # ------------------------------------------------------------------
    # Helpers: nf-core detection
    # ------------------------------------------------------------------

    def _check_nfcore(self, assay_name: str) -> Optional[str]:
        """Return the nf-core pipeline name for this assay type, or None.

        Uses whole-word matching (\\b boundaries) to avoid false positives such
        as "WES" matching "Western blot" or "HiC" matching "stochastic".
        """
        import re
        for data_type, pipeline in NFCORE_MAPPING.items():
            pattern = r"\b" + re.escape(data_type) + r"\b"
            if re.search(pattern, assay_name, re.IGNORECASE):
                return pipeline
        return None

    def _check_nfcore_module(self, tool_name: str) -> Optional[str]:
        """Map well-known tool names to nf-core module identifiers."""
        _tool_to_module: dict[str, str] = {
            "star": "star/align",
            "hisat2": "hisat2/align",
            "bwa": "bwa/mem",
            "bowtie2": "bowtie2/align",
            "samtools": "samtools/sort",
            "picard": "picard/markduplicates",
            "trimgalore": "trimgalore",
            "fastp": "fastp",
            "fastqc": "fastqc",
            "featurecounts": "subread/featurecounts",
            "htseq": "htseq/count",
            "macs2": "macs2/callpeak",
            "macs3": "macs2/callpeak",
            "deseq2": "deseq2/differential",
            "kallisto": "kallisto/quant",
            "salmon": "salmon/quant",
        }
        return _tool_to_module.get(tool_name.lower())

    def _needs_custom_steps(self, config: PipelineConfig) -> bool:
        """True if any step lacks an nf-core module (i.e., needs a custom rule)."""
        return any(step.nf_core_module is None for step in config.steps)

    def _has_nfcore_steps(self, config: PipelineConfig) -> bool:
        """True if the config maps to an nf-core pipeline or any step has an nf-core module."""
        if config.nf_core_pipeline:
            return True
        return any(step.nf_core_module is not None for step in config.steps)

    # ------------------------------------------------------------------
    # Helpers: resource inference & command resolution
    # ------------------------------------------------------------------

    # Rough resource defaults for known tool categories
    _HEAVY_TOOLS = {"star", "hisat2", "bwa", "bowtie2", "cellranger", "bismark"}
    _MEDIUM_TOOLS = {"samtools", "picard", "trimgalore", "fastp", "featurecounts", "htseq", "kallisto", "salmon"}

    def _infer_resources(self, tool_name: str) -> tuple[int, int]:
        """Return (threads, memory_gb) for a tool by name."""
        t = tool_name.lower()
        if t in self._HEAVY_TOOLS:
            return (16, 64)
        if t in self._MEDIUM_TOOLS:
            return (8, 16)
        return (4, 8)

    def _resolve_command(
        self, step: AnalysisStep, sw_obj: Optional[Software]
    ) -> str:
        """Return the best command string for this step.

        Priority:
        1. First command template from the matching Software object.
        2. Templated placeholder based on software name + parameters.
        """
        if sw_obj and sw_obj.commands:
            cmd = sw_obj.commands[0].command_template
            # Substitute any parameters extracted from the step
            for key, val in step.parameters.items():
                cmd = cmd.replace(f"{{{key}}}", val)
            return cmd

        # Fallback: construct a minimal placeholder command
        params_str = " ".join(
            f"--{k} {v}" for k, v in step.parameters.items()
        )
        sw = step.software or "# unknown_tool"
        if params_str:
            return f"{sw} {params_str} {{input}} > {{output}}"
        return f"{sw} {{input}} > {{output}}"

    # ------------------------------------------------------------------
    # Conda environment generation
    # ------------------------------------------------------------------

    def _generate_conda_env(self, software: list[Software]) -> str:
        """Generate a unified conda environment YAML from all software objects.

        Channels: conda-forge, bioconda, defaults.
        Dependencies: bioconda packages first, then pip for PyPI-only tools.
        """
        bioconda_deps: list[str] = []
        condaforge_deps: list[str] = []
        pip_deps: list[str] = []

        seen: set[str] = set()
        for sw in software:
            # Always mark the software name as seen to prevent duplicate fallback entries
            name_key = sw.name.lower()
            if sw.bioconda_package and sw.bioconda_package not in seen:
                pkg = sw.bioconda_package
                if sw.version:
                    pkg = f"{pkg}={sw.version}"
                bioconda_deps.append(f"  - {pkg}")
                seen.add(sw.bioconda_package)
                seen.add(name_key)
            elif sw.pypi_package and sw.pypi_package not in seen:
                pkg = sw.pypi_package
                if sw.version:
                    pkg = f"{pkg}=={sw.version}"
                pip_deps.append(f"    - {pkg}")
                seen.add(sw.pypi_package)
                seen.add(name_key)
            elif sw.bioconda_package in seen or sw.pypi_package in seen:
                # Duplicate package — skip entirely, also mark name as seen
                seen.add(name_key)
            elif name_key not in seen:
                # Generic fallback: try the tool name as a conda-forge package
                condaforge_deps.append(f"  - {name_key}")
                seen.add(name_key)

        lines = [
            "name: researcher_ai_env",
            "channels:",
            "  - conda-forge",
            "  - bioconda",
            "  - defaults",
            "dependencies:",
            "  - python>=3.10",
            "  - pip",
        ]
        lines.extend(bioconda_deps)
        lines.extend(condaforge_deps)
        if pip_deps:
            lines.append("  - pip:")
            lines.extend(pip_deps)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Topological sort of assays
    # ------------------------------------------------------------------

    def _topo_sort_assays(self, method: Method) -> list[str]:
        """Return assay names topologically sorted by AssayGraph dependencies.

        Assays with no dependencies come first; downstream assays follow.
        Falls back to original order if cycles are detected.
        """
        graph = method.assay_graph
        assay_names = [a.name for a in graph.assays]

        # Build adjacency: upstream → downstream
        adj: dict[str, list[str]] = {n: [] for n in assay_names}
        in_degree: dict[str, int] = {n: 0 for n in assay_names}

        for dep in graph.dependencies:
            up = dep.upstream_assay
            down = dep.downstream_assay
            if up in adj and down in adj:
                adj[up].append(down)
                in_degree[down] += 1

        # Kahn's algorithm
        queue = [n for n in assay_names if in_degree[n] == 0]
        result: list[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If topo sort didn't cover all (cycle), fall back to original order
        if len(result) < len(assay_names):
            return assay_names
        return result
