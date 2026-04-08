"""Nextflow pipeline generator.

Supports two modes. In nf-core mode, it creates a params YAML and samplesheet
CSV for launching existing nf-core workflows. In custom mode, it generates a
DSL2 workflow with one process per ``PipelineStep`` plus a workflow block that
wires dependencies.
"""

from __future__ import annotations

import textwrap
from typing import Optional

from researcher_ai.models.pipeline import PipelineConfig, PipelineStep


# Samplesheet column layouts for common nf-core pipelines
_NFCORE_SAMPLESHEET_HEADERS: dict[str, list[str]] = {
    "rnaseq": ["sample", "fastq_1", "fastq_2", "strandedness"],
    "atacseq": ["sample", "fastq_1", "fastq_2"],
    "chipseq": ["sample", "fastq_1", "fastq_2", "antibody", "control"],
    "sarek": ["patient", "sample", "lane", "fastq_1", "fastq_2"],
    "methylseq": ["sample", "fastq_1", "fastq_2"],
    "hic": ["sample", "fastq_1", "fastq_2"],
    "ampliseq": ["sampleID", "forwardReads", "reverseReads", "run"],
    "mag": ["sample", "group", "short_reads_1", "short_reads_2"],
}

# Default placeholder values per column type
_PLACEHOLDER_VALUES: dict[str, str] = {
    "sample": "SAMPLE_ID",
    "sampleID": "SAMPLE_ID",
    "patient": "PATIENT_ID",
    "fastq_1": "/path/to/sample_R1.fastq.gz",
    "fastq_2": "/path/to/sample_R2.fastq.gz",
    "forwardReads": "/path/to/sample_R1.fastq.gz",
    "reverseReads": "/path/to/sample_R2.fastq.gz",
    "short_reads_1": "/path/to/sample_R1.fastq.gz",
    "short_reads_2": "/path/to/sample_R2.fastq.gz",
    "strandedness": "auto",
    "antibody": "H3K27ac",
    "control": "INPUT",
    "lane": "L001",
    "group": "0",
    "run": "1",
}


class NextflowGenerator:
    """Generate Nextflow workflow content from a ``PipelineConfig``."""

    def generate(self, config: PipelineConfig) -> str:
        """Generate Nextflow content as a string.

        If nf_core_pipeline is set, generates params YAML (with embedded
        samplesheet as a comment). Otherwise, generates a custom DSL2 workflow.
        """
        if config.nf_core_pipeline:
            return self._generate_nfcore_config(config)
        return self._generate_custom_workflow(config)

    # ------------------------------------------------------------------
    # nf-core mode
    # ------------------------------------------------------------------

    def _generate_nfcore_config(self, config: PipelineConfig) -> str:
        """Generate nf-core pipeline params YAML.

        Returns a YAML string with:
        - input: path to samplesheet
        - outdir: output directory
        - genome / reference parameters extracted from PipelineStep parameters
        - Any tool-specific parameters captured in PipelineStep.parameters
        """
        pipeline = config.nf_core_pipeline
        params: dict[str, str] = {
            "input": "samplesheet.csv",
            "outdir": "results/",
        }

        # Collect genome/reference hints from step parameters
        for step in config.steps:
            for key, val in step.parameters.items():
                k_lower = key.lower()
                if any(t in k_lower for t in ("genome", "reference", "fasta", "gtf", "index")):
                    params[key] = val

        # Common pipeline-specific defaults
        if pipeline == "rnaseq":
            params.setdefault("genome", "GRCh38")
            params.setdefault("aligner", "star_salmon")
        elif pipeline in ("sarek", "variantcalling"):
            params.setdefault("genome", "GRCh38")
            params.setdefault("tools", "haplotypecaller")
        elif pipeline == "atacseq":
            params.setdefault("genome", "GRCh38")

        # Build YAML lines
        lines = [
            f"# nf-core/{pipeline} parameter file",
            f"# Launch with: nextflow run nf-core/{pipeline} -params-file params.yaml",
            f"# nf-core version: {config.nf_core_version or 'latest'}",
            "#",
            f"# Pipeline: {config.name}",
            f"# Description: {config.description}",
            "#",
        ]
        for k, v in params.items():
            lines.append(f"{k}: '{v}'")

        lines += [
            "",
            "# ---------------------------------------------------------------------------",
            "# Samplesheet (save as samplesheet.csv)",
            "# ---------------------------------------------------------------------------",
            "#",
        ]
        samplesheet = self._generate_samplesheet(config)
        for row in samplesheet.splitlines():
            lines.append(f"# {row}")

        return "\n".join(lines) + "\n"

    def _generate_samplesheet(self, config: PipelineConfig) -> str:
        """Generate an nf-core samplesheet CSV from dataset metadata.

        Format depends on the nf-core pipeline. Returns a CSV string.
        """
        pipeline = config.nf_core_pipeline or "rnaseq"
        headers = _NFCORE_SAMPLESHEET_HEADERS.get(pipeline, ["sample", "fastq_1", "fastq_2"])

        rows = [",".join(headers)]

        # Generate one placeholder row per dataset
        if config.datasets:
            for accession in config.datasets:
                row_vals: list[str] = []
                for col in headers:
                    if col in ("sample", "sampleID", "patient"):
                        row_vals.append(accession)
                    else:
                        row_vals.append(_PLACEHOLDER_VALUES.get(col, f"<{col}>"))
                rows.append(",".join(row_vals))
        else:
            # Fallback: one example row
            row_vals = [_PLACEHOLDER_VALUES.get(col, f"<{col}>") for col in headers]
            rows.append(",".join(row_vals))

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Custom DSL2 mode
    # ------------------------------------------------------------------

    def _generate_custom_workflow(self, config: PipelineConfig) -> str:
        """Generate a custom Nextflow DSL2 .nf workflow.

        Produces:
        - nextflow.config include comment
        - One PROCESS per PipelineStep
        - A WORKFLOW block that wires channel outputs to downstream inputs
        """
        ordered_ids = config.execution_order()
        step_map = {s.step_id: s for s in config.steps}
        ordered_steps = [step_map[sid] for sid in ordered_ids if sid in step_map]

        sections: list[str] = []
        sections.append(self._nf_header(config))
        sections.append(self._params_block(config))
        for step in ordered_steps:
            sections.append(self._process_block(step))
        sections.append(self._workflow_block(ordered_steps))

        return "\n".join(sections) + "\n"

    def _nf_header(self, config: PipelineConfig) -> str:
        """Return the standard Nextflow script preamble for a pipeline config."""
        return textwrap.dedent(f"""\
            #!/usr/bin/env nextflow
            /*
             * Nextflow DSL2 workflow: {config.name}
             *
             * {config.description}
             *
             * Generated by researcher-ai PipelineBuilder.
             * Run: nextflow run main.nf
             */

            nextflow.enable.dsl = 2
        """)

    def _params_block(self, config: PipelineConfig) -> str:
        """Generate params block with dataset and output defaults."""
        lines = ["params {"]
        lines.append('    input       = "samplesheet.csv"')
        lines.append('    outdir      = "results/"')
        lines.append('    publish_dir_mode = "copy"')
        if config.datasets:
            lines.append(f"    // Datasets: {', '.join(config.datasets)}")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _process_block(self, step: PipelineStep) -> str:
        """Generate a single Nextflow DSL2 PROCESS for a PipelineStep."""
        process_name = step.step_id.upper()
        cmd = step.command

        # Input channel val
        input_val = (
            f'path {step.step_id}_input'
            if step.inputs
            else "val sample_id"
        )
        # Output channel val
        output_val = (
            f'path "{step.outputs[0]}"'
            if step.outputs
            else f'path "{step.step_id}_output/"'
        )

        lines = [
            f"process {process_name} {{",
        ]
        if step.container:
            lines.append(f'    container "docker://{step.container}"')
        elif step.conda_env:
            lines.append(f'    conda "{step.conda_env}"')

        lines += [
            f"    cpus    {step.threads}",
            f"    memory  '{step.memory_gb} GB'",
            f"    time    '2h'",
            "",
            f"    publishDir params.outdir, mode: params.publish_dir_mode",
            "",
            "    input:",
            f"        {input_val}",
            "",
            "    output:",
            f"        {output_val}, emit: out",
            "",
            "    script:",
            '    """',
            f"    {cmd}",
            '    """',
            "}",
            "",
        ]
        return "\n".join(lines)

    def _workflow_block(self, ordered_steps: list[PipelineStep]) -> str:
        """Generate a WORKFLOW block that honours the depends_on DAG.

        Wiring strategy:
        - Steps with no deps consume ch_input.
        - Steps with one dep consume <PROC>.out.out of that dep.
        - Steps with multiple deps mix all dep output channels so Nextflow can
          fan-in; the process receives a merged channel.
        """
        if not ordered_steps:
            return "workflow {}"

        lines = ["workflow {"]
        lines.append("")
        lines.append("    // Input channel")
        lines.append("    ch_input = Channel.fromPath(params.input)")
        lines.append("")

        # Build a map from step_id → process name so we can reference outputs
        proc_name: dict[str, str] = {s.step_id: s.step_id.upper() for s in ordered_steps}

        for step in ordered_steps:
            proc = proc_name[step.step_id]

            if not step.depends_on:
                # Root step — consumes the raw input channel
                channel_arg = "ch_input"
            elif len(step.depends_on) == 1:
                dep_proc = proc_name.get(step.depends_on[0], step.depends_on[0].upper())
                channel_arg = f"{dep_proc}.out.out"
            else:
                # Fan-in: mix all upstream outputs
                dep_channels = [
                    f"{proc_name.get(dep, dep.upper())}.out.out"
                    for dep in step.depends_on
                ]
                mix_expr = ".mix(".join(dep_channels[:-1]) + f".mix({dep_channels[-1]})" + ")" * (len(dep_channels) - 2)
                # Simpler flat form — readable for 2-3 deps
                mix_expr = " .mix( ".join(dep_channels)
                channel_arg = f"({mix_expr})"
                lines.append(f"    // Fan-in from: {', '.join(step.depends_on)}")

            lines.append(f"    {proc}({channel_arg})")

        lines.append("}")
        return "\n".join(lines)
