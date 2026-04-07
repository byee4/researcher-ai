# researcher-ai Architecture (Current)

This document consolidates prior architecture/evaluation/calibration/improvement markdown and reflects the current implementation in `researcher_ai/*`.

## Scope and Inputs

Merged source sets:
- Architecture/evaluation/calibration plans: `EVALUATION_ARCHITECTURE.md`, `EVALUATION_PHASE_1.md` ... `EVALUATION_PHASE_8.md`, `BIOC_INTEGRATION_PLAN.md`, `FIGURE_PARSER_CALIBRATION.md`, `FIGURE_PARSER_IMPROVEMENT_PLAN.md`, `ROADMAP.md`
- Code validation anchors: `researcher_ai/parsers/*`, `researcher_ai/pipeline/*`, `researcher_ai/models/*`, `scripts/run_workflow.py`, `tests/*`

## System Overview

`researcher-ai` converts a publication into structured objects and then into executable workflow artifacts.

Main flow:
1. `PaperParser` builds a `Paper` object from PMID/PMCID/DOI/PDF/URL.
2. `FigureParser` converts paper figure IDs/captions/context into structured `Figure` + `SubFigure` outputs.
3. `MethodsParser` extracts assays/steps/dependencies into a `Method` with `AssayGraph`.
4. GEO/SRA parsers expand detected accessions into structured dataset metadata.
5. `SoftwareParser` maps method steps to software tools, environments, commands, licensing metadata.
6. `PipelineBuilder` builds ordered pipeline steps and generators emit Snakemake/Nextflow/Jupyter/conda outputs.

## Core Modules

### Data Models (`researcher_ai/models`)
- Pydantic models define contracts for paper, figure, method, dataset, software, and pipeline components.
- `Method` uses `AssayGraph` + explicit dependency edges.
- Pipeline outputs are carried in `Pipeline`/`PipelineConfig` with backend-specific generated content.

### Paper Parsing (`researcher_ai/parsers/paper_parser.py`)
- Source detection supports PMCID, PMID, DOI, URL, and PDF.
- PMCID path: JATS full text parse first, then BioC context enrichment.
- PMID path: PubMed metadata + PMCID resolution/fallback behavior.
- File detection explicitly rejects non-PDF files.
- Output can include BioC-derived section/figure context attached to `Paper`.

### Figure Parsing + Calibration (`researcher_ai/parsers/figure_parser.py`, `figure_calibration.py`)
- Uses caption/context extraction + LLM structured outputs for panel-level typing.
- Maintains explicit confidence fields (`classification_confidence`, `composite_confidence`, `confidence_scores`).
- Adds deterministic cue reconciliation and calibration pass (`FigureCalibrationEngine`).
- Includes utilities for figure URL retrieval from PMID/PMCID.

### Methods Parsing (`researcher_ai/parsers/methods_parser.py`)
- Finds methods text from section-title heuristics and availability sections.
- Identifies assays and classifies each as experimental/computational/mixed.
- Default behavior is `computational_only=True`; non-computational assays are filtered and logged as parse warnings.
- Builds dependency graph and extracts code/data availability text.

### Dataset Parsing (`researcher_ai/parsers/data/*.py`)
- `GEOParser`: validates GSE/GSM/GPL, fetches E-utilities metadata, handles SuperSeries and optional bounded recursion.
- `SRAParser`: validates SRP/SRX/SRR (+ ENA/DDBJ forms), delegates to `pysradb` and returns project/experiment/run-scoped metadata.

### Software Parsing (`researcher_ai/parsers/software_parser.py`)
- Pulls software mentions from method steps or raw text.
- Uses known-tool registry first, then structured LLM enrichment.
- Classifies license/open-source alternatives and builds command/environment records.

### Pipeline Assembly (`researcher_ai/pipeline/builder.py`)
- Filters method graph to computational assays before pipeline generation.
- Topologically orders assays by dependency edges.
- Creates `PipelineStep`s with intra-assay and inter-assay dependencies.
- Emits Snakemake and/or Nextflow content plus Jupyter reproduction notebook and conda env YAML.

## Operational Entry Points

- Library usage in notebooks (`notebooks/*`).
- Workflow runner script: `scripts/run_workflow.py`.
  - Emits staged progress.
  - Parses paper → figures → method → datasets → software → pipeline.
  - Serializes one JSON result payload.

## Testing and Quality Controls

- Test coverage spans parser modules and pipeline generation (`tests/test_*`).
- Figure calibration has dedicated fixtures and report script (`tests/fixtures/figure_calibration`, `scripts/figure_calibration_report.py`).
- Integration tests and notebook tests are present (`tests/test_integration.py`, `tests/test_notebooks.py`).

## Accuracy Notes (Compared to Older Dev Docs)

- Pipeline builder now enforces computational assay filtering itself (`_computational_only_method`), not only upstream parser filtering.
- Paper source detection behavior now raises on existing non-PDF files rather than silently treating all files as PDFs.
- Evaluation-phase pass counts in older docs are historical snapshots and not treated as current-state metrics here.
