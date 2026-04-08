# Architecture

This document describes the current end-to-end architecture implemented in `researcher_ai/`.

## High-level flow

`researcher-ai` transforms a publication into executable workflow artifacts.

1. Parse source publication into `Paper`.
2. Parse figures into structured `Figure` and `SubFigure` models.
3. Parse methods into computational assay graph (`Method`).
4. Detect and resolve data accessions into `Dataset` records.
5. Map method steps to `Software` and command/environment metadata.
6. Build workflow config and generated artifacts (`Pipeline`).

## Source ingestion and paper parsing

`PaperParser` supports:
- PMCID
- PMID
- DOI
- URL
- PDF

Parsing strategy prioritizes richer sources first:
- PMCID/JATS full text when available
- PMID metadata and PMCID resolution
- DOI to PMID resolution
- PDF text extraction + LLM parsing
- URL text extraction + LLM parsing

BioC enrichment can augment methods/results/figure context.

## Figure parsing and calibration

`FigureParser` combines:
- figure IDs from paper context,
- caption extraction,
- in-text mention context,
- optional panel image extraction,
- structured multimodal LLM extraction.

Outputs include panel-level confidence metrics. `FigureCalibrationEngine` can apply deterministic calibration rules from `researcher_ai/calibration/figure_registry.yaml`.

## Methods parsing

`MethodsParser` extracts:
- assays and categories,
- per-assay computational steps,
- dependency edges,
- data/code availability statements,
- parse warnings.

The orchestrator path defaults to computational assay extraction (`computational_only=True`).

## Dataset parsing

`WorkflowOrchestrator` collects accessions from:
- raw paper text,
- section text,
- method availability statements,
- figure captions.

It then routes accessions to:
- `GEOParser` for GEO families (`GSE`, `GSM`, `GPL`, `GDS`)
- `SRAParser` for SRA/ENA-style IDs (`SRP`, `SRX`, `SRR`, `PRJNA`, `PRJEB`, etc.)

Errors are captured in `dataset_parse_errors` instead of hard-failing the run.

## Software parsing

`SoftwareParser` infers tooling from the method graph and method text using:
- known software registries,
- structured LLM enrichment,
- license/open-source alternative classification,
- command extraction.

## Orchestration model

`WorkflowOrchestrator` runs a stage graph with:
- LangGraph state machine when available,
- deterministic sequential fallback when unavailable.

State keys include parsed artifacts plus `progress`, `stage`, and build retry counters.

BioWorkflow rollout control is managed by `RESEARCHER_AI_BIOWORKFLOW_MODE`:
- `off`: skips method-validation stage.
- `warn` (default): validates and continues with warnings.
- `on`: strict mode; ungrounded validation findings block pipeline build and
  return terminal `stage="needs_human_review"` with `human_review_summary`.

To prevent iterative-query runaway, methods parsing also enforces a hard
retrieval refinement cap via `RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS`.

Per-paper table vision fallback is tracked for benchmark observability as:
- `vision_fallback_count`
- `vision_fallback_latency_seconds`

## Pipeline generation

`PipelineBuilder`:
- filters to computational assays,
- topologically orders steps by dependency,
- generates Snakemake and/or Nextflow code,
- generates Jupyter notebook content,
- generates conda environment YAML,
- validates Snakemake (lint/dry-run) with optional repair loops.

## Data contracts

Pydantic models under `researcher_ai/models/` define stable contracts for:
- paper,
- figure,
- method,
- dataset,
- software,
- pipeline.

These models are the primary interface between parsers and generators.
