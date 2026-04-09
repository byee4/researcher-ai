# researcher-ai

`researcher-ai` converts scientific papers into structured, executable analysis artifacts.

Given a publication (PMID, PMCID, DOI, URL, or PDF), it:
- parses paper metadata and sections,
- extracts figures and panel-level metadata,
- infers computational assays and method dependencies,
- resolves referenced datasets (GEO/SRA),
- extracts software/tooling context,
- generates reproducible workflow artifacts (Snakemake/Nextflow + Jupyter + conda env).

## Documentation

Full docs (including tutorial and API reference) live in `docs/` and are set up for Read the Docs.

- Docs index: `docs/index.rst`
- Architecture: `docs/ARCHITECTURE.md`
- Canonical IR planning roadmap: `docs/WORKFLOW_GRAPH_IR_PLAN.md`
- Full tutorial: `docs/TUTORIAL.md`
- RTD deployment guide: `docs/readthedocs.md`

## Installation

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For docs and development extras:

```bash
python -m pip install -e ".[docs,dev]"
```

## Quick Start

Run end-to-end workflow from a PMID:

```bash
python scripts/run_workflow.py \
  --source 26971820 \
  --source-type pmid \
  --output /tmp/researcher_ai_run.json
```

Run from a local PDF:

```bash
python scripts/run_workflow.py \
  --source /absolute/path/to/paper.pdf \
  --source-type pdf \
  --output /tmp/researcher_ai_pdf_run.json
```

Estimate per-figure parsing latency (p50/p95):

```bash
python scripts/estimate_figure_parse_latency.py \
  --source 40456907 \
  --source-type pmid \
  --max-figures 10 \
  --max-total-seconds 240 \
  --output /tmp/figure_latency_40456907.json \
  --trace-output /tmp/figure_latency_40456907_trace.json
```

Investigate empty structured responses for Figure 2 (PMID 39303722):

```bash
python scripts/investigate_figure2_empty_responses.py \
  --pmid 39303722 \
  --figure-id "Figure 2" \
  --baseline-runs 20 \
  --variant-runs 10
```

## Core Entry Points

- CLI workflow runner: `scripts/run_workflow.py`
- Stateful orchestrator: `researcher_ai/pipeline/orchestrator.py`
- Pipeline builder: `researcher_ai/pipeline/builder.py`
- Parsers: `researcher_ai/parsers/`
- Typed models: `researcher_ai/models/`

## Environment Variables

Most common settings:

- `OPENAI_API_KEY`, `LLM_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (provider keys; no default)
- `RESEARCHER_AI_MODEL` (default: `gpt-5.4`; stable primary routing model)
- `RESEARCHER_AI_LLM_TIMEOUT_SECONDS` (default: `90`; good headroom for long structured calls)
- `RESEARCHER_AI_LITELLM_VERBOSE` (default: `0`; enable only for diagnostics to reduce noisy/sensitive logs)
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS` (default: `0`; disables hard global figure timeout to avoid full figure-drop regressions)
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS` (default: `60`; adaptive per-figure floor informed by measured ~38–52s figure parse times)
- `RESEARCHER_AI_SKIP_FIGURES` (default: `0`; normal runs should parse figures)
- `RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS` (default: `0`; no per-call hard timeout, since decomposition can legitimately take 30–45s)
- `RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS` (default: `3`; circuit-breaker budget per paper)
- `RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS` (default: `1200`; controls long-tail latency for panel decomposition)
- `RESEARCHER_AI_FIGURE_PURPOSE_MAX_TOKENS` (default: `600`; sufficient for purpose/title extraction without excess latency)
- `RESEARCHER_AI_FIGURE_METHODS_DATASETS_MAX_TOKENS` (default: `350`; sufficient for short methods/dataset extraction tasks)
- `RESEARCHER_AI_FIGURE_TRACE_PATH` (default: unset; set only when collecting diagnostics)
- Figure parse warnings now include `subfigure_decomposition_empty_response` when panel decomposition receives persistently empty structured output from the LLM and falls back to best-effort parsing.
- `RESEARCHER_AI_LLM_DEBUG_EMPTY_RESPONSES` (default: `0`; emits per-attempt structured extraction telemetry for empty-response diagnostics)
- `RESEARCHER_AI_LLM_DEBUG_EMPTY_RESPONSES_PATH` (default: unset; optional JSONL sink for telemetry events)
- `RESEARCHER_AI_STRUCTURED_RESPONSE_FORMAT_MODE` (default: `auto`; options: `auto`, `json_schema_only`, `json_object_first`)
- `RESEARCHER_AI_DISABLE_MODEL_FALLBACKS` (default: `0`; when `1`, disables cross-model failover and keeps only the primary model router)
- Methods parser degradation path: per-assay failures still emit `assay_stub` warnings, but now preserve heuristic fallback assay descriptions/steps when source text is available (instead of collapsing all failed assays to `Could not be parsed.`).
- `RESEARCHER_AI_BIOWORKFLOW_MODE` (`off`, `warn`, `on`; default: `warn`)
  - `off`: skip BioWorkflow validation stage
  - `warn`: validate and continue (non-blocking)
  - `on`: strict mode; if ungrounded fields are detected, run ends in `needs_human_review`
- `RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS` (default: `2`; limits call explosion while preserving recall)
- `RESEARCHER_AI_BIOC_ENABLED` (default: `1`; enables richer grounding context)
- `RESEARCHER_AI_FIGURE_CALIBRATION` (default: `on`; keeps calibration active in standard runs)
- `RESEARCHER_AI_HPC_PROFILE` (default: `tscc`; cluster-first runtime profile)

Model/provider routing defaults are defined in:
`researcher_ai/config/models.yaml`

## BioWorkflow Strict-Mode Fallback

When `RESEARCHER_AI_BIOWORKFLOW_MODE=on`, validation can block unsafe pipeline
generation if ungrounded critical fields remain. In that case, the orchestrator
returns a terminal stage:

- `stage = "needs_human_review"`
- `human_review_required = true`
- `human_review_summary` with `ungrounded_count`, `ungrounded_fields`, and
  a recommended next action.

## Testing

Run all tests:

```bash
pytest
```

Run targeted suites:

```bash
pytest tests/test_integration.py
pytest tests/test_pipeline_builder.py
pytest tests/test_figure_parser.py
```

## Build Docs Locally

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Open:
`docs/_build/html/index.html`

## Project Status

This repository includes archived evaluation artifacts under `docs/previous/` and a consolidated changelog in `CHANGELOG.md`.
