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

## Core Entry Points

- CLI workflow runner: `scripts/run_workflow.py`
- Stateful orchestrator: `researcher_ai/pipeline/orchestrator.py`
- Pipeline builder: `researcher_ai/pipeline/builder.py`
- Parsers: `researcher_ai/parsers/`
- Typed models: `researcher_ai/models/`

## Environment Variables

Most common settings:

- `OPENAI_API_KEY`, `LLM_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (provider keys)
- `RESEARCHER_AI_MODEL` (default model router, default: `gpt-5.4`)
- `RESEARCHER_AI_LLM_TIMEOUT_SECONDS` (LLM request timeout)
- `RESEARCHER_AI_BIOC_ENABLED` (enable/disable BioC enrichment)
- `RESEARCHER_AI_FIGURE_CALIBRATION` (on/off)
- `RESEARCHER_AI_HPC_PROFILE` (`tscc` or `local`)

Model/provider routing defaults are defined in:
`researcher_ai/config/models.yaml`

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
