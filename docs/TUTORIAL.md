# Full Tutorial

This tutorial walks through a full publication-to-pipeline run and explains each artifact.

## Tutorial goals

By the end, you will:
- run the orchestrator from a PMID and a PDF,
- inspect parsed outputs (paper, figures, methods, datasets, software),
- inspect generated pipeline artifacts,
- understand how to adjust model/runtime settings,
- know how to debug common failures.

## Prerequisites

- Python 3.10+
- A virtual environment with project dependencies
- At least one LLM API key configured

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Step 1: Run workflow using PMID

```bash
python scripts/run_workflow.py \
  --source 26971820 \
  --source-type pmid \
  --output /tmp/tutorial_pmid.json
```

Progress lines look like:

```text
PROGRESS|5|Initializing state-graph orchestrator
...
PROGRESS|100|Completed
```

## Step 2: Inspect output JSON

```bash
jq 'keys' /tmp/tutorial_pmid.json
```

Expected top-level keys:
- `paper`
- `figures`
- `method`
- `datasets`
- `software`
- `pipeline`
- `dataset_parse_errors`

Quick checks:

```bash
jq '.paper.title' /tmp/tutorial_pmid.json
jq '.figures | length' /tmp/tutorial_pmid.json
jq '.method.assay_graph.assays | length' /tmp/tutorial_pmid.json
jq '.datasets | length' /tmp/tutorial_pmid.json
jq '.software | length' /tmp/tutorial_pmid.json
jq '.pipeline.config.steps | length' /tmp/tutorial_pmid.json
```

## Step 3: Run workflow using a local PDF

```bash
python scripts/run_workflow.py \
  --source /absolute/path/to/paper.pdf \
  --source-type pdf \
  --output /tmp/tutorial_pdf.json
```

Use this mode when papers are not easily resolvable by PMID/PMCID.

## Step 4: Understand each stage

### `PaperParser`

- Detects source type (PMID, PMCID, DOI, URL, PDF).
- For PMCID, parses JATS full text.
- For PMID, combines PubMed metadata with PMCID/BioC enrichment when available.
- For PDF/URL, performs text extraction + structured LLM parsing.

### `FigureParser`

- Uses figure IDs, captions, in-text references, and optional image snippets.
- Produces panel-level structure (`SubFigure`) with confidence metrics.
- Optional calibration layer (`FigureCalibrationEngine`) adjusts figure typing.

### `MethodsParser`

- Extracts assay graph, steps, dependencies, and availability statements.
- Defaults to `computational_only=True` in orchestrator flow.

### Dataset parsers (`GEOParser`, `SRAParser`)

- Detect accession IDs from paper/method/figure text.
- Fetch metadata and normalize into `Dataset` models.

### `SoftwareParser`

- Maps method steps to software tooling, environments, and commands.

### `PipelineBuilder`

- Topologically orders assay dependencies.
- Generates `PipelineConfig` and pipeline artifacts:
  - Snakemake content,
  - optional Nextflow content,
  - Jupyter figure reproduction notebook,
  - conda environment YAML.

## Step 5: Persist generated pipeline files

`run_workflow.py` writes a single JSON blob. Extract artifacts as files:

```bash
mkdir -p /tmp/tutorial_pipeline
jq -r '.pipeline.snakefile_content // empty' /tmp/tutorial_pmid.json > /tmp/tutorial_pipeline/Snakefile
jq -r '.pipeline.nextflow_content // empty' /tmp/tutorial_pmid.json > /tmp/tutorial_pipeline/main.nf
jq -r '.pipeline.jupyter_content // empty' /tmp/tutorial_pmid.json > /tmp/tutorial_pipeline/figure_reproduction.ipynb
jq -r '.pipeline.conda_env_yaml // empty' /tmp/tutorial_pmid.json > /tmp/tutorial_pipeline/environment.yml
```

## Step 6: Optional direct Python usage

```python
from researcher_ai.models.paper import PaperSource
from researcher_ai.pipeline.orchestrator import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator(max_build_attempts=2)
state = orchestrator.run("26971820", PaperSource.PMID)
print(state["paper"].title)
print(len(state.get("figures", [])))
print(len(state.get("datasets", [])))
```

## Step 7: Run and interpret tests

```bash
pytest
```

Useful suites:

```bash
pytest tests/test_integration.py
pytest tests/test_pipeline_builder.py
pytest tests/test_figure_parser.py
pytest tests/test_phase4_state_graph.py
```

## Step 8: Common troubleshooting

## LLM auth errors

- Confirm one provider key is set.
- Confirm `RESEARCHER_AI_MODEL` maps to a configured provider/model.

## Dataset parser failures

- Check `dataset_parse_errors` in workflow output.
- Set `NCBI_API_KEY` to reduce throttling risk for NCBI calls.

## Snakemake validation issues

`PipelineBuilder` attempts lint/dry-run checks and can retry with repairs.
Inspect:
- `pipeline.validation_report`

## Missing figures from PMID path

Some papers expose limited figure assets via metadata APIs. Try PDF source mode for richer extraction.

## Step 9: Customize behavior via environment

```bash
export RESEARCHER_AI_MODEL="gpt-5.4"
export RESEARCHER_AI_FIGURE_CALIBRATION="on"
export RESEARCHER_AI_FIGURE_CALIBRATION_CONFIDENCE_THRESHOLD="0.75"
export RESEARCHER_AI_BIOC_ENABLED="1"
export RESEARCHER_AI_HPC_PROFILE="tscc"
```

## Step 10: Next steps

- Move to the architecture guide: `docs/ARCHITECTURE.md`
- Read API details: `docs/api.rst`
- Configure docs hosting: `docs/readthedocs.md`
