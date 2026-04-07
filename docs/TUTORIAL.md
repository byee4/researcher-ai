# researcher-ai Tutorial (Current)

This document consolidates README/tutorial/quickstart-style guidance for `researcher-ai` and updates steps to match the current code layout.

## Scope and Inputs

Merged source sets:
- `README.md`
- `docs/how_to_read.md`
- `examples/example_paper/README.md`
- legacy root-level docs mapped here where relevant: `ROADMAP.md`

## 1) Environment Setup

From workspace root:

```bash
cd /Users/brianyee/Documents/work/01_active/researcher-ai
python -m pip install -r requirements.txt
python -m pip install -e ./researcher-ai
```

Optional docs build extras:

```bash
cd /Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai
python -m pip install -e ".[docs]"
```

## 2) Core Python Workflow

Typical import path is the installed package `researcher_ai` from inside `researcher-ai` project directory.

```python
from researcher_ai.parsers.paper_parser import PaperParser
from researcher_ai.parsers.figure_parser import FigureParser
from researcher_ai.parsers.methods_parser import MethodsParser
from researcher_ai.parsers.software_parser import SoftwareParser
from researcher_ai.parsers.data.geo_parser import GEOParser
from researcher_ai.parsers.data.sra_parser import SRAParser
from researcher_ai.pipeline.builder import PipelineBuilder
```

Pipeline sequence:
1. Parse a paper (`PaperParser.parse`).
2. Parse figures (`FigureParser.parse_all_figures`).
3. Parse methods (`MethodsParser.parse(..., computational_only=True)` by default).
4. Parse referenced datasets (GEO/SRA).
5. Parse software from method graph.
6. Build executable pipeline artifacts.

## 3) Scripted End-to-End Run

```bash
cd /Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai
python scripts/run_workflow.py \
  --source 26971820 \
  --source-type pmid \
  --output /tmp/researcher_ai_run.json
```

Accepted source types for this script are currently `pmid` and `pdf`.

## 4) Tests

```bash
cd /Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai
python -m pytest
```

Targeted examples:

```bash
python -m pytest tests/test_integration.py
python -m pytest tests/test_pipeline_builder.py
python -m pytest tests/test_figure_parser.py
```

## 5) Build API Docs (Sphinx)

```bash
cd /Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai
sphinx-build -b html docs docs/_build/html
```

Output index:
`/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/docs/_build/html/index.html`

## 6) Example Paper Outputs

`examples/example_paper/` contains generated artifacts (e.g., Snakefile, environment YAML, notebook, pipeline JSON) from a representative eCLIP publication run.

## Accuracy Notes (Compared to Older Tutorial Docs)

- Current package install target is `./researcher-ai`.
- End-to-end orchestration script path is `researcher-ai/scripts/run_workflow.py`.
