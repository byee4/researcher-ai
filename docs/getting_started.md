# Getting Started

This page gets you from clone to first successful workflow run.

## 1. Clone and install

```bash
git clone <your-repo-url>
cd researcher-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install docs/dev extras when needed:

```bash
python -m pip install -e ".[docs,dev]"
```

## 2. Configure credentials

Set at least one LLM provider key:

```bash
export OPENAI_API_KEY="..."
# or
export LLM_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."
# or
export GEMINI_API_KEY="..."
```

Optional:

```bash
export RESEARCHER_AI_MODEL="gpt-5.4"
export NCBI_API_KEY="..."
```

## 3. Run end-to-end from PMID

```bash
python scripts/run_workflow.py \
  --source 26971820 \
  --source-type pmid \
  --output /tmp/researcher_ai_run.json
```

You should see staged progress logs and an output JSON with:
- `paper`
- `figures`
- `method`
- `datasets`
- `software`
- `workflow_graph`
- `workflow_graph_validation_issues`
- `pipeline`

## 4. Build docs

```bash
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## 5. Run tests

```bash
pytest
```
