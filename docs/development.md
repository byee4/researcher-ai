# Development Guide

## Repository layout

- `researcher_ai/models/`: typed Pydantic contracts
- `researcher_ai/parsers/`: paper/figure/method/software/dataset parsing
- `researcher_ai/pipeline/`: orchestrator + workflow generators
- `researcher_ai/utils/`: LLM, PDF, PubMed, RAG utilities
- `scripts/`: runnable workflow and benchmark scripts
- `tests/`: unit + integration + snapshot tests

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Lint and type checks

```bash
ruff check .
mypy researcher_ai
```

## Test strategy

Run all:

```bash
pytest
```

Profile per-test memory with a hard 4 GiB guard:

```bash
.venv/bin/python scripts/profile_test_memory.py \
  --threshold-gib 4 \
  --per-test-timeout-seconds 240 \
  --output-prefix artifacts/test_memory_profile
```

This command writes:
- `artifacts/test_memory_profile.csv`: one row per test with peak RSS.
- `artifacts/test_memory_profile_summary.json`: aggregate counts + top memory tests.
- `artifacts/test_memory_profile_run.log`: full execution log.

Plain-English interpretation:
- If a test crosses 4 GiB RSS, it is terminated immediately and recorded as `killed`.
- If no test crosses 4 GiB, the report still highlights high-memory hotspots for optimization.

Useful focused test runs:

```bash
pytest tests/test_paper_parser.py
pytest tests/test_methods_parser.py
pytest tests/test_pipeline_builder.py
pytest tests/test_phase4_state_graph.py
```

Live/network-dependent tests are marked with `live`.

## Adding or changing parser behavior

1. Update model contracts first if output shape changes.
2. Add/adjust parser logic.
3. Add tests (or snapshot updates) for each changed behavior.
4. Run focused parser tests and end-to-end integration test.

## Documentation maintenance

Docs are built with Sphinx + MyST Markdown.

```bash
python -m sphinx -b html docs docs/_build/html
```

When adding public functions/classes, ensure docstrings are present so API docs remain complete.

## Architecture planning authority

- Current implemented architecture: `docs/ARCHITECTURE.md`
- Canonical IR-first planning roadmap: `docs/WORKFLOW_GRAPH_IR_PLAN.md`
- Deprecated architecture planning artifacts:
  `docs/previous/development/architecture/V2_ARCHITECTURE_PLAN.md` and
  `docs/previous/development/architecture/researcher-ai_architecture_review.md`
