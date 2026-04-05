# Phase 7 Evaluation: Pipeline Builder

**Status: CLOSED**
Date closed: 2026-03-31
Re-evaluation date: 2026-03-31
Targets:
- `README.md` Phase 7 spec
- `researcher_ai/pipeline/builder.py`
- `researcher_ai/pipeline/snakemake_gen.py`
- `researcher_ai/pipeline/nextflow_gen.py`
- `researcher_ai/pipeline/jupyter_gen.py`
- `tests/test_pipeline_builder.py`
- `tests/test_notebooks.py`

---

## Verification Metadata

```
Date    : 2026-03-31
Command : . .venv/bin/activate && python -m pytest tests/test_pipeline_builder.py tests/test_notebooks.py --tb=short -q
Result  : 117 passed in 0.26s
```

---

## Final Verdict

**Phase 7 is complete and closeable.**

All previously open findings have been resolved or addressed with acceptable rebuttals and test-backed evidence.

---

## Findings Resolution

### 1) [High] Snakemake dependency edges dropped when explicit `inputs` existed
**Status:** RESOLVED

- `SnakemakeGenerator._rule()` now emits an additive `input:` block containing:
  - explicit file inputs (`step.inputs`)
  - dependency references (`rules.<dep>.output`) from `step.depends_on`
- This removes the prior mutually-exclusive `if/elif` behavior that could omit DAG edges.

Impact: generated Snakefiles now preserve cross-step/cross-assay dependency enforcement.

### 2) [High] Nextflow custom workflow linearized graph and ignored `depends_on`
**Status:** RESOLVED

- `NextflowGenerator._workflow_block()` now wires by dependency graph semantics:
  - root steps consume `ch_input`
  - single-dependency steps consume upstream process output channel
  - multi-dependency steps use fan-in channel mixing

Impact: custom DSL2 output now respects non-linear DAG execution order.

### 3) [Medium] Phase 7 notebook filename mismatch in README
**Status:** RESOLVED

- README Step 7.6 now references `06_build_pipeline.ipynb`.
- This aligns with repository notebook numbering and existing notebook path.

Impact: removes contributor confusion and documentation drift.

### 4) [Medium] Notebook tests missing for Phase 7 notebook
**Status:** RESOLVED

- Added `TestNotebook06BuildPipeline` in `tests/test_notebooks.py`.
- Coverage includes loading, cell types/count, key imports, and expected topic coverage (Snakemake/Nextflow/Jupyter/conda).

Impact: Tier-4 notebook structure regressions for Phase 7 are now guarded.

### 5) [Low] Snakemake `rule all` overused `expand(...)`
**Status:** RESOLVED

- Generator behavior now validated with a test ensuring `expand()` is only used when output contains `{sample}` and not for literal output files.

Impact: cleaner and more correct terminal target declarations.

---

## Additional Strengths Confirmed

- DAG-focused regression tests were added for both generators (would fail on linearization regressions).
- Pipeline builder, backend generators, and notebook scaffolding are all implemented and test-covered.
- Test runtime remains fast and deterministic.

---

## Deferred / Non-Blocking Items

No new blockers identified for Phase 7 closure.

Future enhancements (outside closure scope) remain valid:
1. richer process I/O typing and channel schemas in Nextflow;
2. execution-time validation against real datasets/containers in Phase 8+.

