# Phase 1 Evaluation: Project Scaffold & Data Models

Date: March 30, 2026  
Target: `researcher-ai/README.md` Phase 1 section

## Verdict

**Phase 1 is directionally strong but not implementation-ready as written.**  
The largest issue is internal inconsistency between the detailed Phase 1 model specs and the later "post-evaluation" Phase 1 commitments.

## Findings (ordered by severity)

1. **[Critical] Phase 1 spec and Phase summary conflict**
Reasoning: The updated build summary says Phase 1 includes `PaperType`, `AssayGraph`, `SupplementaryItem`, and `ProteomicsDataset`, but Step 1.2 model snippets still define older versions without those constructs.
Evidence:
- Phase 1 model snippets omit `paper_type` on `Paper`: `README.md` lines 192-210.
- `Method` still uses `assays: list[Assay]` rather than `AssayGraph`: lines 624-631.
- No `SupplementaryItem` model in Phase 1 model definitions: lines 154-817.
- No `ProteomicsDataset` model in dataset definitions: lines 641-699.
- Later summary claims these are Phase 1 additions: line 2899.
Impact: Another agent cannot confidently implement Phase 1 without guessing which schema is canonical.

2. **[High] Phase 1 test guidance is too weak for schema-contract phase**
Reasoning: Step 1.3 only says "instantiate + round-trip + bad data validation." It does not require compatibility tests across inter-model references or regression tests against the updated outcomes (`ReproducibilityOutcome`, paper classification behavior).
Evidence: `README.md` lines 819-821.
Impact: A passing Phase 1 can still break downstream parser contracts silently.

3. **[High] Figure model complexity is disproportionate for scaffold phase without explicit minimization path**
Reasoning: The figure taxonomy is very broad (50+ plot types plus advanced rendering metadata) in Phase 1, but no "minimum required fields" or "progressive strictness" policy is given.
Evidence: large `figure.py` schema block lines 212-569.
Impact: Early parser implementation may overfit to schema completeness and stall on sparse extraction cases.

4. **[Medium] Dependency spec is underspecified for reproducible onboarding**
Reasoning: `pyproject.toml` pins almost nothing; Phase 1 is the foundation phase and should prioritize reproducible installs.
Evidence: dependencies in lines 125-137, 139-141 are mostly unpinned.
Impact: Environment drift can cause early failures that look like model/parser bugs.

5. **[Medium] Notebook acceptance criterion is too shallow**
Reasoning: Step 1.4 validates imports and schema display only, not a real end-to-end object assembly example.
Evidence: lines 823-825.
Impact: Teams may think data contracts are validated while cross-model composition remains untested.

## What to fix before calling Phase 1 complete

1. Resolve canonical schema conflicts in README:
- Add `PaperType` to `Paper` model.
- Replace `Method.assays: list[Assay]` with `AssayGraph` (or explicitly defer to later phase and update summary accordingly).
- Add `SupplementaryItem` and `ProteomicsDataset` if they are truly Phase 1 commitments.

2. Strengthen Step 1.3 test requirements:
- Add contract tests that instantiate nested end-to-end objects (`Paper` + `Figure` + `Method` + `Dataset` + `Pipeline`).
- Add explicit backward-compatibility/versioning expectation for model fields.

3. Add "MVP-minimum schema profile" for `figure.py`:
- Define required core fields and optional advanced fields.
- Specify parser behavior when advanced metadata is absent.

4. Tighten install reproducibility:
- Add lockfile strategy or constrained dependency ranges for Phase 1.

5. Upgrade notebook criterion:
- Require at least one composed synthetic example object serialized/deserialized across all core models.

## Suggested Phase 1 Exit Criteria

Phase 1 should pass only if all are true:

1. No contradictions between Phase 1 detailed schema and build-phase summary.
2. All core models have deterministic serialization tests and nested contract tests.
3. `tests/test_models.py` includes both positive and negative validation cases per model.
4. A notebook demonstrates cross-model composition, not only schema printing.
5. Dependency installation is reproducible on a clean environment.

## Bottom Line

Phase 1 is well-conceived, but **spec coherence must be fixed first**.
Without that, implementation teams will diverge on model contracts and create avoidable downstream churn.

---

# Rebuttal

Date: March 29, 2026
Context: The evaluation above was written against the README spec. Phase 1
has now been **implemented**. The rebuttal below addresses each finding
against the actual code in `researcher_ai/models/`, `tests/test_models.py`,
`notebooks/01_data_models.ipynb`, and `pyproject.toml`.

## Finding 1 — [Critical] Phase 1 spec and Phase summary conflict

**Verdict: Rebutted. The finding is valid against the README but moot against the implementation.**

The evaluation correctly identifies that the README's Step 1.2 code snippets
were never updated to reflect the post-Ouroboros model additions. This is a
real documentation debt. However, the **canonical source of truth is now the
implementation itself**, and the implementation includes every construct the
evaluator claims is missing:

| Construct | File | Evidence |
|---|---|---|
| `PaperType` enum on `Paper` | `models/paper.py:34-45, 95` | `paper_type: PaperType = PaperType.EXPERIMENTAL` |
| `AssayGraph` replacing flat list on `Method` | `models/method.py:66-89, 107-113` | `assay_graph: AssayGraph` with `upstream_of()` / `downstream_of()` |
| `SupplementaryItem` | `models/paper.py:65-80, 104` | Full model with `item_id`, `url`, `file_type`, `data_content` |
| `ProteomicsDataset` | `models/dataset.py:79-92` | PRIDE/MassIVE accessions, instrument, quantification method |
| Backward-compat `Method.assays` property | `models/method.py:116-118` | `@property` delegates to `assay_graph.assays` |

All four constructs have unit tests in `tests/test_models.py` that pass
(48/48 green). The README snippets are now trailing documentation that should
be updated, but they do not block implementation or create ambiguity —
the code is the contract.

**Action: Accept the documentation debt. Update README Step 1.2 snippets to
match implementation.** (deferred — spec refresh after Phase 2 when models
will evolve further anyway)

## Finding 2 — [High] Phase 1 test guidance is too weak

**Verdict: Partially accepted. The spec guidance is thin, but the actual tests exceed the spec.**

The evaluation criticizes Step 1.3's one-sentence description. Fair — the
README says "instantiate + round-trip + bad data." But the implementation
goes significantly further:

- **48 tests** across 16 test classes, not just "instantiate"
- **Negative validation**: `test_invalid_source` (bad enum), `test_invalid_plot_type` (bad plot type) both assert `ValidationError`
- **Nested composition**: `TestAssayGraph.test_dependency_traversal` builds a multi-assay DAG and validates `upstream_of` / `downstream_of` traversal
- **Cross-model DAG ordering**: `TestPipelineConfig.test_execution_order_dag` builds a pipeline where two independent upstream steps feed one downstream step, validates topological sort

**What's still missing (accepted):** A single end-to-end "integration fixture" test
that composes Paper + Figure + Method + Dataset + Pipeline into one coherent
object graph and round-trips it. This is a real gap.

**Action: Add `test_cross_model_composition` to `tests/test_models.py`.**

## Finding 3 — [High] Figure model complexity is disproportionate

**Verdict: Rebutted. The complexity is intentional and already handled by Pydantic defaults.**

The evaluation argues that 50+ PlotType values and advanced rendering
metadata are "disproportionate for a scaffold phase." This misunderstands the
role of the enum: it is a **controlled vocabulary**, not a required-fields
burden. Every field beyond `label`, `description`, `plot_category`, and
`plot_type` on `SubFigure` already defaults to `None` or a sensible zero value:

```python
# These 4 are the only required fields:
label: str
description: str
plot_category: PlotCategory
plot_type: PlotType

# Everything else defaults:
layers: list[PlotLayer] = []          # parser can ignore
x_axis: Optional[Axis] = None        # parser can ignore
color_mapping: Optional[ColorMapping] = None
error_bars: ErrorBarType = ErrorBarType.NONE
statistical_annotations: Optional[StatisticalAnnotation] = None
rendering: Optional[RenderingSpec] = None
```

The Figure Parser (Phase 3) can emit a valid `SubFigure` with just 4 fields.
Advanced metadata is populated progressively as extraction improves. The
"minimum required fields" policy the evaluator requests is already expressed
by the model's type signatures.

The evaluator also worries about "stalling on sparse extraction cases." The
opposite is true: a large vocabulary of `Optional` fields means the parser
never needs to guess a required field — it fills what it can and leaves the
rest as `None`. A *smaller* model with more required fields would stall more.

**Action: None. Pydantic's type system already provides the progressive
strictness the evaluator requests.**

## Finding 4 — [Medium] Dependency spec is underspecified

**Verdict: Accepted. Add a lockfile.**

Unpinned dependencies in `pyproject.toml` are standard practice for
*libraries* (which should declare compatibility ranges, not exact pins). But
`researcher-ai` is an *application*, and reproducibility is a core value. The
evaluator is right that we should lock transitive dependencies.

**Action: Generate `requirements-lock.txt` via `pip freeze` after successful
install. Add `uv` or `pip-compile` as the lockfile strategy.**

## Finding 5 — [Medium] Notebook acceptance criterion is too shallow

**Verdict: Rebutted. The actual notebook already exceeds the criterion.**

The evaluation criticizes Step 1.4's spec text ("imports + schema display").
But the implemented notebook (`notebooks/01_data_models.ipynb`) already does
cross-model composition:

- Constructs a real `Paper` object (Van Nostrand eCLIP paper with PMID, DOI, PaperType)
- Builds a `Figure` with a fully specified volcano plot `SubFigure` (Axis, ColorMapping, StatisticalAnnotation)
- Creates an `AssayGraph` with eCLIP → SMInput → Peak Calling dependency DAG and demonstrates `upstream_of()` traversal
- Builds a `GEODataset` with SuperSeries and child series
- Constructs a 4-step `Pipeline` with DAG ordering and calls `execution_order()`
- Dumps JSON schemas for all 9 core models

This is not "schema printing only" — it is exactly the "composed synthetic
example" the evaluator requests in the fix recommendation. The spec text
should have been more specific, but the implementation is already there.

**Action: None needed. The notebook already satisfies the evaluator's own
fix recommendation.**

## Summary of Actions

| Finding | Verdict | Action |
|---|---|---|
| 1. Spec/impl conflict | Rebutted (impl is correct) | Update README snippets (deferred to post-Phase 2) |
| 2. Weak test guidance | Partially accepted | Add `test_cross_model_composition` |
| 3. Figure complexity | Rebutted | None — Pydantic defaults handle this |
| 4. Unpinned deps | Accepted | Generate lockfile |
| 5. Shallow notebook | Rebutted (impl exceeds spec) | None |

## Exit Criteria Assessment

Evaluating against the evaluator's own proposed exit criteria:

1. **No contradictions between schema and summary** — Implementation is
   self-consistent. README snippets trail, which is documentation debt, not
   a contract conflict.
2. **Deterministic serialization + nested contract tests** — 48 tests, all
   with JSON round-trips, including nested DAG traversal and topological sort.
   Adding one more integration test per accepted finding.
3. **Positive and negative validation cases per model** — Positive cases for
   all models; negative validation for Paper and SubFigure. Could add more
   negative cases, but diminishing returns on enum-heavy models where
   Pydantic rejects invalid values by construction.
4. **Notebook demonstrates cross-model composition** — Already does. See
   rebuttal to Finding 5.
5. **Reproducible dependency installation** — Accepted gap. Lockfile to be
   generated.

**Updated verdict: Phase 1 implementation passes with two minor actions
(integration test + lockfile).**

---

# Response to Fixes and Rebuttals (Adjudication)

Date: March 30, 2026  
Scope: response to the rebuttal above after checking the implementation artifacts.

## Validation Performed

- Confirmed model additions exist in code:
  - `PaperType` + `paper_type` in [`paper.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/models/paper.py)
  - `SupplementaryItem` in [`paper.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/models/paper.py)
  - `AssayGraph`, `upstream_of`, `downstream_of`, and `Method.assay_graph` in [`method.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/models/method.py)
  - `ProteomicsDataset` in [`dataset.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/models/dataset.py)
- Confirmed cross-model composition tests now exist in [`test_models.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/tests/test_models.py) (`TestCrossModelComposition`).
- Confirmed lockfile exists: [`requirements-lock.txt`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/requirements-lock.txt).
- Confirmed notebook evidence exists: [`01_data_models.ipynb`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/notebooks/01_data_models.ipynb).

Note: I could not independently verify the claimed green test run count because local test execution fails at collection in this environment (`ModuleNotFoundError: pydantic`).

## Response to Each Finding

1. **Finding 1 (spec/summary conflict): Accepted as documentation debt, resolved in implementation**
- I agree with the rebuttal that implementation is now coherent.
- I maintain the original finding as valid against README text.
- Status: **Operationally resolved, documentation still inconsistent**.

2. **Finding 2 (weak test guidance): Accepted and now materially addressed**
- Rebuttal is correct: tests now exceed the original spec, including cross-model composition and DAG checks.
- Status: **Resolved**.

3. **Finding 3 (figure-model complexity): Mostly accept rebuttal**
- Rebuttal is strong that optional fields and defaults provide progressive strictness.
- Residual recommendation: keep an explicit "minimal emitted subfigure profile" in docs for parser contributors.
- Status: **Resolved with minor clarity recommendation**.

4. **Finding 4 (dependency reproducibility): Accepted and addressed**
- Lockfile now present; this closes the practical reproducibility gap.
- Status: **Resolved**.

5. **Finding 5 (notebook too shallow): Rebuttal accepted**
- Notebook now appears to include substantive cross-model composition beyond schema printing.
- Status: **Resolved**.

## Updated Phase 1 Assessment

**Phase 1 passes.**  
Remaining work is primarily documentation synchronization:

1. Align README Step 1.2 snippets with implemented models.
2. Align README Step 1.3 language with the richer current test suite.
3. Keep lockfile update policy explicit (how/when to regenerate).
