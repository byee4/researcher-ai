# BioWorkflow Integration Plan — Review & Improvements

## Executive Summary

After reading the BioWorkflow paper and auditing the researcher-ai codebase against the proposed plan, the plan has **several critical misalignments** with the current code. Some proposed work is redundant with existing capabilities, while the most impactful BioWorkflow innovations are underspecified. This document preserves the original 4-phase structure but corrects each phase with code-grounded recommendations.

---

## Overall Assessment

**What the plan gets right:**

- The phased structure (index → decompose → validate → test) mirrors BioWorkflow's pipeline correctly.
- Identifying the "hidden parameter" problem as the core gap is accurate — `methods_parser.py` currently truncates methods text to 3000 chars at the per-assay level (line ~771) and 1800 chars for RAG inference (line ~992), which means parameters buried in long supplementary sections are genuinely lost.
- The emphasis on hallucination suppression is warranted — the current `_infer_missing_computational_parameters` method (line 933) calls the LLM to "infer" parameters from protocol docs but has no verification that inferred parameters actually appear in the source paper.

**What the plan gets wrong or omits:**

1. **`paper_parser.py` already extracts tables via marker-pdf** (spatial Markdown with table/layout preservation). The plan says "update your parsing logic to systematically extract tables and figure captions alongside the prose" — this is partially done. The gap is that extracted tables are flattened into section text and lose their structured identity.

2. **`figure_parser.py` (79.8 KB) is a sophisticated multimodal parser** that already extracts captions, panels, and in-text mentions. The plan ignores this entirely and proposes building figure caption extraction from scratch.

3. **`rag.py` only indexes local protocol documents**, not paper-derived content. The plan says to "embed both the original chunks and their LLM summaries into a vector space" but doesn't specify that this requires a fundamentally new index — a per-paper ephemeral index — separate from the existing `ProtocolRAGStore`.

4. **The test plan is too abstract.** It names zero existing test files, fixtures, or snapshot data, even though the repo has 25 test modules, YAML fixtures in `tests/snapshots/methods/`, and a parametrized eval framework in `test_methods_parser_eval.py`.

---

## Phase 1: Multimodal Indexing & Chunk-Level Summaries

### What Exists Already

- `utils/pdf.py` uses marker-pdf for spatial Markdown extraction that preserves table structure.
- `figure_parser.py` extracts captions, panel metadata, in-text mentions via multimodal LLM.
- `paper_parser.py` parses JATS XML sections, BioC passages, and supplementary items.
- `rag.py` provides `ProtocolRAGStore` with Chroma + lexical fallback (chunk size 900, overlap 120).

### What's Actually Missing (The Real Gap)

The existing pipeline loses modality provenance. When marker-pdf extracts a table, it becomes inline Markdown within a section's `.text` field. There is no way to later ask "which chunks came from tables vs. prose vs. figure captions?" This is exactly what BioWorkflow solves with its unified multimodal index.

### Improved Implementation

**1a. Add a `ChunkType` enum and `AnnotatedChunk` model to `models/paper.py`:**

```python
class ChunkType(str, Enum):
    prose = "prose"
    table = "table"
    figure_caption = "figure_caption"
    supplementary = "supplementary"

class AnnotatedChunk(BaseModel):
    chunk_id: str
    text: str
    chunk_type: ChunkType
    source_section: str  # which section title this came from
    summary: str = ""    # LLM-generated summary highlighting tools/parameters
    page_number: Optional[int] = None
```

**1b. Create `utils/paper_indexer.py` — a per-paper ephemeral vector index:**

This is a new class, `PaperRAGStore`, distinct from `ProtocolRAGStore`. It takes a parsed `Paper` + `list[Figure]` and builds a transient Chroma collection (no persistence needed — it lives for the duration of one paper parse). It should:

- Split each section's text into chunks, tagging prose vs. table (detect Markdown table syntax `|---|`).
- **Table integrity validation (CRITICAL):** After splitting, run a structural check on candidate table chunks — verify they have consistent column counts and non-empty cells. If a table chunk fails validation (common with older PDFs where `marker-pdf` produces corrupted Markdown), fall back to the multimodal image path: use `extract_figure_panel_images_from_pdf` (already in `utils/pdf.py`, line 191) to crop the table's bounding box from the page and send the raw image bytes to the LLM for structured parameter extraction, bypassing Markdown entirely. This mirrors the existing strategy in `figure_parser.py` where panel images are sent directly to a vision-capable model.
- Ingest figure captions at the **panel level, not the whole-caption level.** The existing `figure_parser.py` already decomposes figures into `SubFigure` objects with individual `panel_id` and `title` fields (via `_decompose_subfigures`, line 886). Use these panel-level chunks rather than whole captions. Tag each as `figure_caption:Figure_1_Panel_C` (not just `figure_caption`). This prevents the "orphaned caption" problem where a single Figure 1 spans multiple assays (Panel A is scRNA-seq, Panel C is ChIP-seq) and a retrieval query for ChIP-seq parameters incorrectly pulls in scRNA-seq context from Panel A.
- For each chunk, call the LLM to generate a 1-sentence summary emphasizing tools, versions, and parameters (BioWorkflow's "context summary" strategy).
- Embed both original text and summary into the same collection with metadata tags for `chunk_type` and `panel_id` (when applicable).

**1c. Wire `PaperRAGStore` into `MethodsParser.__init__`** alongside the existing `ProtocolRAGStore`. Both stores should be queryable, with results merged by relevance score.

### Test: `tests/test_paper_indexer.py`

- Unit test: given a `Paper` with a methods section containing an inline Markdown table, verify that table chunks get `chunk_type=table`.
- Unit test: given a `Figure` with caption mentioning "STAR v2.7.10a", verify the summary chunk contains "STAR" and "2.7.10a".
- Unit test: given a corrupted Markdown table (inconsistent column counts, mangled cell delimiters), verify the fallback image-extraction path is triggered and the chunk is still created with correct parameters.
- Unit test: given a Figure with 3 panels (A: scRNA-seq UMAP, B: volcano plot, C: ChIP-seq peaks), verify that 3 separate panel-level chunks are created, each tagged with their specific `panel_id`, and that querying for "ChIP-seq" returns only the Panel C chunk.
- Integration test: index a real paper from `tests/snapshots/`, query for a known tool, assert the top result includes the table-bound parameters.

---

## Phase 2: Hierarchical Decomposition & Iterative Retrieval

### What Exists Already

- `_identify_assays()` does a single LLM call to list all assays.
- `_parse_assay()` extracts paragraph text per assay and decomposes into `AnalysisStep` objects.
- `_infer_missing_computational_parameters()` does one RAG pass per assay via tool-calling.

### What's Actually Missing

The current pipeline does **breadth-first identification → depth-first per-assay decomposition**, but it never reformulates queries when the first retrieval miss occurs. If the initial paragraph extraction misses a parameter (because it was in a different section or a table), there is no retry mechanism. This is BioWorkflow's key insight: iterative query refinement with dynamic reformulation.

### Improved Implementation

**2a. Replace `_identify_assays` with a two-stage decomposer:**

Stage 1 (unchanged): Identify assay names.
Stage 2 (new): For each assay, ask the LLM to generate a "workflow skeleton" — the expected high-level stages (e.g., "1. Read trimming, 2. Alignment, 3. Quantification, 4. Differential analysis"). This skeleton becomes the query plan for retrieval.

**2b. Add `_iterative_retrieval_loop` to `MethodsParser`:**

```python
def _iterative_retrieval_loop(
    self,
    assay_name: str,
    skeleton_stages: list[str],
    paper_index: PaperRAGStore,
    template: Optional[AssayTemplate] = None,
    max_refinement_rounds: int = 2,
) -> list[AnnotatedChunk]:
    """Multi-round retrieval with dynamic reformulation and early exit.

    Circuit breaker: if round 1 yields hits covering all required fields
    from the assay template (software + version + key parameters), skip
    refinement rounds entirely. Only refine when fields are explicitly missing.
    This reduces worst-case LLM calls from 13/assay to 5/assay for well-
    documented papers.
    """
    collected = []
    for stage in skeleton_stages:
        query = f"{assay_name} {stage}"
        hits = paper_index.query(query, top_k=3)
        collected.extend(hits)

        # Circuit breaker: check completeness against template
        if template and self._stage_fields_complete(hits, stage, template):
            continue  # skip refinement for this stage

        # Refinement rounds (only when fields are missing)
        missing = self._detect_missing_fields(hits, stage, template)
        for round_num in range(max_refinement_rounds):
            if not missing:
                break
            refined_query = f"{missing['software']} {missing['field']} parameters settings"
            refinement_hits = paper_index.query(refined_query, top_k=2)
            collected.extend(refinement_hits)
            missing = self._detect_missing_fields(
                hits + refinement_hits, stage, template
            )
    return collected
```

**Latency and rate-limit mitigation (CRITICAL):** The iterative loop is a massive API call multiplier. Without safeguards, a 5-assay paper could generate 65+ LLM calls, breaching the 90-second timeout (`RESEARCHER_AI_LLM_TIMEOUT_SECONDS`) and triggering 429 rate limits. Three mitigations are required:

1. **Circuit breakers** (shown above): If round 1 yields a high-confidence match with all required template fields, skip refinement entirely. For well-documented papers this reduces calls from ~13/assay to ~5/assay.

2. **Async concurrency in `orchestrator.py`:** The orchestrator is currently fully synchronous (confirmed: zero `async def` or `asyncio` references in `orchestrator.py`). Add an `asyncio.gather`-based dispatcher for per-assay parsing so that the 5 assays run concurrently rather than sequentially. Note: retrieval queries against the local `PaperRAGStore` (Chroma + embeddings) are cheap and do not hit external APIs — only the skeleton-generation LLM calls need rate-limit awareness.

3. **Adaptive timeout:** Scale `RESEARCHER_AI_LLM_TIMEOUT_SECONDS` per paper based on assay count: `timeout = base_timeout * (1 + 0.3 * max(0, assay_count - 3))`. This gives simple papers the default 90s but allows 5-assay papers ~150s.

**2c. Feed retrieved chunks into `_parse_assay` as additional context**, replacing the current 3000-char paragraph truncation. The prompt should receive the concatenated relevant chunks (with provenance tags) rather than a raw substring of the methods text.

### Test: `tests/test_iterative_retrieval.py`

- Mock a paper where the "STAR" tool is mentioned in the methods prose but `--outSAMtype BAM SortedByCoordinate` appears only in a supplementary table. Assert that round 1 finds STAR, round 2 reformulates and finds the parameter.
- Regression test: run against existing `tests/snapshots/methods/*.yaml` fixtures. Assert that step recall does not decrease (≥ 0.7 heading extraction recall baseline from `test_methods_parser_eval.py`).

---

## Phase 3: Staged Assembly & Automated Validation

### What Exists Already

- `pipeline/builder.py` assembles workflows with DAG-aware ordering.
- `pipeline/orchestrator.py` runs a state machine with retry logic (max 2 build attempts).
- `snakemake_gen.py` generates Snakefiles validated via `snakemake --lint` + dry-run.
- `figure_calibration.py` applies deterministic rules to override LLM extraction.

### What's Actually Missing

There is **no evidence-grounding validation agent**. The current pipeline trusts LLM outputs for software versions and parameters. If the LLM hallucinates "STAR v2.7.10a" when the paper says nothing about the version, that hallucination propagates to the generated workflow. BioWorkflow's validation agent cross-checks every extracted field against source evidence.

### Improved Implementation

**3a. Create `parsers/validation_agent.py`:**

This module receives a fully assembled `Method` object + the `PaperRAGStore` index and performs field-level verification:

```python
class EvidenceCategory(str, Enum):
    stated_in_paper = "stated_in_paper"       # exact string found in paper text/table
    inferred_default = "inferred_default"     # well-known default for that tool version
    inferred_from_protocol = "inferred_from_protocol"  # found in protocol docs, not paper
    ungrounded = "ungrounded"                 # not found anywhere — possible hallucination

class ValidationVerdict(BaseModel):
    field: str                    # e.g., "step_3.software_version"
    claimed_value: str            # e.g., "2.7.10a"
    evidence_category: EvidenceCategory
    evidence_source: Optional[str]  # chunk_id, protocol doc name, or "not found"
    action: str                   # "keep", "keep_as_default", "flag_ungrounded", "discard"
    rationale: str = ""           # why this verdict was reached

class ValidationReport(BaseModel):
    verdicts: list[ValidationVerdict]
    ungrounded_count: int
    inferred_default_count: int
    total_fields_checked: int
```

The `inferred_default` category is critical for avoiding false positives. Scientific papers routinely say things like "reads were aligned using standard STAR parameters" — the LLM correctly populates default parameters for that tool, but the exact parameter strings never appear in the text. The validation agent must be prompted to distinguish between "the LLM hallucinated a specific random value" (→ `ungrounded`) and "the LLM correctly populated a well-known default that the paper implied" (→ `inferred_default`, action `keep_as_default`). The `rationale` field captures the reasoning so downstream consumers can audit the decision.

For each `AnalysisStep` in each `Assay`:
1. Check `software`: query the paper index for the tool name. If not found anywhere in the paper, flag it.
2. Check `software_version`: query for the version string. If found only in protocol docs (not the paper), mark as `inferred_from_protocol`. If it's a known default version for a tool mentioned in the paper, mark as `inferred_default`.
3. Check each `parameters` key-value: query for the parameter value. If the value is a documented default for the claimed software version, mark as `inferred_default` with `action=keep_as_default`. Otherwise, ungrounded parameters get `action=flag_ungrounded`.

**3b. Add template-driven assembly** by creating assay-type-specific prompt templates in `parsers/templates/`:

- `rnaseq_template.yaml`: expected stages (QC → trim → align → quantify → DE), required fields per stage.
- `chipseq_template.yaml`: expected stages (QC → trim → align → peak_call → motif).
- `variant_calling_template.yaml`: expected stages (QC → trim → align → call → filter → annotate).

These templates serve as structural priors — if the LLM produces an RNA-seq workflow without a quantification step, the template flags the gap before validation.

**3c. Wire into `orchestrator.py`** as a new stage between "build workflow graph" and "build pipeline":

```
Stage 6: Build workflow graph
Stage 6.5 (NEW): Validate method against paper evidence
Stage 7: Build pipeline
```

### Test: `tests/test_validation_agent.py`

- **Hallucination suppression test**: Create a `Method` where step 3 claims `software_version="5.0.0"` for DESeq2, but the paper text only mentions "DESeq2" with no version. Assert `evidence_category=ungrounded` and `action="flag_ungrounded"`.
- **Grounded parameter test**: Create a `Method` where STAR `--sjdbOverhang 100` is claimed, and the paper table contains exactly that parameter. Assert `evidence_category=stated_in_paper` and `action="keep"`.
- **Inferred default test (false-positive prevention)**: Create a `Method` where STAR `--outSAMtype BAM SortedByCoordinate` is claimed, and the paper says "reads were aligned with STAR using default parameters." Assert `evidence_category=inferred_default` and `action="keep_as_default"` — NOT `flag_ungrounded`. This is the critical test for the false-positive vulnerability.
- **Template gap detection test**: Feed an RNA-seq method missing a quantification step through the `rnaseq_template`. Assert a warning is emitted.

---

## Phase 4: Real-World Testing & Benchmarking

### What Exists Already

- `tests/test_methods_parser_eval.py`: 5 parametrized test methods testing heading extraction recall (≥0.7), compressed summary coverage (≥0.8), paragraph keyword extraction, heading+LLM merge coverage, and computational paragraph content.
- `tests/snapshots/methods/*.yaml`: Fixture files with expected assay names, computational classifications, and keywords.
- `tests/test_methods_parser_snapshot.py`: Snapshot tests with frozen LLM responses.
- `tests/test_phase4_integration.py`: Full pipeline integration tests (paper → JSON).
- `scripts/run_workflow.py`: CLI entry point for end-to-end runs.

### What the Original Plan Gets Wrong

The plan proposes three test cases but doesn't anchor them to existing infrastructure. It should extend the existing parametrized evaluation framework rather than creating ad-hoc standalone tests.

### Improved Implementation

**4a. Extend `tests/snapshots/methods/` with new fixture files targeting BioWorkflow scenarios:**

Each new fixture YAML should include:
- `pmcid` or `doi` for the paper
- `expected_table_parameters`: parameters that appear only in tables (tests Phase 1)
- `expected_iterative_hits`: parameters that require multi-round retrieval (tests Phase 2)
- `expected_ungrounded_fields`: fields the validation agent should flag (tests Phase 3)

**4b. New parametrized test class `tests/test_bioworkflow_integration.py`:**

```python
class TestBioWorkflowIntegration:
    """End-to-end tests for BioWorkflow-style multimodal extraction."""

    @pytest.mark.parametrize("fixture", BIOWORKFLOW_FIXTURES)
    def test_table_parameter_extraction(self, fixture):
        """Phase 1: Parameters in tables are found via multimodal index."""
        method = parse_paper_to_method(fixture.pmcid)
        extracted_params = collect_all_parameters(method)
        for param in fixture.expected_table_parameters:
            assert param in extracted_params, (
                f"Table-bound parameter {param!r} not extracted"
            )

    @pytest.mark.parametrize("fixture", BIOWORKFLOW_FIXTURES)
    def test_hallucination_suppression(self, fixture):
        """Phase 3: Ungrounded fields are flagged, not silently passed."""
        method = parse_paper_to_method(fixture.pmcid)
        report = validate_method_against_paper(method, fixture.paper_index)
        for field in fixture.expected_ungrounded_fields:
            verdict = find_verdict(report, field)
            assert verdict.action in ("flag_ungrounded", "discard"), (
                f"Ungrounded field {field!r} was not flagged"
            )

    @pytest.mark.parametrize("fixture", BIOWORKFLOW_FIXTURES)
    def test_step_recall_not_regressed(self, fixture):
        """Regression: BioWorkflow changes must not reduce step recall below baseline."""
        method = parse_paper_to_method(fixture.pmcid)
        recall = compute_step_recall(method, fixture.expected_steps)
        assert recall >= 0.7, f"Step recall {recall:.2f} below baseline 0.7"
```

**4c. Benchmark script `scripts/benchmark_bioworkflow.py`:**

Run the full pipeline on 10 curated papers, comparing:
- Step recall (before vs. after BioWorkflow integration)
- Parameter extraction precision (grounded vs. ungrounded)
- Multi-round retrieval efficiency (queries per paper)
- LLM cost (total tokens consumed)

Output a summary table and save per-paper JSON reports to `docs/benchmarks/`.

**4d. Specific paper selections for the benchmark set:**

Choose papers that stress different failure modes:
- 2-3 papers with parameters in supplementary tables (tests multimodal indexing)
- 2-3 papers with long methods sections >10K chars (tests iterative retrieval)
- 2-3 papers with missing version numbers (tests hallucination suppression)
- 2-3 papers representing distinct workflows (scRNA-seq, ChIP-seq, WGS) from existing `tests/snapshots/methods/`

---

## Dependency & Migration Risks

**New dependencies:** None required. The existing Chroma + sentence-transformers stack supports the per-paper ephemeral index. The existing LiteLLM interface supports the new prompts.

**Migration risk — `_parse_assay` prompt change:** The current prompt feeds raw paragraph text. Switching to retrieved chunks changes the input distribution for all downstream LLM calls. This must be gated behind a feature flag (e.g., `RESEARCHER_AI_BIOWORKFLOW_MODE=on`) so the existing behavior remains the default until benchmarks confirm improvement.

**Migration risk — token budget and latency:** Multi-round retrieval multiplies LLM calls. Worst case: 5 assays × (1 skeleton call + 4 stages × up to 2 refinement rounds) = 45 LLM-adjacent operations per paper. However, the circuit breaker (Phase 2b) and async concurrency (Phase 2 mitigation) reduce the effective wall-clock time. The retrieval queries themselves hit the local Chroma index (no API calls), so the real bottleneck is skeleton-generation LLM calls (1 per assay, parallelizable). Budget this against `RESEARCHER_AI_LLM_TIMEOUT_SECONDS` (default 90s) with the adaptive timeout scaling described in Phase 2.

**Migration risk — test snapshot invalidation:** If prompts change, existing snapshot tests in `test_methods_parser_snapshot.py` will fail. Plan to regenerate snapshots as part of Phase 2 integration.

---

## Known Vulnerabilities & Defensive Mitigations

Four attack surfaces were identified in the plan and are now addressed inline in the relevant phases. Summary:

| # | Vulnerability | Phase | Mitigation | Where in Code |
|---|--------------|-------|------------|---------------|
| 1 | **Corrupted table extraction** — `marker-pdf` produces mangled Markdown for complex/merged-cell tables, causing `PaperRAGStore` to misclassify table chunks as prose | Phase 1 | Structural integrity check + fallback to multimodal image extraction via `extract_figure_panel_images_from_pdf` (already in `utils/pdf.py:191`) | `utils/paper_indexer.py` |
| 2 | **Validation false positives** — agent flags well-known tool defaults as "ungrounded" when paper says "standard parameters" | Phase 3 | New `inferred_default` evidence category + prompted distinction between hallucination vs. correct default population | `parsers/validation_agent.py` |
| 3 | **Orphaned caption cross-contamination** — whole-caption chunks mix parameters from different assay panels in the same figure | Phase 1 | Panel-level chunking using existing `SubFigure` decomposition from `figure_parser.py:886` | `utils/paper_indexer.py` |
| 4 | **Rate-limit cascade** — iterative loop generates up to 65 LLM calls per paper, breaching 90s timeout and triggering 429s | Phase 2 | Circuit breakers (skip refinement when template fields are complete) + async concurrency in `orchestrator.py` + adaptive timeout scaling | `methods_parser.py`, `orchestrator.py` |

---

## Recommended Implementation Order

1. **Phase 1** (1-2 weeks): `AnnotatedChunk` model + `PaperRAGStore` + unit tests. This is the foundation and can be built independently.
2. **Phase 3 — validation agent only** (1 week): Build `validation_agent.py` and its tests. This provides immediate value even without Phases 2's iterative retrieval.
3. **Phase 2** (2 weeks): Iterative retrieval loop + skeleton decomposer. This is the highest-complexity change and benefits from having the validation agent already in place.
4. **Phase 4** (1 week): Benchmark fixtures, parametrized tests, regression suite.

This reordering front-loads value and risk-mitigation (validation catches hallucinations early) while deferring the most complex retrieval changes until the infrastructure is solid.

---

## Sources

- [BioWorkflow: Retrieving comprehensive bioinformatics workflows from publications (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12596265/)
- [BioWorkflow — Briefings in Bioinformatics (Oxford Academic)](https://academic.oup.com/bib/article/26/6/bbaf571/8315884)
