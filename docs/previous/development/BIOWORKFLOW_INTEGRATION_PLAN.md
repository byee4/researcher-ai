## BioWorkflow Integration Merged Plan (Phases 1–4, Risk-Hardened)

### Summary
Implement BioWorkflow in a staged rollout that reuses existing parsers, adds a per-paper multimodal index, introduces iterative retrieval only after validation guardrails exist, and benchmarks against existing eval infrastructure.  
Execution order: **Phase 1 → Phase 3 (validation-first) → Phase 2 → Phase 4**.

### Implementation Changes
1. **Phase 1: Multimodal Provenance Index**
- Add `ChunkType` and `AnnotatedChunk` models (prose/table/figure_caption/supplementary + provenance metadata + summary).
- Build `PaperRAGStore` (ephemeral, per-paper) separate from `ProtocolRAGStore`.
- Index prose, tables, supplementary text, and **panel-level** figure chunks (not whole-caption chunks).
- Add table integrity checks; if table markdown is corrupted, fall back to table-image extraction and vision parsing.
- Wire `PaperRAGStore` into `MethodsParser` and merge retrieval with protocol retrieval by score.

2. **Phase 3: Evidence Validation + Template Priors (before iterative loop)**
- Add `validation_agent.py` to verify each extracted software/version/parameter field against evidence.
- Use `EvidenceCategory`: `stated_in_paper`, `inferred_default`, `inferred_from_protocol`, `ungrounded`.
- Enforce action policy: `keep`, `keep_as_default`, `flag_ungrounded`, `discard`.
- Add assay templates (`rnaseq`, `chipseq`, `variant_calling`) for required stage coverage checks.
- Insert orchestrator stage **6.5 Validate Method Evidence** between graph build and pipeline build.

3. **Phase 2: Hierarchical Decomposition + Iterative Retrieval**
- Keep assay identification, then add per-assay workflow skeleton generation (stage plan).
- Implement `_iterative_retrieval_loop` with targeted reformulation only for missing fields.
- Replace per-assay raw text truncation context with retrieved, provenance-tagged chunks.
- Add circuit breakers to skip refinement when required template fields are complete.
- Add async per-assay orchestration and adaptive timeout scaling by assay count.

4. **Phase 4: Real-World Benchmarking**
- Extend existing snapshot/eval fixtures with BioWorkflow fields:
  `expected_table_parameters`, `expected_iterative_hits`, `expected_ungrounded_fields`.
- Add end-to-end parametrized integration tests over curated papers.
- Add benchmark script comparing before/after on recall, grounded-parameter precision, retrieval rounds, latency, and token cost.
- Gate rollout behind `RESEARCHER_AI_BIOWORKFLOW_MODE=on`; default remains current behavior until benchmarks pass.

### Public Interfaces / Contracts
- New models: `ChunkType`, `AnnotatedChunk`, `ValidationVerdict`, `ValidationReport`.
- New component: `PaperRAGStore` (ephemeral, parse-scope only; no persistent migration).
- New orchestrator stage: validation step between workflow graph assembly and pipeline build.
- New config behavior:
  `RESEARCHER_AI_BIOWORKFLOW_MODE`,
  adaptive `RESEARCHER_AI_LLM_TIMEOUT_SECONDS`.

### Test Plan & Acceptance Criteria
- **Phase 1 tests:** chunk typing, panel-level chunking, corrupted-table fallback, known table-parameter retrieval.
- **Phase 3 tests:** hallucination suppression, grounded parameter keep, inferred-default keep-as-default, missing-template-stage warnings.
- **Phase 2 tests:** iterative query refinement recovers hidden parameters; no regression below current recall baseline.
- **Phase 4 tests:** end-to-end fixtures + benchmark report generation.
- Acceptance thresholds:
  no recall regression vs baseline,
  higher grounded parameter precision,
  ungrounded fields explicitly flagged,
  latency within adaptive timeout budget.

### Risks and Responses
- **Corrupted table markdown:** structural validation + image fallback extraction.
- **False-positive validation on implied defaults:** explicit `inferred_default` category and policy.
- **Cross-assay caption contamination:** panel-level chunking with panel metadata tags.
- **LLM call explosion / rate limits:** circuit breakers, async assay concurrency, adaptive timeout scaling.
- **Prompt/snapshot churn:** feature-flag rollout and planned snapshot regeneration after stabilization.

### Assumptions / Defaults
- Existing marker-pdf, figure parsing, Chroma, and LiteLLM remain in use; no new core dependency required.
- Per-paper index is transient and recreated per parse.
- Validation is blocking for flagged ungrounded critical fields by default (non-critical can warn).
- Benchmarks run on a curated 10-paper set spanning table-heavy, long-methods, missing-version, and multi-assay cases.
