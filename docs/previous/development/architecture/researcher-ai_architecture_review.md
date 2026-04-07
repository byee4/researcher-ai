> [!WARNING]
> **Deprecated Architecture Review**
>
> This document is retained for historical context only.
> The canonical architecture-planning document is `docs/WORKFLOW_GRAPH_IR_PLAN.md`.
> The current implemented architecture reference remains `docs/ARCHITECTURE.md`.

# researcher-ai: Architecture Review & Improvement Recommendations

**Date:** April 6, 2026
**Scope:** Full codebase review against V2_ARCHITECTURE_PLAN.md, with focus on efficiency, accuracy, and test coverage

---

## V2 Plan Alignment Summary

The V2 plan defines four phases. Here is where each stands:

| Phase | Feature | Completeness | Key Gap |
|-------|---------|:---:|---------|
| 1 | Universal LLM Interface (litellm) | ~95% | Model aliases hardcoded; provider quirks leak through |
| 2 | Multimodal Ingestion Pipeline | ~60% | Vision path exists but text-first dominates; no E2E test |
| 3 | RAG-Augmented Method Inference | ~75% | Only 3 protocol docs; tool-calling is optional/fragile |
| 4 | State-Graph Orchestration | ~70% | Not truly agentic — repair is procedural, not LLM-driven |

---

## Phase 1: Universal LLM Interface

**Status: Substantially Complete**

`researcher_ai/utils/llm.py` successfully wraps litellm behind `extract_structured_data()`. API key routing, provider normalization, multimodal image encoding, and caching all work. The backward-compat alias `ask_claude_structured` is preserved.

**Issues found:**

- **Hardcoded model defaults.** `DEFAULT_MODEL = "gpt-5.4"` and `MODEL_ALIAS_MAP` (mapping `"gemini-3.1-pro"` → `"gemini-2.5-pro"`, `"gpt-5.4-planning"` → `"gpt-4.1"`) are baked into the code. Adding models requires code changes. These should be externalized to a config file.

- **Provider-specific quirks leak through.** GPT-5 enforces `temperature=1.0` (line ~135). Gemini enforces a 400-token minimum. Gemini safety settings disable all filters. These are scattered conditionals that will break when providers change APIs — they should be captured in a provider config registry.

- **Cache key collisions possible.** The LLMCache hashes `json_schema + image_hash`, but semantically equivalent schemas with different field ordering will produce different hashes.

- **No image size validation.** `image_bytes` are base64-encoded inline with no size cap. A 20MB figure image would blow past token limits.

**Recommendations:**
1. Move model aliases and provider configs to `researcher_ai/config/models.yaml`
2. Add image size validation (resize or reject above a threshold)
3. Normalize schema JSON before hashing for cache keys

---

## Phase 2: Multimodal Ingestion Pipeline

**Status: Partially Implemented**

The plumbing exists: `extract_figure_panel_images_from_pdf()` crops panels, `extract_markdown_from_pdf_with_marker()` uses marker-pdf for spatial Markdown, and `extract_structured_data()` accepts `image_bytes`. However, the figure parser doesn't consistently use the vision path.

**Issues found:**

- **Text-first still dominates.** The figure parser's primary flow extracts captions and BioC context as text, then sends that to an LLM. The vision path (sending raw panel images) is secondary. This contradicts Phase 2's stated goal of "passing raw image panels directly to a multimodal vision model."

- **No end-to-end test for the vision path.** `test_phase2_multimodal_pipeline.py` has only 3 tests, all fully mocked. No test actually processes a PDF through marker-pdf or sends a cropped panel to a vision model.

- **Silent fallback hides degradation.** When marker-pdf fails or vision model is unavailable, the system silently falls back to text extraction with only an INFO-level log. Users don't know their parsing quality has degraded.

- **Panel bounding box validation missing.** `PanelBoundingBox` uses `ge=0, le=1` constraints but doesn't validate that `x0 < x1` or `y0 < y1`.

**Recommendations:**
1. Decide whether vision-first or text-first is the intended primary path for PDFs, and refactor accordingly
2. Add at least one integration test that processes a real PDF through the full multimodal pipeline
3. Promote fallback events to WARNING level and surface them in parse results
4. Add cross-field validation for bounding box coordinates

---

## Phase 3: RAG-Augmented Method Inference

**Status: Implemented but Thin**

`ProtocolRAGStore` in `utils/rag.py` supports hybrid vector+lexical retrieval backed by ChromaDB. `MethodsParser` registers a `search_protocol_docs` tool for the LLM to call during assay analysis. The architecture is sound.

**Issues found:**

- **Knowledge base has only 3 documents** (`star_alignment.md`, `eclip_sop.md`, `deseq2_workflow.md`). The V2 plan references "STAR manuals, Seurat vignettes, and internal Yeo Lab SOPs." Most of that content is absent. RAG retrieval will return generic or empty results for most assays.

- **Embedding model is hardcoded** to `all-MiniLM-L6-v2` (a small, general-purpose model). It may not handle domain-specific bioinformatics terminology well. The model, chunk size (900), and overlap (120) are all non-configurable.

- **Tool-calling is fragile.** Lines 54–66 of methods_parser.py show a fallback: if tool-calling fails (some litellm providers don't support it), the system silently drops RAG enrichment entirely. There's no retry or alternative strategy.

- **No explicit "missing parameter" detection.** The plan says to "evaluate the graph for missing computational parameters" before querying RAG. The current code doesn't have a validation step that identifies what's missing — RAG is offered as an optional tool, not triggered by detected gaps.

- **Chroma persistence path is relative** (`.rag_chroma`), making it non-portable and potentially writing into unexpected directories.

- **Lexical fallback drops short tokens** (< 3 chars), which removes bioinformatics abbreviations like "PCR", "ATP", etc.

**Recommendations:**
1. Expand the knowledge base to 15–20 protocol docs covering common bioinformatics tools (BWA, Salmon, Cutadapt, Seurat, scVI, CellRanger, etc.)
2. Add a parameter completeness check after initial LLM extraction — if critical fields are None, explicitly trigger RAG
3. Make embedding model, chunk size, and overlap configurable via constructor parameters
4. Fix the lexical tokenizer to preserve short domain terms
5. Use an absolute path for Chroma persistence, derived from project root

---

## Phase 4: State-Graph Orchestration & Agentic Execution

**Status: Structurally Complete, Not Truly Agentic**

`WorkflowOrchestrator` uses LangGraph (with sequential fallback) to route state through `parse_paper → parse_figures → parse_methods → parse_datasets → parse_software → build_pipeline`. `PipelineBuilder` validates Snakefiles via `snakemake --lint` and `-n`, retrying up to 3 rounds. TSCC SLURM profiles are generated.

**Issues found:**

- **Repair is procedural, not LLM-driven.** The V2 plan says the Engineering Agent should "parse the traceback, rewrite the Snakefile, and re-test." In practice, `_validate_and_repair_snakefile()` and `_repair_snakefile()` use regex-based procedural fixes. No LLM is involved in interpreting errors or reasoning about fixes. This is the biggest gap between plan and reality.

- **Builder is stateless across retries.** Each retry regenerates the Snakefile from scratch rather than learning from the previous attempt's errors. The retry loop in the orchestrator calls the builder as a fresh function, not an agent with memory.

- **No formal bash tool abstraction.** The plan specifies "provide the Engineering Agent with a local bash execution tool." Currently, `subprocess.run()` calls are embedded directly in builder methods. There's no sandboxing, no timeout handling, and no recovery if the subprocess hangs.

- **LangGraph is optional.** If unavailable, `_run_sequential()` executes nodes in fixed order. This works, but the sequential fallback cannot express conditional retry edges, meaning the self-healing loop only works with LangGraph.

- **WorkflowState has no validation.** It's a `TypedDict(total=False)` where all fields are optional. A downstream node could receive a state with `paper=None` and crash. There are no guards or assertions at node boundaries.

- **TSCC SLURM profile is hardcoded.** `_tscc_slurm_profile()` returns fixed `--partition`, `--account`, `--mem` values. There's no abstraction for other HPC systems.

**Recommendations:**
1. Implement `repair_snakefile_with_llm()`: present the Snakefile + lint error to the LLM, let it reason about the fix, validate the result
2. Add state validation at each node boundary (assert required fields are present before proceeding)
3. Abstract HPC profiles into a config system (TSCC, AWS Batch, local, etc.)
4. Wrap subprocess calls in a `BashTool` class with timeout, sandboxing, and error capture
5. Consider making LangGraph a hard dependency if the conditional retry logic is important

---

## Cross-Cutting Code Quality Issues

### Error Handling

Error handling is inconsistent across the codebase. Some modules use careful fallback chains (paper_parser's PMCID → PMID → PDF degradation is well done), while others swallow exceptions silently. Key problems:

- **Silent degradation is pervasive.** Many failures are logged at INFO level and produce stub objects. Users don't know when parsing quality has degraded. The `parse_warnings` list on Method objects accumulates warnings, but nothing surfaces them prominently.

- **No exception hierarchy.** There's no `ParseError`, `ExternalAPIError`, or `ValidationError` hierarchy. All errors are caught as generic `Exception`, making it impossible to handle specific failure modes differently.

- **No validation at module boundaries.** Pydantic validates model fields, but there's no check that LLM-extracted data is semantically valid (e.g., an AnalysisStep with an empty `command` field passes Pydantic but will produce a broken Snakefile).

### Configuration Management

Hardcoded values are scattered throughout the codebase: timeouts (30s for HTTP, 90s for LLM), thresholds (0.75 for calibration confidence), retry counts (3), chunk sizes (900), API endpoints, model names, and SLURM profiles. Every deployment-specific change requires a code change. A centralized config system (YAML or dataclass-based) would significantly improve maintainability.

### Testability & Dependency Injection

Most parsers instantiate their own dependencies internally (creating LLMCache, ProtocolRAGStore, HTTP clients in `__init__`). This makes unit testing difficult — every test must mock at the module level rather than injecting test doubles. Accepting dependencies as constructor parameters would enable much cleaner testing.

---

## Test Suite Assessment

The test suite has **810+ test functions** across 22 files — a solid foundation. However, there are structural gaps.

### What Works Well

- **Pure function testing is excellent.** Regex extraction, section splitting, figure ID parsing, accession validation — these have thorough, fast, isolated tests.
- **Snapshot-based testing** provides regression detection for parser outputs.
- **Test tier separation** (no-network, snapshot, live) via pytest markers is well-designed.
- **Model serialization roundtrips** are comprehensive.

### Critical Gaps

- **No regression baseline.** There's no tracked metric of "X papers parse correctly." Without this, you can't detect slow accuracy degradation.

- **Phase 3–4 tests are minimal.** RAG has 3 tests. The orchestrator has 2. State-graph execution has 3. These are the most architecturally complex components and the least tested.

- **No failure mode tests.** Almost every mock returns a success case. What happens when the LLM returns malformed JSON? When PubMed returns a 429? When marker-pdf crashes? These paths are untested.

- **No pipeline syntax validation.** Pipeline generation tests use string matching (`"rule all:" in output`), not actual `snakemake --lint` or Nextflow syntax validation. A syntactically broken Snakefile could pass all tests.

- **Circular dependency detection missing.** `AssayGraph` doesn't validate that `depends_on` edges form a DAG. No test checks for this. A cycle would cause infinite loops in pipeline generation.

- **Only 1 snapshot fixture for methods parsing** (eCLIP paper). A single test paper can't catch assay-type-specific regressions.

### Recommended Test Additions

**High priority:**
1. Add failure mode tests for every external call (LLM timeout, malformed response, API rate limit, empty PDF)
2. Add real `snakemake --lint` validation in pipeline tests (or at minimum, parse the generated Snakefile AST)
3. Add circular dependency detection to AssayGraph and test it
4. Expand snapshot fixtures to 5+ papers covering RNA-seq, ATAC-seq, proteomics, imaging, and multi-omic workflows
5. Add an integration test that exercises the full RAG path: paper → missing parameter → RAG retrieval → enriched output

**Medium priority:**
6. Add concurrent access tests for LLMCache
7. Add large-input tests (1000+ page PDF, 100+ sample dataset)
8. Test the LangGraph orchestrator path specifically (not just sequential fallback)
9. Add regression tracking: store per-paper accuracy scores and fail CI if they drop

---

## Top 10 Prioritized Recommendations

1. **Expand the RAG knowledge base** from 3 to 15+ protocol docs. This is the highest-leverage change for parsing accuracy — the infrastructure exists but is starved of data.

2. **Add failure mode tests** for LLM calls, API calls, and PDF parsing. The current test suite only validates happy paths, which means production failures will be surprises.

3. **Implement LLM-assisted Snakefile repair.** The procedural repair logic is the biggest divergence from the V2 plan. Presenting the error + Snakefile to an LLM for reasoning-based repair would significantly improve pipeline generation accuracy.

4. **Centralize configuration.** Extract all hardcoded values (model names, timeouts, thresholds, SLURM profiles, chunk sizes) into a single config system. This is a prerequisite for deploying to different environments.

5. **Add state validation to the orchestrator.** Assert required fields are present at each node boundary in WorkflowState. This prevents cryptic downstream crashes.

6. **Clarify the multimodal strategy.** Either promote vision-first as the primary PDF path (and add E2E tests for it) or document it as future work. The current ambiguity means neither path gets proper testing or optimization.

7. **Add circular dependency detection** to AssayGraph. A single `is_dag()` check would prevent infinite loops in pipeline generation.

8. **Introduce dependency injection** for parsers. Accept LLM clients, HTTP clients, and RAG stores as constructor parameters instead of creating them internally.

9. **Add a regression baseline.** Track how many papers from a curated set parse correctly, and fail CI if the number drops.

10. **Define an exception hierarchy.** Create `ParseError`, `ExternalAPIError`, `LLMExtractionError`, and `ValidationError` classes so error handling can be specific rather than catch-all.
