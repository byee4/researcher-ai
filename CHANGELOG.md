# Changelog

This changelog was generated from archived markdown in `docs/previous/` on 2026-04-05.

## 2026-04-10
- feature: document PMID 39303722 benchmark-backed OpenAI-only export profile and include quota/rate-limit interpretation guidance in README/config docs.
- bugfix: stop per-assay LLM call cascades after first quota/rate-limit failure by opening an assay-parse circuit breaker and using text fallback for remaining assays.
- feature: add PMID 39303722 full-run findings document and Beads tracking tree for parser failure/fallback remediation.
- bugfix: add deterministic caption panel-split fallback and explicit warning marker for subfigure decomposition empty-response cases.

## 2026-04-09 (v2.2.3)
- feature: add Figure 2 (PMID 39303722) empty-response investigation tooling with env-gated structured extraction telemetry.
- feature: add reproducibility harness `scripts/investigate_figure2_empty_responses.py` and generated experiment artifacts under `parse_results/`.
- bugfix: allow investigation-mode fallback isolation via `RESEARCHER_AI_DISABLE_MODEL_FALLBACKS`.
- bugfix: keep heuristic MethodsParser fallback descriptions/steps when per-assay LLM parsing fails (for example quota/rate-limit errors), instead of forcing all assays to `Could not be parsed.`.
- bugfix: avoid repeated LiteLLM schema-error loops on OpenAI by preflighting strict `json_schema` compatibility and using `json_object` first when schemas are incompatible.

## 2026-04-09 (v2.2.2)
- Hardened figure parsing timeout behavior for production stability:
  - adaptive orchestrator figure timeout floor by figure count (`RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS`)
  - per-step figure LLM token budgets to reduce long-tail latency:
    - `RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS`
    - `RESEARCHER_AI_FIGURE_PURPOSE_MAX_TOKENS`
    - `RESEARCHER_AI_FIGURE_METHODS_DATASETS_MAX_TOKENS`
- Added regression tests and configuration/docs updates for timeout observability and controls.
- bugfix: fail over structured extraction when primary model returns persistently empty structured content, and surface `subfigure_decomposition_empty_response` in figure `parse_warnings` for observability.
- feature: add env-gated structured-extraction empty-response telemetry and `scripts/investigate_figure2_empty_responses.py` to run Figure 2 (PMID 39303722) experiment matrices with reproducibility reports.

## 2026-04-08 (v2.1.1)
- Documented BioWorkflow rollout controls and strict-mode fallback behavior across README and docs.
- Added explicit user-facing docs for:
  - `RESEARCHER_AI_BIOWORKFLOW_MODE` (`off`/`warn`/`on`)
  - `RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS`
  - strict terminal state `needs_human_review` with `human_review_summary`
  - vision fallback observability fields (`vision_fallback_count`, `vision_fallback_latency_seconds`)

## 2026-04-08 (v2.1.0)
- Completed BioWorkflow staged integration on `main`:
  - multimodal per-paper retrieval indexing (`PaperRAGStore`) with panel-level provenance
  - evidence validation stage with template priors
  - iterative retrieval refinement with circuit-breaker warnings and adaptive timeout controls
  - rollout gating and benchmark tooling for Phase 2/4 safety checks
- Added strict-mode graceful degradation path:
  - terminal `needs_human_review` state (no endless retry loop)
  - structured `human_review_summary` for manual triage

## 2026-04-07
- Adopted canonical architecture-planning document: `docs/WORKFLOW_GRAPH_IR_PLAN.md`.
- Marked `docs/previous/development/architecture/V2_ARCHITECTURE_PLAN.md` as deprecated (planning authority moved to IR plan).
- Marked `docs/previous/development/architecture/researcher-ai_architecture_review.md` as deprecated historical context.

## Architecture, Evaluation, Calibration, Improvement
- **BioC Integration Plan for Parsing Utilities** (BIOC_INTEGRATION_PLAN.md)
- **Architecture Evaluation: researcher-ai** (EVALUATION_ARCHITECTURE.md)
  Date: 2026-03-31
  **Status:** Fixed in this evaluation cycle. Added to both the import block and `__all__`.
- **Phase 1 Evaluation: Project Scaffold & Data Models** (EVALUATION_PHASE_1.md)
  Date: March 30, 2026  
- **Phase 2 Evaluation: Paper Parser** (EVALUATION_PHASE_2.md)
  Date: March 30, 2026  
- **Phase 3 Evaluation: Figure Parser** (EVALUATION_PHASE_3.md)
  Date: March 30, 2026
- **Phase 4 Evaluation: Methods Parser** (EVALUATION_PHASE_4.md)
  Date: 2026-03-30
- **Phase 5 Evaluation: Data Parsers (GEO & SRA)** (EVALUATION_PHASE_5.md)
  Date closed: 2026-03-31
  **Status: CLOSED**
- **Phase 6 Evaluation: Software Parser** (EVALUATION_PHASE_6.md)
  Date closed: 2026-03-31
  **Status: CLOSED**
- **Phase 7 Evaluation: Pipeline Builder** (EVALUATION_PHASE_7.md)
  Date closed: 2026-03-31
  **Status: CLOSED**
- **Phase 8 Evaluation: End-to-End Integration & MVP Validation** (EVALUATION_PHASE_8.md)
  Date closed: 2026-03-31
  **Status: CLOSED**
- **Figure Parser Calibration: Broad Strategy and MVP Roadmap** (FIGURE_PARSER_CALIBRATION.md)
- **Figure Parser Improvement Plan (PMC11633308)** (FIGURE_PARSER_IMPROVEMENT_PLAN.md)
- **Figure Parsing Roadmap** (ROADMAP.md)
- **BioC Confidence Benchmark (39303722)** (bioc_confidence_39303722_report.md)

## Tutorial and Usage Docs
- **Example: eCLIP-seq Pipeline Reproduction** (README.md)
- **How To Read The Function Docs** (how_to_read.md)

## Consolidation Outcome
- Current architecture reference: `docs/ARCHITECTURE.md`.
- Current tutorial reference: `docs/TUTORIAL.md`.
- Prior markdown is preserved under `docs/previous/` for auditability.
