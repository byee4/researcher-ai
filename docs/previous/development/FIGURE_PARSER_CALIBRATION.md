# Figure Parser Calibration: Broad Strategy and MVP Roadmap

## Problem
Figure parsing quality is sensitive to caption style, panel structure, and model drift. A single hardcoded PMCID override can stabilize one benchmark paper, but it does not provide scalable quality control across publications.

## Goals
1. Build a reusable calibration layer that can improve or correct parser outputs for many papers.
2. Keep base parser general-purpose while allowing targeted, versioned calibration rules.
3. Provide measurable quality gains on a benchmark set before broad rollout.
4. Maximize fidelity of extracted titles, captions, x/y-axis labels, and axis scales.
5. Provide fast UI correction paths when figure content cannot be confidently determined.

## Non-Goals (MVP)
- Fully automatic image-based plot recognition from pixels.
- End-to-end active learning infrastructure.
- Real-time human-in-the-loop annotation tooling.

## Calibration Architecture (Generalized)

### 1. Two-Stage Output Pipeline
- Stage A: Base parse (LLM + deterministic heuristics).
- Stage B: Calibration pass (rule registry + confidence-based corrections).

This avoids coupling benchmark fixes into core extraction logic and lets us evolve calibration independently.

Add a metadata-fidelity pass inside Stage A:
- title fallback from caption first sentence when LLM title is generic,
- axis label extraction from caption and in-text references,
- axis scale inference (`linear`, `categorical`, `log2`, `log10`, `ln`, `symlog`, `reversed`).

### 2. Calibration Registry
Create a versioned registry (YAML/JSON) with three rule scopes:
1. Global rules:
   - lexical cues, panel formatting rules, axis extraction normalization.
2. Family rules:
   - journal/format patterns (Nature-style panel captions, supplementary label styles).
3. Benchmark-specific rules:
   - PMCID/PMID-specific expectations for known gold papers.

Suggested path:
- `researcher_ai/calibration/figure_registry.yaml`

Suggested schema:
- `rule_id`
- `scope`: `global | family | paper`
- `match`:
  - `pmcid`, `pmid`, `journal`, `figure_id_pattern`, `panel_label`
- `conditions`:
  - cues in caption/context, current plot_type, confidence threshold
- `actions`:
  - set plot_type/category/layers/facets/axis labels
  - add evidence tag
  - adjust confidence
- `priority`
- `enabled`
- `version`

### 3. Calibration Engine
Implement `FigureCalibrationEngine` that:
1. Loads registry.
2. Resolves applicable rules by scope and priority.
3. Applies rules deterministically and logs all changes.
4. Emits provenance metadata per panel:
   - `original_plot_type`
   - `calibrated_plot_type`
   - `applied_rule_ids`
   - `calibration_version`

Suggested path:
- `researcher_ai/parsers/figure_calibration.py`

### 4. Confidence and Guardrails
Use calibration only when one or more holds:
- Base confidence below threshold (e.g., `<0.75`).
- Strong deterministic cue match.
- Benchmark paper exact match.

Guardrails:
- Never overwrite high-confidence output unless rule is benchmark-specific.
- Preserve original output in provenance for auditability.

### 5. Evaluation Harness
Build repeatable benchmark evaluation:
- Input: fixture set of papers + expected panel labels/types/axes.
- Output:
  - panel-level accuracy,
  - title/caption extraction quality,
  - axis-label extraction F1,
  - axis-scale accuracy,
  - per-plot-type precision/recall,
  - calibration delta vs baseline.

Suggested paths:
- `tests/calibration/test_figure_calibration.py`
- `tests/fixtures/figure_calibration/*.yaml`

## MVP Scope

### MVP Deliverables
1. Calibration registry support (global + paper scopes).
2. Calibration engine integrated after figure parse.
3. Provenance fields in output objects.
4. Figure uncertainty detection (`unknown plot`, `low confidence`, `missing axes`, etc.).
5. Portal UI ground-truth injector for figure/panel/title/caption/x-y axis labels/scales.
6. Benchmark fixtures for at least 5 papers, including PMC11633308.
7. CLI/test runner to compare baseline vs calibrated metrics.

### MVP Success Criteria
- >= 20% relative reduction in panel-type errors on benchmark set.
- >= 20% relative reduction in axis-label extraction errors on benchmark set.
- >= 15% relative reduction in axis-scale classification errors on benchmark set.
- No regression on existing `tests/test_figure_parser.py`.
- Deterministic reproducibility across runs (same inputs => same calibrated outputs).

## Roadmap

### Phase 0: Foundation (1-2 days)
1. Extract current hardcoded overrides into a calibration module.
2. Define registry schema and validation.
3. Add unit tests for registry parsing and rule application order.

### Phase 1: MVP Engine (2-4 days)
1. Implement `FigureCalibrationEngine` with global + paper scopes.
2. Integrate into `FigureParser.parse_figure` and `parse_all_figures`.
3. Add metadata-fidelity helpers (title fallback, axis label + scale inference).
4. Add provenance fields to subfigure metadata.
5. Migrate PMC11633308 rule into registry.

### Phase 2: Benchmarking (2-3 days)
1. Add benchmark fixtures (5 papers, panel-level truth tables).
2. Include truth fields for figure title/caption and axis labels/scales.
3. Build evaluation script (`baseline` vs `calibrated`).
4. Produce summary report in CI logs.

### Phase 3: MVP Hardening (2-3 days)
1. Add confidence gating and override guardrails.
2. Add feature flag/env toggle:
   - `RESEARCHER_AI_FIGURE_CALIBRATION=on|off`
   - `RESEARCHER_AI_FIGURE_CALIBRATION_CONFIDENCE_THRESHOLD=0.75` (default)
3. Add portal UI correction flows for unresolved figures and panel-level ground-truth injection.
4. Add docs for rule authoring and debugging.

### Rule Authoring Notes
- `scope` behavior:
  - `paper`: benchmark-specific and allowed to override high-confidence predictions.
  - `family`/`global`: protected by confidence guardrails by default.
- To override high-confidence outputs in non-paper scopes, set:
  - `actions.allow_high_confidence_override: true`

### Phase 4: Post-MVP Expansion
1. Add family/journal scoped rules.
2. Add optional human feedback loop to generate candidate rules.
3. Add image-assisted cues where captions are underspecified.

## Risks and Mitigations
1. Risk: Rule explosion and maintenance burden.
   - Mitigation: enforce schema, priority, ownership, and review process.
2. Risk: Overfitting benchmark papers.
   - Mitigation: confidence gating + separation of paper-specific vs global rules.
3. Risk: Silent behavior drift.
   - Mitigation: provenance logs + baseline-vs-calibrated CI checks.

## Immediate Next Steps
1. Create `figure_calibration.py` engine skeleton and registry loader.
2. Move current PMC11633308 override into `figure_registry.yaml`.
3. Add calibration engine unit tests and one integration test in `test_figure_parser.py`.
4. Add benchmark fixture format and first 2 papers.

---
This roadmap gets us from one-off overrides to a controlled, auditable, and extensible calibration system without sacrificing parser generality.
