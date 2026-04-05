# Figure Parsing Roadmap

## Goal
Improve discrimination of complex scientific plot types while keeping parsing reliable and testable.

## Phase 1: Cue Coverage Expansion (Short Term)
- Expand lexical cue dictionaries for under-detected classes (oncoprint, alluvial variants, faceted heatmaps, chord/circos variants).
- Add 20-30 hard-example fixtures to `tests/test_figure_parser.py`.
- Success metric: reduce `PlotType.OTHER` rate on fixture set by at least 25%.

## Phase 2: Confidence and Fallback Policy
- Add parser-level thresholds for low-confidence classifications.
- Use top-N alternatives to drive safer fallback behavior in downstream code generation.
- Emit standardized warnings when confidence is below threshold.
- Success metric: no silent ambiguous classifications in test snapshots.

## Phase 3: Family-Specific Classifier Prompts
- Split LLM classification into broad-family routing then family-specific subtype prompts.
- Keep deterministic cue pass as first-stage prior and reconciliation layer.
- Add regression tests for family-routing edge cases.
- Success metric: improve subtype accuracy on complex classes without increasing failures.

## Phase 4: Benchmarking and Reporting
- Build a small benchmark script over curated figures (caption + in-text).
- Track per-class precision/recall and confusion matrix across releases.
- Add a CI gate for major regressions in complex classes.
- Success metric: stable trendline with measurable gains per release.

## Phase 5: Optional Vision Augmentation (Longer Term)
- Prototype image-assisted classification for panels where text cues are weak.
- Fuse text-based and vision-based votes with confidence weighting.
- Keep text-only path as default fallback.
- Success metric: improved classification on sparse-caption figures.

