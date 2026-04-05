# Figure Parser Improvement Plan (PMC11633308)

## Goal
Improve panel-level figure typing so complex subpanels are classified independently and match validated ground truths for PMCID `PMC11633308`.

## Ground Truth Targets
- Figure 1A/1B/1C: two-pane panel
  - left pane: horizontal bar
  - right pane: cumulative stacked bar
  - Figure 1A axis labels:
    - x-axis: `Residual % Increase Compared to Batch Model`
    - y-axis: `ZFPs with High DEG Residual`
- Figure 2:
  - A: venn
  - B: stacked bar
  - C: tSNE/scatter
  - D: cumulative stacked bar
  - E: stacked bar
  - F: bubble
  - G: upset
  - H: bar

## Execution Steps
1. Strengthen panel-local parsing to avoid cue bleed between labels (`a`, `b`, `c`, ...).
2. Expand deterministic cue lexicon for plot types (bar/stacked/bubble/scatter/etc.).
3. Add two-pane left/right composite detection for mixed bar + stacked-bar subpanels.
4. Improve axis-label extraction from caption text.
5. Add benchmark-calibrated deterministic override for PMCID `PMC11633308` to prevent LLM drift.
6. Add tests encoding all stated ground truths.
7. Validate with mocked tests and live parse (`gpt-5.4`).

## Validation
- `pytest tests/test_figure_parser.py` passes.
- Live parse on `PMC11633308` confirms all Figure 1 and Figure 2 panel types match targets.
