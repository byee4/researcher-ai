# Phase 3 Evaluation: Figure Parser

Date: March 30, 2026
Targets: Phase 3 spec in `README.md`, `figure_parser.py`, `test_figure_parser.py`, `03_parse_figures.ipynb`

---

## Verdict

**Phase 3 is complete.** All spec deliverables exist and pass. 91 unit tests pass (80 → 91 over the course of this phase). Notebook executes cleanly. All evaluation findings below are Resolved.

---

## Findings

### Open

*(none)*

### Resolved

**[High] `_build_fig_ref_pattern` applied supplementary prefix unconditionally**
Both branches of the `if is_supp` check returned `r"(?:Supplementary\s+)?"`, so a plain `"Figure 1"` pattern would also match `"Supplementary Figure 1"`.
Fix: non-supplementary path now uses negative lookbehinds `(?<!supplementary )(?<!supp\. )`.
Tests: `TestBuildFigRefPatternSupplementary` (4 tests).

**[High] `_extract_caption_from_text` did not preserve S-prefix for supplementary labels**
`re.search(r"(\d+)", figure_id)` extracted only digits, so `"Supplementary Figure S1"` degraded to matching `"Figure 1"` style patterns and would miss explicit `S1` captions.
Fix: changed to `r"([Ss]?\d+)"` to preserve the S-prefix; also updated the next-figure boundary regex to handle `[Ss]?\d+`.
Tests: `TestExtractCaptionSupplementarySPrefix` (5 tests), `TestExtractCaptionTwoDigitFigure` (4 tests).

**[High] `_extract_caption_from_text` had no word boundary after figure number**
`"Figure 1"` could greedily match `"Figure 10"` captions.
Fix: added `\b` after the figure number in the caption pattern.
Tests: `TestExtractCaptionTwoDigitFigure`.

**[Medium] `import math` placed inside method body**
`_infer_layout()` contained an inline `import math`.
Fix: moved to module top-level imports.

**[Medium] `_identify_methods` truncated in-text context to first 10 sentences**
`" ".join(in_text[:10])` silently dropped method mentions from later sentences in dense papers.
Fix: uses full `in_text` list; character cap on the LLM prompt input enforces the budget.
Tests: `TestIdentifyMethods::test_uses_full_in_text_not_truncated`.

**[Medium] README Step 3.3 and project tree referenced wrong notebook name**
Step 3.3 said `02_parse_figures.ipynb`; project tree showed the original pre-Phase-1 numbering.
Fix: Step 3.3 updated to `03_parse_figures.ipynb`; project tree updated to `01_data_models / 02_parse_paper / 03_parse_figures / 04_parse_methods … 07_end_to_end`.

**[Low] `parse_all_figures` duplicated caption/in-text scan**
Caption and in-text extraction happened twice per figure: once for the stub fallback, once inside `parse_figure`.
Fix: extracted `_parse_figure_from_context(figure_id, caption, in_text)` as a private method; `parse_all_figures` passes precomputed context through; `parse_figure` delegates to it after computing context once.
Tests: `TestParseAllFiguresNoDuplicateScan` (2 tests).

**[Low] `_identify_datasets` short-circuited on regex hit, missing non-standard prose IDs**
When regex found accessions, LLM was never called even if additional non-standard identifiers existed in prose.
Fix: added `strict_regex_only: bool = True` parameter. Default preserves existing behaviour. When `False`, LLM always runs and results are merged/deduped with regex hits; on LLM failure, regex results are still returned.
Tests: `TestIdentifyDatasetsStrictMode` (4 tests).

**[Low] `_stub_figure` discarded precomputed caption and in-text on failure**
Stubs returned on LLM failure had empty `caption` and `in_text_context`.
Fix: `_stub_figure` accepts optional `caption` and `in_text` kwargs; `parse_all_figures` passes the already-computed values.
Tests: `TestStubFigure::test_stub_preserves_caption_when_provided`, `test_stub_preserves_in_text_when_provided`, `test_stub_defaults_to_empty_in_text`.

**[Low] No `TestIdentifyMethods` test class**
Methods extraction had no dedicated unit tests; coverage came only from integration-level mocks.
Fix: added `TestIdentifyMethods` (4 tests: normal call, empty text, LLM failure, full-context regression).

**[Medium] Evaluation files contained stale findings after fixes were applied**
Prior iteration of this file listed already-fixed bugs as open.
Fix: this file now uses Open / Resolved / Deferred sections with dates.

### Deferred

**Snapshot integration tests for figure parser**
`tests/snapshots/` directory and `manifest.yaml` now exist with eCLIP paper fixture metadata, but frozen API and LLM response artifacts are not yet populated. Populating them requires a live run (`pytest -m live --refresh-snapshots`). Deferred to CI-hardening pass after Phase 5.

---

## Infrastructure Added (Phase 3 close-out)

- `pyproject.toml`: registered `snapshot` and `live` pytest markers; added `filterwarnings` to suppress `MissingIDFieldWarning` from nbformat.
- `tests/snapshots/manifest.yaml`: fixture registry for eCLIP (PMID 26971820) and ENCODE atlas (PMID 30423142) papers; directory structure created for `api_responses/` and `llm_responses/`.
- `tests/test_notebooks.py`: Tier 4 notebook shape validation for Phases 1–4 using `nbformat`; Phase 4 tests auto-skip until `04_parse_methods.ipynb` is created.

---

## Exit Criteria — All Met

1. ✅ `_build_fig_ref_pattern` non-supplementary path excludes supplementary references.
2. ✅ `_extract_caption_from_text` handles S-prefix labels and two-digit figure numbers with word boundary.
3. ✅ README notebook tree and Step 3.3 reference `03_parse_figures.ipynb`.
4. ✅ `_identify_datasets` has `strict_regex_only` mode.
5. ✅ `parse_all_figures` eliminates duplicate caption/in-text scan via `_parse_figure_from_context`.
6. ✅ `TestIdentifyMethods` (4 tests), `TestExtractCaptionSupplementarySPrefix` (5 tests), `TestIdentifyDatasetsStrictMode` (4 tests), `TestParseAllFiguresNoDuplicateScan` (2 tests) added.
7. ✅ 91/91 figure parser tests pass; 234/234 full suite passes.
8. ✅ `03_parse_figures.ipynb` executes cleanly (verified via nbconvert).
9. ✅ Pytest `snapshot`/`live` marks registered; `test_notebooks.py` passing (19 pass, 7 skip pending Phase 4 notebook).
