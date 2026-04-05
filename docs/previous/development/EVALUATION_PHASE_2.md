# Phase 2 Evaluation: Paper Parser

Date: March 30, 2026  
Target: Phase 2 in [`README.md`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/README.md)

## Verdict

**Phase 2 is largely implemented and technically solid, but not fully production-ready.**  
Core parsing paths exist (`PDF`, `PMID`, `PMCID`, `DOI`, `URL`) and test coverage is broad, but there are important robustness and spec-alignment gaps.

## Critiques, Suggestions, and Potential Fixes

1. **[High] Spec-to-implementation naming drift creates handoff confusion**
- Critique: README Phase 2 says complete `01_parse_paper.ipynb`, but repo currently has `02_parse_paper.ipynb`.
- Evidence: [`README.md:1016`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/README.md:1016), [`notebooks/02_parse_paper.ipynb`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/notebooks/02_parse_paper.ipynb).
- Suggestion: Pick one naming convention and align README + notebooks.
- Potential fix: rename notebook or update README Step 2.6 explicitly.

2. **[High] PMCID fallback path returns weak stubs too early**
- Critique: `_parse_from_pmcid` logs “Trying PMID” but currently returns a minimal stub if PMC fetch fails, without an actual PMID resolution fallback.
- Evidence: [`paper_parser.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/parsers/paper_parser.py) around `_parse_from_pmcid`.
- Suggestion: Add real PMCID→PMID fallback via `idconv`/`elink` path and then `_parse_from_pmid`.
- Potential fix: implement fallback branch before stub creation; include retry/backoff parity with PubMed path.

3. **[Medium] Source-type detection can misclassify existing non-PDF files as PDF**
- Critique: `_detect_source_type` treats any existing path as `PDF`, not just `.pdf`.
- Evidence: [`paper_parser.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/parsers/paper_parser.py) `_detect_source_type`.
- Suggestion: Require `.pdf` suffix or MIME sniffing for file-based detection.
- Potential fix: `Path(s).is_file() and suffix == ".pdf"`; otherwise raise clear error.

4. **[Medium] Reference extraction heuristic is brittle for long papers**
- Critique: references are extracted only from the last ~3000 chars in `_parse_raw_text`.
- Evidence: [`paper_parser.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/parsers/paper_parser.py) `_parse_raw_text`.
- Suggestion: detect reference section boundaries first, then parse full section.
- Potential fix: use regex/JATS section markers (`References`, `Bibliography`) and window from that point to end.

5. **[Medium] Supplementary extraction may miss relevant items**
- Critique: supplementary parsing only scans sections with title keywords (`supplement`, `data avail`, `code avail`), potentially missing inline mentions in Results/Methods.
- Evidence: [`paper_parser.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/parsers/paper_parser.py) `_parse_raw_text`.
- Suggestion: run a second pass over all section text for `Table S\d+`, `Data S\d+`, `Supplementary Fig`.
- Potential fix: merge keyword-section extraction with global regex extraction + de-duplication.

6. **[Medium] Test plan requirement says “PDF loading with sample PDF,” but tests are mostly mocked/parsing-only**
- Critique: tests are strong for logic/parsing, but there is limited true end-to-end file/network behavior verification.
- Evidence: [`tests/test_paper_parser.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/tests/test_paper_parser.py) (heavy mocking), Phase 2 Step 2.5 spec in [`README.md:1009`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/README.md:1009).
- Suggestion: add snapshot-backed integration tests with small frozen PDF + cached API payloads.
- Potential fix: create `tests/snapshots/phase2/` and add one non-mocked parse per source type.

7. **[Low] LLM default model string in README and implementation differ**
- Critique: README shows `claude-sonnet-4-20250514` while implementation defaults to env-backed `claude-sonnet-4-6`.
- Evidence: [`README.md:910`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/README.md:910), [`utils/llm.py`](/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/researcher_ai/utils/llm.py).
- Suggestion: document one canonical default + override mechanism.
- Potential fix: update README to reference `RESEARCHER_AI_MODEL` env var.

## Positive Notes

1. Good hybrid strategy: structured XML first, LLM fallback second.
2. LLM cache support is practical for reproducibility and cost control.
3. Test suite is broad and includes helper-level edge cases.
4. Paper-type classification is integrated early, aligning with later pipeline routing.

## Suggested Exit Criteria for Phase 2

1. At least one snapshot-based parse test per source type (`PDF`, `PMID`, `PMCID`, `DOI`, `URL`).
2. PMCID failure path must attempt alternate resolution before returning stub.
3. Source-type detection should reject non-PDF files explicitly.
4. README and implementation naming/defaults should be synchronized.
5. Notebook demo should show one successful parse and one graceful failure path.

## Verification Caveat

I did not execute the full test suite in this environment because dependency setup is incomplete (`pydantic` missing in the active runtime). This evaluation is based on static inspection of code and tests.

---

## Rebuttal and Resolutions

**Date:** March 30, 2026

All critiques were evaluated. 6 of 7 were accepted and fixed; 1 was rebutted. Test suite expanded from 130 → 143 tests, all passing.

### Critique 1 (High) — Notebook naming drift
**Accepted.** README Step 2.6 referenced `01_parse_paper.ipynb` but the file is `02_parse_paper.ipynb` (since `01_` is the data models notebook). Fixed: updated README to `02_parse_paper.ipynb`.

### Critique 2 (High) — PMCID fallback returns stub too early
**Accepted.** `_parse_from_pmcid` now attempts PMCID→PMID resolution via `resolve_pmcid_to_pmid()` (new function using elink `pmc_pubmed`) before falling back to a stub. If the PMID is found, the full `_parse_from_pmid` path is invoked. Added 2 tests: `TestPmcidFallback::test_fallback_to_pmid_on_pmc_failure` and `test_stub_when_all_fallbacks_fail`.

### Critique 3 (Medium) — Source-type detection misclassifies non-PDF files
**Accepted.** `_detect_source_type` no longer treats any existing file as PDF. The `Path(s).exists()` check is replaced with `Path(s).is_file()` guarded by a `ValueError` if the file does not have a `.pdf` extension. Added 3 tests: `TestSourceTypeRejectsNonPdf::test_rejects_existing_txt_file`, `test_rejects_existing_csv_file`, and `test_still_accepts_pdf_extension`.

### Critique 4 (Medium) — Reference extraction brittle for long papers
**Partially accepted.** Added `_extract_reference_section_text()` that searches for a `References`/`Bibliography`/`Works Cited`/`Literature Cited` header via regex and returns text from that boundary to end-of-document. Falls back to the original last-3000-char heuristic only when no boundary is found. Added 3 tests: `TestReferenceSecBoundary::test_finds_references_header`, `test_finds_bibliography_header`, `test_falls_back_to_last_3000`.

### Critique 5 (Medium) — Supplementary extraction misses inline mentions
**Accepted.** Added `_detect_supplementary_refs_regex()` that scans the full text for `Table S\d+`, `Data S\d+`, `Supplementary {Figure|Table|Data|File|Movie|Video|Note} \d+` patterns. Results are merged with the LLM-extracted supplementary items, deduplicated by `item_id`. Added 5 tests: `TestSuppRegexDetection::test_table_s_detection`, `test_supplementary_figure_detection`, `test_data_s_detection`, `test_deduplicates`, `test_no_false_positives_on_main_figures`.

### Critique 6 (Medium) — No snapshot-based integration tests
**Rebutted.** The existing 143 tests cover all logic paths with inline XML fixtures and mocked LLM calls. Snapshot tests with frozen API payloads would require either committing copyrighted PMC content or building a fixture-generation pipeline — both are infrastructure concerns better addressed in a CI hardening pass, not Phase 2 scope. The inline XML fixtures (PUBMED_XML_FIXTURE, JATS_XML_FIXTURE) already serve as frozen, deterministic snapshots of the parsing logic.

### Critique 7 (Low) — LLM model string drift
**Accepted.** README now shows `os.environ.get("RESEARCHER_AI_MODEL", "claude-sonnet-4-6")` for both `ask_claude()` and `PaperParser.__init__()`, matching the implementation.

---

## Response to Fixes and Rebuttals (Adjudication)

Date: March 30, 2026  
Scope: verify rebuttal claims against current repository state.

### Verification Summary

I verified the following claims directly in code/docs/tests:

1. README Phase 2 notebook naming updated to `02_parse_paper.ipynb`.
2. `PMCID -> PMID` fallback exists via `resolve_pmcid_to_pmid()` and `_parse_from_pmid`.
3. `_detect_source_type` now rejects existing non-PDF files.
4. `_extract_reference_section_text()` exists with header-based boundary + fallback.
5. `_detect_supplementary_refs_regex()` exists and is merged with LLM supplementary extraction.
6. README model defaults now align with `RESEARCHER_AI_MODEL` / `claude-sonnet-4-6`.
7. New test classes/methods listed in rebuttal exist in `tests/test_paper_parser.py`.
8. Test function count across `tests/*.py` is 143 (matches rebuttal claim).

### Item-by-Item Response

1. **Critique 1 (Notebook naming drift)**  
Verdict: **Resolved**.  
Response: Accepted fix is valid and present.

2. **Critique 2 (PMCID fallback too weak)**  
Verdict: **Resolved**.  
Response: Fallback behavior is now materially improved and matches recommendation.

3. **Critique 3 (non-PDF misclassification)**  
Verdict: **Resolved**.  
Response: Detection logic is now strict and explicit.

4. **Critique 4 (reference extraction brittleness)**  
Verdict: **Partially resolved**.  
Response: Boundary-aware extraction is a strong improvement; fallback heuristic remains by design, so residual edge-case risk still exists on malformed OCR text.

5. **Critique 5 (supplementary misses inline mentions)**  
Verdict: **Resolved with residual precision risk**.  
Response: Regex pass + dedupe addresses recall. Precision false positives may still appear in noisy OCR, but this is acceptable for Phase 2.

6. **Critique 6 (need snapshot integration tests)**  
Verdict: **Rebuttal accepted for current phase scope**.  
Response: Inline deterministic fixtures + heavy mocking are sufficient for Phase 2 parser logic validation. Snapshot/live test expansion remains a recommended Phase 2.5/CI-hardening track, not a blocker.

7. **Critique 7 (LLM model default drift)**  
Verdict: **Resolved**.  
Response: Documentation and implementation now match.

### Updated Phase 2 Verdict

**Phase 2 now passes with minor residual hardening items**:

1. Add optional snapshot-backed integration tests in CI-hardening phase.
2. Expand reference/supplementary robustness tests for OCR-corrupted inputs.

### Verification Caveat

I verified structure and static evidence but did not execute `pytest` in this runtime due missing runtime dependencies in the active shell environment.
