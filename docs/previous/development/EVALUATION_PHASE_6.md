# Phase 6 Evaluation: Software Parser

**Status: CLOSED**
Date closed: 2026-03-31
Re-evaluation date: 2026-03-31
Targets:
- `README.md` Phase 6 spec
- `researcher_ai/parsers/software_parser.py`
- `tests/test_software_parser.py`

---

## Verification Metadata

```
Date    : 2026-03-31
Command : . .venv/bin/activate && python -m pytest tests/test_software_parser.py --tb=short -q
Result  : 84 passed in 0.66s
```

---

## Final Verdict

**Phase 6 is complete and closeable.**

All material findings from the prior evaluation were either implemented directly in Phase 6 scope or explicitly deferred with clear scope boundaries.

---

## Findings Resolution

### 1) [High] LLM schema mismatch in license/alternative calls
**Status:** RESOLVED

- `_check_open_source()` now uses a narrow `_LicenseDecision` schema.
- `_find_alternative()` now uses a narrow `_AlternativeDecision` schema.
- Invalid LLM enum values are handled with explicit fallback to `UNKNOWN` and warning logs.

Impact: removes fragile required-field validation failures from single-field prompts.

### 2) [High] CLI command parsing missing
**Status:** RESOLVED

- Implemented `_extract_commands()`.
- `parse_from_method()` now passes `AnalysisStep` context into resolution.
- `_identify_tool()` now populates `Software.commands`.
- Command extraction supports:
  - Structured path from `AnalysisStep.parameters`.
  - LLM fallback from context text.

Impact: `Software.commands` is now populated and usable by downstream pipeline generation.

### 3) [Medium] Spec/implementation mismatch for external enrichment
**Status:** ACCEPTED AS DEFERRED

- External web search and GitHub license API enrichment are explicitly documented as deferred.
- Current implementation satisfies spec logic via registry + LLM inference path.

Rationale: keeps Phase 6 testability offline and avoids introducing live-network CI fragility.

### 4) [Medium] Alias normalization and dedup risk
**Status:** RESOLVED

- Added `_canonical_tool_name()` normalization.
- Deduplication in `_resolve_mentions()` now uses canonical keys.
- Registry lookup includes canonical alias matching.

Impact: collapses common variants like `Cell Ranger`, `cell-ranger`, `cellranger`.

### 5) [Medium] `_parse_github_code` / `_parse_notebook` are stubs
**Status:** ACCEPTED AS DEFERRED

- Deferred behavior is now explicitly documented in parser module comments.
- Scope boundary points this work to Phase 7 pipeline-driven code reference ingestion.

Rationale: non-blocking for core Phase 6 software extraction goals.

### 6) [Low] Docker image heuristic could fabricate invalid image names
**Status:** RESOLVED

- Removed heuristic `quay.io/biocontainers/{tool}` inference.
- Docker image now set only when explicitly known (registry/LLM-provided value).

Impact: reduces false environment artifacts and broken generated execution specs.

---

## Strengths Confirmed

- Strong unit coverage with fast runtime (`84 passed`).
- Registry-first behavior remains deterministic and robust.
- Graceful degradation on LLM/API failures maintained.
- Phase 6 now produces richer `Software` objects including executable command metadata.

---

## Deferred to Phase 7+

1. Full GitHub repository parsing in `_parse_github_code()`.
2. Full notebook content parsing in `_parse_notebook()`.
3. Optional external enrichment (GitHub license API/web lookup) with snapshot strategy.

These deferrals are acceptable and do not block Phase 6 closure.
