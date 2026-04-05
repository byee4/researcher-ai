# Phase 4 Evaluation: Methods Parser

Date: 2026-03-30
Targets: `README.md` Phase 4 spec, `methods_parser.py`, `test_methods_parser.py`,
`test_methods_parser_snapshot.py`, `04_parse_methods.ipynb`

---

## Verification Metadata

```
Date    : 2026-03-30
Command : python -m pytest --tb=short -q
Python  : 3.10.12
pytest  : 9.0.2
Result  : 351 passed in 0.70s

Bootstrap (required before running):
    pip install -e ".[dev]"          # core deps + pytest/ruff/mypy/nbconvert
    pip install -e ".[all]"          # also installs jupyterlab

Breakdown:
    test_methods_parser.py          77 passed  (13 test classes, Tier 1)
    test_methods_parser_snapshot.py 14 passed  (1 fixture class, Tier 2)
    test_notebooks.py (Phase 4)      7 passed  (Tier 4)
    all other suites               253 passed
```

---

## Verdict

**Phase 4 is complete.** All four findings from the prior re-evaluation have been addressed.

---

## Findings

### Open

*(none)*

### Resolved

**1. [High] Verification claims not reproducible in reviewer's environment**
Critique: reviewer's local `pytest` invocation failed at import (missing `pydantic`, `nbformat`).
Analysis: `pydantic>=2.0` and `nbformat` are listed as core `[project.dependencies]` in
`pyproject.toml` — they are installed by any `pip install -e .`. The failure indicates the
reviewer ran `pytest` without first installing the package. This is a documentation gap, not a
code gap.
Fix: Added `nbconvert` to both `dev` and `jupyter` optional extras; added a unified `all` extra
(`pip install -e ".[all]"`). Verification metadata block now includes the explicit bootstrap
command so any reviewer can reproduce the run in a clean environment.
Rebuttal: The prior test count (335) was correct for that environment. The new count (351) reflects
16 additional tests added in this cycle (14 snapshot + 2 parse_warnings).

**2. [Medium] Snapshot integration testing deferred with no concrete timeline**
Critique: parser behaviour was vulnerable to API/schema drift with no frozen regression anchor.
Fix: Created `tests/snapshots/methods/pmid_26971820_eclip.yaml` — a fully self-contained
frozen fixture for the eCLIP paper (PMID 26971820) containing: frozen methods text excerpt,
per-schema frozen LLM responses, and structural expected_anchors. Created
`tests/test_methods_parser_snapshot.py` with 14 `@pytest.mark.snapshot` tests covering:
assay count, assay names, dependency count and edges, availability statements, parse_warnings
empty, step count and ordering, first/last step software, JSON round-trip, and DAG traversal.
All 14 pass without a live API key.

**3. [Medium] Dropped dependency edges were silently discarded with no downstream visibility**
Critique: `_identify_dependencies` logged a warning but the caller (`parse()`) had no channel
to surface this to users or downstream QA tooling.
Fix: `_identify_dependencies` now returns `(list[AssayDependency], list[str])` — a tuple of
accepted edges and warning strings. Dropped-edge warnings use the prefix `dependency_dropped:`
for machine-parseable filtering. `parse()` accumulates all warnings into `Method.parse_warnings`.
`Method` gained a new `parse_warnings: list[str]` field (default empty list, included in JSON
round-trip). Two new `TestIdentifyDependencies` tests updated to unpack the tuple; existing
`test_drops_dangling_edge` renamed `test_drops_dangling_edge_with_warning` and now asserts the
warning text.

**4. [Low] Assay stub fallback hid failure provenance**
Critique: stubs had `description="Could not be parsed."` but no structured reason.
Fix: `parse()` now formats stub failures as `assay_stub: {name!r} could not be parsed
({ExcType}: {msg})` and appends to `parse_warnings`. Two new `TestParse` tests verify: (a) a
clean parse produces an empty `parse_warnings`; (b) a single-assay LLM failure produces a
warning containing both `"assay_stub"` and the original exception message.

### Deferred

**Live snapshot refresh procedure**
The eCLIP fixture was curated manually from the published paper. A `make snapshot-refresh`
target (or CI job) that replays live API calls and updates the YAML is not yet implemented.
Deferred to the CI-hardening pass after Phase 5.

---

## Infrastructure Added This Cycle

| Artifact | Change |
|---|---|
| `researcher_ai/models/method.py` | `parse_warnings: list[str]` field on `Method` |
| `researcher_ai/parsers/methods_parser.py` | `_identify_dependencies` returns `(deps, warnings)`; `parse()` collects warnings; stub messages include exception type and text |
| `pyproject.toml` | `dev` and `jupyter` extras gain `nbconvert`; new `all` extra |
| `tests/snapshots/methods/pmid_26971820_eclip.yaml` | Frozen eCLIP fixture (methods text, LLM responses, anchors) |
| `tests/test_methods_parser_snapshot.py` | 14 snapshot tests (Tier 2) |
| `tests/test_methods_parser.py` | +2 parse_warnings tests; dependency tuple unpacking; renamed dangling-edge test |

---

## Exit Criteria — All Met

1. ✅ Bootstrap commands documented; any reviewer can reproduce the run with `pip install -e ".[dev]"`.
2. ✅ One frozen snapshot fixture + 14 snapshot tests provide Tier 2 regression coverage.
3. ✅ Dropped dependency edges surface as structured `dependency_dropped:` entries in `Method.parse_warnings`.
4. ✅ Stub assay failures surface as structured `assay_stub:` entries in `Method.parse_warnings`.
5. ✅ `parse_warnings` is empty for clean parses (asserted in test and snapshot).
6. ✅ 351/351 tests pass (77 unit + 14 snapshot + 7 notebook + 253 other).
7. ✅ `04_parse_methods.ipynb` executes cleanly (verified via nbconvert in prior cycle).
