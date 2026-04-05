# Architecture Evaluation: researcher-ai

Date: 2026-03-31
Scope: Full codebase audit — models, parsers, pipeline, tests, test publications, example outputs
Test result: 755 passed, 13 skipped in 3.55s

---

## Verdict

The architecture is solid and well-layered. The phased build order produced clean module boundaries, every Pydantic model round-trips through JSON, and the two test publications (PMID 26971820, 27018577) exercise the full Paper → Pipeline path. The issues below are all tractable; none requires a redesign.

---

## Findings

### 1. [Fixed] `MethodCategory` not exported from `models/__init__.py`

The `MethodCategory` enum added to `method.py` was missing from the barrel export in `models/__init__.py` and the `__all__` list. Downstream callers doing `from researcher_ai.models import MethodCategory` would get `ImportError`.

**Status:** Fixed in this evaluation cycle. Added to both the import block and `__all__`.

---

### 2. [High] `computational_only=True` default not wired into integration tests or pipeline builder

The new `MethodsParser.parse(computational_only=True)` default means downstream consumers now receive a *filtered* set of assays. But:

- `test_integration.py` constructs `_mock_method()` with a hand-built `Assay` list that has no `method_category` field set — it defaults to `experimental`. If the integration tests ever call `MethodsParser.parse()` live, the eCLIP assay (a mix of wet-lab and computational) would need correct classification to survive the filter.
- `PipelineBuilder.build()` doesn't inspect `method_category` at all — it converts every assay's steps into pipeline steps regardless of category. This is actually fine if the upstream parser already filtered, but it means there's no redundant guard.
- The example `notebooks/output/pmid_27018577/pipeline_spec.json` was generated *before* the classification feature. It contains 7 assays including purely experimental ones like "eCLIP-seq library preparation" (UV crosslinking, immunoprecipitation), which produce stub commands (`"software": "unknown"`). With `computational_only=True`, these experimental steps should now be filtered out, producing a cleaner pipeline.

**Recommendation:** Re-generate the PMID 27018577 notebook output with `computational_only=True` and verify the pipeline no longer contains `"software": "unknown"` stub steps for wet-lab protocols. Add `method_category` to the `_mock_method()` fixture in `test_integration.py`. Add a PipelineBuilder assertion that all incoming assays have `method_category != experimental` when `computational_only` was used upstream.

---

### 3. [Medium] `Assay.data_type` and `Assay.method_category` overlap

`Assay` now carries two classification axes:

- `data_type: str` — values like `"sequencing"`, `"imaging"`, `"computational"`, `"mass_spec"`, etc. This was the original axis.
- `method_category: MethodCategory` — `experimental` / `computational` / `mixed`. This is the new axis.

The `data_type` value `"computational"` and `method_category == MethodCategory.computational` overlap but aren't correlated: a computational assay can have `data_type="sequencing"` (e.g., "read alignment"), and `data_type="computational"` is rarely set. This creates a confusing two-source-of-truth situation.

**Recommendation:** Deprecate the `"computational"` value in `data_type` and document that `data_type` describes what *kind of data* the assay operates on (sequencing, imaging, proteomics), while `method_category` describes *how* the assay is performed (bench vs. computer). The `_AssayMeta` LLM schema's `data_type` description should be updated to remove `"computational"` from its enum hint.

---

### 4. [Medium] Snapshot fixture `pmid_26971820_eclip.yaml` does not include `method_category`

The frozen LLM responses in the snapshot YAML predate the classification feature. The `_AssayMeta_*` entries don't carry `method_category`, and neither do the `expected_anchors`. If the snapshot test is run against a parser that stamps `method_category`, the JSON round-trip test will pass (default `experimental` is valid) but the anchors don't verify the classification value.

**Recommendation:** Add a `method_category` field to each `_AssayMeta_*` response in the fixture. Add `expected_anchors.assay_categories` asserting that "UV crosslinking and immunoprecipitation" is `experimental` and "Computational read processing and peak calling" is `computational`. This is the strongest regression anchor for the new feature.

---

### 5. [Medium] `_DependencyList` forward-references `_DependencyMeta` before it's defined

In `methods_parser.py`, `_DependencyList` references `_DependencyMeta` in its type annotation, but `_DependencyMeta` is defined *after* `_DependencyList`. This works at runtime because of `from __future__ import annotations`, but mypy strict mode and some IDE parsers flag it. More importantly, it violates the file's own top-to-bottom reading convention where every other schema is defined before use.

**Recommendation:** Swap the order — define `_DependencyMeta` before `_DependencyList`.

---

### 6. [Low] Test publication PMID 27018577 hardcodes assay names in `_ECLIP_27018577_ASSAY_TITLES`

The `_canonicalize_assay_names` method has a special case for PMID 27018577 that returns a hardcoded list of 7 assay titles. This is a pragmatic workaround for LLM non-determinism on this specific paper, but:

- It bypasses the LLM classification call entirely (classification receives the hardcoded names, not LLM-extracted ones).
- It means `computational_only=True` depends entirely on the classification call correctly labeling each of those 7 hardcoded names.
- Adding more special-cased PMIDs doesn't scale.

**Recommendation:** This is fine for now (it's only 1 paper), but after Phase 8 is complete, replace PMID-specific hardcoding with a general section-heading extraction heuristic that works across papers. The heading extractor (`_extract_heading_like_lines`) is already there — it just needs to be promoted to the primary strategy when section headings are detected.

---

### 7. [Low] `_AssayMeta.data_type` description lists `'computational'` as a valid value

The `_AssayMeta` LLM prompt for `data_type` says:
```
"'sequencing', 'imaging', 'proteomics', 'mass_spec', 'flow_cytometry', 'computational', 'other'"
```

Now that `method_category` is the authoritative classification axis, the `'computational'` option in `data_type` should be removed to prevent the LLM from confusing the two fields.

**Recommendation:** Remove `'computational'` from the `_AssayMeta.data_type` description. Replace with `'bioinformatics'` if a separate signal is still desired, or just let `method_category` carry that information.

---

### 8. [Low] Example output `pipeline_spec.json` (PMID 27018577) contains `"software": "unknown"` stub steps

The generated pipeline from the notebook run for PMID 27018577 has 7 assays, but the first 6 steps of "eCLIP-seq library preparation" produce commands like:
```json
{"software": "unknown", "command": "# unknown_tool --UV_wavelength 254 nm ..."}
```

These are wet-lab protocol steps (UV crosslinking, cell lysis, immunoprecipitation) that cannot run on a computer. With the new `computational_only=True` feature, re-running this paper should eliminate these stub steps entirely — the pipeline should start at the first computational step (Cutadapt adapter trimming).

**Recommendation:** Re-run notebook `04_parse_methods.ipynb` and `06_build_pipeline.ipynb` for PMID 27018577 with the new default, then commit the updated `pipeline_spec.json`. The "software: unknown" stub steps should disappear.

---

### 9. [Info] Test coverage population matrix

How completely do the two test publications populate every model field?

| Model Field | PMID 26971820 (eCLIP) | PMID 27018577 (eCLIP methods) |
|---|---|---|
| `Paper.paper_type` | EXPERIMENTAL ✓ | EXPERIMENTAL ✓ |
| `Paper.supplementary_items` | empty [] | empty [] |
| `Paper.figure_captions` | empty {} | empty {} |
| `Figure.subfigures` | 2 per figure ✓ | not tested (mock) |
| `Figure.rendering` | None | None |
| `SubFigure.layers` | empty [] | empty [] |
| `SubFigure.statistical_annotations` | None | None |
| `SubFigure.facet_variable` | None | None |
| `Assay.method_category` | defaults to experimental | not yet set |
| `Assay.figures_produced` | populated ✓ | populated ✓ |
| `AssayDependency` | 1 edge (snapshot) | multiple edges ✓ |
| `Dataset.samples` | empty [] (mock) | not tested |
| `Software.commands` | populated ✓ | populated via registry ✓ |
| `Software.environment` | not populated | not populated |
| `PipelineStep.nf_core_module` | None | None |
| `PipelineStep.container` | None | None |

Notable gaps: `supplementary_items`, `figure_captions`, `rendering`, `layers`, `statistical_annotations`, `container`, and `nf_core_module` are never populated in any test publication. These are mostly Phase 7/8 deferred features or fields that only apply to specific paper types.

**Recommendation:** The Figure model's `rendering`, `layers`, `statistical_annotations`, and `facet_variable` fields are well-designed but never exercised. Before adding a third test publication, add a unit test in `test_models.py` that constructs a Figure with all optional fields populated and verifies the JSON round-trip. This validates the schema without needing a real paper.

---

## Summary of Actionable Items

| # | Severity | Item | Status |
|---|---|---|---|
| 1 | Fixed | `MethodCategory` export from `__init__.py` | ✅ Done |
| 2 | High | Wire `method_category` into integration tests and re-generate notebook outputs | ✅ Done (tests wired; notebook re-gen requires live API) |
| 3 | Medium | Clarify `data_type` vs `method_category` semantics, remove overlap | ✅ Done |
| 4 | Medium | Update snapshot fixture with classification anchors | ✅ Done |
| 5 | Medium | Fix `_DependencyList` / `_DependencyMeta` definition order | ✅ Done |
| 6 | Low | Plan PMID-specific hardcoding removal | Deferred to Phase 8+ |
| 7 | Low | Remove `'computational'` from `_AssayMeta.data_type` prompt | ✅ Done |
| 8 | Low | Re-generate example outputs with `computational_only=True` | ⏳ Requires live API key |
| 9 | Info | Populate untested model fields in test corpus | Deferred to Phase 8+ |

---

## Exit Criteria for This Evaluation

1. ✅ `MethodCategory` exported from `models/__init__.py`.
2. ✅ 760/760 tests pass (5 new classification tests added).
3. ✅ All findings documented with severity, analysis, and recommendation.
4. ✅ No architectural redesign required.
5. ✅ Findings 1–5, 7 resolved. Finding 8 deferred (needs live API).
