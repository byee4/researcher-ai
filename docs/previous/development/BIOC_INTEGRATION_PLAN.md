# BioC Integration Plan for Parsing Utilities

## Objective
Integrate BioC PMC JSON context into the parsing pipeline so figure and methods parsing are grounded in section-aware evidence, with measurable confidence-score improvements on PMID `39303722`.

## Documentation-Backed Data Contract
Sources reviewed:
- `https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/`
- `https://bioc.sourceforge.net/`
- `https://ftp.ncbi.nlm.nih.gov/pub/wilbur/BioC-PMC/pmc.key`
- `https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/39303722/unicode`

Confirmed contract points:
- REST pattern: `.../RESTful/pmcoa.cgi/BioC_[format]/[ID]/[encoding]`.
- `format` is `xml|json`; `encoding` is `unicode|ascii`.
- Use `unicode` as default because figure/panel cue regexes and caption parsing benefit from preserved punctuation and symbols.
- BioC JSON mirrors BioC XML hierarchy: `collection -> documents -> passages`.
- Core passage fields to consume: `offset`, `text`, `infons`.
- In `39303722` payload, top-level is an array of one collection.
- In `39303722`, `passages[*].infons.section_type` includes:
  - `FIG` (64 passages, IDs `F1..F7`, with `infons.file`, `infons.id`, `infons.type`)
  - `RESULTS` (43 passages, mostly `paragraph` + `title_*`)
  - `METHODS` (56 passages, mostly `paragraph` + `title_*`)

Additional normalization constraints:
- Root shape can be object, one-item array, or multi-collection array.
- Normalize to `collection_list` first, then select canonical collection by PMID/PMCID match in document IDs/infons.
- Do not assume `F1 -> Figure 1`; derive figure mapping from caption/title text when possible, then use ordinal fallback.

## Target Parser Utilities and Responsibilities
1. `researcher-ai/researcher_ai/utils/pubmed.py`
- Add canonical BioC fetch + normalize helpers.
- Keep robust handling for root JSON shapes: object, one-item array, and multi-collection array.

2. `researcher-ai/researcher_ai/parsers/paper_parser.py`
- Attach normalized BioC context to the parsed `Paper` object when PMID/PMCID is available.
- Preserve existing JATS-first behavior; BioC is enrichment, not replacement.

3. `researcher-ai/researcher_ai/parsers/figure_parser.py`
- Use BioC `FIG` + `RESULTS` passages as additional context for each figure.
- Use BioC evidence to refine panel-level confidence fields and composite confidence.

4. `researcher-ai/researcher_ai/parsers/methods_parser.py`
- Use BioC `METHODS` passages to strengthen methods text extraction and assay paragraph retrieval.

5. `researcher-ai/researcher_ai/parsers/software_parser.py`
- No direct BioC call; consume richer assay/step context produced by `MethodsParser`.

## Proposed Data Model Additions
1. Add lightweight BioC context models in `researcher-ai/researcher_ai/models/paper.py` (or a new `models/bioc.py`):
- `BioCPassageContext`: `section_type`, `type`, `text`, `offset`, `figure_id`, `file`.
- `BioCContext`: grouped lists for `fig`, `results`, `methods`, plus metadata (`pmid`, `pmcid`, `source_date`).
- Bound size for memory safety: cap stored passages per paper (e.g., max 200 kept in `Paper.bioc_context`; retain highest-priority passages first).
- `source_date` provenance rule: use HTTP `Last-Modified` when present; else BioC collection date; else fetch timestamp.

2. Add optional field on `Paper`:
- `bioc_context: Optional[BioCContext] = None`

3. Add structured optional field on `SubFigure` for traceability:
- `bioc_evidence_spans: list[BioCPassageContext] = []`
- Population contract:
  - Include passages that explicitly mention the subfigure/panel label (e.g., `Figure 3B`, `(B)` when anchored to the same figure).
  - If panel-specific passages are absent, include parent-figure passages and mark them as figure-level evidence in infons/metadata.
- Storage contract:
  - Store `bioc_evidence_spans` as references keyed by `(section_type, offset)` into `Paper.bioc_context` to avoid passage duplication.
  - Materialize full text only in benchmark/report export paths when needed.

## Implementation Phases

### Phase 1: BioC Ingestion Layer
1. Add `fetch_bioc_json_for_paper(pmid: Optional[str], pmcid: Optional[str], encoding: str = "unicode") -> dict`.
2. Resolution/fetch order:
- Try PMID endpoint first (empirical, not guaranteed by official docs; keep fallback mandatory).
- If empty/failure, resolve PMID -> PMCID (`resolve_pmid_to_pmcid_idconv` fallback to `resolve_pmid_to_pmcid`) and fetch using PMCID without `PMC` prefix.
3. Add root normalization:
- `normalize_bioc_collections(payload) -> list[dict]`
- `select_canonical_bioc_collection(collections, pmid, pmcid) -> dict`
4. Add passage extraction:
- `extract_bioc_passages(collection, section_selector)` where selector is case-insensitive contains matcher.
- Methods aliases: `METHOD`, `METHODS`, `MATERIALS`, `MATERIALS_AND_METHODS`, `METHOD_DETAILS`, `STAR METHODS`.
- Results aliases: `RESULTS`, `RESULTS_DISCUSSION`, `RESULTS|DISCUSSION`.
5. Add figure mapping:
- Primary: parse leading `Figure N` / `Fig. N` token from FIG caption/title text.
- Secondary fallback:
  - Parse literal digit from `infons.id` (`F3 -> Figure 3`) when caption token is unavailable.
  - If `F#` digits are non-sequential/irregular for the article, fall back to FIG-passage index order and emit a warning.
6. Add cache + control plane:
- Feature flag: `RESEARCHER_AI_BIOC_ENABLED` (default `on`).
- Disk cache key: `(pmid_or_pmcid, encoding)` in cache dir.
- TTL env var: `RESEARCHER_AI_BIOC_CACHE_TTL_SEC`.
- Reuse existing `_get` timeout/retry/backoff behavior in `pubmed.py`.
- Flag precedence rule: when `RESEARCHER_AI_BIOC_ENABLED=off`, skip network and skip cache reads/writes.
- Thread/process safety: use atomic cache writes (`tempfile` + `os.replace`).

Deliverable:
- Deterministic fetch/normalize/select utilities with cache, feature flag, and unit tests using saved fixtures.

### Phase 2: PaperParser Enrichment
1. During PMID/PMCID parse path, fetch BioC once and attach normalized context to `Paper`.
2. If BioC fetch fails, continue without failure and log at debug/warn level.
3. Keep existing `figure_ids`, `figure_captions`, `sections` behavior unchanged.

Deliverable:
- Backward-compatible `Paper` outputs with optional BioC context.

### Phase 3: FigureParser Context Fusion (`FIG` + `RESULTS`)
1. Build figure-local context bundle:
- Existing: caption + in-text refs from paper sections.
- New: BioC `FIG` title/caption passages for same figure ID.
- New: BioC `RESULTS` paragraphs mentioning the figure, where mention is defined by the same alias regex family as `_build_fig_ref_pattern` (`Figure N`, `Fig. N`, `Fig N[A-Z]`, panel references).
- Cap appended RESULTS passages per figure (e.g., top `N=8` by mention count, then shortest-distance offset to matching FIG passage) to avoid prompt bloat.
2. Feed fused context to:
- `_decompose_subfigures`
- `_determine_purpose`
- `_identify_methods` (extend signature or add wrapper; do not silently ignore extra context)
- `_disambiguate_subfigure_plot`
3. Add confidence refinement function:
- Inputs: `existing_composite_confidence` (0-100), `confidence_scores`, BioC evidence coverage, cue agreement/contradiction.
- Reuse existing `_composite_from_scores(confidence_scores)` as the weighted field composite.
- Define `distinct_plot_cues` as unique matched cues from `_infer_plot_type_candidates` found in BioC evidence text for that subfigure.
- Compute:
  - `base = max(existing_composite, _composite_from_scores(scores))`
  - `raw_bonus = min(15, 2*evidence_passage_hits + 3*len(distinct_plot_cues))`
  - `bonus = min(raw_bonus, 100 - base)` (headroom-gated)
  - `contradiction_factor = 0.85` if BioC evidence contradicts chosen `plot_type`, else `1.0`
  - `new_composite = clamp((base + bonus) * contradiction_factor, 0, 100)`
- Synchronization rule (mandatory):
  - `classification_confidence = round(new_composite / 100.0, 3)` (0-1 only)
  - Never set `classification_confidence` independently of `composite_confidence`.

Deliverable:
- Figure confidence scores become evidence-aware rather than prompt-only.

### Phase 4: MethodsParser Context Fusion (`METHODS`)
1. Extend `_extract_methods_text` to append/merge BioC `METHODS` passages when available.
2. Prioritize BioC method paragraphs when selecting assay-specific paragraphs.
3. Keep `computational_only` filtering unchanged, but improve upstream assay extraction quality via richer context.

Deliverable:
- Better assay paragraph grounding and fewer low-information assay stubs.

### Phase 5: SoftwareParser Verification (Concrete)
1. Add regression test using `39303722`-style enriched assay steps.
2. Validate that `SoftwareParser.parse_from_method` extracts additional or better-grounded tool mentions from enriched `AnalysisStep` text.
3. Pass condition:
- At least one new `Software` item is surfaced, OR
- At least one existing tool gains new grounding metadata (non-empty version/url/citation-equivalent field vs baseline).

Deliverable:
- Testable improvement criterion for downstream effect, not a no-op statement.

## Test Plan (Including PMID 39303722 Confidence Benchmark)

### A. Fixtures
1. Save endpoint output to:
- `researcher-ai/tests/fixtures/bioc/pmid_39303722_bioc_unicode.json`
2. Add compact derived fixture for faster unit tests (optional).
3. Add negative fixtures:
- BioC unavailable (`404`/empty payload) fixture.
- Contradictory cue fixture where BioC evidence conflicts with selected plot type.

### B. Unit Tests
1. `tests/test_pubmed_bioc_context.py`
- Root-shape normalization (array vs object).
- Canonical collection selection from multi-collection arrays.
- Passage extraction by section type.
- Section alias handling (`METHOD*`, `RESULTS*`).
- Figure mapping from caption token, with ordinal fallback.
2. Cache/feature-flag tests:
- `RESEARCHER_AI_BIOC_ENABLED=off` disables fetch path.
- TTL cache hit/miss behavior.
- Verify `RESEARCHER_AI_BIOC_ENABLED=off` also bypasses cache reads (no stale cached payload use).
3. `tests/test_paper_parser.py`
- Paper includes BioC context when available.
- Graceful fallback when BioC endpoint fails.
4. `tests/test_figure_parser.py`
- BioC FIG/RESULTS context is included in figure-local parsing input.
- Confidence refinement increases/penalizes as expected for controlled cues.
- Contradiction branch applies multiplicative penalty.
- `classification_confidence` always equals `round(composite_confidence / 100.0, 3)`.
5. `tests/test_methods_parser.py`
- BioC METHODS passages influence `_extract_methods_text` and `_extract_assay_paragraph`.

### C. Confidence Benchmark for `39303722`
1. Baseline run:
- Parse `39303722` with BioC context disabled.
2. Enhanced run:
- Parse `39303722` with BioC context enabled.
3. Compare metrics:
- Mean and median `subfigure.composite_confidence`.
- Count of subfigures with `classification_confidence >= 0.75`.
- Count of ambiguous panels (`composite_confidence` in `40-74` range).
- Contradiction count (defined): number of subfigures where BioC cue set conflicts with chosen `plot_type` under refinement rules.
4. Report output:
- Create benchmark dir if missing: `researcher-ai/parse_results/bioc/`
- `researcher-ai/parse_results/bioc/bioc_confidence_39303722_report.json`
- `researcher-ai/parse_results/bioc/bioc_confidence_39303722_report.md`
- Include explicit fields: `contradiction_count_baseline`, `contradiction_count_enhanced`, and per-subfigure contradiction flags.
5. Golden diff artifacts:
- Baseline and enhanced serialized figure outputs.
- Human-readable diff file for `Figure`/`SubFigure` JSON changes.

## Acceptance Criteria
1. All existing parser tests pass unchanged.
2. New BioC tests pass with deterministic fixtures.
3. `39303722` benchmark meets quantitative gates:
- Mean `composite_confidence` delta >= `+3.0` points.
- Ambiguous-panel count drops by >= `10%` OR absolute decrease is >= `2` panels.
- Contradiction count is non-increasing vs baseline.
4. Pipeline remains backward compatible when BioC fetch is unavailable.

## Execution Checklist
- [ ] Implement BioC normalization helpers in `utils/pubmed.py`.
- [ ] Implement root collection selection logic for multi-collection payloads.
- [ ] Implement disk cache + TTL + `RESEARCHER_AI_BIOC_ENABLED` feature flag.
- [ ] Add `Paper` BioC context model fields.
- [ ] Wire BioC context into `PaperParser`.
- [ ] Fuse `FIG` + `RESULTS` into `FigureParser` context path.
- [ ] Implement RESULTS mention matching using `_build_fig_ref_pattern`-compatible aliases.
- [ ] Add confidence refinement logic in `FigureParser`.
- [ ] Enforce `classification_confidence = composite_confidence/100` synchronization.
- [ ] Fuse `METHODS` into `MethodsParser`.
- [ ] Add fixtures and tests.
- [ ] Add contradiction-penalty unit test with synthetic passages.
- [ ] Add BioC 404/unavailable fallback-path test.
- [ ] Run `pytest` for parser test suites.
- [ ] Run `39303722` baseline vs enhanced benchmark, assert quantitative gates, and publish snapshot diffs.

## Risks and Mitigations
1. Risk: BioC payload shape differences across articles.
- Mitigation: normalize object, one-item array, and multi-collection array roots; tolerate missing keys.
2. Risk: Confidence inflation without true quality gains.
- Mitigation: headroom-gated bonus, contradiction multiplicative penalty, and quantitative benchmark thresholds.
3. Risk: Runtime/network overhead.
- Mitigation: cache BioC responses with TTL and make enrichment optional via feature flag.
