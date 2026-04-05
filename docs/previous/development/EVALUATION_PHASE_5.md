# Phase 5 Evaluation: Data Parsers (GEO & SRA)

**Status: CLOSED**
Date closed: 2026-03-31
Re-evaluation date: 2026-03-30
Implemented files:
- [`researcher_ai/parsers/data/base.py`](researcher_ai/parsers/data/base.py)
- [`researcher_ai/parsers/data/geo_parser.py`](researcher_ai/parsers/data/geo_parser.py)
- [`researcher_ai/parsers/data/sra_parser.py`](researcher_ai/parsers/data/sra_parser.py)
- [`tests/test_geo_parser.py`](tests/test_geo_parser.py)
- [`tests/test_sra_parser.py`](tests/test_sra_parser.py)
- [`tests/conftest.py`](tests/conftest.py)
- [`notebooks/05_parse_data.ipynb`](notebooks/05_parse_data.ipynb)

---

## Exit Criteria — All Met

1. **README Step 5.5 notebook name** — confirmed correct as `05_parse_data.ipynb` at README:1435.
   Already resolved before close.

2. **SuperSeries recursion** — resolved with `parse(accession, recursive: bool = False)`.
   Shallow parse by default; bounded one-level-deep recursive mode available when `recursive=True`.
   Spec updated in `geo_parser.py` docstring; README already reflects the `recursive` parameter.

3. **Evaluation doc scoped to Phase 5** — this document (previously contained stale
   `MethodsParser._parse_assay` references); rewritten clean.

4. **GPL normalization hardened** — `_normalise_gpl()` guards against double-prefixing:
   tokens already starting with `GPL` are returned as-is; otherwise `GPL` is prepended.
   Covered by `test_gpl_already_prefixed_not_doubled`.

5. **`_gse_to_srp` fallback implemented** — two-path strategy:
   (1) pysradb `gse_to_srp()` fast path; (2) NCBI elink `db=gds → db=sra` + esummary SRP
   extraction as fallback. Both paths have exception guards; `None` returned on total failure.

---

## Additional Fix Applied at Close

**`GEOParser` lazy HTTP client** — `httpx.Client` was instantiated eagerly in `__init__`,
causing `ImportError: socksio not installed` in SOCKS-proxy environments (including the
Cowork sandbox) even for tests that make no network calls.  Fixed by lazy-loading the client
via a `client` property, consistent with `SRAParser._db`.  All 102 unit tests now pass
without any proxy-related errors.

**`tests/conftest.py` added** — `@pytest.mark.live` tests are now automatically skipped
unless `pytest --run-live` is passed.  Eliminates confusing failures in offline/sandbox
environments; the two live tests are properly deselected in all default runs.

---

## Test Results at Close

```
pytest tests/test_geo_parser.py tests/test_sra_parser.py tests/test_notebooks.py
135 passed, 2 skipped [live]
```

---

## What Was Built

### `BaseDataParser` (base.py)
Abstract interface with `parse(accession) -> Dataset` and `validate_accession(accession) -> bool`.

### `GEOParser` (geo_parser.py)
- Handles GSE (Series + SuperSeries), GSM (Sample), GPL (Platform).
- SuperSeries detection via `gdstype` field; child IDs extracted from `extrelations`.
- SRA bridging: pysradb `gse_to_srp()` fast path + NCBI elink fallback.
- Processed-data FTP URLs constructed deterministically (no live fetch).
- Resilience: all network helpers return stubs on failure.
- Lazy `httpx.Client` via property (defers proxy configuration until first use).

### `SRAParser` (sra_parser.py)
- Auto-detects level from prefix: SRP/ERP/DRP → project, SRX/ERX/DRX → experiment, SRR/ERR/DRR → run.
- Covers NCBI, EBI (ENA), and DDBJ accession families.
- Lazy pysradb `SRAweb` client via `_db` property.
- Network failures return stub `SRADataset` without raising.

### Notebook `05_parse_data.ipynb`
9 cells: accession validation, esummary normalisation, SuperSeries detection, mocked GEO parse,
SRA helper functions, mocked SRA parse (SRP/SRX/SRR), JSON round-trip, live GEO demo,
live SRA demo. Cells 1–7 require no network; live cells auto-skip when offline.
