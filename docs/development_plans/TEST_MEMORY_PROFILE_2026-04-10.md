# Test Memory Profiling Findings (2026-04-10)

## Goal

Profile memory usage across the full pytest suite and enforce a 4 GiB safety threshold.

## Method

We ran every collected pytest node (`1115` tests) in isolation using:

```bash
.venv/bin/python scripts/profile_test_memory.py \
  --threshold-gib 4 \
  --per-test-timeout-seconds 240 \
  --output-prefix artifacts/test_memory_profile
```

Profiler behavior:
- Tracks per-test peak RSS (resident memory).
- Immediately terminates a running test if RSS exceeds 4 GiB.
- Records per-test status, runtime, and peak memory.

## Results

- Total tests profiled: `1115`
- Passed: `1115`
- Failed: `0`
- Killed for memory >4 GiB: `0`
- Highest observed memory: `0.726 GiB`

Top memory consumers were regression-style snapshot/integration checks in:
- `tests/test_integration.py::TestSnapshotPipelineFromRealFixture::*`
- `tests/test_methods_parser_snapshot.py::TestMethodsParserSnapshoteCLIP::*`

These were all under 1 GiB and therefore below the 4 GiB termination threshold.

## Artifacts

- `artifacts/test_memory_profile.csv`
- `artifacts/test_memory_profile_summary.json`
- `artifacts/test_memory_profile_run.log`

## Recommended Efficiency Alternative (if memory pressure increases)

No test required replacement under the current 4 GiB policy. If these regression suites trend upward in future runs, the preferred optimization is:

1. Parse once per fixture scope (module/session) for expensive real-parser regression classes.
2. Keep assertions split into lightweight tests that reuse the shared parsed object.
3. Reserve full end-to-end re-parses for a small smoke subset.

Plain-English impact: this preserves behavioral coverage while reducing repeated heavy object construction in many near-duplicate tests.
