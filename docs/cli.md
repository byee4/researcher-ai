# CLI and Scripts

## Main workflow runner

Primary command:

```bash
python scripts/run_workflow.py --source <PMID-or-PDF-path> --source-type <pmid|pdf> --output <json-path>
```

Arguments:
- `--source`: PMID value or absolute PDF path
- `--source-type`: `pmid` or `pdf`
- `--output`: output JSON file path

The script emits progress lines and writes one consolidated JSON payload.

## Calibration and benchmarking scripts

## Figure calibration report

```bash
python scripts/figure_calibration_report.py
```

Generates aggregate calibration metrics from fixtures in `tests/fixtures/figure_calibration/`.

## BioC confidence benchmark

```bash
python scripts/bioc_confidence_benchmark.py \
  --baseline /path/to/baseline.json \
  --enhanced /path/to/enhanced.json \
  --outdir /tmp/bioc
```

Generates summary JSON/Markdown and panel-level diff artifacts.
