# BioC Confidence Benchmark (39303722)

## Summary
- Mean composite confidence: baseline `78.924` -> enhanced `83.847` (delta `4.923`)
- Ambiguous panels (40-74): baseline `13` -> enhanced `7`
- High-confidence panels (classification >= 0.75): baseline `36` -> enhanced `42`
- Contradiction count: baseline `0` -> enhanced `0`

## Gates
- Mean delta >= +3.0: `True`
- Ambiguous drop >=10% OR >=2 absolute: `True`
- Contradictions non-increasing: `True`
- Overall pass: `True`

## Artifacts
- `bioc_confidence_39303722_report.json`
- `pmid_39303722_figures_baseline.json`
- `pmid_39303722_figures_enhanced.json`
- `pmid_39303722_figures_diff.patch`
