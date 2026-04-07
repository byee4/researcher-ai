# Cell Ranger Count

10x Genomics primary processing:
- `cellranger count --id=<sample> --transcriptome=<refdata> --fastqs=<dir> --sample=<name>`
- Core parameters: transcriptome reference, FASTQ path, chemistry overrides when needed.
- Outputs include filtered feature-barcode matrix and QC summaries.
