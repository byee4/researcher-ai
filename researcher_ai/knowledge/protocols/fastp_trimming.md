# fastp Trimming

Read trimming and QC in one step:
- `fastp -i R1.fq.gz -I R2.fq.gz -o R1.trim.fq.gz -O R2.trim.fq.gz -w <threads>`
- Common settings: adapter detection, quality cutoff, minimum read length.
- JSON/HTML reports are useful for QC tracking.
