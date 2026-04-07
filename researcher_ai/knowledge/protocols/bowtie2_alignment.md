# Bowtie2 Alignment

Typical short-read alignment command:
- `bowtie2 -x <index_base> -1 R1.fq.gz -2 R2.fq.gz -p <threads> --very-sensitive`
- Important parameters: index base, threads, sensitivity preset, paired/single mode.
- Output SAM should be converted to sorted BAM for downstream tools.
