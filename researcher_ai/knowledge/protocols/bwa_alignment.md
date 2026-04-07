# BWA-MEM Alignment

DNA-seq read mapping pattern:
- `bwa mem -t <threads> <ref.fa> R1.fq.gz R2.fq.gz | samtools sort -@ <threads> -o sample.bam`
- Important parameters: reference FASTA, thread count, read group (`-R`) metadata.
- Downstream duplicate marking and indexing are standard.
