# STAR RNA-seq Alignment Notes

Typical parameters:
- `runThreadN`: number of threads
- `genomeDir`: STAR index path
- `readFilesIn`: FASTQ input path(s)
- `outSAMtype`: often `BAM SortedByCoordinate`
- `outFilterMultimapNmax`: optional multimapper threshold

Recommended to record explicitly for reproducibility.
