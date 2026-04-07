# Picard MarkDuplicates

Duplicate marking for aligned BAM:
- `picard MarkDuplicates I=sorted.bam O=dedup.bam M=metrics.txt REMOVE_DUPLICATES=false`
- Important parameters: input/output BAM, metrics path, duplicate handling mode.
- Follow with BAM indexing before downstream analysis.
