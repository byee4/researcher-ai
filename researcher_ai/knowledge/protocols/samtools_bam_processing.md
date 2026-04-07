# Samtools BAM Processing

Standard BAM preparation sequence:
- `samtools view -bS in.sam > out.bam`
- `samtools sort -@ <threads> -o sorted.bam out.bam`
- `samtools index sorted.bam`
- Common quality filtering: `samtools view -q <mapq>`.
