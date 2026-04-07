# GATK Variant Calling

Common germline calling flow:
- `gatk HaplotypeCaller -R ref.fa -I dedup.bam -O sample.g.vcf.gz -ERC GVCF`
- Joint genotyping often follows with `GenotypeGVCFs`.
- Key parameters: reference FASTA, known-sites resources, ploidy, emit mode.
