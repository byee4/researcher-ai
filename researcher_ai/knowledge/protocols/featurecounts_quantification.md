# featureCounts Quantification

Gene-level counting from aligned BAM:
- `featureCounts -a annotation.gtf -o counts.txt -T <threads> -p -t exon -g gene_id sample.bam`
- Important parameters: annotation file, feature type, grouping attribute, paired-end mode.
- Output count matrix is commonly used with DESeq2/edgeR.
