# Kallisto Quantification

Pseudoalignment for transcript abundance:
- Build index: `kallisto index -i transcripts.idx transcripts.fa`.
- Quantify: `kallisto quant -i transcripts.idx -o out -t <threads> -b 100 R1.fq.gz R2.fq.gz`.
- Key settings: index path, read mode (single/paired), thread count, bootstrap count.
