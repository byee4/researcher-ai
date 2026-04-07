# HISAT2 Alignment

Typical paired-end RNA-seq alignment settings:
- `hisat2 -x <index_base> -1 <reads_1.fq.gz> -2 <reads_2.fq.gz> -p <threads> --dta`
- Important parameters: `-x` (index), `-p` (threads), `--rna-strandness`, `--dta`.
- Output is SAM/BAM that should be sorted and indexed before quantification.
