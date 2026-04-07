# Salmon Transcript Quantification

Common quantification workflow:
- Build index: `salmon index -t transcripts.fa -i salmon_index`.
- Quantify: `salmon quant -i salmon_index -l A -1 R1.fq.gz -2 R2.fq.gz -p <threads> -o quant`.
- Key parameters: index path, library type (`-l`), thread count (`-p`), and bootstraps (`--numBootstraps`).
