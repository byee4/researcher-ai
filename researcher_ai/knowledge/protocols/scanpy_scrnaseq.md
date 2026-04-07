# Scanpy scRNA-seq Workflow

Typical Scanpy analysis pattern:
- `pp.normalize_total`, `pp.log1p`, `pp.highly_variable_genes`, `pp.scale`.
- Dimensionality reduction and neighbors: `tl.pca`, `pp.neighbors`.
- Clustering and embedding: `tl.leiden`, `tl.umap`.
- Key reproducibility settings: random seed, number of PCs, resolution.
