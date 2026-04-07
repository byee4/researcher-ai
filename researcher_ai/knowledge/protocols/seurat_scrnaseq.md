# Seurat scRNA-seq Workflow

Common Seurat preprocessing sequence:
- `NormalizeData` -> `FindVariableFeatures` -> `ScaleData` -> `RunPCA`.
- Clustering: `FindNeighbors` + `FindClusters` with explicit resolution.
- Visualization: `RunUMAP` or `RunTSNE` with selected PCs.
