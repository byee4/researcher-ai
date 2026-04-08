"""researcher_ai.models — all Pydantic data models.

Import everything from here so callers don't need to know which
sub-module a model lives in:

    from researcher_ai.models import Paper, Figure, Method, Dataset, Software, Pipeline
"""

from researcher_ai.models.paper import (
    AnnotatedChunk,
    BioCContext,
    BioCPassageContext,
    ChunkType,
    Paper,
    PaperSource,
    PaperType,
    Reference,
    Section,
    SupplementaryItem,
)
from researcher_ai.models.figure import (
    Axis,
    AxisScale,
    ColorMapping,
    ColormapType,
    ErrorBarType,
    Figure,
    PanelLayout,
    PlotCategory,
    PlotLayer,
    PlotType,
    RenderingSpec,
    StatisticalAnnotation,
    SubFigure,
)
from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayDependency,
    AssayGraph,
    Method,
    MethodCategory,
)
from researcher_ai.models.dataset import (
    DataSource,
    Dataset,
    GEODataset,
    ProteomicsDataset,
    SampleMetadata,
    SRADataset,
)
from researcher_ai.models.software import (
    Command,
    Environment,
    LicenseType,
    Software,
)
from researcher_ai.models.pipeline import (
    Pipeline,
    PipelineBackend,
    PipelineConfig,
    PipelineStep,
)
from researcher_ai.models.confidence import (
    StepConfidence,
    AssayConfidence,
    PipelineConfidence,
)
from researcher_ai.models.workflow_graph import (
    ExecutionBackend,
    GraphEdge,
    GraphNode,
    GraphPort,
    GraphValidationIssue,
    NodeKind,
    NodeResources,
    PortDirection,
    PortMultiplicity,
    PortType,
    ValidationSeverity,
    WorkflowGraph,
)

__all__ = [
    # paper
    "AnnotatedChunk", "BioCContext", "BioCPassageContext", "ChunkType",
    "Paper", "PaperSource", "PaperType", "Reference", "Section", "SupplementaryItem",
    # figure
    "Axis", "AxisScale", "ColorMapping", "ColormapType", "ErrorBarType",
    "Figure", "PanelLayout", "PlotCategory", "PlotLayer", "PlotType",
    "RenderingSpec", "StatisticalAnnotation", "SubFigure",
    # method
    "AnalysisStep", "Assay", "AssayDependency", "AssayGraph", "Method", "MethodCategory",
    # dataset
    "DataSource", "Dataset", "GEODataset", "ProteomicsDataset",
    "SampleMetadata", "SRADataset",
    # software
    "Command", "Environment", "LicenseType", "Software",
    # pipeline
    "Pipeline", "PipelineBackend", "PipelineConfig", "PipelineStep",
    # confidence
    "StepConfidence", "AssayConfidence", "PipelineConfidence",
    # workflow graph
    "ExecutionBackend",
    "GraphEdge",
    "GraphNode",
    "GraphPort",
    "GraphValidationIssue",
    "NodeKind",
    "NodeResources",
    "PortDirection",
    "PortMultiplicity",
    "PortType",
    "ValidationSeverity",
    "WorkflowGraph",
]
