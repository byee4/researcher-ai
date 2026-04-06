from __future__ import annotations

from pydantic import BaseModel, Field


class StepConfidence(BaseModel):
    has_software: bool = False
    has_version: bool = False
    has_parameters: bool = False
    has_input_output: bool = False
    parameter_completeness: float = 0.0
    overall: float = 50.0


class AssayConfidence(BaseModel):
    step_confidences: list[StepConfidence] = Field(default_factory=list)
    dataset_resolved: bool = False
    figure_confidence_mean: float = 50.0
    parse_warning_count: int = 0
    overall: float = 50.0


class PipelineConfidence(BaseModel):
    assay_confidences: dict[str, AssayConfidence] = Field(default_factory=dict)
    validation_passed: bool = False
    overall: float = 50.0
    human_edited_steps: int = 0
