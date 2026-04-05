"""Software and environment data models.

Represents software tools identified in a paper's methods section,
along with their execution environment and CLI commands.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LicenseType(str, Enum):
    OPEN_SOURCE = "open_source"
    CLOSED_SOURCE = "closed_source"
    FREEMIUM = "freemium"
    UNKNOWN = "unknown"


class Command(BaseModel):
    """A CLI command or function call for a software tool."""

    command_template: str               # e.g., "STAR --runMode genomeGenerate ..."
    description: str
    required_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    parameters: dict[str, str] = Field(default_factory=dict)


class Environment(BaseModel):
    """Environment specification for running software."""

    conda_yaml: Optional[str] = None    # Conda environment YAML string
    docker_image: Optional[str] = None
    pip_requirements: list[str] = Field(default_factory=list)
    system_dependencies: list[str] = Field(default_factory=list)
    setup_instructions: str = ""


class Software(BaseModel):
    """A software tool used in analysis."""

    name: str                           # e.g., "STAR", "DESeq2", "samtools"
    version: Optional[str] = None
    source_url: Optional[str] = None    # GitHub URL, website, etc.
    license_type: LicenseType = LicenseType.UNKNOWN
    language: Optional[str] = None      # e.g., "C++", "R", "Python"
    description: str = ""
    environment: Optional[Environment] = None
    commands: list[Command] = Field(default_factory=list)
    open_source_alternative: Optional[str] = Field(
        default=None,
        description="If closed-source, name of best open-source alternative",
    )
    bioconda_package: Optional[str] = None
    cran_package: Optional[str] = None
    pypi_package: Optional[str] = None
    github_repo: Optional[str] = None
    notebooks_available: list[str] = Field(
        default_factory=list,
        description="URLs to Jupyter notebooks demonstrating usage",
    )
