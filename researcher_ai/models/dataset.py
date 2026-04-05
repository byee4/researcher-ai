"""Dataset data models.

Represents datasets referenced or used in a paper, with GEO, SRA,
ENCODE, and supplementary file variants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    GEO = "geo"
    SRA = "sra"
    ENCODE = "encode"
    SUPPLEMENTARY = "supplementary"
    OTHER = "other"


class SampleMetadata(BaseModel):
    """Metadata for a single sample/run."""

    sample_id: str                      # e.g., SRR ID, GSM ID
    title: Optional[str] = None
    organism: Optional[str] = None
    source: Optional[str] = None
    selection: Optional[str] = None
    layout: Optional[str] = None        # SINGLE or PAIRED
    platform: Optional[str] = None      # e.g., Illumina HiSeq
    attributes: dict[str, str] = Field(default_factory=dict)
    fastq_urls: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    """A dataset referenced or used in a paper."""

    accession: str                      # e.g., GSE12345, SRP12345
    source: DataSource
    title: Optional[str] = None
    organism: Optional[str] = None
    summary: Optional[str] = None
    experiment_type: Optional[str] = None  # e.g., "RNA-seq", "ChIP-seq"
    samples: list[SampleMetadata] = Field(default_factory=list)
    total_samples: int = 0
    processed_data_urls: list[str] = Field(default_factory=list)
    supplementary_files: list[str] = Field(default_factory=list)
    related_datasets: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Full metadata dict from source API",
    )


class GEODataset(Dataset):
    """GEO-specific dataset with series/superseries awareness."""

    source: DataSource = DataSource.GEO
    series_type: Optional[str] = None  # "Series" or "SuperSeries"
    child_series: list[str] = Field(
        default_factory=list,
        description="GSE IDs of child series (if SuperSeries)",
    )
    platform_id: Optional[str] = None  # e.g., GPL570
    platform_name: Optional[str] = None


class SRADataset(Dataset):
    """SRA-specific dataset with project/experiment/run hierarchy."""

    source: DataSource = DataSource.SRA
    srp: Optional[str] = None          # Project ID
    srx_list: list[str] = Field(default_factory=list)  # Experiment IDs
    srr_list: list[str] = Field(default_factory=list)  # Run IDs


class ProteomicsDataset(Dataset):
    """Proteomics dataset from PRIDE or similar archives.

    Added after Ouroboros evaluation identified proteomics papers (mass
    spec, phosphoproteomics) in the test corpus that require PRIDE/MassIVE
    accessions rather than GEO/SRA accessions.
    """

    source: DataSource = DataSource.OTHER
    pride_accession: Optional[str] = None   # PXD* accession from PRIDE
    massIVE_accession: Optional[str] = None # MSV* accession
    instrument: Optional[str] = None        # e.g., "Orbitrap Fusion Lumos"
    fragmentation: Optional[str] = None     # e.g., "HCD", "CID"
    quantification_method: Optional[str] = None  # e.g., "TMT", "LFQ", "iTRAQ"
    database_search_software: Optional[str] = None  # e.g., "MaxQuant", "Proteome Discoverer"
