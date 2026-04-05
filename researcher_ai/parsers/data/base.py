"""Base data parser interface.

All dataset-source parsers (GEO, SRA, …) implement this ABC so that
callers can treat them uniformly regardless of the underlying database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from researcher_ai.models.dataset import Dataset


class BaseDataParser(ABC):
    """Abstract base class for all data parsers."""

    @abstractmethod
    def parse(self, accession: str) -> Dataset:
        """Parse a dataset by accession ID.

        Args:
            accession: Database-specific accession (e.g., GSE12345, SRP12345).

        Returns:
            Structured Dataset object with metadata and sample information.
        """
        ...

    @abstractmethod
    def validate_accession(self, accession: str) -> bool:
        """Check if an accession string is valid for this parser.

        Returns True if the accession format is accepted by this parser,
        False otherwise.  Does NOT perform a live lookup.
        """
        ...
