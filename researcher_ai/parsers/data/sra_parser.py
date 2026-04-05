"""SRA dataset parser using pysradb.

Supports project (SRP/ERP/DRP), experiment (SRX/ERX/DRX), and run
(SRR/ERR/DRR) level accessions from INSDC partners (NCBI, EBI, DDBJ).

Strategy
--------
1. Detect accession level (project / experiment / run) via prefix regex.
2. Delegate to the appropriate ``_parse_*`` private method.
3. All network calls go through pysradb.SRAweb which wraps the SRA REST API.
4. Return a fully-populated SRADataset.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from researcher_ai.models.dataset import DataSource, SampleMetadata, SRADataset
from researcher_ai.parsers.data.base import BaseDataParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accession pattern
# ---------------------------------------------------------------------------

# Covers NCBI (SRP/SRX/SRR), EBI (ERP/ERX/ERR), and DDBJ (DRP/DRX/DRR)
_SRA_RE = re.compile(
    r"^(SRP|SRX|SRR|ERP|ERX|ERR|DRP|DRX|DRR)\d+$",
    re.IGNORECASE,
)

_PROJECT_PREFIXES = frozenset({"SRP", "ERP", "DRP"})
_EXPERIMENT_PREFIXES = frozenset({"SRX", "ERX", "DRX"})
_RUN_PREFIXES = frozenset({"SRR", "ERR", "DRR"})


# ---------------------------------------------------------------------------
# SRAParser
# ---------------------------------------------------------------------------

class SRAParser(BaseDataParser):
    """Parse SRA datasets (SRP/SRX/SRR and EBI/DDBJ equivalents).

    Args:
        detailed: Whether to fetch detailed sample attributes (slower but richer).
        timeout: Timeout passed to pysradb's underlying HTTP client.
    """

    def __init__(
        self,
        detailed: bool = True,
        timeout: float = 60.0,
    ) -> None:
        """Initialize SRA parser options and defer SRAweb client construction."""
        self.detailed = detailed
        self.timeout = timeout
        self._db: Any = None  # lazy-loaded SRAweb instance

    @property
    def db(self) -> Any:
        """Lazy-load the pysradb SRAweb client."""
        if self._db is None:
            from pysradb.sraweb import SRAweb
            self._db = SRAweb()
        return self._db

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, accession: str) -> SRADataset:
        """Parse an SRA accession at project, experiment, or run level.

        Args:
            accession: SRP/ERP/DRP (project), SRX/ERX/DRX (experiment),
                       or SRR/ERR/DRR (run) identifier.

        Returns:
            SRADataset with populated metadata, samples, and run lists.

        Raises:
            ValueError: If the accession format is not recognised.
        """
        accession = accession.strip().upper()
        if not self.validate_accession(accession):
            raise ValueError(f"Invalid SRA accession: {accession!r}")

        prefix = accession[:3]
        if prefix in _PROJECT_PREFIXES:
            return self._parse_project(accession)
        if prefix in _EXPERIMENT_PREFIXES:
            return self._parse_experiment(accession)
        # Run level
        return self._parse_run(accession)

    def validate_accession(self, accession: str) -> bool:
        """Return True if accession matches an SRA/ENA/DDBJ prefix pattern."""
        return bool(_SRA_RE.match((accession or "").strip().upper()))

    # ── Level parsers ─────────────────────────────────────────────────────────

    def _parse_project(self, srp: str) -> SRADataset:
        """Parse a project-level SRA accession (SRP/ERP/DRP).

        Fetches all experiments and runs; builds a flat sample list.
        """
        try:
            df = self.db.sra_metadata(srp, detailed=self.detailed)
        except Exception as exc:
            logger.warning("Failed to fetch SRA metadata for %s: %s", srp, exc)
            return SRADataset(accession=srp, source=DataSource.SRA, srp=srp)

        if df is None or df.empty:
            return SRADataset(accession=srp, source=DataSource.SRA, srp=srp)

        samples = _df_to_samples(df)
        srx_list = _unique_col(df, "experiment_accession")
        srr_list = _unique_col(df, "run_accession")

        # Infer experiment type from library_strategy
        experiment_type = _infer_experiment_type(df)
        organism = _first_col(df, "scientific_name")
        title = _first_col(df, "study_title") or _first_col(df, "experiment_title")

        return SRADataset(
            accession=srp,
            source=DataSource.SRA,
            title=title,
            organism=organism,
            experiment_type=experiment_type,
            samples=samples,
            total_samples=len(srr_list) or len(samples),
            srp=srp,
            srx_list=srx_list,
            srr_list=srr_list,
            raw_metadata={"n_experiments": len(srx_list), "n_runs": len(srr_list)},
        )

    def _parse_experiment(self, srx: str) -> SRADataset:
        """Parse an experiment-level SRA accession (SRX/ERX/DRX).

        Fetches all runs under this experiment.
        """
        try:
            df = self.db.sra_metadata(srx, detailed=self.detailed)
        except Exception as exc:
            logger.warning("Failed to fetch SRA metadata for %s: %s", srx, exc)
            return SRADataset(accession=srx, source=DataSource.SRA)

        if df is None or df.empty:
            return SRADataset(accession=srx, source=DataSource.SRA)

        samples = _df_to_samples(df)
        srr_list = _unique_col(df, "run_accession")
        srp = _first_col(df, "study_accession") or ""

        return SRADataset(
            accession=srx,
            source=DataSource.SRA,
            title=_first_col(df, "experiment_title"),
            organism=_first_col(df, "scientific_name"),
            experiment_type=_infer_experiment_type(df),
            samples=samples,
            total_samples=len(srr_list) or len(samples),
            srp=srp,
            srx_list=[srx],
            srr_list=srr_list,
            raw_metadata={"srx": srx, "n_runs": len(srr_list)},
        )

    def _parse_run(self, srr: str) -> SRADataset:
        """Parse a single run-level SRA accession (SRR/ERR/DRR).

        Returns a minimal SRADataset with one sample.
        """
        try:
            df = self.db.sra_metadata(srr, detailed=self.detailed)
        except Exception as exc:
            logger.warning("Failed to fetch SRA metadata for %s: %s", srr, exc)
            return SRADataset(accession=srr, source=DataSource.SRA, srr_list=[srr])

        if df is None or df.empty:
            return SRADataset(accession=srr, source=DataSource.SRA, srr_list=[srr])

        samples = _df_to_samples(df)
        srp = _first_col(df, "study_accession") or ""
        srx = _first_col(df, "experiment_accession") or ""
        spots = _first_col(df, "total_spots")
        bases = _first_col(df, "total_bases")

        raw: dict[str, Any] = {"srr": srr}
        if spots:
            raw["total_spots"] = spots
        if bases:
            raw["total_bases"] = bases

        return SRADataset(
            accession=srr,
            source=DataSource.SRA,
            title=_first_col(df, "experiment_title"),
            organism=_first_col(df, "scientific_name"),
            experiment_type=_infer_experiment_type(df),
            samples=samples,
            total_samples=1,
            srp=srp,
            srx_list=[srx] if srx else [],
            srr_list=[srr],
            raw_metadata=raw,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _df_to_samples(df: Any) -> list[SampleMetadata]:
    """Convert a pysradb DataFrame to a list of SampleMetadata objects."""
    samples: list[SampleMetadata] = []
    reserved = {
        "run_accession", "experiment_accession", "sample_accession",
        "study_accession", "sample_title", "experiment_title", "study_title",
        "organism_taxid", "scientific_name", "library_selection", "library_layout",
        "instrument_model", "instrument_platform", "library_source",
        "library_strategy", "total_spots", "total_bases", "fastq_ftp",
    }
    for _, row in df.iterrows():
        attrs: dict[str, str] = {
            col: str(row[col])
            for col in df.columns
            if col not in reserved and row.get(col) not in (None, "", "nan")
        }
        fq = row.get("fastq_ftp", "")
        fastq_urls = (
            [f"ftp://{u.strip()}" for u in str(fq).split(";") if u.strip()]
            if fq and str(fq) != "nan"
            else []
        )
        samples.append(SampleMetadata(
            sample_id=str(row.get("run_accession") or row.get("sample_accession") or ""),
            title=str(row.get("sample_title") or row.get("experiment_title") or ""),
            organism=str(row.get("scientific_name") or ""),
            source=str(row.get("library_source") or ""),
            selection=str(row.get("library_selection") or ""),
            layout=str(row.get("library_layout") or ""),
            platform=str(row.get("instrument_model") or row.get("instrument_platform") or ""),
            attributes=attrs,
            fastq_urls=fastq_urls,
        ))
    return samples


def _unique_col(df: Any, col: str) -> list[str]:
    """Return unique non-null values from a DataFrame column."""
    if col not in df.columns:
        return []
    return [str(v) for v in df[col].dropna().unique().tolist() if str(v) not in ("", "nan")]


def _first_col(df: Any, col: str) -> Optional[str]:
    """Return the first non-null value from a DataFrame column, or None."""
    if col not in df.columns:
        return None
    for val in df[col].dropna():
        s = str(val).strip()
        if s and s != "nan":
            return s
    return None


def _infer_experiment_type(df: Any) -> Optional[str]:
    """Infer experiment type from library_strategy column (most common value)."""
    col = "library_strategy"
    if col not in df.columns:
        return None
    counts = df[col].dropna().value_counts()
    if counts.empty:
        return None
    top = str(counts.index[0]).strip()
    return top if top and top != "nan" else None
