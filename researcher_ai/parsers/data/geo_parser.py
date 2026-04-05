"""GEO dataset parser.

Fetches structured metadata for GEO Series (GSE), Samples (GSM), and
Platforms (GPL) using the NCBI E-utilities API and pysradb for SRA bridging.

Strategy
--------
1. Validate the accession (GSE/GSM/GPL regex).
2. Fetch the GEO DataSets esummary record via E-utilities.
3. Detect SuperSeries vs plain Series from the record type field.
4. Bridge to SRA via pysradb (GSE → SRP → SRX/SRR + sample attributes).
5. Collect processed/supplementary file URLs from GEO FTP.
6. Return a fully-populated GEODataset.

Live network calls are contained in thin ``_fetch_*`` helpers so the
rest of the class can be exercised with mocked HTTP responses.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from researcher_ai.models.dataset import DataSource, GEODataset, SampleMetadata
from researcher_ai.parsers.data.base import BaseDataParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
_RETRY_DELAY = 1.0          # seconds between E-utilities retries
_MAX_RETRIES = 3

# Accession patterns
_GSE_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)
_GSM_RE = re.compile(r"^GSM\d+$", re.IGNORECASE)
_GPL_RE = re.compile(r"^GPL\d+$", re.IGNORECASE)

# E-utilities db=gds type codes that indicate a SuperSeries
_SUPERSERIES_TYPES = frozenset({"GSE", "superseries"})


# ---------------------------------------------------------------------------
# GEOParser
# ---------------------------------------------------------------------------

class GEOParser(BaseDataParser):
    """Parse GEO datasets (GSE/GSM/GPL) into structured GEODataset objects.

    Args:
        timeout: HTTP request timeout in seconds.
        api_key: Optional NCBI API key (increases rate limit from 3 to 10 req/s).
    """

    def __init__(
        self,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize GEO parser options and lazy HTTP client state."""
        self.timeout = timeout
        self.api_key = api_key
        self._client: Optional[httpx.Client] = None  # lazy-loaded

    @property
    def client(self) -> httpx.Client:
        """Lazy-load the HTTP client to defer proxy configuration until first use."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, accession: str, recursive: bool = False) -> GEODataset:
        """Parse a GEO accession into a structured GEODataset.

        Handles GSE (Series / SuperSeries), GSM (Sample), and GPL (Platform).

        Args:
            accession: A GSE, GSM, or GPL identifier (case-insensitive).
            recursive: If True and the accession is a SuperSeries, parse each
                child GSE one level deep and populate
                ``GEODataset.related_datasets`` with their accessions.
                Child parsing uses ``recursive=False`` to prevent unbounded
                API request chains.  Default False (shallow parse).

        Returns:
            GEODataset with populated metadata, samples, and file URLs.
        """
        accession = accession.strip().upper()
        if not self.validate_accession(accession):
            raise ValueError(f"Invalid GEO accession: {accession!r}")

        metadata = self._fetch_geo_metadata(accession)
        if not metadata:
            raise LookupError(f"GEO accession not found: {accession}")

        # Branch on accession type
        if _GSM_RE.match(accession):
            return self._parse_sample(accession, metadata)
        if _GPL_RE.match(accession):
            return self._parse_platform(accession, metadata)
        # GSE (Series or SuperSeries)
        return self._parse_series(accession, metadata, recursive=recursive)

    def validate_accession(self, accession: str) -> bool:
        """Return True if accession matches GSE/GSM/GPL format."""
        a = (accession or "").strip().upper()
        return bool(_GSE_RE.match(a) or _GSM_RE.match(a) or _GPL_RE.match(a))

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_series(
        self,
        gse_id: str,
        metadata: dict[str, Any],
        recursive: bool = False,
    ) -> GEODataset:
        """Parse a GSE Series or SuperSeries record.

        Args:
            gse_id: The GSE accession.
            metadata: Normalised esummary record.
            recursive: When True and this is a SuperSeries, each child GSE is
                parsed one level deep (recursive=False) and its accession is
                appended to ``related_datasets``.
        """
        child_series = self._get_child_series(gse_id, metadata)
        is_super = self._is_superseries(metadata) or bool(child_series)

        samples: list[SampleMetadata] = []
        if not is_super:
            samples = self._fetch_samples(gse_id)

        processed_urls = self._fetch_processed_data(gse_id)

        # Organism: may be multi-organism; take first non-empty
        organism = _first(metadata.get("taxon", []))

        # Recursive child parsing (one level deep only)
        related_datasets: list[str] = []
        if is_super and recursive and child_series:
            for child_gse in child_series:
                try:
                    child_ds = self.parse(child_gse, recursive=False)
                    related_datasets.append(child_ds.accession)
                    # Aggregate child samples into the SuperSeries dataset
                    samples.extend(child_ds.samples)
                except Exception as exc:
                    logger.warning(
                        "Failed to recursively parse child series %s: %s", child_gse, exc
                    )

        return GEODataset(
            accession=gse_id,
            source=DataSource.GEO,
            title=metadata.get("title", ""),
            organism=organism,
            summary=metadata.get("summary", ""),
            experiment_type=metadata.get("gdstype", "") or metadata.get("type", ""),
            samples=samples,
            total_samples=metadata.get("n_samples", len(samples)),
            processed_data_urls=processed_urls,
            series_type="SuperSeries" if is_super else "Series",
            child_series=child_series,
            related_datasets=related_datasets,
            platform_id=_first(metadata.get("gpl", [])),
            platform_name=metadata.get("platform_name", ""),
            raw_metadata=metadata,
        )

    def _parse_sample(self, gsm_id: str, metadata: dict[str, Any]) -> GEODataset:
        """Parse a single GSM Sample record."""
        sample = SampleMetadata(
            sample_id=gsm_id,
            title=metadata.get("title", ""),
            organism=_first(metadata.get("taxon", [])),
            platform=_first(metadata.get("gpl", [])),
            attributes=metadata.get("characteristics", {}),
        )
        return GEODataset(
            accession=gsm_id,
            source=DataSource.GEO,
            title=metadata.get("title", ""),
            organism=sample.organism,
            series_type="Sample",
            samples=[sample],
            total_samples=1,
            raw_metadata=metadata,
        )

    def _parse_platform(self, gpl_id: str, metadata: dict[str, Any]) -> GEODataset:
        """Parse a GPL Platform record (minimal — no samples fetched)."""
        return GEODataset(
            accession=gpl_id,
            source=DataSource.GEO,
            title=metadata.get("title", ""),
            organism=_first(metadata.get("taxon", [])),
            series_type="Platform",
            platform_id=gpl_id,
            platform_name=metadata.get("title", ""),
            raw_metadata=metadata,
        )

    # ── Network helpers ───────────────────────────────────────────────────────

    def _fetch_geo_metadata(self, accession: str) -> dict[str, Any]:
        """Fetch metadata from NCBI GEO E-utilities (esummary, db=gds).

        Returns a flat metadata dict normalised from the esummary JSON response.
        Raises ``httpx.HTTPError`` on network failure after ``_MAX_RETRIES`` attempts.
        """
        uid = self._esearch_uid(accession)
        if uid is None:
            logger.warning("No GEO UID found for accession %s", accession)
            return {}

        params: dict[str, Any] = {
            "db": "gds",
            "id": uid,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{_EUTILS_BASE}/esummary.fcgi?{urlencode(params)}"
        data = self._get_json(url)

        result_map = data.get("result", {})
        record = result_map.get(str(uid), {})
        return _normalise_esummary(record)

    def _esearch_uid(self, accession: str) -> Optional[str]:
        """Convert a GEO accession to its numeric UID via esearch."""
        params: dict[str, Any] = {
            "db": "gds",
            "term": f"{accession}[Accession]",
            "retmode": "json",
            "retmax": "1",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{_EUTILS_BASE}/esearch.fcgi?{urlencode(params)}"
        data = self._get_json(url)
        ids = data.get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else None

    def _is_superseries(self, metadata: dict[str, Any]) -> bool:
        """Return True if the metadata record indicates a SuperSeries."""
        gds_type = (metadata.get("gdstype") or metadata.get("type") or "").lower()
        return "superseries" in gds_type

    def _get_child_series(
        self, gse_id: str, metadata: dict[str, Any]
    ) -> list[str]:
        """Extract child GSE IDs from a SuperSeries record."""
        # NCBI esummary may include relations with SubSeries links.
        relations: list[dict] = metadata.get("relations", [])
        children: list[str] = []
        for rel in relations:
            url = rel.get("target", "")
            match = re.search(r"(GSE\d+)", url, re.IGNORECASE)
            if match:
                children.append(match.group(1).upper())
        children = _dedupe_ordered(children)
        if children:
            return children

        # Fallback: some GEO records omit extrelations in esummary but still
        # expose child links in SOFT text as:
        #   !Series_relation = SuperSeries of: GSE12345
        try:
            soft_text = self._fetch_series_soft_text(gse_id)
        except Exception as exc:
            logger.debug("SOFT child-series lookup failed for %s: %s", gse_id, exc)
            return []
        return _extract_child_series_from_soft_text(soft_text)

    def _fetch_series_soft_text(self, gse_id: str) -> str:
        """Fetch GEO SOFT text for a series accession."""
        url = (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            f"?acc={gse_id}&targ=self&form=text&view=full"
        )
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.get(url)
                resp.raise_for_status()
                return resp.text
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
        raise last_exc

    def _fetch_samples(self, gse_id: str) -> list[SampleMetadata]:
        """Fetch sample-level metadata for a GSE via pysradb SRA bridge.

        Bridges GSE → SRP (via elink) → SRX/SRR (via pysradb).
        Falls back to an empty list if SRA bridging fails.
        """
        try:
            from pysradb.sraweb import SRAweb
            db = SRAweb()
            srp = self._gse_to_srp(gse_id)
            if srp is None:
                return []
            df = db.sra_metadata(srp, detailed=True)
            if df is None or df.empty:
                return []
            return _df_to_samples(df)
        except Exception as exc:
            logger.warning("SRA bridging failed for %s: %s", gse_id, exc)
            return []

    def _gse_to_srp(self, gse_id: str) -> Optional[str]:
        """Find the SRA Project (SRP) accession for a GSE accession.

        Strategy (in order):
        1. pysradb ``gse_to_srp()`` — fast, wraps the SRA REST API.
        2. NCBI elink ``db=gds → db=sra`` — fallback when pysradb fails.

        Returns the SRP accession string, or None if both paths fail.
        """
        # 1. pysradb fast path
        try:
            from pysradb.sraweb import SRAweb
            df = SRAweb().gse_to_srp(gse_id)
            if df is not None and not df.empty and "study_accession" in df.columns:
                return df["study_accession"].iloc[0]
        except Exception as exc:
            logger.debug("pysradb gse_to_srp failed for %s: %s", gse_id, exc)

        # 2. NCBI elink fallback: GEO UID → SRA IDs
        try:
            # GEO UIDs for GSE accessions are the numeric part prefixed with 200
            geo_uid = "200" + re.sub(r"^GSE", "", gse_id, flags=re.IGNORECASE)
            params: dict[str, Any] = {
                "dbfrom": "gds",
                "db": "sra",
                "id": geo_uid,
                "retmode": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key
            url = f"{_EUTILS_BASE}/elink.fcgi?{urlencode(params)}"
            data = self._get_json(url)
            link_sets = data.get("linksets", [])
            for ls in link_sets:
                for link_set_db in ls.get("linksetdbs", []):
                    if link_set_db.get("dbto") == "sra":
                        sra_ids = link_set_db.get("links", [])
                        if sra_ids:
                            # Fetch the SRP accession from the first SRA ID
                            srp = self._sra_id_to_srp(str(sra_ids[0]))
                            if srp:
                                return srp
        except Exception as exc:
            logger.debug("elink gse_to_srp failed for %s: %s", gse_id, exc)

        return None

    def _sra_id_to_srp(self, sra_uid: str) -> Optional[str]:
        """Convert a numeric SRA UID to an SRP study accession via esummary."""
        try:
            params: dict[str, Any] = {
                "db": "sra",
                "id": sra_uid,
                "retmode": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key
            url = f"{_EUTILS_BASE}/esummary.fcgi?{urlencode(params)}"
            data = self._get_json(url)
            result = data.get("result", {}).get(sra_uid, {})
            # SRA esummary embeds accession in 'expxml' or 'runs'
            expxml = result.get("expxml", "")
            match = re.search(r"<Study\s+acc=\"(SRP\d+)\"", expxml)
            if match:
                return match.group(1)
        except Exception as exc:
            logger.debug("_sra_id_to_srp failed for uid %s: %s", sra_uid, exc)
        return None

    def _fetch_processed_data(self, gse_id: str) -> list[str]:
        """Build GEO FTP URLs for the supplementary/processed data directory.

        GEO FTP layout: /geo/series/GSEnnn/GSExxxxxx/suppl/
        No live fetch needed — URL is deterministic from accession.
        """
        # Construct the FTP stem from the accession (e.g. GSE72987 → GSE72nnn)
        nnn = re.sub(r"\d{3}$", "nnn", gse_id)
        base = f"{_GEO_FTP_BASE}/{nnn}/{gse_id}/suppl/"
        return [base]

    # ── HTTP utility ──────────────────────────────────────────────────────────

    def _get_json(self, url: str) -> dict[str, Any]:
        """GET a URL and return the parsed JSON, retrying on transient errors."""
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.get(url)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
        raise last_exc

    def close(self) -> None:
        """Close the underlying HTTP client (no-op if never opened)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "GEOParser":
        """Context-manager entry; returns self."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Context-manager exit; closes the lazy HTTP client."""
        self.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _first(seq: list) -> Optional[str]:
    """Return the first element of a list, or None if empty."""
    return seq[0] if seq else None


def _normalise_gpl(token: str) -> str:
    """Ensure a GPL platform token has exactly one 'GPL' prefix.

    NCBI esummary returns numeric-only IDs (e.g. ``"11154"``) in most cases but
    has been observed to also return the full form (``"GPL11154"``).  This helper
    normalises both to ``"GPL11154"`` without double-prefixing.
    """
    t = token.strip()
    if t.upper().startswith("GPL"):
        return t.upper()          # already correct; preserve case-canonical form
    return f"GPL{t}"


def _normalise_esummary(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten an NCBI GEO esummary result record into a simpler dict."""
    if not record:
        return {}

    # Taxon may be a string or list
    taxon_raw = record.get("taxon", "")
    if isinstance(taxon_raw, str):
        taxon = [t.strip() for t in taxon_raw.split(";") if t.strip()]
    else:
        taxon = list(taxon_raw)

    # GPL may also be a string or list.
    # NCBI usually returns numeric-only IDs (e.g. "11154") but may also
    # return the full identifier ("GPL11154").  Guard against double-prefix.
    gpl_raw = record.get("gpl", "")
    if isinstance(gpl_raw, str):
        gpl = [_normalise_gpl(g.strip()) for g in gpl_raw.split(";") if g.strip()]
    else:
        gpl = [_normalise_gpl(str(g)) for g in gpl_raw if str(g).strip()]

    # Relations (SubSeries, SRA links)
    relations = []
    for rel in record.get("extrelations", []):
        relations.append({
            "type": rel.get("relationtype", ""),
            "target": rel.get("targetobject", ""),
        })

    n_samples_raw = record.get("n_samples", record.get("samplecount", 0))
    try:
        n_samples = int(n_samples_raw)
    except (TypeError, ValueError):
        n_samples = 0

    return {
        "title": record.get("title", ""),
        "summary": record.get("summary", ""),
        "gdstype": record.get("gdstype", ""),
        "type": record.get("entrytype", ""),
        "taxon": taxon,
        "gpl": gpl,
        "platform_name": record.get("platformtitle", ""),
        "n_samples": n_samples,
        "relations": relations,
        "characteristics": {},          # populated only for GSM records
        "accession": record.get("accession", ""),
        "uid": record.get("uid", ""),
    }


def _df_to_samples(df: Any) -> list[SampleMetadata]:
    """Convert a pysradb DataFrame to a list of SampleMetadata objects."""
    samples: list[SampleMetadata] = []
    for _, row in df.iterrows():
        attrs: dict[str, str] = {}
        for col in df.columns:
            val = row.get(col)
            if val and col not in {
                "run_accession", "experiment_accession", "sample_accession",
                "study_accession", "sample_title", "organism_taxid",
                "scientific_name", "library_selection", "library_layout",
                "instrument_model",
            }:
                attrs[col] = str(val)

        fastq_urls: list[str] = []
        fq = row.get("fastq_ftp", "")
        if fq:
            fastq_urls = [f"ftp://{u.strip()}" for u in str(fq).split(";") if u.strip()]

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


def _dedupe_ordered(values: list[str]) -> list[str]:
    """Return case-insensitive deduplicated values, preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = (value or "").strip().upper()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _extract_child_series_from_soft_text(text: str) -> list[str]:
    """Extract child GSE IDs from GEO SOFT `!Series_relation` lines."""
    children: list[str] = []
    for line in (text or "").splitlines():
        if "!Series_relation" not in line:
            continue
        # SuperSeries record format:
        #   !Series_relation = SuperSeries of: GSE12345
        m = re.search(r"SuperSeries\s+of:\s*(GSE\d+)\b", line, flags=re.IGNORECASE)
        if m:
            children.append(m.group(1).upper())
    return _dedupe_ordered(children)
