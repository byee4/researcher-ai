"""PubMed and PMC API helpers.

Provides thin wrappers around the NCBI eutils REST API for fetching
article metadata, full-text XML, and open-access PDFs.

All network calls use httpx with reasonable timeouts and retry logic.
No API key is required for small-volume access; pass NCBI_API_KEY env var
to raise the rate limit from 3 to 10 requests/second.
"""

from __future__ import annotations

import logging
import os
import json
import hashlib
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI migrated the OAI-PMH endpoint in 2024; the old URL returns 301.
PMC_OA_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
PMC_IDCONV_BASE = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_BIOC_JSON_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
PMC_OA_S3_HTTPS_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"

# Raise to 10 req/s when API key is present
_API_KEY = os.environ.get("NCBI_API_KEY")
_REQUEST_DELAY = 0.11 if _API_KEY else 0.34  # seconds between calls

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_PMC_IMAGE_HEADERS = {
    "User-Agent": "researcher-ai/figure-fetcher (+https://pmc.ncbi.nlm.nih.gov/)",
    "Referer": "https://pmc.ncbi.nlm.nih.gov/",
}
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|tiff?|svg|webp)$", re.IGNORECASE)
_BIOC_FIG_RE = re.compile(r"^\s*F(\d+)\s*$", re.IGNORECASE)
_FIGURE_TOKEN_RE = re.compile(r"\bFig(?:ure)?\.?\s*(S?\d+)\b", re.IGNORECASE)
_BIOC_METHOD_SECTION_TOKENS = (
    "METHOD",
    "METHODS",
    "MATERIALS",
    "MATERIALS METHODS",
    "METHOD DETAILS",
    "STAR METHODS",
)
_BIOC_RESULTS_SECTION_TOKENS = (
    "RESULTS",
    "RESULTS DISCUSSION",
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get(
    url: str,
    params: dict | None = None,
    retries: int = 3,
    include_api_key: bool = True,
    headers: dict[str, str] | None = None,
) -> str:
    """GET a URL, returning the response text. Retries on 5xx errors."""
    if include_api_key and _API_KEY:
        params = params or {}
        params["api_key"] = _API_KEY

    for attempt in range(retries):
        try:
            response = httpx.get(url, params=params, timeout=_DEFAULT_TIMEOUT,
                                   follow_redirects=True, headers=headers)
            response.raise_for_status()
            time.sleep(_REQUEST_DELAY)
            return response.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500 and attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("HTTP %d on attempt %d, retrying in %ds…",
                               exc.response.status_code, attempt + 1, wait)
                time.sleep(wait)
            else:
                raise
        except httpx.TimeoutException:
            if attempt < retries - 1:
                logger.warning("Timeout on attempt %d, retrying…", attempt + 1)
                time.sleep(2 ** attempt)
            else:
                raise
        except httpx.HTTPError:
            if attempt < retries - 1:
                logger.warning("HTTP transport error on attempt %d, retrying…", attempt + 1)
                time.sleep(2 ** attempt)
            else:
                raise

    raise RuntimeError(f"All {retries} attempts failed for {url}")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _bioc_cache_dir() -> Path:
    raw = os.environ.get("RESEARCHER_AI_BIOC_CACHE_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "researcher_ai_bioc_cache"


def _bioc_cache_ttl_seconds() -> int:
    raw = os.environ.get("RESEARCHER_AI_BIOC_CACHE_TTL_SEC", "86400").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 86400


def _bioc_cache_key(identifier: str, encoding: str) -> str:
    base = f"{identifier.strip().lower()}::{encoding.strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _bioc_cache_path(identifier: str, encoding: str) -> Path:
    return _bioc_cache_dir() / f"{_bioc_cache_key(identifier, encoding)}.json"


def _read_bioc_cache(identifier: str, encoding: str) -> Optional[Any]:
    path = _bioc_cache_path(identifier, encoding)
    if not path.exists():
        return None
    ttl = _bioc_cache_ttl_seconds()
    if ttl >= 0:
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_bioc_cache(identifier: str, encoding: str, payload: Any) -> None:
    cache_dir = _bioc_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _bioc_cache_path(identifier, encoding)
    fd, tmp_name = tempfile.mkstemp(prefix="bioc_", suffix=".json", dir=str(cache_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def normalize_bioc_collections(payload: Any) -> list[dict[str, Any]]:
    """Normalize BioC payload shapes into a list of collection dicts."""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _collection_document_ids(collection: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    docs = collection.get("documents")
    if not isinstance(docs, list):
        return ids
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("id", "")).strip()
        if doc_id:
            ids.add(doc_id.upper())
        infons = doc.get("infons")
        if isinstance(infons, dict):
            pmid = str(infons.get("article-id_pmid", "")).strip()
            pmc = str(infons.get("article-id_pmc", "")).strip()
            if pmid:
                ids.add(pmid.upper())
            if pmc:
                ids.add(pmc.upper())
    return ids


def select_canonical_bioc_collection(
    collections: list[dict[str, Any]],
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Select the collection matching PMID/PMCID, falling back to first."""
    if not collections:
        return None

    candidates: set[str] = set()
    if pmid:
        candidates.add(pmid.strip().upper())
    if pmcid:
        pmc = pmcid.strip().upper()
        candidates.add(pmc)
        if pmc.startswith("PMC"):
            candidates.add(pmc[3:])

    for collection in collections:
        ids = _collection_document_ids(collection)
        if ids & candidates:
            return collection
    return collections[0]


def _normalize_bioc_section_type(section_type: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", (section_type or "").upper()).strip()


def make_bioc_section_selector(tokens: Iterable[str]) -> Callable[[str], bool]:
    normalized = tuple(_normalize_bioc_section_type(tok) for tok in tokens if tok)

    def _matches(section_type: str) -> bool:
        sec = _normalize_bioc_section_type(section_type)
        if not sec:
            return False
        return any(tok in sec for tok in normalized)

    return _matches


def bioc_methods_section_selector(section_type: str) -> bool:
    return make_bioc_section_selector(_BIOC_METHOD_SECTION_TOKENS)(section_type)


def bioc_results_section_selector(section_type: str) -> bool:
    return make_bioc_section_selector(_BIOC_RESULTS_SECTION_TOKENS)(section_type)


def extract_bioc_passages(
    collection: dict[str, Any],
    section_selector: Optional[Callable[[str], bool]] = None,
) -> list[dict[str, Any]]:
    """Extract passages from a collection, optionally filtered by section type."""
    out: list[dict[str, Any]] = []
    for passage in _iter_bioc_passages(collection):
        if not isinstance(passage, dict):
            continue
        if section_selector is None:
            out.append(passage)
            continue
        infons = passage.get("infons")
        section_type = str(infons.get("section_type", "")) if isinstance(infons, dict) else ""
        if section_selector(section_type):
            out.append(passage)
    return out


def _figure_id_from_caption_text(text: str) -> Optional[str]:
    m = _FIGURE_TOKEN_RE.search(text or "")
    if not m:
        return None
    token = m.group(1).upper()
    if token.startswith("S"):
        return f"Supplementary Figure {token[1:]}"
    try:
        return f"Figure {int(token)}"
    except ValueError:
        return f"Figure {token}"


def map_bioc_figure_id(
    infon_id: str,
    text: str,
    fallback_index: int,
    irregular_f_sequence: bool = False,
) -> Optional[str]:
    """Map BioC FIG identifiers to canonical figure labels."""
    from_caption = _figure_id_from_caption_text(text)
    if from_caption:
        return from_caption

    raw = (infon_id or "").strip()
    m = _BIOC_FIG_RE.match(raw)
    if m:
        if irregular_f_sequence:
            return f"Figure {fallback_index}"
        return f"Figure {int(m.group(1))}"

    if fallback_index > 0:
        return f"Figure {fallback_index}"
    return None


def fetch_bioc_json_for_paper(
    pmid: Optional[str],
    pmcid: Optional[str],
    encoding: str = "unicode",
) -> dict[str, Any]:
    """Fetch canonical BioC collection for a paper with cache + fallback logic."""
    if not _env_flag("RESEARCHER_AI_BIOC_ENABLED", default=True):
        return {}

    pmc_norm = _normalize_pmcid(pmcid or "") if pmcid else ""
    ids_ordered: list[str] = []
    if pmid:
        ids_ordered.append(pmid.strip())
    if pmc_norm:
        ids_ordered.append(pmc_norm.replace("PMC", ""))

    attempted: set[str] = set()

    def _try_identifier(identifier: str) -> Optional[dict[str, Any]]:
        ident = identifier.strip()
        if not ident or ident in attempted:
            return None
        attempted.add(ident)
        payload = _read_bioc_cache(identifier, encoding)
        if payload is None:
            try:
                url = f"{PMC_BIOC_JSON_BASE}/{ident}/{encoding}"
                text = _get(url, include_api_key=False, headers=_PMC_IMAGE_HEADERS)
                payload = json.loads(text)
                _write_bioc_cache(ident, encoding, payload)
            except Exception:
                return None

        collections = normalize_bioc_collections(payload)
        return select_canonical_bioc_collection(
            collections,
            pmid=pmid,
            pmcid=pmc_norm,
        )

    for identifier in ids_ordered:
        collection = _try_identifier(identifier)
        if collection:
            return collection

    # Empirical fallback: if PMID fetch path failed, resolve PMCID and retry.
    if pmid and not pmc_norm:
        resolved = resolve_pmid_to_pmcid_idconv(pmid.strip()) or resolve_pmid_to_pmcid(pmid.strip())
        if resolved:
            resolved_id = _normalize_pmcid(resolved).replace("PMC", "")
            collection = _try_identifier(resolved_id)
            if collection:
                return collection

    return {}


# ── PubMed eutils ─────────────────────────────────────────────────────────────

def fetch_article_xml(pmid: str) -> str:
    """Fetch the PubMed article XML for a given PMID via efetch.

    Returns PubMed XML format (not JATS). Use parse_pubmed_xml() to
    extract structured metadata.

    Args:
        pmid: PubMed ID (numeric string, e.g. "26971820").

    Returns:
        Raw XML string.
    """
    pmid = pmid.strip()
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
        "retmode": "xml",
    }
    return _get(f"{EUTILS_BASE}/efetch.fcgi", params=params)


def fetch_pmc_fulltext(pmcid: str) -> str:
    """Fetch the full-text JATS XML for an article from PMC.

    Only works for open-access articles in PMC.

    Args:
        pmcid: PMC ID, with or without the 'PMC' prefix (e.g. "PMC4878918"
               or "4878918").

    Returns:
        JATS XML string.

    Raises:
        httpx.HTTPStatusError: If the article is not in PMC or not OA.
    """
    # Normalise: ensure PMC prefix
    pmcid = pmcid.strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmcid.replace('PMC', '')}",
        "metadataPrefix": "pmc",
    }
    # NOTE:
    # PMC OAI-PMH endpoint does not accept NCBI eutils api_key. Passing it can
    # trigger HTTP 400 responses even for valid records. Keep api_key attached
    # for eutils calls, but explicitly disable it for OAI fetches.
    return _get(PMC_OA_BASE, params=params, include_api_key=False)


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed with a text query and return a list of PMIDs.

    Args:
        query: PubMed search query string (supports boolean operators).
        max_results: Maximum number of PMIDs to return.

    Returns:
        List of PMID strings, in relevance order.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "usehistory": "n",
    }
    text = _get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
    import json
    data = json.loads(text)
    return data.get("esearchresult", {}).get("idlist", [])


def resolve_doi_to_pmid(doi: str) -> Optional[str]:
    """Convert a DOI to a PMID using the PubMed eutils search.

    Args:
        doi: DOI string (with or without "https://doi.org/" prefix).

    Returns:
        PMID string if found, None otherwise.
    """
    # Strip URL prefix if present
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip())
    results = search_pubmed(f"{doi}[DOI]", max_results=1)
    return results[0] if results else None


def resolve_pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Convert a PMID to a PMCID using elink.

    Args:
        pmid: PubMed ID string.

    Returns:
        PMCID string (e.g. "PMC4878918") if the article is in PMC, else None.
    """
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid.strip(),
        "retmode": "json",
        "linkname": "pubmed_pmc",
    }
    text = _get(f"{EUTILS_BASE}/elink.fcgi", params=params)
    import json
    data = json.loads(text)
    try:
        linksets = data["linksets"][0]["linksetdbs"]
        for ls in linksets:
            if ls.get("linkname") == "pubmed_pmc":
                ids = ls.get("links", [])
                if ids:
                    return f"PMC{ids[0]}"
    except (KeyError, IndexError):
        pass
    return None


def resolve_pmid_to_pmcid_idconv(pmid: str) -> Optional[str]:
    """Convert PMID to PMCID via the PMC idconv API."""
    params = {
        "ids": pmid.strip(),
        "format": "json",
    }
    text = _get(
        PMC_IDCONV_BASE,
        params=params,
        include_api_key=False,
        headers=_PMC_IMAGE_HEADERS,
    )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    for rec in payload.get("records", []):
        rec_pmid = str(rec.get("pmid", "")).strip()
        if rec_pmid and rec_pmid != pmid.strip():
            continue
        pmcid = (rec.get("pmcid") or "").strip()
        if pmcid:
            if not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            return pmcid
    return None


def get_figure_urls_from_pmid(pmid: str) -> list[str]:
    """Resolve a PMID and return direct figure URLs.

    Priority:
    1) PMC OA S3 metadata
    2) BioC PMC JSON
    """
    pmid = pmid.strip()
    pmcid = resolve_pmid_to_pmcid_idconv(pmid)
    if not pmcid:
        pmcid = resolve_pmid_to_pmcid(pmid)
    if not pmcid:
        return []
    return get_figure_urls_from_pmcid(pmcid)


def get_figure_urls_from_pmcid(pmcid: str) -> list[str]:
    """Return direct figure URLs from PMCID using S3-first, BioC-second strategy."""
    pmcid = _normalize_pmcid(pmcid)

    s3_urls = _get_figure_urls_from_pmc_s3_metadata(pmcid)
    if s3_urls:
        return s3_urls
    return _get_figure_urls_from_bioc(pmcid)


def resolve_pmcid_to_pmid(pmcid: str) -> Optional[str]:
    """Convert a PMCID to a PMID using elink.

    Args:
        pmcid: PMC ID (with or without 'PMC' prefix).

    Returns:
        PMID string if found, None otherwise.
    """
    pmcid = pmcid.strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    numeric_id = pmcid.replace("PMC", "")

    params = {
        "dbfrom": "pmc",
        "db": "pubmed",
        "id": numeric_id,
        "retmode": "json",
        "linkname": "pmc_pubmed",
    }
    text = _get(f"{EUTILS_BASE}/elink.fcgi", params=params)
    import json
    data = json.loads(text)
    try:
        linksets = data["linksets"][0]["linksetdbs"]
        for ls in linksets:
            if ls.get("linkname") == "pmc_pubmed":
                ids = ls.get("links", [])
                if ids:
                    return str(ids[0])
    except (KeyError, IndexError):
        pass
    return None


def download_pdf_from_pmc(pmcid: str, output_path: str | Path) -> Optional[Path]:
    """Attempt to download an open-access PDF from PubMed Central.

    PMC provides direct PDF downloads for OA articles via the ftp.ncbi.nlm.nih.gov
    server. Falls back to the web endpoint if the FTP path cannot be resolved.

    Args:
        pmcid: PMC ID (with or without 'PMC' prefix).
        output_path: Destination file path for the downloaded PDF.

    Returns:
        Path to the downloaded PDF file, or None if unavailable.
    """
    pmcid = pmcid.strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    numeric_id = pmcid.replace("PMC", "")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try the PMC PDF link endpoint
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    try:
        with httpx.stream("GET", pdf_url, timeout=_DEFAULT_TIMEOUT,
                          follow_redirects=True) as response:
            if response.status_code == 200 and "pdf" in response.headers.get("content-type", ""):
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                logger.info("Downloaded PDF to %s", output_path)
                return output_path
            else:
                logger.warning(
                    "PMC PDF not available for %s (status %d)",
                    pmcid, response.status_code,
                )
                return None
    except httpx.HTTPError as exc:
        logger.warning("Could not download PDF for %s: %s", pmcid, exc)
        return None


# ── Figure URL discovery ─────────────────────────────────────────────────────

def _normalize_pmcid(pmcid: str) -> str:
    pmcid = (pmcid or "").strip()
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return pmcid.upper()


def _normalize_pmc_article_url(url: str) -> str:
    """Normalize legacy PMC article URLs to the modern pmc.ncbi.nlm.nih.gov domain."""
    url = re.sub(
        r"^https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/",
        "https://pmc.ncbi.nlm.nih.gov/articles/",
        (url or "").strip(),
        flags=re.IGNORECASE,
    )
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def _get_figure_urls_from_pmc_s3_metadata(pmcid: str) -> list[str]:
    metadata_key = _select_latest_s3_metadata_key(pmcid)
    if not metadata_key:
        return []

    metadata_text = _get(
        f"{PMC_OA_S3_HTTPS_BASE}/{metadata_key}",
        include_api_key=False,
        headers=_PMC_IMAGE_HEADERS,
    )
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError:
        return []

    raw_candidates = _extract_image_candidates(metadata)
    s3_base_url: Optional[str] = None
    m_meta_ver = re.search(rf"metadata/{re.escape(pmcid)}\.(\d+)\.json$", metadata_key)
    if m_meta_ver:
        s3_base_url = f"{PMC_OA_S3_HTTPS_BASE}/{pmcid}.{m_meta_ver.group(1)}/"

    # Some metadata records do not list media keys directly but include xml_url.
    xml_url = str(metadata.get("xml_url", "")).strip()
    if xml_url:
        xml_https = _candidate_to_url(xml_url.split("?", 1)[0], f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
        if xml_https:
            if xml_https.startswith(PMC_OA_S3_HTTPS_BASE):
                s3_base_url = xml_https.rsplit("/", 1)[0] + "/"
            try:
                article_xml = _get(
                    xml_https,
                    include_api_key=False,
                    headers=_PMC_IMAGE_HEADERS,
                )
                raw_candidates.extend(_extract_image_refs_from_xml(article_xml))
            except Exception:
                logger.debug("Could not extract image refs from article XML for %s", pmcid)
    if not raw_candidates:
        return []
    return _resolve_candidate_urls(raw_candidates, pmcid, s3_base_url=s3_base_url)


def _select_latest_s3_metadata_key(pmcid: str) -> Optional[str]:
    prefix = f"metadata/{pmcid}."
    listing_xml = _get(
        PMC_OA_S3_HTTPS_BASE,
        params={"prefix": prefix, "max-keys": "1000"},
        include_api_key=False,
        headers=_PMC_IMAGE_HEADERS,
    )
    try:
        root = ET.fromstring(listing_xml)
    except ET.ParseError:
        return None

    keys: list[str] = []
    for el in root.iter():
        if el.tag.endswith("Key") and (el.text or "").strip():
            key = el.text.strip()
            if key.startswith(prefix) and key.endswith(".json"):
                keys.append(key)
    if not keys:
        return None

    def _version_num(k: str) -> int:
        m = re.search(rf"{re.escape(pmcid)}\.(\d+)\.json$", k)
        return int(m.group(1)) if m else -1

    keys.sort(key=_version_num, reverse=True)
    return keys[0]


def _get_figure_urls_from_bioc(pmcid: str) -> list[str]:
    url = f"{PMC_BIOC_JSON_BASE}/{pmcid}/unicode"
    text = _get(
        url,
        include_api_key=False,
        headers=_PMC_IMAGE_HEADERS,
    )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    collection = select_canonical_bioc_collection(
        normalize_bioc_collections(payload),
        pmcid=pmcid,
    ) or {}
    if not collection:
        return []

    figure_candidates: list[str] = []
    for passage in extract_bioc_passages(
        collection,
        section_selector=make_bioc_section_selector(("FIG",)),
    ):
        infons = passage.get("infons") if isinstance(passage, dict) else {}
        if not isinstance(infons, dict):
            continue
        figure_candidates.extend(_extract_image_candidates(infons))
        figure_candidates.extend(_extract_image_candidates(passage))

    if not figure_candidates:
        return []
    return _resolve_candidate_urls(figure_candidates, pmcid)


def _iter_bioc_passages(node: Any):
    if isinstance(node, dict):
        passages = node.get("passages")
        if isinstance(passages, list):
            for passage in passages:
                if isinstance(passage, dict):
                    yield passage
        for value in node.values():
            yield from _iter_bioc_passages(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_bioc_passages(item)


def _extract_image_candidates(node: Any) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            out.extend(_extract_image_candidates(value))
        return out
    if isinstance(node, list):
        for item in node:
            out.extend(_extract_image_candidates(item))
        return out
    if isinstance(node, str):
        candidate = node.strip()
        if _looks_like_image_ref(candidate):
            out.append(candidate)
        return out
    return out


def _looks_like_image_ref(value: str) -> bool:
    if not value:
        return False
    if _IMAGE_EXT_RE.search(value):
        return True
    return False


def _resolve_candidate_urls(
    candidates: list[str],
    pmcid: str,
    s3_base_url: Optional[str] = None,
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    pmc_root = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    for cand in candidates:
        normalized = _candidate_to_url(cand, pmc_root, s3_base_url=s3_base_url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def _extract_image_refs_from_xml(xml_text: str) -> list[str]:
    refs: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return refs

    root = _strip_namespaces(root)

    for fig in root.findall(".//fig"):
        label = " ".join((fig.findtext("label") or "").split()).lower()
        title = " ".join((fig.findtext("caption/title") or "").split()).lower()
        if "supplement" in label or "supplement" in title:
            continue

        for el in fig.iter():
            for attr_name, attr_value in el.attrib.items():
                if not attr_value:
                    continue
                if not attr_name.lower().endswith("href"):
                    continue
                value = attr_value.strip()
                if _looks_like_image_ref(value):
                    refs.append(value)
    return refs


def _candidate_to_url(
    candidate: str,
    pmc_root: str,
    s3_base_url: Optional[str] = None,
) -> Optional[str]:
    cand = (candidate or "").strip()
    if not cand:
        return None
    if cand.startswith("s3://"):
        cand = cand.replace("s3://pmc-oa-opendata/", f"{PMC_OA_S3_HTTPS_BASE}/", 1)
    if cand.startswith("http://") or cand.startswith("https://"):
        return _normalize_pmc_article_url(cand)

    clean = cand.lstrip("/")
    if clean.startswith("pmc-oa-opendata/"):
        return f"https://{clean}"
    if clean.startswith("bin/") or clean.startswith("media/"):
        if s3_base_url:
            return f"{s3_base_url}{clean}"
        return _normalize_pmc_article_url(f"{pmc_root}{clean}")
    if "/" in clean:
        return f"{PMC_OA_S3_HTTPS_BASE}/{clean}"
    # filename-only fallback: try both common PMC static folders.
    if _IMAGE_EXT_RE.search(clean):
        if s3_base_url:
            return f"{s3_base_url}{clean}"
        return _normalize_pmc_article_url(f"{pmc_root}bin/{clean}")
    return None


# ── XML parsing ───────────────────────────────────────────────────────────────

def parse_pubmed_xml(xml_text: str) -> dict:
    """Parse PubMed efetch XML into a structured dict.

    Extracts: pmid, pmcid, doi, title, authors, abstract, journal, year.

    Args:
        xml_text: Raw XML string from fetch_article_xml().

    Returns:
        Dict with keys: pmid, pmcid, doi, title, authors (list), abstract,
        journal, year.
    """
    root = ET.fromstring(xml_text)

    # Navigate PubMed XML structure
    article = root.find(".//PubmedArticle")
    if article is None:
        return {}

    medline = article.find("MedlineCitation")
    pubmed_data = article.find("PubmedData")
    art = medline.find("Article") if medline is not None else None

    result: dict = {}

    # PMID
    pmid_el = medline.find("PMID") if medline is not None else None
    result["pmid"] = pmid_el.text.strip() if pmid_el is not None else None

    # IDs from PubmedData
    #
    # IMPORTANT:
    # Use only the top-level ArticleIdList under PubmedData.
    # A broad ".//ArticleId" search can pick up nested IDs from comment/
    # correction/reference structures and overwrite the paper's own DOI/PMCID.
    if pubmed_data is not None:
        article_ids = pubmed_data.findall("./ArticleIdList/ArticleId")
        if not article_ids:
            # Fallback for variant XML layouts without direct ArticleIdList
            article_ids = pubmed_data.findall(".//ArticleId")

        for aid in article_ids:
            id_type = aid.get("IdType", "")
            if id_type == "doi":
                if not result.get("doi"):
                    result["doi"] = aid.text.strip() if aid.text else None
            elif id_type == "pmc":
                if not result.get("pmcid"):
                    result["pmcid"] = aid.text.strip() if aid.text else None

    if art is not None:
        # Title
        title_el = art.find("ArticleTitle")
        result["title"] = _elem_text(title_el)

        # Journal
        journal_el = art.find(".//Journal/Title")
        result["journal"] = _elem_text(journal_el)

        # Year
        year_el = art.find(".//JournalIssue/PubDate/Year")
        if year_el is None:
            year_el = art.find(".//PubDate/MedlineDate")
        if year_el is not None and year_el.text:
            year_match = re.search(r"\d{4}", year_el.text)
            result["year"] = int(year_match.group()) if year_match else None

        # Authors
        authors = []
        for author in art.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            initials = author.find("Initials")
            if last is not None:
                given = (first.text if first is not None else
                         initials.text if initials is not None else "")
                authors.append(f"{last.text}, {given}".strip(", "))
            else:
                # CollectiveName
                collective = author.find("CollectiveName")
                if collective is not None:
                    authors.append(collective.text or "")
        result["authors"] = authors

        # Abstract
        abstract_parts = []
        for abs_el in art.findall(".//AbstractText"):
            label = abs_el.get("Label")
            text = _elem_text(abs_el)
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        result["abstract"] = "\n".join(abstract_parts)

    return result


def parse_jats_xml(xml_text: str) -> dict:
    """Parse PMC JATS XML (from fetch_pmc_fulltext) into a structured dict.

    Extracts full-text sections, figure captions, and references.

    Args:
        xml_text: Raw JATS XML string from fetch_pmc_fulltext().

    Returns:
        Dict with keys: title, authors, abstract, doi, pmid, pmcid,
        sections (list of {title, text}), figure_captions (dict of id→caption),
        references (list of {ref_id, authors, title, journal, year}).
    """
    # Strip OAI wrapper if present
    if "<OAI-PMH" in xml_text or "<GetRecord" in xml_text:
        # Extract the article element from the OAI envelope
        start = xml_text.find("<article")
        end = xml_text.rfind("</article>")
        if start != -1 and end != -1:
            xml_text = xml_text[start:end + len("</article>")]

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse JATS XML: %s", exc)
        return {}
    root = _strip_namespaces(root)

    result: dict = {
        "sections": [],
        "figure_captions": {},
        "references": [],
    }
    seen_table_keys: set[str] = set()

    def _table_key(table_wrap: ET.Element) -> str:
        tid = (table_wrap.get("id", "") or "").strip()
        if tid:
            return f"id:{tid}"
        title = _table_wrap_title(table_wrap).strip()
        text = _elem_text_full(table_wrap).strip()
        return f"content:{title}|{text[:160]}"

    def _append_table_section(table_wrap: ET.Element) -> None:
        key = _table_key(table_wrap)
        if key in seen_table_keys:
            return
        seen_table_keys.add(key)
        tw_title = _table_wrap_title(table_wrap)
        tw_text = _elem_text_full(table_wrap)
        if tw_text.strip():
            result["sections"].append(
                {
                    "title": tw_title,
                    "text": tw_text.strip(),
                    "section_type": "table",
                    "is_methods": False,
                }
            )

    # Article metadata (front matter)
    front = root.find(".//front")
    if front is not None:
        meta = front.find("article-meta")
        if meta is not None:
            # Title
            title_el = meta.find(".//article-title")
            result["title"] = _elem_text_full(title_el)

            # IDs
            for article_id in meta.findall("article-id"):
                id_type = article_id.get("pub-id-type", "")
                if id_type == "doi":
                    result["doi"] = article_id.text
                elif id_type == "pmid":
                    result["pmid"] = article_id.text
                elif id_type in ("pmc", "pmcid"):
                    pmcid = (article_id.text or "").strip()
                    if pmcid and not pmcid.upper().startswith("PMC"):
                        pmcid = f"PMC{pmcid}"
                    result["pmcid"] = pmcid

            # Authors
            authors = []
            for contrib in meta.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given = contrib.find(".//given-names")
                if surname is not None:
                    name = surname.text or ""
                    if given is not None:
                        name = f"{name}, {given.text}"
                    authors.append(name.strip(", "))
            result["authors"] = authors

            # Abstract
            abstract_el = meta.find(".//abstract")
            result["abstract"] = _elem_text_full(abstract_el)

    # Body sections
    body = root.find(".//body")
    if body is not None:
        # Parse only top-level sections, but include nested subsection text so
        # methods parsers retain protocol subheadings (e.g., "eCLIP-seq library preparation").
        for sec in body.findall("./sec"):
            title_el = sec.find("title")
            sec_title = _elem_text(title_el) if title_el is not None else "Untitled"
            sec_type = (sec.get("sec-type", "") or "").strip()
            sec_text = _collect_section_text(sec)
            if sec_text.strip():
                result["sections"].append({
                    "title": sec_title,
                    "text": sec_text.strip(),
                    "section_type": sec_type or None,
                    "is_methods": _is_methods_section(sec_title, sec_type),
                })
            # Tables nested inside section trees are already represented in
            # sec_text via _collect_section_text; mark them as seen to avoid
            # duplicating as standalone sections in the global table sweep.
            for tw in sec.findall(".//table-wrap"):
                seen_table_keys.add(_table_key(tw))

        # Some JATS articles place key resource / accession tables directly
        # under <body> (outside <sec>). Preserve these as standalone sections.
        for tw in body.findall("./table-wrap"):
            _append_table_section(tw)

    # Figure captions
    # Parse from the full article tree (not only body) since some JATS variants
    # place figures in floats-groups or alternative containers.
    for fig in root.findall(".//fig"):
        fig_id = fig.get("id", "")
        label_el = fig.find("label")
        caption_el = fig.find("caption")
        label = _elem_text(label_el)
        caption = _elem_text_full(caption_el)
        display_id = label if label else fig_id
        if display_id:
            result["figure_captions"][display_id] = caption

    # References
    back = root.find(".//back")
    if back is not None:
        # Back matter sections can contain Data/Code Availability statements.
        for sec in back.findall(".//sec"):
            title_el = sec.find("title")
            sec_title = _elem_text(title_el)
            sec_type = (sec.get("sec-type", "") or "").strip()
            if not sec_title:
                sec_title = sec_type.replace("-", " ").title() or "Back Matter"
            sec_text = _collect_section_text(sec)
            if sec_text.strip():
                result["sections"].append({
                    "title": sec_title,
                    "text": sec_text.strip(),
                    "section_type": sec_type or None,
                    "is_methods": _is_methods_section(sec_title, sec_type),
                })
            for tw in sec.findall(".//table-wrap"):
                seen_table_keys.add(_table_key(tw))

        # Footnotes frequently contain accession codes in PMC manuscripts.
        # Example: <fn><bold>Accession codes</bold> ... <ext-link>GSE77634</ext-link></fn>
        for fn in back.findall(".//fn"):
            fn_text = _elem_text_full(fn)
            if not fn_text.strip():
                continue

            # Prefer a bold heading as section title when available.
            title = _elem_text(fn.find(".//bold")) or "Footnote"
            result["sections"].append({
                "title": title,
                "text": fn_text.strip(),
                "section_type": "footnote",
                "is_methods": False,
            })

        # Back-matter tables can also carry accession IDs.
        for tw in back.findall("./table-wrap"):
            _append_table_section(tw)

        for ref in back.findall(".//ref"):
            ref_id = ref.get("id", "")
            citation = ref.find(".//element-citation") or ref.find(".//mixed-citation")
            if citation is None:
                continue
            pub_type = citation.get("publication-type", "")
            ref_data: dict = {"ref_id": ref_id}

            article_title = citation.find(".//article-title")
            ref_data["title"] = _elem_text(article_title)

            source = citation.find(".//source")
            ref_data["journal"] = _elem_text(source)

            year_el = citation.find(".//year")
            ref_data["year"] = int(year_el.text) if year_el is not None and year_el.text and year_el.text.isdigit() else None

            pub_id = citation.find(".//pub-id[@pub-id-type='doi']")
            ref_data["doi"] = pub_id.text if pub_id is not None else None

            ref_authors = []
            for name in citation.findall(".//name"):
                sn = name.find("surname")
                gn = name.find("given-names")
                if sn is not None:
                    n = sn.text or ""
                    if gn is not None:
                        n = f"{n}, {gn.text}"
                    ref_authors.append(n.strip(", "))
            ref_data["authors"] = ref_authors

            result["references"].append(ref_data)

    # Some publishers place key-resources/accession tables in float groups or
    # other containers outside body/back sections. Sweep all table-wrap nodes
    # so dataset IDs (e.g., GEO accessions) are not silently dropped.
    for tw in root.findall(".//table-wrap"):
        _append_table_section(tw)

    return result


# ── XML text extraction helpers ───────────────────────────────────────────────

def _elem_text(el: Optional[ET.Element]) -> str:
    """Return text content of an element, or empty string."""
    if el is None:
        return ""
    return (el.text or "").strip()


def _elem_text_full(el: Optional[ET.Element]) -> str:
    """Return full text content of an element including all descendant text."""
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def _collect_section_text(sec: ET.Element) -> str:
    """Collect section text, recursively including subsection titles and paragraphs."""
    chunks: list[str] = []
    for child in sec:
        if child.tag == "title":
            continue
        if child.tag == "p":
            text = _elem_text_full(child)
            if text:
                chunks.append(text)
            continue
        if child.tag == "sec":
            sub_title = _elem_text(child.find("title"))
            if sub_title:
                chunks.append(sub_title)
            sub_text = _collect_section_text(child)
            if sub_text:
                chunks.append(sub_text)
            continue
        if child.tag == "table-wrap":
            table_title = _table_wrap_title(child)
            table_text = _elem_text_full(child)
            if table_text:
                chunks.append(f"{table_title}\n{table_text}".strip())
            continue
        # Fallback for other inline containers (lists, boxed-text, etc.).
        # This preserves accession IDs that are not wrapped in <p>.
        other_text = _elem_text_full(child)
        if other_text:
            chunks.append(other_text)
    return "\n".join(chunks)


def _table_wrap_title(table_wrap: ET.Element) -> str:
    """Best-effort table title from label/caption/title."""
    label = _elem_text(table_wrap.find("label"))
    caption = _elem_text(table_wrap.find("caption/title")) or _elem_text(table_wrap.find("caption"))
    if label and caption:
        return f"{label}: {caption}"
    if caption:
        return caption
    if label:
        return label
    return "Table"


def _is_methods_section(title: str, sec_type: str) -> bool:
    """Return True if JATS metadata or title indicates a methods section."""
    sec_type_norm = (sec_type or "").strip().lower().replace("-", " ")
    title_norm = (title or "").strip().lower()
    if any(k in sec_type_norm for k in ("method", "materials", "experimental procedures", "protocol")):
        return True
    if any(k in title_norm for k in ("methods", "materials and methods", "experimental procedures", "star methods")):
        return True
    return False


def _strip_namespaces(root: ET.Element) -> ET.Element:
    """Strip XML namespaces in-place so simple tag queries work across JATS variants."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root
