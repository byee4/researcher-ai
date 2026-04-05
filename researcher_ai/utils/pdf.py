"""PDF text and figure extraction utilities.

Uses pdfplumber for text extraction (handles multi-column layouts better
than PyPDF2) and Pillow for image I/O.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# pdfplumber is listed as a required dependency; import lazily so the module
# can be imported in test environments where only models are needed.
try:
    import pdfplumber  # type: ignore[import]
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFPLUMBER_AVAILABLE = False


def _require_pdfplumber() -> None:
    """Raise a clear ImportError when pdfplumber-dependent APIs are called without it."""
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError(
            "pdfplumber is required for PDF operations. "
            "Install with: pip install pdfplumber"
        )


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file.

    Concatenates text from every page with a newline separator.
    Handles multi-column layouts by preserving pdfplumber's extraction order.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text content as a single string.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ImportError: If pdfplumber is not installed.
    """
    _require_pdfplumber()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages_text.append(text)

    return "\n".join(pages_text)


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract per-page text and metadata from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts, each with keys:
            page_number (int, 1-indexed)
            text (str)
            width (float, points)
            height (float, points)
            n_chars (int)
    """
    _require_pdfplumber()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            pages.append({
                "page_number": i,
                "text": text,
                "width": float(page.width),
                "height": float(page.height),
                "n_chars": len(text),
            })

    return pages


def extract_images_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_size_px: int = 50,
) -> list[Path]:
    """Extract embedded images from a PDF and save them to output_dir.

    Only images larger than min_size_px in both dimensions are extracted
    to avoid extracting decorative lines and logos.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory where extracted images will be saved.
        min_size_px: Minimum image dimension (width and height) in pixels.

    Returns:
        List of paths to the extracted image files.

    Notes:
        Requires Pillow (PIL) for image I/O.
    """
    _require_pdfplumber()
    try:
        from PIL import Image  # type: ignore[import]
        import io
    except ImportError as exc:
        raise ImportError("Pillow is required for image extraction.") from exc

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for img_idx, img_obj in enumerate(page.images):
                try:
                    # pdfplumber returns image objects with raw stream data
                    img_stream = img_obj.get("stream")
                    if img_stream is None:
                        continue
                    raw_data = img_stream.get_data()
                    img = Image.open(io.BytesIO(raw_data))
                    w, h = img.size
                    if w < min_size_px or h < min_size_px:
                        continue
                    out_path = output_dir / f"page{page_num:03d}_img{img_idx:02d}.png"
                    img.save(out_path, format="PNG")
                    saved_paths.append(out_path)
                    logger.debug("Extracted image %s (%dx%d)", out_path.name, w, h)
                except Exception as exc:
                    logger.warning(
                        "Could not extract image %d on page %d: %s",
                        img_idx, page_num, exc,
                    )

    return saved_paths


# ── Section detection helpers (regex-based, no LLM) ──────────────────────────

# Common scientific paper section headers (case-insensitive).
# Ordered from most to least specific to avoid false-positive matches.
_SECTION_PATTERNS = [
    r"^(abstract)\s*$",
    r"^(introduction)\s*$",
    r"^(background)\s*$",
    r"^(results(?:\s+and\s+discussion)?)\s*$",
    r"^(discussion)\s*$",
    r"^(methods?(?:\s+and\s+materials?)?|materials?\s+and\s+methods?|experimental\s+(?:procedures?|methods?))\s*$",
    r"^(conclusions?)\s*$",
    r"^(acknowledgements?|acknowledgments?)\s*$",
    r"^(references?|bibliography)\s*$",
    r"^(supplementar[y\s](?:information|materials?|methods?|figures?|tables?))\s*$",
    r"^(data\s+availability)\s*$",
    r"^(code\s+availability)\s*$",
    r"^(author\s+contributions?)\s*$",
    r"^(competing\s+interests?|conflict[s]?\s+of\s+interest)\s*$",
    r"^(figure\s+legends?|extended\s+data)\s*$",
]

_SECTION_RE = re.compile(
    "|".join(_SECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def detect_section_boundaries(text: str) -> list[tuple[str, int]]:
    """Find section header names and their character offsets in raw PDF text.

    Returns:
        List of (section_title, char_offset) tuples in document order.
        Useful for splitting raw text into sections before LLM processing.
    """
    boundaries: list[tuple[str, int]] = []
    for match in _SECTION_RE.finditer(text):
        title = match.group(0).strip()
        boundaries.append((title, match.start()))
    return boundaries


def split_text_into_sections(text: str) -> dict[str, str]:
    """Split raw PDF text into a dict of {section_title: section_text}.

    Uses regex-detected section boundaries. Returns an 'PREAMBLE' key
    for text before the first recognised section header (usually title +
    authors + abstract in papers that embed abstract in the header).

    Args:
        text: Full raw text from extract_text_from_pdf().

    Returns:
        Ordered dict of section_title → section_text.
    """
    boundaries = detect_section_boundaries(text)
    if not boundaries:
        return {"FULL_TEXT": text}

    sections: dict[str, str] = {}

    # Text before first section header
    first_offset = boundaries[0][1]
    preamble = text[:first_offset].strip()
    if preamble:
        sections["PREAMBLE"] = preamble

    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        # Skip the header line itself — jump past first newline
        body_start = text.find("\n", start)
        body = text[body_start:end].strip() if body_start != -1 else text[start:end].strip()
        canonical = title.upper()
        sections[canonical] = body

    return sections


# ── Figure reference extraction (regex, no LLM) ──────────────────────────────

# Primary pattern: matches the anchor "Fig." / "Figure" / "Figs." prefix and
# the first number that follows. A second pass (below) handles enumeration
# continuations like "Figs. 3 and 4B" or "Figures 1, 2, and 3A".
_FIGURE_REF_RE = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?\s*)(\d+[A-Za-z]?(?:[–\-]\d*[A-Za-z]?)?)",
    re.IGNORECASE,
)

# Continuation pattern: after a plural "Figs." / "Figures" anchor capture
# subsequent comma- or "and"-separated number+letter tokens that follow
# immediately (within the same phrase, before a sentence boundary).
# E.g., "Figs. 3 and 4B" → also captures "4B"
_FIGURE_ENUM_RE = re.compile(
    r"\b(?:Fig(?:ure)?s\.?\s+)"           # plural anchor (Figs. / Figures)
    r"(\d+[A-Za-z]?)"                      # first id
    r"(?:(?:[,\s]+(?:and\s+)?)"           # separator: ", " / " and "
    r"(\d+[A-Za-z]?))*",                   # additional ids
    re.IGNORECASE,
)

# Sub-pattern used to pull all id tokens out of an enumeration match
_FIGURE_TOKEN_RE = re.compile(r"\d+[A-Za-z]?")

# Supplementary figures: Supplementary Figure 1, Fig. S1, Supplementary Fig. S1
_SUPP_FIGURE_RE = re.compile(
    r"\b(?:Supplementar[y\s]+Fig(?:ure)?s?\.?\s*)([A-Za-z]?\d+[A-Za-z]?)",
    re.IGNORECASE,
)


def extract_figure_ids_from_text(text: str) -> list[str]:
    """Extract all unique figure ID references from a text string.

    Handles both main figures (Fig. 1, Figure 2A) and supplementary
    figures (Supplementary Figure S1, Fig. S2).

    Returns:
        Sorted, deduplicated list of figure ID strings,
        e.g., ['Figure 1', 'Figure 2A', 'Supplementary Figure S1'].
    """
    ids: set[str] = set()

    for match in _FIGURE_REF_RE.finditer(text):
        # Avoid treating supplementary references as main figures.
        # Example to block: "Supplementary Fig. 8" -> should not add "Figure 8".
        prefix = text[max(0, match.start() - 24): match.start()]
        if re.search(r"(?i)(?:supplementary|supp\.)\s*$", prefix):
            continue
        num_part = match.group(1).strip()
        ids.add(f"Figure {num_part}")

    # Handle plural/enumerated forms: "Figs. 3 and 4B", "Figures 1, 2, and 3A"
    for match in _FIGURE_ENUM_RE.finditer(text):
        prefix = text[max(0, match.start() - 24): match.start()]
        if re.search(r"(?i)(?:supplementary|supp\.)\s*$", prefix):
            continue
        for token in _FIGURE_TOKEN_RE.findall(match.group(0)):
            ids.add(f"Figure {token}")

    for match in _SUPP_FIGURE_RE.finditer(text):
        num_part = match.group(1).strip()
        ids.add(f"Supplementary Figure {num_part}")

    return sorted(ids, key=_figure_sort_key)


def _figure_sort_key(fig_id: str) -> tuple[int, str, int, str]:
    """Sort figure IDs: main figures first, then supplementary, numerically."""
    is_supp = 1 if fig_id.lower().startswith("supplementary") else 0
    m = re.search(r"(\d+)([A-Za-z]?)", fig_id)
    num = int(m.group(1)) if m else 0
    letter = m.group(2) if m else ""
    return (is_supp, "", num, letter)
