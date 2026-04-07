"""PDF text and figure extraction utilities.

Uses pdfplumber for text extraction (handles multi-column layouts better
than PyPDF2) and Pillow for image I/O.
"""

from __future__ import annotations

from collections import deque
import os
import logging
import re
import io
import math
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


def extract_markdown_from_pdf_with_marker(pdf_path: str | Path) -> str:
    """Extract spatially faithful Markdown using marker-pdf when available.

    Falls back to plain text extraction when marker-pdf is unavailable or fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Marker API variants differ by release; keep this path resilient.
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore[import]
        from marker.models import create_model_dict  # type: ignore[import]

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        markdown = getattr(rendered, "markdown", None)
        if isinstance(markdown, str) and markdown.strip():
            return markdown
    except Exception as exc:
        logger.warning("marker-pdf Markdown extraction unavailable/failed; falling back to plain text: %s", exc)

    return extract_text_from_pdf(pdf_path)


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


def extract_figure_panel_images_from_pdf(
    pdf_path: str | Path,
    figure_id: str,
    *,
    caption: str = "",
    dpi: int = 180,
    max_panels: int = 8,
    return_diagnostics: bool = False,
    max_image_bytes: Optional[int] = None,
) -> list[bytes] | tuple[list[bytes], list[str]]:
    """Extract figure panel crops as PNG bytes for multimodal LLM calls.

    Strategy:
    1. Find all pages mentioning the figure id (handles multi-page figures).
    2. Rasterize each candidate page.
    3. Try connected-components panel detection first.
    4. Fall back to a caption-informed grid split when detection is ambiguous.
    5. Enforce per-panel byte cap via adaptive resize/quantize before returning.
    """
    _require_pdfplumber()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    diagnostics: list[str] = []
    max_image_bytes = max_image_bytes or int(os.environ.get("RESEARCHER_AI_MAX_IMAGE_BYTES", 5 * 1024 * 1024))

    fig_num_match = re.search(r"(?i)(?:fig(?:ure)?\.?\s*)(\d+)", figure_id or "")
    if not fig_num_match:
        diagnostics.append("figure_id_unparseable")
        return ([], diagnostics) if return_diagnostics else []
    fig_num = fig_num_match.group(1)
    fig_pat = re.compile(rf"(?i)\bfig(?:ure)?\.?\s*{re.escape(fig_num)}[A-Za-z]?\b")

    panel_count = _estimate_panel_count_from_caption(caption)
    panel_count = max(1, min(panel_count, max_panels))

    page_indices: list[int] = []
    page_images: list = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if fig_pat.search(text):
                page_indices.append(idx)
        if not page_indices:
            diagnostics.append("figure_page_not_found")
            return ([], diagnostics) if return_diagnostics else []

        for page_index in page_indices:
            page = pdf.pages[page_index]
            page_images.append(page.to_image(resolution=dpi).original.convert("RGB"))

    panel_images: list[bytes] = []
    for page_image in page_images:
        remaining = max(1, panel_count - len(panel_images))
        page_panels, page_diagnostics = _extract_panels_from_page_image(
            page_image,
            panel_count=remaining,
            max_panels=max_panels,
            max_image_bytes=max_image_bytes,
        )
        panel_images.extend(page_panels)
        diagnostics.extend(page_diagnostics)
        if len(panel_images) >= panel_count:
            break

    if not panel_images:
        diagnostics.append("no_panel_images_extracted")
        return ([], _dedupe_preserve_order(diagnostics)) if return_diagnostics else []

    panel_images = panel_images[:max_panels]
    deduped_diagnostics = _dedupe_preserve_order(diagnostics)
    if return_diagnostics:
        return panel_images, deduped_diagnostics
    return panel_images


def _estimate_panel_count_from_caption(caption: str) -> int:
    text = caption or ""
    labels: set[str] = set()

    # Parenthesized single-letter labels, e.g. (a), (B), (a1), (b2)
    for match in re.finditer(r"\(([a-zA-Z])(?:\d+)?\)", text):
        labels.add(match.group(1).lower())
    # Bare single-letter labels followed by punctuation or whitespace, e.g. "A.", "b:"
    for match in re.finditer(r"(?<![A-Za-z0-9])([A-Ha-h])(?:[.)\]:]|\s)", text):
        labels.add(match.group(1).lower())
    # Numeric labels, e.g. (1), 2), 3.
    numeric_labels = {
        int(match.group(1))
        for match in re.finditer(r"(?<!\d)\(?([1-9])\)?(?:[.)\]:]|\s)", text)
    }

    letter_count = len(labels)
    numeric_count = len(numeric_labels)
    inferred = max(letter_count, numeric_count, 1)
    return inferred


def _image_to_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def _extract_panels_from_page_image(
    image,
    *,
    panel_count: int,
    max_panels: int,
    max_image_bytes: int,
) -> tuple[list[bytes], list[str]]:
    diagnostics: list[str] = []

    boxes = _detect_panel_boxes_from_image(image, expected_count=panel_count)
    if len(boxes) >= 2:
        diagnostics.append("panel_detection_connected_components")
    else:
        boxes = _grid_boxes(image.size[0], image.size[1], panel_count)
        diagnostics.append("panel_detection_grid_fallback")

    boxes = boxes[:max(1, min(max_panels, panel_count))]
    panels: list[bytes] = []
    for box in boxes:
        crop = image.crop(box)
        panel_bytes, panel_diag = _image_to_png_bytes_with_limit(crop, max_image_bytes=max_image_bytes)
        panels.append(panel_bytes)
        diagnostics.extend(panel_diag)
    return panels, diagnostics


def _detect_panel_boxes_from_image(image, *, expected_count: int) -> list[tuple[int, int, int, int]]:
    """Find likely panel boxes using connected components on a binarized thumbnail."""
    try:
        from PIL import ImageFilter  # type: ignore[import]
    except ImportError:
        return []

    orig_w, orig_h = image.size
    max_dim = max(orig_w, orig_h)
    scale = max(1.0, max_dim / 900.0)
    work_w = max(64, int(orig_w / scale))
    work_h = max(64, int(orig_h / scale))

    gray = image.convert("L").resize((work_w, work_h))
    binary = gray.point(lambda p: 255 if p < 245 else 0, mode="L")
    binary = binary.filter(ImageFilter.MaxFilter(size=7))
    binary = binary.filter(ImageFilter.MinFilter(size=3))

    pixels = binary.load()
    visited = bytearray(work_w * work_h)

    min_area = max(200, int(work_w * work_h * 0.008))
    min_w = max(20, int(work_w * 0.08))
    min_h = max(20, int(work_h * 0.08))

    components: list[tuple[int, int, int, int, int]] = []
    for y in range(work_h):
        for x in range(work_w):
            idx = y * work_w + x
            if visited[idx] or pixels[x, y] == 0:
                continue
            visited[idx] = 1
            queue: deque[tuple[int, int]] = deque([(x, y)])
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while queue:
                cx, cy = queue.popleft()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= work_w or ny >= work_h:
                        continue
                    n_idx = ny * work_w + nx
                    if visited[n_idx] or pixels[nx, ny] == 0:
                        continue
                    visited[n_idx] = 1
                    queue.append((nx, ny))
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            if area < min_area or width < min_w or height < min_h:
                continue
            components.append((min_x, min_y, max_x, max_y, area))

    if len(components) < 2:
        return []

    components.sort(key=lambda item: item[4], reverse=True)
    cap = min(len(components), max(2, expected_count * 2))
    selected = components[:cap]

    scale_x = orig_w / work_w
    scale_y = orig_h / work_h
    boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1, _ in selected:
        pad_x = max(2, int((x1 - x0 + 1) * 0.02))
        pad_y = max(2, int((y1 - y0 + 1) * 0.02))
        ox0 = max(0, int((x0 - pad_x) * scale_x))
        oy0 = max(0, int((y0 - pad_y) * scale_y))
        ox1 = min(orig_w, int((x1 + pad_x + 1) * scale_x))
        oy1 = min(orig_h, int((y1 + pad_y + 1) * scale_y))
        if ox1 - ox0 < 8 or oy1 - oy0 < 8:
            continue
        boxes.append((ox0, oy0, ox1, oy1))

    boxes = _remove_near_duplicate_boxes(boxes)
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def _remove_near_duplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    deduped: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if any(_iou(box, existing) > 0.85 for existing in deduped):
            continue
        deduped.append(box)
    return deduped


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter_area / float(area_a + area_b - inter_area)


def _grid_boxes(width: int, height: int, panel_count: int) -> list[tuple[int, int, int, int]]:
    cols = math.ceil(math.sqrt(panel_count))
    rows = math.ceil(panel_count / cols)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if len(boxes) >= panel_count:
                break
            x0 = int((c / cols) * width)
            x1 = int(((c + 1) / cols) * width)
            y0 = int((r / rows) * height)
            y1 = int(((r + 1) / rows) * height)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _image_to_png_bytes_with_limit(image, *, max_image_bytes: int) -> tuple[bytes, list[str]]:
    from PIL import Image  # type: ignore[import]

    diagnostics: list[str] = []

    encoded = _image_to_png_bytes(image)
    if len(encoded) <= max_image_bytes:
        return encoded, diagnostics

    diagnostics.append("panel_resize_applied")
    candidate = image
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    adaptive_palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", Image.ADAPTIVE)
    for _ in range(12):
        encoded = _image_to_png_bytes(candidate)
        if len(encoded) <= max_image_bytes:
            return encoded, diagnostics

        for n_colors in (128, 64, 32, 16, 8):
            quantized = candidate.convert("P", palette=adaptive_palette, colors=n_colors).convert("RGB")
            quantized_encoded = _image_to_png_bytes(quantized)
            if len(quantized_encoded) <= max_image_bytes:
                diagnostics.append("panel_quantization_applied")
                return quantized_encoded, diagnostics

        next_w = max(32, int(candidate.size[0] * 0.75))
        next_h = max(32, int(candidate.size[1] * 0.75))
        if (next_w, next_h) == candidate.size:
            break
        candidate = candidate.resize((next_w, next_h), resample=resample)

    diagnostics.append("panel_still_over_limit")
    emergency = candidate.resize((32, 32), resample=resample)
    return _image_to_png_bytes(emergency), diagnostics


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
