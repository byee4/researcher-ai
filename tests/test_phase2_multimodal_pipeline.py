from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import types
import sys
import os

from PIL import Image, ImageDraw
import pytest
from pydantic import ValidationError

from researcher_ai.models.figure import PanelBoundingBox
from researcher_ai.models.paper import Paper, PaperSource, PaperType, Section
from researcher_ai.parsers.figure_parser import FigureParser, _SubFigureMeta, _VisionFigureExtraction
from researcher_ai.parsers.paper_parser import PaperParser
from researcher_ai.utils.pdf import extract_figure_panel_images_from_pdf, extract_markdown_from_pdf_with_marker

REAL_PDF_FIXTURE = Path(__file__).parent / "fixtures" / "figure_calibration" / "Sison_Nature_2026.pdf"


def test_paper_parser_pdf_uses_marker_markdown():
    parser = PaperParser(llm_model="test-model")
    sentinel = object()
    with patch("researcher_ai.parsers.paper_parser.extract_markdown_from_pdf_with_marker") as mock_marker, patch.object(
        PaperParser, "_parse_raw_text"
    ) as mock_parse_raw:
        mock_marker.return_value = "# Title\n\n## Methods\nTable A | B"
        mock_parse_raw.return_value = sentinel
        out = parser._parse_from_pdf("/tmp/fake.pdf")

    assert out is sentinel
    mock_marker.assert_called_once_with("/tmp/fake.pdf")
    assert mock_parse_raw.call_count == 1
    args, _ = mock_parse_raw.call_args
    assert args[1] == "/tmp/fake.pdf"
    assert args[2] == PaperSource.PDF
    assert "## Methods" in args[0]


def test_figure_parser_pdf_multimodal_path_skips_legacy_heuristics(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")

    paper = Paper(
        title="Test PDF",
        source=PaperSource.PDF,
        source_path=str(pdf_path),
        paper_type=PaperType.EXPERIMENTAL,
        sections=[Section(title="Results", text="See Figure 1 for details.")],
        figure_ids=["Figure 1"],
        figure_captions={"Figure 1": "Figure 1. (a) Heatmap. (b) Volcano plot."},
    )
    parser = FigureParser(llm_model="test-model", vision_model="gemini-3.1-pro")

    extraction = _VisionFigureExtraction(
        title="Figure 1 multimodal",
        purpose="Shows two complementary analyses.",
        methods_used=["RNA-seq"],
        datasets_used=["GSE12345"],
        subfigures=[
            _SubFigureMeta(
                label="a",
                description="Heatmap of marker genes",
                plot_type="heatmap",
                plot_category="matrix",
            ),
            _SubFigureMeta(
                label="b",
                description="Volcano plot of differential expression",
                plot_type="volcano",
                plot_category="genomic",
            ),
        ],
    )

    with patch("researcher_ai.parsers.figure_parser.extract_figure_panel_images_from_pdf") as mock_panels, patch(
        "researcher_ai.parsers.figure_parser._extract_structured_data"
    ) as mock_extract, patch.object(
        FigureParser, "_get_bioc_context_for_figure", side_effect=AssertionError("legacy BioC path should not run")
    ):
        mock_panels.return_value = [b"img-1", b"img-2"]
        mock_extract.return_value = extraction
        figure = parser.parse_figure(paper, "Figure 1")

    assert figure.figure_id == "Figure 1"
    assert figure.title == "Figure 1 multimodal"
    assert figure.methods_used == ["RNA-seq"]
    assert figure.datasets_used == ["GSE12345"]
    assert len(figure.subfigures) == 2
    mock_panels.assert_called_once()
    assert mock_extract.call_count == 1


def test_extract_figure_panel_images_from_pdf_multimodal_fixture(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")

    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 199, 199), fill="red")
    draw.rectangle((200, 0, 399, 199), fill="green")
    draw.rectangle((0, 200, 199, 399), fill="blue")
    draw.rectangle((200, 200, 399, 399), fill="yellow")

    class _FakePage:
        def extract_text(self, **kwargs):  # noqa: ARG002
            return "Figure 1. complex multi-panel layout"

        def to_image(self, resolution):  # noqa: ARG002
            return types.SimpleNamespace(original=img)

    class _FakePdf:
        pages = [_FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    with patch("researcher_ai.utils.pdf._PDFPLUMBER_AVAILABLE", True), patch(
        "researcher_ai.utils.pdf.pdfplumber.open", return_value=_FakePdf()
    ):
        panels = extract_figure_panel_images_from_pdf(
            pdf_path,
            figure_id="Figure 1",
            caption="Figure 1. (a) panel one (b) panel two (c) panel three (d) panel four.",
        )

    assert len(panels) == 4
    assert all(isinstance(p, bytes) and len(p) > 0 for p in panels)


def test_panel_bounding_box_rejects_non_increasing_bounds():
    with pytest.raises(ValidationError):
        PanelBoundingBox(x0=0.8, y0=0.0, x1=0.2, y1=0.9)
    with pytest.raises(ValidationError):
        PanelBoundingBox(x0=0.1, y0=0.9, x1=0.8, y1=0.3)


def test_pdf_multimodal_fallback_records_warning_when_no_panels(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")
    paper = Paper(
        title="Fallback PDF",
        source=PaperSource.PDF,
        source_path=str(pdf_path),
        paper_type=PaperType.EXPERIMENTAL,
        sections=[],
        figure_ids=["Figure 1"],
        figure_captions={"Figure 1": "Figure 1. Panel A and B."},
    )
    parser = FigureParser(llm_model="test-model", vision_model="gemini-3.1-pro")

    with patch(
        "researcher_ai.parsers.figure_parser.extract_figure_panel_images_from_pdf",
        return_value=([], ["no_panel_images_extracted"]),
    ):
        figure = parser.parse_figure(paper, "Figure 1")

    assert figure.purpose == "Could not be parsed."
    assert any("no_panel_images_extracted" in warning for warning in figure.parse_warnings)


def test_pdf_multimodal_fallback_records_warning_on_vision_error(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")
    paper = Paper(
        title="Fallback PDF",
        source=PaperSource.PDF,
        source_path=str(pdf_path),
        paper_type=PaperType.EXPERIMENTAL,
        sections=[],
        figure_ids=["Figure 1"],
        figure_captions={"Figure 1": "Figure 1. Panel A and B."},
    )
    parser = FigureParser(llm_model="test-model", vision_model="gemini-3.1-pro")

    with patch(
        "researcher_ai.parsers.figure_parser.extract_figure_panel_images_from_pdf",
        return_value=([b"img"], []),
    ), patch(
        "researcher_ai.parsers.figure_parser._extract_structured_data", side_effect=RuntimeError("vision down")
    ):
        figure = parser.parse_figure(paper, "Figure 1")

    assert figure.purpose == "Could not be parsed."
    assert any("vision_extraction_failed:RuntimeError" in warning for warning in figure.parse_warnings)


def test_marker_fallback_logs_warning_and_returns_plain_text(tmp_path: Path, caplog):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")

    class _BrokenConverter:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("marker unavailable")

    marker_converters_pdf = types.ModuleType("marker.converters.pdf")
    marker_converters_pdf.PdfConverter = _BrokenConverter
    marker_models = types.ModuleType("marker.models")
    marker_models.create_model_dict = lambda: {}

    with patch.dict(
        sys.modules,
        {
            "marker.converters.pdf": marker_converters_pdf,
            "marker.models": marker_models,
        },
    ), patch("researcher_ai.utils.pdf.extract_text_from_pdf", return_value="plain-text-fallback"), caplog.at_level(
        "WARNING"
    ):
        output = extract_markdown_from_pdf_with_marker(pdf_path)

    assert output == "plain-text-fallback"
    assert "falling back to plain text" in caplog.text


def test_extract_figure_panel_images_resizes_to_limit(tmp_path: Path):
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy")
    noisy_bytes = os.urandom(1600 * 1600 * 3)
    noisy = Image.frombytes("RGB", (1600, 1600), noisy_bytes)

    class _FakePage:
        def extract_text(self, **kwargs):  # noqa: ARG002
            return "Figure 1"

        def to_image(self, resolution):  # noqa: ARG002
            return types.SimpleNamespace(original=noisy)

    class _FakePdf:
        pages = [_FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    with patch("researcher_ai.utils.pdf._PDFPLUMBER_AVAILABLE", True), patch(
        "researcher_ai.utils.pdf.pdfplumber.open", return_value=_FakePdf()
    ):
        panels, diagnostics = extract_figure_panel_images_from_pdf(
            pdf_path,
            figure_id="Figure 1",
            caption="Figure 1",
            max_panels=1,
            max_image_bytes=120_000,
            return_diagnostics=True,
        )

    assert len(panels) == 1
    assert len(panels[0]) <= 120_000
    assert "panel_resize_applied" in diagnostics


@pytest.mark.skipif(not REAL_PDF_FIXTURE.exists(), reason="Sison_Nature_2026.pdf not found")
def test_real_pdf_panel_extraction_path_integration():
    paper = Paper(
        title="Sison Nature 2026",
        source=PaperSource.PDF,
        source_path=str(REAL_PDF_FIXTURE),
        paper_type=PaperType.EXPERIMENTAL,
        sections=[],
        figure_ids=["Figure 1"],
        figure_captions={"Figure 1": "Figure 1. (a) (b) (c) (d)"},
    )
    parser = FigureParser(llm_model="test-model", vision_model="gemini-3.1-pro")

    def _capture_images(*args, **kwargs):  # noqa: ARG001
        image_bytes = kwargs.get("image_bytes", [])
        assert image_bytes
        return _VisionFigureExtraction(
            title="Integrated extraction",
            purpose="Real PDF panel extraction fed image bytes to vision parser.",
            methods_used=[],
            datasets_used=[],
            subfigures=[
                _SubFigureMeta(label="a", description="Panel A", plot_type="image", plot_category="image"),
            ],
        )

    with patch("researcher_ai.parsers.figure_parser._extract_structured_data", side_effect=_capture_images):
        figure = parser.parse_figure(paper, "Figure 1")

    assert figure.title == "Integrated extraction"
    assert figure.purpose.startswith("Real PDF panel extraction")
