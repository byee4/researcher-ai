from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import types

from PIL import Image, ImageDraw

from researcher_ai.models.paper import Paper, PaperSource, PaperType, Section
from researcher_ai.parsers.figure_parser import FigureParser, _SubFigureMeta, _VisionFigureExtraction
from researcher_ai.parsers.paper_parser import PaperParser
from researcher_ai.utils.pdf import extract_figure_panel_images_from_pdf


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
