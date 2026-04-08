from __future__ import annotations

from pathlib import Path

from researcher_ai.models.figure import Figure, PlotCategory, PlotType, SubFigure
from researcher_ai.models.paper import ChunkType, Paper, PaperSource, Section
from researcher_ai.utils.paper_indexer import PaperRAGStore, merge_retrieval_results


def _summary(text: str, chunk_type: ChunkType) -> str:
    return f"{chunk_type.value}: {text[:40]}"


def _make_paper(*, section_text: str, source: PaperSource = PaperSource.PMCID, source_path: str = "PMC1") -> Paper:
    return Paper(
        title="Test Paper",
        source=source,
        source_path=source_path,
        sections=[Section(title="Methods", text=section_text)],
    )


def _make_figure() -> Figure:
    return Figure(
        figure_id="Figure 1",
        title="Panelized figure",
        caption="(A) scRNA-seq UMAP. (B) Volcano. (C) ChIP-seq peaks with STAR v2.7.10a.",
        purpose="Demonstrates distinct assay outputs.",
        subfigures=[
            SubFigure(
                label="A",
                description="scRNA-seq UMAP embedding",
                plot_category=PlotCategory.DIMENSIONALITY,
                plot_type=PlotType.UMAP,
            ),
            SubFigure(
                label="C",
                description="ChIP-seq peak profile generated after STAR alignment",
                plot_category=PlotCategory.GENOMIC,
                plot_type=PlotType.COVERAGE_TRACK,
            ),
        ],
    )


def test_table_chunk_typing_from_markdown():
    paper = _make_paper(
        section_text=(
            "Reads were aligned with STAR.\n\n"
            "| tool | version | parameter | value |\n"
            "| --- | --- | --- | --- |\n"
            "| STAR | 2.7.10a | --runThreadN | 8 |"
        )
    )
    store = PaperRAGStore(
        summary_builder=_summary,
        enable_vector_index=False,
    )
    store.build_from(paper=paper, figures=[])

    table_chunks = [c for c in store.chunks if c.chunk_type == ChunkType.TABLE]
    assert table_chunks, "Expected at least one table chunk"
    assert any("--runThreadN" in c.text for c in table_chunks)


def test_panel_level_figure_chunks_are_created_and_tagged():
    paper = _make_paper(section_text="Methods mention figures.")
    figure = _make_figure()
    store = PaperRAGStore(
        summary_builder=_summary,
        enable_vector_index=False,
    )
    store.build_from(paper=paper, figures=[figure])

    panel_chunks = [c for c in store.chunks if c.chunk_type == ChunkType.FIGURE_CAPTION and c.panel_id]
    panel_ids = {c.panel_id for c in panel_chunks}
    assert {"A", "C"}.issubset(panel_ids)


def test_malformed_table_triggers_fallback_path(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake")
    paper = _make_paper(
        section_text=(
            "Malformed table below\n\n"
            "| tool | version | parameter |\n"
            "| --- | --- | --- |\n"
            "| STAR | 2.7.10a | --runThreadN | 8 |"
        ),
        source=PaperSource.PDF,
        source_path=str(pdf_path),
    )

    store = PaperRAGStore(
        summary_builder=_summary,
        enable_vector_index=False,
    )
    calls: list[str] = []

    def _fake_recover(*, malformed_table_text: str, source_pdf: Path | None, section_title: str) -> str | None:
        calls.append(section_title)
        assert source_pdf is not None
        return "STAR 2.7.10a --runThreadN 8"

    monkeypatch.setattr(store, "_recover_table_chunk_with_vision", _fake_recover)
    store.build_from(paper=paper, figures=[])

    assert calls, "Expected malformed table fallback to be called"
    assert any(c.chunk_type == ChunkType.TABLE and "--runThreadN" in c.text for c in store.chunks)
    assert store.vision_fallback_count >= 1
    assert store.vision_fallback_latency_seconds >= 0.0


def test_table_only_parameter_retrieval_and_paper_first_merge():
    paper = _make_paper(
        section_text=(
            "STAR was used for alignment.\n\n"
            "| tool | parameter | value |\n"
            "| --- | --- | --- |\n"
            "| STAR | --outSAMtype | BAM SortedByCoordinate |"
        )
    )
    store = PaperRAGStore(
        summary_builder=_summary,
        enable_vector_index=False,
    )
    store.build_from(paper=paper, figures=[])
    paper_hits = store.query("STAR outSAMtype BAM SortedByCoordinate", top_k=2)
    assert paper_hits
    assert any("--outSAMtype" in str(hit["text"]) for hit in paper_hits)

    protocol_hits = [
        {"source": "star.md", "text": "General STAR defaults", "score": 0.95},
    ]
    merged = merge_retrieval_results(
        paper_hits=paper_hits,
        protocol_hits=protocol_hits,
        top_k=1,
        paper_bias=0.10,
        protocol_bias=0.0,
    )
    assert merged
    assert merged[0]["source_type"] == "paper"
