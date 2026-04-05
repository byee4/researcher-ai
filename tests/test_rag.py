from __future__ import annotations

from pathlib import Path

from researcher_ai.utils.rag import ProtocolRAGStore, search_protocol_docs


def test_protocol_rag_store_returns_relevant_chunk(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "star.md").write_text(
        "STAR alignment uses runThreadN and genomeDir for index selection.",
        encoding="utf-8",
    )
    (docs_dir / "other.md").write_text(
        "Unrelated microscopy protocol.", encoding="utf-8"
    )
    store = ProtocolRAGStore(docs_dir=docs_dir, persist_dir=tmp_path / "chroma")
    hits = store.query("STAR runThreadN", top_k=2)
    assert hits
    assert any("runThreadN" in h["text"] for h in hits)


def test_search_protocol_docs_formats_sources(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "deseq2.md").write_text(
        "DESeq2 uses a design formula and BH correction.",
        encoding="utf-8",
    )
    store = ProtocolRAGStore(docs_dir=docs_dir, persist_dir=tmp_path / "chroma")
    result = search_protocol_docs("DESeq2 design", top_k=1, store=store)
    assert "source=deseq2.md" in result
    assert "DESeq2" in result


def test_default_protocol_store_eclip_query_returns_star_signal():
    store = ProtocolRAGStore()
    result = search_protocol_docs(
        "What is the standard read mapping tool for eCLIP?",
        top_k=3,
        store=store,
    )
    assert "eclip" in result.lower()
    assert "star" in result.lower()
