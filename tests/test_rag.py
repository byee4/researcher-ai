from __future__ import annotations

import types
import sys
from unittest.mock import patch
from pathlib import Path

from researcher_ai.utils.rag import ProtocolRAGStore, search_protocol_docs, _default_persist_dir


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


def test_protocol_rag_store_defaults_to_absolute_project_persist_dir():
    store = ProtocolRAGStore()
    assert store.persist_dir == _default_persist_dir()
    assert store.persist_dir.is_absolute()


def test_protocol_rag_store_preserves_short_domain_tokens(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "protocol.md").write_text(
        "PCR uses ATP-dependent polymerase extension.",
        encoding="utf-8",
    )
    store = ProtocolRAGStore(
        docs_dir=docs_dir,
        persist_dir=tmp_path / "chroma",
        lexical_min_token_len=2,
    )
    hits = store.query("PCR ATP", top_k=1)
    assert hits
    assert "PCR" in hits[0]["text"]


def test_protocol_rag_store_honors_chunk_config(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    long_text = " ".join(f"token{i}" for i in range(1200))
    (docs_dir / "long.md").write_text(long_text, encoding="utf-8")
    store = ProtocolRAGStore(
        docs_dir=docs_dir,
        persist_dir=tmp_path / "chroma",
        chunk_size=240,
        chunk_overlap=24,
    )
    assert store._chunks
    assert len(store._chunks) > 1


def test_protocol_rag_store_uses_configured_embedding_model(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "star.md").write_text("STAR alignment", encoding="utf-8")
    fake_chromadb = types.ModuleType("chromadb")
    fake_sentence = types.ModuleType("sentence_transformers")

    mock_col = type("C", (), {})()
    mock_col.get = lambda include=None: {"ids": []}  # noqa: ARG005
    mock_col.add = lambda **kwargs: None  # noqa: ARG005
    mock_col.delete = lambda **kwargs: None  # noqa: ARG005
    fake_client = type("Client", (), {"get_or_create_collection": lambda self, name: mock_col})()  # noqa: ARG005
    fake_chromadb.PersistentClient = lambda path: fake_client  # noqa: ARG005

    class _FakeEncoded:
        def tolist(self):
            return [[0.1, 0.2]]

    called_with: list[str] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str):
            called_with.append(model_name)

        def encode(self, docs):
            return _FakeEncoded()

    fake_sentence.SentenceTransformer = _FakeSentenceTransformer

    with patch.dict(
        sys.modules,
        {"chromadb": fake_chromadb, "sentence_transformers": fake_sentence},
    ):
        ProtocolRAGStore(
            docs_dir=docs_dir,
            persist_dir=tmp_path / "chroma",
            embedding_model="sentence-t5-base",
        )
    assert called_with == ["sentence-t5-base"]
