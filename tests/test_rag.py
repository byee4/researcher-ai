from __future__ import annotations

import types
import sys
from unittest.mock import patch
from pathlib import Path

import researcher_ai.utils.rag as rag_mod
from researcher_ai.utils.rag import (
    ProtocolRAGStore,
    _chunk_text,
    _default_persist_dir,
    search_protocol_docs,
)


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
    lowered = result.lower()
    assert "source=" in lowered
    assert any(tool in lowered for tool in ("star", "bwa", "bowtie2", "hisat2"))


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


def test_chunk_text_empty_and_single_chunk():
    assert _chunk_text("", chunk_size=100, overlap=20) == []
    assert _chunk_text("abc", chunk_size=100, overlap=20) == ["abc"]


def test_chunk_text_large_overlap_still_advances():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = _chunk_text(text, chunk_size=80, overlap=79)
    assert len(chunks) < len(text) // 2
    assert len(set(chunks)) == len(chunks)


def test_search_protocol_docs_uses_singleton_store_when_not_provided(monkeypatch):
    rag_mod._DEFAULT_RAG_STORE = None
    created: list[str] = []

    class _FakeStore:
        def __init__(self):
            created.append("x")

        def query(self, query: str, top_k: int = 3):
            return [{"source": "fake.md", "text": f"{query}:{top_k}"}]

    monkeypatch.setattr(rag_mod, "ProtocolRAGStore", _FakeStore)
    first = search_protocol_docs("q1", top_k=1)
    second = search_protocol_docs("q2", top_k=1)
    assert len(created) == 1
    assert "q1" in first
    assert "q2" in second


def test_rag_store_skips_reindex_when_signature_unchanged(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "star.md").write_text("STAR runThreadN genomeDir", encoding="utf-8")

    fake_chromadb = types.ModuleType("chromadb")
    fake_sentence = types.ModuleType("sentence_transformers")

    class _FakeCollection:
        def __init__(self):
            self.add_calls = 0
            self.delete_calls = 0
            self._ids = []

        def get(self, include=None):  # noqa: ARG002
            return {"ids": self._ids}

        def add(self, ids, documents, metadatas, embeddings):  # noqa: ARG002
            self.add_calls += 1
            self._ids = list(ids)

        def delete(self, ids):  # noqa: ARG002
            self.delete_calls += 1
            self._ids = []

    col = _FakeCollection()

    class _FakeClient:
        def get_or_create_collection(self, name):  # noqa: ARG002
            return col

    fake_chromadb.PersistentClient = lambda path: _FakeClient()  # noqa: ARG005

    class _FakeEncoded:
        def tolist(self):
            return [[0.1, 0.2]]

    class _FakeSentenceTransformer:
        def __init__(self, model_name):  # noqa: ARG002
            pass

        def encode(self, docs):  # noqa: ARG002
            return _FakeEncoded()

    fake_sentence.SentenceTransformer = _FakeSentenceTransformer

    with patch.dict(
        sys.modules,
        {"chromadb": fake_chromadb, "sentence_transformers": fake_sentence},
    ):
        ProtocolRAGStore(docs_dir=docs_dir, persist_dir=tmp_path / "chroma")
        ProtocolRAGStore(docs_dir=docs_dir, persist_dir=tmp_path / "chroma")

    assert col.add_calls == 1
