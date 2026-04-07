"""Local protocol-document retrieval utilities for MethodsParser Phase 3."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_DEFAULT_RAG_STORE: Optional["ProtocolRAGStore"] = None


def _default_docs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge" / "protocols"


def _default_persist_dir() -> Path:
    # Absolute path rooted at project directory (not process working directory).
    return Path(__file__).resolve().parents[2] / ".rag_chroma"


def _tokenize(text: str, *, min_token_len: int = 2) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
        if len(t) >= min_token_len
    }


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]
    chunks: list[str] = []
    advance = max(chunk_size - overlap, max(1, chunk_size // 4))
    i = 0
    n = len(clean)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(clean[i:j])
        if j >= n:
            break
        i += advance
    return chunks


@dataclass
class _Chunk:
    source: str
    text: str


class ProtocolRAGStore:
    """Local protocol-doc retriever with Chroma/embedding optional acceleration."""

    def __init__(
        self,
        *,
        docs_dir: Optional[str | Path] = None,
        persist_dir: Optional[str | Path] = None,
        collection_name: str = "protocol_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        lexical_min_token_len: int = 2,
    ):
        self.docs_dir = Path(docs_dir) if docs_dir else _default_docs_dir()
        self.persist_dir = Path(persist_dir).resolve() if persist_dir else _default_persist_dir()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = max(200, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.lexical_min_token_len = max(1, int(lexical_min_token_len))
        self._chunks: list[_Chunk] = []
        self._chroma = None
        self._embedder = None
        self._init_index()

    def _init_index(self) -> None:
        self._chunks = self._load_chunks()
        if not self._chunks:
            return
        try:
            import chromadb  # type: ignore[import]
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._embedder = SentenceTransformer(self.embedding_model)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            col = client.get_or_create_collection(name=self.collection_name)
            ids = [f"doc-{i}" for i in range(len(self._chunks))]
            docs = [c.text for c in self._chunks]
            metas = [{"source": c.source} for c in self._chunks]
            signature = self._compute_index_signature()
            if self._should_reindex(col=col, signature=signature):
                try:
                    existing = col.get(include=[])
                    existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
                    if existing_ids:
                        col.delete(ids=existing_ids)
                except Exception:
                    pass
                embeddings = self._embedder.encode(docs).tolist()
                col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
                self._write_index_signature(signature)
            self._chroma = col
        except Exception as exc:
            logger.info("RAG vector backend unavailable, using lexical retrieval fallback: %s", exc)
            self._chroma = None
            self._embedder = None

    def _load_chunks(self) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        if not self.docs_dir.exists():
            return chunks
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for chunk in _chunk_text(
                text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            ):
                chunks.append(_Chunk(source=path.name, text=chunk))
        return chunks

    def _index_meta_path(self) -> Path:
        return self.persist_dir / f"{self.collection_name}.index_meta.json"

    def _compute_index_signature(self) -> str:
        payload = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "docs": [],
        }
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            payload["docs"].append(
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_index_signature(self) -> Optional[str]:
        meta_path = self._index_meta_path()
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            sig = data.get("signature")
            return str(sig) if isinstance(sig, str) else None
        except Exception:
            return None

    def _write_index_signature(self, signature: str) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._index_meta_path()
        meta_path.write_text(
            json.dumps({"signature": signature}, indent=2),
            encoding="utf-8",
        )

    def _should_reindex(self, *, col, signature: str) -> bool:
        existing_signature = self._read_index_signature()
        if existing_signature != signature:
            return True
        try:
            existing = col.get(include=[])
            existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
            return len(existing_ids) != len(self._chunks)
        except Exception:
            return True

    def query(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        if top_k <= 0:
            return []
        if self._chroma is not None and self._embedder is not None:
            try:
                emb = self._embedder.encode([query]).tolist()
                out = self._chroma.query(query_embeddings=emb, n_results=top_k)
                docs = out.get("documents", [[]])[0]
                metas = out.get("metadatas", [[]])[0]
                res: list[dict[str, str]] = []
                for d, m in zip(docs, metas):
                    source = (m or {}).get("source", "unknown.md")
                    res.append({"source": source, "text": d})
                return res
            except Exception as exc:
                logger.info("Chroma query failed, falling back to lexical retrieval: %s", exc)

        q_tokens = _tokenize(query, min_token_len=self.lexical_min_token_len)
        scored: list[tuple[int, _Chunk]] = []
        for c in self._chunks:
            overlap = len(q_tokens & _tokenize(c.text, min_token_len=self.lexical_min_token_len))
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"source": c.source, "text": c.text} for _, c in scored[:top_k]]


def search_protocol_docs(
    query: str,
    *,
    top_k: int = 3,
    store: Optional[ProtocolRAGStore] = None,
) -> str:
    """Tool-facing retrieval function returning the top protocol chunks as text."""
    rag = store or _get_default_rag_store()
    hits = rag.query(query, top_k=top_k)
    if not hits:
        return "No protocol documents matched the query."
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] source={h['source']}\n{h['text']}")
    return "\n\n".join(lines)


def _get_default_rag_store() -> ProtocolRAGStore:
    global _DEFAULT_RAG_STORE
    if _DEFAULT_RAG_STORE is None:
        _DEFAULT_RAG_STORE = ProtocolRAGStore()
    return _DEFAULT_RAG_STORE
