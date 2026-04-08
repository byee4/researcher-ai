"""Per-paper multimodal retrieval index used for BioWorkflow-style parsing."""

from __future__ import annotations

import logging
import re
import uuid
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from researcher_ai.models.figure import Figure
from researcher_ai.models.paper import AnnotatedChunk, ChunkType, Paper, PaperSource
from researcher_ai.utils import llm as llm_utils
from researcher_ai.utils.llm import LLMCache, SYSTEM_METHODS_PARSER
from researcher_ai.utils.pdf import extract_figure_panel_images_from_pdf
from researcher_ai.utils.rag import _chunk_text, _tokenize

logger = logging.getLogger(__name__)


class _VisionTableText(BaseModel):
    """Fallback schema for vision-based malformed-table recovery."""

    extracted_text: str = Field(default="")


@dataclass
class _IndexedChunk:
    chunk: AnnotatedChunk
    searchable_text: str


class PaperRAGStore:
    """Ephemeral per-paper retrieval store with modality provenance."""

    def __init__(
        self,
        *,
        llm_model: str = llm_utils.DEFAULT_MODEL,
        cache: Optional[LLMCache] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        lexical_min_token_len: int = 2,
        summary_builder: Optional[Callable[[str, ChunkType], str]] = None,
        enable_vector_index: bool = True,
    ):
        self.llm_model = llm_model
        self.cache = cache
        self.embedding_model = embedding_model
        self.chunk_size = max(200, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.lexical_min_token_len = max(1, int(lexical_min_token_len))
        self.summary_builder = summary_builder
        self.enable_vector_index = bool(enable_vector_index)

        self._chunks: list[_IndexedChunk] = []
        self._chroma = None
        self._embedder = None
        self._collection_name = f"paper_{uuid.uuid4().hex[:12]}"
        self._vision_fallback_count = 0
        self._vision_fallback_latency_seconds = 0.0

    @property
    def chunks(self) -> list[AnnotatedChunk]:
        return [c.chunk for c in self._chunks]

    @property
    def vision_fallback_count(self) -> int:
        return int(self._vision_fallback_count)

    @property
    def vision_fallback_latency_seconds(self) -> float:
        return float(self._vision_fallback_latency_seconds)

    def build_from(
        self,
        *,
        paper: Paper,
        figures: Optional[list[Figure]] = None,
    ) -> "PaperRAGStore":
        """Build/rebuild the in-memory index for one paper parse context."""
        self._vision_fallback_count = 0
        self._vision_fallback_latency_seconds = 0.0
        raw_chunks = self._collect_chunks(paper=paper, figures=figures or [])
        indexed: list[_IndexedChunk] = []
        for chunk in raw_chunks:
            summary = self._summarize_chunk(chunk.text, chunk.chunk_type)
            entry = chunk.model_copy(update={"summary": summary})
            searchable = f"{entry.text}\n\nSummary: {entry.summary}" if entry.summary else entry.text
            indexed.append(_IndexedChunk(chunk=entry, searchable_text=searchable))
        self._chunks = indexed
        self._init_vector_index()
        return self

    def query(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        """Return top per-paper hits with provenance metadata and scores."""
        if top_k <= 0 or not self._chunks:
            return []
        if self._chroma is not None and self._embedder is not None:
            try:
                emb = self._embedder.encode([query]).tolist()
                out = self._chroma.query(
                    query_embeddings=emb,
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                docs = out.get("documents", [[]])[0]
                metas = out.get("metadatas", [[]])[0]
                distances = out.get("distances", [[]])[0]
                results: list[dict[str, object]] = []
                for doc, meta, dist in zip(docs, metas, distances):
                    distance = float(dist) if dist is not None else 1.0
                    score = 1.0 / (1.0 + max(distance, 0.0))
                    m = meta or {}
                    results.append(
                        {
                            "source": m.get("source", "paper"),
                            "source_type": "paper",
                            "text": doc,
                            "score": score,
                            "chunk_id": m.get("chunk_id"),
                            "chunk_type": m.get("chunk_type", ChunkType.PROSE.value),
                            "source_section": m.get("source_section", ""),
                            "figure_id": m.get("figure_id"),
                            "panel_id": m.get("panel_id"),
                        }
                    )
                if results:
                    return results
            except Exception as exc:
                logger.info("PaperRAG vector query failed, using lexical fallback: %s", exc)

        q_tokens = _tokenize(query, min_token_len=self.lexical_min_token_len)
        scored: list[tuple[float, _IndexedChunk]] = []
        for entry in self._chunks:
            overlap = len(
                q_tokens
                & _tokenize(entry.searchable_text, min_token_len=self.lexical_min_token_len)
            )
            if overlap <= 0:
                continue
            text_tokens = _tokenize(
                entry.searchable_text,
                min_token_len=self.lexical_min_token_len,
            )
            denom = max(len(q_tokens), 1)
            score = float(overlap) / float(denom)
            score += min(0.25, float(overlap) / float(max(len(text_tokens), 1)))
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, object]] = []
        for score, entry in scored[:top_k]:
            chunk = entry.chunk
            results.append(
                {
                    "source": f"paper:{chunk.source_section or 'unknown'}",
                    "source_type": "paper",
                    "text": chunk.text,
                    "score": score,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type.value,
                    "source_section": chunk.source_section,
                    "figure_id": chunk.figure_id,
                    "panel_id": chunk.panel_id,
                    "summary": chunk.summary,
                }
            )
        return results

    def _collect_chunks(self, *, paper: Paper, figures: list[Figure]) -> list[AnnotatedChunk]:
        chunks: list[AnnotatedChunk] = []
        counter = 0
        source_pdf = self._source_pdf_path(paper)

        for section in paper.sections:
            section_title = (section.title or "").strip() or "Untitled section"
            for block in _split_blocks(section.text):
                if len(block.strip()) < 20:
                    continue
                if _looks_like_markdown_table(block):
                    normalized = block.strip()
                    chunk_type = ChunkType.TABLE if _is_valid_markdown_table(normalized) else ChunkType.PROSE
                    if chunk_type != ChunkType.TABLE:
                        started = time.perf_counter()
                        recovered = self._recover_table_chunk_with_vision(
                            malformed_table_text=normalized,
                            source_pdf=source_pdf,
                            section_title=section_title,
                        )
                        self._vision_fallback_latency_seconds += max(0.0, time.perf_counter() - started)
                        self._vision_fallback_count += 1
                        if recovered:
                            normalized = recovered
                            chunk_type = ChunkType.TABLE
                    chunks.append(
                        AnnotatedChunk(
                            chunk_id=f"chunk-{counter}",
                            text=normalized,
                            chunk_type=chunk_type,
                            source_section=section_title,
                        )
                    )
                    counter += 1
                    continue

                for piece in _chunk_text(
                    block,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                ):
                    chunks.append(
                        AnnotatedChunk(
                            chunk_id=f"chunk-{counter}",
                            text=piece.strip(),
                            chunk_type=ChunkType.PROSE,
                            source_section=section_title,
                        )
                    )
                    counter += 1

        for item in paper.supplementary_items:
            text = " ".join(
                [
                    (item.item_id or "").strip(),
                    (item.label or "").strip(),
                    (item.description or "").strip(),
                ]
            ).strip()
            if not text:
                continue
            chunks.append(
                AnnotatedChunk(
                    chunk_id=f"chunk-{counter}",
                    text=text,
                    chunk_type=ChunkType.SUPPLEMENTARY,
                    source_section="Supplementary",
                )
            )
            counter += 1

        for figure in figures:
            if figure.subfigures:
                for sub in figure.subfigures:
                    panel_text = _build_panel_chunk_text(figure, panel_label=sub.label, panel_description=sub.description)
                    if not panel_text.strip():
                        continue
                    chunks.append(
                        AnnotatedChunk(
                            chunk_id=f"chunk-{counter}",
                            text=panel_text,
                            chunk_type=ChunkType.FIGURE_CAPTION,
                            source_section=f"Figure {figure.figure_id}",
                            figure_id=figure.figure_id,
                            panel_id=sub.label,
                        )
                    )
                    counter += 1
            else:
                text = _clean_text(figure.caption)
                if not text:
                    continue
                chunks.append(
                    AnnotatedChunk(
                        chunk_id=f"chunk-{counter}",
                        text=text,
                        chunk_type=ChunkType.FIGURE_CAPTION,
                        source_section=f"Figure {figure.figure_id}",
                        figure_id=figure.figure_id,
                    )
                )
                counter += 1

        if not figures and paper.figure_captions:
            for fig_id, caption in paper.figure_captions.items():
                text = _clean_text(caption)
                if not text:
                    continue
                chunks.append(
                    AnnotatedChunk(
                        chunk_id=f"chunk-{counter}",
                        text=text,
                        chunk_type=ChunkType.FIGURE_CAPTION,
                        source_section=f"Figure {fig_id}",
                        figure_id=fig_id,
                    )
                )
                counter += 1

        return chunks

    def _source_pdf_path(self, paper: Paper) -> Optional[Path]:
        if paper.source != PaperSource.PDF:
            return None
        path = Path(paper.source_path)
        return path if path.exists() else None

    def _recover_table_chunk_with_vision(
        self,
        *,
        malformed_table_text: str,
        source_pdf: Optional[Path],
        section_title: str,
    ) -> Optional[str]:
        """Attempt table-text recovery from PDF panel crops for malformed tables."""
        if source_pdf is None:
            return None

        table_num = _extract_table_number(section_title + "\n" + malformed_table_text)
        proxy_figure_id = f"Figure {table_num}" if table_num else "Figure 1"

        try:
            panel_images_result = extract_figure_panel_images_from_pdf(
                str(source_pdf),
                figure_id=proxy_figure_id,
                caption=malformed_table_text[:240],
                include_diagnostics=True,
            )
            if isinstance(panel_images_result, tuple):
                panel_images, diagnostics = panel_images_result
                if diagnostics:
                    logger.debug(
                        "Table fallback diagnostics (%s): %s",
                        proxy_figure_id,
                        ",".join(diagnostics),
                    )
            else:
                panel_images = panel_images_result
            if not panel_images:
                return None
            extracted = llm_utils.extract_structured_data(
                model_router=self.llm_model,
                prompt=(
                    "Extract only the table content from this scientific image as compact plain text. "
                    "Preserve software names, versions, and parameters."
                ),
                schema=_VisionTableText,
                system=SYSTEM_METHODS_PARSER,
                cache=self.cache,
                image_bytes=[panel_images[0]],
            )
            text = _clean_text(extracted.extracted_text)
            return text or None
        except Exception as exc:
            logger.info("Malformed-table vision fallback failed: %s", exc)
            return None

    def _summarize_chunk(self, text: str, chunk_type: ChunkType) -> str:
        clean = _clean_text(text)
        if not clean:
            return ""
        if self.summary_builder is not None:
            try:
                out = _clean_text(self.summary_builder(clean, chunk_type))
                if out:
                    return out
            except Exception:
                pass
        try:
            summary = llm_utils.generate_text(
                model_router=self.llm_model,
                prompt=(
                    "Write one concise sentence summarizing this methods chunk. "
                    "Prioritize software, versions, commands, and parameters.\n\n"
                    f"Chunk type: {chunk_type.value}\n"
                    f"Text:\n{clean[:1800]}"
                ),
                system=SYSTEM_METHODS_PARSER,
                max_tokens=140,
                temperature=0.0,
                cache=self.cache,
            )
            summary = _clean_text(summary)
            if summary:
                return summary
        except Exception:
            pass
        return _heuristic_summary(clean)

    def _init_vector_index(self) -> None:
        self._chroma = None
        self._embedder = None
        if not self.enable_vector_index or not self._chunks:
            return
        try:
            import chromadb  # type: ignore[import]
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._embedder = SentenceTransformer(self.embedding_model)
            client_factory = getattr(chromadb, "EphemeralClient", None)
            if callable(client_factory):
                client = client_factory()
            else:
                client = chromadb.Client()
            col = client.get_or_create_collection(name=self._collection_name)
            ids = [entry.chunk.chunk_id for entry in self._chunks]
            docs = [entry.searchable_text for entry in self._chunks]
            metas = [
                {
                    "chunk_id": entry.chunk.chunk_id,
                    "chunk_type": entry.chunk.chunk_type.value,
                    "source_section": entry.chunk.source_section,
                    "figure_id": entry.chunk.figure_id or "",
                    "panel_id": entry.chunk.panel_id or "",
                    "source": f"paper:{entry.chunk.source_section or 'unknown'}",
                }
                for entry in self._chunks
            ]
            embeddings = self._embedder.encode(docs).tolist()
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            self._chroma = col
        except Exception as exc:
            logger.info("PaperRAG vector backend unavailable; lexical retrieval only: %s", exc)
            self._chroma = None
            self._embedder = None


def merge_retrieval_results(
    *,
    paper_hits: list[dict[str, object]],
    protocol_hits: list[dict[str, object]],
    top_k: int,
    paper_bias: float = 0.10,
    protocol_bias: float = 0.0,
) -> list[dict[str, object]]:
    """Merge per-paper and protocol retrieval with paper-first score bias."""
    if top_k <= 0:
        return []

    def _with_normalized_scores(
        hits: list[dict[str, object]],
        *,
        bias: float,
        source_type: str,
    ) -> list[dict[str, object]]:
        if not hits:
            return []
        raw_scores: list[float] = []
        for i, h in enumerate(hits):
            if isinstance(h.get("score"), (float, int)):
                raw_scores.append(float(h["score"]))
            else:
                raw_scores.append(max(0.0, 1.0 - (float(i) / float(max(len(hits), 1)))))
        s_min = min(raw_scores)
        s_max = max(raw_scores)
        out: list[dict[str, object]] = []
        for h, raw in zip(hits, raw_scores):
            if s_max == s_min:
                norm = 1.0
            else:
                norm = (raw - s_min) / (s_max - s_min)
            entry = dict(h)
            entry["source_type"] = source_type
            entry["normalized_score"] = norm
            entry["final_score"] = norm + bias
            out.append(entry)
        return out

    merged = _with_normalized_scores(
        paper_hits,
        bias=paper_bias,
        source_type="paper",
    ) + _with_normalized_scores(
        protocol_hits,
        bias=protocol_bias,
        source_type="protocol",
    )
    merged.sort(
        key=lambda h: (
            float(h.get("final_score", 0.0)),
            1 if h.get("source_type") == "paper" else 0,
        ),
        reverse=True,
    )
    return merged[:top_k]


def _split_blocks(text: str) -> list[str]:
    clean = (text or "").strip()
    if not clean:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", clean) if b.strip()]
    return blocks


def _looks_like_markdown_table(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) < 2:
        return False
    separator = any(re.match(r"^\|?\s*[:\-]+(?:\s*\|\s*[:\-]+)+\s*\|?$", ln) for ln in lines)
    return separator or len(pipe_lines) >= 3


def _is_valid_markdown_table(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "|" in ln]
    if len(lines) < 2:
        return False

    parsed_rows: list[list[str]] = []
    for ln in lines:
        row = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(row) < 2:
            return False
        parsed_rows.append(row)

    col_count = len(parsed_rows[0])
    if col_count < 2:
        return False
    if any(len(r) != col_count for r in parsed_rows):
        return False

    all_cells = [cell for row in parsed_rows for cell in row]
    non_empty = [cell for cell in all_cells if cell and not re.fullmatch(r"[:\-\s]+", cell)]
    ratio = len(non_empty) / max(len(all_cells), 1)
    return ratio >= 0.35


def _extract_table_number(text: str) -> Optional[int]:
    m = re.search(r"(?i)table\s*S?(\d+)", text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _heuristic_summary(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    if not sentence:
        return ""
    sentence = sentence[:220].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _build_panel_chunk_text(figure: Figure, panel_label: str, panel_description: str) -> str:
    parts = [
        f"{figure.figure_id} panel {panel_label}",
        _clean_text(panel_description),
        _clean_text(figure.caption),
    ]
    return " ".join(p for p in parts if p)
