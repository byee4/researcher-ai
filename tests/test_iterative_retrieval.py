from __future__ import annotations

from pathlib import Path

from researcher_ai.parsers.methods_parser import MethodsParser
from researcher_ai.models.paper import Paper, PaperSource, PaperType, Section
from researcher_ai.utils.paper_indexer import PaperRAGStore


def _make_parser() -> MethodsParser:
    parser = MethodsParser.__new__(MethodsParser)
    parser.llm_model = "test-model"
    parser.cache = None
    return parser


def test_detect_missing_fields_and_stage_completeness():
    parser = _make_parser()
    hits = [{"text": "Reads were aligned with STAR."}]
    missing = parser._detect_missing_fields(hits, "align")
    assert "parameters" in missing
    assert parser._stage_fields_complete(hits, "align") is False

    completed_hits = [{"text": "STAR was run with --outSAMtype BAM SortedByCoordinate."}]
    assert parser._detect_missing_fields(completed_hits, "align") == []
    assert parser._stage_fields_complete(completed_hits, "align") is True


def test_iterative_retrieval_refines_when_parameters_missing(monkeypatch):
    parser = _make_parser()

    def _fake_query(query: str, top_k: int = 3):  # noqa: ARG001
        q = query.lower()
        if "settings arguments" in q:
            return [{"text": "STAR --outSAMtype BAM SortedByCoordinate", "source": "paper"}]
        return [{"text": "STAR aligner used for mapping", "source": "paper"}]

    monkeypatch.setattr(parser, "_query_evidence_hits", _fake_query)
    hits, warnings = parser._iterative_retrieval_loop(
        assay_name="RNA-seq",
        skeleton_stages=["align"],
        max_refinement_rounds=2,
    )

    joined = "\n".join(str(h.get("text", "")) for h in hits)
    assert "--outSAMtype" in joined
    assert warnings == []


def test_render_retrieved_context_preserves_provenance_tags():
    parser = _make_parser()
    text = parser._render_retrieved_context(
        [
            {
                "source": "paper:Methods",
                "source_type": "paper",
                "chunk_type": "table",
                "text": "STAR --runThreadN 8",
            }
        ],
        max_chars=300,
    )
    assert "[paper:paper:Methods" in text
    assert "chunk_type=table" in text


def test_iterative_retrieval_uses_recovered_table_evidence(monkeypatch, tmp_path: Path):
    parser = _make_parser()
    parser.protocol_rag = type("ProtocolStub", (), {"query": lambda self, query, top_k=3: []})()
    parser.paper_rag = PaperRAGStore(
        llm_model="test-model",
        enable_vector_index=False,
        summary_builder=lambda text, chunk_type: "",
    )

    malformed_table = (
        "| Tool | Version |\n"
        "| STAR | 2.7.11a |\n"
        "| Param | --outSAMtype BAM SortedByCoordinate |\n"
    )
    paper = Paper(
        title="Malformed table methods",
        authors=["A"],
        abstract="",
        source=PaperSource.PDF,
        source_path=str(tmp_path / "paper.pdf"),
        paper_type=PaperType.EXPERIMENTAL,
        sections=[Section(title="Methods", text=malformed_table)],
    )

    monkeypatch.setattr(
        parser.paper_rag,
        "_recover_table_chunk_with_vision",
        lambda *, malformed_table_text, source_pdf, section_title: (
            "| Tool | Version |\n"
            "| --- | --- |\n"
            "| STAR | 2.7.11a |\n"
            "| Param | --outSAMtype BAM SortedByCoordinate |\n"
        ),
    )
    parser.paper_rag.build_from(paper=paper, figures=[])

    hits, warnings = parser._iterative_retrieval_loop(
        assay_name="RNA-seq",
        skeleton_stages=["align"],
        max_refinement_rounds=1,
    )
    joined = "\n".join(str(h.get("text", "")) for h in hits)
    assert "--outSAMtype BAM SortedByCoordinate" in joined
    assert warnings == []


def test_iterative_retrieval_emits_circuit_breaker_warning_when_unresolved(monkeypatch):
    parser = _make_parser()
    monkeypatch.setattr(
        parser,
        "_query_evidence_hits",
        lambda query, top_k=3: [{"text": "Alignment was performed.", "source": "paper"}],
    )
    _, warnings = parser._iterative_retrieval_loop(
        assay_name="RNA-seq",
        skeleton_stages=["align"],
        max_refinement_rounds=1,
    )
    assert any("retrieval_circuit_breaker" in msg for msg in warnings)
