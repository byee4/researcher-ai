from __future__ import annotations

from researcher_ai.parsers.methods_parser import MethodsParser


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
    hits = parser._iterative_retrieval_loop(
        assay_name="RNA-seq",
        skeleton_stages=["align"],
        max_refinement_rounds=2,
    )

    joined = "\n".join(str(h.get("text", "")) for h in hits)
    assert "--outSAMtype" in joined


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
