"""BioC utility tests for pubmed.py Phase 1 integration."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from researcher_ai.utils import pubmed


def _collection(doc_id: str, *, pmid: str = "", pmcid: str = "") -> dict:
    infons = {}
    if pmid:
        infons["article-id_pmid"] = pmid
    if pmcid:
        infons["article-id_pmc"] = pmcid
    return {
        "documents": [
            {
                "id": doc_id,
                "infons": infons,
                "passages": [],
            }
        ]
    }


class TestNormalizeCollections:
    def test_object_payload_normalizes_to_single_list(self):
        payload = {"documents": []}
        out = pubmed.normalize_bioc_collections(payload)
        assert len(out) == 1
        assert out[0]["documents"] == []

    def test_list_payload_filters_non_dict(self):
        payload = [{"documents": []}, "x", 1, {"documents": [1]}]
        out = pubmed.normalize_bioc_collections(payload)
        assert len(out) == 2
        assert isinstance(out[0], dict)
        assert isinstance(out[1], dict)

    def test_invalid_payload_returns_empty(self):
        assert pubmed.normalize_bioc_collections("bad") == []


class TestCanonicalSelection:
    def test_selects_collection_by_pmid_match(self):
        c1 = _collection("PMC111", pmid="100")
        c2 = _collection("PMC222", pmid="200")
        chosen = pubmed.select_canonical_bioc_collection([c1, c2], pmid="200")
        assert chosen is c2

    def test_selects_collection_by_pmcid_match(self):
        c1 = _collection("PMC111")
        c2 = _collection("PMC999")
        chosen = pubmed.select_canonical_bioc_collection([c1, c2], pmcid="PMC999")
        assert chosen is c2

    def test_falls_back_to_first_when_no_match(self):
        c1 = _collection("PMC111")
        c2 = _collection("PMC222")
        chosen = pubmed.select_canonical_bioc_collection([c1, c2], pmid="999")
        assert chosen is c1


class TestPassageSelectors:
    def test_methods_selector_matches_aliases(self):
        assert pubmed.bioc_methods_section_selector("METHODS")
        assert pubmed.bioc_methods_section_selector("materials_and_methods")
        assert pubmed.bioc_methods_section_selector("STAR METHODS")
        assert not pubmed.bioc_methods_section_selector("RESULTS")

    def test_results_selector_matches_aliases(self):
        assert pubmed.bioc_results_section_selector("RESULTS")
        assert pubmed.bioc_results_section_selector("results_discussion")
        assert not pubmed.bioc_results_section_selector("METHODS")

    def test_extract_bioc_passages_with_selector(self):
        collection = {
            "documents": [
                {
                    "id": "PMC1",
                    "passages": [
                        {"infons": {"section_type": "METHODS"}, "text": "m"},
                        {"infons": {"section_type": "RESULTS"}, "text": "r"},
                        {"infons": {"section_type": "FIG"}, "text": "f"},
                    ],
                }
            ]
        }
        methods = pubmed.extract_bioc_passages(collection, pubmed.bioc_methods_section_selector)
        results = pubmed.extract_bioc_passages(collection, pubmed.bioc_results_section_selector)
        assert len(methods) == 1
        assert methods[0]["text"] == "m"
        assert len(results) == 1
        assert results[0]["text"] == "r"


class TestFigureIdMapping:
    def test_figure_id_from_caption_wins(self):
        fid = pubmed.map_bioc_figure_id("F5", "Figure 2. Caption text", fallback_index=1)
        assert fid == "Figure 2"

    def test_fallback_uses_literal_f_digit(self):
        fid = pubmed.map_bioc_figure_id("F7", "No explicit figure token", fallback_index=1)
        assert fid == "Figure 7"

    def test_irregular_sequence_uses_index(self):
        fid = pubmed.map_bioc_figure_id("F9", "No token", fallback_index=2, irregular_f_sequence=True)
        assert fid == "Figure 2"


class TestBiocCacheAndFlag:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RESEARCHER_AI_BIOC_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("RESEARCHER_AI_BIOC_CACHE_TTL_SEC", "3600")
        payload = {"documents": [{"id": "PMC1"}]}
        pubmed._write_bioc_cache("xid", "unicode", payload)
        loaded = pubmed._read_bioc_cache("xid", "unicode")
        assert loaded == payload

    def test_flag_off_bypasses_cache_and_network(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RESEARCHER_AI_BIOC_ENABLED", "off")
        monkeypatch.setenv("RESEARCHER_AI_BIOC_CACHE_DIR", str(tmp_path))
        pubmed._write_bioc_cache("39303722", "unicode", {"documents": [{"id": "PMC11633308"}]})

        with patch("researcher_ai.utils.pubmed._read_bioc_cache") as mock_read, patch(
            "researcher_ai.utils.pubmed._get"
        ) as mock_get:
            out = pubmed.fetch_bioc_json_for_paper("39303722", None, encoding="unicode")
            assert out == {}
            mock_read.assert_not_called()
            mock_get.assert_not_called()

    def test_fetch_uses_pmid_then_selects_canonical_collection(self, monkeypatch):
        monkeypatch.setenv("RESEARCHER_AI_BIOC_ENABLED", "on")
        payload = [
            _collection("PMC0000000", pmid="123"),
            _collection("PMC11633308", pmid="39303722", pmcid="PMC11633308"),
        ]

        with patch("researcher_ai.utils.pubmed._read_bioc_cache", return_value=None), patch(
            "researcher_ai.utils.pubmed._write_bioc_cache"
        ), patch("researcher_ai.utils.pubmed._get", return_value=json.dumps(payload)) as mock_get:
            out = pubmed.fetch_bioc_json_for_paper("39303722", None, encoding="unicode")
            assert out["documents"][0]["id"] == "PMC11633308"
            called_url = mock_get.call_args[0][0]
            assert "/BioC_json/39303722/unicode" in called_url

    def test_fetch_falls_back_to_resolved_pmcid(self, monkeypatch):
        monkeypatch.setenv("RESEARCHER_AI_BIOC_ENABLED", "on")
        payload = [_collection("PMC11633308", pmid="39303722", pmcid="PMC11633308")]

        def _fake_get(url, *args, **kwargs):
            if "/BioC_json/39303722/unicode" in url:
                raise RuntimeError("pmid route unavailable")
            return json.dumps(payload)

        with patch("researcher_ai.utils.pubmed._read_bioc_cache", return_value=None), patch(
            "researcher_ai.utils.pubmed._write_bioc_cache"
        ), patch("researcher_ai.utils.pubmed._get", side_effect=_fake_get) as mock_get, patch(
            "researcher_ai.utils.pubmed.resolve_pmid_to_pmcid_idconv",
            return_value="PMC11633308",
        ):
            out = pubmed.fetch_bioc_json_for_paper("39303722", None, encoding="unicode")
            assert out["documents"][0]["id"] == "PMC11633308"
            urls = [call.args[0] for call in mock_get.call_args_list]
            assert any("/BioC_json/11633308/unicode" in u for u in urls)
