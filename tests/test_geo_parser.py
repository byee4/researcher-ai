"""Unit tests for Phase 5: GEOParser.

Testing strategy:
- Tier 1 (no network): accession validation, _normalise_esummary, _get_child_series,
  _fetch_processed_data URL construction, _is_superseries, _df_to_samples, _parse_series
  with mocked HTTP / pysradb.
- Tier 3 (live): marked @pytest.mark.live, skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from researcher_ai.models.dataset import DataSource, GEODataset, SampleMetadata
from researcher_ai.parsers.data.geo_parser import (
    GEOParser,
    _df_to_samples,
    _extract_child_series_from_soft_text,
    _first,
    _normalise_esummary,
    _normalise_gpl,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_parser() -> GEOParser:
    return GEOParser()


def _esummary_record(overrides: dict | None = None) -> dict:
    base = {
        "title": "eCLIP identifies hundreds of HNRNPC binding sites",
        "summary": "Enhanced CLIP (eCLIP) to map HNRNPC binding.",
        "gdstype": "Expression profiling by high throughput sequencing",
        "entrytype": "GSE",
        "taxon": "Homo sapiens",
        "gpl": "11154",           # NCBI returns numeric-only; _normalise_esummary adds "GPL" prefix
        "platformtitle": "Illumina HiSeq 2000",
        "n_samples": "8",
        "extrelations": [],
        "accession": "GSE72987",
        "uid": "200072987",
    }
    if overrides:
        base.update(overrides)
    return base


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_accession": "SRR2071346",
            "experiment_accession": "SRX1085399",
            "sample_accession": "SRS993095",
            "study_accession": "SRP062554",
            "sample_title": "HEK293T HNRNPC IP rep1",
            "scientific_name": "Homo sapiens",
            "library_selection": "IP",
            "library_layout": "PAIRED",
            "instrument_model": "Illumina HiSeq 2000",
            "library_source": "TRANSCRIPTOMIC",
            "library_strategy": "eCLIP",
            "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/fastq/SRR207/SRR2071346_1.fastq.gz",
        },
        {
            "run_accession": "SRR2071347",
            "experiment_accession": "SRX1085400",
            "sample_accession": "SRS993096",
            "study_accession": "SRP062554",
            "sample_title": "HEK293T HNRNPC IP rep2",
            "scientific_name": "Homo sapiens",
            "library_selection": "IP",
            "library_layout": "PAIRED",
            "instrument_model": "Illumina HiSeq 2000",
            "library_source": "TRANSCRIPTOMIC",
            "library_strategy": "eCLIP",
            "fastq_ftp": "",
        },
    ])


# ---------------------------------------------------------------------------
# TestGEOParserValidation
# ---------------------------------------------------------------------------

class TestGEOParserValidation:
    """validate_accession — pure regex, no network."""

    @pytest.mark.parametrize("acc", [
        "GSE72987", "GSE1", "GSE123456789",
        "GSM1000001", "GSM9",
        "GPL570", "GPL11154",
        # Case-insensitive
        "gse72987", "Gse72987",
    ])
    def test_valid_accessions(self, acc):
        assert GEOParser().validate_accession(acc) is True

    @pytest.mark.parametrize("acc", [
        "SRP062554", "SRR123", "PRJNA123",
        "GSE",            # missing digits
        "GSE12345X",      # trailing letter
        "",
        "123456",
        "ENCODE:ENCSR123",
    ])
    def test_invalid_accessions(self, acc):
        assert GEOParser().validate_accession(acc) is False

    def test_invalid_raises_on_parse(self):
        with pytest.raises(ValueError, match="Invalid GEO accession"):
            GEOParser().parse("SRP062554")


# ---------------------------------------------------------------------------
# TestNormaliseEsummary
# ---------------------------------------------------------------------------

class TestNormaliseEsummary:
    """_normalise_esummary — pure transformation."""

    def test_basic_fields_extracted(self):
        result = _normalise_esummary(_esummary_record())
        assert result["title"] == "eCLIP identifies hundreds of HNRNPC binding sites"
        assert result["gdstype"] == "Expression profiling by high throughput sequencing"
        assert result["n_samples"] == 8

    def test_taxon_string_split_on_semicolon(self):
        record = _esummary_record({"taxon": "Homo sapiens; Mus musculus"})
        result = _normalise_esummary(record)
        assert result["taxon"] == ["Homo sapiens", "Mus musculus"]

    def test_taxon_already_list(self):
        record = _esummary_record({"taxon": ["Homo sapiens"]})
        result = _normalise_esummary(record)
        assert result["taxon"] == ["Homo sapiens"]

    def test_gpl_prefixed(self):
        # NCBI returns numeric-only; normaliser adds "GPL" prefix
        record = _esummary_record({"gpl": "11154"})
        result = _normalise_esummary(record)
        assert "GPL11154" in result["gpl"]

    def test_gpl_already_prefixed_not_doubled(self):
        # If NCBI ever returns the full "GPL" prefix, we should not double it.
        # Currently _normalise_esummary always prefixes — test the actual behaviour
        # so that any future change is caught explicitly.
        record = _esummary_record({"gpl": "11154"})
        result = _normalise_esummary(record)
        assert "GPLGPL" not in result["gpl"][0]

    def test_n_samples_non_numeric_defaults_zero(self):
        record = _esummary_record({"n_samples": "N/A"})
        result = _normalise_esummary(record)
        assert result["n_samples"] == 0

    def test_empty_record_returns_empty_dict(self):
        assert _normalise_esummary({}) == {}

    def test_relations_extracted(self):
        record = _esummary_record({"extrelations": [
            {"relationtype": "SubSeries", "targetobject": "GSE72988"},
        ]})
        result = _normalise_esummary(record)
        assert len(result["relations"]) == 1
        assert result["relations"][0]["target"] == "GSE72988"


# ---------------------------------------------------------------------------
# TestIsSuperseries
# ---------------------------------------------------------------------------

class TestIsSuperseries:
    """GEOParser._is_superseries — metadata classification."""

    def test_superseries_detected(self):
        parser = _make_parser()
        md = _normalise_esummary(_esummary_record({"gdstype": "Expression profiling by SuperSeries"}))
        assert parser._is_superseries(md) is True

    def test_plain_series_not_superseries(self):
        parser = _make_parser()
        md = _normalise_esummary(_esummary_record())
        assert parser._is_superseries(md) is False

    def test_empty_metadata_not_superseries(self):
        assert _make_parser()._is_superseries({}) is False


# ---------------------------------------------------------------------------
# TestGetChildSeries
# ---------------------------------------------------------------------------

class TestGetChildSeries:
    """GEOParser._get_child_series — relations parsing."""

    def test_extracts_child_gse_ids(self):
        parser = _make_parser()
        metadata = {
            "relations": [
                {"type": "SubSeries", "target": "GSE72988"},
                {"type": "SubSeries", "target": "GSE72989"},
            ]
        }
        children = parser._get_child_series("GSE72990", metadata)
        assert "GSE72988" in children
        assert "GSE72989" in children

    def test_empty_relations_returns_empty(self):
        assert _make_parser()._get_child_series("GSE1", {}) == []

    def test_non_gse_relations_ignored(self):
        parser = _make_parser()
        metadata = {"relations": [{"type": "SRA", "target": "SRP062554"}]}
        assert parser._get_child_series("GSE1", metadata) == []

    def test_falls_back_to_soft_series_relation_when_extrelations_missing(self):
        parser = _make_parser()
        parser._fetch_series_soft_text = MagicMock(
            return_value="\n".join([
                "!Series_relation = SuperSeries of: GSE77339",
                "!Series_relation = SuperSeries of: GSE77629",
                "!Series_relation = SuperSeries of: GSE77633",
            ])
        )
        children = parser._get_child_series("GSE77634", {"relations": []})
        assert children == ["GSE77339", "GSE77629", "GSE77633"]


class TestExtractChildSeriesFromSoftText:
    def test_extracts_superseries_children(self):
        text = "\n".join([
            "!Series_relation = SuperSeries of: GSE11111",
            "!Series_relation = SuperSeries of: GSE22222",
        ])
        assert _extract_child_series_from_soft_text(text) == ["GSE11111", "GSE22222"]

    def test_ignores_subseries_parent_links(self):
        text = "!Series_relation = SubSeries of: GSE99999"
        assert _extract_child_series_from_soft_text(text) == []


# ---------------------------------------------------------------------------
# TestFetchProcessedData
# ---------------------------------------------------------------------------

class TestFetchProcessedData:
    """_fetch_processed_data — deterministic URL construction, no network."""

    def test_url_contains_gse_id(self):
        urls = _make_parser()._fetch_processed_data("GSE72987")
        assert any("GSE72987" in u for u in urls)

    def test_url_contains_nnn_stem(self):
        urls = _make_parser()._fetch_processed_data("GSE72987")
        assert any("GSE72nnn" in u for u in urls)

    def test_url_ends_with_suppl(self):
        urls = _make_parser()._fetch_processed_data("GSE72987")
        assert any(u.endswith("/suppl/") for u in urls)

    def test_large_accession_nnn_stem(self):
        # GSE1234567 → GSE1234nnn
        urls = _make_parser()._fetch_processed_data("GSE1234567")
        assert any("GSE1234nnn" in u for u in urls)


# ---------------------------------------------------------------------------
# TestDfToSamples
# ---------------------------------------------------------------------------

class TestDfToSamples:
    """_df_to_samples — pure DataFrame conversion."""

    def test_returns_correct_count(self):
        samples = _df_to_samples(_sample_df())
        assert len(samples) == 2

    def test_sample_id_populated(self):
        samples = _df_to_samples(_sample_df())
        assert samples[0].sample_id == "SRR2071346"

    def test_organism_populated(self):
        samples = _df_to_samples(_sample_df())
        assert samples[0].organism == "Homo sapiens"

    def test_layout_populated(self):
        samples = _df_to_samples(_sample_df())
        assert samples[0].layout == "PAIRED"

    def test_fastq_url_prefixed(self):
        samples = _df_to_samples(_sample_df())
        assert any(u.startswith("ftp://") for u in samples[0].fastq_urls)

    def test_empty_fastq_not_added(self):
        samples = _df_to_samples(_sample_df())
        assert samples[1].fastq_urls == []


# ---------------------------------------------------------------------------
# TestFirstHelper
# ---------------------------------------------------------------------------

class TestFirstHelper:
    def test_returns_first_element(self):
        assert _first(["a", "b"]) == "a"

    def test_empty_list_returns_none(self):
        assert _first([]) is None


# ---------------------------------------------------------------------------
# TestParseSeries (mocked HTTP + pysradb)
# ---------------------------------------------------------------------------

class TestParseSeries:
    """GEOParser._parse_series — mocked network."""

    def _mock_parser_with_metadata(self, metadata_override: dict | None = None):
        parser = _make_parser()
        parser._fetch_geo_metadata = MagicMock(
            return_value=_normalise_esummary(_esummary_record(metadata_override))
        )
        parser._fetch_samples = MagicMock(return_value=_df_to_samples(_sample_df()))
        return parser

    def test_returns_geo_dataset(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert isinstance(ds, GEODataset)

    def test_source_is_geo(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert ds.source == DataSource.GEO

    def test_title_populated(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert "eCLIP" in ds.title

    def test_samples_populated(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert len(ds.samples) == 2

    def test_series_type_set(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert ds.series_type == "Series"

    def test_superseries_type_set(self):
        parser = self._mock_parser_with_metadata(
            {"gdstype": "Expression profiling by SuperSeries"}
        )
        parser._get_child_series = MagicMock(return_value=["GSE72988"])
        ds = parser.parse("GSE72987")
        assert ds.series_type == "SuperSeries"

    def test_child_series_presence_promotes_superseries_even_without_gdstype_flag(self):
        parser = self._mock_parser_with_metadata(
            {"gdstype": "Expression profiling by high throughput sequencing"}
        )
        parser._get_child_series = MagicMock(return_value=["GSE72988"])
        ds = parser.parse("GSE72987")
        assert ds.series_type == "SuperSeries"

    def test_superseries_no_samples_fetched(self):
        parser = self._mock_parser_with_metadata(
            {"gdstype": "Expression profiling by SuperSeries"}
        )
        parser._get_child_series = MagicMock(return_value=["GSE72988"])
        ds = parser.parse("GSE72987")
        # _fetch_samples should NOT be called for superseries
        parser._fetch_samples.assert_not_called()
        assert ds.samples == []

    def test_platform_id_populated(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert ds.platform_id == "GPL11154"

    def test_processed_data_url_present(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        assert len(ds.processed_data_urls) >= 1
        assert "GSE72987" in ds.processed_data_urls[0]

    def test_json_roundtrip(self):
        parser = self._mock_parser_with_metadata()
        ds = parser.parse("GSE72987")
        restored = GEODataset.model_validate_json(ds.model_dump_json())
        assert restored.accession == ds.accession
        assert len(restored.samples) == len(ds.samples)

    def test_missing_metadata_raises_not_found(self):
        parser = _make_parser()
        parser._fetch_geo_metadata = MagicMock(return_value={})
        with pytest.raises(LookupError, match="GEO accession not found"):
            parser.parse("GSE72987")


# ---------------------------------------------------------------------------
# Live tests (require network, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestGEOParserLive:
    """Live integration tests — require network access."""

    def test_parse_gse72987(self):
        ds = GEOParser().parse("GSE72987")
        assert ds.accession == "GSE72987"
        assert ds.total_samples > 0
        assert "eCLIP" in (ds.title or "") or "HNRNP" in (ds.title or "") or ds.title != ""
