"""Unit tests for Phase 5: SRAParser.

Testing strategy:
- Tier 1 (no network): accession validation, helper functions (_df_to_samples,
  _unique_col, _first_col, _infer_experiment_type), _parse_project / _parse_experiment
  / _parse_run with mocked pysradb.
- Tier 3 (live): marked @pytest.mark.live, skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from researcher_ai.models.dataset import DataSource, SRADataset, SampleMetadata
from researcher_ai.parsers.data.sra_parser import (
    SRAParser,
    _df_to_samples,
    _first_col,
    _infer_experiment_type,
    _unique_col,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_parser() -> SRAParser:
    return SRAParser()


def _project_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_accession": "SRR2071346",
            "experiment_accession": "SRX1085399",
            "sample_accession": "SRS993095",
            "study_accession": "SRP062554",
            "study_title": "eCLIP of HNRNPC in HEK293T",
            "experiment_title": "HNRNPC IP rep1",
            "sample_title": "HEK293T HNRNPC IP rep1",
            "scientific_name": "Homo sapiens",
            "library_selection": "IP",
            "library_layout": "PAIRED",
            "instrument_model": "Illumina HiSeq 2000",
            "instrument_platform": "ILLUMINA",
            "library_source": "TRANSCRIPTOMIC",
            "library_strategy": "OTHER",
            "total_spots": "25000000",
            "total_bases": "3750000000",
            "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/fastq/SRR207/SRR2071346_1.fastq.gz",
        },
        {
            "run_accession": "SRR2071347",
            "experiment_accession": "SRX1085400",
            "sample_accession": "SRS993096",
            "study_accession": "SRP062554",
            "study_title": "eCLIP of HNRNPC in HEK293T",
            "experiment_title": "HNRNPC IP rep2",
            "sample_title": "HEK293T HNRNPC IP rep2",
            "scientific_name": "Homo sapiens",
            "library_selection": "IP",
            "library_layout": "PAIRED",
            "instrument_model": "Illumina HiSeq 2000",
            "instrument_platform": "ILLUMINA",
            "library_source": "TRANSCRIPTOMIC",
            "library_strategy": "OTHER",
            "total_spots": "22000000",
            "total_bases": "3300000000",
            "fastq_ftp": "",
        },
    ])


def _make_parser_with_mock_db(df: pd.DataFrame | None = None) -> SRAParser:
    """Return SRAParser with mocked pysradb SRAweb instance."""
    parser = _make_parser()
    mock_db = MagicMock()
    mock_db.sra_metadata.return_value = df if df is not None else _project_df()
    parser._db = mock_db
    return parser


# ---------------------------------------------------------------------------
# TestSRAParserValidation
# ---------------------------------------------------------------------------

class TestSRAParserValidation:
    """validate_accession — pure regex, no network."""

    @pytest.mark.parametrize("acc", [
        # NCBI
        "SRP062554", "SRX1085399", "SRR2071346",
        # EBI
        "ERP001942", "ERX000001", "ERR000001",
        # DDBJ
        "DRP000001", "DRX000001", "DRR000001",
        # Case-insensitive
        "srp062554", "Srr2071346",
    ])
    def test_valid_accessions(self, acc):
        assert SRAParser().validate_accession(acc) is True

    @pytest.mark.parametrize("acc", [
        "GSE72987",     # GEO
        "PRJNA123",     # BioProject
        "SRP",          # missing digits
        "SRA123456",    # wrong prefix
        "",
        "123456",
    ])
    def test_invalid_accessions(self, acc):
        assert SRAParser().validate_accession(acc) is False

    def test_invalid_raises_on_parse(self):
        with pytest.raises(ValueError, match="Invalid SRA accession"):
            SRAParser().parse("GSE72987")


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Module-level helper functions — pure, no network."""

    def test_unique_col_deduplicates(self):
        df = pd.DataFrame({"a": ["x", "x", "y", None]})
        result = _unique_col(df, "a")
        assert sorted(result) == ["x", "y"]

    def test_unique_col_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        assert _unique_col(df, "missing") == []

    def test_first_col_returns_first_non_null(self):
        df = pd.DataFrame({"a": [None, "second", "third"]})
        assert _first_col(df, "a") == "second"

    def test_first_col_all_null_returns_none(self):
        df = pd.DataFrame({"a": [None, None]})
        assert _first_col(df, "a") is None

    def test_first_col_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        assert _first_col(df, "missing") is None

    def test_infer_experiment_type_most_common(self):
        df = pd.DataFrame({"library_strategy": ["RNA-Seq", "RNA-Seq", "OTHER"]})
        assert _infer_experiment_type(df) == "RNA-Seq"

    def test_infer_experiment_type_missing_col(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        assert _infer_experiment_type(df) is None


# ---------------------------------------------------------------------------
# TestDfToSamples
# ---------------------------------------------------------------------------

class TestDfToSamples:
    """_df_to_samples — pure DataFrame → SampleMetadata conversion."""

    def test_returns_correct_count(self):
        assert len(_df_to_samples(_project_df())) == 2

    def test_run_accession_used_as_sample_id(self):
        samples = _df_to_samples(_project_df())
        assert samples[0].sample_id == "SRR2071346"

    def test_organism_populated(self):
        samples = _df_to_samples(_project_df())
        assert samples[0].organism == "Homo sapiens"

    def test_platform_from_instrument_model(self):
        samples = _df_to_samples(_project_df())
        assert "Illumina" in samples[0].platform

    def test_fastq_url_ftp_prefixed(self):
        samples = _df_to_samples(_project_df())
        assert samples[0].fastq_urls[0].startswith("ftp://")

    def test_empty_fastq_not_added(self):
        samples = _df_to_samples(_project_df())
        assert samples[1].fastq_urls == []

    def test_reserved_cols_excluded_from_attributes(self):
        samples = _df_to_samples(_project_df())
        reserved = {"run_accession", "study_accession", "scientific_name"}
        for key in reserved:
            assert key not in samples[0].attributes


# ---------------------------------------------------------------------------
# TestParseProject
# ---------------------------------------------------------------------------

class TestParseProject:
    """SRAParser._parse_project — mocked pysradb."""

    def test_returns_sra_dataset(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert isinstance(ds, SRADataset)

    def test_source_is_sra(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert ds.source == DataSource.SRA

    def test_srp_set(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert ds.srp == "SRP062554"

    def test_srx_list_populated(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert "SRX1085399" in ds.srx_list
        assert "SRX1085400" in ds.srx_list

    def test_srr_list_populated(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert "SRR2071346" in ds.srr_list

    def test_samples_populated(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert len(ds.samples) == 2

    def test_organism_populated(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        assert ds.organism == "Homo sapiens"

    def test_empty_df_returns_stub(self):
        parser = _make_parser_with_mock_db(df=pd.DataFrame())
        ds = parser.parse("SRP062554")
        assert ds.srp == "SRP062554"
        assert ds.samples == []

    def test_network_failure_returns_stub(self):
        parser = _make_parser()
        mock_db = MagicMock()
        mock_db.sra_metadata.side_effect = RuntimeError("network down")
        parser._db = mock_db
        ds = parser.parse("SRP062554")
        assert isinstance(ds, SRADataset)
        assert ds.samples == []


# ---------------------------------------------------------------------------
# TestParseExperiment
# ---------------------------------------------------------------------------

class TestParseExperiment:
    """SRAParser._parse_experiment — mocked pysradb."""

    def test_returns_sra_dataset(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRX1085399")
        assert isinstance(ds, SRADataset)

    def test_srx_list_contains_input_id(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRX1085399")
        assert "SRX1085399" in ds.srx_list

    def test_srp_inherited_from_df(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRX1085399")
        assert ds.srp == "SRP062554"


# ---------------------------------------------------------------------------
# TestParseRun
# ---------------------------------------------------------------------------

class TestParseRun:
    """SRAParser._parse_run — mocked pysradb."""

    def test_returns_sra_dataset(self):
        # Single-row df
        parser = _make_parser_with_mock_db(df=_project_df().head(1))
        ds = parser.parse("SRR2071346")
        assert isinstance(ds, SRADataset)

    def test_srr_list_contains_input_id(self):
        parser = _make_parser_with_mock_db(df=_project_df().head(1))
        ds = parser.parse("SRR2071346")
        assert "SRR2071346" in ds.srr_list

    def test_total_samples_is_one(self):
        parser = _make_parser_with_mock_db(df=_project_df().head(1))
        ds = parser.parse("SRR2071346")
        assert ds.total_samples == 1


# ---------------------------------------------------------------------------
# TestSRADatasetJsonRoundtrip
# ---------------------------------------------------------------------------

class TestSRADatasetJsonRoundtrip:
    def test_roundtrip_preserves_all_fields(self):
        parser = _make_parser_with_mock_db()
        ds = parser.parse("SRP062554")
        restored = SRADataset.model_validate_json(ds.model_dump_json())
        assert restored.accession == ds.accession
        assert restored.srx_list == ds.srx_list
        assert len(restored.samples) == len(ds.samples)


# ---------------------------------------------------------------------------
# Live tests (require network, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestSRAParserLive:
    """Live integration tests — require network access."""

    def test_parse_srp062554(self):
        ds = SRAParser().parse("SRP062554")
        assert ds.accession == "SRP062554"
        assert ds.total_samples > 0
        assert ds.organism is not None
