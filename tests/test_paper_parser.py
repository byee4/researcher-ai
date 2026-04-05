"""Unit tests for Phase 2: Paper Parser and utilities.

Testing strategy:
- pdf.py:    Pure functions tested directly (no pdfplumber needed for logic tests).
- pubmed.py: XML parsers tested against inline fixture XML; network calls mocked.
- llm.py:    Cache tested directly; ask_claude/ask_claude_structured mocked.
- paper_parser.py: PaperParser tested with mocked network + LLM dependencies.
  Source type detection is pure logic — tested without mocking.
"""

import json
import textwrap
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from researcher_ai.models.paper import Paper, PaperSource, PaperType, Section
from researcher_ai.utils.pdf import (
    detect_section_boundaries,
    extract_figure_ids_from_text,
    split_text_into_sections,
    _figure_sort_key,
)
from researcher_ai.utils.pubmed import (
    fetch_pmc_fulltext,
    get_figure_urls_from_pmid,
    parse_pubmed_xml,
    parse_jats_xml,
    resolve_pmid_to_pmcid_idconv,
    _elem_text,
    _elem_text_full,
)
from researcher_ai.utils.llm import LLMCache
from researcher_ai.parsers.paper_parser import (
    PaperParser,
    _build_bioc_context_from_collection,
    _deduplicate_ordered,
    _strip_html,
)


# ── pdf.py tests ──────────────────────────────────────────────────────────────

class TestExtractFigureIds:
    """extract_figure_ids_from_text — regex-based, no I/O."""

    def test_basic_figure(self):
        text = "As shown in Figure 1, the expression levels..."
        ids = extract_figure_ids_from_text(text)
        assert "Figure 1" in ids

    def test_abbreviated_fig(self):
        text = "See Fig. 2A for the volcano plot."
        ids = extract_figure_ids_from_text(text)
        assert "Figure 2A" in ids

    def test_plural_figs(self):
        text = "As shown in Figs. 3 and 4B, the data indicate..."
        ids = extract_figure_ids_from_text(text)
        assert "Figure 3" in ids
        assert "Figure 4B" in ids

    def test_supplementary_figure(self):
        text = "Supplementary Figure S1 shows the QC metrics."
        ids = extract_figure_ids_from_text(text)
        assert "Supplementary Figure S1" in ids
        assert "Figure S1" not in ids
        assert "Figure 1" not in ids

    def test_supplementary_abbreviated(self):
        text = "Data shown in Supplementary Fig. 2B."
        ids = extract_figure_ids_from_text(text)
        assert "Supplementary Figure 2B" in ids
        assert "Figure 2B" not in ids

    def test_deduplicated(self):
        text = "Figure 1 is important. See Figure 1 for details."
        ids = extract_figure_ids_from_text(text)
        assert ids.count("Figure 1") == 1

    def test_no_figures(self):
        assert extract_figure_ids_from_text("No figures here.") == []

    def test_mixed_case(self):
        text = "FIGURE 5 shows the results."
        ids = extract_figure_ids_from_text(text)
        assert "Figure 5" in ids

    def test_sort_order_main_before_supplementary(self):
        text = "Supplementary Figure S1 and Figure 1 and Figure 2."
        ids = extract_figure_ids_from_text(text)
        assert ids.index("Figure 1") < ids.index("Supplementary Figure S1")

    def test_sort_order_numeric(self):
        text = "Figure 3, Figure 1, Figure 2."
        ids = extract_figure_ids_from_text(text)
        assert ids == ["Figure 1", "Figure 2", "Figure 3"]


class TestDetectSectionBoundaries:
    def test_finds_methods(self):
        text = "Introduction text.\n\nMethods\n\nWe did stuff."
        bounds = detect_section_boundaries(text)
        titles = [t for t, _ in bounds]
        assert any("method" in t.lower() for t in titles)

    def test_finds_multiple(self):
        text = "Abstract\n\nSome abstract text.\n\nResults\n\nResults text.\n\nMethods\n\nMethods text."
        bounds = detect_section_boundaries(text)
        assert len(bounds) >= 3

    def test_empty_text(self):
        assert detect_section_boundaries("") == []

    def test_no_sections(self):
        assert detect_section_boundaries("Just some random text without headers.") == []


class TestSplitTextIntoSections:
    def test_splits_known_sections(self):
        text = "Abstract\n\nThis is the abstract.\n\nMethods\n\nWe did this."
        sections = split_text_into_sections(text)
        assert len(sections) >= 2
        assert any("abstract" in k.lower() for k in sections)

    def test_fallback_full_text(self):
        text = "No headers here at all."
        sections = split_text_into_sections(text)
        assert "FULL_TEXT" in sections
        assert sections["FULL_TEXT"] == text

    def test_preamble_captured(self):
        text = "Title: A Great Paper\nAuthors: Smith et al.\n\nAbstract\n\nThe abstract."
        sections = split_text_into_sections(text)
        assert "PREAMBLE" in sections
        assert "Smith" in sections["PREAMBLE"]


# ── pubmed.py XML parser tests ─────────────────────────────────────────────────

# Minimal but structurally valid PubMed XML fixture
PUBMED_XML_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID Version="1">26971820</PMID>
          <Article PubModel="Print-Electronic">
            <Journal>
              <Title>Nature Methods</Title>
              <JournalIssue>
                <PubDate><Year>2016</Year></PubDate>
              </JournalIssue>
            </Journal>
            <ArticleTitle>Robust transcriptome-wide discovery of RNA-binding protein binding sites with enhanced CLIP (eCLIP)</ArticleTitle>
            <AuthorList CompleteYN="Y">
              <Author ValidYN="Y">
                <LastName>Van Nostrand</LastName>
                <ForeName>Eric L</ForeName>
                <Initials>EL</Initials>
              </Author>
              <Author ValidYN="Y">
                <LastName>Pratt</LastName>
                <ForeName>Gabriel A</ForeName>
                <Initials>GA</Initials>
              </Author>
              <Author ValidYN="Y">
                <LastName>Yeo</LastName>
                <ForeName>Gene W</ForeName>
                <Initials>GW</Initials>
              </Author>
            </AuthorList>
            <Abstract>
              <AbstractText>eCLIP is a method for identifying RNA-binding protein binding sites.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">26971820</ArticleId>
            <ArticleId IdType="doi">10.1038/nmeth.3810</ArticleId>
            <ArticleId IdType="pmc">PMC4878918</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
""")

# Fixture with nested ArticleId values that should NOT override top-level IDs
PUBMED_XML_NESTED_IDS_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID Version="1">27018577</PMID>
          <Article>
            <ArticleTitle>Robust transcriptome-wide discovery of RNA-binding protein binding sites with enhanced CLIP (eCLIP)</ArticleTitle>
            <Abstract><AbstractText>eCLIP abstract text.</AbstractText></Abstract>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">27018577</ArticleId>
            <ArticleId IdType="doi">10.1038/nmeth.3810</ArticleId>
            <ArticleId IdType="pmc">PMC4887338</ArticleId>
          </ArticleIdList>
          <ReferenceList>
            <Reference>
              <ArticleIdList>
                <ArticleId IdType="doi">10.1016/j.molcel.2010.08.011</ArticleId>
                <ArticleId IdType="pmc">PMC4158944</ArticleId>
              </ArticleIdList>
            </Reference>
          </ReferenceList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
""")

# Minimal JATS XML fixture (as returned by PMC OAI)
JATS_XML_FIXTURE = textwrap.dedent("""\
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">26971820</article-id>
          <article-id pub-id-type="doi">10.1038/nmeth.3810</article-id>
          <title-group>
            <article-title>Robust transcriptome-wide discovery of RNA-binding protein binding sites with enhanced CLIP (eCLIP)</article-title>
          </title-group>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Van Nostrand</surname><given-names>Eric L</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>Yeo</surname><given-names>Gene W</given-names></name>
            </contrib>
          </contrib-group>
          <abstract>
            <p>eCLIP is a method for identifying RNA-binding protein binding sites genome-wide.</p>
          </abstract>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Introduction</title>
          <p>RNA-binding proteins (RBPs) regulate post-transcriptional gene expression. See Figure 1 for an overview.</p>
        </sec>
        <sec>
          <title>Results</title>
          <p>We identified 356 high-confidence binding sites (Fig. 2A). Supplementary Figure S1 shows QC metrics.</p>
          <fig id="fig1">
            <label>Figure 1</label>
            <caption><p>Overview of the eCLIP method.</p></caption>
          </fig>
          <fig id="fig2">
            <label>Figure 2</label>
            <caption><p>(A) Volcano plot of enrichment. (B) Heatmap of top peaks.</p></caption>
          </fig>
        </sec>
        <sec>
          <title>Methods</title>
          <p>Cells were grown in DMEM. eCLIP was performed as described previously.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref1">
            <element-citation publication-type="journal">
              <person-group>
                <name><surname>Ule</surname><given-names>Jernej</given-names></name>
              </person-group>
              <article-title>CLIP: a method for identifying protein-RNA interaction sites in living cells</article-title>
              <source>Science</source>
              <year>2003</year>
              <pub-id pub-id-type="doi">10.1126/science.1090942</pub-id>
            </element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
""")

JATS_XML_NAMESPACED_FIXTURE = textwrap.dedent("""\
    <article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">27018577</article-id>
          <article-id pub-id-type="doi">10.1038/nmeth.3810</article-id>
          <article-id pub-id-type="pmcid">PMC4887338</article-id>
          <title-group>
            <article-title>Robust transcriptome-wide discovery of RNA-binding protein binding sites with enhanced CLIP (eCLIP)</article-title>
          </title-group>
          <abstract><p>Abstract text for namespaced JATS.</p></abstract>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Results</title>
          <p>See Figure 1 for overview and Figure 2 for controls.</p>
          <fig id="f1">
            <label>Figure 1</label>
            <caption><p>Overview panel caption.</p></caption>
          </fig>
        </sec>
      </body>
      <back>
        <floats-group>
          <fig id="f2">
            <label>Figure 2</label>
            <caption><p>Control panel caption in floats group.</p></caption>
          </fig>
        </floats-group>
      </back>
    </article>
""")

JATS_XML_NESTED_METHODS_FIXTURE = textwrap.dedent("""\
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">27018577</article-id>
          <title-group><article-title>Test nested methods</article-title></title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Methods</title>
          <sec>
            <title>eCLIP-seq library preparation</title>
            <p>(See Supplementary Protocol 1 for detailed SOPs.)</p>
          </sec>
          <sec>
            <title>Computational analysis</title>
            <p>Reads were aligned with STAR and peaks called with CLIPper.</p>
          </sec>
        </sec>
      </body>
    </article>
""")

JATS_XML_BACK_DATA_AVAIL_FIXTURE = textwrap.dedent("""\
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">99999999</article-id>
          <title-group><article-title>Back matter data availability test</article-title></title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Results</title>
          <p>Figure 1 overview.</p>
          <fig id="f1">
            <label>Figure 1</label>
            <caption><p>A result figure.</p></caption>
          </fig>
        </sec>
      </body>
      <back>
        <sec sec-type="data-availability">
          <title>Data Availability</title>
          <p>Sequencing data were deposited at GEO under accession GSE77634.</p>
        </sec>
      </back>
    </article>
""")

JATS_XML_BACK_FN_ACCESSION_FIXTURE = textwrap.dedent("""\
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">27018577</article-id>
          <title-group><article-title>Accession footnote test</article-title></title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Results</title>
          <p>Main text.</p>
        </sec>
      </body>
      <back>
        <fn id="FN4">
          <p><bold>Accession codes</bold></p>
          <p>Datasets deposited at GEO (<ext-link ext-link-type="pmc:entrez-geo" xlink:href="GSE77634">GSE77634</ext-link>).</p>
        </fn>
      </back>
    </article>
""")

JATS_XML_KEY_RESOURCES_TABLE_FIXTURE = textwrap.dedent("""\
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmid">11633308</article-id>
          <title-group><article-title>Key resources table test</article-title></title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Methods</title>
          <table-wrap id="t1">
            <label>Table 1</label>
            <caption><title>Key Resources Table</title></caption>
            <table>
              <tbody>
                <tr><td>Data source</td><td>GEO</td><td>GSE314176</td></tr>
                <tr><td>BioProject</td><td>NCBI</td><td>PRJNA123456</td></tr>
              </tbody>
            </table>
          </table-wrap>
        </sec>
      </body>
    </article>
""")


class TestParsePubmedXml:
    def test_pmid_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert result["pmid"] == "26971820"

    def test_doi_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert result["doi"] == "10.1038/nmeth.3810"

    def test_pmcid_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert result["pmcid"] == "PMC4878918"

    def test_title_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert "eCLIP" in result["title"]

    def test_nested_article_ids_do_not_override_primary_ids(self):
        """Nested reference IDs must not overwrite the paper's own DOI/PMCID."""
        result = parse_pubmed_xml(PUBMED_XML_NESTED_IDS_FIXTURE)
        assert result["pmid"] == "27018577"
        assert result["doi"] == "10.1038/nmeth.3810"
        assert result["pmcid"] == "PMC4887338"


class TestPubmedNetworkWrappers:
    """Wrapper behavior around pubmed.py network calls (no live HTTP)."""

    @patch("researcher_ai.utils.pubmed._get")
    def test_fetch_pmc_fulltext_disables_api_key_injection(self, mock_get):
        """PMC OAI requests must not forward NCBI api_key query params."""
        mock_get.return_value = "<article></article>"
        fetch_pmc_fulltext("PMC4158944")
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs.get("include_api_key") is False

    def test_authors_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert len(result["authors"]) == 3
        assert any("Van Nostrand" in a for a in result["authors"])
        assert any("Yeo" in a for a in result["authors"])

    def test_abstract_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert "eCLIP" in result["abstract"]

    def test_year_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert result["year"] == 2016

    def test_journal_extracted(self):
        result = parse_pubmed_xml(PUBMED_XML_FIXTURE)
        assert result["journal"] == "Nature Methods"

    def test_empty_xml(self):
        result = parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>")
        assert result == {}


class TestPmcFigureUrlWorkaround:
    @patch("researcher_ai.utils.pubmed._get")
    def test_idconv_resolves_pmid_to_pmcid_with_headers(self, mock_get):
        mock_get.return_value = json.dumps(
            {"records": [{"pmid": "40054464", "pmcid": "PMC12283108"}]}
        )
        pmcid = resolve_pmid_to_pmcid_idconv("40054464")
        assert pmcid == "PMC12283108"
        _, kwargs = mock_get.call_args
        headers = kwargs.get("headers") or {}
        assert "User-Agent" in headers
        assert headers.get("Referer") == "https://pmc.ncbi.nlm.nih.gov/"

    @patch("researcher_ai.utils.pubmed._get")
    def test_get_figure_urls_prefers_s3_metadata(self, mock_get):
        listing_xml = textwrap.dedent("""\
            <ListBucketResult>
              <Contents><Key>metadata/PMC12283108.1.json</Key></Contents>
            </ListBucketResult>
        """)
        metadata_json = json.dumps(
            {
                "figures": [
                    {"key": "oa_noncomm/xml/PMC12283108.1/media/F1.jpg"},
                    {"filename": "F2.png"},
                ]
            }
        )

        def side_effect(url, *args, **kwargs):
            if "idconv" in url:
                return json.dumps({"records": [{"pmid": "40054464", "pmcid": "PMC12283108"}]})
            if url.endswith("pmc-oa-opendata.s3.amazonaws.com"):
                return listing_xml
            if url.endswith("metadata/PMC12283108.1.json"):
                return metadata_json
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect
        urls = get_figure_urls_from_pmid("40054464")
        assert len(urls) == 2
        assert urls[0] == "https://pmc-oa-opendata.s3.amazonaws.com/oa_noncomm/xml/PMC12283108.1/media/F1.jpg"
        assert urls[1] == "https://pmc-oa-opendata.s3.amazonaws.com/PMC12283108.1/F2.png"
        for call in mock_get.call_args_list:
            headers = call.kwargs.get("headers") or {}
            assert headers.get("Referer") == "https://pmc.ncbi.nlm.nih.gov/"
            assert "User-Agent" in headers

    @patch("researcher_ai.utils.pubmed._get")
    def test_get_figure_urls_falls_back_to_bioc_when_s3_empty(self, mock_get):
        listing_xml = "<ListBucketResult></ListBucketResult>"
        bioc_json = json.dumps(
            {
                "documents": [
                    {
                        "passages": [
                            {
                                "infons": {
                                    "section_type": "FIG",
                                    "graphic": "bin/F3.tif",
                                }
                            }
                        ]
                    }
                ]
            }
        )

        def side_effect(url, *args, **kwargs):
            if "idconv" in url:
                return json.dumps({"records": [{"pmid": "39303722", "pmcid": "PMC11765923"}]})
            if url.endswith("pmc-oa-opendata.s3.amazonaws.com"):
                return listing_xml
            if "/BioC_json/" in url:
                return bioc_json
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect
        urls = get_figure_urls_from_pmid("39303722")
        assert urls == ["https://pmc.ncbi.nlm.nih.gov/articles/PMC11765923/bin/F3.tif"]
        for call in mock_get.call_args_list:
            headers = call.kwargs.get("headers") or {}
            assert headers.get("Referer") == "https://pmc.ncbi.nlm.nih.gov/"
            assert "User-Agent" in headers

    @patch("researcher_ai.utils.pubmed._get")
    def test_get_figure_urls_uses_xml_url_when_metadata_has_no_image_keys(self, mock_get):
        listing_xml = textwrap.dedent("""\
            <ListBucketResult>
              <Contents><Key>metadata/PMC12283108.1.json</Key></Contents>
            </ListBucketResult>
        """)
        metadata_json = json.dumps(
            {"xml_url": "s3://pmc-oa-opendata/PMC12283108.1/PMC12283108.1.xml"}
        )
        article_xml = (
            '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
            '<fig><graphic xlink:href="nihms-2062735-f0001.jpg"/></fig>'
            "</article>"
        )

        def side_effect(url, *args, **kwargs):
            if "idconv" in url:
                return json.dumps({"records": [{"pmid": "40054464", "pmcid": "PMC12283108"}]})
            if url.endswith("pmc-oa-opendata.s3.amazonaws.com"):
                return listing_xml
            if url.endswith("metadata/PMC12283108.1.json"):
                return metadata_json
            if url.endswith("/PMC12283108.1/PMC12283108.1.xml"):
                return article_xml
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect
        urls = get_figure_urls_from_pmid("40054464")
        assert urls == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC12283108.1/nihms-2062735-f0001.jpg"]

    @patch("researcher_ai.utils.pubmed._get")
    def test_xml_url_extraction_skips_supplementary_figures(self, mock_get):
        listing_xml = textwrap.dedent("""\
            <ListBucketResult>
              <Contents><Key>metadata/PMC90000000.1.json</Key></Contents>
            </ListBucketResult>
        """)
        metadata_json = json.dumps(
            {"xml_url": "s3://pmc-oa-opendata/PMC90000000.1/PMC90000000.1.xml"}
        )
        article_xml = (
            '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
            '<fig><label>Figure 1</label><graphic xlink:href="main-f1.jpg"/></fig>'
            '<fig><label>Supplementary Figure 1</label><graphic xlink:href="supp-s1.jpg"/></fig>'
            "</article>"
        )

        def side_effect(url, *args, **kwargs):
            if "idconv" in url:
                return json.dumps({"records": [{"pmid": "99999999", "pmcid": "PMC90000000"}]})
            if url.endswith("pmc-oa-opendata.s3.amazonaws.com"):
                return listing_xml
            if url.endswith("metadata/PMC90000000.1.json"):
                return metadata_json
            if url.endswith("/PMC90000000.1/PMC90000000.1.xml"):
                return article_xml
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect
        urls = get_figure_urls_from_pmid("99999999")
        assert urls == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC90000000.1/main-f1.jpg"]


class TestParseJatsXml:
    def test_title_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        assert "eCLIP" in result["title"]

    def test_authors_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        assert len(result["authors"]) == 2
        assert any("Yeo" in a for a in result["authors"])

    def test_abstract_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        assert "RNA-binding" in result["abstract"]

    def test_sections_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        titles = [s["title"] for s in result["sections"]]
        assert "Introduction" in titles
        assert "Results" in titles
        assert "Methods" in titles

    def test_section_text_populated(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        intro = next(s for s in result["sections"] if s["title"] == "Introduction")
        assert "RNA-binding" in intro["text"]

    def test_figure_captions_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        captions = result["figure_captions"]
        assert len(captions) >= 1
        # At least one caption contains figure content
        all_captions = " ".join(captions.values())
        assert "eCLIP" in all_captions or "Volcano" in all_captions or "Overview" in all_captions

    def test_references_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        refs = result["references"]
        assert len(refs) == 1
        assert refs[0]["ref_id"] == "ref1"
        assert refs[0]["journal"] == "Science"
        assert refs[0]["year"] == 2003
        assert refs[0]["doi"] == "10.1126/science.1090942"

    def test_oai_wrapper_stripped(self):
        """OAI envelope is stripped before parsing."""
        oai_wrapped = (
            "<OAI-PMH><GetRecord><metadata>"
            + JATS_XML_FIXTURE
            + "</metadata></GetRecord></OAI-PMH>"
        )
        result = parse_jats_xml(oai_wrapped)
        assert "title" in result
        assert "eCLIP" in result.get("title", "")

    def test_doi_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        assert result.get("doi") == "10.1038/nmeth.3810"

    def test_pmid_extracted(self):
        result = parse_jats_xml(JATS_XML_FIXTURE)
        assert result.get("pmid") == "26971820"

    def test_namespaced_jats_extracts_sections_and_figures(self):
        result = parse_jats_xml(JATS_XML_NAMESPACED_FIXTURE)
        assert result.get("pmid") == "27018577"
        assert result.get("pmcid") == "PMC4887338"
        assert any(s["title"] == "Results" for s in result.get("sections", []))
        assert "Figure 1" in result.get("figure_captions", {})
        assert "Figure 2" in result.get("figure_captions", {})

    def test_nested_methods_subsections_are_preserved_in_methods_text(self):
        result = parse_jats_xml(JATS_XML_NESTED_METHODS_FIXTURE)
        methods = next(s for s in result["sections"] if s["title"] == "Methods")
        assert "eCLIP-seq library preparation" in methods["text"]
        assert "Supplementary Protocol 1" in methods["text"]
        assert methods["is_methods"] is True

    def test_back_matter_data_availability_section_is_extracted(self):
        result = parse_jats_xml(JATS_XML_BACK_DATA_AVAIL_FIXTURE)
        data_sec = next(s for s in result["sections"] if "availability" in s["title"].lower())
        assert "GSE77634" in data_sec["text"]

    def test_back_fn_accession_codes_are_extracted(self):
        result = parse_jats_xml(JATS_XML_BACK_FN_ACCESSION_FIXTURE)
        acc_sec = next(s for s in result["sections"] if "accession" in s["title"].lower())
        assert "GSE77634" in acc_sec["text"]

    def test_key_resources_table_accessions_are_extracted(self):
        result = parse_jats_xml(JATS_XML_KEY_RESOURCES_TABLE_FIXTURE)
        methods = next(s for s in result["sections"] if s["title"] == "Methods")
        assert "Key Resources Table" in methods["text"]
        assert "GSE314176" in methods["text"]
        assert "PRJNA123456" in methods["text"]


# ── llm.py cache tests ────────────────────────────────────────────────────────

class TestLLMCache:
    def test_miss_returns_none(self, tmp_path):
        cache = LLMCache(tmp_path)
        assert cache.get("prompt", "system", "model") is None

    def test_set_then_get(self, tmp_path):
        cache = LLMCache(tmp_path)
        cache.set("prompt", "system", "model", "response text")
        assert cache.get("prompt", "system", "model") == "response text"

    def test_different_prompts_dont_collide(self, tmp_path):
        cache = LLMCache(tmp_path)
        cache.set("prompt A", "", "model", "response A")
        cache.set("prompt B", "", "model", "response B")
        assert cache.get("prompt A", "", "model") == "response A"
        assert cache.get("prompt B", "", "model") == "response B"

    def test_different_models_dont_collide(self, tmp_path):
        cache = LLMCache(tmp_path)
        cache.set("prompt", "", "model-1", "r1")
        cache.set("prompt", "", "model-2", "r2")
        assert cache.get("prompt", "", "model-1") == "r1"
        assert cache.get("prompt", "", "model-2") == "r2"

    def test_len(self, tmp_path):
        cache = LLMCache(tmp_path)
        assert len(cache) == 0
        cache.set("p1", "", "m", "r")
        cache.set("p2", "", "m", "r")
        assert len(cache) == 2

    def test_clear(self, tmp_path):
        cache = LLMCache(tmp_path)
        cache.set("p", "", "m", "r")
        cache.clear()
        assert len(cache) == 0
        assert cache.get("p", "", "m") is None

    def test_cache_dir_created(self, tmp_path):
        subdir = tmp_path / "nested" / "cache"
        cache = LLMCache(subdir)
        assert subdir.exists()

    def test_unicode_content_survives(self, tmp_path):
        cache = LLMCache(tmp_path)
        content = "RNA–protein interactions: α, β, γ"
        cache.set("p", "", "m", content)
        assert cache.get("p", "", "m") == content


# ── paper_parser.py tests ─────────────────────────────────────────────────────

class TestDetectSourceType:
    """_detect_source_type — pure logic, no mocking needed."""

    def setup_method(self):
        self.parser = PaperParser.__new__(PaperParser)
        self.parser.llm_model = "test-model"
        self.parser.cache = None

    def test_pmcid_with_prefix(self):
        assert self.parser._detect_source_type("PMC4878918") == PaperSource.PMCID

    def test_pmcid_lowercase(self):
        assert self.parser._detect_source_type("pmc4878918") == PaperSource.PMCID

    def test_doi_bare(self):
        assert self.parser._detect_source_type("10.1038/nmeth.3810") == PaperSource.DOI

    def test_doi_url(self):
        assert self.parser._detect_source_type("https://doi.org/10.1038/nmeth.3810") == PaperSource.DOI

    def test_doi_dx_url(self):
        assert self.parser._detect_source_type("https://dx.doi.org/10.1038/s41586-020-2012-7") == PaperSource.DOI

    def test_http_url(self):
        assert self.parser._detect_source_type("https://www.nature.com/articles/nmeth.3810") == PaperSource.URL

    def test_http_plain(self):
        assert self.parser._detect_source_type("http://biorxiv.org/content/123") == PaperSource.URL

    def test_pdf_extension(self):
        assert self.parser._detect_source_type("/some/path/paper.pdf") == PaperSource.PDF

    def test_pdf_extension_uppercase(self):
        assert self.parser._detect_source_type("/path/to/PAPER.PDF") == PaperSource.PDF

    def test_pmid_digits(self):
        assert self.parser._detect_source_type("26971820") == PaperSource.PMID

    def test_pmid_short(self):
        assert self.parser._detect_source_type("1234567") == PaperSource.PMID


class TestParseRawText:
    """_parse_raw_text — mocks LLM calls; tests the parser's orchestration logic."""

    SAMPLE_TEXT = textwrap.dedent("""\
        Robust transcriptome-wide discovery of RNA-binding protein binding sites
        Van Nostrand EL, Pratt GA, Yeo GW
        Nature Methods 2016

        Abstract
        eCLIP is a method for identifying RNA-binding protein binding sites.

        Introduction
        RNA-binding proteins regulate gene expression. See Figure 1 for an overview.

        Results
        We identified 356 peaks (Fig. 2A). Supplementary Figure S1 shows QC.

        Methods
        Cells were grown in DMEM. eCLIP was performed as described.

        References
        [1] Ule J et al. Science 2003. doi:10.1126/science.1090942
    """)

    def _make_parser(self):
        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test-model"
        parser.cache = None
        return parser

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_figure_ids_extracted(self, mock_structured):
        """Figure IDs are extracted by regex — not dependent on LLM."""
        from researcher_ai.parsers.paper_parser import (
            _HeaderMeta, _ExtractedSections, _PaperTypeClassification, _SupplementaryItems
        )

        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _HeaderMeta:
                return _HeaderMeta(
                    title="Robust transcriptome-wide discovery",
                    authors=["Van Nostrand, EL", "Yeo, GW"],
                    abstract="eCLIP is a method.",
                )
            elif output_schema is _ExtractedSections:
                return _ExtractedSections(sections=[
                    type("S", (), {"title": "Introduction", "text": "See Figure 1.", "figures_referenced": []})(),
                    type("S", (), {"title": "Results", "text": "Fig. 2A shows.", "figures_referenced": []})(),
                ])
            elif output_schema is _PaperTypeClassification:
                return _PaperTypeClassification(paper_type="experimental")
            elif output_schema is _SupplementaryItems:
                return _SupplementaryItems(items=[])
            return MagicMock()

        mock_structured.side_effect = side_effect

        parser = self._make_parser()
        paper = parser._parse_raw_text(self.SAMPLE_TEXT, "/tmp/test.pdf", PaperSource.PDF)

        assert "Figure 1" in paper.figure_ids
        assert "Figure 2A" in paper.figure_ids
        assert "Supplementary Figure S1" not in paper.figure_ids

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_sections_extracted_by_regex(self, mock_structured):
        """When regex finds sections, LLM section extraction is skipped."""
        from researcher_ai.parsers.paper_parser import (
            _HeaderMeta, _PaperTypeClassification, _SupplementaryItems
        )

        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _HeaderMeta:
                return _HeaderMeta(title="Test Paper", authors=[], abstract="")
            elif output_schema is _PaperTypeClassification:
                return _PaperTypeClassification(paper_type="experimental")
            elif output_schema is _SupplementaryItems:
                return _SupplementaryItems(items=[])
            return MagicMock()

        mock_structured.side_effect = side_effect

        parser = self._make_parser()
        paper = parser._parse_raw_text(self.SAMPLE_TEXT, "/tmp/test.pdf", PaperSource.PDF)

        assert len(paper.sections) >= 2
        titles = [s.title.lower() for s in paper.sections]
        assert any("result" in t or "method" in t or "introduction" in t for t in titles)

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_paper_type_classified(self, mock_structured):
        from researcher_ai.parsers.paper_parser import (
            _HeaderMeta, _PaperTypeClassification, _SupplementaryItems
        )

        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _HeaderMeta:
                return _HeaderMeta(title="Test", authors=[], abstract="")
            elif output_schema is _PaperTypeClassification:
                return _PaperTypeClassification(paper_type="multi_omic")
            elif output_schema is _SupplementaryItems:
                return _SupplementaryItems(items=[])
            return MagicMock()

        mock_structured.side_effect = side_effect

        parser = self._make_parser()
        paper = parser._parse_raw_text(self.SAMPLE_TEXT, "/tmp/test.pdf", PaperSource.PDF)
        assert paper.paper_type == PaperType.MULTI_OMIC

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_empty_text_returns_stub(self, mock_structured):
        parser = self._make_parser()
        paper = parser._parse_raw_text("", "/tmp/empty.pdf", PaperSource.PDF)
        assert paper.source == PaperSource.PDF
        assert paper.raw_text == ""
        mock_structured.assert_not_called()

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_metadata_applied(self, mock_structured):
        from researcher_ai.parsers.paper_parser import (
            _HeaderMeta, _PaperTypeClassification, _SupplementaryItems
        )

        def side_effect(prompt, output_schema, **kwargs):
            if output_schema is _HeaderMeta:
                return _HeaderMeta(
                    title="eCLIP Paper",
                    authors=["Yeo, GW"],
                    abstract="Abstract text here.",
                    doi="10.1038/nmeth.3810",
                )
            elif output_schema is _PaperTypeClassification:
                return _PaperTypeClassification(paper_type="experimental")
            elif output_schema is _SupplementaryItems:
                return _SupplementaryItems(items=[])
            return MagicMock()

        mock_structured.side_effect = side_effect

        parser = self._make_parser()
        paper = parser._parse_raw_text(self.SAMPLE_TEXT, "/tmp/test.pdf", PaperSource.PDF)

        assert paper.title == "eCLIP Paper"
        assert "Yeo, GW" in paper.authors
        assert paper.doi == "10.1038/nmeth.3810"


class TestBuildPaperFromJats:
    """_build_paper_from_jats — no network, no LLM."""

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_sections_populated(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats_data = parse_jats_xml(JATS_XML_FIXTURE)
        paper = parser._build_paper_from_jats(jats_data, "PMC4878918", PaperSource.PMCID)

        assert paper.source == PaperSource.PMCID
        assert len(paper.sections) == 3
        titles = [s.title for s in paper.sections]
        assert "Introduction" in titles
        assert "Methods" in titles

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_figure_ids_from_captions_and_text(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats_data = parse_jats_xml(JATS_XML_FIXTURE)
        paper = parser._build_paper_from_jats(jats_data, "PMC4878918", PaperSource.PMCID)

        # JATS fixture has Figure 1 and Figure 2 as fig labels,
        # plus in-text references to Figure 1, Fig. 2A, Supplementary Figure S1
        assert len(paper.figure_ids) >= 2
        assert any("Figure" in fid for fid in paper.figure_ids)
        assert not any(fid.startswith("Supplementary Figure") for fid in paper.figure_ids)

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_regression_pmid_40054464_supplementary_intext_not_promoted_to_main(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats = {
            "title": "Test",
            "authors": [],
            "abstract": "",
            "sections": [
                {
                    "title": "Results",
                    "text": "Figure 1 and Figure 2 show main results. See Supplementary Fig. 8 for controls.",
                }
            ],
            "figure_captions": {
                "Figure 1": "Main figure 1",
                "Figure 2": "Main figure 2",
            },
            "references": [],
        }
        paper = parser._build_paper_from_jats(jats, "PMC12283108", PaperSource.PMCID)
        assert paper.figure_ids == ["Figure 1", "Figure 2"]

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_regression_pmid_39303722_keeps_seven_main_figures(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats = {
            "title": "Test",
            "authors": [],
            "abstract": "",
            "sections": [
                {
                    "title": "Results",
                    "text": (
                        "Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6, and Figure 7 "
                        "summarize the primary analyses. Supplementary Fig. 1 provides QC."
                    ),
                }
            ],
            "figure_captions": {f"Figure {i}": f"Main figure {i}" for i in range(1, 8)},
            "references": [],
        }
        paper = parser._build_paper_from_jats(jats, "PMC11765923", PaperSource.PMCID)
        assert paper.figure_ids == [f"Figure {i}" for i in range(1, 8)]

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_references_populated(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats_data = parse_jats_xml(JATS_XML_FIXTURE)
        paper = parser._build_paper_from_jats(jats_data, "PMC4878918", PaperSource.PMCID)

        assert len(paper.references) == 1
        assert paper.references[0].journal == "Science"
        assert paper.references[0].year == 2003

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_metadata_fields(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats_data = parse_jats_xml(JATS_XML_FIXTURE)
        paper = parser._build_paper_from_jats(jats_data, "PMC4878918", PaperSource.PMCID)

        assert "eCLIP" in paper.title
        assert paper.doi == "10.1038/nmeth.3810"
        assert paper.pmid == "26971820"
        assert any("Yeo" in a for a in paper.authors)

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_methods_flags_propagated_to_section_model(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        jats_data = parse_jats_xml(JATS_XML_NESTED_METHODS_FIXTURE)
        paper = parser._build_paper_from_jats(jats_data, "PMC999", PaperSource.PMCID)
        methods = next(s for s in paper.sections if s.title == "Methods")
        assert methods.is_methods is True


class TestParsePMID:
    """_parse_from_pmid — mocks network calls."""

    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.fetch_pmc_fulltext")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_parse_pmid_with_pmcid(
        self, mock_structured, mock_fetch_xml, mock_fetch_pmc, mock_resolve_pmcid
    ):
        """When a PMCID is available, full text is fetched from PMC."""
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        mock_fetch_xml.return_value = PUBMED_XML_FIXTURE
        mock_fetch_pmc.return_value = JATS_XML_FIXTURE
        mock_resolve_pmcid.return_value = "PMC4878918"
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")

        mock_fetch_xml.assert_called_once_with("26971820")
        mock_fetch_pmc.assert_called_once()
        assert "eCLIP" in paper.title
        assert paper.pmid == "26971820"

    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.fetch_pmc_fulltext")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_parse_pmid_retries_with_resolved_pmcid_if_meta_pmcid_fails(
        self, mock_structured, mock_fetch_xml, mock_fetch_pmc, mock_resolve_pmcid
    ):
        """If meta PMCID fails, parser retries with resolve_pmid_to_pmcid() value."""
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        xml_with_wrong_pmcid = PUBMED_XML_FIXTURE.replace(
            '<ArticleId IdType="pmc">PMC4878918</ArticleId>',
            '<ArticleId IdType="pmc">PMC4158944</ArticleId>',
        )
        mock_fetch_xml.return_value = xml_with_wrong_pmcid
        mock_resolve_pmcid.return_value = "PMC4878918"
        mock_fetch_pmc.side_effect = [Exception("400 bad request"), JATS_XML_FIXTURE]
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")

        assert mock_fetch_pmc.call_count == 2
        assert paper.pmid == "26971820"
        assert "eCLIP" in paper.title

    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_parse_pmid_no_pmcid_falls_back(
        self, mock_structured, mock_fetch_xml, mock_resolve_pmcid
    ):
        """When no PMCID, builds Paper from PubMed metadata alone."""
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        # PubMed XML without a PMC ID
        xml_no_pmc = PUBMED_XML_FIXTURE.replace(
            '<ArticleId IdType="pmc">PMC4878918</ArticleId>', ""
        )
        mock_fetch_xml.return_value = xml_no_pmc
        mock_resolve_pmcid.return_value = None
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")

        assert paper.pmid == "26971820"
        assert "eCLIP" in paper.title or paper.title != ""

    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    def test_parse_pmid_network_failure_graceful(self, mock_fetch_xml):
        """Network failure returns a stub Paper, not an exception."""
        mock_fetch_xml.side_effect = Exception("Network error")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")

        assert paper.source == PaperSource.PMID
        assert paper.source_path == "26971820"


class TestParseDOI:
    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.resolve_doi_to_pmid")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_doi_resolved_to_pmid(self, mock_structured, mock_fetch_xml, mock_resolve, mock_resolve_pmc):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        mock_resolve.return_value = "26971820"
        mock_resolve_pmc.return_value = None
        mock_fetch_xml.return_value = PUBMED_XML_FIXTURE.replace(
            '<ArticleId IdType="pmc">PMC4878918</ArticleId>', ""
        )
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_doi("10.1038/nmeth.3810")

        mock_resolve.assert_called_once_with("10.1038/nmeth.3810")
        assert paper.doi == "10.1038/nmeth.3810"

    @patch("researcher_ai.parsers.paper_parser.resolve_doi_to_pmid")
    def test_doi_unresolvable_returns_stub(self, mock_resolve):
        mock_resolve.return_value = None

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_doi("10.9999/nonexistent")

        assert paper.source == PaperSource.DOI
        assert paper.source_path == "10.9999/nonexistent"


class TestBiocContextAttachment:
    @patch("researcher_ai.parsers.paper_parser.fetch_bioc_json_for_paper")
    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_parse_pmid_attaches_bioc_context(
        self,
        mock_structured,
        mock_fetch_xml,
        mock_resolve_pmc,
        mock_fetch_bioc,
    ):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        mock_resolve_pmc.return_value = None
        mock_fetch_xml.return_value = PUBMED_XML_FIXTURE.replace(
            '<ArticleId IdType="pmc">PMC4878918</ArticleId>', ""
        )
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")
        mock_fetch_bioc.return_value = {
            "date": "20260218",
            "documents": [
                {
                    "id": "PMC11633308",
                    "passages": [
                        {"offset": 10, "infons": {"section_type": "FIG", "id": "F1", "type": "fig_caption"}, "text": "Figure 1. Caption"},
                        {"offset": 20, "infons": {"section_type": "RESULTS", "type": "paragraph"}, "text": "Results text"},
                        {"offset": 30, "infons": {"section_type": "METHODS", "type": "paragraph"}, "text": "Methods text"},
                    ],
                }
            ],
        }

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")

        assert paper.bioc_context is not None
        assert paper.bioc_context.source_date == "20260218"
        assert len(paper.bioc_context.fig) == 1
        assert paper.bioc_context.fig[0].figure_id == "Figure 1"
        assert len(paper.bioc_context.results) == 1
        assert len(paper.bioc_context.methods) == 1

    @patch("researcher_ai.parsers.paper_parser.fetch_bioc_json_for_paper")
    @patch("researcher_ai.parsers.paper_parser.resolve_pmid_to_pmcid")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_parse_pmid_bioc_failure_is_graceful(
        self,
        mock_structured,
        mock_fetch_xml,
        mock_resolve_pmc,
        mock_fetch_bioc,
    ):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        mock_resolve_pmc.return_value = None
        mock_fetch_xml.return_value = PUBMED_XML_FIXTURE.replace(
            '<ArticleId IdType="pmc">PMC4878918</ArticleId>', ""
        )
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")
        mock_fetch_bioc.side_effect = RuntimeError("boom")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmid("26971820")
        assert paper.bioc_context is None

    def test_build_bioc_context_caps_passages(self):
        passages = []
        for i in range(0, 250):
            section = "FIG" if i < 120 else "METHODS" if i < 190 else "RESULTS"
            passages.append(
                {"offset": i, "infons": {"section_type": section, "id": f"F{i+1}"}, "text": f"text {i}"}
            )
        collection = {"documents": [{"id": "PMC1", "passages": passages}]}
        ctx = _build_bioc_context_from_collection(collection, pmid="1", pmcid="PMC1", max_passages=200)
        assert ctx is not None
        assert (len(ctx.fig) + len(ctx.methods) + len(ctx.results)) == 200


class TestClassifyPaperType:
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_experimental_classified(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        result = parser._classify_paper_type("We performed RNA-seq experiments.")
        assert result == PaperType.EXPERIMENTAL

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_review_classified(self, mock_structured):
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="review")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        result = parser._classify_paper_type("This systematic review covers...")
        assert result == PaperType.REVIEW

    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_invalid_type_falls_back(self, mock_structured):
        """Invalid paper_type from LLM should fall back to EXPERIMENTAL."""
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification
        mock_structured.return_value = _PaperTypeClassification(paper_type="not_a_real_type")

        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        result = parser._classify_paper_type("Some abstract.")
        assert result == PaperType.EXPERIMENTAL

    def test_empty_abstract_returns_experimental(self):
        parser = PaperParser.__new__(PaperParser)
        parser.llm_model = "test"
        parser.cache = None

        result = parser._classify_paper_type("")
        assert result == PaperType.EXPERIMENTAL


class TestHelpers:
    def test_deduplicate_ordered(self):
        items = ["Figure 1", "Figure 2", "Figure 1", "Figure 3", "Figure 2"]
        result = _deduplicate_ordered(items)
        assert result == ["Figure 1", "Figure 2", "Figure 3"]

    def test_deduplicate_empty(self):
        assert _deduplicate_ordered([]) == []

    def test_strip_html_removes_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strip_html_removes_script(self):
        html = "<html><script>alert('xss')</script><p>Content</p></html>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_strip_html_removes_style(self):
        html = "<style>body { color: red; }</style><p>Text</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "Text" in result

    def test_figure_sort_key_main_before_supp(self):
        main = _figure_sort_key("Figure 1")
        supp = _figure_sort_key("Supplementary Figure S1")
        assert main < supp

    def test_figure_sort_key_numeric_order(self):
        k1 = _figure_sort_key("Figure 1")
        k2 = _figure_sort_key("Figure 2")
        k10 = _figure_sort_key("Figure 10")
        assert k1 < k2 < k10


# ── Evaluation Phase 2 fix tests ─────────────────────────────────────────────

class TestSourceTypeRejectsNonPdf:
    """Fix #3: _detect_source_type should reject existing non-PDF files."""

    def setup_method(self):
        self.parser = PaperParser.__new__(PaperParser)
        self.parser.llm_model = "test-model"
        self.parser.cache = None

    def test_rejects_existing_txt_file(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("some text")
        with pytest.raises(ValueError, match="not a PDF"):
            self.parser._detect_source_type(str(txt_file))

    def test_rejects_existing_csv_file(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c")
        with pytest.raises(ValueError, match="not a PDF"):
            self.parser._detect_source_type(str(csv_file))

    def test_still_accepts_pdf_extension(self):
        # A .pdf path that doesn't exist should still be classified as PDF
        assert self.parser._detect_source_type("/nonexistent/paper.pdf") == PaperSource.PDF


class TestReferenceSecBoundary:
    """Fix #4: Reference extraction prefers section boundary over last-3000."""

    def setup_method(self):
        self.parser = PaperParser.__new__(PaperParser)
        self.parser.llm_model = "test"
        self.parser.cache = None

    def test_finds_references_header(self):
        text = "Some intro text.\n\nReferences\n[1] Ule J et al. Science 2003."
        result = self.parser._extract_reference_section_text(text)
        assert result.startswith("\n")
        assert "References" in result
        assert "[1] Ule J" in result

    def test_finds_bibliography_header(self):
        text = "Body content.\n\nBibliography\n1. Smith A. Nature 2020."
        result = self.parser._extract_reference_section_text(text)
        assert "Bibliography" in result

    def test_falls_back_to_last_3000(self):
        text = "A" * 5000 + "\n[1] Ref without header"
        result = self.parser._extract_reference_section_text(text)
        assert len(result) == 3000


class TestSuppRegexDetection:
    """Fix #5: Regex-based supplementary item detection."""

    def setup_method(self):
        self.parser = PaperParser.__new__(PaperParser)
        self.parser.llm_model = "test"
        self.parser.cache = None

    def test_table_s_detection(self):
        text = "Peak counts are in Table S1. See also Table S2 for enrichment."
        items = self.parser._detect_supplementary_refs_regex(text)
        ids = [i.item_id for i in items]
        assert "Table S1" in ids
        assert "Table S2" in ids

    def test_supplementary_figure_detection(self):
        text = "QC metrics in Supplementary Figure 3."
        items = self.parser._detect_supplementary_refs_regex(text)
        ids = [i.item_id for i in items]
        assert "Supplementary Figure 3" in ids

    def test_data_s_detection(self):
        text = "Raw counts available in Data S1."
        items = self.parser._detect_supplementary_refs_regex(text)
        ids = [i.item_id for i in items]
        assert "Data S1" in ids

    def test_deduplicates(self):
        text = "Table S1 shows peaks. As noted in Table S1 above."
        items = self.parser._detect_supplementary_refs_regex(text)
        ids = [i.item_id for i in items]
        assert ids.count("Table S1") == 1

    def test_no_false_positives_on_main_figures(self):
        text = "Figure 1 shows the overview. Table 2 has the counts."
        items = self.parser._detect_supplementary_refs_regex(text)
        assert len(items) == 0


class TestPmcidFallback:
    """Fix #2: PMCID fallback should try PMID resolution before stub."""

    @patch("researcher_ai.parsers.paper_parser.resolve_pmcid_to_pmid")
    @patch("researcher_ai.parsers.paper_parser.fetch_pmc_fulltext")
    @patch("researcher_ai.parsers.paper_parser.fetch_article_xml")
    @patch("researcher_ai.parsers.paper_parser.ask_claude_structured")
    def test_fallback_to_pmid_on_pmc_failure(
        self, mock_structured, mock_fetch_xml, mock_fetch_pmc, mock_resolve
    ):
        """When PMC fetch fails, parser should try resolving to PMID."""
        from researcher_ai.parsers.paper_parser import _PaperTypeClassification

        # PMC fetch fails
        mock_fetch_pmc.side_effect = Exception("PMC unavailable")
        # PMCID→PMID resolves
        mock_resolve.return_value = "26971820"
        # PMID fetch succeeds
        mock_fetch_xml.return_value = PUBMED_XML_FIXTURE
        mock_structured.return_value = _PaperTypeClassification(paper_type="experimental")

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmcid("PMC4878918", "PMC4878918")

        # Should have called resolve and gotten metadata from PMID path
        mock_resolve.assert_called_once_with("PMC4878918")
        assert paper.title != "PMC4878918"  # Not a bare stub

    @patch("researcher_ai.parsers.paper_parser.resolve_pmcid_to_pmid")
    @patch("researcher_ai.parsers.paper_parser.fetch_pmc_fulltext")
    def test_stub_when_all_fallbacks_fail(self, mock_fetch_pmc, mock_resolve):
        """When both PMC and PMID resolution fail, a stub is returned."""
        mock_fetch_pmc.side_effect = Exception("PMC unavailable")
        mock_resolve.return_value = None

        parser = PaperParser(llm_model="test-model")
        paper = parser._parse_from_pmcid("PMC9999999", "PMC9999999")

        assert paper.title == "PMC9999999"
        assert paper.source == PaperSource.PMCID
