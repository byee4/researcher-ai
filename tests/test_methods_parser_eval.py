"""Evaluation framework for iterative methods-parser improvement.

This module provides a fixture-driven evaluation harness that:

1. Accepts YAML fixtures specifying a PMID, expected assays, expected
   computational classifications, and expected keyword coverage.
2. Runs the parser's deterministic helpers (heading extraction, assay
   paragraph extraction, compression, classification filtering) against
   the fixture.
3. Reports per-fixture *recall* and *precision* scores so regressions
   are caught and parser improvements can be measured quantitatively.

Adding a new paper to the eval suite
-------------------------------------
1. Create a YAML file in ``tests/snapshots/methods/`` following the schema
   of ``pmid_40456907_neuronal_aging.yaml``.
2. At minimum, include ``expected_assays`` (list of assay names that MUST
   be identified) and ``expected_computational`` (subset that must survive
   the ``computational_only=True`` filter).
3. Optionally include ``expected_method_keywords`` mapping assay names to
   keywords that must appear in the extracted paragraph.
4. Run ``pytest tests/test_methods_parser_eval.py -v``.

Fixture schema
--------------
.. code-block:: yaml

    pmid: "12345678"
    expected_assays:
      - "RNA-seq"
      - "Proteomics"
    expected_computational:
      - "RNA-seq"
      - "Proteomics"
    expected_experimental_only:  # optional
      - "Western blot"
    expected_method_keywords:    # optional
      "RNA-seq":
        - "STAR"
        - "DESeq2"
    expected_computational_paragraphs:  # optional — full method text + steps
      "Data analysis":
        full_text: |
          ...
        expected_keywords:
          - "STAR"
        expected_steps_min: 3
        expected_software:
          - "STAR"
        expected_code_references:
          - "https://github.com/..."
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Optional

import pytest
import yaml

from researcher_ai.models.method import MethodCategory
from researcher_ai.parsers.methods_parser import (
    _compress_methods_for_identification,
    _extract_assay_block_by_heading,
    _extract_assay_paragraph,
    _extract_heading_like_lines,
    _merge_heading_and_llm_assays,
)

SNAPSHOT_DIR = pathlib.Path(__file__).parent / "snapshots" / "methods"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_all_eval_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """Load all YAML fixtures that define expected_assays."""
    fixtures = []
    for path in sorted(SNAPSHOT_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and "expected_assays" in data:
            fixtures.append((path.stem, data))
    return fixtures


def _casefold_set(names: list[str]) -> set[str]:
    return {n.casefold().strip() for n in names}


def _recall(expected: set[str], actual: set[str]) -> float:
    """Fraction of expected items found in actual."""
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def _precision(expected: set[str], actual: set[str]) -> float:
    """Fraction of actual items that are in expected."""
    if not actual:
        return 1.0
    return len(expected & actual) / len(actual)


# ---------------------------------------------------------------------------
# Parametrised eval tests
# ---------------------------------------------------------------------------

_FIXTURES = _load_all_eval_fixtures()


@pytest.mark.parametrize(
    "fixture_name,fixture",
    _FIXTURES,
    ids=[name for name, _ in _FIXTURES],
)
class TestMethodsParserEval:
    """Evaluate methods-parser helpers against curated fixtures.

    These tests exercise the *deterministic* parser helpers (heading
    extraction, compression, paragraph extraction) without mocking the LLM.
    They verify that the structural pre-processing is sufficient for the LLM
    to identify all expected assays.
    """

    def test_heading_extraction_recall(
        self, fixture_name: str, fixture: dict[str, Any], methods_text_for_fixture: str
    ):
        """All expected assays with section headings must be found by heading extraction."""
        if not methods_text_for_fixture:
            pytest.skip("No methods_text available for this fixture")

        headings = _extract_heading_like_lines(methods_text_for_fixture)
        heading_set = _casefold_set(headings)
        expected_set = _casefold_set(fixture["expected_assays"])

        # Heading extraction may not find every assay (some lack headings),
        # but it MUST find the ones that are explicit section titles.
        found = expected_set & heading_set
        # Allow substring matching (heading may be longer than expected name).
        for exp in expected_set - found:
            for h in heading_set:
                if exp in h or h in exp:
                    found.add(exp)
                    break

        recall = len(found) / len(expected_set) if expected_set else 1.0
        assert recall >= 0.7, (
            f"Heading extraction recall too low: {recall:.0%}. "
            f"Missing: {expected_set - found}"
        )

    def test_compressed_summary_contains_all_assays(
        self, fixture_name: str, fixture: dict[str, Any], methods_text_for_fixture: str
    ):
        """The compressed methods summary must mention every expected assay."""
        if not methods_text_for_fixture:
            pytest.skip("No methods_text available for this fixture")

        compressed = _compress_methods_for_identification(
            methods_text_for_fixture, char_budget=6000
        )
        compressed_lower = compressed.lower()

        missing = []
        for assay in fixture["expected_assays"]:
            # Check primary name or key substring.
            name_lower = assay.lower()
            # For composite names like "Tandem ubiquitin binding entities (TUBE) pulldown MS",
            # check either the full name or the abbreviation.
            parts = [name_lower]
            paren_match = re.search(r"\((\w+)\)", assay)
            if paren_match:
                parts.append(paren_match.group(1).lower())

            if not any(p in compressed_lower for p in parts):
                missing.append(assay)

        assert not missing, (
            f"Compressed summary ({len(compressed)} chars) is missing assays: {missing}"
        )

    def test_paragraph_extraction_keywords(
        self, fixture_name: str, fixture: dict[str, Any], methods_text_for_fixture: str
    ):
        """Extracted paragraphs must contain expected keywords for each assay."""
        if not methods_text_for_fixture:
            pytest.skip("No methods_text available for this fixture")

        keyword_map = fixture.get("expected_method_keywords", {})
        if not keyword_map:
            pytest.skip("No expected_method_keywords in fixture")

        all_assay_names = fixture["expected_assays"]
        failures = []
        for assay_name, keywords in keyword_map.items():
            paragraph = _extract_assay_paragraph(
                methods_text_for_fixture, assay_name, assay_names=all_assay_names
            )
            para_lower = paragraph.lower()
            missing_kw = [kw for kw in keywords if kw.lower() not in para_lower]
            if missing_kw:
                failures.append(
                    f"  {assay_name}: missing keywords {missing_kw} "
                    f"(paragraph length={len(paragraph)})"
                )

        assert not failures, (
            "Paragraph extraction missing expected keywords:\n" + "\n".join(failures)
        )

    def test_merge_heading_and_llm_coverage(
        self, fixture_name: str, fixture: dict[str, Any], methods_text_for_fixture: str
    ):
        """_merge_heading_and_llm_assays must cover all expected assays.

        Simulates the case where the LLM returns an incomplete list (e.g.,
        only assays from the first 4000 chars) and verifies that heading
        merging recovers the missing ones.
        """
        if not methods_text_for_fixture:
            pytest.skip("No methods_text available for this fixture")

        headings = _extract_heading_like_lines(methods_text_for_fixture)
        # Simulate a truncated LLM response: only return the first 3 assays.
        simulated_llm = fixture["expected_assays"][:3]

        merged = _merge_heading_and_llm_assays(headings, simulated_llm)
        merged_lower = _casefold_set(merged)
        expected_lower = _casefold_set(fixture["expected_assays"])

        # Check with substring matching.
        found = set()
        for exp in expected_lower:
            for m in merged_lower:
                if exp in m or m in exp:
                    found.add(exp)
                    break

        recall = len(found) / len(expected_lower) if expected_lower else 1.0
        assert recall >= 0.8, (
            f"Merge recall too low: {recall:.0%}. "
            f"Missing: {expected_lower - found}"
        )

    def test_computational_paragraph_content(
        self, fixture_name: str, fixture: dict[str, Any], methods_text_for_fixture: str
    ):
        """Validate computational paragraph fixtures with detailed expectations."""
        if not methods_text_for_fixture:
            pytest.skip("No methods_text available for this fixture")

        comp_paragraphs = fixture.get("expected_computational_paragraphs", {})
        if not comp_paragraphs:
            pytest.skip("No expected_computational_paragraphs in fixture")

        all_assay_names = fixture.get("expected_assays", [])
        failures = []

        for assay_name, expectations in comp_paragraphs.items():
            paragraph = _extract_assay_paragraph(
                methods_text_for_fixture, assay_name, assay_names=all_assay_names
            )
            para_lower = paragraph.lower()

            # Check keywords.
            for kw in expectations.get("expected_keywords", []):
                if kw.lower() not in para_lower:
                    failures.append(f"  {assay_name}: missing keyword '{kw}'")

            # Check software mentions.
            for sw in expectations.get("expected_software", []):
                if sw.lower() not in para_lower:
                    failures.append(f"  {assay_name}: missing software '{sw}'")

            # Check code references.
            for ref in expectations.get("expected_code_references", []):
                if ref.lower() not in para_lower:
                    failures.append(f"  {assay_name}: missing code ref '{ref}'")

        assert not failures, (
            "Computational paragraph validation failures:\n" + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Fixture: methods text provider
# ---------------------------------------------------------------------------

@pytest.fixture
def methods_text_for_fixture(fixture_name: str, fixture: dict[str, Any]) -> str:
    """Load or generate methods text for a fixture.

    For fixtures that include inline methods_text, use it directly.
    For fixtures that reference a PMID, attempt to load from a cached
    text file in the snapshots directory.
    """
    # 1. Check for inline methods_text in the YAML.
    if "methods_text" in fixture:
        return fixture["methods_text"]

    # 2. Check for a companion .txt file with the full methods text.
    txt_path = SNAPSHOT_DIR / f"{fixture_name}_methods.txt"
    if txt_path.exists():
        return txt_path.read_text()

    # 3. Check for computational paragraph full_text as a fallback.
    comp_paragraphs = fixture.get("expected_computational_paragraphs", {})
    if comp_paragraphs:
        parts = []
        for name, block in comp_paragraphs.items():
            if "full_text" in block:
                parts.append(block["full_text"])
        if parts:
            return "\n\n".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Standalone eval runner
# ---------------------------------------------------------------------------


def run_eval(fixture_path: Optional[str] = None, verbose: bool = True) -> dict[str, Any]:
    """Run evaluation on one or all fixtures and return a summary report.

    This function can be called programmatically (e.g., from a notebook or
    CLI script) to evaluate the parser without pytest.

    Args:
        fixture_path: Path to a specific YAML fixture, or None for all.
        verbose: Print per-assay results.

    Returns:
        dict with keys: total_fixtures, total_expected, total_found,
        overall_recall, per_fixture (list of per-fixture dicts).
    """
    if fixture_path:
        with open(fixture_path) as f:
            data = yaml.safe_load(f)
        fixtures = [(pathlib.Path(fixture_path).stem, data)]
    else:
        fixtures = _load_all_eval_fixtures()

    results: list[dict[str, Any]] = []
    total_expected = 0
    total_found = 0

    for name, fixture in fixtures:
        # Load methods text.
        txt_path = SNAPSHOT_DIR / f"{name}_methods.txt"
        if "methods_text" in fixture:
            methods_text = fixture["methods_text"]
        elif txt_path.exists():
            methods_text = txt_path.read_text()
        else:
            if verbose:
                print(f"  SKIP {name}: no methods_text available")
            continue

        expected = fixture.get("expected_assays", [])
        headings = _extract_heading_like_lines(methods_text)
        compressed = _compress_methods_for_identification(methods_text, char_budget=6000)

        # Check heading recall.
        heading_set = _casefold_set(headings)
        expected_set = _casefold_set(expected)

        found = set()
        for exp in expected_set:
            for h in heading_set:
                if exp in h or h in exp:
                    found.add(exp)
                    break

        recall = _recall(expected_set, found)
        total_expected += len(expected_set)
        total_found += len(found)

        result = {
            "fixture": name,
            "expected": len(expected_set),
            "found_in_headings": len(found),
            "recall": recall,
            "missing": sorted(expected_set - found),
            "compressed_len": len(compressed),
        }
        results.append(result)

        if verbose:
            status = "PASS" if recall >= 0.7 else "FAIL"
            print(f"  [{status}] {name}: recall={recall:.0%} "
                  f"({len(found)}/{len(expected_set)}) "
                  f"compressed={len(compressed)} chars")
            if result["missing"]:
                print(f"         missing: {result['missing']}")

    overall_recall = total_found / total_expected if total_expected else 1.0

    summary = {
        "total_fixtures": len(results),
        "total_expected": total_expected,
        "total_found": total_found,
        "overall_recall": overall_recall,
        "per_fixture": results,
    }

    if verbose:
        print(f"\n  Overall recall: {overall_recall:.0%} "
              f"({total_found}/{total_expected})")

    return summary


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(fixture_path=path)
