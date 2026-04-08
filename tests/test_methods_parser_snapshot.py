"""Snapshot integration tests for MethodsParser (Tier 2).

These tests replay frozen LLM responses from YAML fixtures stored in
tests/snapshots/methods/ and assert structural anchors on the parsed output.

They require no live API key but ARE sensitive to schema changes — if a
Pydantic model field is renamed or a parser helper is refactored, the
fixture may need updating.

Run with:
    pytest tests/test_methods_parser_snapshot.py -m snapshot -v
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from researcher_ai.models.method import Method
from researcher_ai.models.paper import Paper, PaperSource, Section
from researcher_ai.models.method import MethodCategory
from researcher_ai.parsers.methods_parser import (
    MethodsParser,
    _AssayCategoryItem,
    _AssayClassificationList,
    _AssayList,
    _AssayMeta,
    _AvailabilityStatement,
    _DependencyList,
    _StepMeta,
)

SNAPSHOT_DIR = pathlib.Path(__file__).parent / "snapshots" / "methods"
ECLIP_FIXTURE = SNAPSHOT_DIR / "pmid_26971820_eclip.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(path: pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_paper(methods_text: str, doi: str = "10.1234/test") -> Paper:
    return Paper(
        title="Test paper",
        doi=doi,
        source=PaperSource.DOI,
        source_path=doi,
        sections=[Section(title="Materials and Methods", text=methods_text)],
    )


def _build_assay_meta(raw: dict) -> _AssayMeta:
    steps = [
        _StepMeta(
            step_number=s["step_number"],
            description=s["description"],
            input_data=s["input_data"],
            output_data=s["output_data"],
            software=s.get("software"),
            software_version=s.get("software_version"),
            parameters=s.get("parameters", {}),
        )
        for s in raw.get("steps", [])
    ]
    return _AssayMeta(
        name=raw["name"],
        description=raw["description"],
        data_type=raw["data_type"],
        raw_data_source=raw.get("raw_data_source"),
        steps=steps,
        figures_produced=raw.get("figures_produced", []),
    )


def _make_side_effect(llm_responses: dict) -> Any:
    """Build a mock side_effect that replays frozen responses by schema class."""
    assay_meta_order = [
        v for k, v in llm_responses.items() if k.startswith("_AssayMeta_")
    ]
    assay_meta_idx = [0]

    def side_effect(prompt, output_schema, **kw):
        if output_schema is _AssayList:
            raw = llm_responses["_AssayList"]
            return _AssayList(assay_names=raw["assay_names"])

        if output_schema is _AssayClassificationList:
            raw = llm_responses.get("_AssayClassificationList", {"assays": []})
            items = [
                _AssayCategoryItem(
                    name=item["name"],
                    method_category=item["method_category"],
                )
                for item in raw.get("assays", [])
            ]
            return _AssayClassificationList(assays=items)

        if output_schema is _AssayMeta:
            raw = assay_meta_order[assay_meta_idx[0]]
            assay_meta_idx[0] += 1
            return _build_assay_meta(raw)

        if output_schema is _DependencyList:
            raw = llm_responses["_DependencyList"]
            from researcher_ai.parsers.methods_parser import _DependencyMeta
            deps = [
                _DependencyMeta(
                    upstream_assay=d["upstream_assay"],
                    downstream_assay=d["downstream_assay"],
                    dependency_type=d["dependency_type"],
                    description=d.get("description", ""),
                )
                for d in raw.get("dependencies", [])
            ]
            return _DependencyList(dependencies=deps)

        if output_schema is _AvailabilityStatement:
            raw = llm_responses["_AvailabilityStatement"]
            return _AvailabilityStatement(
                data_statement=raw.get("data_statement", ""),
                code_statement=raw.get("code_statement", ""),
            )

        raise ValueError(f"Unexpected schema in snapshot replay: {output_schema}")

    return side_effect


# ---------------------------------------------------------------------------
# TestMethodsParserSnapshoteCLIP
# ---------------------------------------------------------------------------

@pytest.mark.snapshot
class TestMethodsParserSnapshoteCLIP:
    """Snapshot tests for PMID 26971820 (eCLIP, Van Nostrand 2016).

    All LLM responses are replayed from the frozen YAML fixture.
    No network access required.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixture = _load_fixture(ECLIP_FIXTURE)
        self.anchors = self.fixture["expected_anchors"]
        self.paper = _make_paper(
            methods_text=self.fixture["methods_text"],
            doi=self.fixture["doi"],
        )
        self.side_effect = _make_side_effect(self.fixture["llm_responses"])

    def _parse(self, computational_only: bool = False) -> Method:
        with patch(
            "researcher_ai.parsers.methods_parser.ask_claude_structured",
            side_effect=self.side_effect,
        ):
            return MethodsParser().parse(self.paper, computational_only=computational_only)

    def test_assay_count(self):
        method = self._parse()
        assert len(method.assay_graph.assays) == self.anchors["assay_count"]

    def test_assay_names(self):
        method = self._parse()
        names = [a.name for a in method.assay_graph.assays]
        for expected_name in self.anchors["assay_names"]:
            assert expected_name in names

    def test_dependency_count(self):
        method = self._parse()
        assert len(method.assay_graph.dependencies) == self.anchors["dependency_count"]

    def test_dependency_edges(self):
        method = self._parse()
        edges = {
            (d.upstream_assay, d.downstream_assay)
            for d in method.assay_graph.dependencies
        }
        for edge in self.anchors["dependency_edges"]:
            assert (edge["upstream"], edge["downstream"]) in edges

    def test_data_availability(self):
        method = self._parse()
        assert self.anchors["data_availability_contains"] in method.data_availability

    def test_code_availability(self):
        method = self._parse()
        assert self.anchors["code_availability_contains"] in method.code_availability

    def test_parse_warnings_empty(self):
        method = self._parse()
        if self.anchors["parse_warnings_empty"]:
            # Live/provider-dependent RAG enrichment may emit informative
            # inferred-parameter warnings even when no parse degradation occurs.
            non_informational = [
                w for w in method.parse_warnings
                if not str(w).startswith("inferred_parameters:")
            ]
            assert non_informational == [], (
                f"Expected no non-informational parse warnings; got: {method.parse_warnings}"
            )

    def test_computational_assay_step_count(self):
        method = self._parse()
        comp = method.assay_graph.get_assay("Computational read processing and peak calling")
        assert comp is not None
        assert len(comp.steps) == self.anchors["computational_assay_step_count"]

    def test_computational_assay_steps_sorted(self):
        """Steps must be ordered by step_number regardless of fixture order."""
        method = self._parse()
        comp = method.assay_graph.get_assay("Computational read processing and peak calling")
        step_numbers = [s.step_number for s in comp.steps]
        assert step_numbers == sorted(step_numbers)

    def test_computational_assay_first_step_software(self):
        method = self._parse()
        comp = method.assay_graph.get_assay("Computational read processing and peak calling")
        assert comp.steps[0].software == self.anchors["computational_assay_first_step_software"]

    def test_computational_assay_last_step_software(self):
        method = self._parse()
        comp = method.assay_graph.get_assay("Computational read processing and peak calling")
        assert comp.steps[-1].software == self.anchors["computational_assay_last_step_software"]

    def test_json_roundtrip(self):
        """Frozen parse result survives Method → JSON → Method without data loss."""
        method = self._parse()
        restored = Method.model_validate_json(method.model_dump_json())
        assert len(restored.assay_graph.assays) == len(method.assay_graph.assays)
        assert len(restored.assay_graph.dependencies) == len(method.assay_graph.dependencies)
        assert restored.data_availability == method.data_availability
        assert restored.parse_warnings == method.parse_warnings

    def test_graph_traversal_upstream(self):
        """AssayGraph DAG helpers return correct upstream assays."""
        method = self._parse()
        upstream = method.assay_graph.upstream_of("eCLIP library preparation")
        assert "UV crosslinking and immunoprecipitation" in upstream

    def test_graph_traversal_downstream(self):
        """AssayGraph DAG helpers return correct downstream assays."""
        method = self._parse()
        downstream = method.assay_graph.downstream_of("eCLIP library preparation")
        assert "Computational read processing and peak calling" in downstream

    def test_assay_categories(self):
        """Each assay has the correct method_category from classification."""
        method = self._parse()
        expected = self.anchors.get("assay_categories", {})
        for assay_name, expected_cat in expected.items():
            assay = method.assay_graph.get_assay(assay_name)
            assert assay is not None, f"Missing assay: {assay_name}"
            assert assay.method_category.value == expected_cat, (
                f"{assay_name}: expected {expected_cat}, got {assay.method_category.value}"
            )

    def test_computational_only_filters_experimental(self):
        """With computational_only=True, only computational assays survive."""
        method = self._parse(computational_only=True)
        names = [a.name for a in method.assay_graph.assays]
        assert "Computational read processing and peak calling" in names
        assert "UV crosslinking and immunoprecipitation" not in names
        assert "eCLIP library preparation" not in names

    def test_computational_only_parse_warnings(self):
        """Filtered experimental assays are logged in parse_warnings."""
        method = self._parse(computational_only=True)
        warning_text = " ".join(method.parse_warnings)
        assert "UV crosslinking and immunoprecipitation" in warning_text
        assert "eCLIP library preparation" in warning_text
