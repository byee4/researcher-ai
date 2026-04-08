from __future__ import annotations

import asyncio
import threading
import time

import pytest

from researcher_ai.models.method import Assay, MethodCategory
from researcher_ai.parsers.methods_parser import MethodsParser


def _make_parser() -> MethodsParser:
    parser = MethodsParser.__new__(MethodsParser)
    parser.assay_parse_concurrency = 2
    parser.assay_parse_base_timeout_seconds = 90.0
    return parser


def test_adaptive_timeout_scaling_increases_for_multi_assay():
    parser = _make_parser()
    assert parser._adaptive_assay_timeout_seconds(1) == pytest.approx(90.0)
    assert parser._adaptive_assay_timeout_seconds(3) == pytest.approx(90.0)
    assert parser._adaptive_assay_timeout_seconds(4) == pytest.approx(117.0)
    assert parser._adaptive_assay_timeout_seconds(10) == pytest.approx(270.0)


def test_parse_assays_async_respects_concurrency_limit(monkeypatch):
    parser = _make_parser()
    parser.assay_parse_concurrency = 2

    lock = threading.Lock()
    inflight = {"count": 0, "max": 0}

    def _fake_parse_single_assay(**kwargs):
        with lock:
            inflight["count"] += 1
            inflight["max"] = max(inflight["max"], inflight["count"])
        try:
            time.sleep(0.03)
            assay_name = kwargs["assay_name"]
            return (
                Assay(
                    name=assay_name,
                    description="ok",
                    data_type="computational",
                    method_category=MethodCategory.computational,
                    steps=[],
                ),
                [],
            )
        finally:
            with lock:
                inflight["count"] -= 1

    monkeypatch.setattr(parser, "_parse_single_assay", _fake_parse_single_assay)

    assays, warnings = asyncio.run(
        parser._parse_assays_async(
            assay_names=["A", "B", "C", "D"],
            category_map={},
            assay_skeletons={},
            methods_text="methods",
            paper=None,
            figure_context={},
            code_refs=[],
            grounded_accessions=[],
        )
    )
    assert [a.name for a in assays] == ["A", "B", "C", "D"]
    assert warnings == []
    assert inflight["max"] <= 2
