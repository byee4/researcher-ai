from __future__ import annotations

import json
import sys
import types

import pytest
from pydantic import BaseModel

from researcher_ai.utils.llm import extract_structured_data


class MockAssay(BaseModel):
    name: str
    n_steps: int


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def test_extract_structured_data_provider_loop_schema_stable(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"name": "RNA-seq", "n_steps": 4}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    model_routers = ["gemini-3.1-pro", "gpt-5.4-planning", "claude-4.6-opus"]
    outputs = [
        extract_structured_data(model_router=m, prompt="extract assay", schema=MockAssay)
        for m in model_routers
    ]

    assert [o.model_dump() for o in outputs] == [
        {"name": "RNA-seq", "n_steps": 4},
        {"name": "RNA-seq", "n_steps": 4},
        {"name": "RNA-seq", "n_steps": 4},
    ]
    assert [c["model"] for c in calls] == model_routers


@pytest.mark.live
def test_extract_structured_data_provider_loop_live():
    model_routers = ["gemini-3.1-pro", "gpt-5.4-planning", "claude-4.6-opus"]
    outputs = [
        extract_structured_data(
            model_router=m,
            prompt="Return assay name RNA-seq with n_steps 4.",
            schema=MockAssay,
            max_tokens=200,
        )
        for m in model_routers
    ]
    assert all(isinstance(o, MockAssay) for o in outputs)
