from __future__ import annotations

import json
import sys
import types

import pytest
from pydantic import BaseModel

from researcher_ai.utils.llm import extract_structured_data, extract_structured_data_with_tools


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
    assert [c["model"] for c in calls] == [
        "gemini/gemini-2.5-pro",
        "openai/gpt-4.1",
        "anthropic/claude-4.6-opus",
    ]


def test_extract_structured_data_with_tools_executes_search_tool(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            msg = types.SimpleNamespace(
                content="",
                tool_calls=[
                    types.SimpleNamespace(
                        id="call_1",
                        function=types.SimpleNamespace(
                            name="search_protocol_docs",
                            arguments=json.dumps({"query": "STAR runThreadN", "top_k": 2}),
                        ),
                    )
                ],
            )
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        return _DummyResponse(json.dumps({"name": "RNA-seq", "n_steps": 4}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_protocol_docs",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        }
    ]
    seen_queries: list[str] = []

    out = extract_structured_data_with_tools(
        model_router="gpt-5.4-planning",
        prompt="Fill missing STAR params",
        schema=MockAssay,
        tools=tools,
        tool_handlers={
            "search_protocol_docs": lambda args: seen_queries.append(str(args["query"])) or "doc chunk",
        },
    )
    assert out.name == "RNA-seq"
    assert seen_queries == ["STAR runThreadN"]
    assert len(calls) >= 2


@pytest.mark.live
def test_extract_structured_data_provider_loop_live():
    model_routers = ["gemini-3.1-pro", "gpt-5.4-planning", "claude-4.6-opus"]
    unavailable_markers = [
        "credit balance is too low",
        "quota",
        "rate limit",
        "api key not valid",
        "not found for api version",
        "does not exist or you do not have access",
        "model_not_found",
    ]

    outputs: list[MockAssay] = []
    unavailable: list[str] = []
    for model_router in model_routers:
        try:
            outputs.append(
                extract_structured_data(
                    model_router=model_router,
                    prompt="Return assay name RNA-seq with n_steps 4.",
                    schema=MockAssay,
                    max_tokens=200,
                )
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(marker in msg for marker in unavailable_markers):
                unavailable.append(model_router)
                continue
            raise

    # Require live verification to succeed for at least two providers.
    assert len(outputs) >= 2, f"Insufficient live provider coverage; unavailable={unavailable}"
    assert all(isinstance(o, MockAssay) for o in outputs)
