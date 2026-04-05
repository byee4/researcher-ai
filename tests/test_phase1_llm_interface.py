from __future__ import annotations

import json
import sys
import types

from pydantic import BaseModel

from researcher_ai.utils.llm import extract_structured_data, generate_text


class _Out(BaseModel):
    value: str


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def test_extract_structured_data_routes_openai_key(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    out = extract_structured_data("gpt-5.4", "prompt", _Out)
    assert out.value == "ok"
    assert calls[-1]["api_key"] == "openai-test-key"


def test_extract_structured_data_routes_anthropic_key(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    out = extract_structured_data("claude-3-7-sonnet", "prompt", _Out)
    assert out.value == "ok"
    assert calls[-1]["api_key"] == "anthropic-test-key"


def test_extract_structured_data_routes_gemini_key(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    out = extract_structured_data("gemini-3.1-pro", "prompt", _Out)
    assert out.value == "ok"
    assert calls[-1]["api_key"] == "gemini-test-key"


def test_extract_structured_data_multimodal_sends_images(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "vision-ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    out = extract_structured_data(
        "gemini-3.1-pro",
        "caption and context",
        _Out,
        image_bytes=[b"\x89PNG\r\n\x1a\nfake"],
    )
    assert out.value == "vision-ok"
    user_content = calls[-1]["messages"][0 if calls[-1]["messages"][0]["role"] == "user" else 1]["content"]
    assert isinstance(user_content, list)
    assert any(part.get("type") == "image_url" for part in user_content if isinstance(part, dict))


def test_generate_text_uses_universal_router(monkeypatch):
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse("hello")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    out = generate_text("gpt-5.4-planning", "prompt")
    assert out == "hello"
    assert calls[-1]["api_key"] == "openai-test-key"


def test_extract_structured_data_router_models_requested(monkeypatch):
    seen_models: list[str] = []

    def _completion(**kwargs):
        seen_models.append(kwargs["model"])
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    extract_structured_data("gpt-5.4-planning", "p", _Out)
    extract_structured_data("claude-4.6-opus", "p", _Out)
    extract_structured_data("gemini-3.1-pro", "p", _Out)

    assert seen_models == ["gpt-5.4-planning", "claude-4.6-opus", "gemini-3.1-pro"]
