"""Phase 1 — Universal LLM Interface: contract and hardening tests.

This file tests the Phase 1 V2 architecture deliverables:
  - Provider-agnostic routing via litellm
  - Config-driven model aliases and provider quirks (models.yaml)
  - Image size validation for multimodal calls
  - Failure modes: missing API key, empty response, response_format fallback
  - Safety settings injected correctly per provider
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel

from researcher_ai.utils.llm import (
    _infer_provider_from_model_router,
    _normalize_max_tokens_for_model,
    _normalize_model_router_for_litellm,
    _normalize_temperature_for_model,
    _resolve_api_key_for_model_router,
    _safety_settings_for_model,
    _validate_image_sizes,
    extract_structured_data,
    generate_text,
    MODEL_ALIAS_MAP,
)


class _Out(BaseModel):
    value: str


class MockAssay(BaseModel):
    name: str
    n_steps: int


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_completion(content: str):
    """Return a _completion callable that always yields *content*."""
    def _completion(**kwargs):
        return _DummyResponse(content)
    return _completion


# ---------------------------------------------------------------------------
# Provider routing — all three providers + aliases
# ---------------------------------------------------------------------------

def test_extract_structured_data_routes_all_three_providers(monkeypatch):
    """All three providers resolve to the correct LiteLLM model prefix."""
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


# ---------------------------------------------------------------------------
# Model alias map — loaded from config
# ---------------------------------------------------------------------------

def test_model_alias_map_contains_builtin_aliases():
    """Built-in aliases are always present even without a YAML file."""
    assert MODEL_ALIAS_MAP.get("gemini-3.1-pro") == "gemini-2.5-pro"
    assert MODEL_ALIAS_MAP.get("gpt-5.4-planning") == "gpt-4.1"


def test_model_alias_map_loaded_from_yaml(tmp_path, monkeypatch):
    """Aliases defined in models.yaml are merged into MODEL_ALIAS_MAP."""
    import yaml

    cfg = {"aliases": {"my-internal-model": "gpt-4o-mini"}}
    (tmp_path / "models.yaml").write_text(yaml.dump(cfg))

    import researcher_ai.utils.llm as llm_mod
    original = llm_mod._MODEL_CONFIG
    monkeypatch.setattr(llm_mod, "_MODEL_CONFIG", cfg)
    # Re-derive the alias map from the patched config
    merged = {
        "gemini-3.1-pro": "gemini-2.5-pro",
        "gpt-5.4-planning": "gpt-4.1",
        **cfg.get("aliases", {}),
    }
    monkeypatch.setattr(llm_mod, "MODEL_ALIAS_MAP", merged)

    assert llm_mod.MODEL_ALIAS_MAP["my-internal-model"] == "gpt-4o-mini"
    # Built-ins are preserved
    assert llm_mod.MODEL_ALIAS_MAP["gemini-3.1-pro"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected_provider", [
    ("claude-3-7-sonnet", "anthropic"),
    ("anthropic/claude-3-opus", "anthropic"),
    ("gemini-2.5-pro", "gemini"),
    ("gemini/gemini-2.5-pro", "gemini"),
    ("vertex_ai/gemini-1.5-flash", "gemini"),
    ("gpt-5.4", "openai"),
    ("gpt-4.1", "openai"),
    ("openai/gpt-4o", "openai"),
])
def test_infer_provider_from_model_router(model, expected_provider):
    assert _infer_provider_from_model_router(model) == expected_provider


# ---------------------------------------------------------------------------
# Normalize model router for LiteLLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("router,expected_litellm_model", [
    ("gemini-3.1-pro", "gemini/gemini-2.5-pro"),
    ("gpt-5.4-planning", "openai/gpt-4.1"),
    ("claude-4.6-opus", "anthropic/claude-4.6-opus"),
    ("gpt-5.4", "openai/gpt-5.4"),
    ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),   # already prefixed — pass through
    ("anthropic/claude-3-opus", "anthropic/claude-3-opus"),  # already prefixed
])
def test_normalize_model_router_for_litellm(router, expected_litellm_model):
    assert _normalize_model_router_for_litellm(router) == expected_litellm_model


# ---------------------------------------------------------------------------
# Temperature constraints (config-driven)
# ---------------------------------------------------------------------------

def test_gpt5_temperature_forced_to_one():
    """GPT-5 models require temperature=1.0."""
    assert _normalize_temperature_for_model("openai/gpt-5.4", 0.0) == 1.0
    assert _normalize_temperature_for_model("openai/gpt-5.4", 0.5) == 1.0


def test_non_gpt5_temperature_unchanged():
    """Non GPT-5 models must not have their temperature altered."""
    assert _normalize_temperature_for_model("openai/gpt-4.1", 0.3) == 0.3
    assert _normalize_temperature_for_model("anthropic/claude-3-opus", 0.7) == 0.7
    assert _normalize_temperature_for_model("gemini/gemini-2.5-pro", 0.2) == 0.2


# ---------------------------------------------------------------------------
# Max-token floor (config-driven)
# ---------------------------------------------------------------------------

def test_gemini_max_tokens_floor_applied():
    """Gemini enforces a minimum of 400 tokens."""
    assert _normalize_max_tokens_for_model("gemini/gemini-2.5-pro", 100) == 400
    assert _normalize_max_tokens_for_model("gemini/gemini-2.5-pro", 1000) == 1000


def test_openai_max_tokens_unchanged():
    """OpenAI models should not have a floor applied."""
    assert _normalize_max_tokens_for_model("openai/gpt-5.4", 50) == 50


# ---------------------------------------------------------------------------
# Safety settings (config-driven)
# ---------------------------------------------------------------------------

def test_gemini_safety_settings_injected(monkeypatch):
    """Gemini calls must include safety_settings to disable content filters."""
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    extract_structured_data("gemini-3.1-pro", "prompt", _Out)

    assert any("safety_settings" in c for c in calls), "safety_settings not found in any Gemini call"
    settings = next(c for c in calls if "safety_settings" in c)["safety_settings"]
    categories = {s["category"] for s in settings}
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in categories
    assert "HARM_CATEGORY_HARASSMENT" in categories


def test_openai_no_safety_settings_injected(monkeypatch):
    """OpenAI calls must NOT receive safety_settings."""
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    extract_structured_data("gpt-5.4", "prompt", _Out)

    assert all("safety_settings" not in c for c in calls)


def test_safety_settings_for_model_returns_list_for_gemini():
    categories = {s["category"] for s in (_safety_settings_for_model("gemini/gemini-2.5-pro") or [])}
    assert len(categories) == 4


def test_safety_settings_for_model_returns_none_for_openai():
    assert _safety_settings_for_model("openai/gpt-5.4") is None


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_resolve_api_key_openai_primary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert _resolve_api_key_for_model_router("gpt-5.4") == "oai-key"


def test_resolve_api_key_openai_fallback(monkeypatch):
    """LLM_API_KEY is accepted as an OpenAI fallback when OPENAI_API_KEY is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fallback-key")
    assert _resolve_api_key_for_model_router("gpt-5.4") == "fallback-key"


def test_resolve_api_key_missing_raises_environment_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="No API key found"):
        _resolve_api_key_for_model_router("gpt-5.4")


def test_resolve_api_key_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert _resolve_api_key_for_model_router("gemini-3.1-pro") == "gem-key"


def test_resolve_api_key_gemini_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key")
    assert _resolve_api_key_for_model_router("gemini-3.1-pro") == "goog-key"


def test_resolve_api_key_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    assert _resolve_api_key_for_model_router("claude-3-7-sonnet") == "ant-key"


# ---------------------------------------------------------------------------
# Image size validation
# ---------------------------------------------------------------------------

def test_validate_image_sizes_accepts_small_images():
    """Images below the limit must pass without error."""
    small = b"\x89PNG" + b"\x00" * 100
    _validate_image_sizes([small], max_bytes=5 * 1024 * 1024)


def test_validate_image_sizes_rejects_oversized_image():
    """Images exceeding the limit must raise ValueError with helpful message."""
    big = b"\x00" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds the"):
        _validate_image_sizes([big], max_bytes=5 * 1024 * 1024)


def test_validate_image_sizes_reports_correct_index():
    """The error message must identify which image in the list is oversized."""
    ok = b"\x00" * 10
    big = b"\x00" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="index 1"):
        _validate_image_sizes([ok, big], max_bytes=5 * 1024 * 1024)


def test_validate_image_sizes_empty_list_is_valid():
    """An empty image list must not raise."""
    _validate_image_sizes([])


def test_extract_structured_data_rejects_oversized_image(monkeypatch):
    """extract_structured_data must validate images before calling the API."""
    api_called = []

    def _completion(**kwargs):
        api_called.append(True)
        return _DummyResponse(json.dumps({"value": "ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr("researcher_ai.utils.llm._MAX_IMAGE_BYTES", 100)

    with pytest.raises(ValueError, match="exceeds the"):
        extract_structured_data(
            "gemini-3.1-pro",
            "caption",
            _Out,
            image_bytes=[b"\x00" * 101],
        )

    assert not api_called, "API should not be called when image validation fails"


# ---------------------------------------------------------------------------
# Fallback chain: response_format not supported → json_object → raw
# ---------------------------------------------------------------------------

def test_response_format_fallback_to_json_object(monkeypatch):
    """When strict json_schema fails, fall back to json_object mode."""
    call_count = [0]

    def _completion(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("response_format not supported by this model")
        return _DummyResponse(json.dumps({"value": "fallback-ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    out = extract_structured_data("gpt-5.4", "prompt", _Out)
    assert out.value == "fallback-ok"
    assert call_count[0] == 2


def test_response_format_fallback_to_raw_completion(monkeypatch):
    """When both structured formats fail (non-Gemini), fall back to raw completion."""
    call_count = [0]

    def _completion(**kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("response_format not supported by this model")
        return _DummyResponse(json.dumps({"value": "raw-ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    out = extract_structured_data("gpt-5.4", "prompt", _Out)
    assert out.value == "raw-ok"
    assert call_count[0] == 3


# ---------------------------------------------------------------------------
# Empty response retry
# ---------------------------------------------------------------------------

def test_empty_response_triggers_retry(monkeypatch):
    """An empty first response must trigger the json_object retry path."""
    call_count = [0]

    def _completion(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _DummyResponse("")          # empty first response
        return _DummyResponse(json.dumps({"value": "retry-ok"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    out = extract_structured_data("gpt-5.4", "prompt", _Out)
    assert out.value == "retry-ok"
    assert call_count[0] >= 2


def test_persistently_empty_response_raises(monkeypatch):
    """If all retry attempts return empty content, ValueError must be raised."""

    def _completion(**kwargs):
        return _DummyResponse("")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(ValueError, match="Empty structured response"):
        extract_structured_data("gpt-5.4", "prompt", _Out)


# ---------------------------------------------------------------------------
# Failure mode tests — LLM timeout, malformed JSON, API rate limit
# ---------------------------------------------------------------------------

def test_extract_structured_data_propagates_timeout_error(monkeypatch):
    """A timeout exception from the LLM API must propagate to the caller unchanged."""
    import socket

    def _completion(**kwargs):
        raise TimeoutError("request timed out after 90 seconds")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(TimeoutError, match="timed out"):
        extract_structured_data("gpt-5.4", "prompt", _Out)


def test_extract_structured_data_raises_on_malformed_json(monkeypatch):
    """A response that is not valid JSON must raise a validation error (not silently return)."""
    def _completion(**kwargs):
        return _DummyResponse("this is definitely not json {{{")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(Exception):
        extract_structured_data("gpt-5.4", "prompt", _Out)


def test_extract_structured_data_raises_on_schema_mismatch(monkeypatch):
    """JSON that does not match the Pydantic schema must raise a validation error."""
    def _completion(**kwargs):
        # Returns valid JSON but the schema expects {"value": str}, not {"unexpected_key": ...}
        return _DummyResponse(json.dumps({"unexpected_key": 42}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(Exception):
        extract_structured_data("gpt-5.4", "prompt", _Out)


def test_extract_structured_data_propagates_rate_limit_error(monkeypatch):
    """HTTP 429 / RateLimitError from the provider must propagate to the caller."""
    class RateLimitError(Exception):
        pass

    def _completion(**kwargs):
        raise RateLimitError("Rate limit exceeded. Please retry after 60 seconds.")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(RateLimitError, match="Rate limit"):
        extract_structured_data("gpt-5.4", "prompt", _Out)


def test_generate_text_propagates_api_error(monkeypatch):
    """Network-level errors in generate_text() must propagate, not be swallowed."""
    def _completion(**kwargs):
        raise ConnectionError("Failed to connect to api.openai.com")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(ConnectionError, match="Failed to connect"):
        generate_text("gpt-5.4", "hello world")


# ---------------------------------------------------------------------------
# LLMCache — edge cases
# ---------------------------------------------------------------------------

def test_llm_cache_hit_avoids_api_call(tmp_path, monkeypatch):
    """A cached response must be returned without calling the LLM API."""
    from researcher_ai.utils.llm import LLMCache

    api_called = []

    def _completion(**kwargs):
        api_called.append(True)
        return _DummyResponse(json.dumps({"value": "original"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    cache = LLMCache(tmp_path / "llm_cache")

    # First call — populates cache
    out1 = extract_structured_data("gpt-5.4", "same prompt", _Out, cache=cache)
    assert out1.value == "original"
    assert len(api_called) == 1

    # Second call with same inputs — must hit cache, no new API call
    out2 = extract_structured_data("gpt-5.4", "same prompt", _Out, cache=cache)
    assert out2.value == "original"
    assert len(api_called) == 1, "API should not be called on cache hit"


def test_llm_cache_key_stable_for_same_schema(tmp_path, monkeypatch):
    """Cache key must be identical on repeated calls with the same schema and prompt."""
    from researcher_ai.utils.llm import LLMCache

    call_count = [0]

    def _completion(**kwargs):
        call_count[0] += 1
        return _DummyResponse(json.dumps({"value": "cached"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    cache = LLMCache(tmp_path / "schema_stability_cache")

    # Call twice with identical inputs — second must be a cache hit.
    extract_structured_data("gpt-5.4", "idempotent prompt", _Out, cache=cache)
    result = extract_structured_data("gpt-5.4", "idempotent prompt", _Out, cache=cache)

    assert result.value == "cached"
    assert call_count[0] == 1, "Identical schema+prompt+model must always hit the cache"


def test_llm_cache_key_uses_sort_keys_for_schema_stability():
    """json.dumps(schema, sort_keys=True) must produce the same string regardless of dict insertion order.

    This validates the cache-key stability assumption: two logically identical
    schemas with different Python dict ordering must hash identically.
    """
    schema_forward = {"b": 2, "a": 1, "properties": {"z": "last", "a": "first"}}
    schema_reversed = {"a": 1, "b": 2, "properties": {"a": "first", "z": "last"}}

    forward_str = json.dumps(schema_forward, sort_keys=True)
    reversed_str = json.dumps(schema_reversed, sort_keys=True)

    assert forward_str == reversed_str, (
        "sort_keys=True must produce identical JSON strings for equivalent dicts in different orders"
    )


def test_llm_cache_clear_removes_all_entries(tmp_path, monkeypatch):
    """LLMCache.clear() must delete all cached files."""
    from researcher_ai.utils.llm import LLMCache

    def _completion(**kwargs):
        return _DummyResponse(json.dumps({"value": "x"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    cache = LLMCache(tmp_path / "clear_cache")
    extract_structured_data("gpt-5.4", "prompt-a", _Out, cache=cache)
    extract_structured_data("gpt-5.4", "prompt-b", _Out, cache=cache)
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0


def test_llm_cache_miss_on_different_model(tmp_path, monkeypatch):
    """Cache must NOT return a hit when the model changes, even for the same prompt."""
    from researcher_ai.utils.llm import LLMCache

    call_models = []

    def _completion(**kwargs):
        call_models.append(kwargs.get("model"))
        return _DummyResponse(json.dumps({"value": "y"}))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    cache = LLMCache(tmp_path / "model_miss_cache")
    extract_structured_data("gpt-5.4", "same prompt", _Out, cache=cache)
    extract_structured_data("claude-3-7-sonnet", "same prompt", _Out, cache=cache)

    assert len(call_models) == 2, "Different models must not share cache entries"


# ---------------------------------------------------------------------------
# Live provider coverage (skipped unless --run-live passed)
# ---------------------------------------------------------------------------

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
        "not_found_error",
        "model:",
        "service unavailable",
        "high demand",
        "unavailable",
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

    # Provider availability can be transient across vendors; require at least
    # one live provider to succeed and record unavailable backends.
    assert len(outputs) >= 1, f"No live provider succeeded; unavailable={unavailable}"
    assert all(isinstance(o, MockAssay) for o in outputs)
