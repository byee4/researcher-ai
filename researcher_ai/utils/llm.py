"""LLM prompting helpers for provider-agnostic LLM usage.

Design principles:

- ``generate_text()``: provider-agnostic text generation.
- ``extract_structured_data()``: universal structured extraction interface for
  OpenAI, Anthropic, and Gemini via ``litellm``.
- ``LLMCache``: thin file-based cache to avoid re-running identical prompts
  during development and testing.

Set ``LLM_API_KEY`` (preferred) or provider-specific keys before use.
`RESEARCHER_AI_MODEL` controls which provider/model is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model — configurable via environment variable
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("RESEARCHER_AI_MODEL", "gpt-5.4")
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("RESEARCHER_AI_LLM_TIMEOUT_SECONDS", "90"))

T = TypeVar("T", bound=BaseModel)

MODEL_ALIAS_MAP: dict[str, str] = {
    # Keep stable project aliases while routing to currently supported vendor ids.
    "gemini-3.1-pro": "gemini-2.5-pro",
    "gpt-5.4-planning": "gpt-4.1",
}



def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences around JSON, if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)
    return cleaned.strip()


def _extract_message_text(response: Any) -> str:
    """Extract assistant text content from either OpenAI or LiteLLM responses."""
    try:
        content = response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unexpected response shape: {type(response).__name__}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content or "")


def _infer_provider_from_model_router(model_router: str) -> Literal["anthropic", "openai", "gemini"]:
    """Infer provider family from model-router string."""
    m = (model_router or "").strip().lower()
    if m.startswith("claude") or m.startswith("anthropic/"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("google/") or m.startswith("vertex_ai/"):
        return "gemini"
    return "openai"


def _resolve_api_key_for_model_router(model_router: str) -> str:
    """Resolve API key env var by model router/provider."""
    provider = _infer_provider_from_model_router(model_router)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if key:
            return key
        raise EnvironmentError(
            f"No API key found for model '{model_router}'. "
            "Set OPENAI_API_KEY (or LLM_API_KEY fallback)."
        )
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
        raise EnvironmentError(
            f"No API key found for model '{model_router}'. "
            "Set ANTHROPIC_API_KEY."
        )

    # Gemini / Google AI Studio
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    raise EnvironmentError(
        f"No API key found for model '{model_router}'. "
        "Set GEMINI_API_KEY or GOOGLE_API_KEY."
    )


def _normalize_model_router_for_litellm(model_router: str) -> str:
    """Map project model aliases to provider-prefixed LiteLLM model ids."""
    model = (model_router or "").strip()
    if not model:
        return model
    model = MODEL_ALIAS_MAP.get(model, model)
    if "/" in model:
        return model

    provider = _infer_provider_from_model_router(model)
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    return f"openai/{model}"


def _normalize_temperature_for_model(llm_model: str, temperature: float) -> float:
    """Apply provider/model-specific temperature constraints."""
    if "/gpt-5" in llm_model and temperature != 1.0:
        return 1.0
    return temperature


def _normalize_max_tokens_for_model(llm_model: str, max_tokens: int) -> int:
    """Apply provider/model-specific max token floors for reliable responses."""
    if llm_model.startswith("gemini/"):
        return max(max_tokens, 400)
    return max_tokens


def _litellm_completion(**kwargs: Any) -> Any:
    try:
        from litellm import completion  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "litellm is required for LLM operations. "
            "Install with: pip install litellm"
        ) from exc
    return completion(**kwargs)


# ---------------------------------------------------------------------------
# Simple text -> text
# ---------------------------------------------------------------------------

def generate_text(
    model_router: str,
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    cache: Optional["LLMCache"] = None,
) -> str:
    """Universal text-generation interface via LiteLLM."""
    llm_model = _normalize_model_router_for_litellm(model_router)
    normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
    normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
    if cache is not None:
        cached = cache.get(prompt, system, llm_model)
        if cached is not None:
            logger.debug("LLM text cache hit")
            return cached

    api_key = _resolve_api_key_for_model_router(model_router)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _litellm_completion(
        model=llm_model,
        messages=messages,
        max_tokens=normalized_max_tokens,
        temperature=normalized_temperature,
        api_key=api_key,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    text = _extract_message_text(response)
    if cache is not None:
        cache.set(prompt, system, llm_model, text)
    return text


# ---------------------------------------------------------------------------
# Structured extraction via tool_use
# ---------------------------------------------------------------------------

def extract_structured_data(
    model_router: Optional[str] = None,
    prompt: str = "",
    schema: Optional[type[T]] = None,
    *,
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    cache: Optional["LLMCache"] = None,
    image_bytes: Optional[list[bytes]] = None,
    **legacy_kwargs: Any,
) -> T:
    """Universal structured extraction interface via LiteLLM.

    Args:
        model_router: Target model/router string, e.g. "gpt-5.4",
            "claude-3-7-sonnet", "gemini-3.1-pro".
        prompt: User prompt.
        schema: Pydantic model class to validate structured output.
        system: Optional system prompt.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        cache: Optional LLM cache.
        image_bytes: Optional list of raw image bytes for multimodal calls.
    """
    # Backward-compatible kwargs support for migrated call-sites.
    if schema is None:
        schema = legacy_kwargs.pop("output_schema", None)
    if model_router is None:
        model_router = legacy_kwargs.pop("model", None)
    if schema is None:
        raise ValueError("extract_structured_data requires a Pydantic schema.")
    if not model_router:
        raise ValueError("extract_structured_data requires model_router.")
    llm_model = _normalize_model_router_for_litellm(model_router)
    normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
    normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)

    json_schema = schema.model_json_schema()
    image_bytes = image_bytes or []
    image_hash = hashlib.sha256(b"".join(image_bytes)).hexdigest()[:16] if image_bytes else ""
    cache_key_extra = json.dumps(json_schema, sort_keys=True) + image_hash
    if cache is not None:
        cached_text = cache.get(prompt, system + cache_key_extra, llm_model)
        if cached_text is not None:
            logger.debug("LLM structured cache hit")
            return schema.model_validate_json(cached_text)

    api_key = _resolve_api_key_for_model_router(model_router)

    user_content: Any
    if image_bytes:
        multimodal_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in image_bytes:
            b64 = base64.b64encode(img).decode("ascii")
            multimodal_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        user_content = multimodal_parts
    else:
        user_content = prompt

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append(
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON matching this schema exactly:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            ),
        }
    )

    kwargs: dict[str, Any] = {
        "model": llm_model,
        "messages": messages,
        "max_tokens": normalized_max_tokens,
        "temperature": normalized_temperature,
        "api_key": api_key,
        "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
    }
    if llm_model.startswith("gemini/"):
        kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    strict_schema = json.loads(json.dumps(json_schema))
    strict_schema["additionalProperties"] = False
    try:
        response = _litellm_completion(
            **kwargs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__ or "ExtractionSchema",
                    "strict": True,
                    "schema": strict_schema,
                },
            },
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "response_format" in msg or "schema" in msg or "unsupported" in msg:
            try:
                response = _litellm_completion(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as inner_exc:
                inner_msg = str(inner_exc).lower()
                if not llm_model.startswith("gemini/") and "response_format" in inner_msg:
                    response = _litellm_completion(**kwargs)
                else:
                    raise inner_exc
        else:
            raise exc

    text = _strip_json_fences(_extract_message_text(response))
    if not text:
        # Guard against provider responses that return empty content despite a 200.
        retry_response = _litellm_completion(
            **kwargs,
            response_format={"type": "json_object"},
        )
        text = _strip_json_fences(_extract_message_text(retry_response))
        if not text and not llm_model.startswith("gemini/"):
            retry_response = _litellm_completion(**kwargs)
            text = _strip_json_fences(_extract_message_text(retry_response))
        if not text:
            raise ValueError(f"Empty structured response from model '{llm_model}'.")
    try:
        result = schema.model_validate_json(text)
    except Exception:
        if llm_model.startswith("gemini/"):
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = max(int(retry_kwargs.get("max_tokens", 0)), 800)
            retry_response = _litellm_completion(
                **retry_kwargs,
                response_format={"type": "json_object"},
            )
            retry_text = _strip_json_fences(_extract_message_text(retry_response))
            result = schema.model_validate_json(retry_text)
        else:
            raise

    if cache is not None:
        cache.set(prompt, system + cache_key_extra, llm_model, result.model_dump_json())

    return result


def extract_structured_data_with_tools(
    *,
    model_router: str,
    prompt: str,
    schema: type[T],
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[[dict[str, Any]], str]],
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    max_tool_rounds: int = 6,
) -> T:
    """Structured extraction with tool-calling loop via LiteLLM."""
    llm_model = _normalize_model_router_for_litellm(model_router)
    normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
    normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
    api_key = _resolve_api_key_for_model_router(model_router)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    messages.append(
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON matching this schema after using tools as needed:\n"
                f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
            ),
        }
    )

    for _ in range(max_tool_rounds):
        response = _litellm_completion(
            model=llm_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=normalized_max_tokens,
            temperature=normalized_temperature,
            api_key=api_key,
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            text = _strip_json_fences(_extract_message_text(response))
            return schema.model_validate_json(text)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        assistant_tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            tc_id = getattr(tc, "id", None) or tc.get("id")
            fn_obj = getattr(tc, "function", None) or tc.get("function", {})
            fn_name = getattr(fn_obj, "name", None) or fn_obj.get("name")
            fn_args = getattr(fn_obj, "arguments", None) or fn_obj.get("arguments", "{}")
            assistant_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }
            )
        assistant_msg["tool_calls"] = assistant_tool_calls
        messages.append(assistant_msg)

        for call in assistant_tool_calls:
            fn_name = call["function"]["name"]
            args_raw = call["function"]["arguments"] or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
            handler = tool_handlers.get(fn_name)
            if handler is None:
                result = f"Unknown tool: {fn_name}"
            else:
                try:
                    result = handler(args)
                except Exception as exc:  # pragma: no cover - defensive
                    result = f"Tool error: {type(exc).__name__}: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn_name,
                    "content": result,
                }
            )

    # Fallback after exhausting tool rounds.
    final = _litellm_completion(
        model=llm_model,
        messages=messages,
        max_tokens=normalized_max_tokens,
        temperature=normalized_temperature,
        api_key=api_key,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    text = _strip_json_fences(_extract_message_text(final))
    return schema.model_validate_json(text)


# ---------------------------------------------------------------------------
# Batch helper (multiple independent extractions in one call)
# ---------------------------------------------------------------------------

def generate_text_batch(
    prompts: list[str],
    system: str = "",
    model_router: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[str]:
    """Run multiple independent generation prompts sequentially."""
    return [generate_text(prompt=p, system=system, model_router=model_router, max_tokens=max_tokens)
            for p in prompts]


# ---------------------------------------------------------------------------
# File-based cache
# ---------------------------------------------------------------------------

class LLMCache:
    """Simple file-based cache for LLM responses.

    Stores responses as JSON files named by a SHA256 hash of the inputs.
    Useful during development to avoid re-running expensive API calls.

    Usage:
        cache = LLMCache("/path/to/cache/dir")
        result = generate_text("gpt-5.4", "...", cache=cache)
    """

    def __init__(self, cache_dir: str | Path):
        """Create a cache directory (if needed) for structured LLM response files."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, system: str, model: str) -> str:
        """Compute a stable short hash key for cache lookup."""
        h = hashlib.sha256(f"{model}||{system}||{prompt}".encode()).hexdigest()
        return h[:16]

    def _path(self, key: str) -> Path:
        """Return the cache-file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, system: str, model: str) -> Optional[str]:
        """Return cached response text for a prompt/system/model tuple, if present."""
        p = self._path(self._key(prompt, system, model))
        if p.exists():
            return json.loads(p.read_text())["response"]
        return None

    def set(self, prompt: str, system: str, model: str, response: str) -> None:
        """Persist a response string for a prompt/system/model tuple."""
        p = self._path(self._key(prompt, system, model))
        p.write_text(json.dumps({"response": response}, ensure_ascii=False, indent=2))

    def clear(self) -> None:
        """Delete all cached response files for this cache directory."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()

    def __len__(self) -> int:
        """Return the number of cached response files."""
        return len(list(self.cache_dir.glob("*.json")))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PAPER_PARSER = """\
You are a scientific paper parser specialising in biomedical and life-sciences \
publications. Extract information accurately and completely from the provided text. \
Return only what is explicitly stated in the text; do not infer or hallucinate data. \
If a field cannot be determined from the text, leave it empty or null.\
"""

SYSTEM_FIGURE_PARSER = """\
You are an expert at interpreting scientific figures and captions from biomedical \
publications. When asked to analyse a figure, extract structured information about \
each panel's plot type, axes, data, and statistical annotations based solely on the \
caption and in-text references provided.\
"""

SYSTEM_METHODS_PARSER = """\
You are an expert bioinformatician analysing the methods sections of scientific papers. \
Extract structured protocol information including assay types, analysis steps, software \
tools, and their parameters. Be specific about tool versions and parameter values \
when they appear in the text.\
"""
