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
from typing import Any, Literal, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model — configurable via environment variable
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("RESEARCHER_AI_MODEL", "gpt-5.4")
DEFAULT_MAX_TOKENS = 4096

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Lazy client (avoid import-time auth errors in test environments)
# ---------------------------------------------------------------------------
_client_anthropic = None
_client_openai = None


def _infer_provider(model: str) -> Literal["anthropic", "openai"]:
    """Infer provider from model id."""
    m = (model or "").strip().lower()
    if m.startswith("claude"):
        return "anthropic"
    if (
        m.startswith("gpt")
        or m.startswith("chatgpt")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    ):
        return "openai"
    return "anthropic"


def _get_client(provider: Literal["anthropic", "openai"]):
    """Return a shared LLM client for the selected provider."""
    global _client_anthropic, _client_openai
    if provider == "anthropic":
        if _client_anthropic is None:
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
            if not key:
                raise EnvironmentError(
                    "No LLM API key found. Set LLM_API_KEY (preferred) or ANTHROPIC_API_KEY:\n"
                    "  import os; os.environ['LLM_API_KEY'] = '...'\n"
                    "  or set it in your shell before launching."
                )
            from anthropic import Anthropic  # type: ignore[import]
            _client_anthropic = Anthropic(api_key=key)
        return _client_anthropic
    if _client_openai is None:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not key:
            raise EnvironmentError(
                "No LLM API key found. Set LLM_API_KEY (preferred) or OPENAI_API_KEY:\n"
                "  import os; os.environ['LLM_API_KEY'] = '...'\n"
                "  or set it in your shell before launching."
            )
        from openai import OpenAI  # type: ignore[import]
        _client_openai = OpenAI(api_key=key)
    return _client_openai


def _extract_openai_text(response: Any) -> str:
    """Extract text from OpenAI chat completion response."""
    try:
        content = response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unexpected OpenAI response shape: {type(response).__name__}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content or "")


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
    key_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    key_name = key_env_map[provider]
    key = os.environ.get(key_name) or os.environ.get("LLM_API_KEY")
    if key:
        return key
    raise EnvironmentError(
        f"No API key found for model '{model_router}'. "
        f"Set {key_name} (or LLM_API_KEY fallback)."
    )


def _openai_chat_create(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format: Optional[dict[str, Any]] = None,
) -> Any:
    """Create an OpenAI chat completion with compatibility fallbacks."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature != 0.0:
        kwargs["temperature"] = temperature
    # Newer OpenAI models expect max_completion_tokens.
    kwargs["max_completion_tokens"] = max_tokens
    try:
        return client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = max_tokens
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Simple text → text
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
    try:
        from litellm import completion  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "litellm is required for generate_text(). "
            "Install with: pip install litellm"
        ) from exc

    if cache is not None:
        cached = cache.get(prompt, system, model_router)
        if cached is not None:
            logger.debug("LLM text cache hit")
            return cached

    api_key = _resolve_api_key_for_model_router(model_router)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = completion(
        model=model_router,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )
    text = _extract_message_text(response)
    if cache is not None:
        cache.set(prompt, system, model_router, text)
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
    try:
        from litellm import completion  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "litellm is required for extract_structured_data(). "
            "Install with: pip install litellm"
        ) from exc

    # Backward-compatible kwargs support for migrated call-sites.
    if schema is None:
        schema = legacy_kwargs.pop("output_schema", None)
    if model_router is None:
        model_router = legacy_kwargs.pop("model", None)
    if schema is None:
        raise ValueError("extract_structured_data requires a Pydantic schema.")
    if not model_router:
        raise ValueError("extract_structured_data requires model_router.")

    json_schema = schema.model_json_schema()
    image_bytes = image_bytes or []
    image_hash = hashlib.sha256(b"".join(image_bytes)).hexdigest()[:16] if image_bytes else ""
    cache_key_extra = json.dumps(json_schema, sort_keys=True) + image_hash
    if cache is not None:
        cached_text = cache.get(prompt, system + cache_key_extra, model_router)
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
        "model": model_router,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "api_key": api_key,
    }
    try:
        response = completion(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = completion(**kwargs)

    text = _strip_json_fences(_extract_message_text(response))
    result = schema.model_validate_json(text)

    if cache is not None:
        cache.set(prompt, system + cache_key_extra, model_router, result.model_dump_json())

    return result


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
