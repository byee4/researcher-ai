"""LLM prompting helpers for provider-agnostic LLM usage.

Design principles:

- ``ask_claude()``: fire-and-forget text to text.
- ``ask_claude_structured()``: uses tool use to guarantee JSON output matching
  a Pydantic model schema, so no post-hoc string parsing is required.
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

def ask_claude(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    cache: Optional["LLMCache"] = None,
) -> str:
    """Send a prompt to the selected model provider and return text.

    Args:
        prompt: User-turn message.
        system: System prompt (e.g., role description).
        model: Model identifier string.
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature (0 = deterministic).
        cache: Optional LLMCache instance; if provided, results are cached
               on disk and returned from cache on subsequent identical calls.

    Returns:
        Response text string.
    """
    if cache is not None:
        cached = cache.get(prompt, system, model)
        if cached is not None:
            logger.debug("LLM cache hit")
            return cached

    provider = _infer_provider(model)
    client = _get_client(provider)
    if provider == "anthropic":
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if temperature != 0.0:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)
        text = response.content[0].text
    else:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = _openai_chat_create(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = _extract_openai_text(response)

    if cache is not None:
        cache.set(prompt, system, model, text)

    return text


# ---------------------------------------------------------------------------
# Structured extraction via tool_use
# ---------------------------------------------------------------------------

def ask_claude_structured(
    prompt: str,
    output_schema: type[T],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache: Optional["LLMCache"] = None,
) -> T:
    """Extract structured data from text using provider-native JSON features.

    Uses the tool_use feature to guarantee that Claude returns JSON
    conforming to the Pydantic model's schema — no regex parsing required.

    :param prompt: User-turn message describing what to extract.
    :param output_schema: Pydantic model class used as the target schema.
    :param system: Optional system prompt.
    :param model: Model identifier.
    :param max_tokens: Maximum response tokens.
    :param cache: Optional on-disk LLM cache.
    :return: Parsed instance of ``output_schema``.

    Example::

        class HeaderMeta(BaseModel):
            title: str
            authors: list[str]
            doi: Optional[str] = None

        meta = ask_claude_structured(
            prompt=f"Extract from: {paper_text[:2000]}",
            output_schema=HeaderMeta,
            system="You are a scientific paper parser.",
        )
    """
    schema = output_schema.model_json_schema()
    tool_name = "extract_data"

    cache_key_extra = json.dumps(schema, sort_keys=True)
    if cache is not None:
        cached_text = cache.get(prompt, system + cache_key_extra, model)
        if cached_text is not None:
            logger.debug("LLM structured cache hit")
            return output_schema.model_validate_json(cached_text)

    provider = _infer_provider(model)
    client = _get_client(provider)
    if provider == "anthropic":
        tools = [
            {
                "name": tool_name,
                "description": (
                    f"Extract structured information and return it as JSON "
                    f"conforming to the schema. Schema: {schema.get('description', schema['title'] if 'title' in schema else 'output')}"
                ),
                "input_schema": schema,
            }
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        # Find the tool_use block
        tool_block = next(
            (b for b in response.content if b.type == "tool_use" and b.name == tool_name),
            None,
        )
        if tool_block is None:
            raise ValueError(
                f"Claude did not return a tool_use block with name '{tool_name}'. "
                f"Response: {response.content}"
            )

        result = output_schema.model_validate(tool_block.input)
    else:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        schema_name = (
            schema.get("title", output_schema.__name__) if isinstance(schema, dict) else output_schema.__name__
        )
        try:
            response = _openai_chat_create(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": str(schema_name).replace(" ", "_")[:64],
                        "schema": schema,
                        # Some OpenAI models reject strict=true unless every
                        # object node has explicit additionalProperties=false.
                        # Keep schema mode, but relax strictness for compatibility.
                        "strict": False,
                    },
                },
            )
            text = _strip_json_fences(_extract_openai_text(response))
            result = output_schema.model_validate_json(text)
        except Exception as exc:
            logger.warning(
                "OpenAI json_schema structured output failed (%s). "
                "Retrying with json_object response format.",
                exc,
            )
            fallback_messages = list(messages)
            fallback_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return the result as a valid JSON object that conforms to the schema. "
                        "Do not include markdown fences."
                    ),
                }
            )
            response = _openai_chat_create(
                client=client,
                model=model,
                messages=fallback_messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = _strip_json_fences(_extract_openai_text(response))
            result = output_schema.model_validate_json(text)

    if cache is not None:
        cache.set(prompt, system + cache_key_extra, model, result.model_dump_json())

    return result


# ---------------------------------------------------------------------------
# Batch helper (multiple independent extractions in one call)
# ---------------------------------------------------------------------------

def ask_claude_batch(
    prompts: list[str],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[str]:
    """Run multiple independent prompts sequentially.

    Convenience wrapper — does not use the Batch API. For large batches
    (>100 items) prefer the Anthropic Batch API directly.
    """
    return [ask_claude(p, system=system, model=model, max_tokens=max_tokens)
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
        result = ask_claude("...", cache=cache)
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
