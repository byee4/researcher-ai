"""LLM prompting helpers for provider-agnostic LLM usage.

Design principles:

- ``generate_text()``: provider-agnostic text generation.
- ``extract_structured_data()``: universal structured extraction interface for
  OpenAI, Anthropic, and Gemini via ``litellm``.
- ``LLMCache``: thin file-based cache to avoid re-running identical prompts
  during development and testing.

Provider behaviour (temperature constraints, token floors, safety settings,
model aliases) is driven by ``researcher_ai/config/models.yaml`` so that
deployment-specific tuning never requires source-code changes.

Set ``LLM_API_KEY`` (preferred) or provider-specific keys before use.
`RESEARCHER_AI_MODEL` controls which provider/model is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class TokenLimitExceededError(ValueError):
    """Raised when prompt/input tokens exceed configured context window."""


class LLMStreamChunk(BaseModel):
    """Unified streaming chunk payload across providers."""

    text: str = ""
    done: bool = False

# ---------------------------------------------------------------------------
# Config loading — reads researcher_ai/config/models.yaml once at import time
# ---------------------------------------------------------------------------

def _load_model_config() -> dict[str, Any]:
    """Load LLM model/provider config from models.yaml.

    Falls back silently to an empty dict if the file is missing or unparseable
    so that the module always works even in minimal environments.
    """
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    if not config_path.exists():
        logger.debug("models.yaml not found at %s — using built-in defaults", config_path)
        return {}
    try:
        import yaml  # PyYAML is a project dependency
        data = yaml.safe_load(config_path.read_text()) or {}
        logger.debug("Loaded LLM config from %s", config_path)
        return data
    except Exception as exc:
        logger.warning(
            "Failed to load models.yaml (%s) — falling back to built-in defaults: %s",
            config_path,
            exc,
        )
        return {}


_MODEL_CONFIG: dict[str, Any] = _load_model_config()

# ---------------------------------------------------------------------------
# Defaults — values from YAML take precedence; env vars take precedence over YAML
# ---------------------------------------------------------------------------

_DEFAULTS = _MODEL_CONFIG.get("defaults", {})

DEFAULT_MAX_TOKENS: int = int(_DEFAULTS.get("max_tokens", 4096))
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = float(
    os.environ.get(
        "RESEARCHER_AI_LLM_TIMEOUT_SECONDS",
        str(_DEFAULTS.get("timeout_seconds", 90)),
    )
)
DEFAULT_MODEL: str = os.environ.get("RESEARCHER_AI_MODEL", "gpt-5.4")


@contextmanager
def temporary_request_timeout(timeout_seconds: float):
    """Temporarily override global request timeout for nested LLM calls."""
    global DEFAULT_REQUEST_TIMEOUT_SECONDS
    previous = DEFAULT_REQUEST_TIMEOUT_SECONDS
    DEFAULT_REQUEST_TIMEOUT_SECONDS = float(timeout_seconds)
    try:
        yield
    finally:
        DEFAULT_REQUEST_TIMEOUT_SECONDS = previous

# Maximum raw bytes allowed per image in a multimodal call.
# Configurable via env var (bytes) or models.yaml defaults.max_image_bytes.
_MAX_IMAGE_BYTES: int = int(
    os.environ.get(
        "RESEARCHER_AI_MAX_IMAGE_BYTES",
        str(_DEFAULTS.get("max_image_bytes", 5 * 1024 * 1024)),
    )
)

# ---------------------------------------------------------------------------
# Model alias map — loaded from YAML, safe to extend without code changes
# ---------------------------------------------------------------------------

# Built-in fallback aliases (used when YAML is absent or the key is not found).
_BUILTIN_ALIASES: dict[str, str] = {
    "gemini-3.1-pro": "gemini-2.5-pro",
    "gpt-5.4-planning": "gpt-4.1",
}

MODEL_ALIAS_MAP: dict[str, str] = {
    **_BUILTIN_ALIASES,
    **_MODEL_CONFIG.get("aliases", {}),
}

# ---------------------------------------------------------------------------
# Provider config helpers
# ---------------------------------------------------------------------------

_PROVIDER_CONFIG: dict[str, Any] = _MODEL_CONFIG.get("providers", {})


def _provider_cfg(provider: str) -> dict[str, Any]:
    """Return the config block for a specific provider (empty dict if absent)."""
    return _PROVIDER_CONFIG.get(provider, {})


def _retry_cfg() -> dict[str, Any]:
    return _MODEL_CONFIG.get("retry", {})


def _fallback_chain_for_model_router(model_router: str) -> list[str]:
    """Return ordered model-router chain: primary model followed by failovers."""
    chain = [str(model_router)]
    fallback_cfg = _MODEL_CONFIG.get("fallbacks", {})
    fallback_raw = (
        fallback_cfg.get(model_router)
        or fallback_cfg.get(_normalize_model_router_for_litellm(model_router))
        or []
    )
    if isinstance(fallback_raw, str):
        fallback_candidates = [fallback_raw]
    elif isinstance(fallback_raw, list):
        fallback_candidates = [str(x) for x in fallback_raw if str(x).strip()]
    else:
        fallback_candidates = []
    for candidate in fallback_candidates:
        if candidate not in chain:
            chain.append(candidate)
    return chain


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
    """Resolve API key from environment using config-driven env-var precedence.

    The list of env vars to check per provider is defined in models.yaml
    under ``providers.<provider>.api_key_envs``.  Built-in defaults are used
    as a fallback when the YAML key is absent.
    """
    provider = _infer_provider_from_model_router(model_router)
    cfg = _provider_cfg(provider)

    _builtin_envs: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY", "LLM_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    }
    env_vars: list[str] = cfg.get("api_key_envs", _builtin_envs.get(provider, []))

    for env_var in env_vars:
        key = os.environ.get(env_var)
        if key:
            return key

    raise EnvironmentError(
        f"No API key found for model '{model_router}' (provider: {provider}). "
        f"Set one of: {', '.join(env_vars)}."
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
    """Apply provider/model-specific temperature constraints from config.

    Constraints are defined in models.yaml under
    ``providers.<provider>.temperature_constraints`` as a mapping of model
    substring → required temperature value.
    """
    provider = _infer_provider_from_model_router(llm_model)
    constraints: dict[str, float] = _provider_cfg(provider).get(
        "temperature_constraints",
        # Built-in fallback: GPT-5 generation models require temperature=1.0
        {"/gpt-5": 1.0} if provider == "openai" else {},
    )
    for pattern, forced_temp in constraints.items():
        if pattern in llm_model:
            return float(forced_temp)
    return temperature


def _normalize_max_tokens_for_model(llm_model: str, max_tokens: int) -> int:
    """Apply provider/model-specific max-token floors from config.

    The floor is defined in models.yaml under
    ``providers.<provider>.min_max_tokens``.
    """
    provider = _infer_provider_from_model_router(llm_model)
    floor: int = _provider_cfg(provider).get(
        "min_max_tokens",
        # Built-in fallback: Gemini needs at least 400 tokens for reliable JSON
        400 if provider == "gemini" else 0,
    )
    return max(max_tokens, floor)


def _safety_settings_for_model(llm_model: str) -> list[dict[str, str]] | None:
    """Return provider safety-setting overrides from config, or None.

    Used to disable over-aggressive content filters for scientific literature.
    Settings are defined in models.yaml under
    ``providers.<provider>.safety_settings``.
    """
    provider = _infer_provider_from_model_router(llm_model)
    settings = _provider_cfg(provider).get("safety_settings")
    if settings:
        return [dict(s) for s in settings]
    # Built-in fallback for Gemini when YAML config is absent
    if provider == "gemini" and not _provider_cfg(provider):
        return [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    return None


def _max_context_tokens_for_model(llm_model: str) -> Optional[int]:
    """Return configured max context window tokens for provider, if set."""
    provider = _infer_provider_from_model_router(llm_model)
    raw = _provider_cfg(provider).get("max_context_tokens")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort 429 / rate-limit classification."""
    try:
        import litellm  # type: ignore[import]
        rate_limit_cls = getattr(litellm, "RateLimitError", None)
        if isinstance(rate_limit_cls, type) and isinstance(exc, rate_limit_cls):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "ratelimit" in cls
    )


def _is_transient_provider_error(exc: Exception) -> bool:
    """Classify failover-eligible transient provider errors (5xx/network)."""
    try:
        import litellm  # type: ignore[import]
        transient_classes = []
        for name in ("APIConnectionError", "APIStatusError", "ServiceUnavailableError", "InternalServerError"):
            cls = getattr(litellm, name, None)
            if isinstance(cls, type):
                transient_classes.append(cls)
        if transient_classes and isinstance(exc, tuple(transient_classes)):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    try:
        if status_code is not None and int(status_code) >= 500:
            return True
    except Exception:
        pass
    return (
        "apiconnectionerror" in cls
        or "serviceunavailable" in cls
        or "internalservererror" in cls
        or "apistatuserror" in cls
        or "bad gateway" in msg
        or "gateway timeout" in msg
        or "service unavailable" in msg
        or " 500" in msg
        or " 502" in msg
        or " 503" in msg
        or " 504" in msg
    )


def _rate_limit_retry_limit() -> int:
    raw = _retry_cfg().get("rate_limit_max_retries", 2)
    try:
        return max(0, int(raw))
    except Exception:
        return 2


def _rate_limit_backoff_seconds(attempt_index: int) -> float:
    cfg = _retry_cfg()
    base = float(cfg.get("rate_limit_backoff_base_seconds", 0.6))
    cap = float(cfg.get("rate_limit_backoff_cap_seconds", 8.0))
    # Exponential backoff without jitter keeps tests deterministic.
    return min(cap, base * (2 ** max(0, attempt_index)))


def _token_overflow_strategy() -> str:
    return str(_DEFAULTS.get("token_overflow_strategy", "raise")).strip().lower()


def _token_count_for_messages(llm_model: str, messages: list[dict[str, Any]]) -> Optional[int]:
    try:
        from litellm import token_counter  # type: ignore[import]
    except Exception:
        return None
    try:
        return int(token_counter(model=llm_model, messages=messages))
    except Exception:
        return None


def _truncate_text_preserving_tail(value: str, *, remove_chars: int) -> str:
    if remove_chars <= 0 or not value:
        return value
    if remove_chars >= len(value):
        # Keep at least a small tail.
        return value[-120:] if len(value) > 120 else value
    return value[remove_chars:]


def _truncate_oldest_message_content(messages: list[dict[str, Any]], *, remove_chars: int) -> bool:
    """Trim oldest non-system/non-final message content from the front."""
    if len(messages) < 2:
        return False
    # Preserve system (typically index 0) and the most recent instruction (last message).
    candidate_indices = [
        i for i in range(1, len(messages) - 1)
        if isinstance(messages[i], dict)
    ]
    if not candidate_indices:
        return False
    idx = candidate_indices[0]
    msg = messages[idx]
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = _truncate_text_preserving_tail(content, remove_chars=remove_chars)
        return True
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = _truncate_text_preserving_tail(
                    part.get("text", ""),
                    remove_chars=remove_chars,
                )
                return True
    return False


def _preflight_token_budget(
    *,
    llm_model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Apply token budget strategy before dispatch, returning possibly-trimmed messages."""
    context_limit = _max_context_tokens_for_model(llm_model)
    if not context_limit:
        return messages
    counted = _token_count_for_messages(llm_model, messages)
    if counted is None:
        return messages
    total = counted + int(max_output_tokens)
    if total <= context_limit:
        return messages

    strategy = _token_overflow_strategy()
    if strategy not in {"trim_oldest", "raise"}:
        strategy = "raise"
    if strategy == "raise":
        raise TokenLimitExceededError(
            f"Token budget exceeded for model '{llm_model}': "
            f"input={counted}, requested_output={int(max_output_tokens)}, "
            f"total={total} > max_context_tokens={context_limit}."
        )

    adjusted = json.loads(json.dumps(messages))
    max_rounds = 6
    for _ in range(max_rounds):
        overflow = (counted + int(max_output_tokens)) - context_limit
        if overflow <= 0:
            return adjusted
        # Approximate chars-to-token conversion.
        removed = _truncate_oldest_message_content(
            adjusted,
            remove_chars=max(160, overflow * 5),
        )
        if not removed:
            break
        recounted = _token_count_for_messages(llm_model, adjusted)
        if recounted is None:
            return adjusted
        counted = recounted
    total = counted + int(max_output_tokens)
    if total > context_limit:
        raise TokenLimitExceededError(
            f"Token budget exceeded for model '{llm_model}' after trim_oldest strategy: "
            f"input={counted}, requested_output={int(max_output_tokens)}, "
            f"total={total} > max_context_tokens={context_limit}."
        )
    return adjusted


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def _validate_image_sizes(image_bytes: list[bytes], max_bytes: int = _MAX_IMAGE_BYTES) -> None:
    """Raise ValueError if any image exceeds the configured byte limit.

    This prevents accidentally sending multi-megabyte figures that would
    silently blow through token limits or cause cryptic API errors.

    Args:
        image_bytes: List of raw image byte strings to validate.
        max_bytes: Per-image byte limit.  Defaults to ``_MAX_IMAGE_BYTES``
            (configured via ``models.yaml`` or ``RESEARCHER_AI_MAX_IMAGE_BYTES``
            env var; factory default is 5 MB).
    """
    for idx, img in enumerate(image_bytes):
        size = len(img)
        if size > max_bytes:
            raise ValueError(
                f"Image at index {idx} is {size:,} bytes, which exceeds the "
                f"{max_bytes:,}-byte limit. Resize or crop the image before "
                "passing it to extract_structured_data()."
            )


# ---------------------------------------------------------------------------
# litellm wrapper
# ---------------------------------------------------------------------------

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
    primary_llm_model = _normalize_model_router_for_litellm(model_router)
    model_chain = _fallback_chain_for_model_router(model_router)
    if cache is not None:
        cached = cache.get(prompt, system, primary_llm_model)
        if cached is not None:
            logger.debug("LLM text cache hit")
            return cached

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = None
    active_llm_model = primary_llm_model
    last_exc: Optional[Exception] = None
    for idx, candidate_router in enumerate(model_chain):
        llm_model = _normalize_model_router_for_litellm(candidate_router)
        normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
        normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
        try:
            api_key = _resolve_api_key_for_model_router(candidate_router)
            candidate_messages = _preflight_token_budget(
                llm_model=llm_model,
                messages=messages,
                max_output_tokens=normalized_max_tokens,
            )
            retry_limit = _rate_limit_retry_limit()
            for attempt in range(retry_limit + 1):
                try:
                    response = _litellm_completion(
                        model=llm_model,
                        messages=candidate_messages,
                        max_tokens=normalized_max_tokens,
                        temperature=normalized_temperature,
                        api_key=api_key,
                        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                    )
                    active_llm_model = llm_model
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit_error(exc) and attempt < retry_limit:
                        sleep_s = _rate_limit_backoff_seconds(attempt)
                        logger.warning(
                            "Rate limit on %s (attempt %d/%d). Backing off %.2fs.",
                            llm_model,
                            attempt + 1,
                            retry_limit + 1,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise
            if response is not None:
                break
        except Exception as exc:
            last_exc = exc
            if idx < len(model_chain) - 1 and (_is_rate_limit_error(exc) or _is_transient_provider_error(exc)):
                logger.warning(
                    "Failing over from %s to %s after %s",
                    llm_model,
                    _normalize_model_router_for_litellm(model_chain[idx + 1]),
                    type(exc).__name__,
                )
                continue
            raise
    if response is None:
        raise last_exc or RuntimeError("LLM completion failed with no response.")
    text = _extract_message_text(response)
    if cache is not None:
        cache.set(prompt, system, primary_llm_model, text)
    if active_llm_model != primary_llm_model:
        logger.info("generate_text used fallback model %s for primary %s", active_llm_model, primary_llm_model)
    return text


def generate_text_stream(
    model_router: str,
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> list[LLMStreamChunk]:
    """Provider-agnostic streaming text generation with normalized chunks.

    Returns a list of `LLMStreamChunk` so callers can consume one normalized
    payload shape regardless of provider.
    """
    llm_model = _normalize_model_router_for_litellm(model_router)
    normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
    normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
    api_key = _resolve_api_key_for_model_router(model_router)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    stream = _litellm_completion(
        model=llm_model,
        messages=messages,
        max_tokens=normalized_max_tokens,
        temperature=normalized_temperature,
        api_key=api_key,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    chunks: list[LLMStreamChunk] = []
    for event in stream:
        try:
            delta = event.choices[0].delta
            text = getattr(delta, "content", None)
            if text is None and isinstance(delta, dict):
                text = delta.get("content")
        except Exception:
            text = None
        if isinstance(text, str) and text:
            chunks.append(LLMStreamChunk(text=text, done=False))
    chunks.append(LLMStreamChunk(text="", done=True))
    return chunks


# ---------------------------------------------------------------------------
# Structured extraction via response_format / tool_use
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
        model_router: Target model/router string, e.g. ``"gpt-5.4"``,
            ``"claude-3-7-sonnet"``, ``"gemini-3.1-pro"``.
        prompt: User prompt.
        schema: Pydantic model class to validate structured output.
        system: Optional system prompt.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        cache: Optional LLM cache.
        image_bytes: Optional list of raw image bytes for multimodal calls.
            Each image must be smaller than ``RESEARCHER_AI_MAX_IMAGE_BYTES``
            (default 5 MB) or a ``ValueError`` is raised before the API call.
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

    primary_llm_model = _normalize_model_router_for_litellm(model_router)
    model_chain = _fallback_chain_for_model_router(model_router)

    image_bytes = image_bytes or []
    # Validate image sizes before making any API call.  Pass _MAX_IMAGE_BYTES
    # explicitly rather than relying on the default argument so that
    # monkeypatching the module-level variable in tests takes effect correctly.
    _validate_image_sizes(image_bytes, max_bytes=_MAX_IMAGE_BYTES)

    json_schema = schema.model_json_schema()
    image_hash = hashlib.sha256(b"".join(image_bytes)).hexdigest()[:16] if image_bytes else ""
    # sort_keys=True ensures the cache key is stable regardless of dict insertion order.
    cache_key_extra = json.dumps(json_schema, sort_keys=True) + image_hash
    if cache is not None:
        cached_text = cache.get(prompt, system + cache_key_extra, primary_llm_model)
        if cached_text is not None:
            logger.debug("LLM structured cache hit")
            return schema.model_validate_json(cached_text)

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

    strict_schema = json.loads(json.dumps(json_schema))
    strict_schema["additionalProperties"] = False
    last_exc: Optional[Exception] = None

    for idx, candidate_router in enumerate(model_chain):
        llm_model = _normalize_model_router_for_litellm(candidate_router)
        normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
        normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
        try:
            api_key = _resolve_api_key_for_model_router(candidate_router)
            candidate_messages = _preflight_token_budget(
                llm_model=llm_model,
                messages=messages,
                max_output_tokens=normalized_max_tokens,
            )
            kwargs: dict[str, Any] = {
                "model": llm_model,
                "messages": candidate_messages,
                "max_tokens": normalized_max_tokens,
                "temperature": normalized_temperature,
                "api_key": api_key,
                "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
            }
            # Inject provider safety settings from config (e.g. Gemini content filters).
            safety = _safety_settings_for_model(llm_model)
            if safety:
                kwargs["safety_settings"] = safety
            retry_limit = _rate_limit_retry_limit()
            response = None
            for attempt in range(retry_limit + 1):
                try:
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
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit_error(exc) and attempt < retry_limit:
                        sleep_s = _rate_limit_backoff_seconds(attempt)
                        logger.warning(
                            "Rate limit on %s (attempt %d/%d). Backing off %.2fs.",
                            llm_model,
                            attempt + 1,
                            retry_limit + 1,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise
            if response is None:
                raise last_exc or RuntimeError("No response returned from structured extraction.")

            text = _strip_json_fences(_extract_message_text(response))
            if not text:
                # Guard against provider responses that return empty content despite 200.
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
                cache.set(prompt, system + cache_key_extra, primary_llm_model, result.model_dump_json())
            if llm_model != primary_llm_model:
                logger.info(
                    "extract_structured_data used fallback model %s for primary %s",
                    llm_model,
                    primary_llm_model,
                )
            return result
        except Exception as exc:
            last_exc = exc
            if idx < len(model_chain) - 1 and (_is_rate_limit_error(exc) or _is_transient_provider_error(exc)):
                logger.warning(
                    "Failing over structured extraction from %s to %s after %s",
                    llm_model,
                    _normalize_model_router_for_litellm(model_chain[idx + 1]),
                    type(exc).__name__,
                )
                continue
            raise
    raise last_exc or RuntimeError("Structured extraction failed with no response.")


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
    primary_llm_model = _normalize_model_router_for_litellm(model_router)
    model_chain = _fallback_chain_for_model_router(model_router)
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

    last_exc: Optional[Exception] = None
    for idx, candidate_router in enumerate(model_chain):
        llm_model = _normalize_model_router_for_litellm(candidate_router)
        normalized_temperature = _normalize_temperature_for_model(llm_model, temperature)
        normalized_max_tokens = _normalize_max_tokens_for_model(llm_model, max_tokens)
        try:
            api_key = _resolve_api_key_for_model_router(candidate_router)
            safety = _safety_settings_for_model(llm_model)
            retry_limit = _rate_limit_retry_limit()

            for _ in range(max_tool_rounds):
                candidate_messages = _preflight_token_budget(
                    llm_model=llm_model,
                    messages=messages,
                    max_output_tokens=normalized_max_tokens,
                )
                response = None
                for attempt in range(retry_limit + 1):
                    try:
                        call_kwargs: dict[str, Any] = {
                            "model": llm_model,
                            "messages": candidate_messages,
                            "tools": tools,
                            "tool_choice": "auto",
                            "max_tokens": normalized_max_tokens,
                            "temperature": normalized_temperature,
                            "api_key": api_key,
                            "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
                        }
                        if safety:
                            call_kwargs["safety_settings"] = safety
                        response = _litellm_completion(**call_kwargs)
                        break
                    except Exception as exc:
                        last_exc = exc
                        if _is_rate_limit_error(exc) and attempt < retry_limit:
                            sleep_s = _rate_limit_backoff_seconds(attempt)
                            logger.warning(
                                "Rate limit on %s (attempt %d/%d). Backing off %.2fs.",
                                llm_model,
                                attempt + 1,
                                retry_limit + 1,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        raise
                if response is None:
                    raise last_exc or RuntimeError("No response returned from tool extraction.")
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    text = _strip_json_fences(_extract_message_text(response))
                    if llm_model != primary_llm_model:
                        logger.info(
                            "extract_structured_data_with_tools used fallback model %s for primary %s",
                            llm_model,
                            primary_llm_model,
                        )
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
                        result_str = f"Unknown tool: {fn_name}"
                    else:
                        try:
                            result_str = handler(args)
                        except Exception as exc:  # pragma: no cover - defensive
                            result_str = f"Tool error: {type(exc).__name__}: {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": fn_name,
                            "content": result_str,
                        }
                    )

            # Fallback after exhausting tool rounds.
            candidate_messages = _preflight_token_budget(
                llm_model=llm_model,
                messages=messages,
                max_output_tokens=normalized_max_tokens,
            )
            final_kwargs: dict[str, Any] = {
                "model": llm_model,
                "messages": candidate_messages,
                "max_tokens": normalized_max_tokens,
                "temperature": normalized_temperature,
                "api_key": api_key,
                "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
            }
            if safety:
                final_kwargs["safety_settings"] = safety
            final = _litellm_completion(**final_kwargs)
            text = _strip_json_fences(_extract_message_text(final))
            if llm_model != primary_llm_model:
                logger.info(
                    "extract_structured_data_with_tools used fallback model %s for primary %s",
                    llm_model,
                    primary_llm_model,
                )
            return schema.model_validate_json(text)
        except Exception as exc:
            last_exc = exc
            if idx < len(model_chain) - 1 and (_is_rate_limit_error(exc) or _is_transient_provider_error(exc)):
                logger.warning(
                    "Failing over tool extraction from %s to %s after %s",
                    llm_model,
                    _normalize_model_router_for_litellm(model_chain[idx + 1]),
                    type(exc).__name__,
                )
                continue
            raise
    raise last_exc or RuntimeError("Tool extraction failed with no response.")


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
    return [
        generate_text(prompt=p, system=system, model_router=model_router, max_tokens=max_tokens)
        for p in prompts
    ]


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
