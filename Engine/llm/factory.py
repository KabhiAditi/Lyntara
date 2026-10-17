"""Provider factory.

Resolves a *role* ("sql", "context") to a configured :class:`LLMProvider` using
environment variables. Adding a backend means registering it in ``_BUILDERS``;
changing the model in use means editing .env, not code.

Environment variables, per role (ROLE is upper-cased, e.g. LLM_SQL_*):

    LLM_<ROLE>_PROVIDER     ollama | openai | gemini      (required in practice)
    LLM_<ROLE>_BASE_URL     server root, e.g. http://localhost:11434
    LLM_<ROLE>_MODEL        model tag, e.g. lyntara-sql:ckpt200
    LLM_<ROLE>_API_KEY      bearer token, if the server wants one
    LLM_<ROLE>_TEMPERATURE  float
    LLM_<ROLE>_MAX_TOKENS   int
    LLM_<ROLE>_TOP_P        float
    LLM_<ROLE>_NUM_CTX      int   (ollama only)
    LLM_<ROLE>_TIMEOUT      seconds
    LLM_<ROLE>_RETRIES      generation attempts
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable

from .base import LLMError, LLMProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

# Sensible defaults per role, overridden by env.
_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "sql": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "lyntara-sql:ckpt200",
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.9,
        "num_ctx": 8192,
        "timeout": 120.0,
        "retries": 1,
    },
    # SQL self-repair. Defaults to the same local server as "sql" but WITHOUT
    # the fine-tuned adapter, since the Spider LoRA never saw error-correction
    # examples. Override with LLM_SQLFIX_*.
    "sqlfix": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "qwen3.5:9b",
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.9,
        "num_ctx": 8192,
        "timeout": 120.0,
        "retries": 1,
    },
    "context": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "timeout": 120.0,
        "retries": 3,
    },
}


def _env(role: str, key: str) -> str | None:
    return os.getenv(f"LLM_{role.upper()}_{key.upper()}")


def _cfg(role: str) -> dict[str, Any]:
    defaults = dict(_ROLE_DEFAULTS.get(role, {}))

    def pick(key: str, cast: Callable[[str], Any] | None = None) -> Any:
        raw = _env(role, key)
        if raw is None or raw == "":
            return defaults.get(key)
        if cast is None:
            return raw
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return defaults.get(key)

    return {
        "provider": (pick("provider") or "").strip().lower(),
        "base_url": pick("base_url"),
        "model": pick("model"),
        "api_key": pick("api_key"),
        "temperature": pick("temperature", float),
        "max_tokens": pick("max_tokens", int),
        "top_p": pick("top_p", float),
        "num_ctx": pick("num_ctx", int),
        "timeout": pick("timeout", float),
        "retries": pick("retries", int),
    }


def _build_ollama(cfg: dict[str, Any]) -> LLMProvider:
    return OllamaProvider(
        base_url=cfg["base_url"] or "http://localhost:11434",
        model=cfg["model"],
        timeout=cfg["timeout"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        top_p=cfg["top_p"],
        num_ctx=cfg["num_ctx"] or 8192,
        stop=["<|im_end|>", "<|im_start|>"],
    )


def _build_openai(cfg: dict[str, Any]) -> LLMProvider:
    if not cfg["base_url"]:
        raise LLMError("openai-compatible provider requires a base_url")
    return OpenAICompatProvider(
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key=cfg["api_key"],
        timeout=cfg["timeout"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        top_p=cfg["top_p"],
        stop=["<|im_end|>"],
    )


def _build_gemini(cfg: dict[str, Any]) -> LLMProvider:
    return GeminiProvider(
        model=cfg["model"] or "gemini-2.5-flash",
        api_key=cfg["api_key"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


_BUILDERS: dict[str, Callable[[dict[str, Any]], LLMProvider]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "openai-compat": _build_openai,
    "llamacpp": _build_openai,
    "lmstudio": _build_openai,
    "vllm": _build_openai,
    "gemini": _build_gemini,
}


def get_retries(role: str = "sql") -> int:
    """Generation attempts configured for ``role`` (at least 1)."""
    return max(1, int(_cfg(role).get("retries") or 1))


def build_provider(role: str = "sql") -> LLMProvider:
    """Construct a fresh provider for ``role`` (uncached)."""
    cfg = _cfg(role)
    name = cfg["provider"]
    builder = _BUILDERS.get(name)
    if builder is None:
        raise LLMError(
            f"Unknown LLM provider {name!r} for role {role!r}. "
            f"Set LLM_{role.upper()}_PROVIDER to one of: {', '.join(sorted(_BUILDERS))}."
        )
    if not cfg["model"]:
        raise LLMError(f"No model configured for role {role!r}; set LLM_{role.upper()}_MODEL.")
    return builder(cfg)


@lru_cache(maxsize=8)
def _cached(role: str, fingerprint: tuple) -> LLMProvider:
    return build_provider(role)


def get_provider(role: str = "sql") -> LLMProvider:
    """Cached provider for ``role``.

    The cache key includes the resolved config, so editing .env and restarting
    (or changing os.environ at runtime) yields a new provider rather than a
    stale one.
    """
    cfg = _cfg(role)
    fingerprint = tuple(sorted((k, str(v)) for k, v in cfg.items()))
    return _cached(role, fingerprint)
