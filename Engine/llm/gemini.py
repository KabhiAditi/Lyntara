"""Gemini provider.

Wraps the google-genai client behind :class:`LLMProvider` so the Gemini path is
one provider among several rather than special-cased code. Still used for the
per-table context/JSON enrichment, where the Spider SQL fine-tune is a poor fit.
"""

from __future__ import annotations

import os
from typing import Any

from .base import STYLE_INSTRUCTION, LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    prompt_style = STYLE_INSTRUCTION

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("google-genai is not installed; `pip install google-genai`") from exc

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set.")

        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"gemini:{model}"

    def _config(self, params: dict[str, Any]):
        from google.genai import types

        fields: dict[str, Any] = {}
        temperature = params.pop("temperature", self.temperature)
        if temperature is not None:
            fields["temperature"] = temperature
        max_tokens = params.pop("max_tokens", self.max_tokens)
        if max_tokens is not None:
            fields["max_output_tokens"] = max_tokens
        if mime := params.pop("response_mime_type", None):
            fields["response_mime_type"] = mime
        if system := params.pop("system_instruction", None):
            fields["system_instruction"] = system
        if stop := params.pop("stop", None):
            fields["stop_sequences"] = stop
        return types.GenerateContentConfig(**fields) if fields else None

    def complete(self, prompt: str, **params: Any) -> str:
        params = dict(params)
        model = params.pop("model", self.model)
        config = self._config(params)
        try:
            kwargs: dict[str, Any] = {"model": model, "contents": prompt}
            if config is not None:
                kwargs["config"] = config
            response = self.client.models.generate_content(**kwargs)
        except Exception as exc:
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        return (response.text or "").strip()

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        body = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") != "system"
        )
        if system:
            params.setdefault("system_instruction", system)
        return self.complete(body, **params)
