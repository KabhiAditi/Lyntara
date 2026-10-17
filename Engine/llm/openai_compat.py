"""OpenAI-compatible HTTP provider.

Works against any server exposing /v1/chat/completions and /v1/completions:
Ollama, llama.cpp's llama-server, LM Studio, vLLM, TGI, or the OpenAI API
itself. Point ``base_url`` at the server and set ``model``; nothing else needs
to change to move between them.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import STYLE_CHATML, LLMError, LLMProvider


class OpenAICompatProvider(LLMProvider):
    name = "openai-compat"
    prompt_style = STYLE_CHATML

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        prompt_style: str = STYLE_CHATML,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop or []
        self.prompt_style = prompt_style
        self.extra_body = extra_body or {}
        self.name = f"openai-compat:{model}"

    # ── internals ────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _merged(self, params: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": params.pop("model", self.model),
            "temperature": params.pop("temperature", self.temperature),
            "max_tokens": params.pop("max_tokens", self.max_tokens),
            "top_p": params.pop("top_p", self.top_p),
            "stream": False,
        }
        stop = params.pop("stop", self.stop)
        if stop:
            body["stop"] = stop
        body.update(self.extra_body)
        body.update(params)
        return body

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=body, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self.name}: timed out after {self.timeout}s calling {url}. "
                "Local 9B models are slow on Apple Silicon; raise LLM_SQL_TIMEOUT."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise LLMError(f"{self.name}: HTTP {exc.response.status_code} from {url}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: cannot reach {url} ({type(exc).__name__}). Is the server running?") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.name}: malformed JSON response from {url}") from exc

    # ── LLMProvider ──────────────────────────────────────────────

    def complete(self, prompt: str, **params: Any) -> str:
        body = self._merged(dict(params))
        body["prompt"] = prompt
        data = self._post("/v1/completions", body)
        try:
            return (data["choices"][0]["text"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: unexpected completions payload: {str(data)[:300]}") from exc

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        body = self._merged(dict(params))
        body["messages"] = messages
        data = self._post("/v1/chat/completions", body)
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: unexpected chat payload: {str(data)[:300]}") from exc
