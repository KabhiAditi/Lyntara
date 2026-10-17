"""Ollama provider using the native /api/generate endpoint.

Ollama's OpenAI shim always applies the model's built-in chat template, which
would wrap our prompt in a *second* ChatML envelope and drift from the format
the Spider LoRA was trained on. The native endpoint accepts ``raw: true``,
which sends the prompt through untouched -- that is what :meth:`complete` uses.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import STYLE_CHATML, LLMError
from .openai_compat import OpenAICompatProvider


class OllamaProvider(OpenAICompatProvider):
    prompt_style = STYLE_CHATML

    def __init__(self, base_url: str = "http://localhost:11434", num_ctx: int = 8192, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self.num_ctx = num_ctx
        self.name = f"ollama:{self.model}"

    def _options(self, params: dict[str, Any]) -> dict[str, Any]:
        opts = {
            "temperature": params.pop("temperature", self.temperature),
            "num_predict": params.pop("max_tokens", self.max_tokens),
            "top_p": params.pop("top_p", self.top_p),
            "num_ctx": params.pop("num_ctx", self.num_ctx),
        }
        stop = params.pop("stop", self.stop)
        if stop:
            opts["stop"] = stop
        opts.update(params.pop("options", {}))
        return opts

    def complete(self, prompt: str, **params: Any) -> str:
        params = dict(params)
        model = params.pop("model", self.model)
        body = {
            "model": model,
            "prompt": prompt,
            # raw=True bypasses Ollama's template so the caller's exact ChatML
            # string (including the <think></think> prefill) reaches the model.
            "raw": True,
            "stream": False,
            "options": self._options(params),
        }
        data = self._generate(body)
        return (data.get("response") or "").strip()

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        params = dict(params)
        model = params.pop("model", self.model)
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._options(params),
        }
        url = f"{self.base_url}/api/chat"
        data = self._request(url, body)
        return ((data.get("message") or {}).get("content") or "").strip()

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(f"{self.base_url}/api/generate", body)

    def _request(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self.name}: timed out after {self.timeout}s. A 9B Q4 model on an "
                "M2 Air runs ~10 tok/s; raise LLM_SQL_TIMEOUT if this keeps happening."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise LLMError(
                f"{self.name}: HTTP {exc.response.status_code} from {url}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{self.name}: cannot reach Ollama at {self.base_url} ({type(exc).__name__}). "
                "Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.name}: malformed JSON from {url}") from exc
