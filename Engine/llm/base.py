"""Provider-agnostic LLM interface.

Every backend (Ollama, llama.cpp, LM Studio, vLLM, Gemini, ...) implements
:class:`LLMProvider`. Call sites depend only on this protocol, so swapping the
model or the serving runtime is a configuration change, not a code change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Prompt styles a provider can consume.
#   "chatml"      -> the caller renders the full ChatML string itself and the
#                    provider must send it verbatim (no server-side templating).
#                    Required for the Spider fine-tune, whose training format
#                    must be reproduced exactly.
#   "instruction" -> a plain instruction prompt; the provider may apply whatever
#                    chat template it likes.
STYLE_CHATML = "chatml"
STYLE_INSTRUCTION = "instruction"


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface every LLM backend must expose."""

    #: Prompt style this provider expects from callers.
    prompt_style: str

    #: Human-readable identifier, used in logs and error messages.
    name: str

    def complete(self, prompt: str, **params: Any) -> str:
        """Send ``prompt`` as-is and return the raw completion text.

        No chat template is applied server-side: what the caller builds is what
        the model sees. This is the path used for the fine-tuned SQL model.
        """
        ...

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        """Send OpenAI-style ``messages`` and return the assistant reply text."""
        ...


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion."""
