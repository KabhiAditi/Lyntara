"""Pluggable LLM providers for Lyntara.

Call sites depend on :class:`LLMProvider`; the concrete backend is chosen by
environment variables through :func:`get_provider`.
"""

from .base import STYLE_CHATML, STYLE_INSTRUCTION, LLMError, LLMProvider
from .factory import build_provider, get_provider, get_retries
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider
from . import prompts

__all__ = [
    "LLMProvider",
    "LLMError",
    "STYLE_CHATML",
    "STYLE_INSTRUCTION",
    "get_provider",
    "build_provider",
    "get_retries",
    "OllamaProvider",
    "OpenAICompatProvider",
    "GeminiProvider",
    "prompts",
]
