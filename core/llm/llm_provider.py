"""
Factory for the text-only LLM client, wired to Ollama.

Deliberately a thin wrapper: agents/chains should always get their LLM
through get_llm(), never construct ChatOllama directly -- so swapping
providers, or adjusting context_window/temperature globally, stays a
one-file change instead of a find-and-replace across every agent.
"""

from langchain_ollama import ChatOllama
from core.config.settings import settings

# Our settings use hyphenated names (matches the blueprint's Literal types);
# Ollama's actual model tags use colons. This maps between the two.
_OLLAMA_MODEL_MAP = {
    "qwen2.5-7b": "qwen2.5:7b",
    "llama3.1-8b": "llama3.1:8b",
}


def get_llm() -> ChatOllama:
    model_tag = _OLLAMA_MODEL_MAP.get(settings.llm.provider, settings.llm.provider)
    return ChatOllama(
        model=model_tag,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        num_predict=settings.llm.max_tokens,
        num_ctx=settings.llm.context_window,  # replaces Ollama's small default context window
    )
