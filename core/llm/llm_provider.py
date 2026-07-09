"""
Factory for the text-only LLM client, wired to Ollama.

Deliberately a thin wrapper: agents/chains should always get their LLM
through get_llm(), never construct ChatOllama directly -- so swapping
providers, or adjusting context_window/temperature globally, stays a
one-file change instead of a find-and-replace across every agent.
"""

from typing import Optional
from langchain_ollama import ChatOllama
from core.config.settings import settings

# Our settings use hyphenated names (matches the blueprint's Literal types);
# Ollama's actual model tags use colons. This maps between the two.
_OLLAMA_MODEL_MAP = {
    "qwen2.5-7b": "qwen2.5:7b",
    "llama3.1-8b": "llama3.1:8b",
}


def get_llm(json_mode: Optional[bool] = None) -> ChatOllama:
    """
    json_mode: forces Ollama's native JSON-syntax enforcement (format="json")
    when True. Defaults to settings.llm.json_mode. Agents that need structured
    output should use the default (True); free-form chat (like the Step 4
    connectivity test) can pass json_mode=False explicitly.

    Note: format="json" guarantees syntactically valid JSON, NOT schema
    conformance -- the prompt still has to describe the desired shape, and
    the caller still validates against the actual Pydantic schema.
    """
    model_tag = _OLLAMA_MODEL_MAP.get(settings.llm.provider, settings.llm.provider)
    use_json = settings.llm.json_mode if json_mode is None else json_mode

    kwargs = dict(
        model=model_tag,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        num_predict=settings.llm.max_tokens,
        num_ctx=settings.llm.context_window,  # replaces Ollama's small default context window
    )
    if use_json:
        kwargs["format"] = "json"

    return ChatOllama(**kwargs)
