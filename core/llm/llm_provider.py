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
# "qwen2.5-7b" -> the q8_0 instruct tag, not the bare "qwen2.5:7b" tag: the
# latter was never pulled on this machine and get_llm() would 404 against it.
_OLLAMA_MODEL_MAP = {
    "qwen2.5-7b": "qwen2.5:7b-instruct-q8_0",
    "llama3.1-8b": "llama3.1:8b",
}

# Vision settings.provider values predate Ollama's current library naming
# (Qwen2-VL was superseded by Qwen2.5-VL there) -- map to the real pullable tags.
_OLLAMA_VISION_MODEL_MAP = {
    "qwen2-vl-7b": "qwen2.5vl:7b",
    "llama3.2-vision-11b": "llama3.2-vision:11b",
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


def get_vision_llm() -> ChatOllama:
    """Local multimodal (image+text) client for figure/table analysis.

    Deliberately separate from get_llm(): a vision-capable model is a
    different (larger, slower) model from the text-reasoning model, and
    settings.vision.enabled lets the whole capability be toggled off without
    touching the text pipeline.
    """
    model_tag = _OLLAMA_VISION_MODEL_MAP.get(settings.vision.provider, settings.vision.provider)
    return ChatOllama(
        model=model_tag,
        base_url=settings.vision.base_url,
        temperature=settings.vision.temperature,
        num_predict=settings.vision.max_tokens,
    )
