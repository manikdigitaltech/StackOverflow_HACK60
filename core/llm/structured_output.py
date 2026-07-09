"""
Shared helper: invoke an LLM and parse its response into a specific Pydantic
schema. Every agent uses this rather than parsing JSON ad hoc.

Design choice: rather than relying on LangChain's with_structured_output()
(which, with Ollama, depends on function-calling support that varies by
model/version), this takes the more predictable path -- prompt the model
explicitly for JSON, use Ollama's native format="json" enforcement
(syntactic validity only), then validate against the real Pydantic schema
ourselves. If validation fails, the raw response is included in the error
so debugging doesn't require re-running the LLM call blind.
"""

import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class StructuredOutputError(Exception):
    """Raised when the LLM's response can't be parsed/validated into the target schema."""
    pass


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def invoke_for_json(llm, system: str, user: str, output_model: Type[T], max_attempts: int = 3) -> T:
    """format="json" only guarantees syntactically valid JSON, not schema
    conformance (see module docstring) -- a model occasionally emits a
    plausible-looking value outside a Literal's allowed set (e.g. "good"
    where only "adequate"/"weak"/"missing" are valid). On a validation
    failure, retry once with the concrete error fed back to the model rather
    than failing the whole agent call over a single fixable mistake.
    """
    messages = [("system", system), ("human", user)] if system else [("human", user)]

    last_error: StructuredOutputError = None  # type: ignore[assignment]
    for attempt in range(1, max_attempts + 1):
        response = llm.invoke(messages)
        raw_text = response.content
        cleaned = _strip_code_fences(raw_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = StructuredOutputError(
                f"LLM response was not valid JSON even with format='json' enforcement.\n"
                f"Error: {e}\nRaw response:\n{raw_text}"
            )
            if attempt < max_attempts:
                messages = messages + [
                    ("ai", raw_text),
                    ("human", f"That was not valid JSON ({e}). Reply again with ONLY a valid JSON object, no other text."),
                ]
            continue

        try:
            return output_model.model_validate(data)
        except ValidationError as e:
            last_error = StructuredOutputError(
                f"LLM's JSON didn't match the expected {output_model.__name__} schema.\n"
                f"Validation errors: {e}\nParsed JSON:\n{json.dumps(data, indent=2)}"
            )
            if attempt < max_attempts:
                messages = messages + [
                    ("ai", raw_text),
                    ("human", f"That JSON didn't match the required schema. Validation errors:\n{e}\n"
                              f"Reply again with ONLY a corrected JSON object fixing exactly these errors."),
                ]
            continue

    raise last_error from None
