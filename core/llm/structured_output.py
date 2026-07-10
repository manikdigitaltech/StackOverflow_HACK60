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
import logging
import re
import typing
from typing import Any, Dict, Literal, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel, ValidationError

from core.utils.guardrails import verify_output_safety

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Several of our schemas place a 3-point Literal ("adequate"/"weak"/"missing")
# next to a 4-point Literal ("poor"/"fair"/"good"/"excellent") on a sibling
# field in the same object (e.g. MethodologyAssessment.aspect_verdicts[].assessment
# vs MethodologyAssessment.soundness_rating). Weaker local models occasionally
# bleed a value from one scale into the other despite the prompt explicitly
# telling them not to. This is a deterministic, logged repair for exactly that
# known confusion -- it only fires if the mapped value is actually valid for
# the field that failed, so it can't paper over unrelated schema violations.
_LITERAL_SYNONYM_REPAIRS = {
    "excellent": "adequate",
    "good": "adequate",
    "fair": "weak",
    "poor": "weak",
}


class StructuredOutputError(Exception):
    """Raised when the LLM's response can't be parsed/validated into the target schema."""
    pass


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def _literal_choices_at_path(model: Type[BaseModel], loc: Tuple[Any, ...]) -> Optional[tuple]:
    """Walks a Pydantic error `loc` path (field names / list indices) through
    the model's type annotations and returns the allowed values if it lands
    on a Literal field, else None."""
    current: Any = model
    for key in loc:
        if isinstance(key, int):
            args = typing.get_args(current)
            if not args:
                return None
            current = args[0]
            continue
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return None
        field = current.model_fields.get(key)
        if field is None:
            return None
        current = field.annotation
    if typing.get_origin(current) is Literal:
        return typing.get_args(current)
    return None


def _set_at_path(data: Dict[str, Any], loc: Tuple[Any, ...], value: Any) -> None:
    target = data
    for key in loc[:-1]:
        target = target[key]
    target[loc[-1]] = value


def _attempt_literal_repair(data: Dict[str, Any], error: ValidationError, output_model: Type[BaseModel]) -> bool:
    """Deterministically fixes known cross-scale Literal mistakes in place.
    Returns True if at least one field was repaired."""
    repaired = False
    for err in error.errors():
        if err["type"] != "literal_error":
            continue
        invalid = err.get("input")
        if not isinstance(invalid, str):
            continue
        loc = err["loc"]
        choices = _literal_choices_at_path(output_model, loc)
        if choices is None:
            continue

        candidate = _LITERAL_SYNONYM_REPAIRS.get(invalid.strip().lower())
        if candidate is None and not invalid.strip() and "other" in choices:
            # A model that produces an entry it shouldn't (e.g.
            # VisualReferenceAgent's model inventing a "missing_target"
            # verdict for a figure it was told never to report on) has
            # nothing real to classify that entry's Literal field with, and
            # leaves it "" rather than picking a listed value. Map it to the
            # schema's own catch-all rather than failing the whole call over
            # one hallucinated-but-otherwise-harmless entry.
            candidate = "other"
        if candidate is None or candidate not in choices:
            continue

        _set_at_path(data, loc, candidate)
        logger.warning(
            "Repaired %s: field %s had out-of-scale value %r, mapped to %r.",
            output_model.__name__, ".".join(str(p) for p in loc), invalid, candidate,
        )
        repaired = True
    return repaired


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

        # Output guardrail: a syntactically valid response can still be a
        # hijacked one -- e.g. the model was steered into echoing our own
        # system prompt back instead of doing the review. verify_output_safety
        # only flags a *non-empty* leak; an empty response is a normal failure
        # handled by the JSON path below, so don't mislabel it as a breach.
        if raw_text and not verify_output_safety(raw_text):
            last_error = StructuredOutputError(
                "Output guardrail tripped: the LLM response looked like a leaked "
                "system prompt rather than a review answer (see "
                "core/utils/guardrails.verify_output_safety).\n"
                f"Raw response:\n{raw_text}"
            )
            if attempt < max_attempts:
                messages = messages + [
                    ("ai", raw_text),
                    ("human", "That response leaked internal prompt text instead of "
                              "answering. Reply again with ONLY the requested JSON object."),
                ]
            continue

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
            if _attempt_literal_repair(data, e, output_model):
                try:
                    return output_model.model_validate(data)
                except ValidationError as e2:
                    e = e2  # repaired the known confusion; feed back only what's still wrong

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
