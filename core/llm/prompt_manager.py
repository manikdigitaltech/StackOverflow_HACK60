"""
Loads and renders prompt templates from prompts.yaml.

Templates use simple {variable} placeholders (Python str.format style)
rather than full Jinja2 -- our prompts don't need conditionals or loops,
so avoiding an extra templating dependency keeps this simpler.

Known limitation: since this uses str.format(), any LITERAL curly braces
in a template's text (e.g. showing a raw JSON example) would need to be
escaped as {{ }}. None of our current templates do this, but worth
remembering if a future prompt needs to show literal JSON syntax.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


class PromptManager:
    def __init__(self, prompts_path: Path = _PROMPTS_PATH):
        with open(prompts_path, "r", encoding="utf-8") as f:
            self._templates: Dict[str, Any] = yaml.safe_load(f) or {}

    def render(self, template_name: str, **kwargs) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt), both fully rendered."""
        template = self._templates.get(template_name)
        if template is None:
            raise KeyError(f"No prompt template named '{template_name}' in prompts.yaml")

        system_raw = template.get("system", "") or ""
        user_raw = template.get("user", "") or ""

        system = system_raw.format(**kwargs) if system_raw else ""
        user = user_raw.format(**kwargs)
        return system, user

    def get_output_schema_name(self, template_name: str) -> Optional[str]:
        template = self._templates.get(template_name)
        return template.get("output_schema") if template else None
