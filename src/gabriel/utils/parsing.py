from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

DEFAULT_JSON_LLM_MODEL = os.getenv("JSON_LLM_MODEL", "gpt-4o-mini")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_json(txt: Any) -> dict | list:
    """Synchronously parse JSON with multiple fallbacks."""
    if isinstance(txt, (dict, list)):
        return txt

    if isinstance(txt, list) and txt:
        return parse_json(txt[0])

    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode(errors="ignore")

    if txt is None:
        return {}

    cleaned = str(txt).strip()

    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    m = _JSON_FENCE_RE.search(cleaned)
    if m:
        cleaned = m.group(1).strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(cleaned)
        except Exception:
            pass

    brace = re.search(r"\{[\s\S]*\}", cleaned)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass

    return {}


async def safe_json(
    txt: Any,
    *,
    json_llm_model: str | None = None,
    use_dummy: bool = False,
) -> dict | list:
    """Parse JSON, optionally reformatted by an LLM on failure."""
    result = parse_json(txt)
    if result:
        return result

    cleaned = str(txt).strip() if txt is not None else ""
    if not cleaned:
        return {}

    model = json_llm_model or DEFAULT_JSON_LLM_MODEL
    try:
        from .openai_utils import get_response
        fixed, _ = await get_response(
            prompt=(
                "Please parse the following text **without changing any "
                "content** into valid JSON. This is a pure formatting task.\n\n"
                + cleaned
            ),
            model=model,
            json_mode=True,
            use_dummy=use_dummy,
        )
        return parse_json(fixed[0])
    except Exception:
        return {}
