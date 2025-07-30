from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Union, Optional

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# model used when an LLM is required to reformat malformed JSON
JSON_LLM_MODEL = os.getenv("JSON_LLM_MODEL", "gpt-4o-mini")


def _parse_json(txt: Any) -> Union[dict, list]:
    """Strict JSON parsing with common cleaning heuristics."""
    if isinstance(txt, (dict, list)):
        return txt

    if isinstance(txt, list) and txt:
        return _parse_json(txt[0])

    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode(errors="ignore")

    if txt is None:
        raise ValueError("None provided")

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
            out = parser(cleaned)
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    brace = re.search(r"\{[\s\S]*\}", cleaned)
    if brace:
        try:
            out = json.loads(brace.group(0))
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    raise ValueError(f"Failed to parse JSON: {cleaned[:200]}")


def safe_json(txt: Any) -> Union[dict, list]:
    """Best-effort JSON parser returning ``{}`` on failure."""
    try:
        return _parse_json(txt)
    except Exception:
        return {}


async def safest_json(txt: Any, *, model: Optional[str] = None) -> Union[dict, list]:
    """Async wrapper around :func:`safe_json` with optional LLM fixup."""
    try:
        return _parse_json(txt)
    except Exception:
        if model is None:
            model = JSON_LLM_MODEL
        from gabriel.utils.openai_utils import get_response
        use_dummy = model == "dummy"
        fixed, _ = await get_response(
            prompt=(
                "Please parse the following text **without changing any content** "
                "into valid JSON. This is a pure formatting task.\n\n" + str(txt)
            ),
            model=model,
            json_mode=True,
            use_dummy=use_dummy,
        )
        if fixed:
            try:
                return _parse_json(fixed[0])
            except Exception:
                return {}
        return {}
