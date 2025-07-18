from __future__ import annotations

import ast
import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def safe_json(txt: Any) -> dict | list:
    """Best-effort JSON parser returning ``{}`` on failure."""
    if isinstance(txt, (dict, list)):
        return txt

    if isinstance(txt, list) and txt:
        return safe_json(txt[0])

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

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    try:
        brace = re.search(r"\{[\s\S]*\}", cleaned)
        if brace:
            return json.loads(brace.group(0))
    except Exception:
        pass

    return {}
