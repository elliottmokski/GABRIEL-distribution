# src/gabriel/tasks/ratings.py
# ---------------------------------------------------------------------
# Robust “rate-a-passage” task – now shares the same parsing philosophy
# as BasicClassifier, so no more empty {} rows 🎉
# ---------------------------------------------------------------------
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class RatingsConfig:
    """Configuration for :class:`Ratings`."""

    attributes: Dict[str, str]          # {"clarity": "desc", ...}
    model: str = "gpt-4o-mini"
    n_parallels: int = 50
    save_path: str = "ratings.csv"
    use_dummy: bool = False
    timeout: float = 60.0

# ---------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------
class Ratings:
    """Rate passages on a set of attributes (0-100)."""

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    # -----------------------------------------------------------------
    # Helpers copied from BasicClassifier for identical robustness
    # -----------------------------------------------------------------
    @staticmethod
    def _safe_json(txt: Any) -> dict:
        """Best-effort JSON → dict conversion (handles strings, lists…)."""
        try:
            if isinstance(txt, dict):
                return txt
            if isinstance(txt, list):
                if txt and isinstance(txt[0], dict):
                    return txt[0]
                if txt and isinstance(txt[0], str):
                    txt = txt[0]
            cleaned = str(txt).strip()
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                cleaned.startswith("'") and cleaned.endswith("'")
            ):
                cleaned = cleaned[1:-1]
            try:
                return json.loads(cleaned)
            except Exception:
                try:
                    return ast.literal_eval(cleaned)
                except Exception:
                    return {}
        except Exception:
            return {}
        return {}

    # -----------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------
    def __init__(self, cfg: RatingsConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    # -----------------------------------------------------------------
    # Parsing – now mirrors classifier logic + numeric handling
    # -----------------------------------------------------------------
    def _parse(self, raw: Any) -> Dict[str, Optional[float]]:
        """Convert raw LLM output to {attribute: float}."""
        # unwrap single-element lists / bytes
        if isinstance(raw, list) and len(raw) == 1:
            raw = raw[0]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()

        # strip markdown/code fences
        if isinstance(raw, str):
            m = self._FENCE_RE.search(raw)
            if m:
                raw = m.group(1).strip()

        obj = self._safe_json(raw)
        out: Dict[str, Optional[float]] = {}

        # -- Shape A: {"data":[{"attribute":"clarity","rating":77},…]}
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            obj = obj["data"]

        # -- Shape B: [{"attribute":"clarity","rating":77},…]
        if isinstance(obj, list):
            for entry in obj:
                if not isinstance(entry, dict):
                    continue
                attr = str(entry.get("attribute", "")).strip()
                val = entry.get("rating")
                try:
                    out[attr] = float(val)
                except Exception:
                    out[attr] = None
            return out

        # -- Shape C: {"clarity":77, "humor":12}
        if isinstance(obj, dict):
            for attr, val in obj.items():
                try:
                    out[attr] = float(val)
                except Exception:
                    out[attr] = None
            return out

        # -- Shape D: fallback regex "clarity: 77"
        text = str(raw)
        for attr in self.cfg.attributes:
            m = re.search(rf"{re.escape(attr)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
            out[attr] = float(m.group(1)) if m else None
        return out

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    async def run(self, texts: List[str]) -> pd.DataFrame:
        """Call the LLM on every unique passage and return a DataFrame."""
        prompts: List[str] = []
        unique_ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)

        # Build prompts (deduplicate identical passages by SHA-1)
        for row, passage in enumerate(texts):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue  # duplicate passage, no need to re-prompt
            prompts.append(
                self.template.render(
                    attributes=list(self.cfg.attributes.keys()),
                    descriptions=list(self.cfg.attributes.values()),
                    passage=passage,
                    object_category="text",
                    attribute_category="attributes",
                )
            )
            unique_ids.append(sha8)

        # JSON schema hint improves compliance
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "attribute": {"type": "string"},
                            "rating": {"type": "number"},
                        },
                        "required": ["attribute", "rating"],
                    },
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        }

        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=unique_ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=self.cfg.save_path,
            use_dummy=self.cfg.use_dummy,
            timeout=self.cfg.timeout,
            json_mode=True,
            expected_schema=schema,
        )

        # Map responses back to all rows
        id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {}
        for ident, resp in zip(df_resp["Identifier"], df_resp["Response"]):
            main = resp[0] if isinstance(resp, list) and resp else resp
            id_to_ratings[ident] = self._parse(main)

        ratings_list: List[Dict[str, Optional[float]]] = []
        for passage in texts:
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            ratings_list.append(id_to_ratings.get(sha8, {}))

        return pd.DataFrame({"text": texts, "ratings": ratings_list})

