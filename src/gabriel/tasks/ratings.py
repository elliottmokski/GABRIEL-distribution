# src/gabriel/tasks/ratings.py
# ════════════════════════════════════════════════════════════════════
# Robust passage-rating task with optional debug logging.
# ════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safe_json


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class RatingsConfig:
    attributes: Dict[str, str]          # {"clarity": "description", ...}
    model: str = "o4-mini"
    n_parallels: int = 50
    save_path: str = "ratings.csv"
    use_dummy: bool = False
    timeout: float = 60.0
    rating_scale: str = "0-100"
    additional_guidelines: str = ""


# ────────────────────────────
# Main Ratings task
# ────────────────────────────
class Ratings:
    """Rate passages on specified attributes (0–100)."""


    # -----------------------------------------------------------------
    def __init__(self, cfg: RatingsConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    # -----------------------------------------------------------------
    # Parse raw LLM output into {attribute: float}
    # -----------------------------------------------------------------
    def _parse(self, raw: Any) -> Dict[str, Optional[float]]:
        obj = safe_json(raw)
        out: Dict[str, Optional[float]] = {}

        # shape A: {"data":[{"attribute":"clarity","rating":88}, …]}
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            obj = obj["data"]

        # shape B: list of dicts
        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                attr = str(item.get("attribute", "")).strip()
                try:
                    out[attr] = float(item.get("rating"))
                except Exception:
                    out[attr] = None
            return out

        # shape C: flat dict {"clarity": 88, "humor": 12}
        if isinstance(obj, dict):
            for attr, val in obj.items():
                try:
                    out[attr] = float(val)
                except Exception:
                    out[attr] = None
            return out

        # shape D: regex fallback "clarity: 88"
        text = str(raw)
        for attr in self.cfg.attributes:
            m = re.search(rf"{re.escape(attr)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
            out[attr] = float(m.group(1)) if m else None
        return out

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(self, texts: List[str], *, debug: bool = False) -> pd.DataFrame:
        """Return DataFrame with a 'ratings' column (dict per row)."""

        prompts: List[str] = []
        ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)

        # Build prompts, deduplicating identical passages
        for row, passage in enumerate(texts):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            prompts.append(
                self.template.render(
                    attributes=list(self.cfg.attributes.keys()),
                    descriptions=list(self.cfg.attributes.values()),
                    passage=passage,
                    object_category="text",
                    attribute_category="attributes",
                    rating_scale=self.cfg.rating_scale,
                    additional_guidelines=self.cfg.additional_guidelines,
                )
            )
            ids.append(sha8)

        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=self.cfg.save_path,
            use_dummy=self.cfg.use_dummy,
            timeout=self.cfg.timeout,
            json_mode=True,
        )

        # optional debug dump
        if debug:
            print("\n── raw LLM responses ──")
            for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                r = raw[0] if isinstance(raw, list) and raw else raw
                print(f"{ident} →\n{r}\n")
            print("────────────────────────\n")

        # parse and map back to every row
        id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {}
        for ident, raw in zip(df_resp.Identifier, df_resp.Response):
            main = raw[0] if isinstance(raw, list) and raw else raw
            id_to_ratings[ident] = self._parse(main)

        ratings_list: List[Dict[str, Optional[float]]] = []
        for passage in texts:
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            ratings_list.append(id_to_ratings.get(sha8, {}))

        return pd.DataFrame({"text": texts, "ratings": ratings_list})

