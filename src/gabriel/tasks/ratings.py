from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class RatingsConfig:
    """Configuration for :class:`Ratings`."""

    attributes: Dict[str, str]
    model: str = "gpt-3.5-turbo"
    n_parallels: int = 50
    save_path: str = "ratings.csv"
    use_dummy: bool = False
    timeout: float = 60.0


class Ratings:
    """Rate passages on a set of attributes."""

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    def __init__(self, cfg: RatingsConfig, template: PromptTemplate | None = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    @staticmethod
    def _safe_json(txt: Any) -> Optional[dict]:
        """Best-effort JSON parser."""
        candidate: Any = txt
        if candidate is None:
            return None
        if isinstance(candidate, list) and len(candidate) == 1:
            candidate = candidate[0]
        if isinstance(candidate, (bytes, bytearray)):
            candidate = candidate.decode()
        if isinstance(candidate, str):
            m = Ratings._FENCE_RE.search(candidate)
            cleaned = m.group(1) if m else candidate
            cleaned = cleaned.strip()
            try:
                return json.loads(cleaned)
            except Exception:
                pass
            try:
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                pass
            return None
        if isinstance(candidate, dict):
            return candidate
        return None

    def _parse(self, txt: Any) -> Dict[str, Optional[float]]:
        data = self._safe_json(txt) or {}
        items = data.get("data") if isinstance(data, dict) else None
        out: Dict[str, Optional[float]] = {}
        if isinstance(items, list):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                attr = str(entry.get("attribute", "")).strip()
                val = entry.get("rating")
                try:
                    out[attr] = float(val)
                except Exception:
                    out[attr] = None if val is None else str(val)
        return out

    async def run(self, texts: List[str]) -> pd.DataFrame:
        prompts: List[str] = []
        unique_ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        for row, passage in enumerate(texts):
            key = hashlib.sha1(passage.encode()).hexdigest()[:8]
            id_to_rows[key].append(row)
            if len(id_to_rows[key]) > 1:
                continue
            prompts.append(
                self.template.render(
                    attributes=list(self.cfg.attributes.keys()),
                    descriptions=list(self.cfg.attributes.values()),
                    passage=passage,
                    object_category="text",
                    attribute_category="attributes",
                )
            )
            unique_ids.append(key)

        df = await get_all_responses(
            prompts=prompts,
            identifiers=unique_ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=self.cfg.save_path,
            use_dummy=self.cfg.use_dummy,
            timeout=self.cfg.timeout,
            json_mode=True,
        )

        id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {}
        for ident, resp in zip(df["Identifier"], df["Response"]):
            main = resp[0] if isinstance(resp, list) and resp else resp
            id_to_ratings[ident] = self._parse(main)

        ratings_list: List[Dict[str, Optional[float]]] = []
        for row in range(len(texts)):
            sha = hashlib.sha1(texts[row].encode()).hexdigest()[:8]
            ratings_list.append(id_to_ratings.get(sha, {}))

        return pd.DataFrame({"text": texts, "ratings": ratings_list})
