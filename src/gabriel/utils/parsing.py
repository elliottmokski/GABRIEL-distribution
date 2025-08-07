from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Union, Optional, List

import pandas as pd

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

    # attempt to strip `//` and `/* */` style comments before parsing
    try:
        no_line_comments = re.sub(r"(?<!:)//.*$", "", cleaned, flags=re.MULTILINE)
        no_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.S)
        out = json.loads(no_comments)
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

    bracket = re.search(r"\[[\s\S]*\]", cleaned)
    if bracket:
        candidate = bracket.group(0).strip()
        try:
            out = json.loads(candidate)
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

        m = re.fullmatch(r"\[\s*(['\"])(.*)\1\s*\]", candidate, re.S)
        if m:
            inner = m.group(2).strip()
            try:
                out = json.loads(inner)
                if isinstance(out, (dict, list)):
                    return out
            except Exception:
                inner_bracket = re.search(r"\[[\s\S]*\]", inner)
                if inner_bracket:
                    try:
                        out = json.loads(inner_bracket.group(0))
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


async def clean_json_df(
    df: pd.DataFrame,
    columns: List[str],
    *,
    id_col: str,
    model: str = "o4-mini",
    exclude_valid_json: bool = False,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Ensure specified DataFrame columns contain valid JSON.

    Parameters
    ----------
    df:
        Input DataFrame whose columns may contain malformed JSON strings.
    columns:
        Names of columns to inspect and clean.
    id_col:
        Name of a **unique** column in ``df`` used to merge cleaned JSON
        responses back into the original DataFrame. A :class:`ValueError` is
        raised if the column is missing or contains duplicate values.
    model:
        Model name passed to :func:`get_all_responses` when attempting to
        repair invalid JSON. Defaults to ``"o4-mini"``.
    exclude_valid_json:
        When ``False`` (default), only entries that fail to parse are sent to
        the model. When ``True``, all entries are processed regardless of
        validity.
    save_path:
        Optional path where the intermediate CSV from
        :func:`get_all_responses` will be saved. When ``None`` (default), a
        temporary file is created and deleted after processing.

    Returns
    -------
    DataFrame with new ``<column>_cleaned`` columns containing the cleaned
    JSON structures. Rows that were already valid retain their original value.
    """

    from gabriel.utils.openai_utils import get_all_responses
    import tempfile
    df = df.copy()

    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame")
    if not df[id_col].is_unique:
        raise ValueError(f"Column '{id_col}' must contain unique values")

    prompts: List[str] = []
    identifiers: List[str] = []
    # ``mapping`` maps each identifier to its originating column and ``id_col``
    # value so responses can be merged back using a stable key.
    mapping: dict[str, tuple[str, Any]] = {}

    for col in columns:
        cleaned_col = f"{col}_cleaned"
        df[cleaned_col] = None
        col_idx = df.columns.get_loc(cleaned_col)
        for row_pos, (id_val, val) in enumerate(zip(df[id_col], df[col])):
            valid = True
            try:
                _parse_json(val)
            except Exception:
                valid = False
            if exclude_valid_json or not valid:
                prompt = (
                    "Please parse the following text **without changing any content** "
                    "into valid JSON. This is a pure formatting task.\n\n" + str(val)
                )
                ident = f"{id_val}__{col}"
                prompts.append(prompt)
                identifiers.append(ident)
                mapping[ident] = (col, id_val)
            else:
                df.iat[row_pos, col_idx] = val

    if prompts:
        use_dummy = model == "dummy"
        cleanup = False
        if save_path is None:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            os.close(tmp_fd)
            os.remove(tmp_path)
            cleanup = True
        else:
            tmp_path = save_path
        try:
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                model=model,
                json_mode=True,
                use_dummy=use_dummy,
                print_example_prompt=False,
                save_path=tmp_path,
                reset_files=True,
            )
        finally:
            if cleanup:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        for _, row in resp_df.iterrows():
            ident = str(row.get("Identifier", "")).strip()
            if ident not in mapping:
                continue
            col, id_val = mapping[ident]
            col_idx = df.columns.get_loc(f"{col}_cleaned")
            df.loc[df[id_col] == id_val, df.columns[col_idx]] = row["Response"]

    return df
