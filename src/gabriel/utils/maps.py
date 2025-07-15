from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def create_county_choropleth(
    df: pd.DataFrame,
    fips_col: str,
    value_col: str,
    title: str = "County Ratings",
    color_scale: str = "RdBu",
    font_family: str = "monospace",
    save_path: Optional[str] = None,
    county_col: Optional[str] = None,
    z_score: bool = False,
):
    """Create a simple county-level choropleth map using Plotly."""
    import json
    import requests
    import numpy as np
    import plotly.express as px
    from scipy.stats import zscore

    geojson_url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    )
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "county_geo.json")
    if not os.path.exists(cache_path):
        r = requests.get(geojson_url, timeout=30)
        r.raise_for_status()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(cache_path, encoding="utf-8") as f:
        counties = json.load(f)

    df = df.copy()
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)

    if county_col is None:
        for cand in ["county", "County", "region", "Region"]:
            if cand in df.columns:
                county_col = cand
                break
    if county_col is None:
        df["_county_name"] = ""
        county_col = "_county_name"

    plot_col = value_col
    if z_score:
        vals = df[value_col].values.astype(float)
        if len(vals) > 1 and np.nanstd(vals) > 0:
            zs = zscore(vals, nan_policy="omit")
        else:
            zs = np.zeros_like(vals)
        plot_col = f"_zscore_{value_col}"
        df[plot_col] = zs
        color_scale = "RdBu" if color_scale == "RdBu" else "PuOr"

    hover_data = {county_col: True, fips_col: True, value_col: True}

    fig = px.choropleth(
        df,
        geojson=counties,
        locations=fips_col,
        color=plot_col,
        color_continuous_scale=color_scale,
        scope="usa",
        hover_data=hover_data,
    )
    if z_score:
        fig.update_coloraxes(cmid=0)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            fig.write_image(save_path, scale=3)
        else:
            fig.write_html(save_path)
    return fig
