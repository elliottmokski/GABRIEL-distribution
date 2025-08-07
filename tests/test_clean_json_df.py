import asyncio
import pandas as pd
from gabriel.utils import clean_json_df


def test_clean_json_df_dummy():
    df = pd.DataFrame({"col": ['{"a": 1}', "{bad json"]})
    out = asyncio.run(clean_json_df(df, ["col"], model="dummy"))
    assert out.loc[0, "col_cleaned"] == '{"a": 1}'
    assert out.loc[1, "col_cleaned"][0].startswith("DUMMY")


def test_duplicate_index():
    df = pd.DataFrame({"col": ['{"a": 1}', "{bad json"]}, index=[0, 0])
    out = asyncio.run(clean_json_df(df, ["col"], model="dummy"))
    assert out.iloc[0]["col_cleaned"] == '{"a": 1}'
    assert out.iloc[1]["col_cleaned"][0].startswith("DUMMY")


def test_custom_save_path(tmp_path):
    df = pd.DataFrame({"col": ['{"a": 1}', "{bad json"]})
    cache_file = tmp_path / "cache.csv"
    out = asyncio.run(
        clean_json_df(df, ["col"], model="dummy", save_path=str(cache_file))
    )
    assert cache_file.exists()
    assert out.loc[0, "col_cleaned"] == '{"a": 1}'
    assert out.loc[1, "col_cleaned"][0].startswith("DUMMY")
