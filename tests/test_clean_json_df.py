import asyncio
import pandas as pd
import pytest
from gabriel.utils.parsing import clean_json_df


def test_clean_json_df_dummy():
    df = pd.DataFrame({"id": [1, 2], "col": ['{"a": 1}', "{bad json"]})
    out = asyncio.run(clean_json_df(df, ["col"], id_col="id", model="dummy"))
    assert out.loc[0, "col_cleaned"] == '{"a": 1}'
    assert out.loc[1, "col_cleaned"].startswith("DUMMY")


def test_duplicate_index():
    df = pd.DataFrame({"id": [1, 2], "col": ['{"a": 1}', "{bad json"]}, index=[0, 0])
    out = asyncio.run(clean_json_df(df, ["col"], id_col="id", model="dummy"))
    assert out.iloc[0]["col_cleaned"] == '{"a": 1}'
    assert out.iloc[1]["col_cleaned"].startswith("DUMMY")


def test_non_unique_id_col():
    df = pd.DataFrame({"id": [1, 1], "col": ['{"a": 1}', "{bad json"]})
    with pytest.raises(ValueError):
        asyncio.run(clean_json_df(df, ["col"], id_col="id", model="dummy"))


def test_custom_save_path(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "col": ['{"a": 1}', "{bad json"]})
    cache_file = tmp_path / "cache.csv"
    out = asyncio.run(
        clean_json_df(df, ["col"], id_col="id", model="dummy", save_path=str(cache_file))
    )
    assert cache_file.exists()
    assert out.loc[0, "col_cleaned"] == '{"a": 1}'
    assert out.loc[1, "col_cleaned"].startswith("DUMMY")
