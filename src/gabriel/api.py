import asyncio
import os
import pandas as pd
from typing import Optional, Union

from .tasks import (
    Rate,
    RateConfig,
    Classify,
    ClassifyConfig,
    Rank,
    RankConfig,
    Deidentifier,
    DeidentifyConfig,
    Codify,
)
from .utils.openai_utils import get_all_responses
from .utils.passage_viewer import view_coded_passages as _view_coded_passages

async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 400,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "ratings.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Rate`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = RateConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        **cfg_kwargs,
    )
    return await Rate(cfg).run(df, column_name, reset_files=reset_files)

async def classify(
    df: pd.DataFrame,
    column_name: str,
    *,
    labels: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 400,
    n_runs: int = 1,
    min_frequency: float = 0.6,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "classify_responses.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Classify`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = ClassifyConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        min_frequency=min_frequency,
        additional_instructions=additional_instructions or "",
        use_dummy=use_dummy,
        **cfg_kwargs,
    )
    return await Classify(cfg).run(df, column_name, reset_files=reset_files)


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 400,
    use_dummy: bool = False,
    file_name: str = "deidentified.csv",
    max_words_per_call: int = 7500,
    guidelines: str = "",
    additional_guidelines: str = "",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Deidentifier`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = DeidentifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        max_words_per_call=max_words_per_call,
        guidelines=guidelines,
        additional_guidelines=additional_guidelines,
        **cfg_kwargs,
    )
    return await Deidentifier(cfg).run(df, column_name, grouping_column=grouping_column)


async def rank(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Union[dict[str, str], list[str]],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_rounds: int = 4,
    matches_per_round: int = 5,
    power_matching: bool = True,
    add_zscore: bool = True,
    compute_se: bool = True,
    learning_rate: float = 0.1,
    n_parallels: int = 400,
    use_dummy: bool = False,
    file_name: str = "rankings",
    reset_files: bool = False,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Rank`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = RankConfig(
        attributes=attributes,
        n_rounds=n_rounds,
        matches_per_round=matches_per_round,
        power_matching=power_matching,
        add_zscore=add_zscore,
        compute_se=compute_se,
        learning_rate=learning_rate,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        save_dir=save_dir,
        file_name=file_name,
        additional_instructions=additional_instructions or "",
        **cfg_kwargs,
    )
    return await Rank(cfg).run(df, column_name, reset_files=reset_files)


async def codify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[dict[str, str]] = None,
    user_instructions: str = "",
    additional_instructions: str = "",
    model: str = "gpt-5-mini",
    n_parallels: int = 400,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    file_name: str = "coding_results.csv",
    reset_files: bool = False,
    debug_print: bool = False,
    use_dummy: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Codify`."""
    os.makedirs(save_dir, exist_ok=True)
    coder = Codify()
    return await coder.codify(
        df,
        column_name,
        categories=categories,
        user_instructions=user_instructions,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        additional_instructions=additional_instructions,
        n_parallels=n_parallels,
        model=model,
        save_dir=save_dir,
        file_name=file_name,
        reset_files=reset_files,
        debug_print=debug_print,
        use_dummy=use_dummy,
    )


async def whatever(
    prompts: list[str],
    identifiers: list[str],
    *,
    save_dir: str,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    use_web_search: bool = False,
    n_parallels: int = 400,
    use_dummy: bool = False,
    reset_files: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper around :func:`get_all_responses` for arbitrary prompts.

    Results are saved to ``save_dir/file_name``.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    return await get_all_responses(
        prompts=prompts,
        identifiers=identifiers,
        save_path=save_path,
        model=model,
        json_mode=json_mode,
        use_web_search=use_web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reset_files=reset_files,
        **kwargs,
    )


async def custom_prompt(
    prompts: list[str],
    identifiers: list[str],
    *,
    save_dir: str,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    use_web_search: bool = False,
    n_parallels: int = 400,
    use_dummy: bool = False,
    reset_files: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Backward compatible alias for :func:`whatever`."""

    return await whatever(
        prompts,
        identifiers,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        use_web_search=use_web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reset_files=reset_files,
        **kwargs,
    )


def view_coded_passages(
    df: pd.DataFrame,
    text_column: str,
    categories: Optional[Union[list[str], str]] = None,
    colab: bool = False,
):
    """Convenience wrapper for the passage viewer utility."""
    return _view_coded_passages(df, text_column, categories, colab=colab)
