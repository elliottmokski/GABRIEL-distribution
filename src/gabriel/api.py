import asyncio
import os
import pandas as pd

from .tasks import (
    Ratings,
    RatingsConfig,
    Classification,
    ClassificationConfig,
    Deidentifier,
    DeidentifyConfig,
)

async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    additional_instructions: str | None = None,
    model: str = "o4-mini",
    n_parallels: int = 100,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "ratings.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Ratings`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = RatingsConfig(
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
    return await Ratings(cfg).run(df, column_name, reset_files=reset_files)

async def classify(
    df: pd.DataFrame,
    column_name: str,
    *,
    labels: dict[str, str],
    save_dir: str,
    additional_instructions: str | None = None,
    model: str = "o4-mini",
    n_parallels: int = 400,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "basic_classifier_responses.csv",
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Classification`."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = ClassificationConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        additional_instructions=additional_instructions or "",
        use_dummy=use_dummy,
        **cfg_kwargs,
    )
    return await Classification(cfg).run(df, column_name, reset_files=reset_files)


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: str | None = None,
    model: str = "o4-mini",
    n_parallels: int = 50,
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
