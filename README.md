# GABRIEL

TRIAL NOTEBOOK: https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing. See this notebook for the most updated example set.

GABRIEL (Generalized Attribute Based Ratings Information Extraction Library) is a collection of utilities for running large language model driven analyses.  The library provides high level tasks such as passage rating, text classification, de‑identification, regional report generation and several Elo style ranking utilities.

The current `src` directory contains a cleaned up and asynchronous implementation.  Each task exposes an easy to use `run()` coroutine and sensible configuration dataclasses. 

## Quick Start

```python
from gabriel.tasks import Rate, RateConfig

cfg = RateConfig(
    attributes={"clarity": "How understandable is the text?"},
    save_path="ratings.csv",
    use_dummy=True  # set to False to call the OpenAI API
)

texts = ["This is an example passage"]
ratings = asyncio.run(Rate(cfg).run(texts))
print(ratings)
```

Each task returns a `pandas.DataFrame` and saves raw responses to disk.  Set `use_dummy=False` and provide your OpenAI credentials via the `OPENAI_API_KEY` environment variable to perform real API calls.

### Image inputs

`get_response` and `get_all_responses` can optionally include images with your prompts. Pass the `images` parameter to `get_response` or a mapping `prompt_images` to `get_all_responses`, where each key is a prompt identifier and the value is a list of base64 strings. A helper `encode_image` is provided:

```python
from gabriel.utils import encode_image

img_b64 = encode_image("picture.jpg")
responses = asyncio.run(
    get_response("Describe this", images=[img_b64], use_dummy=True)
)
```


## Tasks

### `Rate`
Rate passages on a set of numeric attributes.  The task builds prompts using `gabriel.prompts.ratings_prompt.jinja2` and parses the JSON style output into a `dict` for each passage.

Key options (see `RateConfig`):
- `attributes` – mapping of attribute name to description.
- `model` – model name (default `o4-mini`).
- `n_parallels` – number of concurrent API calls.
- `save_path` – CSV file for intermediate results.
- `rating_scale` – optional custom rating scale text. If omitted, the default 0–100 scale from the template is used.

### `Classify`
Classify passages into boolean labels.  Uses a prompt in `basic_classifier_prompt.jinja2` and expects JSON `{label: true/false}` responses.

Options include the label dictionary, output directory, model and timeout.  Results are joined back onto the input DataFrame with one column per label.

### `Deidentifier`
Iteratively remove identifying information from text.  Texts are split into manageable chunks and the model returns JSON replacement mappings which are applied across all rows.

Configuration allows controlling the maximum words per call, LLM model and any additional guidelines for the prompt.

### `EloRater`
Pairwise Elo / Bradley–Terry rating of items across any set of attributes.  Prompts are built from the `rankings_prompt.jinja2` template and include explicit support for win/loss/draw outcomes.

`EloConfig` controls the number of rounds, matches per round, rating method (Elo, BT or Plackett–Luce), parallelism and more.  The final DataFrame includes rating, optional standard error and z-score columns.

### `RecursiveEloRater`
Higher level orchestrator that repeatedly applies `EloRater` on progressively filtered subsets.  Items can optionally be rewritten between stages and cumulative scores are tracked across recursion steps.

### `Regional`
Generate short reports for topics across regions (for example counties or states).  Results are stored in a wide DataFrame with one column per topic.

### `CountyCounter`
Convenience wrapper that chains a `Regional` run followed by Elo rating of each regional report.  Optionally produces Plotly choropleth maps if FIPS codes are provided.

## Utilities

The `gabriel.utils` module contains helpers for interacting with the OpenAI API, rendering prompt templates and creating visualisations.  The `OpenAIClient` class in `gabriel.core` provides a minimal asynchronous interface for customised pipelines.

## Running the Tests

Install the development dependencies and run `pytest`:

```bash
pip install -e .[dev]
pytest
```

All tests use `use_dummy=True` so no API key is required.

## Citation

If you use GABRIEL in your research, please cite:

> The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2023). <https://github.com/elliottmokski/GABRIEL-distribution>
