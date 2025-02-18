import asyncio
import json
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from gabriel.teleprompter import teleprompter

# Instead of directly calling get_response, we will accept a `client` argument that has `get_response`.
async def async_get_single_response(prompt, system_instruction, timeout, temperature, model, client, max_tokens=1000,
                                    exclude_system_prompt=False):
    responses, time_taken = await client.get_response(
        prompt,
        model=model,
        system_instruction=system_instruction,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=False,
        timeout=timeout,
        exclude_system_prompt=exclude_system_prompt
    )
    return responses[0]

async def get_description_for_attribute(attribute, attribute_category, description_length,
                                        timeout=75, temperature=0.8, model='gpt-4o-mini', client=None,
                                        exclude_system_prompt=False, **kwargs):
    system_instruction = "Please provide a brief description."
    prompt = teleprompter.attribute_description_prompt(attribute, attribute_category, description_length)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    return response.strip()

async def generate_category_items(category, n_items, mode='item',
                                  object_clarification=None, attribute_clarification=None,
                                  timeout=90, temperature=0.8, model='gpt-4o-mini',
                                  client=None, exclude_system_prompt=False, **kwargs):
    system_instruction = "Please list the items or attributes."
    prompt = teleprompter.list_generation_prompt(category, n_items, mode, object_clarification, attribute_clarification)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    items = re.split(r',\s|\n', response)
    items = [re.sub(r'\d+\.\s', '', item).strip() for item in items if item.strip()]
    items = [re.sub(r'[^a-zA-Z0-9\s]', '', item) for item in items]
    return items

async def generate_simple_ratings(attributes, descriptions, passage, object_category, attribute_category, format='json',
                                  classification_clarification=None,
                                  timeout=90, temperature=0.8, model='gpt-4o-mini', client=None,
                                  exclude_system_prompt=False, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.ratings_prompt(attributes, descriptions, passage, object_category, attribute_category,
                                         classification_clarification, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    return response

async def generate_full_ratings(attribute_dict, passage, entity_category, attribute_category, attributes, format='json',
                                timeout=90, temperature=0.8, model='gpt-4o-mini',
                                client=None, exclude_system_prompt=False, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.ratings_prompt_full(attribute_dict, passage, entity_category, attribute_category, attributes, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    return response

async def generate_simple_classification(attributes, descriptions, passage, object_category, attribute_category, format='json',
                                         classification_clarification=None,
                                         timeout=90, temperature=0.8, model='gpt-4o-mini',
                                         client=None, exclude_system_prompt=False, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.classification_prompt(attributes, descriptions, passage, object_category, attribute_category,
                                                classification_clarification, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    return response

async def identify_categories(task_description, format='json',
                              timeout=90, temperature=0.8, model='gpt-4o-mini',
                              client=None, exclude_system_prompt=False, **kwargs):
    if format == 'json':
        system_instruction = 'Please output well-defined categories as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output well-defined categories as requested, using the provided format.'

    prompt = teleprompter.identify_categories_prompt(task_description, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature,
                                               model, client, exclude_system_prompt=exclude_system_prompt)
    return response

def ensure_no_duplicates(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def stabilize_schema_on_load(df, attributes, num_runs):
    df = ensure_no_duplicates(df)
    if num_runs == 1:
        for attr in attributes:
            run_col = f"{attr}_run_1"
            if run_col in df.columns:
                if attr in df.columns:
                    mask = df[attr].isna() & df[run_col].notna()
                    df.loc[mask, attr] = df.loc[mask, run_col]
                    df.drop(columns=[run_col], inplace=True)
                else:
                    df.rename(columns={run_col: attr}, inplace=True)
    else:
        pass
    df = ensure_no_duplicates(df)
    return df

def update_dataframe(existing_df, new_df, attributes, word_merge=250):
    new_df = ensure_no_duplicates(new_df)
    existing_df['merge_words'] = existing_df['Text'].apply(lambda x: ' '.join(x.split()[:word_merge]))
    new_df['merge_words'] = new_df['Text'].apply(lambda x: ' '.join(x.split()[:word_merge]))

    attribute_like_columns = []
    for col in new_df.columns:
        if col in ['Text', 'merge_words', 'ID']:
            continue
        if any(col == attr or col.startswith(attr + '_run_') for attr in attributes):
            attribute_like_columns.append(col)
            if col not in existing_df.columns:
                existing_df[col] = np.nan

    for _, new_row in new_df.iterrows():
        text = new_row['merge_words']
        mask = existing_df['merge_words'] == text
        if mask.any():
            for col in attribute_like_columns:
                existing_df.loc[mask, col] = new_row[col]

    existing_df.drop(columns=['merge_words'], inplace=True)
    new_df.drop(columns=['merge_words'], inplace=True)

    existing_df = ensure_no_duplicates(existing_df)
    new_df = ensure_no_duplicates(new_df)
    return existing_df, new_df

def create_batch_info_dataframe(batch_instance):
    data = {
        'id': None,
        'completion_window': None,
        'created_at': None,
        'endpoint': None,
        'input_file_id': None,
        'object': None,
        'status': None
    }
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    return df

# NEW robust JSON loader
async def robust_json_loads(
    json_str: str,
    client=None,
    json_model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    temperature: float = 0.0,
    clean_json_prompt_file: str = "clean_json_prompt.j2"
) -> dict:
    """
    Attempt to parse JSON robustly:
      1) Try direct json.loads()
      2) If that fails, try extracting the largest {...} via regex and parse
      3) If that fails, call the client with a 'clean_json_prompt.j2' template prompt
         (passing the entire dirty JSON), then parse again.

    Returns a dict (empty if all attempts fail).
    """
    # Step 1: Direct parse
    try:
        return json.loads(json_str)
    except:
        pass

    # Step 2: Attempt regex extraction of a JSON object
    match = re.search(r"\{[\s\S]*\}", json_str)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except:
            pass

    # Step 3: Use the LLM to repair the JSON, if a client is provided
    if client:
        from gabriel.teleprompter import teleprompter
        system_instruction = "Please fix the JSON so that it is valid JSON. Return only valid JSON with no extra text."
        prompt = teleprompter.render_template(clean_json_prompt_file, {"dirty_json_output": json_str})
        try:
            responses, _ = await client.get_response(
                prompt=prompt,
                model=json_model,
                temperature=temperature,
                max_tokens=max_tokens,
                exclude_system_prompt=True
            )
            if responses:
                for r in responses:
                    try:
                        return json.loads(r)
                    except:
                        continue
        except:
            pass

    # If all else fails, return empty
    return {}
