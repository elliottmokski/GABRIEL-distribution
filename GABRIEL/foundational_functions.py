import asyncio
import json
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from gabriel.teleprompter import teleprompter

# Instead of directly calling get_response, we will accept a `client` argument that has `get_response`.
async def async_get_single_response(prompt, system_instruction, timeout, temperature, model, client, max_tokens=1000):
    responses, time_taken = await client.get_response(
        prompt, 
        model=model, 
        system_instruction=system_instruction, 
        max_tokens=max_tokens, 
        temperature=temperature,
        json_mode=False,
        timeout=timeout
    )
    return responses[0]

async def get_description_for_attribute(attribute, attribute_category, description_length,
                                        timeout=75, temperature=0.8, model='gpt-4o-mini', client=None, **kwargs):
    system_instruction = "Please provide a brief description."
    prompt = teleprompter.attribute_description_prompt(attribute, attribute_category, description_length)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)
    return response.strip()

async def generate_category_items(category, n_items, mode='item', 
                                  object_clarification=None, attribute_clarification=None, 
                                  timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125', 
                                  client=None, **kwargs):
    system_instruction = "Please list the items or attributes."
    prompt = teleprompter.list_generation_prompt(category, n_items, mode, object_clarification, attribute_clarification)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)

    items = re.split(r',\s|\n', response)
    items = [re.sub(r'\d+\.\s', '', item).strip() for item in items if item.strip()]
    items = [re.sub(r'[^a-zA-Z0-9\s]', '', item) for item in items]
    return items

async def generate_simple_ratings(attributes, descriptions, passage, object_category, attribute_category, format='json',
                                  classification_clarification=None,
                                  timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125', client=None, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.ratings_prompt(attributes, descriptions, passage, object_category, attribute_category, classification_clarification, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)
    return response


async def generate_full_ratings(attribute_dict, passage, entity_category, attribute_category, attributes, format='json',
                                timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125',
                                client=None, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.ratings_prompt_full(attribute_dict, passage, entity_category, attribute_category, attributes, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)
    return response

async def generate_simple_classification(attributes, descriptions, passage, object_category, attribute_category, format='json',
                                         classification_clarification=None,
                                         timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125',
                                         client=None, **kwargs):
    if format == 'json':
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    prompt = teleprompter.classification_prompt(attributes, descriptions, passage, object_category, attribute_category, classification_clarification, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)
    return response

async def identify_categories(task_description, format='json',
                              timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125',
                              client=None, **kwargs):
    if format == 'json':
        system_instruction = 'Please output well-defined categories as requested, following the detailed JSON template.'
    else:
        system_instruction = 'Please output well-defined categories as requested, using the provided format.'

    prompt = teleprompter.identify_categories_prompt(task_description, format=format)
    response = await async_get_single_response(prompt, system_instruction, timeout, temperature, model, client)
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