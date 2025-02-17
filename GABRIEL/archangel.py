import os
import json
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional

from gabriel.openai_api_calls import OpenAIClient
from gabriel.teleprompter import teleprompter

import tiktoken

def count_message_tokens(messages, model_name):
    """
    Helper function to approximate or count tokens in chat messages.
    Uses tiktoken if possible; otherwise a fallback encoding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["content"]))
    return total_tokens

class Archangel:
    def __init__(
        self, 
        api_key: str, 
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize Archangel, which internally sets up an OpenAIClient.

        Args:
            api_key (str): Your OpenAI API key.
            endpoint_url (str, optional): Custom endpoint for the OpenAIClient. 
                                          Defaults to OpenAI's standard if None.
        """
        self.api_key = api_key
        self.client = OpenAIClient(
            api_key=api_key,
            endpoint_url=endpoint_url or "https://api.openai.com/v1/chat/completions"
        )
        self.openai_pricing = self.client.openai_pricing

    def ensure_no_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utility to remove duplicate columns if they appear.
        """
        return df.loc[:, ~df.columns.duplicated()]

    def compute_final_results(
        self, 
        df: pd.DataFrame, 
        mode: str, 
        attributes: List[str], 
        multiple_runs: bool, 
        n_responses: int
    ):
        """
        Post-processing:
         - classification mode: tie-break among multiple runs
         - ratings mode: compute average among multiple runs
        """
        if mode == 'classification':
            if multiple_runs:
                label_run_cols = [
                    f"label_run_{r}" for r in range(1, n_responses + 1)
                    if f"label_run_{r}" in df.columns
                ]
                def tie_break(row):
                    labels = [row[c] for c in label_run_cols if pd.notna(row[c])]
                    if labels:
                        freq = {}
                        for lbl in labels:
                            freq[lbl] = freq.get(lbl, 0) + 1
                        max_count = max(freq.values())
                        for lbl, count in freq.items():
                            if count == max_count:
                                return lbl
                    return None
                df['winning_label'] = df.apply(tie_break, axis=1)
        else:  # "ratings"
            if multiple_runs:
                for attr in attributes:
                    run_cols = [c for c in df.columns if c.startswith(f"{attr}_run_")]
                    if run_cols:
                        # convert all runs to numeric
                        for c in run_cols:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                        # create an average column
                        df[f"{attr}_average"] = df[run_cols].mean(axis=1, skipna=True)

    async def estimate_analysis_cost(
        self,
        texts: List[str],
        attribute_dict: Dict[str, str],
        mode: str = 'ratings',
        num_runs: int = 1,
        save_folder: str = '.',
        file_name: str = 'analysis_results.csv',
        model: str = 'gpt-4o-mini',
        max_tokens: int = 1000,
        format: str = 'json',
        truncate_len: Optional[int] = None,
        truncate: bool = False,
        reset_files: bool = False,
        entity_category: str = 'entities',
        topic: str = '',
        prompt_output_format: str = 'json',
        task_description: Optional[str] = None,
        reasoning_model: bool = False,
        reasoning_token_scaling: float = 1.0
    ) -> float:
        """
        Standalone function to compute the total estimated cost for unprocessed rows,
        *without* calling identify_categories (i.e., no additional LLM usage).

        For "ratings" mode, we simply use placeholder values for `entity_category_` 
        and `attribute_category_` to build rating prompts.
        """
        if mode not in ['ratings', 'classification']:
            raise ValueError("mode must be 'ratings' or 'classification'.")

        if mode == 'ratings':
            if not task_description or not isinstance(task_description, str) or not task_description.strip():
                raise ValueError("task_description must be provided and non-empty for ratings mode.")

        attributes = list(attribute_dict.keys())
        n_responses = num_runs if (num_runs and num_runs > 0) else 1
        multiple_runs = (n_responses > 1)

        full_path = os.path.join(save_folder, file_name)
        if reset_files and os.path.isfile(full_path):
            os.remove(full_path)

        if not os.path.isfile(full_path):
            # If no CSV, create minimal DF
            if mode == 'classification':
                df = pd.DataFrame({'Text': texts})
            else:
                if multiple_runs:
                    df = pd.DataFrame({'Text': texts})
                else:
                    df = pd.DataFrame(columns=['Text'] + attributes)
                    df['Text'] = texts
            df['ID'] = df.index
        else:
            df = pd.read_csv(full_path)
            if 'ID' not in df.columns:
                df['ID'] = df.index

        df = self.ensure_no_duplicates(df)

        # For ratings mode, do NOT call identify_categories. Use placeholders:
        if mode == 'ratings':
            entity_category_ = 'entities'
            attribute_category_ = 'N/A'
        else:
            entity_category_ = entity_category or 'entities'
            attribute_category_ = None

        # Create or ensure columns to detect unprocessed rows
        if mode == 'classification':
            if multiple_runs:
                run_cols = [f"label_run_{r}" for r in range(1, n_responses + 1)]
                for col in run_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                rows_to_process = df.loc[df[run_cols].isna().any(axis=1)]
            else:
                if 'winning_label' not in df.columns:
                    df['winning_label'] = np.nan
                rows_to_process = df.loc[df['winning_label'].isna()]
        else:
            # "ratings"
            if multiple_runs:
                run_cols = []
                for attr in attributes:
                    for r in range(1, n_responses + 1):
                        col = f"{attr}_run_{r}"
                        if col not in df.columns:
                            df[col] = np.nan
                        run_cols.append(col)
                rows_to_process = df.loc[df[run_cols].isna().any(axis=1)]
            else:
                if not attributes:
                    # if no attributes, entire DF is "unprocessed"
                    rows_to_process = df
                else:
                    for attr in attributes:
                        if attr not in df.columns:
                            df[attr] = np.nan
                    rows_to_process = df.loc[df[attributes].isna().all(axis=1)]

        if not isinstance(texts, list):
            texts = list(texts)

        if rows_to_process.empty:
            # No new rows => cost is zero
            return 0.0

        process_texts = rows_to_process['Text'].tolist()

        if truncate and truncate_len:
            process_texts = [
                ' '.join(txt.split()[:truncate_len]) 
                for txt in process_texts
            ]

        # Basic system instruction
        if mode == 'classification':
            system_instruction = (
                "Please output precise classification as requested, "
                "following the detailed JSON template."
            )
        else:
            system_instruction = (
                "Please output precise ratings as requested, "
                "following the detailed JSON template."
            )

        # Build prompts
        prompts = []
        for text_val in process_texts:
            if mode == 'classification':
                possible_classes = list(attribute_dict.keys())
                defs_str = ""
                for cls, definition in attribute_dict.items():
                    defs_str += f"'{cls}': {definition}\n\n"
                prompt = teleprompter.generic_classification_prompt(
                    entity_list=[text_val],
                    possible_classes=possible_classes,
                    class_definitions=defs_str,
                    entity_category=entity_category_,
                    output_format=prompt_output_format
                )
            else:
                prompt = teleprompter.ratings_prompt_full(
                    attribute_dict=attribute_dict,
                    passage=text_val,
                    entity_category=entity_category_,
                    attribute_category=attribute_category_,
                    attributes=attributes,
                    format=format
                )
            prompts.append(prompt)

        if not prompts:
            return 0.0

        # Sum cost using self.client.get_cost
        total_cost = 0.0
        for prompt in prompts:
            cost_for_this_prompt = self.client.get_cost(
                prompt=prompt,
                model=model,
                system_instruction=system_instruction,
                n=n_responses,
                max_tokens=max_tokens,
                reasoning_model=reasoning_model,
                reasoning_token_scaling=reasoning_token_scaling
            )
            total_cost += cost_for_this_prompt

        return total_cost

    async def run_analysis(
        self,
        texts: List[str],
        attribute_dict: Dict[str, str],
        mode: str = 'ratings',
        num_runs: int = 1,
        save_folder: str = '.',
        file_name: str = 'analysis_results.csv',
        model: str = 'gpt-4o-mini',
        n_parallels: int = 100,
        temperature: float = 0.8,
        timeout: int = 75,
        requests_per_minute: int = 40000,
        tokens_per_minute: int = 150000000,
        max_tokens: int = 1000,
        format: str = 'json',
        truncate_len: Optional[int] = None,
        seed: int = 42,
        truncate: bool = False,
        reset_files: bool = False,
        entity_category: str = 'entities',
        topic: str = '',
        guidance_template: Optional[str] = None,
        prompt_output_format: str = 'json',
        max_prompts_per_batch: int = 25000,
        terms_per_prompt: int = 40,
        task_description: Optional[str] = None,
        reasoning_model: bool = False,
        reasoning_token_scaling: float = 1.0
    ) -> pd.DataFrame:
        """
        Actually *executes* calls to the model (via get_all_responses) and merges results 
        into a CSV. For RATINGS mode, here we DO call identify_categories (if you want).
        
        **Now also prints** the estimated cost before running.
        """
        if mode not in ['ratings', 'classification']:
            raise ValueError("mode must be 'ratings' or 'classification'.")

        if mode == 'ratings':
            if not task_description or not isinstance(task_description, str) or not task_description.strip():
                raise ValueError("task_description must be provided for ratings mode.")

        # --------------------------------------------
        # 1) Estimate cost and print it
        # --------------------------------------------
        estimated_cost = await self.estimate_analysis_cost(
            texts=texts,
            attribute_dict=attribute_dict,
            mode=mode,
            num_runs=num_runs,
            save_folder=save_folder,
            file_name=file_name,
            model=model,
            max_tokens=max_tokens,
            format=format,
            truncate_len=truncate_len,
            truncate=truncate,
            reset_files=reset_files,
            entity_category=entity_category,
            topic=topic,
            prompt_output_format=prompt_output_format,
            task_description=task_description,
            reasoning_model=reasoning_model,
            reasoning_token_scaling=reasoning_token_scaling
        )

        # Print cost with reasoning info
        if reasoning_model:
            print(
                f"Estimated cost for this run: ${estimated_cost:.4f} "
                f"(includes reasoning, scaling factor={reasoning_token_scaling})"
            )
        else:
            print(f"Estimated cost for this run: ${estimated_cost:.4f} (no extra reasoning)")

        # --------------------------------------------
        # 2) Proceed with the actual run
        # --------------------------------------------
        attributes = list(attribute_dict.keys())
        n_responses = num_runs if (num_runs and num_runs > 0) else 1
        multiple_runs = (n_responses > 1)

        full_path = os.path.join(save_folder, file_name)
        responses_save_path = full_path.replace('.csv', '_responses.csv')

        if reset_files and os.path.isfile(full_path):
            os.remove(full_path)
        if reset_files and os.path.isfile(responses_save_path):
            os.remove(responses_save_path)

        if not os.path.isfile(full_path):
            # Create or initialize the CSV
            if mode == 'classification':
                df = pd.DataFrame({'Text': texts})
            else:
                if multiple_runs:
                    df = pd.DataFrame({'Text': texts})
                else:
                    df = pd.DataFrame(columns=['Text'] + attributes)
                    df['Text'] = texts
            df['ID'] = df.index
            df.to_csv(full_path, index=False)
            print(f"Created new file at {full_path}")
        else:
            df = pd.read_csv(full_path)
            print("File exists; loaded DataFrame.")
            if 'ID' not in df.columns:
                df['ID'] = df.index
                df.to_csv(full_path, index=False)

        df = self.ensure_no_duplicates(df)

        # For RATINGS, we DO call identify_categories here if you need real categories
        if mode == 'ratings':
            from gabriel.foundational_functions import identify_categories
            cats_json = await identify_categories(
                task_description=task_description, 
                format='json',
                client=self.client
            )
            cats = json.loads(cats_json)
            entity_category_ = cats['entity category']
            attribute_category_ = cats['attribute category']
        else:
            entity_category_ = entity_category or 'entities'
            attribute_category_ = None

        # Add missing columns
        if mode == 'classification':
            if multiple_runs:
                for col in [f"label_run_{r}" for r in range(1, n_responses + 1)]:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                if 'winning_label' not in df.columns:
                    df['winning_label'] = np.nan
        else:
            # "ratings"
            if multiple_runs:
                for attr in attributes:
                    for r in range(1, n_responses + 1):
                        run_col = f"{attr}_run_{r}"
                        if run_col not in df.columns:
                            df[run_col] = np.nan
            else:
                for attr in attributes:
                    if attr not in df.columns:
                        df[attr] = np.nan

        df = self.ensure_no_duplicates(df)

        # Identify which rows are unprocessed
        if mode == 'classification':
            if multiple_runs:
                label_run_cols = [f"label_run_{r}" for r in range(1, n_responses + 1)]
                rows_to_process = df.loc[df[label_run_cols].isna().any(axis=1)]
            else:
                rows_to_process = df.loc[df['winning_label'].isna()]
        else:
            if multiple_runs:
                expected_run_cols = [
                    f"{attr}_run_{r}" for attr in attributes
                    for r in range(1, n_responses + 1)
                ]
                rows_to_process = df.loc[df[expected_run_cols].isna().any(axis=1)]
            else:
                if attributes:
                    rows_to_process = df.loc[df[attributes].isna().all(axis=1)]
                else:
                    rows_to_process = df

        if not isinstance(texts, list):
            texts = list(texts)

        if rows_to_process.empty:
            print("No new texts to process.")
            self.compute_final_results(df, mode, attributes, multiple_runs, n_responses)
            df.to_csv(full_path, index=False)
            return df

        process_ids = rows_to_process['ID'].tolist()
        process_texts = rows_to_process['Text'].tolist()

        if truncate and truncate_len:
            process_texts = [
                ' '.join(txt.split()[:truncate_len])
                for txt in process_texts
            ]

        # Basic system instruction
        if mode == 'classification':
            system_instruction = (
                "Please output precise classification as requested, "
                "following the detailed JSON template."
            )
        else:
            system_instruction = (
                "Please output precise ratings as requested, "
                "following the detailed JSON template."
            )

        # Build prompts
        prompts = []
        prompt_identifiers = []
        for text_id, text_val in zip(process_ids, process_texts):
            if mode == 'classification':
                defs_str = ""
                for cls, definition in attribute_dict.items():
                    defs_str += f"'{cls}': {definition}\n\n"
                prompt = teleprompter.generic_classification_prompt(
                    entity_list=[text_val],
                    possible_classes=list(attribute_dict.keys()),
                    class_definitions=defs_str,
                    entity_category=entity_category_,
                    output_format=prompt_output_format
                )
            else:
                # "ratings" prompt w/ real categories from identify_categories
                prompt = teleprompter.ratings_prompt_full(
                    attribute_dict=attribute_dict,
                    passage=text_val,
                    entity_category=entity_category_,
                    attribute_category=attribute_category_ if attribute_category_ else "N/A",
                    attributes=attributes,
                    format=format
                )
            prompts.append(prompt)
            prompt_identifiers.append(f"id_{text_id}")

        if not prompts:
            print("No new prompts to process.")
            self.compute_final_results(df, mode, attributes, multiple_runs, n_responses)
            df.to_csv(full_path, index=False)
            return df

        # Actually call the model
        get_response_kwargs = {
            "system_instruction": system_instruction,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n_responses
        }

        responses_df = await self.client.get_all_responses(
            prompts=prompts,
            identifiers=prompt_identifiers,
            n_parallels=n_parallels,
            save_path=responses_save_path,
            reset_files=reset_files,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            timeout=timeout,
            **get_response_kwargs
        )

        if responses_df is None or responses_df.empty:
            print("No responses returned.")
            self.compute_final_results(df, mode, attributes, multiple_runs, n_responses)
            df.to_csv(full_path, index=False)
            return df

        # Parse + merge
        parsed_rows = []
        for idx, row in responses_df.iterrows():
            identifier = row['Identifier']
            response_list = row['Response']
            if not response_list:
                continue
            text_id = int(identifier.split('_')[-1])
            row_data = {'ID': text_id}

            if mode == 'classification':
                # Each response presumably JSON with a single label
                for run_num, response_str in enumerate(response_list, start=1):
                    try:
                        response_dict = json.loads(response_str) if response_str else {}
                        label = response_dict.get(process_texts[process_ids.index(text_id)], None)
                        if multiple_runs:
                            run_col = f"label_run_{run_num}"
                            if pd.isna(df.loc[df['ID'] == text_id, run_col].values[0]) and label is not None:
                                row_data[run_col] = label
                        else:
                            if pd.isna(df.loc[df['ID'] == text_id, 'winning_label'].values[0]) and label is not None:
                                row_data['winning_label'] = label
                    except Exception as e:
                        print(f"Failed to parse classification run {run_num} for {identifier}: {e}")
                        if multiple_runs:
                            row_data[f"label_run_{run_num}"] = None
                        else:
                            row_data['winning_label'] = None
            else:
                # "ratings" mode
                for run_num, response_str in enumerate(response_list, start=1):
                    try:
                        response_dict = json.loads(response_str) if response_str else {}
                        # example usage: each attr in attribute_dict
                        attr_ratings = {}
                        for attr in attributes:
                            val = response_dict.get(attr, None)
                            if val is not None:
                                try:
                                    val = float(val)
                                except:
                                    pass
                                attr_ratings[attr] = val

                        if multiple_runs:
                            for attr in attributes:
                                run_col = f"{attr}_run_{run_num}"
                                val = attr_ratings.get(attr, None)
                                if val is not None and pd.isna(df.loc[df['ID'] == text_id, run_col].values[0]):
                                    row_data[run_col] = val
                        else:
                            for attr in attributes:
                                val = attr_ratings.get(attr, None)
                                if val is not None and pd.isna(df.loc[df['ID'] == text_id, attr].values[0]):
                                    row_data[attr] = val
                    except Exception as e:
                        print(f"Failed to parse ratings run {run_num} for {identifier}: {e}")

            parsed_rows.append(row_data)

        if not parsed_rows:
            print("No valid responses parsed.")
            self.compute_final_results(df, mode, attributes, multiple_runs, n_responses)
            df.to_csv(full_path, index=False)
            return df

        parsed_df = pd.DataFrame(parsed_rows)
        parsed_df = self.ensure_no_duplicates(parsed_df)

        # Merge
        for i, new_row in parsed_df.iterrows():
            row_id = new_row['ID']
            mask = (df['ID'] == row_id)
            if mask.any():
                for col in new_row.index:
                    if col == 'ID':
                        continue
                    val = new_row[col]
                    if val is not None and pd.isna(df.loc[mask, col].values[0]):
                        df.loc[mask, col] = val

        df = self.ensure_no_duplicates(df)

        # Final tie-break or averaging
        self.compute_final_results(df, mode, attributes, multiple_runs, n_responses)
        df.to_csv(full_path, index=False)
        return df