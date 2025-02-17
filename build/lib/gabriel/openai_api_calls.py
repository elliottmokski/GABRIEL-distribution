import aiohttp
import asyncio
import ast
import math
import time
import json
import pandas as pd
import numpy as np
import os
from typing import Any, Dict, Optional, List, Tuple
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

openai_pricing = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00, "cached_input": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40, "cached_input": 0.30},
    "o1-preview": {"input": 15.00, "output": 60.00, "cached_input": 7.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o1-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "text-embedding-3-small": {"input": 0.020, "batch_input": 0.010},
    "text-embedding-3-large": {"input": 0.130, "batch_input": 0.065},
    "ada-v2": {"input": 0.10, "batch_input": 0.050},
    "gpt-3.5-turbo": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-vision-preview": {"input": 10.00, "output": 30.00},
    "whisper": {"input": 0.006},
    "tts": {"input": 15.00, "hd": 30.00},
    "dall-e-3": {
        "standard_1024x1024": 0.04,
        "standard_1792x1024": 0.08,
        "hd_1024x1024": 0.08,
        "hd_1792x1024": 0.12
    },
    "dall-e-2": {
        "256x256": 0.016,
        "512x512": 0.018,
        "1024x1024": 0.020
    }
}

class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        endpoint_url: str = "https://api.openai.com/v1/chat/completions"
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key (str): Your OpenAI API key.
            endpoint_url (str): URL endpoint for the chat completion API 
                                (defaults to OpenAI's standard endpoint).
        """
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.openai_pricing = openai_pricing

    def get_cost(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        system_instruction: str = "Please provide a helpful response for academic research.",
        n: int = 1,
        max_tokens: int = 4000,
        temperature: float = 0.9,
        json_mode: bool = False,
        expected_schema: Optional[Dict[str, Any]] = None,
        reasoning_model: bool = False,
        reasoning_token_scaling: float = 1.0,
        **kwargs,
    ) -> float:
        """
        Estimate the cost (in USD) for a single prompt/model call *before* actually running it.

        - If `reasoning_model` is True, add cost for additional "reasoning tokens" which we assume
          equals `reasoning_token_scaling * total_output_tokens`, billed at the same rate as output.

        Args:
            prompt (str): The user prompt for which to estimate cost.
            model (str, optional): The model used. Defaults to "gpt-4o-mini".
            system_instruction (str, optional): System instruction for the AI model.
            n (int, optional): Number of responses (affects output tokens multiplied).
            max_tokens (int, optional): The maximum tokens in the response. Defaults to 4000.
            temperature (float, optional): Sampling temperature. Unused for cost, 
                                           but included for completeness.
            json_mode (bool, optional): If True, modifies system_instruction for JSON.
            expected_schema (Dict[str, Any], optional): Unused for cost, but included for completeness.
            reasoning_model (bool, optional): If True, we add additional "reasoning" token costs.
            reasoning_token_scaling (float, optional): Multiplier for extra "reasoning" tokens,
                                                       default = 1.0 (same amount as output tokens).
            **kwargs: Additional parameters that do not affect token count in this example.

        Returns:
            float: Estimated cost in USD.
        """
        if json_mode:
            system_instruction = system_instruction + " Output the response in JSON format."

        # Prepare chat messages similarly to get_response
        if model.startswith("o"):
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]

        # --- Count tokens (approximation or with tiktoken if possible) ---
        def approximate_token_count(text: str) -> int:
            # Very rough approximation: 1 word ~ 1.5 tokens
            words = text.split()
            return int(math.ceil(len(words) * 1.5))

        try:
            encoding = tiktoken.encoding_for_model(model)
            total_input_tokens = 0
            for msg in messages:
                # Role + message content
                total_input_tokens += len(encoding.encode(msg["role"]))
                total_input_tokens += len(encoding.encode(msg["content"]))
                # A few tokens overhead per message
                total_input_tokens += 4
            # Add final reply separator tokens
            total_input_tokens += 2
        except:
            # Fallback to approximate
            concatenated = " ".join([m["content"] for m in messages])
            total_input_tokens = approximate_token_count(concatenated)

        # The output tokens are at most `max_tokens` for each of the n responses
        total_output_tokens = max_tokens * n

        # Look up model pricing
        if model not in self.openai_pricing:
            # If not found, default to 0 cost
            return 0.0

        input_rate = self.openai_pricing[model].get("input", 0.0)
        output_rate = self.openai_pricing[model].get("output", 0.0)

        cost_input = (total_input_tokens / 1_000_000) * input_rate
        cost_output = (total_output_tokens / 1_000_000) * output_rate

        # If reasoning_model is True, add cost for "reasoning tokens"
        # reasoning tokens = reasoning_token_scaling * total_output_tokens (billed at output rate)
        if reasoning_model:
            reasoning_tokens = reasoning_token_scaling * total_output_tokens
            cost_reasoning = (reasoning_tokens / 1_000_000) * output_rate
            return cost_input + cost_output + cost_reasoning

        return cost_input + cost_output

    async def get_response(
        self, 
        prompt: str,
        model: str = "gpt-4o-mini",
        system_instruction: str = "Please provide a helpful response to this inquiry for purposes of academic research.",
        n: int = 1,
        max_tokens: int = 4000,
        temperature: float = 0.9,
        json_mode: bool = False,
        expected_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,  # Timeout in seconds
        **kwargs,
    ) -> Tuple[List[str], float]:
        """
        Fetches responses from the OpenAI API based on the provided prompt and parameters.
        This version uses asynchronous execution with aiohttp and supports multiple responses.
        JSON schema is optional even if json_mode is True.

        Args:
            prompt (str): The user prompt to send to the API.
            model (str, optional): The model to use. Defaults to 'gpt-4o-mini'.
            system_instruction (str, optional): The system instruction for the AI model.
                                                Defaults to a predefined instruction.
            n (int, optional): Number of responses to generate. Defaults to 1.
            max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 4000.
            temperature (float, optional): The sampling temperature. Defaults to 0.9.
            json_mode (bool, optional): If True, expects the response in JSON format.
            expected_schema (Dict[str, Any], optional): A dictionary defining the expected JSON schema of the response.
            timeout (float, optional): Timeout in seconds for the API call.
            **kwargs: Additional optional parameters for the OpenAI API.

        Returns:
            Tuple[List[str], float]: The list of AI model's responses and the time taken to retrieve them.

        Raises:
            Exception: If the API call fails or encounters an exception.
        """

        if json_mode:
            system_instruction = system_instruction + " Output the response in JSON format."

        # Prepare the messages for the API call
        if model.startswith("o"):
            messages = [{"role": "user", "content": prompt}]
            params = {
                "model": model,
                "messages": messages,
                "n": n,
            }
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": n,
            }
        # Additional parameters
        params.update(kwargs)

        # Handle JSON mode (hypothetical usage; not an actual OpenAI param)
        if json_mode:
            if expected_schema is not None:
                params["response_format"] = {"type": "json_schema", "json_schema": expected_schema}
            else:
                params["response_format"] = {"type": "json_object"}

        url = self.endpoint_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=params, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise Exception(
                            f"API call failed with status code {resp.status}: {await resp.text()}"
                        )
                    data = await resp.json()
        except asyncio.TimeoutError:
            raise Exception(f"API call timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"API call resulted in exception: {e}")

        # Extract the responses
        responses = []
        for choice in data.get("choices", []):
            response_message = choice["message"]["content"]
            responses.append(response_message)

        end_time = time.time()
        total_time = end_time - start_time

        return responses, total_time

    async def get_all_responses(
        self, 
        prompts: List[str],
        identifiers: Optional[List[str]] = None,
        n_parallels: int = 100,
        save_path: str = "temp.csv",
        reset_files: bool = False,
        rich_print: bool = False,
        n: int = 1,
        max_tokens: int = 1000,
        requests_per_minute: int = 40000,
        tokens_per_minute: int = 15000000000,
        rate_limit_factor: float = 0.8,
        truncate_middle: bool = True,
        timeout: int = 60,
        max_retries: int = 7,
        save_every_x_responses: int = 1000,
        save_every_x_seconds: Optional[int] = None,
        format_template: Optional[str] = None,
        **get_response_kwargs,
    ) -> pd.DataFrame:
        """
        Fetches responses for a list of prompts using specified LLM models and records the time taken for each response.
        Processes prompts using an asyncio.Queue and a fixed number of worker coroutines.
        """
        if identifiers is None:
            identifiers = prompts  # Use prompts as identifiers if none provided

        model = get_response_kwargs.get("model", "gpt-4o-mini")
        print(f"Model used: {model}")

        # Load existing data if save_path exists
        if os.path.exists(save_path) and not reset_files:
            df = pd.read_csv(save_path)
            existing_identifiers = set(df["Identifier"])
            prompts_to_process = [
                (p, id) for p, id in zip(prompts, identifiers) if id not in existing_identifiers
            ]

            # Parse the 'Response' column back into lists
            df["Response"] = df["Response"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        else:
            df = pd.DataFrame(columns=["Identifier", "Response", "Time Taken"])
            prompts_to_process = list(zip(prompts, identifiers))

        total_prompts = len(prompts_to_process)
        print(f"Total prompts to process: {total_prompts}")

        if total_prompts == 0:
            print("No new prompts to process.")
            return df

        # Prompt truncation
        max_context_tokens = 128000
        max_allowed_prompt_tokens = max_context_tokens - max_tokens
        truncated_count = 0

        def truncate_prompt(prompt: str) -> Tuple[str, bool]:
            words = prompt.split()
            num_tokens = int(math.ceil(len(words) * 1.5))  # Approx tokens

            if num_tokens <= max_allowed_prompt_tokens:
                return prompt, False

            allowed_words_count = int(max_allowed_prompt_tokens / 1.5)

            if allowed_words_count <= 0:
                return "\n\n...\n\n", True

            half = allowed_words_count // 2
            truncated_prompt = " ".join(words[:half]) + "\n\n...\n\n" + " ".join(words[-half:])
            return truncated_prompt, True

        updated_prompts_to_process = []
        for prompt, identifier in prompts_to_process:
            truncated_prompt, was_truncated = (
                truncate_prompt(prompt) if truncate_middle else (prompt, False)
            )
            if was_truncated:
                truncated_count += 1
            updated_prompts_to_process.append((truncated_prompt, identifier))

        truncation_percent = (truncated_count / total_prompts) * 100 if total_prompts > 0 else 0
        print(
            f"Truncated {truncated_count} prompts out of {total_prompts} ({truncation_percent:.2f}%)."
        )

        # Rate limits
        effective_requests_per_minute = requests_per_minute * rate_limit_factor
        effective_tokens_per_minute = tokens_per_minute * rate_limit_factor

        request_limiter = AsyncLimiter(
            max_rate=int(effective_requests_per_minute), time_period=60
        )
        token_limiter = AsyncLimiter(
            max_rate=int(effective_tokens_per_minute), time_period=60
        )

        # Storage
        results = []
        processed_responses = 0
        last_save_time = time.time()

        queue = asyncio.Queue()
        for prompt, identifier in updated_prompts_to_process:
            queue.put_nowait((prompt, identifier))

        total_tasks = queue.qsize()

        async def worker(worker_id: int):
            nonlocal processed_responses
            while True:
                try:
                    prompt, identifier = await queue.get()
                except asyncio.CancelledError:
                    break

                base_timeout = timeout
                model_name = get_response_kwargs.get("model", "gpt-4o-mini")

                # Increase timeout for certain "o"-models
                if model_name.startswith("o"):
                    if model_name in ["o1", "o3"]:
                        base_timeout *= 6
                    else:
                        base_timeout *= 2

                attempt = 1
                while attempt <= max_retries:
                    attempt_timeout = base_timeout + 30 * (attempt - 1)
                    try:
                        # Rate limiting
                        prompt_words = prompt.split()
                        prompt_tokens = int(math.ceil(len(prompt_words) * 1.5))
                        total_tokens_for_request = (prompt_tokens + max_tokens) * n

                        await request_limiter.acquire()
                        await token_limiter.acquire(total_tokens_for_request)

                        responses, time_taken = await asyncio.wait_for(
                            self.get_response(
                                prompt,
                                n=n,
                                max_tokens=max_tokens,
                                timeout=attempt_timeout,
                                **get_response_kwargs,
                            ),
                            timeout=attempt_timeout,
                        )

                        results.append(
                            {
                                "Identifier": identifier,
                                "Response": responses,
                                "Time Taken": time_taken,
                            }
                        )

                        processed_responses += 1
                        if processed_responses % save_every_x_responses == 0:
                            await save_results()

                        break  # success

                    except asyncio.TimeoutError:
                        print(
                            f"Worker {worker_id}, attempt {attempt}: Prompt {identifier} timed out after {attempt_timeout} seconds."
                        )
                        attempt += 3
                        if attempt > max_retries:
                            print(
                                f"Prompt {identifier} failed after {max_retries} attempts due to timeout."
                            )
                            results.append(
                                {"Identifier": identifier, "Response": None, "Time Taken": None}
                            )
                            processed_responses += 1
                            if processed_responses % save_every_x_responses == 0:
                                await save_results()
                            break
                        else:
                            await asyncio.sleep(5)
                    except Exception as e:
                        print(
                            f"Worker {worker_id}, Error for Identifier {identifier}, attempt {attempt}: {e}"
                        )
                        attempt += 1
                        if attempt > max_retries:
                            results.append(
                                {"Identifier": identifier, "Response": None, "Time Taken": None}
                            )
                            processed_responses += 1
                            if processed_responses % save_every_x_responses == 0:
                                await save_results()
                            break
                        else:
                            await asyncio.sleep(5)
                queue.task_done()

        async def save_results():
            nonlocal results, last_save_time, df
            if results:
                batch_df = pd.DataFrame(results)
                df = pd.concat([df, batch_df], ignore_index=True)
                df.to_csv(save_path, index=False)
                results = []
                last_save_time = time.time()
                print(f"Results saved to {save_path}")

        async def periodic_save():
            while True:
                await asyncio.sleep(save_every_x_seconds)
                await save_results()

        # Start workers
        workers = []
        for i in range(n_parallels):
            worker_task = asyncio.create_task(worker(i))
            workers.append(worker_task)

        if save_every_x_seconds is not None:
            periodic_save_task = asyncio.create_task(periodic_save())

        pbar = tqdm(total=total_tasks, desc="Processing prompts")

        try:
            while True:
                await asyncio.sleep(1)
                processed = processed_responses
                pbar.n = processed
                pbar.refresh()
                if queue.empty() and processed >= total_tasks:
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving current progress and stopping.")
            for worker_task in workers:
                worker_task.cancel()
            await save_results()
            return df

        # Wait for queue to empty
        await queue.join()

        # Cancel worker tasks
        for worker_task in workers:
            worker_task.cancel()

        await save_results()
        if save_every_x_seconds is not None:
            periodic_save_task.cancel()

        pbar.close()
        print(f"All responses saved to {save_path}")

        # Time summary
        time_taken_series = df["Time Taken"].dropna()
        if len(time_taken_series) > 0:
            percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
            percentile_values = np.percentile(time_taken_series, percentiles)

            print("\nTime taken summary (in seconds):")
            for p, val in zip(percentiles, percentile_values):
                print(f"{p}th percentile: {val:.2f}")

        return df