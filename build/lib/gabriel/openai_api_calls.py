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
    "gpt-4o": {"input": 2.50,"output":10.00,"cached_input":1.25},
    "gpt-4o-mini": {"input":0.15,"output":0.60,"cached_input":0.075},
    "o1-preview": {"input":15.00,"output":60.00,"cached_input":7.50},
    "o1-mini": {"input":3.00,"output":12.00,"cached_input":1.50},
    "text-embedding-3-small":{"input":0.020,"batch_input":0.010},
    "text-embedding-3-large":{"input":0.130,"batch_input":0.065},
    "ada-v2":{"input":0.10,"batch_input":0.050},
    "gpt-3.5-turbo":{"input":1.50,"output":2.00},
    "gpt-3.5-turbo-16k":{"input":3.00,"output":4.00},
    "gpt-4":{"input":30.00,"output":60.00},
    "gpt-4-32k":{"input":60.00,"output":120.00},
    "gpt-4-turbo":{"input":10.00,"output":30.00},
    "gpt-4-vision-preview":{"input":10.00,"output":30.00},
    "whisper":{"input":0.006},
    "tts":{"input":15.00,"hd":30.00},
    "dall-e-3":{
        "standard_1024x1024":0.04,
        "standard_1792x1024":0.08,
        "hd_1024x1024":0.08,
        "hd_1792x1024":0.12
    },
    "dall-e-2":{
        "256x256":0.016,
        "512x512":0.018,
        "1024x1024":0.020
    }
}

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.openai_pricing = openai_pricing

    async def get_response(
        self,
        prompt: str,
        model: str = 'gpt-4o-mini',
        system_instruction: str = 'Please provide a helpful response.',
        n: int = 1,
        max_tokens: int = 1000,
        temperature: float = 0.9,
        json_mode: bool = False,
        expected_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
        **kwargs
    ) -> (List[str], float):
        import aiohttp
        import time
        if json_mode:
            system_instruction = system_instruction + " Output the response in JSON format."

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        }
        params.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        url = "https://api.openai.com/v1/chat/completions"

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=params, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise Exception(f"API call failed: {resp.status}: {await resp.text()}")
                    data = await resp.json()
        except asyncio.TimeoutError:
            raise Exception(f"API call timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"API call resulted in exception: {e}")

        responses = []
        for choice in data.get('choices', []):
            response_message = choice['message']['content']
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
        tokens_per_minute: int = 1500000000,
        rate_limit_factor: float = 0.8,
        truncate_middle: bool = True,
        timeout: int = 60,
        max_retries: int = 7,
        save_every_x_responses: int = 1000,
        save_every_x_seconds: Optional[int] = None,
        format_template: Optional[str] = None,
        model="gpt-4o-mini",
        system_instruction="Please provide a helpful response.",
        temperature=0.9,
        json_mode=False,
        **kwargs,
    ) -> pd.DataFrame:
        # Similar logic as before, but worker uses self.get_response
        from aiolimiter import AsyncLimiter
        import aiohttp

        if identifiers is None:
            identifiers = prompts

        print(f"Model used: {model}")

        if os.path.exists(save_path) and not reset_files:
            df = pd.read_csv(save_path)
            existing_identifiers = set(df["Identifier"])
            prompts_to_process = [
                (p, id) for p, id in zip(prompts, identifiers) if id not in existing_identifiers
            ]
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

        max_context_tokens = 128000
        max_allowed_prompt_tokens = max_context_tokens - max_tokens
        truncated_count = 0

        def truncate_prompt(prompt: str) -> Tuple[str, bool]:
            words = prompt.split()
            num_tokens = int(math.ceil(len(words) * 1.5))
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

        effective_requests_per_minute = requests_per_minute * rate_limit_factor
        effective_tokens_per_minute = tokens_per_minute * rate_limit_factor

        request_limiter = AsyncLimiter(max_rate=int(effective_requests_per_minute), time_period=60)
        token_limiter = AsyncLimiter(max_rate=int(effective_tokens_per_minute), time_period=60)

        results = []
        processed_responses = 0
        last_save_time = time.time()

        queue = asyncio.Queue()
        for prompt, identifier in updated_prompts_to_process:
            queue.put_nowait((prompt, identifier))

        total_tasks = queue.qsize()

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

        async def worker(worker_id: int):
            nonlocal processed_responses
            while True:
                try:
                    prompt, identifier = await queue.get()
                except asyncio.CancelledError:
                    break

                base_timeout = timeout
                if "o1" in model:
                    if model == "o1":
                        base_timeout *= 6
                    else:
                        base_timeout *= 3

                attempt = 1
                while attempt <= max_retries:
                    attempt_timeout = base_timeout + 30 * (attempt - 1)
                    try:
                        prompt_words = prompt.split()
                        prompt_tokens = int(math.ceil(len(prompt_words) * 1.5))
                        total_tokens_for_request = (prompt_tokens + max_tokens)*n
                        await request_limiter.acquire()
                        await token_limiter.acquire(total_tokens_for_request)

                        responses, time_taken = await asyncio.wait_for(
                            self.get_response(
                                prompt,
                                model=model,
                                system_instruction=system_instruction,
                                n=n,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                json_mode=json_mode,
                                timeout=attempt_timeout,
                                **kwargs,
                            ),
                            timeout=attempt_timeout,
                        )
                        total_time_taken = time_taken

                        if format_template is not None and "o1" in model:
                            clean_responses = []
                            from gabriel.teleprompter import teleprompter
                            for response in responses:
                                cleaning_prompt = teleprompter.clean_json_prompt(
                                    response, format_template
                                )
                                clean_response, clean_time_taken = await asyncio.wait_for(
                                    self.get_response(
                                        cleaning_prompt,
                                        n=1,
                                        max_tokens=max_tokens,
                                        timeout=attempt_timeout,
                                        model="gpt-4o-mini",
                                        json_mode=True,
                                        **kwargs,
                                    ),
                                    timeout=attempt_timeout,
                                )
                                total_time_taken += clean_time_taken
                                clean_responses.append(clean_response)
                            responses = clean_responses

                        results.append(
                            {
                                "Identifier": identifier,
                                "Response": responses,
                                "Time Taken": total_time_taken,
                            }
                        )

                        processed_responses += 1

                        if processed_responses % save_every_x_responses == 0:
                            await save_results()

                        break  # Success
                    except asyncio.TimeoutError:
                        print(f"Worker {worker_id}, attempt {attempt}: Prompt {identifier} timed out.")
                        attempt += 3
                        if attempt > max_retries:
                            print(f"Prompt {identifier} failed after max retries due to timeout.")
                            results.append({"Identifier": identifier, "Response":None,"Time Taken":None})
                            processed_responses+=1
                            if processed_responses%save_every_x_responses==0:
                                await save_results()
                            break
                        else:
                            await asyncio.sleep(5)
                    except Exception as e:
                        print(f"Worker {worker_id}, Error for {identifier}, attempt {attempt}: {e}")
                        attempt+=1
                        if attempt>max_retries:
                            results.append({"Identifier":identifier,"Response":None,"Time Taken":None})
                            processed_responses+=1
                            if processed_responses%save_every_x_responses==0:
                                await save_results()
                            break
                        else:
                            await asyncio.sleep(5)
                queue.task_done()

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
                if queue.empty() and processed>=total_tasks:
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving current progress.")
            for worker_task in workers:
                worker_task.cancel()
            await save_results()
            return df

        await queue.join()

        for worker_task in workers:
            worker_task.cancel()

        await save_results()
        if save_every_x_seconds is not None:
            periodic_save_task.cancel()

        pbar.close()
        print(f"All responses saved to {save_path}")

        time_taken_series = df["Time Taken"].dropna()
        if len(time_taken_series)>0:
            percentiles=[0,10,25,50,75,90,95,99,100]
            vals=np.percentile(time_taken_series,percentiles)
            print("\nTime taken summary (seconds):")
            for p,v in zip(percentiles,vals):
                print(f"{p}th: {v:.2f}")

        return df
