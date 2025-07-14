from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import openai

# single connection pool per process, created lazily
client_async: Optional[openai.AsyncOpenAI] = None


def _build_params(
    *,
    model: str,
    input_data: List[Dict[str, str]],
    max_tokens: int,
    system_instruction: str,
    temperature: float,
    json_mode: bool,
    expected_schema: Optional[Dict[str, Any]],
    **extra: Any,
) -> Dict[str, Any]:
    params = {
        "model": model,
        "input": input_data,
        "max_output_tokens": max_tokens,
        "truncation": "auto",
    }

    if json_mode:
        params["text"] = (
            {"format": {"type": "json_schema", "schema": expected_schema}}
            if expected_schema
            else {"format": {"type": "json_object"}}
        )

    if model.startswith("o"):
        params["reasoning"] = {"effort": "medium"}
    else:
        params["temperature"] = temperature

    params.update(extra)
    return params

async def get_response(
    prompt: str,
    *,
    model: str = "gpt-3.5-turbo",
    n: int = 1,
    max_tokens: int = 256,
    timeout: float = 90.0,
    temperature: float = 0.0,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    use_dummy: bool = False,
    **kwargs: Any,
) -> Tuple[List[str], float]:
    """Minimal async call to OpenAI's /responses endpoint or dummy response."""
    if use_dummy:
        return [f"DUMMY {prompt}" for _ in range(max(n, 1))], 0.0

    system_instruction = (
        "Please provide a helpful response to this inquiry for purposes of academic research."
    )

    input_data = (
        [{"role": "user", "content": prompt}]
        if model.startswith("o")
        else [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
    )

    params = _build_params(
        model=model,
        input_data=input_data,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
        temperature=temperature,
        json_mode=json_mode,
        expected_schema=expected_schema,
        **kwargs,
    )

    global client_async
    if client_async is None:
        client_async = openai.AsyncOpenAI()

    start = time.time()
    tasks = [client_async.responses.create(**params, timeout=timeout) for _ in range(max(n, 1))]
    try:
        raw = await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        raise Exception(f"API call timed out after {timeout} s")
    except Exception as e:
        raise Exception(f"API call resulted in exception: {e}")

    return [r.output_text for r in raw], time.time() - start


def _ser(x: Any) -> Optional[str]:
    """Serialize Python objects deterministically."""
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _de(x: Any) -> Any:
    """Deserialize JSON strings back to Python objects."""
    if pd.isna(x):
        return None
    try:
        return json.loads(x)
    except Exception:
        try:
            import ast

            return ast.literal_eval(x)
        except Exception:
            return None


async def get_all_responses(
    *,
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    n_parallels: int = 100,
    save_path: str = "temp.csv",
    reset_files: bool = False,
    n: int = 1,
    max_tokens: int = 25_000,
    requests_per_minute: int = 40_000,
    tokens_per_minute: int = 15_000_000_000,
    rate_limit_factor: float = 0.8,
    timeout: int = 90,
    max_retries: int = 7,
    save_every_x_responses: int = 1_000,
    save_every_x_seconds: Optional[int] = None,
    use_dummy: bool = False,
    print_example_prompt: bool = False,
    **get_response_kwargs: Any,
) -> pd.DataFrame:
    if identifiers is None:
        identifiers = prompts

    if os.path.exists(save_path) and not reset_files:
        df = pd.read_csv(save_path)
        df["Response"] = df["Response"].apply(_de)
        done = set(df["Identifier"])
    else:
        df = pd.DataFrame(columns=["Identifier", "Response", "Time Taken"])
        done = set()

    todo = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    total = len(todo)
    if total == 0:
        return df

    if print_example_prompt:
        print(f"Example prompt: {todo[0][0]}\n")

    if use_dummy:
        rows = [
            {"Identifier": i, "Response": [f"DUMMY {i}"] * n, "Time Taken": 0.0}
            for _, i in todo
        ]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(save_path, index=False)
        return df

    req_lim = AsyncLimiter(int(requests_per_minute * rate_limit_factor), 60)
    tok_lim = AsyncLimiter(int(tokens_per_minute * rate_limit_factor), 60)

    queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)

    results: List[Dict[str, Any]] = []
    processed = 0
    pbar = tqdm(total=total, desc="Processing prompts")

    async def flush() -> None:
        nonlocal results
        if results:
            batch = pd.DataFrame(results)
            batch["Response"] = batch["Response"].apply(_ser)
            batch.to_csv(
                save_path,
                mode="a",
                header=not os.path.exists(save_path),
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            results = []

    async def worker() -> None:
        nonlocal processed
        while True:
            try:
                prompt, ident = await queue.get()
            except asyncio.CancelledError:
                break

            attempt = 1
            while attempt <= max_retries:
                try:
                    approx = int(len(prompt.split()) * 1.5)
                    await req_lim.acquire()
                    await tok_lim.acquire((approx + max_tokens) * n)

                    resps, t = await asyncio.wait_for(
                        get_response(
                            prompt,
                            n=n,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            use_dummy=use_dummy,
                            **get_response_kwargs,
                        ),
                        timeout=timeout,
                    )

                    results.append({"Identifier": ident, "Response": resps, "Time Taken": t})
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x_responses == 0:
                        await flush()
                    break
                except Exception:
                    if attempt >= max_retries:
                        results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    await asyncio.sleep(5 * attempt)
                    attempt += 1
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(n_parallels)]
    if save_every_x_seconds:
        async def periodic() -> None:
            while True:
                await asyncio.sleep(save_every_x_seconds)
                await flush()
        ticker = asyncio.create_task(periodic())

    await queue.join()
    for w in workers:
        w.cancel()
    await flush()
    if save_every_x_seconds:
        ticker.cancel()
    pbar.close()

    return pd.read_csv(save_path).assign(Response=lambda d: d.Response.apply(_de))
