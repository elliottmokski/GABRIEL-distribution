from __future__ import annotations

import asyncio
import csv
import json
import os
import time
import tempfile
import statistics
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import openai
try:
    from openai import APIError, AuthenticationError, BadRequestError, InvalidRequestError, RateLimitError
except Exception:  # pragma: no cover - fallback for older openai versions
    RateLimitError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    InvalidRequestError = Exception  # type: ignore
from .parsing import safe_json

# single connection pool per process, created lazily
client_async: Optional[openai.AsyncOpenAI] = None


def _build_params(
    *,
    model: str,
    input_data: List[Dict[str, str]],
    max_tokens: int,
    system_instruction: str,
    temperature: float,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    search_context_size: str = "medium",
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "medium",
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

    all_tools = list(tools) if tools else []
    if web_search:
        all_tools.append({"type": "web_search_preview", "search_context_size": search_context_size})
    if all_tools:
        params["tools"] = all_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice

    if model.startswith("o"):
        params["reasoning"] = {"effort": reasoning_effort}
    else:
        params["temperature"] = temperature

    params.update(extra)
    return params

async def get_response(
    prompt: str,
    *,
    model: str = "o4-mini",
    n: int = 1,
    max_tokens: int = 25_000,
    timeout: float = 90.0,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    search_context_size: str = "medium",
    reasoning_effort: str = "medium",
    use_dummy: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Tuple[List[str], float]:
    """Minimal async call to OpenAI's /responses endpoint or dummy response.

    Parameters
    ----------
    verbose : bool
        If ``True``, print any API errors encountered before retrying.
    """
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
        tools=tools,
        tool_choice=tool_choice,
        web_search=web_search,
        search_context_size=search_context_size,
        json_mode=json_mode,
        expected_schema=expected_schema,
        reasoning_effort=reasoning_effort,
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
        err = Exception(f"API call timed out after {timeout} s")
        if verbose:
            print(f"[get_response] {err}")
        raise err
    except Exception as e:
        err = Exception(f"API call resulted in exception: {e!r}")
        if verbose:
            print(f"[get_response] {err}")
        raise err

    return [r.output_text for r in raw], time.time() - start


def _ser(x: Any) -> Optional[str]:
    """Serialize Python objects deterministically."""
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _de(x: Any) -> Any:
    """Deserialize JSON strings back to Python objects."""
    if pd.isna(x):
        return None
    parsed = safe_json(x)
    return parsed if parsed else None

"""
This module contains a reimplementation of the `get_all_responses` function from
GABRIEL's `openai_utils.py` with optional Batch API support.  The aim is to
preserve backwards‑compatible behaviour for synchronous parallel execution while
adding a mechanism to submit large workloads via the Batch API and resume
processing later if a notebook times out.

Key differences from the original implementation:

* A new ``use_batch`` flag determines whether to use the Batch API.  When
  ``use_batch=False`` (default), the function behaves identically to the
  original version, spawning asynchronous workers that call ``get_response`` in
  parallel with rate limiting.

* When ``use_batch=True``, requests are collected into a single JSONL file and
  submitted to OpenAI’s Batch API.  The function can either wait for the job
  to complete or return immediately, allowing a long‑running batch to process
  in the background.  The job state (input file ID and batch ID) is saved to
  disk so that subsequent calls can resume polling and download the results
  once available.

* A small metadata file named ``<save_path>.batch_state.json`` stores the
  outstanding batch ID.  If it exists, the function will attempt to retrieve
  and finalise the results rather than submitting a new batch.  Once results
  are downloaded and merged into the existing CSV, the metadata file is
  removed.

This implementation uses ``openai.AsyncOpenAI`` for non‑blocking I/O when
interacting with the Batch API.  See the inline documentation for more
details.
"""

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
    use_web_search: bool = False,
    search_context_size: str = "medium",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    verbose: bool = True,
    use_batch: bool = False,
    batch_completion_window: str = "24h",
    batch_poll_interval: int = 10,
    batch_wait_for_completion: bool = True,
    dynamic_timeout: bool = True,
    timeout_factor: float = 1.5,
    max_timeout: int = 300,
    dynamic_rate_limit: bool = True,
    rate_limit_adjust_factor: float = 0.9,
    cancel_existing_batch: bool = False,
    max_batch_requests: int = 50_000,
    max_batch_file_bytes: int = 200_000_000,
    **get_response_kwargs: Any,
) -> pd.DataFrame:
    """Query an LLM for multiple prompts, optionally via the Batch API.

    Parameters
    ----------
    prompts : List[str]
        Prompts to send to the model.
    identifiers : Optional[List[str]], optional
        Unique identifiers for each prompt.  Defaults to the prompt itself.
    n_parallels : int, optional
        Number of concurrent workers to spawn when not using batch.
    save_path : str, optional
        Path to a CSV file where results are incrementally written.  A
        corresponding ``<save_path>.batch_state.json`` file is used to track
        outstanding batch jobs.
    reset_files : bool, optional
        If ``True``, ignore any existing CSV or batch state and start fresh.
    n : int, optional
        Number of completions to request per prompt.
    max_tokens : int, optional
        Maximum number of output tokens for each completion.
    requests_per_minute : int, optional
        Global request rate limit used by the non‑batch path.
    tokens_per_minute : int, optional
        Global token rate limit used by the non‑batch path.
    rate_limit_factor : float, optional
        Fraction of the rate limits to actually use (adds headroom for retries).
    timeout : int, optional
        Timeout in seconds for each individual API call.
    max_retries : int, optional
        Maximum number of retries per prompt when not using batch.
    save_every_x_responses : int, optional
        Flush accumulated responses to disk every this many responses in
        non‑batch mode.
    save_every_x_seconds : Optional[int], optional
        Flush accumulated responses to disk every this many seconds in
        non‑batch mode.  If ``None``, no time‑based flushing is performed.
    use_dummy : bool, optional
        If ``True``, return dummy responses instead of calling the API.
    print_example_prompt : bool, optional
        If ``True``, print the first prompt before processing.
    use_web_search : bool, optional
        Whether to include the web_search tool when building the request.
    search_context_size : str, optional
        Context size for the web_search tool.
    tools : Optional[List[dict]], optional
        Additional tools to include in the request body.
    tool_choice : Optional[dict], optional
        Tool choice override.
    verbose : bool, optional
        If ``True``, print progress and error messages.
    use_batch : bool, optional
        If ``True``, submit requests via the Batch API rather than
        synchronously.
    batch_completion_window : str, optional
        Completion window to pass to ``client.batches.create``.  Only "24h"
        is currently supported by OpenAI.
    batch_poll_interval : int, optional
        Seconds to wait between polling the batch status.
    batch_wait_for_completion : bool, optional
        If ``True``, wait until the batch job finishes before returning.
        If ``False``, submit the batch (or resume polling) but return
        immediately.

    dynamic_timeout : bool, optional
        If ``True``, adjust the timeout dynamically based on observed response
        times.  After at least five successful calls, the timeout will be
        updated to ``timeout_factor`` times the 95th percentile of observed
        response durations, clamped between the initial ``timeout`` and
        ``max_timeout``.

    timeout_factor : float, optional
        Multiplier applied to the 95th percentile of response times when
        computing a new timeout.  Ignored if ``dynamic_timeout`` is
        ``False``.

    max_timeout : int, optional
        Maximum allowed timeout in seconds when using dynamic timeouts.

    dynamic_rate_limit : bool, optional
        If ``True``, automatically reduce the effective rate limit factor
        when a rate‑limit error (HTTP 429) is encountered.  Each time a
        ``RateLimitError`` occurs, the ``rate_limit_factor`` is multiplied
        by ``rate_limit_adjust_factor`` and new limiters are created.

    rate_limit_adjust_factor : float, optional
        Factor by which to multiply ``rate_limit_factor`` when a rate
        limit error is detected.  Should be between 0 and 1.  Ignored if
        ``dynamic_rate_limit`` is ``False``.

    cancel_existing_batch : bool, optional
        If ``True`` and a batch state file exists, cancel the outstanding
        batch(es) before submitting a new job.  Otherwise, existing
        unfinished batches will be resumed and no new batch will be
        submitted.

    max_batch_requests : int, optional
        Maximum number of requests per batch when using the Batch API.
        OpenAI currently limits batches to 50 000 requests; you can lower
        this for testing.

    max_batch_file_bytes : int, optional
        Maximum size of the JSONL input file in bytes when using the Batch
        API.  OpenAI currently allows up to 100 MB per batch file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Identifier``, ``Response``, and
        ``Time Taken``.  In batch mode, ``Time Taken`` is ``None`` because
        individual timings are not available.
    """

    # Assign identifiers if none provided
    if identifiers is None:
        identifiers = prompts

    # Default per-call parameters
    get_response_kwargs.setdefault("web_search", use_web_search)
    get_response_kwargs.setdefault("search_context_size", search_context_size)
    get_response_kwargs.setdefault("tools", tools)
    get_response_kwargs.setdefault("tool_choice", tool_choice)

    # Load or initialise result CSV
    if os.path.exists(save_path) and not reset_files:
        df = pd.read_csv(save_path)
        df["Response"] = df["Response"].apply(_de)
        done = set(df["Identifier"])
    else:
        df = pd.DataFrame(columns=["Identifier", "Response", "Time Taken"])
        done = set()

    # Filter out already processed prompts
    todo_pairs = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    if not todo_pairs:
        return df

    # ------------------------------ BATCH PATH ------------------------------
    if use_batch:
        state_path = save_path + ".batch_state.json"

        def _append_results(rows: List[Dict[str, Any]]) -> None:
            """Serialize results and append to CSV and in-memory DataFrame."""
            nonlocal df
            if not rows:
                return
            batch_df = pd.DataFrame(rows)
            batch_df["Response"] = batch_df["Response"].apply(_ser)
            batch_df.to_csv(
                save_path,
                mode="a",
                header=not os.path.exists(save_path),
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            batch_df["Response"] = batch_df["Response"].apply(_de)
            df = pd.concat([df, batch_df], ignore_index=True)

        client = openai.AsyncOpenAI()

        # Load existing batch state if present
        if os.path.exists(state_path) and not reset_files:
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {}

        # Upgrade old state format
        if state.get("batch_id"):
            state = {"batches": [
                {
                    "batch_id": state["batch_id"],
                    "input_file_id": state.get("input_file_id"),
                    "total": None,
                    "submitted_at": None,
                }
            ]}

        # Cancel unfinished batches if requested
        if cancel_existing_batch and state.get("batches"):
            if verbose:
                print("Cancelling unfinished batch jobs...")
            for b in state["batches"]:
                bid = b.get("batch_id")
                try:
                    await client.batches.cancel(bid)
                    if verbose:
                        print(f"Cancelled batch {bid}.")
                except Exception as exc:
                    if verbose:
                        print(f"Failed to cancel batch {bid}: {exc}")
            try:
                os.remove(state_path)
            except OSError:
                pass
            state = {}

        # If no unfinished batches remain, create new ones
        if not state.get("batches"):
            tasks: List[Dict[str, Any]] = []
            for prompt, ident in todo_pairs:
                # Build per-request parameters
                input_data = (
                    [{"role": "user", "content": prompt}]
                    if get_response_kwargs.get("model", "o4-mini").startswith("o")
                    else [
                        {
                            "role": "system",
                            "content": "Please provide a helpful response to this inquiry for purposes of academic research.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                body = _build_params(
                    model=get_response_kwargs.get("model", "o4-mini"),
                    input_data=input_data,
                    max_tokens=max_tokens,
                    system_instruction="Please provide a helpful response to this inquiry for purposes of academic research.",
                    temperature=get_response_kwargs.get("temperature", 0.9),
                    tools=get_response_kwargs.get("tools"),
                    tool_choice=get_response_kwargs.get("tool_choice"),
                    web_search=get_response_kwargs.get("web_search", False),
                    search_context_size=get_response_kwargs.get("search_context_size", "medium"),
                    json_mode=get_response_kwargs.get("json_mode", False),
                    expected_schema=get_response_kwargs.get("expected_schema"),
                    reasoning_effort=get_response_kwargs.get("reasoning_effort", "medium"),
                )
                tasks.append(
                    {
                        "custom_id": str(ident),
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }
                )

            if tasks:
                # Split into multiple batches if needed
                batches: List[List[Dict[str, Any]]] = []
                current_batch: List[Dict[str, Any]] = []
                current_size = 0
                for obj in tasks:
                    line_bytes = len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) + 1
                    if (
                        len(current_batch) >= max_batch_requests
                        or current_size + line_bytes > max_batch_file_bytes
                    ):
                        if current_batch:
                            batches.append(current_batch)
                        current_batch = []
                        current_size = 0
                    current_batch.append(obj)
                    current_size += line_bytes
                if current_batch:
                    batches.append(current_batch)

                state["batches"] = []
                for batch_tasks in batches:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                        for obj in batch_tasks:
                            tmp.write(json.dumps(obj).encode("utf-8") + b"\n")
                        input_filename = tmp.name
                    uploaded = await client.files.create(file=open(input_filename, "rb"), purpose="batch")
                    batch = await client.batches.create(
                        input_file_id=uploaded.id,
                        endpoint="/v1/responses",
                        completion_window=batch_completion_window,
                    )
                    state["batches"].append(
                        {
                            "batch_id": batch.id,
                            "input_file_id": uploaded.id,
                            "total": len(batch_tasks),
                            "submitted_at": int(time.time()),
                        }
                    )
                    if verbose:
                        print(f"Submitted batch {batch.id} with {len(batch_tasks)} requests.")
                with open(state_path, "w") as f:
                    json.dump(state, f)

        # Return immediately if not waiting for completion
        if not batch_wait_for_completion:
            return df

        # Poll each unfinished batch until completion
        unfinished_batches: List[Dict[str, Any]] = list(state.get("batches", []))
        completed_rows: List[Dict[str, Any]] = []
        while unfinished_batches:
            for b in list(unfinished_batches):
                bid = b.get("batch_id")
                try:
                    job = await client.batches.retrieve(bid)
                except Exception as exc:
                    if verbose:
                        print(f"Failed to retrieve batch {bid}: {exc}")
                    continue
                status = job.status
                if status == "completed":
                    output_file_id = job.output_file_id
                    error_file_id = job.error_file_id
                    if verbose:
                        print(f"Batch {bid} completed. Downloading results...")
                    # Download output file (string, bytes, or response object)
                    try:
                        file_response = await client.files.content(output_file_id)
                    except Exception as exc:
                        if verbose:
                            print(f"Failed to download output file for batch {bid}: {exc}")
                        unfinished_batches.remove(b)
                        continue
                    # Normalise file content to text
                    text_data: Optional[str] = None
                    try:
                        if isinstance(file_response, str):
                            text_data = file_response
                        elif isinstance(file_response, bytes):
                            text_data = file_response.decode("utf-8", errors="replace")
                        elif hasattr(file_response, "text"):
                            attr = getattr(file_response, "text")
                            text_data = await attr() if callable(attr) else attr
                        if text_data is None and hasattr(file_response, "read"):
                            content_bytes = await file_response.read()
                            if isinstance(content_bytes, bytes):
                                text_data = content_bytes.decode("utf-8", errors="replace")
                            else:
                                text_data = str(content_bytes)
                    except Exception as exc:
                        if verbose:
                            print(f"Failed to extract text from output file for batch {bid}: {exc}")
                    if text_data is None:
                        if verbose:
                            print(f"No data found in output file for batch {bid}.")
                        unfinished_batches.remove(b)
                        continue
                    # Download and parse error file if present
                    errors: Dict[str, Any] = {}
                    if error_file_id:
                        try:
                            err_response = await client.files.content(error_file_id)
                        except Exception as exc:
                            if verbose:
                                print(f"Failed to download error file for batch {bid}: {exc}")
                            err_response = None
                        if err_response is not None:
                            err_text: Optional[str] = None
                            try:
                                if isinstance(err_response, str):
                                    err_text = err_response
                                elif isinstance(err_response, bytes):
                                    err_text = err_response.decode("utf-8", errors="replace")
                                elif hasattr(err_response, "text"):
                                    attr = getattr(err_response, "text")
                                    err_text = await attr() if callable(attr) else attr
                                if err_text is None and hasattr(err_response, "read"):
                                    content_bytes = await err_response.read()
                                    if isinstance(content_bytes, bytes):
                                        err_text = content_bytes.decode("utf-8", errors="replace")
                                    else:
                                        err_text = str(content_bytes)
                            except Exception:
                                err_text = None
                            if err_text:
                                for line in err_text.splitlines():
                                    try:
                                        rec = json.loads(line)
                                        errors[rec.get("custom_id")] = rec.get("error")
                                    except Exception:
                                        pass
                    # Parse output file
                    for line in text_data.splitlines():
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ident = rec.get("custom_id")
                        if not ident:
                            continue
                        if rec.get("response"):
                            body = rec["response"].get("body")
                            if isinstance(body, dict) and "output_text" in body:
                                resp_text = body["output_text"]
                            elif isinstance(body, dict) and "choices" in body:
                                choice = body["choices"][0]
                                msg = choice.get("message") or choice.get("delta") or {}
                                resp_text = msg.get("content")
                            else:
                                resp_text = None
                            completed_rows.append({"Identifier": ident, "Response": [resp_text], "Time Taken": None})
                        else:
                            err = rec.get("error") or errors.get(ident)
                            completed_rows.append({"Identifier": ident, "Response": None, "Time Taken": None, "Error": err})
                    # Remove completed batch from state and disk
                    unfinished_batches.remove(b)
                    state["batches"] = [bb for bb in state.get("batches", []) if bb.get("batch_id") != bid]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                elif status in {"failed", "cancelled", "expired"}:
                    if verbose:
                        print(f"Batch {bid} finished with status {status}.")
                    unfinished_batches.remove(b)
                    state["batches"] = [bb for bb in state.get("batches", []) if bb.get("batch_id") != bid]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                else:
                    if verbose:
                        rc = job.request_counts
                        print(f"Batch {bid} in progress: {status}; completed {rc.completed}/{rc.total}.")
            if unfinished_batches:
                await asyncio.sleep(batch_poll_interval)
        # Clean up and append results
        if os.path.exists(state_path):
            os.remove(state_path)
        _append_results(completed_rows)
        return df

    # -------------------------- NON-BATCH PATH --------------------------
    # Print example if requested
    if print_example_prompt:
        print(f"Example prompt: {todo_pairs[0][0]}\n")

    # Dummy mode
    if use_dummy:
        rows = [
            {
                "Identifier": ident,
                "Response": [f"DUMMY {ident}"] * n,
                "Time Taken": 0.0,
            }
            for _, ident in todo_pairs
        ]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(save_path, index=False)
        return df

    # Initialize rate limiters
    nonlocal_timeout = timeout
    current_rate_limit_factor = rate_limit_factor
    req_lim = AsyncLimiter(int(requests_per_minute * current_rate_limit_factor), 60)
    tok_lim = AsyncLimiter(int(tokens_per_minute * current_rate_limit_factor), 60)

    # Dynamic timeout tracking
    response_times: List[float] = []
    timeout_errors = 0
    call_count = 0
    min_samples_for_timeout = max(100, n_parallels)  # threshold for stable estimates

    # Queue for workers
    queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
    for item in todo_pairs:
        queue.put_nowait(item)

    results: List[Dict[str, Any]] = []
    processed = 0
    pbar = tqdm(total=len(todo_pairs), desc="Processing prompts")

    async def flush() -> None:
        nonlocal results, df
        if results:
            batch_df = pd.DataFrame(results)
            batch_df["Response"] = batch_df["Response"].apply(_ser)
            batch_df.to_csv(
                save_path,
                mode="a",
                header=not os.path.exists(save_path),
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            batch_df["Response"] = batch_df["Response"].apply(_de)
            df = pd.concat([df, batch_df], ignore_index=True)
            results = []

    async def adjust_timeout() -> None:
        nonlocal nonlocal_timeout
        if not dynamic_timeout:
            return
        # Only adjust when we have enough samples
        if len(response_times) < min_samples_for_timeout:
            return
        try:
            sorted_times = sorted(response_times)
            q95_index = max(0, int(0.95 * (len(sorted_times) - 1)))
            q95 = sorted_times[q95_index]
            new_timeout = min(max_timeout, max(timeout, timeout_factor * q95))
            if new_timeout > nonlocal_timeout * 1.2 or new_timeout < nonlocal_timeout * 0.8:
                if verbose:
                    print(f"[dynamic timeout] Updating timeout from {nonlocal_timeout:.1f}s to {new_timeout:.1f}s based on observed latency.")
                nonlocal_timeout = new_timeout
        except Exception:
            pass

    async def rebuild_limiters() -> None:
        nonlocal req_lim, tok_lim, current_rate_limit_factor
        current_rate_limit_factor = max(0.1, current_rate_limit_factor)
        req_lim = AsyncLimiter(int(requests_per_minute * current_rate_limit_factor), 60)
        tok_lim = AsyncLimiter(int(tokens_per_minute * current_rate_limit_factor), 60)
        if verbose:
            print(
                f"[dynamic rate-limit] Adjusted rate_limit_factor to {current_rate_limit_factor:.2f}. "
                f"New RPM limit: {int(requests_per_minute * current_rate_limit_factor)}, "
                f"TPM limit: {int(tokens_per_minute * current_rate_limit_factor)}."
            )

    async def worker() -> None:
        nonlocal processed, timeout_errors, call_count, current_rate_limit_factor, nonlocal_timeout
        while True:
            try:
                prompt, ident = await queue.get()
            except asyncio.CancelledError:
                break
            attempt = 1
            while attempt <= max_retries:
                try:
                    # Estimate token usage and acquire rate-limit tokens
                    approx = int(len(prompt.split()) * 1.5)
                    await req_lim.acquire()
                    await tok_lim.acquire((approx + max_tokens) * n)
                    call_count += 1
                    # Make API call with dynamic timeout
                    resps, t = await asyncio.wait_for(
                        get_response(
                            prompt,
                            n=n,
                            max_tokens=max_tokens,
                            timeout=nonlocal_timeout,
                            use_dummy=use_dummy,
                            verbose=verbose,
                            **get_response_kwargs,
                        ),
                        timeout=nonlocal_timeout,
                    )
                    response_times.append(t)
                    await adjust_timeout()
                    results.append({"Identifier": ident, "Response": resps, "Time Taken": t})
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x_responses == 0:
                        await flush()
                    break
                except asyncio.TimeoutError:
                    timeout_errors += 1
                    if verbose:
                        print(f"[get_all_responses] Timeout on attempt {attempt} for {ident} after {nonlocal_timeout:.1f}s. "
                              "Consider increasing the 'timeout' parameter if timeouts persist.")
                    # If more than 5% of calls are timing out, raise the timeout using 95th percentile or fallback
                    if dynamic_timeout and call_count > 0 and timeout_errors / call_count > 0.05:
                        if len(response_times) >= min_samples_for_timeout:
                            try:
                                sorted_times = sorted(response_times)
                                q95_index = max(0, int(0.95 * (len(sorted_times) - 1)))
                                q95 = sorted_times[q95_index]
                                new_t = min(max_timeout, max(nonlocal_timeout, timeout_factor * q95))
                            except Exception:
                                new_t = min(max_timeout, nonlocal_timeout * timeout_factor)
                        else:
                            new_t = min(max_timeout, nonlocal_timeout * timeout_factor)
                        if new_t > nonlocal_timeout:
                            if verbose:
                                print(f"[dynamic timeout] Increasing timeout to {new_t:.1f}s due to high timeout rate.")
                            nonlocal_timeout = new_t
                    if attempt >= max_retries:
                        results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    await asyncio.sleep(5 * attempt)
                    attempt += 1
                except RateLimitError as e:
                    if verbose:
                        print(f"[get_all_responses] Rate limit error on attempt {attempt} for {ident}: {e}")
                    if dynamic_rate_limit:
                        current_rate_limit_factor *= rate_limit_adjust_factor
                        await rebuild_limiters()
                    if attempt >= max_retries:
                        results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    await asyncio.sleep(5 * attempt)
                    attempt += 1
                except (APIError, BadRequestError, AuthenticationError, InvalidRequestError) as e:
                    if verbose:
                        print(f"[get_all_responses] API error for {ident}: {e}")
                    results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                    processed += 1
                    pbar.update(1)
                    await flush()
                    break
                except Exception as e:
                    if verbose:
                        print(f"[get_all_responses] Error on attempt {attempt} for {ident}: {e}")
                    if attempt >= max_retries:
                        results.append({"Identifier": ident, "Response": None, "Time Taken": None})
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    await asyncio.sleep(5 * attempt)
                    attempt += 1
            queue.task_done()

    # Spawn workers
    workers = [asyncio.create_task(worker()) for _ in range(n_parallels)]
    ticker = None
    if save_every_x_seconds:
        async def periodic() -> None:
            while True:
                await asyncio.sleep(save_every_x_seconds)
                await flush()
        ticker = asyncio.create_task(periodic())

    # Wait until all tasks are processed
    await queue.join()
    for w in workers:
        w.cancel()
    await flush()
    if ticker:
        ticker.cancel()
    pbar.close()
    return df
