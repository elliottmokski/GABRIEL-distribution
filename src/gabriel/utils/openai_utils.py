"""
This module reimplements the original GABRIEL `openai_utils.py` for the
OpenAI Responses API with several improvements:

* Rate limit introspection – a helper fetches the current token/request
  budget from the ``x‑ratelimit-*`` response headers returned by a cheap
  ``GET /v1/models`` call.  These values are used to display how many
  tokens and requests remain per minute.
* User‑friendly summary – before a long job starts, the module prints a
  summary showing the number of prompts, input words, remaining rate‑limit
  capacity, usage tier qualifications, and an estimated cost.  It also
  explains the purpose of the ``max_output_tokens`` parameter.
* Dynamic ``max_output_tokens`` – when a user does not specify
  ``max_output_tokens`` explicitly, the library inspects the current
  token quota.  If fewer than one million tokens remain in the minute
  budget, a safety cutoff of 2 500 tokens is applied; otherwise, the
  parameter is left ``None`` so the model’s default output limit is used.
  This prevents long responses from being rejected due to an overly high
  token estimate, while removing unnecessary complexity when there is
  ample capacity.
* Improved rate‑limit gating – the token limiter now estimates the worst
  possible output length when the cutoff is unspecified by assuming
  the response could be as long as the input.  This avoids grossly
  underestimating throughput while still honouring the per‑minute token
  budget.
* Exponential backoff with jitter – the retry logic uses a random
  exponential backoff when rate‑limit errors occur, following OpenAI’s
  guidelines for handling 429 responses.

The overall API surface remains compatible with the original file: the
public functions ``get_response`` and ``get_all_responses`` still
exist, but the argument ``max_tokens`` has been renamed to
``max_output_tokens`` to match the Responses API.  A legacy alias
``max_tokens`` is accepted for backward compatibility.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import openai
import statistics

# Try to import requests/httpx for rate‑limit introspection
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# Bring in specific error classes for granular handling
try:
    from openai import RateLimitError, APIError, BadRequestError, AuthenticationError, InvalidRequestError  # type: ignore
except Exception:
    RateLimitError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    InvalidRequestError = Exception  # type: ignore

from gabriel.utils.parsing import safe_json

# single connection pool per process, created lazily
client_async: Optional[openai.AsyncOpenAI] = None

# Default safety cutoff when token capacity is low
DEFAULT_MAX_OUTPUT_TOKENS = 2500

# Usage tiers with qualifications and monthly limits for printing
TIER_INFO = [
    {
        "tier": "Free",
        "qualification": "User must be in an allowed geography",
        "monthly_quota": "$100 / month",
    },
    {"tier": "Tier 1", "qualification": "$5 paid", "monthly_quota": "$100 / month"},
    {
        "tier": "Tier 2",
        "qualification": "$50 paid and 7+ days since first payment",
        "monthly_quota": "$500 / month",
    },
    {
        "tier": "Tier 3",
        "qualification": "$100 paid and 7+ days since first payment",
        "monthly_quota": "$1 000 / month",
    },
    {
        "tier": "Tier 4",
        "qualification": "$250 paid and 14+ days since first payment",
        "monthly_quota": "$5 000 / month",
    },
    {
        "tier": "Tier 5",
        "qualification": "$1 000 paid and 30+ days since first payment",
        "monthly_quota": "$200 000 / month",
    },
]

# Truncated pricing table (USD per million tokens) for a few common models
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # model family       input   cached_input   output   batch_factor
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60, "batch": 0.5},
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40,
        "batch": 0.5,
    },
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00, "batch": 0.5},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "batch": 0.5},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40, "batch": 0.5},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40, "batch": 0.5},
    "o3-deep-research": {
        "input": 10.00,
        "cached_input": 2.50,
        "output": 40.00,
        "batch": 0.5,
    },
    "o4-mini-deep-research": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
        "batch": 0.5,
    },
}


def _print_tier_explainer(verbose: bool = True) -> None:
    """Print a helpful explanation of usage tiers and how to increase them.

    This helper can be called when a user encounters errors that may be
    related to low quotas or tier limitations.  It summarises the
    qualifications for each tier and encourages users to check their
    payment status and billing page.  The message is only printed when
    ``verbose`` is ``True``.
    """
    if not verbose:
        return
    print("\n===== Tier explainer =====")
    print(
        "Your organization’s ability to call the OpenAI API is governed by usage tiers."
    )
    print(
        "As you spend more on the API, you are automatically graduated to higher tiers with larger token and request limits."
    )
    print("Here are the current tiers and how to qualify:")
    for tier in TIER_INFO:
        print(
            f"  • {tier['tier']}: qualify by {tier['qualification']}; monthly quota {tier['monthly_quota']}"
        )
    print("If you are encountering rate limits or truncated outputs, consider:")
    print(
        "  – Checking your current spend and ensuring you have met the payment criteria for a higher tier."
    )
    print(
        "  – Adding funds or updating billing details at https://platform.openai.com/settings/organization/billing/."
    )
    print("  – Reducing the number of parallel requests or batching your workload.")


def _approx_tokens(text: str) -> int:
    """Roughly estimate the token count from a string by assuming ~1.5 tokens per word."""
    return int(len(str(text).split()) * 1.5)


def _lookup_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """Find a pricing entry for ``model`` by prefix match (case‑insensitive)."""
    key = model.lower()
    for prefix, pricing in MODEL_PRICING.items():
        if key.startswith(prefix):
            return pricing
    return None


def _estimate_cost(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
) -> Optional[Dict[str, float]]:
    """Estimate input/output tokens and cost for a set of prompts.

    Returns a dict with keys ``input_tokens``, ``output_tokens``, ``input_cost``, ``output_cost``, and ``total_cost``.
    If the model pricing is unavailable, returns ``None``.
    """
    pricing = _lookup_model_pricing(model)
    if pricing is None:
        return None
    # Estimate tokens: input tokens are sum of tokens per prompt times number of responses
    input_tokens = sum(_approx_tokens(p) for p in prompts) * max(1, n)
    # If no cutoff, conservatively assume output tokens equal input tokens
    if max_output_tokens is None:
        output_tokens = input_tokens
    else:
        output_tokens = max_output_tokens * max(1, n) * len(prompts)
    cost_in = (input_tokens / 1_000_000) * pricing["input"]
    cost_out = (output_tokens / 1_000_000) * pricing["output"]
    if use_batch:
        cost_in *= pricing["batch"]
        cost_out *= pricing["batch"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": cost_in,
        "output_cost": cost_out,
        "total_cost": cost_in + cost_out,
    }


def _require_api_key() -> str:
    """Return the API key or raise a runtime error if missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable must be set or passed via OpenAIClient(api_key)."
        )
    return api_key


def _get_rate_limit_headers() -> Optional[Dict[str, str]]:
    """Retrieve rate‑limit headers via a cheap API request.

    Performs a ``GET /v1/models`` request (which does not consume tokens) using
    whatever HTTP client is available.  Returns a dict of the relevant
    ``x‑ratelimit-*`` headers or ``None`` if the request fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    # Try requests then httpx
    for client in (requests, httpx):
        if client is None:
            continue
        try:
            resp = client.get(url, headers=headers, timeout=10)  # type: ignore
            # Some libraries store headers in different attributes (dict or case‑insensitive)
            h = getattr(resp, "headers", {})  # type: ignore
            # Normalize keys to lower case
            new_h = {k.lower(): v for k, v in h.items()}
            return {
                "limit_requests": new_h.get("x-ratelimit-limit-requests"),
                "remaining_requests": new_h.get("x-ratelimit-remaining-requests"),
                "reset_requests": new_h.get("x-ratelimit-reset-requests"),
                "limit_tokens": new_h.get("x-ratelimit-limit-tokens"),
                "remaining_tokens": new_h.get("x-ratelimit-remaining-tokens"),
                "reset_tokens": new_h.get("x-ratelimit-reset-tokens"),
            }
        except Exception:
            continue
    return None


def _print_usage_overview(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
    requests_per_minute: int,
    tokens_per_minute: int,
    rate_limit_factor: float,
    *,
    verbose: bool = True,
    rate_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Print a summary of usage limits, cost estimate and tier information.

    Optionally takes a pre‑fetched ``rate_headers`` dict to avoid calling
    ``_get_rate_limit_headers`` multiple times per job.  When ``rate_headers``
    is ``None``, the helper will fetch the headers itself.
    """
    if not verbose:
        return
    print("\n===== OpenAI API usage summary =====")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Total input words: {sum(len(str(p).split()) for p in prompts):,}")
    rl = rate_headers or _get_rate_limit_headers()
    if rl:
        try:
            lim_r = int(float(rl.get("limit_requests") or 0))
            rem_r = int(float(rl.get("remaining_requests") or 0))
            reset_r = rl.get("reset_requests")
            print(
                f"Requests per minute: {lim_r:,} (remaining {rem_r:,}); resets in {reset_r}"
            )
            print(f"Approx. requests per day: {lim_r * 60 * 24:,}")
        except Exception:
            pass
        try:
            lim_t = int(float(rl.get("limit_tokens") or 0))
            rem_t = int(float(rl.get("remaining_tokens") or 0))
            reset_t = rl.get("reset_tokens")
            print(
                f"Tokens per minute: {lim_t:,} (remaining {rem_t:,}); resets in {reset_t}"
            )
            words_per_min = lim_t // 2
            words_per_day = lim_t * 60 * 24 // 2
            print(
                f"≈ {words_per_min:,} words per minute, ≈ {words_per_day:,} words per day"
            )
        except Exception:
            pass
    else:
        eff_rpm = int(requests_per_minute * rate_limit_factor)
        eff_tpm = int(tokens_per_minute * rate_limit_factor)
        print(f"Requests per minute: {eff_rpm:,} (effective)")
        print(
            f"Tokens per minute: {eff_tpm:,} (≈ {eff_tpm // 2:,} words); per day ≈ {(eff_tpm * 60 * 24) // 2:,} words"
        )
    print("\nUsage tiers:")
    for tier in TIER_INFO:
        print(
            f"  • {tier['tier']}: qualifies by {tier['qualification']}; monthly quota {tier['monthly_quota']}"
        )
    pricing = _lookup_model_pricing(model)
    est = _estimate_cost(prompts, n, max_output_tokens, model, use_batch)
    if pricing and est:
        print(
            f"\nPricing for model '{model}': input ${pricing['input']}/1M, output ${pricing['output']}/1M"
        )
        if use_batch:
            print("Batch API prices are half the synchronous rates.")
        print(
            f"Estimated token usage: input {est['input_tokens']:,}, output {est['output_tokens']:,}"
        )
        print(
            f"Estimated {'batch' if use_batch else 'synchronous'} cost: ${est['total_cost']:.4f}"
        )
    else:
        print(f"\nPricing for model '{model}' is unavailable; cannot estimate cost.")
    print(
        "\nAdd funds or manage your billing here: https://platform.openai.com/settings/organization/billing/"
    )
    if max_output_tokens is None:
        print(
            "\nmax_output_tokens: None (using model default – note this does not cap total tokens)"
        )
    else:
        print(
            f"\nmax_output_tokens: {max_output_tokens} (safety cutoff; generation will stop if this is reached)"
        )


def _decide_default_max_output_tokens(
    user_specified: Optional[int], rate_headers: Optional[Dict[str, str]] = None
) -> Optional[int]:
    """Decide a default ``max_output_tokens`` based on current token budget.

    If ``user_specified`` is not ``None``, return it unchanged.  Otherwise,
    use the supplied ``rate_headers`` dict (or fetch one if ``None``) to
    determine how many tokens remain in the per‑minute budget.  If fewer than
    one million tokens remain, return ``DEFAULT_MAX_OUTPUT_TOKENS``; else
    return ``None`` to indicate no cutoff.
    """
    if user_specified is not None:
        return user_specified
    rl = rate_headers or _get_rate_limit_headers()
    if rl and rl.get("remaining_tokens"):
        try:
            rem = int(float(rl["remaining_tokens"]))
            if rem < 1_000_000:
                return DEFAULT_MAX_OUTPUT_TOKENS
        except Exception:
            pass
    return None


def _build_params(
    *,
    model: str,
    input_data: List[Dict[str, Any]],
    max_output_tokens: Optional[int],
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
    """Build the parameter dict for ``client.responses.create``.

    The ``max_output_tokens`` key is only included when a non‑None value is
    provided; otherwise, the API will use its own maximum.
    """
    params: Dict[str, Any] = {
        "model": model,
        "input": input_data,
        "truncation": "auto",
    }
    if max_output_tokens is not None:
        params["max_output_tokens"] = max_output_tokens
    if json_mode:
        params["text"] = (
            {"format": {"type": "json_schema", "schema": expected_schema}}
            if expected_schema
            else {"format": {"type": "json_object"}}
        )
    all_tools = list(tools) if tools else []
    if web_search:
        all_tools.append(
            {"type": "web_search_preview", "search_context_size": search_context_size}
        )
    if all_tools:
        params["tools"] = all_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    # For o‑series models, reasoning_effort controls hidden reasoning tokens; for others, temperature
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
    max_output_tokens: Optional[int] = None,
    # legacy alias for backwards compatibility
    max_tokens: Optional[int] = None,
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
    images: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[List[str], float]:
    """Minimal async call to OpenAI's /responses endpoint or dummy response.

    The caller may specify either ``max_output_tokens`` or the deprecated
    ``max_tokens`` argument.  If both are provided, ``max_output_tokens``
    takes precedence.  When both are ``None``, the model’s default output
    limit is used.
    """
    # Use dummy for testing without calling the API
    if use_dummy:
        return [f"DUMMY {prompt}" for _ in range(max(n, 1))], 0.0
    _require_api_key()
    # Derive the effective cutoff
    cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    # Build system message only for non‑o series
    system_instruction = (
        "Please provide a helpful response to this inquiry for purposes of academic research."
    )
    if images:
        contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for img in images:
            img_url = img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
            contents.append({"type": "input_image", "image_url": img_url})
        input_data = (
            [{"role": "user", "content": contents}]
            if model.startswith("o")
            else [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": contents},
            ]
        )
    else:
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
        max_output_tokens=cutoff,
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
    # Create parallel tasks for `n` completions
    tasks = [
        client_async.responses.create(**params, timeout=timeout)
        for _ in range(max(n, 1))
    ]
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
    # Extract ``output_text`` from the responses.  For Responses API
    # the SDK returns an object with an ``output_text`` attribute.
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


async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    *,
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    # legacy alias
    max_tokens: Optional[int] = None,
    timeout: float = 90.0,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    use_web_search: bool = False,
    search_context_size: str = "medium",
    reasoning_effort: str = "medium",
    use_dummy: bool = False,
    print_example_prompt: bool = True,
    save_path: str = "responses.csv",
    reset_files: bool = False,
    n_parallels: int = 5,
    max_retries: int = 5,
    timeout_factor: float = 1.5,
    max_timeout: int = 300,
    dynamic_timeout: bool = True,
    requests_per_minute: int = 300,
    tokens_per_minute: int = 150_000,
    dynamic_rate_limit: bool = True,
    rate_limit_factor: float = 1.0,
    rate_limit_adjust_factor: float = 0.7,
    cancel_existing_batch: bool = False,
    use_batch: bool = False,
    batch_completion_window: str = "24h",
    batch_poll_interval: int = 10,
    batch_wait_for_completion: bool = False,
    max_batch_requests: int = 50_000,
    max_batch_file_bytes: int = 100 * 1024 * 1024,
    save_every_x_responses: int = 25,
    verbose: bool = True,
    **get_response_kwargs: Any,
) -> pd.DataFrame:
    """Retrieve responses for a list of prompts, with optional batch support.

    This function handles rate limiting, optional batch submission, dynamic
    timeout adjustment and printing of helpful usage summaries.  It is
    backwards compatible with the original version, except that the
    parameter ``max_tokens`` has been renamed to ``max_output_tokens``.
    When both are provided, ``max_output_tokens`` takes precedence.
    """
    if not use_dummy:
        _require_api_key()
    # Backwards compatibility for identifiers
    if identifiers is None:
        identifiers = prompts
    # Pull default values into kwargs for get_response
    get_response_kwargs.setdefault("web_search", use_web_search)
    get_response_kwargs.setdefault("search_context_size", search_context_size)
    get_response_kwargs.setdefault("tools", tools)
    get_response_kwargs.setdefault("tool_choice", tool_choice)
    get_response_kwargs.setdefault("json_mode", json_mode)
    get_response_kwargs.setdefault("expected_schema", expected_schema)
    get_response_kwargs.setdefault("temperature", temperature)
    get_response_kwargs.setdefault("reasoning_effort", reasoning_effort)
    # Decide default cutoff once per job using cached rate headers
    # Fetch rate headers once to avoid multiple API calls
    rate_headers = _get_rate_limit_headers()
    user_cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    cutoff = _decide_default_max_output_tokens(user_cutoff, rate_headers)
    get_response_kwargs.setdefault("max_output_tokens", cutoff)
    # Always load or initialise the CSV
    if os.path.exists(save_path) and not reset_files:
        df = pd.read_csv(save_path)
        df["Response"] = df["Response"].apply(_de)
        done = set(df["Identifier"])
    else:
        df = pd.DataFrame(columns=["Identifier", "Response", "Time Taken"])
        done = set()
    # Filter prompts/identifiers based on what is already completed
    todo_pairs = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    if not todo_pairs:
        return df
    # Print usage summary and example prompt
    if print_example_prompt and todo_pairs:
        # Build prompt list for cost estimate
        prompt_list = [p for p, _ in todo_pairs]
        _print_usage_overview(
            prompts=prompt_list,
            n=n,
            max_output_tokens=cutoff,
            model=get_response_kwargs.get("model", "o4-mini"),
            use_batch=use_batch,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            rate_limit_factor=rate_limit_factor,
            verbose=verbose,
            rate_headers=rate_headers,
        )
        example_prompt, _ = todo_pairs[0]
        print(f"\nExample prompt: {example_prompt}\n")
    # Batch submission path
    if use_batch:
        state_path = save_path + ".batch_state.json"

        # Helper to append batch rows
        def _append_results(rows: List[Dict[str, Any]]) -> None:
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
        # Load existing state
        if os.path.exists(state_path) and not reset_files:
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {}
        # Convert single batch format
        if state.get("batch_id"):
            state = {
                "batches": [
                    {
                        "batch_id": state["batch_id"],
                        "input_file_id": state.get("input_file_id"),
                        "total": None,
                        "submitted_at": None,
                    }
                ]
            }
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
        # If there are no unfinished batches, create new ones
        if not state.get("batches"):
            tasks: List[Dict[str, Any]] = []
            for prompt, ident in todo_pairs:
                imgs = prompt_images.get(str(ident)) if prompt_images else None
                if imgs:
                    contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
                    for img in imgs:
                        img_url = img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                        contents.append({"type": "input_image", "image_url": img_url})
                    input_data = (
                        [{"role": "user", "content": contents}]
                        if get_response_kwargs.get("model", "o4-mini").startswith("o")
                        else [
                            {"role": "system", "content": "Please provide a helpful response to this inquiry for purposes of academic research."},
                            {"role": "user", "content": contents},
                        ]
                    )
                else:
                    input_data = (
                        [{"role": "user", "content": prompt}]
                        if get_response_kwargs.get("model", "o4-mini").startswith("o")
                        else [
                            {"role": "system", "content": "Please provide a helpful response to this inquiry for purposes of academic research."},
                            {"role": "user", "content": prompt},
                        ]
                    )
                body = _build_params(
                    model=get_response_kwargs.get("model", "o4-mini"),
                    input_data=input_data,
                    max_output_tokens=cutoff,
                    system_instruction="Please provide a helpful response to this inquiry for purposes of academic research.",
                    temperature=get_response_kwargs.get("temperature", 0.9),
                    tools=get_response_kwargs.get("tools"),
                    tool_choice=get_response_kwargs.get("tool_choice"),
                    web_search=get_response_kwargs.get("web_search", False),
                    search_context_size=get_response_kwargs.get(
                        "search_context_size", "medium"
                    ),
                    json_mode=get_response_kwargs.get("json_mode", False),
                    expected_schema=get_response_kwargs.get("expected_schema"),
                    reasoning_effort=get_response_kwargs.get(
                        "reasoning_effort", "medium"
                    ),
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
                batches: List[List[Dict[str, Any]]] = []
                current_batch: List[Dict[str, Any]] = []
                current_size = 0
                for obj in tasks:
                    line_bytes = (
                        len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) + 1
                    )
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
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jsonl"
                    ) as tmp:
                        for obj in batch_tasks:
                            tmp.write(json.dumps(obj).encode("utf-8") + b"\n")
                        input_filename = tmp.name
                    uploaded = await client.files.create(
                        file=open(input_filename, "rb"), purpose="batch"
                    )
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
                        print(
                            f"Submitted batch {batch.id} with {len(batch_tasks)} requests."
                        )
                with open(state_path, "w") as f:
                    json.dump(state, f)
        # Return immediately if not waiting for completion
        if not batch_wait_for_completion:
            return df
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
                    try:
                        file_response = await client.files.content(output_file_id)
                    except Exception as exc:
                        if verbose:
                            print(
                                f"Failed to download output file for batch {bid}: {exc}"
                            )
                        unfinished_batches.remove(b)
                        continue
                    # Normalize file response to plain text
                    text_data: Optional[str] = None
                    try:
                        if isinstance(file_response, str):
                            text_data = file_response
                        elif isinstance(file_response, bytes):
                            text_data = file_response.decode("utf-8", errors="replace")
                        elif hasattr(file_response, "text"):
                            attr = getattr(file_response, "text")
                            text_data = await attr() if callable(attr) else attr  # type: ignore
                        if text_data is None and hasattr(file_response, "read"):
                            content_bytes = await file_response.read()  # type: ignore
                            text_data = (
                                content_bytes.decode("utf-8", errors="replace")
                                if isinstance(content_bytes, bytes)
                                else str(content_bytes)
                            )
                    except Exception:
                        pass
                    if text_data is None:
                        if verbose:
                            print(f"No data found in output file for batch {bid}.")
                        unfinished_batches.remove(b)
                        continue
                    errors: Dict[str, Any] = {}
                    if error_file_id:
                        try:
                            err_response = await client.files.content(error_file_id)
                        except Exception as exc:
                            if verbose:
                                print(
                                    f"Failed to download error file for batch {bid}: {exc}"
                                )
                            err_response = None
                        if err_response is not None:
                            err_text: Optional[str] = None
                            try:
                                if isinstance(err_response, str):
                                    err_text = err_response
                                elif isinstance(err_response, bytes):
                                    err_text = err_response.decode(
                                        "utf-8", errors="replace"
                                    )
                                elif hasattr(err_response, "text"):
                                    attr = getattr(err_response, "text")
                                    err_text = await attr() if callable(attr) else attr  # type: ignore
                                if err_text is None and hasattr(err_response, "read"):
                                    content_bytes = await err_response.read()  # type: ignore
                                    err_text = (
                                        content_bytes.decode("utf-8", errors="replace")
                                        if isinstance(content_bytes, bytes)
                                        else str(content_bytes)
                                    )
                            except Exception:
                                err_text = None
                            if err_text:
                                for line in err_text.splitlines():
                                    try:
                                        rec = json.loads(line)
                                        errors[rec.get("custom_id")] = rec.get("error")
                                    except Exception:
                                        pass
                    for line in text_data.splitlines():
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ident = rec.get("custom_id")
                        if not ident:
                            continue
                        if rec.get("response") is None:
                            err = rec.get("error") or errors.get(ident)
                            completed_rows.append(
                                {
                                    "Identifier": ident,
                                    "Response": None,
                                    "Time Taken": None,
                                    "Error": err,
                                }
                            )
                            continue
                        resp_obj = rec["response"]
                        resp_text: Optional[str] = None
                        # Determine candidate payload
                        candidate = (
                            resp_obj.get("body", resp_obj)
                            if isinstance(resp_obj, dict)
                            else None
                        )
                        search_objs: List[Dict[str, Any]] = []
                        if isinstance(candidate, dict):
                            search_objs.append(candidate)
                        if isinstance(resp_obj, dict):
                            search_objs.append(resp_obj)
                        for obj in search_objs:
                            if resp_text is None and isinstance(
                                obj.get("output_text"), (str, bytes)
                            ):
                                resp_text = obj["output_text"]
                                break
                            if resp_text is None and isinstance(
                                obj.get("choices"), list
                            ):
                                choices = obj.get("choices")
                                if choices:
                                    choice = choices[0]
                                    if isinstance(choice, dict):
                                        msg = (
                                            choice.get("message")
                                            or choice.get("delta")
                                            or {}
                                        )
                                        if isinstance(msg, dict):
                                            content = msg.get("content")
                                            if isinstance(content, str):
                                                resp_text = content
                                                break
                            if resp_text is None and isinstance(
                                obj.get("output"), list
                            ):
                                out_list = obj.get("output")
                                for item in out_list:
                                    if not isinstance(item, dict):
                                        continue
                                    content_list = item.get("content")
                                    if isinstance(content_list, list):
                                        for piece in content_list:
                                            if (
                                                isinstance(piece, dict)
                                                and "text" in piece
                                            ):
                                                txt = piece.get("text")
                                                if isinstance(txt, str):
                                                    resp_text = txt
                                                    break
                                        if resp_text is not None:
                                            break
                                    if resp_text is None and isinstance(
                                        item.get("text"), str
                                    ):
                                        resp_text = item["text"]
                                        break
                                    if resp_text is not None:
                                        break
                                if resp_text is not None:
                                    break
                        completed_rows.append(
                            {
                                "Identifier": ident,
                                "Response": [resp_text],
                                "Time Taken": None,
                            }
                        )
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                elif status in {"failed", "cancelled", "expired"}:
                    if verbose:
                        print(f"Batch {bid} finished with status {status}.")
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                else:
                    if verbose:
                        rc = job.request_counts
                        print(
                            f"Batch {bid} in progress: {status}; completed {rc.completed}/{rc.total}."
                        )
            if unfinished_batches:
                await asyncio.sleep(batch_poll_interval)
        # Append and return
        _append_results(completed_rows)
        return df
    # Non‑batch path
    # Initialise limiters; will be rebuilt if dynamic_rate_limit kicks in
    nonlocal_timeout = timeout
    current_rate_limit_factor = rate_limit_factor
    req_lim = AsyncLimiter(int(requests_per_minute * current_rate_limit_factor), 60)
    tok_lim = AsyncLimiter(int(tokens_per_minute * current_rate_limit_factor), 60)
    response_times: List[float] = []
    timeout_errors = 0
    call_count = 0
    min_samples_for_timeout = max(100, n_parallels)
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
        if len(response_times) < min_samples_for_timeout:
            return
        try:
            sorted_times = sorted(response_times)
            q95_index = max(0, int(0.95 * (len(sorted_times) - 1)))
            q95 = sorted_times[q95_index]
            new_timeout = min(max_timeout, max(timeout, timeout_factor * q95))
            if (
                new_timeout > nonlocal_timeout * 1.2
                or new_timeout < nonlocal_timeout * 0.8
            ):
                if verbose:
                    print(
                        f"[dynamic timeout] Updating timeout from {nonlocal_timeout:.1f}s to {new_timeout:.1f}s based on observed latency."
                    )
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
                f"[dynamic rate-limit] Adjusted rate_limit_factor to {current_rate_limit_factor:.2f}. New RPM limit: {int(requests_per_minute * current_rate_limit_factor)}, TPM limit: {int(tokens_per_minute * current_rate_limit_factor)}."
            )

    async def worker() -> None:
        nonlocal processed, timeout_errors, call_count, current_rate_limit_factor, nonlocal_timeout
        while True:
            try:
                prompt, ident = await queue.get()
            except asyncio.CancelledError:
                break
            try:
                attempt = 1
                while attempt <= max_retries:
                    try:
                        approx = _approx_tokens(prompt)
                        # Estimate tokens for gating: assume output could be as long as input when cutoff is None
                        gating_output = cutoff if cutoff is not None else approx
                        await req_lim.acquire()
                        await tok_lim.acquire((approx + gating_output) * n)
                        call_count += 1
                        resps, t = await asyncio.wait_for(
                            get_response(
                                prompt,
                                n=n,
                                timeout=nonlocal_timeout,
                                use_dummy=use_dummy,
                                verbose=verbose,
                                **get_response_kwargs,
                            ),
                            timeout=nonlocal_timeout,
                            use_dummy=use_dummy,
                            verbose=verbose,
                            images=prompt_images.get(str(ident)) if prompt_images else None,
                            **get_response_kwargs,
                        ),
                        timeout=nonlocal_timeout,
                    )
                    response_times.append(t)
                    await adjust_timeout()
                    # Check for empty outputs.  If all returned strings are empty or whitespace,
                    # notify the user that the safety cutoff or tier limits may have truncated the output.
                    if resps and all((isinstance(r, str) and not r.strip()) for r in resps):
                        if verbose:
                            print(
                                f"[get_all_responses] Timeout on attempt {attempt} for {ident} after {nonlocal_timeout:.1f}s. Consider increasing the 'timeout' parameter if timeouts persist."
                            )
                        if (
                            dynamic_timeout
                            and call_count > 0
                            and timeout_errors / call_count > 0.05
                        ):
                            if len(response_times) >= min_samples_for_timeout:
                                try:
                                    sorted_times = sorted(response_times)
                                    q95_index = max(0, int(0.95 * (len(sorted_times) - 1)))
                                    q95 = sorted_times[q95_index]
                                    new_t = min(
                                        max_timeout,
                                        max(nonlocal_timeout, timeout_factor * q95),
                                    )
                                except Exception:
                                    new_t = min(
                                        max_timeout, nonlocal_timeout * timeout_factor
                                    )
                            else:
                                new_t = min(max_timeout, nonlocal_timeout * timeout_factor)
                            if new_t > nonlocal_timeout:
                                if verbose:
                                    print(
                                        f"[dynamic timeout] Increasing timeout to {new_t:.1f}s due to high timeout rate."
                                    )
                                nonlocal_timeout = new_t
                        if attempt >= max_retries:
                            results.append(
                                {"Identifier": ident, "Response": None, "Time Taken": None}
                            )
                            processed += 1
                            pbar.update(1)
                            await flush()
                            break
                        # Exponential backoff with jitter
                        await asyncio.sleep(random.uniform(1, 2) * (2 ** (attempt - 1)))
                        attempt += 1
                    except RateLimitError as e:
                        if verbose:
                            print(
                                f"[get_all_responses] Rate limit error on attempt {attempt} for {ident}: {e}"
                            )
                        if dynamic_rate_limit:
                            current_rate_limit_factor *= rate_limit_adjust_factor
                            await rebuild_limiters()
                        if attempt >= max_retries:
                            results.append(
                                {"Identifier": ident, "Response": None, "Time Taken": None}
                            )
                            processed += 1
                            pbar.update(1)
                            await flush()
                            break
                        await asyncio.sleep(random.uniform(1, 2) * (2 ** (attempt - 1)))
                        attempt += 1
                    except (
                        APIError,
                        BadRequestError,
                        AuthenticationError,
                        InvalidRequestError,
                    ) as e:
                        if verbose:
                            print(f"[get_all_responses] API error for {ident}: {e}")
                        results.append(
                            {"Identifier": ident, "Response": None, "Time Taken": None}
                        )
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
                    except Exception as e:
                        if verbose:
                            print(f"[get_all_responses] Unexpected error for {ident}: {e}")
                        results.append(
                            {"Identifier": ident, "Response": None, "Time Taken": None}
                        )
                        processed += 1
                        pbar.update(1)
                        await flush()
                        break
            finally:
                queue.task_done()

    # Spawn workers
    workers = [asyncio.create_task(worker()) for _ in range(n_parallels)]
    await queue.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    await flush()
    return df
