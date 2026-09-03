"""Async load generator for an LLM endpoint: the online measurement.

The Locust load test (../../classic/load_testing/) reported QPS and end-to-end latency. An
LLM streams tokens, so the metrics that matter are different:

  - TTFT (time-to-first-token): prompt submitted -> first token. Dominated by
    prefill + queueing. The latency a user feels before anything appears.
  - TPOT (time-per-output-token): mean gap between subsequent tokens. Dominated
    by decode throughput.
  - throughput: total output tokens / wall-clock across all requests.

This drives the OpenAI streaming API at bounded concurrency and reports those,
with p50/p99 on TTFT. Use it to validate autoscaling and the engine knobs
against real numbers.

Importable from a notebook:
    from load_test import load_test
    stats = await load_test("http://localhost:8000/v1", "qwen-0.5b", total_requests=200)

CLI (after `cd ../serving && serve run app:build_app`):
    python load_test.py --total-requests 200 --concurrency 16
"""

import argparse
import asyncio
import time

import numpy as np
from openai import AsyncOpenAI

DEFAULT_PROMPT = "Explain the benefit of continuous batching in two sentences."


async def _one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed completion; measure TTFT, total time, and output tokens."""
    start = time.perf_counter()
    ttft = None
    tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start   # first token landed
            tokens += 1
    total = time.perf_counter() - start
    return {"ttft": ttft or total, "total": total, "tokens": tokens}


async def load_test(
    base_url: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    total_requests: int = 200,
    concurrency: int = 16,
    max_tokens: int = 128,
    api_key: str = "fake-key",
) -> dict:
    """Drive `total_requests` streamed completions, `concurrency` in flight.

    Returns {ok, errors, elapsed_s, throughput_tok_s, ttft_p50_ms, ttft_p99_ms,
    tpot_ms_mean}. TTFT tail is what a user perceives; throughput is the fleet's
    token rate.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    ttfts_ms: list[float] = []
    tpots_ms: list[float] = []
    total_tokens = 0
    errors = 0

    async def one() -> None:
        nonlocal total_tokens, errors
        async with semaphore:
            try:
                r = await _one_request(client, model, prompt, max_tokens)
                ttfts_ms.append(r["ttft"] * 1000.0)
                total_tokens += r["tokens"]
                if r["tokens"] > 1:
                    # mean per-output-token time after the first token
                    tpots_ms.append((r["total"] - r["ttft"]) / (r["tokens"] - 1) * 1000.0)
            except Exception:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(total_requests)))
    elapsed_s = time.perf_counter() - start

    ttft_arr = np.array(ttfts_ms) if ttfts_ms else np.array([0.0])
    return {
        "ok": total_requests - errors,
        "errors": errors,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_tok_s": round(total_tokens / elapsed_s, 1) if elapsed_s else 0.0,
        "ttft_p50_ms": round(float(np.percentile(ttft_arr, 50)), 1),
        "ttft_p99_ms": round(float(np.percentile(ttft_arr, 99)), 1),
        "tpot_ms_mean": round(float(np.mean(tpots_ms)), 1) if tpots_ms else 0.0,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--total-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    stats = asyncio.run(
        load_test(
            args.base_url,
            args.model,
            total_requests=args.total_requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    )
    for key, value in stats.items():
        print(f"{key:>18}: {value}")


if __name__ == "__main__":
    _main()
