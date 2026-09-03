"""Verify that prefix-cache-aware routing is actually routing on the prefix cache.

`app.py` attaches `PrefixCacheAffinityRouter`, but a deployment that comes up
HEALTHY proves nothing about where requests land. This script proves it, by measuring the
one thing the router exists to change: how many prompt tokens get their KV reused instead
of recomputed.

The experiment is two requests carrying the SAME long prefix, fired at the same moment at
two replicas that can each hold one request:

    default router  ->  the second request goes to the free replica    ->  ~0% reuse
    affinity router ->  it waits for the replica holding the prefix    ->  ~50% reuse

50% rather than 100% because only one of the two requests can reuse anything: the first
one populates the cache. The pair carries ~2,900 prompt tokens of which the shared prefix
is ~1,440, so "the whole prefix, reused once" is ~49%.

Three details decide whether the measurement means anything, and getting any of them wrong
produces a confidently wrong answer:

  * `max_ongoing_requests=1`. Ray Serve LLM defaults this and `target_ongoing_requests` to
    ~1e9, so one in-flight request is not detectable load and Power of Two has nothing to
    act on -- both routers then send both requests to the same replica and score ~49%. One
    slot per replica makes admission control force the placement instead. This is a
    MEASUREMENT INSTRUMENT: it disables continuous batching and must never be deployed.
  * Both requests fired together with `asyncio.gather`. Sequentially, both replicas sit
    idle, Power of Two ties, and the tie-break is deterministic rather than random -- it
    picks the same replica every time, so both arms score ~49%.
  * The counters live on each node's Ray metrics agent, not on the Serve port, and they
    count prompt TOKENS rather than cache blocks.

The config under test comes from `app.routed_config`, so this verifies the shipped
reference rather than a copy of it.

Run (from the `code/` directory, on a 2-GPU workspace):
    python verify.py

Exits non-zero if the affinity arm fails to beat the default arm.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import ray
import requests
from openai import AsyncOpenAI
from ray import serve
from ray.serve.llm import build_openai_app

from app import DEFAULT_MODEL_ID, DEFAULT_MODEL_SOURCE, routed_config

HITS_METRIC = "ray_vllm_prefix_cache_hits_total"
QUERIES_METRIC = "ray_vllm_prefix_cache_queries_total"
EXPORT_WAIT_S = 15          # Ray's metrics agents republish roughly every 10s
REQUEST_TIMEOUT_S = 180.0
BASE_URL = "http://localhost:8000/v1"

# Long enough to span many 16-token cache blocks, short enough to leave room inside
# max_model_len=4096 for the reply.
SHARED_PREFIX = (
    "You are a meticulous assistant. Follow these standing instructions exactly "
    "and answer in one short sentence. " * 80
)


def measurement_config(use_router: bool, model_id: str, model_source: str):
    """The shipped routed_config, perturbed only for measurability.

    Starts from app.routed_config so this verifies the reference implementation.
    Two changes: one request slot per replica (so placement is forced rather than
    preferred), and optionally strip the router to get the load-only baseline.
    """
    config = routed_config(model_id, model_source, min_replicas=2)
    config.deployment_config["max_ongoing_requests"] = 1        # instrument, never production
    config.deployment_config["autoscaling_config"]["max_replicas"] = 2
    if not use_router:
        config.deployment_config.pop("request_router_config", None)
    return config


def _sum_metric(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        if line[len(name):len(name) + 1] not in ("{", " "):     # skip *_created companions
            continue
        total += float(line.rsplit(" ", 1)[-1])
    return total


def prefix_cache_counters() -> tuple[float, float]:
    """(hits, queries) in prompt TOKENS, summed over every node's Ray metrics agent.

    vLLM publishes these into Ray's metrics system, so they surface on the agents rather
    than on the port serving the model.
    """
    text = "".join(
        requests.get(
            f"http://{node['NodeManagerAddress']}:{node['MetricsExportPort']}/metrics",
            timeout=10,
        ).text
        for node in ray.nodes()
        if node["Alive"]
    )
    return _sum_metric(text, HITS_METRIC), _sum_metric(text, QUERIES_METRIC)


async def time_to_first_token(client: AsyncOpenAI, question: str,
                              prefix: str = SHARED_PREFIX) -> float:
    """Seconds to the first streamed chunk. Closes the stream rather than abandoning it."""
    started = time.perf_counter()
    stream = await client.chat.completions.create(
        model=DEFAULT_MODEL_ID,
        messages=[{"role": "system", "content": prefix},
                  {"role": "user", "content": question}],
        max_tokens=16, stream=True,
    )
    try:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                return time.perf_counter() - started
        return float("nan")
    finally:
        await stream.close()


async def measure_arm(use_router: bool, model_id: str, model_source: str) -> tuple[float, float]:
    """Deploy one configuration, run the two-request experiment, tear it down.

    Returns (reuse_fraction, slower_ttft_seconds).
    """
    serve.run(
        build_openai_app({"llm_configs": [measurement_config(use_router, model_id, model_source)]}),
        blocking=False,
    )
    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed",
                        timeout=REQUEST_TIMEOUT_S, max_retries=0)

    # Warm both replicas past first-call costs using an UNRELATED prefix, so SHARED_PREFIX
    # stays out of both caches. Its cold prefill is what the experiment counts.
    for _ in range(4):
        await time_to_first_token(client, "hello", prefix="Warm up the engine. ")

    time.sleep(EXPORT_WAIT_S)
    hits_before, queries_before = prefix_cache_counters()
    latencies = await asyncio.gather(
        time_to_first_token(client, "Summarize the instructions."),
        time_to_first_token(client, "Now restate them briefly."),
    )
    time.sleep(EXPORT_WAIT_S)
    hits_after, queries_after = prefix_cache_counters()

    serve.shutdown()
    time.sleep(5)

    queried = queries_after - queries_before
    reuse = (hits_after - hits_before) / queried if queried else float("nan")
    return reuse, max(latencies)


async def main(model_id: str, model_source: str) -> int:
    ray.init(ignore_reinit_error=True)
    gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"cluster GPUs: {gpus}")
    if gpus < 2:
        print("SKIP: need 2 GPUs to route between two replicas", file=sys.stderr)
        return 0

    default_reuse, default_ttft = await measure_arm(False, model_id, model_source)
    print(f"  default router : prompt tokens reused {default_reuse:6.1%} "
          f"| slower TTFT {default_ttft * 1000:7.1f} ms")

    router_reuse, router_ttft = await measure_arm(True, model_id, model_source)
    print(f"  affinity router: prompt tokens reused {router_reuse:6.1%} "
          f"| slower TTFT {router_ttft * 1000:7.1f} ms")

    # The affinity arm must reuse the prefix and the default arm must not. A 25-point gap
    # is a wide margin around the expected 0% vs ~49%, so this fails on a real regression
    # rather than on noise.
    gap = router_reuse - default_reuse
    print(f"\n  reuse gap: {gap * 100:+.1f} points")
    if not (gap > 0.25 and router_reuse > 0.25):
        print("FAIL: prefix-affinity routing is not reusing the shared prefix.\n"
              "      Check enable_prefix_caching=True, that both replicas came up, and\n"
              "      that imbalanced_threshold is not capping affinity at this load.",
              file=sys.stderr)
        return 1
    print("PASS: the affinity router steers a shared prefix to the replica holding its KV.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify prefix-cache-aware routing.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.model_id, args.model_source)))
