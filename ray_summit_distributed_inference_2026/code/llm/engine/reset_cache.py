"""Measure the vLLM prefix cache: warm replay vs cold replay after a reset.

This makes the KV prefix cache visible. Send the same long prompt twice and the
second is fast (its prefix KV is cached). Reset the cache and the next identical
request is slow again (prefill recomputed). The gap IS the cache's value.

Uses DevIngress, which adds control-plane endpoints (including
`/reset_prefix_cache`) on top of the normal OpenAI app. Two reset paths, same
effect:
  - in-cluster: `broadcast(handle, "reset_prefix_cache")` to every replica.
  - HTTP:       POST /reset_prefix_cache (for external clients).

Runnable on one small GPU.

Run:
    python reset_cache.py            # in-cluster broadcast reset
    python reset_cache.py --use-http # HTTP endpoint reset
"""

import argparse
import asyncio
import time

import httpx
from ray import serve
from ray.llm._internal.serve.core.ingress.dev_ingress import build_dev_openai_app
from ray.llm._internal.serve.utils.broadcast import broadcast
from ray.serve.llm import LLMConfig

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
# MODEL is the client-facing id (the OpenAI `model=` field); MODEL_SOURCE is where the
# weights come from. Keep the id matching the rest of the course so a reset issued
# against an app started elsewhere addresses the same model.
MODEL = "qwen-0.5b"
MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})
# DevIngress names the model deployment "LLMServer:<name>" from deployment_config.
DEPLOYMENT_NAME = "llm"
BASE_URL = "http://localhost:8000"


def build() -> None:
    """Start the model behind DevIngress with prefix caching ON."""
    llm_config = LLMConfig(
        model_loading_config=dict(model_id=MODEL, model_source=MODEL_SOURCE),
        deployment_config=dict(num_replicas=2, name=DEPLOYMENT_NAME),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read the s3:// model_source
            enable_prefix_caching=True,   # the feature under test
            enforce_eager=True,
            max_num_batched_tokens=4096,
        ),
    )
    serve.run(build_dev_openai_app({"llm_configs": [llm_config]}))


async def time_request(prompt: str) -> float:
    """Send one 1-token completion and return its wall-clock seconds."""
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{BASE_URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 1},
        )
        resp.raise_for_status()
    return time.perf_counter() - start


def reset_via_handle() -> None:
    """Reset every replica's prefix cache with one in-cluster broadcast."""
    handle = serve.get_deployment_handle(f"LLMServer:{DEPLOYMENT_NAME}", app_name="default")
    broadcast(handle, "reset_prefix_cache")


async def reset_via_http() -> None:
    """Reset the prefix cache through the DevIngress HTTP endpoint."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{BASE_URL}/reset_prefix_cache", json={"model": MODEL})
        resp.raise_for_status()


async def main(use_http: bool) -> None:
    build()
    # A long shared prefix so prefill dominates and the cache effect is visible.
    prompt = "The quick brown fox jumps over the lazy dog. " * 3000

    await time_request(prompt)                       # 1st call: populates the cache
    warm = await time_request(prompt)                # 2nd call: served from cache
    print(f"warm (cached) request:  {warm:.4f}s")

    await reset_via_http() if use_http else reset_via_handle()

    cold = await time_request(prompt)                # prefill recomputed
    print(f"cold (post-reset):      {cold:.4f}s")
    print(f"slowdown after reset:   {cold / warm:.1f}x")

    serve.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the vLLM prefix cache.")
    parser.add_argument("--use-http", action="store_true", help="Reset via HTTP, not broadcast.")
    args = parser.parse_args()
    asyncio.run(main(use_http=args.use_http))
