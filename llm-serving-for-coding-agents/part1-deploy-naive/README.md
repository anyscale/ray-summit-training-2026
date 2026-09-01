# Part 1 — Deploy `qwen3.6-27b` on Anyscale (the naive way)

Goal: get a working OpenAI-compatible endpoint with the **least** configuration. It's deliberately
un-optimized — the point is to prove the model serves and to give you a baseline to optimize later.

**What "naive" means here:** 4× L4 (`g6.12xlarge`, TP=4), single replica, default engine settings —
deliberately **highly un-optimized**. It works; it's just the wrong *shape* for a team (≈ one concurrent
user, slow cold start), and 4× L4 isn't the right GPU for this model either — the FP8 weights fit on a
single bigger GPU. [Part 3](../part3-optimize/) fixes both: it moves to 1× RTX PRO 6000 96GB (TP=1) and
adds the software optimizations.

## Files
- `PROMPT.md` — the prompt that generates this deployment.
- `serve_qwen3_6_27b_naive.py` — the Ray Serve LLM app (one `LLMConfig`, built with `build_openai_app`).
- `service_naive.yaml` — the Anyscale Service config (compute + image + import path).
- `client.py` — a tiny OpenAI-SDK script to sanity-check the endpoint.

## Deploy

Give the prompt in [`PROMPT.md`](PROMPT.md) to your coding agent. It writes the serve app and the service
config, then deploys them as an Anyscale **Service** for you. The committed files here are what it should
produce — use them to compare, or deploy them directly:

```bash
cd part1-deploy-naive
anyscale service deploy -f service_naive.yaml
```

When the deploy reports **RUNNING**, grab the endpoint from the console (**Services → your service →
Query**) — that gives you the `base_url` and bearer token Part 2 needs.

## Verify

```bash
cd part1-deploy-naive
export ANYSCALE_BASE_URL="https://<your-service-host>/v1"   # from the Query panel, include /v1
export ANYSCALE_API_KEY="<your service bearer token>"
python client.py        # sends a chat completion, prints the reply
```

The first request after the deploy (or after idle) **cold-starts** for ~2–3 min. That slowness is one of
the things [Part 3](../part3-optimize/) removes.

## Native multi-API endpoint (direct streaming)

This deployment turns on **direct streaming**, so this one endpoint speaks all three agent APIs natively —
no proxy needed:

| Path | Used by |
|---|---|
| `POST /v1/chat/completions` | Cursor (OpenAI) |
| `POST /v1/messages` | Claude Code (Anthropic) |
| `POST /v1/responses` | Codex (OpenAI Responses) |

Connect your agents in **[Part 2](../part2-connect-clients-production/)** — point Claude Code, Codex, and
Cursor **directly** at the paths above, **no proxy, no `pip install`**. (A LiteLLM-gateway alternative
also exists — handy for a service *without* direct streaming — but this repo uses the direct path.)

> **How it's enabled (and a gotcha):** direct streaming needs
> `RAY_SERVE_ENABLE_HA_PROXY=1` + `RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1` in the Ray Serve
> **controller's** environment — the controller reads the flag at startup (`build_app.py`). A
> per-deployment `runtime_env` (e.g. `LLMConfig(runtime_env=…)`) reaches only the **replicas**, not the
> controller, so setting them there makes the app fail with *"ingress_request_router requires
> HAProxy."*
>
> That's why they sit in `service_naive.yaml`'s top-level `env_vars:` — Anyscale applies those
> cluster-wide, so the controller picks them up. To check, curl the endpoint: `/v1/messages` and
> `/v1/responses` should respond (not `404`); a `404` means direct streaming isn't active.
> *(Validated: all three native endpoints return 200.)*

## Why this image / GPU

- **Image `anyscale/ray-llm:2.57.0-py312-cu130`** — the stock Anyscale base image (public on Docker Hub,
  pullable with no creds), shipping **vLLM 0.25.1**, which accepts Claude Code's `/v1/messages` request
  schema natively. No custom image is needed here. Earlier releases don't work out of the box:
  `ray-llm:2.56.0` ships vLLM 0.22.0, which serves Codex and Cursor but rejects Claude Code's `system` role
  (this tutorial used to carry a prebuilt image that force-upgraded it to 0.23.0), and `ray-llm:2.55.x`
  ships vLLM 0.18, too old to load Qwen3.6.
- **4× L4 / TP=4** (`g6.12xlarge`) — a common, widely-available GPU shape, used here as the baseline. It's
  not optimal: L4 has the lowest memory bandwidth of the serving GPUs (~300 GB/s) and the 4 cards talk over
  PCIe with no NVLink, so tensor-parallel comms cost you. The FP8 weights fit on a single bigger GPU, so
  [Part 3](../part3-optimize/) moves to **1× RTX PRO 6000 96GB** (TP=1) to serve the model's full 256K
  context in FP8.

## KV cache dtype

`serve_qwen3_6_27b_naive.py` leaves `kv_cache_dtype` at the vLLM default (bf16).

**Validated capacity** (vLLM 0.22.0, TP=4, `gpu_memory_utilization=0.85`, `max_model_len=131072`) — measured
on the older 2.56.0 image and **not re-measured on vLLM 0.25.1**:

| Metric | Value |
|---|---|
| Available KV cache memory | **10.38 GiB / GPU** (~41.5 GiB total) |
| GPU KV cache size | **652,346 tokens** |
| Max concurrency @ 128K tokens/request | **4.98×** (raw cache capacity) |

> ⚠️ The 4.98× is the **raw KV cache capacity** (pure storage). Practical concurrency under
> real serving load will be lower due to decode-phase memory fragmentation, vLLM's safety margins,
> and PCIe bandwidth saturation between the 4 L4 GPUs (~300 GB/s each). Actual concurrent user
> capacity should be measured with real workloads.

→ Next: **[Part 2 — connect Claude Code / Codex / Cursor](../part2-connect-clients-production/README.md)** (no proxy).
