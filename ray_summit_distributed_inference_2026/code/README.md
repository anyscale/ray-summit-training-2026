# Serving reference

Production-grade Ray Serve reference implementations, organized by concept. Three buckets:

- **`llm/`**: distributed LLM inference with `ray.serve.llm`: an OpenAI-compatible
  endpoint, composition with ordinary deployments, KV-cache-aware routing,
  prefill/decode disaggregation, wide expert parallelism, engine measurement, and
  offline evaluation.
- **`classic/`**: non-LLM serving: a batched two-tower Ranker, load testing, custom routing and
  autoscaling, async submit/poll inference, and queue-driven stream processing.
- **`ops/`**: operational tooling that applies to either track: local debugging and
  distributed tracing.

**Every concept directory is self-contained**, and **every file holds one idea.** A
directory's `service.yaml` sets `working_dir: .` and `import_path: app:build_app`, so you
deploy or run from inside it and nothing reaches across the tree. Within a directory,
`app.py` is always the entry point and the pieces it composes sit in their own modules,
so any single file can be shown on its own:

```bash
cd llm/routing && serve run app:build_app
cd classic/ranking && anyscale service deploy -f service.yaml
```

Runnable examples are Tier-1: a small model on 1-2 GPUs. Production configs live in
`service_prod.yaml` and are shown, not run.

The custom request-router and custom autoscaling-policy APIs are alpha and experimental
respectively, so expect their signatures to move.

## `llm/`: distributed LLM inference

| Directory | What it is |
|---|---|
| `serving/` | `app.py` serves one model through `build_openai_app`. `multimodel.py` puts two models behind one ingress and routes on the client's `model=` field. `query.py` holds the OpenAI-client helpers (one-shot chat, and streaming for TTFT). `service.yaml` is Tier-1; `service_prod.yaml` is the same builder scaled by config. |
| `composition/` | `build_llm_deployment` drops one level below `build_openai_app` so an LLM binds like any other Serve deployment. `triage.py` is a CPU deployment that answers from a lookup table and knows nothing about LLMs; `app.py` composes it with the engine behind two endpoints, `/ask` (one JSON body) and `/ask/stream` (OpenAI SSE). |
| `routing/` | `PrefixCacheAffinityRouter` steers a request to the replica already holding its prefix's KV, falling back to Power of Two when replicas go load-imbalanced. `verify.py` measures prompt-token reuse with and without affinity and exits non-zero if affinity does not win, so it doubles as a regression check. Needs 2 GPUs. |
| `disaggregation/` | Prefill is compute-bound, decode is memory-bound. `build_pd_openai_app` runs them on separate replica pools and ships KV between them over NIXL, so each pool scales on its own bottleneck. |
| `expert_parallel/` | `build_dp_openai_app` replicates the attention path via `data_parallel_size` and gang-schedules the ranks. Add `enable_expert_parallel=True` to shard MoE experts across them. |
| `engine/` | `load_test.py` drives streamed completions at bounded concurrency and reports TTFT, time per output token, and throughput. `reset_cache.py` clears the vLLM prefix cache so a cold and warm replay can be timed against each other. |
| `offline_inference/` | Evaluation is offline: score a dataset, throughput matters, no endpoint. `ray.data.llm` runs vLLM inside a Ray Data pipeline, so the same engine optimizations apply to a bulk job. Runs as a Job, not a Service. |

### Serve an LLM on one GPU

```bash
cd llm/serving
serve run app:build_app                  # Qwen2.5-0.5B behind /v1
python query.py                          # chat + streaming
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen-0.5b","messages":[{"role":"user","content":"Hello!"}]}'
```

Deploy the same graph as a Service with `anyscale service deploy -f service.yaml`.

### The Tier-1 model constants are duplicated on purpose

Each `llm/` directory declares its own model id, S3 source, and `AWS_REGION` runtime
env rather than importing them, so a directory reads top to bottom with nothing to
chase. The cost is that changing the mirror, the region, or the Tier-1 model is a
multi-file edit. Every site carries the same marker, so one command finds them all:

```bash
grep -rn "TIER1 MODEL" .
```

## `classic/`: non-LLM serving

| Directory | What it is |
|---|---|
| `ranking/` | A two-tower Ranker behind a FastAPI ingress. `schemas.py` is the API contract, `model.py` the CPU-only PyTorch model, `feature_store.py` a mock online store whose `batch_get` fetches many users in one round trip. `ranker.py` splits `RankerLogic` (pure, Serve-free, testable) from the `Ranker` deployment (batched feature lookup, `@serve.batch`, autoscaling, live `reconfigure`). `app.py` composes the graph. |
| `load_testing/` | Locust drives the Ranker open-loop, with a `LoadTestShape` that ramps, sustains, spikes, and drains, so one headless run shows both a scale-up and a scale-down. `benchmark.yaml` pins the app to a single replica with backpressure lifted, which is how you read `target_ongoing_requests` off the latency knee: deploy it first, or the numbers mean nothing. |
| `custom_routing/` | The default router balances on queue length and knows nothing about what a replica has cached. `router.py` subclasses `RequestRouter` to route on user id instead, so a replica's per-user feature cache actually hits, and sheds affinity when the preferred replica runs hot. Shows the full hook set: `choose_replicas` returning ranked fallbacks, `initialize_state` for tuning, `on_request_routed` for outcomes, and `record_routing_stats()` in `app.py` feeding `replica.routing_stats`. |
| `custom_autoscaling/` | One policy per `policy_*.py`, each scaling on something the default queue-depth loop cannot see: a schedule (capacity leads the traffic instead of chasing it), replica CPU and memory via `record_autoscaling_stats()`, the queued fraction plus its trend through `policy_state`, and a class-based policy taking its target from outside the cluster. Its own README compares them and documents the `AutoscalingContext` fields. |
| `async_inference/` | Long-running inference through Ray Serve's task-consumer API. `scorer.py` is the `@task_consumer` worker that drains a queue; `app.py` is the HTTP ingress that enqueues a job and hands back a ticket, so no connection is held across the work; `task_queue.py` is the config both share. `client.py` is a plain HTTP client, needing no source and no broker access. Runs broker-free on a filesystem broker. |
| `stream_processing/` | Work arrives from SQS rather than a queue Serve manages. `poller.py` forwards messages to the `image_model.py` GPU pool, handles `BackPressureError` with exponential backoff, and deletes a message only after the result lands, giving at-least-once delivery. The contrast with `async_inference/` is built-in against do-it-yourself. |

### Run the Ranker locally

```bash
cd classic/ranking
python model.py                                            # writes /tmp/ranker_serve/ranker.pt
serve run app:build_app weights_path=/tmp/ranker_serve/ranker.pt
```

```bash
curl -s localhost:8000/healthz
curl -s -XPOST localhost:8000/rank \
  -H 'content-type: application/json' \
  -d '{"user_id": "u1", "candidate_ids": ["a", "b", "c"]}'
```

On an Anyscale workspace, weights live on the shared mount
(`/mnt/cluster_storage/ranker_serve/ranker.pt`), which is the default `RANKER_WEIGHTS`,
so `serve run app:app` works with no arguments.

## `ops/`: operational tooling

| Directory | What it is |
|---|---|
| `debugging/` | `serve.run(app, _local_testing_mode=True)` runs a deployment in-process, so a handle call is an ordinary function call you can step through in a debugger with no cluster involved. |
| `tracing/` | Anyscale's `tracing_config` emits OpenTelemetry spans for every proxy and replica hop, which is how you find which deployment in a multi-hop graph owns the latency. Spans land under `/tmp/ray/session_latest/logs/serve/spans/`. |
