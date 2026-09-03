# Distributed LLM Inference with Ray Serve — Ray Summit 2026

© 2026, Anyscale. All Rights Reserved

Serve high-throughput, low-latency inference on Ray Serve, from a classic ML model to a
distributed LLM. Each notebook is self-contained and defines every helper it needs, so you can
open any one of them and run it top to bottom.

## Agenda

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [Ray and Ray Serve fundamentals](01_ray_serve_fundamentals.ipynb) | Deployments, request routing and admission control, FastAPI ingress, composing several deployments |
| 02 | [Architecture, fault tolerance, and scaling](02_architecture_fault_tolerance_scaling.ipynb) | The control plane behind an app: health checks, self-healing, and how the autoscaling loop works |
| 03 | [Ray Serve LLM](03_ray_serve_llm_framework.ipynb) | An OpenAI-compatible endpoint with `ray.serve.llm`, and scaling at the framework and engine levels |
| 04 | [Inside the inference engine](04_inside_the_inference_engine.ipynb) | TTFT and ITL, the KV cache, paged attention, prefix caching, continuous batching, chunked prefill |
| 05 | [Scaling LLM inference](05_scaling_llm_inference.ipynb) | KV-cache-aware routing, prefill/decode disaggregation, mixture-of-experts at scale |

Notebooks 03 and 04 need one GPU; the rest run on CPU. Runnable cells use a small model, while the
multi-node production configs appear as read-only blocks and as `code/**/service_prod*.yaml`.

## Also here

- **`code/`** holds deployable versions of the apps the notebooks build, grouped by concept under
  `llm/`, `classic/`, and `ops/`. Each concept directory is self-contained: run or deploy from inside
  it. See [code/README.md](code/README.md).
