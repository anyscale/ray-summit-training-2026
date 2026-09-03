# syntax=docker/dockerfile:1
# =============================================================================
# Container image for the course notebooks.
#
# The Anyscale Ray LLM base image already bundles ray[serve,llm], a
# version-matched vLLM, and their dependencies (openai, httpx, numpy, pandas,
# fastapi, aiohttp, requests). This adds only what the notebooks need on top.
#
# Ray, vLLM, Python, and CUDA move together, so the base image supplies vLLM
# rather than a loose `pip install vllm`. Bump the tag and the two together.
# =============================================================================
FROM anyscale/ray-llm:2.56.0-py312-cu130

# runai-model-streamer(-s3)
#     Every GPU lesson and every app under code/ loads Qwen2.5-0.5B from a public
#     S3 mirror with engine_kwargs.load_format="runai_streamer". These provide the
#     Run:ai Model Streamer and its S3 backend, so vLLM streams safetensors
#     straight from the bucket into GPU memory instead of downloading to local
#     disk first. A classroom cold-starting replicas together would otherwise pull
#     the same weights from an external service.
#
# locust
#     The load generator behind code/classic/load_testing.
#
# celery
#     The broker client behind Ray's task-consumer adapter, used by
#     code/classic/async_inference. Ray declares it only under the
#     ray[serve-async-inference] extra, so the import succeeds without it and then
#     fails when the adapter is instantiated. Installed directly, because the extra
#     would reinstall ray and could swap the base image's vLLM-matched wheel.
RUN pip install --no-cache-dir \
        runai-model-streamer \
        runai-model-streamer-s3 \
        locust \
        celery

# Fail the build on a broken pin or a missing API instead of at runtime. vLLM is
# not imported here: importing it initializes CUDA, which fails on a GPU-less
# builder even when the image is correct.
RUN python -c "import ray, openai, httpx; \
from ray.serve.llm import LLMConfig, build_openai_app, build_pd_openai_app, build_dp_openai_app; \
from ray.serve.llm.request_router import PrefixCacheAffinityRouter; \
print('ray   ', ray.__version__); \
print('openai', openai.__version__); \
print('ray.serve.llm builders import OK')"

# Suppress the spurious accelerator warning Ray emits on CPU-only tasks, where
# num_gpus is 0 or None. Harmless on GPU tasks.
ENV RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Verify on a GPU node, not during `docker build`:
#   python -c "import torch, vllm; \
#     print('cuda available:', torch.cuda.is_available()); \
#     print('device:', torch.cuda.get_device_name(0)); \
#     print('vllm:', vllm.__version__)"
#
# Then serve a model end to end:
#   cd code/llm/serving && serve run app:build_app
