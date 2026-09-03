"""Serve several models behind one OpenAI ingress.

`build_openai_app` takes a LIST of LLMConfigs. Each becomes its own LLMServer
deployment, autoscaled independently, and the shared ingress routes by the
client's `model=` field. One endpoint, many models, one Service.

Tier-1: two small Qwen models on small GPUs. Each scales on its own load.

Import targets (same pattern as app.py):
  - `app`:       the two-model graph; `serve run multimodel:app`.
  - `build_app`: the builder for service.yaml / CLI args.

Local run (one GPU per model, so ~2 GPUs to serve both):
    serve run multimodel:build_app
    # then query either model by name:
    #   client.chat.completions.create(model="qwen-0.5b", ...)
    #   client.chat.completions.create(model="qwen-1.5b", ...)
"""

from pydantic import BaseModel
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_openai_app


def model_config(model_id: str, model_source: str) -> LLMConfig:
    """One small model, autoscaled 1-2 replicas on one GPU."""
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(
            autoscaling_config=dict(min_replicas=1, max_replicas=2),
        ),
        engine_kwargs=dict(max_model_len=4096, enforce_eager=True),
    )


# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
# Two models the client selects between with the OpenAI `model=` field.
# Hugging Face ids, not s3:// URIs: only the 0.5B is on the course S3 mirror, and a
# mixed list would need per-model load_format. The other Tier-1 apps stream from S3.
MODELS = [
    ("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
]


class MultiModelArgs(BaseModel):
    """No required args: the model list is fixed for the demo. Extend with a
    list field to parameterize from service.yaml."""


def build_app(args: MultiModelArgs) -> Application:
    """Build one OpenAI app serving every model in MODELS."""
    configs = [model_config(mid, src) for mid, src in MODELS]
    return build_openai_app({"llm_configs": configs})


app = build_app(MultiModelArgs())
