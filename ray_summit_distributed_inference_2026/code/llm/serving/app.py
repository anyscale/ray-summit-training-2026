"""Serve an LLM behind an OpenAI-compatible endpoint.

The Ranker (../../classic/ranking/) showed the bare Serve mechanics: a
`@serve.deployment`, a FastAPI ingress, composition with `.bind()`. Serving an
LLM does not hand-roll any of that. `ray.serve.llm.build_openai_app` returns a
ready Application: an OpenAI-compatible router in front of one `LLMServer`
deployment per model, each wrapping a vLLM engine.

Two tiers, one API:
  - Tier-1 (this file, runs now): Qwen2.5-0.5B on one GPU,
    enforce_eager + max_model_len=4096 so it loads fast.
  - Tier-2 (service_prod.yaml, shown not run): the SAME LLMConfig with a bigger
    model, tensor_parallel_size, and a longer context: config, not new code.

Import targets (same pattern as app.py):
  - `app`       the default single-model graph; `serve run app:app`.
  - `build_app` the builder; `serve run app:build_app model_source=...`
                  and service.yaml (`import_path: app:build_app`).

Local run (one GPU):
    serve run app:build_app
    python query.py           # or see query.py
"""

import os

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_openai_app

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
# The model id is how a client addresses the model (the OpenAI `model=` field);
# model_source is where the weights come from: an s3:// URI (streamed straight into
# GPU memory via load_format=runai_streamer), a Hugging Face id, or a local path.
DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})


def tier1_llm_config(model_id: str, model_source: str, accelerator_type: str = "") -> LLMConfig:
    """The runnable Tier-1 config: one small model on one GPU.

    enforce_eager skips CUDA-graph capture (faster cold start, fine for a demo);
    max_model_len caps the KV cache and keeps cold start fast. accelerator_type is
    optional: leave it unset on a homogeneous fleet (every node the same GPU) and
    Serve schedules on the only accelerator there is; pin it to steer models onto
    specific hardware in a mixed fleet. The autoscaling_config keys are the same
    ones the Ranker used earlier; here they live under deployment_config because an
    LLM replica is a GPU actor.
    """
    config = dict(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(
            autoscaling_config=dict(min_replicas=1, max_replicas=2),
        ),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read an s3:// model_source
            max_model_len=4096,
            enforce_eager=True,
        ),
    )
    if accelerator_type:
        config["accelerator_type"] = accelerator_type   # pin a GPU type in a mixed fleet
    return LLMConfig(**config)


class LLMAppArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI.

    Defaults read env vars so `serve run app:build_app` works with no
    arguments on a single-GPU workspace.
    """

    model_id: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL_ID", DEFAULT_MODEL_ID)
    )
    model_source: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL_SOURCE", DEFAULT_MODEL_SOURCE)
    )
    accelerator_type: str = Field(
        default_factory=lambda: os.environ.get("LLM_ACCELERATOR", "")
    )


def build_app(args: LLMAppArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service.

    Returns the OpenAI-compatible app for one model. `build_openai_app` takes a
    dict conforming to LLMServingArgs (`llm_configs` is a list, so the same
    builder serves several models, see multimodel.py).
    """
    llm_config = tier1_llm_config(args.model_id, args.model_source, args.accelerator_type)
    return build_openai_app({"llm_configs": [llm_config]})


# Module-scope default so `serve run app:app` and the notebook work
# with no args. Production deploys use the builder with args from service.yaml.
app = build_app(LLMAppArgs())
