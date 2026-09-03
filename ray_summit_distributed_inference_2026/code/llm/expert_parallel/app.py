"""Data-parallel attention and wide expert parallelism: the frontier-scaling DP app.

Mixture-of-Experts (MoE) models (DeepSeek-V3, Kimi-class) have many expert
FFNs; only a few fire per token. Sharding experts across many GPUs is "wide
expert parallelism" (WideEP). Its partner is data-parallel attention: replicate
the attention path N ways so the experts stay fed.

`build_dp_openai_app` builds a data-parallel deployment from ONE llm_config whose
engine_kwargs set `data_parallel_size`. Serve gang-schedules the N ranks
together automatically. Add `enable_expert_parallel=True` for the MoE expert
sharding (WideEP).

Tier-1: Qwen2.5-0.5B with data_parallel_size=2, two ranks, co-locatable on a
2-GPU box (or one if small). It demonstrates the API; true WideEP needs a real
MoE model and many GPUs (see service_prod.yaml).

Note: build_dp_openai_app takes a SINGULAR `llm_config` (not the `llm_configs`
list that build_openai_app takes).

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run:
    serve run app:build_app
"""

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_dp_openai_app

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})


def dp_config(model_id: str, model_source: str, dp_size: int) -> LLMConfig:
    """Data-parallel config. data_parallel_size replicates the attention path
    dp_size ways; Serve gang-schedules the ranks and PACKs them onto the fewest
    nodes automatically (rank co-location needs no config knob; dp_size_per_node
    is a no-op in current Ray). For a real MoE model you would also pass
    enable_expert_parallel=True (kept off here: Qwen-0.5B is dense).
    """
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read an s3:// model_source
            data_parallel_size=dp_size,
            tensor_parallel_size=1,
            max_model_len=4096,
            enforce_eager=True,
        ),
    )


class DPArgs(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    model_source: str = Field(default=DEFAULT_MODEL_SOURCE)
    dp_size: int = Field(default=2, ge=1)


def build_app(args: DPArgs) -> Application:
    """Build the data-parallel app. Note the singular `llm_config` key."""
    cfg = dp_config(args.model_id, args.model_source, args.dp_size)
    return build_dp_openai_app({"llm_config": cfg})


app = build_app(DPArgs())
