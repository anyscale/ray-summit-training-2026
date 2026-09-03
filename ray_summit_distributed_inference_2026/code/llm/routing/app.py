"""KV-cache-aware routing.

The default router (Power of Two Choices) balances on queue length and ignores
what each replica has cached. For LLMs that wastes the prefix cache: a request
sharing a long system prompt should land on the replica that already holds that
prefix's KV, not a random one.

`PrefixCacheAffinityRouter` does that. It tracks processed prefixes in a tree and
routes a request to the replica with the highest prefix-match rate, falling back
to Power of Two when replicas are load-imbalanced (so a hot prefix cannot
overload one replica). You attach it through `deployment_config.request_router_config`.

Tier-1: one model, multiple replicas (so routing has somewhere to choose), with
prefix-affinity routing on. Needs ~2 small GPUs to run 2 replicas; on 1 GPU the
config still builds with a single replica (routing is then a no-op).

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run:
    serve run app:build_app

Verify the routing actually routes on the cache (2 GPUs, exits non-zero on failure):
    python verify.py
"""

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import LLMConfig, build_openai_app

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})


def routed_config(model_id: str, model_source: str, min_replicas: int) -> LLMConfig:
    """One model, prefix-cache-aware routing across its replicas.

    request_router_class is given as an import string so the config stays
    JSON-serializable (it travels to the Service). imbalanced_threshold is the
    queue-length gap beyond which the router stops chasing prefix affinity and
    rebalances on load; its default is infinity (affinity always wins), so we set
    a finite 20 here to cap how lopsided a hot prefix may make the fleet.
    """
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=min_replicas, max_replicas=4, target_ongoing_requests=8
            ),
            request_router_config=RequestRouterConfig(
                request_router_class="ray.serve.llm.request_router.PrefixCacheAffinityRouter",
                request_router_kwargs={"imbalanced_threshold": 20},
            ),
        ),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read an s3:// model_source
            max_model_len=4096,
            enforce_eager=True,
            enable_prefix_caching=True,  # the cache the router is steering toward
        ),
    )


class RoutingArgs(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    model_source: str = Field(default=DEFAULT_MODEL_SOURCE)
    # 2 replicas need ~2 GPUs; drop to 1 to run on a single-GPU workspace.
    min_replicas: int = Field(default=2, ge=1)


def build_app(args: RoutingArgs) -> Application:
    cfg = routed_config(args.model_id, args.model_source, args.min_replicas)
    return build_openai_app({"llm_configs": [cfg]})


app = build_app(RoutingArgs())
