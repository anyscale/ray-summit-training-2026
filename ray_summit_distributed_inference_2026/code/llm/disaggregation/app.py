"""Prefill/decode disaggregation: the frontier-scaling P/D app.

One engine doing both phases is a compromise: prefill is compute-bound (process
the whole prompt at once), decode is memory-bound (one token per step). P/D
disaggregation runs them on SEPARATE replica pools and ships the KV cache from
prefill to decode over a fast transport (NIXL). Each pool scales on its own
bottleneck.

`build_pd_openai_app` builds the 3-tier graph (ingress -> decode -> prefill) from
a prefill_config and a decode_config. Two validated requirements (enforced by the
builder):
  - prefill and decode must use the SAME model_id (it is one model, split by phase).
  - both configs must set engine_kwargs.kv_transfer_config (the KV transport).

Tier-1: Qwen2.5-0.5B, same config for both phases, NixlConnector / kv_both.
Needs ~2 GPUs (a prefill replica + a decode replica) on a NIXL-capable image.

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run (2 GPUs, NIXL):
    serve run app:build_app
"""

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_pd_openai_app

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})


def phase_config(model_id: str, model_source: str) -> LLMConfig:
    """One phase's config. Prefill and decode share this shape; the builder wires
    them into separate pools. kv_transfer_config is mandatory for P/D: it selects
    the connector (NIXL) that moves KV blocks between the pools. kv_role="kv_both"
    lets a replica act as producer or consumer.
    """
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(
            autoscaling_config=dict(min_replicas=1, max_replicas=2),
        ),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read an s3:// model_source
            max_model_len=4096,
            enforce_eager=True,
            kv_transfer_config=dict(
                kv_connector="NixlConnector",
                kv_role="kv_both",
            ),
        ),
    )


class PDArgs(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    model_source: str = Field(default=DEFAULT_MODEL_SOURCE)


def build_app(args: PDArgs) -> Application:
    """Build the disaggregated app: same model, split into prefill and decode."""
    cfg = phase_config(args.model_id, args.model_source)
    # prefill_config and decode_config are independent LLMConfigs; here they are
    # identical, but in production they differ (decode wants more replicas,
    # prefill wants bigger token budgets).
    return build_pd_openai_app(
        dict(
            prefill_config=phase_config(args.model_id, args.model_source),
            decode_config=cfg,
        )
    )


app = build_app(PDArgs())
