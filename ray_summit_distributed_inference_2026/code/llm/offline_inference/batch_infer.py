"""Offline batch LLM inference: the evaluation job.

Serving (everything before this) is online: one request, one response, latency
matters. Evaluation is offline: score a whole dataset, throughput matters, no
endpoint needed. `ray.data.llm` runs vLLM inside a Ray Data pipeline, so the
engine optimizations (continuous batching, prefix caching) apply to a
bulk job, and Ray Data streams the dataset through a (possibly autoscaled) pool
of engine actors.

This is a Ray Data pipeline, not a Serve app, so it runs as an Anyscale Job, not
a Service. The shape: a vLLMEngineProcessorConfig, a preprocess that turns each
row into chat messages + sampling params, and a postprocess that extracts the
generated text.

Tier-1: Qwen2.5-0.5B over a tiny prompt set on one GPU.

Run as a job (one GPU):
    python batch_infer.py
"""

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})


def build_eval_processor(concurrency: int = 1, batch_size: int = 32):
    """A vLLM batch processor: same engine knobs as serving, applied to a
    bulk dataset. concurrency is the engine-actor pool size; pass a (min, max)
    tuple to autoscale it for a large dataset (the Tier-2 shape).
    """
    config = vLLMEngineProcessorConfig(
        model_source=MODEL_SOURCE,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read the s3:// model_source
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            max_model_len=4096,
        ),
        runtime_env=MODEL_SOURCE_ENV,   # reaches the engine actors via map_batches
        concurrency=concurrency,
        batch_size=batch_size,
    )
    return build_processor(
        config,
        # Each input row -> OpenAI-style chat messages + sampling params.
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(temperature=0.0, max_tokens=64),
        ),
        # Engine output row -> just the fields we want to keep.
        postprocess=lambda row: dict(
            prompt=row["prompt"],
            response=row["generated_text"],
        ),
    )


def main() -> None:
    prompts = [
        "Summarize what Ray Serve does in one sentence.",
        "What is a KV cache?",
        "Name two reasons to use prefill/decode disaggregation.",
        "What does tensor parallelism do?",
    ]
    ds = ray.data.from_items([{"prompt": p} for p in prompts])

    processor = build_eval_processor()
    # log_input_column_names() prints what columns the first stage needs, handy
    # when wiring a real dataset.
    processor.log_input_column_names()

    ds = processor(ds)
    for row in ds.take_all():
        print(f"Q: {row['prompt']}\nA: {row['response']}\n")


if __name__ == "__main__":
    main()
