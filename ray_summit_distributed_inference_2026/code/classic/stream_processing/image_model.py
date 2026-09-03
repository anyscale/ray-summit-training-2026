"""The GPU model the poller feeds. Knows nothing about queues.

Only two decorator settings here matter to the pattern, and both are about
backpressure rather than about the model:

  - `max_queued_requests=3` is what makes a handle call RAISE instead of queueing
    without bound. Without it the poller can shovel work in forever and the
    backoff in poller.py never fires.
  - `max_ongoing_requests=1` because generation is GPU-bound: a second concurrent
    request on one replica buys nothing and doubles the latency of both.

Together with `target_ongoing_requests: 1`, a busy replica is a full replica, so
the autoscaler adds capacity as soon as one is working.
"""

import logging

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from ray import serve

logger = logging.getLogger("ray.serve")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 1},
    max_queued_requests=3,   # low capacity on purpose: triggers backpressure early
    max_ongoing_requests=1,  # GPU-bound, one image at a time
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 1,   # a busy replica is a full replica
    },
)
class ImageGenerator:
    """Text to image."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")

    def __call__(self, prompt: str, img_size: int = 512):
        assert prompt, "prompt cannot be empty"
        logger.info("Prompt: [%s]", prompt)
        return self.pipe(prompt, height=img_size, width=img_size).images[0]
