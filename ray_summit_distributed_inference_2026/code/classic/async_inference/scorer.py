"""The queue worker. Nothing in here knows about HTTP.

`@task_consumer` is what makes this deployment pull from a queue instead of
receiving calls, and `@task_handler` marks the method that handles a named task.
Enqueueing is the ingress's job (see app.py); this side only consumes.

Two constraints the decorators impose:
  - Handlers must be SYNCHRONOUS. An `async def` handler raises
    NotImplementedError.
  - `@task_consumer` takes its config when the class is DECORATED, so the queue
    location is resolved at import time and cannot come from `args:` the way a
    weights path can. Set ASYNC_QUEUE_ROOT in the environment instead.
"""

import os
import time
import zlib

import torch
from ray import serve
from ray.serve.task_consumer import task_consumer, task_handler

from task_queue import build_config

# Queue state lives on the shared mount so every replica sees the same broker
# directories. Override with ASYNC_QUEUE_ROOT to run outside a workspace.
DEFAULT_QUEUE_ROOT = "/mnt/cluster_storage/async_inference"
QUEUE_ROOT = os.environ.get("ASYNC_QUEUE_ROOT", DEFAULT_QUEUE_ROOT)

# Built once here and imported by app.py, so both halves share one queue.
TASK_PROCESSOR_CONFIG = build_config(QUEUE_ROOT)

EMBED_DIM = 64
NUM_ITEMS = 10_000


class ScorerLogic:
    """Pure batch scoring: no Serve, no queue, unit-testable on its own.

    The same logic/wrapper split the Ranker uses. Construct it and call `score(...)`
    directly with no Ray runtime involved.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, num_items: int = NUM_ITEMS) -> None:
        torch.manual_seed(0)  # reproducible weights, so scores compare across runs
        self.item_emb = torch.nn.Embedding(num_items, embed_dim)
        self.item_emb.eval()

    def score(self, user_id: str, num_candidates: int) -> dict:
        """Score `num_candidates` items for one user and summarize the result.

        Returns summary statistics rather than every score: a result backend is for
        handoff, not for shipping a large payload.
        """
        # CRC32, not the built-in hash: hash() is salted per process, so the same
        # user would score differently on each replica.
        generator = torch.Generator().manual_seed(zlib.crc32(user_id.encode("utf-8")))
        user_vec = torch.randn(EMBED_DIM, generator=generator)
        item_ids = torch.randint(0, NUM_ITEMS, (num_candidates,), generator=generator)
        with torch.no_grad():
            scores = self.item_emb(item_ids) @ user_vec
        top = torch.topk(scores, k=min(5, num_candidates))
        return {
            "user_id": user_id,
            "scored": num_candidates,
            "top_indices": item_ids[top.indices].tolist(),
            "top_scores": [round(s, 4) for s in top.values.tolist()],
        }


@serve.deployment(
    ray_actor_options={"num_cpus": 2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
)
@task_consumer(task_processor_config=TASK_PROCESSOR_CONFIG)
class Scorer:
    """Drains the queue, one job at a time per replica."""

    def __init__(self, default_candidates: int = 50_000) -> None:
        self.logic = ScorerLogic()
        self.default_candidates = default_candidates

    @task_handler(name="score_batch")
    def score_batch(self, user_id: str, num_candidates: int | None = None) -> dict:
        """Handle one enqueued job. Synchronous, as the decorator requires.

        The default candidate count is large enough that a job takes long enough to
        be worth queueing, which is the whole premise of this pattern.
        """
        started = time.perf_counter()
        result = self.logic.score(user_id, num_candidates or self.default_candidates)
        result["elapsed_s"] = round(time.perf_counter() - started, 3)
        return result
