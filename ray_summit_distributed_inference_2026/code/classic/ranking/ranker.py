"""The Ranker: the production synthesis the notebooks build up to.

Two classes, split along the seam the "Designing Serve Applications" material
draws between business logic and the deployment wrapper:

  - `RankerLogic`: plain, Serve-free, unit-testable: the model plus the one
    vectorized forward pass. Construct it and call `forward(...)` in a test with
    no Ray runtime, no metrics, no feature store.
  - `Ranker(RankerLogic)`: the `@serve.deployment` wrapper that adds the
    serving seams: one batched online feature lookup per batch, dynamic batching
    via @serve.batch with an unbatched twin, autoscaling + max_ongoing_requests
    sized to the batch, live retuning of the batch knobs via reconfigure(),
    structured logging, and metric emission.

The model is loaded once per replica in __init__, never per request.

Run the whole graph with:  serve run app:app   (see app.py)
"""

import asyncio
import logging

import torch
from ray import serve
from ray.serve import metrics

from feature_store import FeatureStore
from model import item_id_to_index, load_model
from schemas import (
    DEFAULT_BATCH_WAIT_TIMEOUT_S,
    DEFAULT_MAX_BATCH_SIZE,
    BatchConfig,
    RankRequest,
    RankResponse,
    ScoredCandidate,
)

# Serve routes deployment logs through the "ray.serve" logger.
logger = logging.getLogger("ray.serve")


class RankerLogic:
    """Pure two-tower scoring: the model and one vectorized forward pass.

    No Ray Serve, no I/O, no metrics. Construct it with a weights path and call
    `forward(requests, user_feats)` directly in a unit test. The serving
    concerns live in the `Ranker` deployment that extends this class.
    """

    def __init__(self, weights_path: str) -> None:
        # Load weights ONCE. This is the expensive step; paying it at
        # construction (not per request) is the single most important habit.
        self.model = load_model(weights_path)

    def forward(
        self, requests: list[RankRequest], user_feats: torch.Tensor
    ) -> list[RankResponse]:
        """One vectorized scoring pass over every candidate in the batch."""
        counts = [len(r.candidate_ids) for r in requests]
        flat_ids = torch.tensor(
            [item_id_to_index(c) for r in requests for c in r.candidate_ids],
            dtype=torch.long,
        )
        with torch.no_grad():
            user_embs = self.model.embed_users(user_feats)  # (N, hidden)
            item_vecs = self.model.embed_items(flat_ids)  # (total_C, hidden)
            # Expand each user's embedding across its own candidates, then a
            # single elementwise-multiply-and-sum scores every pair at once.
            expanded = user_embs.repeat_interleave(
                torch.tensor(counts), dim=0
            )  # (total_C, hidden)
            flat_scores = (expanded * item_vecs).sum(dim=1).tolist()

        responses: list[RankResponse] = []
        offset = 0
        for request, count in zip(requests, counts):
            scored = [
                ScoredCandidate(candidate_id=cid, score=flat_scores[offset + j])
                for j, cid in enumerate(request.candidate_ids)
            ]
            scored.sort(key=lambda s: s.score, reverse=True)
            responses.append(RankResponse(user_id=request.user_id, ranked=scored))
            offset += count
        return responses


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 32,
    },
    # Allow enough in-flight requests for a full batch to form before the
    # autoscaler reacts. Keep max_ongoing_requests >= max_batch_size.
    max_ongoing_requests=64,
    ray_actor_options={"num_cpus": 2},  # GPU variant: {"num_gpus": 1, "num_cpus": 2}
    # Drives reconfigure() at startup and on every live update.
    user_config=BatchConfig().model_dump(),
)
class Ranker(RankerLogic):
    """Serving wrapper around RankerLogic: online feature lookup, dynamic
    batching, metrics, and live reconfigure."""

    def __init__(self, weights_path: str) -> None:
        super().__init__(weights_path)  # loads the model once per replica
        self.store = FeatureStore()
        # Metric-emission seam: these surface in the Serve dashboard panels.
        self.requests_ranked = metrics.Counter(
            "ranker_requests_ranked_total",
            description="Total ranking requests served.",
        )
        self.candidate_set_size = metrics.Histogram(
            "ranker_candidate_set_size",
            description="Number of candidates scored per request.",
            boundaries=[1, 5, 10, 25, 50, 100, 250, 500],
        )
        logger.info("Ranker replica ready (weights=%s)", weights_path)

    def reconfigure(self, config: dict) -> None:
        """Apply batch-tuning knobs live, without a replica restart.

        Called by Serve at startup with `user_config` and again on every
        config update, so changing max_batch_size / batch_wait_timeout_s is a
        zero-downtime operation.
        """
        cfg = BatchConfig(**config)
        self.score_batch.set_max_batch_size(cfg.max_batch_size)
        self.score_batch.set_batch_wait_timeout_s(cfg.batch_wait_timeout_s)
        logger.info(
            "Ranker reconfigured: max_batch_size=%d batch_wait_timeout_s=%s",
            cfg.max_batch_size,
            cfg.batch_wait_timeout_s,
        )

    async def score_unbatched(self, request: RankRequest) -> RankResponse:
        """Score one request on its own, with no cross-request batching."""
        results = await self._score([request])
        return results[0]

    @serve.batch(
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
        batch_wait_timeout_s=DEFAULT_BATCH_WAIT_TIMEOUT_S,
    )
    async def score_batch(self, requests: list[RankRequest]) -> list[RankResponse]:
        """Coalesce concurrent requests into one batch (list in, list out)."""
        logger.info("score_batch coalesced %d request(s)", len(requests))
        return await self._score(requests)

    async def _score(self, requests: list[RankRequest]) -> list[RankResponse]:
        # ONE feature-store round trip for every user in the batch.
        user_ids = [r.user_id for r in requests]
        user_feats = await self.store.batch_get(user_ids)
        # Offload the blocking torch forward so the event loop stays free.
        responses = await asyncio.to_thread(self.forward, requests, user_feats)
        for request in requests:
            self.candidate_set_size.observe(len(request.candidate_ids))
            self.requests_ranked.inc()
        return responses

    async def __call__(
        self, request: RankRequest, use_batching: bool = True
    ) -> RankResponse:
        """Entry point. Batched by default; unbatched twin for comparison."""
        if use_batching:
            return await self.score_batch(request)
        return await self.score_unbatched(request)
