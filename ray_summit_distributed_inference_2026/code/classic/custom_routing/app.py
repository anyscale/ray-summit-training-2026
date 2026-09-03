"""An app that shows a custom router earning its keep.

The deployment holds a per-replica cache of user features. That cache is the
whole reason to route on user id: a hit skips the feature-store round trip, and a
hit only happens if the same user keeps landing on the same replica.

Two halves make the router work, and both live here rather than in router.py:

  - `record_routing_stats()` is how a replica tells the router about itself. Serve
    calls it every `request_routing_stats_period_s` and publishes the dict as
    `replica.routing_stats`, which is what UserAffinityRouter reads to decide
    whether affinity has gone lopsided.
  - `request_router_config` attaches the router. `request_router_class` is given
    as an import STRING, not the class: the config is serialized to the
    controller, so a class object would not survive the trip.

The endpoint returns `cache_hit` so you can watch affinity work. Fire the same
user id repeatedly and hits climb; fire random ids and they do not.

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run:
    serve run app:build_app
    # same user, twice: the second is a cache hit
    curl -s -XPOST localhost:8000/rank -H 'content-type: application/json' \
        -d '{"user_id": "u1", "candidate_ids": ["a", "b", "c"]}'
"""

import logging
import zlib
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle

from feature_store import EMBED_DIM, FeatureStore

logger = logging.getLogger("ray.serve")

NUM_ITEMS = 10_000

api = FastAPI(title="User-affinity ranking API")


class RankRequest(BaseModel):
    """The router reads `user_id` off this object to pick a replica."""

    user_id: str
    candidate_ids: List[str]


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=16,
    autoscaling_config={"min_replicas": 2, "max_replicas": 6},
    request_router_config=RequestRouterConfig(
        # Import string, not the class: this config travels to the controller.
        request_router_class="router:UserAffinityRouter",
        request_router_kwargs={"imbalance_threshold": 10},
        # How often Serve polls record_routing_stats() below. The default 10s is
        # far too slow to steer on load; 1s keeps the router's view current.
        request_routing_stats_period_s=1,
        request_routing_stats_timeout_s=5,
    ),
)
class Ranker:
    """Scores candidates, caching each user's features on the replica that saw them."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.item_emb = torch.nn.Embedding(NUM_ITEMS, EMBED_DIM)
        self.item_emb.eval()
        self.store = FeatureStore()
        self.feature_cache: Dict[str, torch.Tensor] = {}
        self.ongoing = 0
        self.hits = 0
        self.misses = 0

    def record_routing_stats(self) -> Dict[str, Any]:
        """Publish this replica's state to the router.

        Whatever this returns becomes `replica.routing_stats`. `ongoing_requests`
        is what the router compares across replicas to decide if affinity has gone
        lopsided; the cache numbers are here so you can read hit rate per replica.
        """
        total = self.hits + self.misses
        return {
            "ongoing_requests": self.ongoing,
            "cached_users": len(self.feature_cache),
            "cache_hit_rate": round(self.hits / total, 3) if total else 0.0,
        }

    async def rank(self, request: RankRequest) -> dict:
        self.ongoing += 1
        try:
            cache_hit = request.user_id in self.feature_cache
            if cache_hit:
                self.hits += 1
                user_features = self.feature_cache[request.user_id]
            else:
                self.misses += 1
                user_features = await self.store.get(request.user_id)
                self.feature_cache[request.user_id] = user_features

            item_ids = torch.tensor(
                [zlib.crc32(c.encode("utf-8")) % NUM_ITEMS for c in request.candidate_ids],
                dtype=torch.long,
            )
            with torch.no_grad():
                scores = (self.item_emb(item_ids) @ user_features).tolist()

            ranked = sorted(
                zip(request.candidate_ids, scores), key=lambda p: p[1], reverse=True
            )
            return {
                "user_id": request.user_id,
                "ranked": [{"candidate_id": c, "score": round(s, 4)} for c, s in ranked],
                "cache_hit": cache_hit,
                # Which replica answered, so you can see affinity holding.
                "replica": serve.get_replica_context().replica_id.unique_id,
            }
        finally:
            self.ongoing -= 1


@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class Ingress:
    def __init__(self, ranker: DeploymentHandle) -> None:
        self.ranker = ranker

    @api.post("/rank")
    async def rank(self, request: RankRequest) -> dict:
        # The router inspects this call's arguments, which is how RankRequest's
        # user_id reaches choose_replicas.
        return await self.ranker.rank.remote(request)

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok"}


class RoutingArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI."""

    imbalance_threshold: int = Field(default=10, ge=1)
    min_replicas: int = Field(default=2, ge=1)


def build_app(args: RoutingArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service.

    Two replicas minimum, because a router with one replica to choose from has
    nothing to demonstrate.
    """
    ranker = Ranker.options(
        autoscaling_config={"min_replicas": args.min_replicas, "max_replicas": 6},
        request_router_config=RequestRouterConfig(
            request_router_class="router:UserAffinityRouter",
            request_router_kwargs={"imbalance_threshold": args.imbalance_threshold},
            request_routing_stats_period_s=1,
            request_routing_stats_timeout_s=5,
        ),
    )
    return Ingress.bind(ranker.bind())


app = build_app(RoutingArgs())
