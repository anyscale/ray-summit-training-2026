"""An app whose replicas report the signal their autoscaler scales on.

`record_autoscaling_stats()` is the other half of a custom-metrics policy. A
policy can only scale on what it can see, and the controller cannot see inside a
replica: it knows the queue depth because it owns the queue, and nothing else.
This hook is how a replica publishes its own state.

The round trip:

  1. Each replica records its stats every
     `RAY_SERVE_REPLICA_AUTOSCALING_METRIC_RECORD_INTERVAL_S` (0.5s) and pushes
     them every `RAY_SERVE_REPLICA_AUTOSCALING_METRIC_PUSH_INTERVAL_S` (10s).
     The hook must return within `RAY_SERVE_RECORD_AUTOSCALING_STATS_TIMEOUT_S`
     (10s), so it can never do slow work.
  2. The values arrive at the controller as `ctx.aggregated_metrics[name]`,
     keyed by replica id, with `ctx.raw_metrics[name]` holding the timestamped
     series behind it.
  3. The policy named in `autoscaling_config["policy"]` reads them and returns a
     target. One policy per `policy_*.py` module; README.md compares them.

The policy is named as an import STRING (`policy_resource_pressure:policy`),
not a callable. A callable works in-process, so it works from a notebook, but a
Service's config is serialized to the controller and the string is what survives
the trip.

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run:
    serve run app:build_app
    # then drive load and watch replica count track CPU rather than queue depth:
    curl -s -XPOST localhost:8000/work -H 'content-type: application/json' \
        -d '{"iterations": 2000000}'
"""

import logging
import time
import zlib
from typing import Any, Dict

import psutil
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy

logger = logging.getLogger("ray.serve")

api = FastAPI(title="Resource-scaled work API")

# One policy per module, so any single one can be shown on its own. See README.md
# for the contract they share and the AutoscalingContext fields they read.
DEFAULT_POLICY = "policy_resource_pressure:policy"


class WorkRequest(BaseModel):
    """`iterations` sets how CPU-heavy one request is, which is the whole point:
    it lets one request be expensive without the queue ever getting deep."""

    iterations: int = Field(default=1_000_000, ge=1)


@serve.deployment(ray_actor_options={"num_cpus": 1}, max_ongoing_requests=5)
class Worker:
    """CPU-bound work, reporting its own resource use to the autoscaler."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory().total
        self.completed = 0

    def record_autoscaling_stats(self) -> Dict[str, float]:
        """Publish this replica's resource use to the autoscaling policy.

        The metric NAMES are the contract with policy_resource_pressure.py, which
        reads `cpu_percent` and `memory_percent`. Rename one here and the policy
        silently sees an empty dict, so the two move together.

        `cpu_percent(interval=None)` is non-blocking: it reports use since the
        previous call rather than sleeping to sample. Passing an interval would
        block a replica thread on every collection.
        """
        memory = self.process.memory_full_info()
        return {
            "cpu_percent": self.process.cpu_percent(interval=None),
            "memory_percent": (memory.uss / self.system_memory) * 100,
        }

    async def work(self, request: WorkRequest) -> dict:
        started = time.perf_counter()
        checksum = 0
        payload = b"payload"
        for _ in range(request.iterations):
            checksum = zlib.crc32(payload, checksum)
        self.completed += 1
        return {
            "checksum": checksum,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "completed_by_this_replica": self.completed,
            "replica": serve.get_replica_context().replica_id.unique_id,
        }


@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class Ingress:
    def __init__(self, worker) -> None:
        self.worker = worker

    @api.post("/work")
    async def work(self, request: WorkRequest) -> dict:
        return await self.worker.work.remote(request)

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok"}


class AutoscalingArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI.

    `policy_function` is swappable so you can compare the policy_*.py modules
    against each other, and against the default, without editing code.
    """

    policy_function: str = Field(default=DEFAULT_POLICY)
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=6, ge=1)


def build_app(args: AutoscalingArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service."""
    autoscaling = AutoscalingConfig(
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        # Serve applies these AROUND whatever the policy returns, so the policy
        # itself never implements bounds or damping.
        upscale_delay_s=10,
        downscale_delay_s=120,
        policy=AutoscalingPolicy(policy_function=args.policy_function),
    )
    # Collection cadence is NOT set here. `metrics_interval_s` is deprecated in
    # favour of RAY_SERVE_REPLICA_AUTOSCALING_METRIC_PUSH_INTERVAL_S, which
    # defaults to 10s: slower than upscale_delay_s above, so the policy would
    # decide on a reading up to 10s stale. service.yaml lowers it in runtime_env.
    return Ingress.bind(Worker.options(autoscaling_config=autoscaling).bind())


app = build_app(AutoscalingArgs())
