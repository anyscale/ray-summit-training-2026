"""A multi-hop graph worth tracing.

Tracing answers a question metrics cannot: a request took 800 ms, so which hop
spent it? Per-deployment latency histograms tell you each hop's distribution, but
they cannot tie one slow request's hops together. A trace can, because every hop
of a single request shares one trace id.

This graph exists to make that visible. Three deployments in a chain, each with a
different latency profile, so the resulting spans have an obvious shape:

    Ingress  ->  Enricher  ->  Scorer
    (thin)       (I/O bound)   (compute bound)

Tracing itself is configured, not coded: Anyscale's `tracing_config` in
tracing_service.yaml turns it on for the proxy and every replica. Nothing in this
file imports OpenTelemetry, which is the point. Instrument your own spans only
when you need detail INSIDE a deployment; the hop boundaries come free.

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for tracing_service.yaml / CLI args.

Local run (spans are only exported on Anyscale, so this just serves the graph):
    serve run app:build_app
    curl -s -XPOST localhost:8000/score -H 'content-type: application/json' \
        -d '{"text": "a request to trace"}'
"""

import asyncio
import zlib

from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle

api = FastAPI(title="Traced scoring API")


class ScoreRequest(BaseModel):
    text: str


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Scorer:
    """Compute-bound leaf. Its span is the one that should dominate."""

    def __init__(self, work_iterations: int = 200_000) -> None:
        self.work_iterations = work_iterations

    async def score(self, text: str) -> float:
        # Real CPU work, so the span has honest width rather than a sleep.
        checksum = 0
        payload = text.encode("utf-8")
        for _ in range(self.work_iterations):
            checksum = zlib.crc32(payload, checksum)
        return (checksum % 1_000) / 1_000


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Enricher:
    """I/O-bound middle hop, standing in for a feature or metadata lookup."""

    def __init__(self, scorer: DeploymentHandle, lookup_latency_s: float = 0.05) -> None:
        self.scorer = scorer
        self.lookup_latency_s = lookup_latency_s

    async def enrich(self, text: str) -> dict:
        await asyncio.sleep(self.lookup_latency_s)   # the simulated round trip
        score = await self.scorer.score.remote(text)
        return {"text": text, "score": score, "enriched": True}


@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class Ingress:
    """Thin HTTP boundary. Its span should be nearly all child time."""

    def __init__(self, enricher: DeploymentHandle) -> None:
        self.enricher = enricher

    @api.post("/score")
    async def score(self, request: ScoreRequest) -> dict:
        return await self.enricher.enrich.remote(request.text)

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok"}


class TracingArgs(BaseModel):
    """Builder arguments, validated from tracing_service.yaml `args:` or the CLI.

    Both knobs exist so you can shift where the time goes and watch the spans
    change shape: raise work_iterations and the Scorer span dominates, raise
    lookup_latency_s and the Enricher does.
    """

    work_iterations: int = Field(default=200_000, ge=1)
    lookup_latency_s: float = Field(default=0.05, ge=0.0)


def build_app(args: TracingArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service."""
    return Ingress.bind(
        Enricher.bind(
            Scorer.bind(work_iterations=args.work_iterations),
            lookup_latency_s=args.lookup_latency_s,
        )
    )


app = build_app(TracingArgs())
