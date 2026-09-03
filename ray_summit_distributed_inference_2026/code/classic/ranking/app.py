"""The Ranker Serve application: a FastAPI ingress in front of the Ranker.

This is the graph the notebooks build toward and the Anyscale Service deploys.
Two application-structure patterns from the "Designing Serve Applications"
material shape it:

  - FastAPI builder + dependency injection: `build_ingress_app()` constructs the
    FastAPI app at replica init, and its routes reach the Ranker by name via
    `serve.get_deployment_handle(...)` inside a FastAPI `Depends`, rather than
    decorating a module-scope `FastAPI()` object and threading handles through a
    constructor.
  - Application builder: `build_app(args)` is the import target, so the model
    weights path arrives as validated `args:` from `service.yaml` instead of an
    environment variable.

Feature prep (a candidate-id dedup) is cheap, so it is fused into the ingress as
a plain in-process function instead of a separate deployment behind a network
hop. The Ranker stays its own deployment: it is the heavy, dynamically batched,
independently scaled model.

Import targets:
  - `app`:       the default graph; `serve run app:app` (notebooks, local).
  - `build_app`: the builder; `serve run app:build_app weights_path=...` and
                  `service.yaml` (`import_path: app:build_app` + `args:`).

Local run:
    python model.py                         # writes /tmp/ranker_serve/ranker.pt
    serve run app:build_app weights_path=/tmp/ranker_serve/ranker.pt
"""

import os

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle

from ranker import Ranker
from schemas import RankRequest, RankResponse


def prepare_features(request: RankRequest) -> RankRequest:
    """Inline feature prep, fused from the old FeaturePrep deployment: drop
    duplicate candidate ids while preserving order. The work is microseconds of
    CPU, so a separate deployment plus an RPC round-trip would be pure overhead;
    it runs in-process in the ingress instead. Real prep might also drop blocked
    items, clip the slate, or normalize ids.
    """
    unique_ids = list(dict.fromkeys(request.candidate_ids))
    return RankRequest(user_id=request.user_id, candidate_ids=unique_ids)


def build_ingress_app() -> FastAPI:
    """Builder: construct the FastAPI ingress at replica init.

    Returning the app from a function (instead of decorating a module-scope
    FastAPI object) defers construction to the replica. The routes are plain
    functions with no `self`, so they reach the Ranker by name through
    `serve.get_deployment_handle` injected via FastAPI `Depends`.
    """
    fastapi_app = FastAPI(
        title="Ranker API",
        description="Two-tower candidate ranking served with Ray Serve.",
    )

    # Resolve the Ranker handle once per replica, not once per request: a fresh
    # handle would spin up a new router each time. app_name is omitted so the
    # handle binds to the current application ("default" under `serve run`,
    # "ranker" under the Service), which keeps this portable.
    ranker_handle: DeploymentHandle | None = None

    def get_ranker() -> DeploymentHandle:
        nonlocal ranker_handle
        if ranker_handle is None:
            ranker_handle = serve.get_deployment_handle("Ranker")
        return ranker_handle

    @fastapi_app.post("/rank", response_model=RankResponse)
    async def rank(
        request: RankRequest,
        ranker: DeploymentHandle = Depends(get_ranker),
    ) -> RankResponse:
        prepared = prepare_features(request)  # fused, in-process
        return await ranker.remote(prepared)

    @fastapi_app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return fastapi_app


# Wrap the builder in an ASGI ingress deployment. Passing no class to
# `serve.ingress` generates one whose routes come from `build_ingress_app`.
# name="Ingress" is deliberate: it keeps the per-deployment override in
# service.yaml matching (the generated default would be "ASGIIngressDeployment").
Ingress = serve.deployment(
    name="Ingress",
    ray_actor_options={"num_cpus": 1},
)(serve.ingress(build_ingress_app)())


# Default weights live on the Anyscale shared mount; override locally with the
# RANKER_WEIGHTS env var. Create the file with model.save_random_weights(path).
DEFAULT_WEIGHTS_PATH = "/mnt/cluster_storage/ranker_serve/ranker.pt"


class RankerAppArgs(BaseModel):
    """Arguments for the application builder.

    `weights_path` is parameterized so `service.yaml` can point production at the
    right weights through its `args:` block, with Pydantic validating the value.
    The default reads RANKER_WEIGHTS so `serve run app:app` keeps working locally.
    """

    weights_path: str = Field(
        default_factory=lambda: os.environ.get("RANKER_WEIGHTS", DEFAULT_WEIGHTS_PATH)
    )


def build_app(args: RankerAppArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service.

    Composes the graph (Ingress -> Ranker; feature prep is fused into the
    ingress) and returns it. Used as `import_path: app:build_app`, with `args`
    supplied from the config file or the `serve run key=value` CLI.
    """
    return Ingress.bind(Ranker.bind(weights_path=args.weights_path))


# Module-scope default graph so `serve run app:app` and the notebooks keep
# working. Production deploys use the builder (app:build_app) with args from
# service.yaml.
app = build_app(RankerAppArgs())
