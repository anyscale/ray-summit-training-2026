"""Async inference: the HTTP front door for a queue-backed worker.

An HTTP request holds a connection open until the reply lands. That is fine for a
30 ms ranking call and wrong for a job that takes minutes: the client blocks, a
timeout somewhere in the path kills the request, and a retry re-runs work that
already succeeded.

This ingress breaks that coupling. A POST enqueues a job and returns a ticket in
milliseconds; a GET reads the result once there is one. The work happens in
`scorer.py`, pulled off the queue by a `@task_consumer` deployment that has no
HTTP surface of its own.

The two are bound as one application, `AsyncScoringAPI.bind(CONFIG, Scorer.bind())`.
Note what that binding is NOT: the ingress never calls the worker's handle. The
data path is the queue, and the handle exists only to pull the consumer into the
same application so one deploy starts both.

Import targets:
  - `app`:       `serve run app:app`.
  - `build_app`: builder for service.yaml / CLI args.

Local run:
    serve run app:build_app
    python client.py            # submit over HTTP, poll until it finishes
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import TaskProcessorConfig
from ray.serve.task_consumer import instantiate_adapter_from_config

from scorer import TASK_PROCESSOR_CONFIG, Scorer

api = FastAPI(title="Async scoring API")


class ScoreRequest(BaseModel):
    """What a client POSTs. No task ids or queue names: those are the server's."""

    user_id: str
    num_candidates: int | None = Field(default=None, ge=1)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
)
@serve.ingress(api)
class AsyncScoringAPI:
    """Accept a job, hand back a ticket, never block on the work.

    This is what makes the pattern usable from anywhere. The adapter needs the
    broker, but only this deployment touches it, so a client needs nothing but an
    HTTP address.
    """

    def __init__(
        self, task_processor_config: TaskProcessorConfig, scorer: DeploymentHandle
    ) -> None:
        # The same config the worker was decorated with, so this enqueues onto the
        # queue that worker drains.
        self.adapter = instantiate_adapter_from_config(
            task_processor_config=task_processor_config
        )
        # Held, never called. Binding the consumer here is what puts it in this
        # application; the work reaches it through the queue, not this handle.
        self._scorer = scorer

    @api.post("/score")
    async def submit(self, request: ScoreRequest) -> dict:
        """Enqueue and return immediately: accepted, not done."""
        task = self.adapter.enqueue_task_sync(
            task_name="score_batch",
            kwargs={
                "user_id": request.user_id,
                "num_candidates": request.num_candidates,
            },
        )
        return {"task_id": task.id, "status": task.status}

    @api.get("/status/{task_id}")
    async def status(self, task_id: str) -> dict:
        """Read a job's state, and its result once there is one."""
        task = self.adapter.get_task_status_sync(task_id)
        return {"task_id": task.id, "status": task.status, "result": task.result}

    @api.delete("/status/{task_id}")
    async def cancel(self, task_id: str) -> dict:
        """Withdraw a job that has not started.

        The operation a synchronous endpoint cannot offer: once a blocking call is
        in flight, a client's only move is to hang up, and the work continues.
        """
        try:
            self.adapter.cancel_task_sync(task_id)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task_id": task_id, "status": "CANCELLED"}

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok"}


class AsyncInferenceArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI.

    `default_candidates` is the job size a client gets when it enqueues without
    one. The queue location is deliberately absent: see the note in scorer.py.
    """

    default_candidates: int = Field(default=50_000, ge=1)


def build_app(args: AsyncInferenceArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service.

    Ingress in front, worker bound behind it, one application. Deploying this
    starts both, and the queue between them is the only thing they share.
    """
    return AsyncScoringAPI.bind(
        TASK_PROCESSOR_CONFIG,
        Scorer.bind(default_candidates=args.default_candidates),
    )


app = build_app(AsyncInferenceArgs())
