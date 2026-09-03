"""Compose an LLM deployment with an ordinary Serve deployment in one app.

`build_openai_app` hands you a finished OpenAI-compatible service. When you want
the LLM to be one part of a larger application instead, drop one level:
`build_llm_deployment` returns the `LLMServer` deployment on its own, and it
binds exactly like any other Serve deployment.

The example is cheap-then-escalate: a CPU triage deployment answers what it can
from a lookup table, and only the rest reaches the GPU. The routing decision is
an ordinary Python `if`, which is the point -- the composition logic is code,
not a rule in another language.

Two endpoints, so both response shapes are covered:
  - POST /ask         one JSON body, the whole answer at once.
  - POST /ask/stream  OpenAI SSE, forwarded from the engine as tokens arrive.

Import targets (same pattern as ../serving/app.py):
  - `app`       the default graph; `serve run app:app`.
  - `build_app` the builder; `serve run app:build_app model_source=...`
                  and service.yaml (`import_path: app:build_app`).

Local run (one GPU):
    serve run app:build_app
    curl -s localhost:8000/ask        -H 'content-type: application/json' \
         -d '{"text": "what are your support hours?"}'
    curl -N localhost:8000/ask/stream -H 'content-type: application/json' \
         -d '{"text": "Explain a KV cache in three sentences."}'
"""

import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest

from triage import FAQ, Triage

# --- TIER1 MODEL: duplicated by design (self-contained dirs). Sync all sites:
#     grep -rn "TIER1 MODEL" .
DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "s3://anyscale-public-materials-use2/models/Qwen/Qwen2.5-0.5B-Instruct"
# The Run:ai streamer's S3 client reads its region from the AWS SDK env chain; with
# none set it probes instance metadata and fails against a bucket elsewhere.
MODEL_SOURCE_ENV = dict(env_vars={"AWS_REGION": "us-east-2"})

DONE = "data: [DONE]\n\n"

api = FastAPI()


class Ask(BaseModel):
    text: str


def sse_packet(text: str, model_id: str) -> str:
    """One OpenAI chat.completion.chunk, for an answer no engine produced.

    The cheap path still has to speak the streaming wire format, so a client
    consuming /ask/stream sees one contract regardless of which branch answered.
    """
    chunk = {
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": text},
             "finish_reason": None}
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@serve.deployment
@serve.ingress(api)
class Assistant:
    """The ingress. Both collaborators arrive as plain DeploymentHandles."""

    def __init__(self, triage: DeploymentHandle, llm: DeploymentHandle,
                 model_id: str = DEFAULT_MODEL_ID) -> None:
        self.triage = triage
        # LLMServer.chat is an async generator, so the handle must stream.
        self.llm = llm.options(stream=True)
        self.model_id = model_id

    def _chat_request(self, text: str, stream: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": text}],
            max_tokens=96,
            stream=stream,
        )

    @api.post("/ask")
    async def ask(self, body: Ask) -> dict:
        hit = await self.triage.route.remote(body.text)
        if hit is not None:
            return {"answer": FAQ[hit], "used_llm": False}

        # stream is False, so the engine yields exactly ONE ChatCompletionResponse.
        # `return` takes that single item; it is not discarding later chunks.
        request = self._chat_request(body.text, stream=False)
        async for response in self.llm.chat.remote(request, None):
            return {"answer": response.choices[0].message.content, "used_llm": True}

    @api.post("/ask/stream")
    async def ask_stream(self, body: Ask) -> StreamingResponse:
        hit = await self.triage.route.remote(body.text)
        if hit is not None:
            return StreamingResponse(
                iter([sse_packet(FAQ[hit], self.model_id), DONE]),
                media_type="text/event-stream",
            )

        request = self._chat_request(body.text, stream=True)

        async def sse():
            # With stream=True the engine yields a LIST of already-formatted
            # `data: {...}` strings per stream-batching window, so forwarding is
            # a join. The window is 50 ms by default and sets the ITL floor.
            async for batch in self.llm.chat.remote(request, None):
                yield "".join(batch)
            yield DONE

        return StreamingResponse(sse(), media_type="text/event-stream")


def tier1_llm_config(model_id: str, model_source: str) -> LLMConfig:
    """The runnable Tier-1 config: one small model on one GPU."""
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(autoscaling_config=dict(min_replicas=1, max_replicas=1)),
        runtime_env=MODEL_SOURCE_ENV,
        engine_kwargs=dict(
            load_format="runai_streamer",   # required to read an s3:// model_source
            max_model_len=4096,
            enforce_eager=True,
        ),
    )


class ComposeArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI."""

    model_id: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL_ID", DEFAULT_MODEL_ID)
    )
    model_source: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL_SOURCE", DEFAULT_MODEL_SOURCE)
    )


def build_app(args: ComposeArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service.

    The last line is the whole point: `build_llm_deployment(...)` and
    `Triage.bind()` both return an `Application`, so they compose through the
    same `.bind()` seam and the ingress cannot tell them apart.
    """
    llm_config = tier1_llm_config(args.model_id, args.model_source)
    return Assistant.bind(
        Triage.bind(FAQ),
        build_llm_deployment(llm_config),
        model_id=args.model_id,
    )


# Module-scope default so `serve run app:app` works with no args.
app = build_app(ComposeArgs())
