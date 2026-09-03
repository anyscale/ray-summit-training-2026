"""Wire the poller to the model: QueuePoller -> ImageGenerator.

One deployment drains SQS (poller.py), one runs on the GPU (image_model.py), and
this file joins them. The split matters at runtime, not just for reading: the
poller is pinned to one replica while the model pool autoscales, because scaling
the poller multiplies duplicate handling without adding throughput.

Import target:
  - `build_app`: builder for service.yaml / CLI args.

There is deliberately no module-scope `app`. The builder raises when the queue and
bucket are unset, and a module that raises on import cannot even be read by
tooling.

Local run (needs the queue and bucket to exist):
    export SQS_QUEUE_NAME=... S3_BUCKET_NAME=... AWS_REGION=...
    serve run app:build_app
"""

from pydantic import BaseModel, Field
from ray.serve import Application

from image_model import MODEL_ID, ImageGenerator
from poller import S3_BUCKET_NAME, SQS_QUEUE_NAME, QueuePoller


class StreamArgs(BaseModel):
    """Builder arguments, validated from service.yaml `args:` or the CLI.

    Defaults read the environment so the same builder works from a shell with
    AWS_* exported and from a Service config that names them explicitly.
    """

    queue_name: str = Field(default_factory=lambda: SQS_QUEUE_NAME)
    bucket: str = Field(default_factory=lambda: S3_BUCKET_NAME)
    model_id: str = Field(default=MODEL_ID)


def build_app(args: StreamArgs) -> Application:
    """Application builder: the import target for `serve run` and the Service."""
    if not args.queue_name or not args.bucket:
        raise ValueError(
            "set SQS_QUEUE_NAME and S3_BUCKET_NAME, or pass queue_name= and "
            "bucket= as builder args"
        )
    return QueuePoller.bind(
        ImageGenerator.bind(args.model_id), args.queue_name, args.bucket
    )
