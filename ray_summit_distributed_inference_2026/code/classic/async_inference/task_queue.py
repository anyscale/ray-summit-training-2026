"""The queue both halves of the app share.

The ingress enqueues onto this queue and the worker drains it, so they must be
built from the SAME config. A mismatch is the quiet failure mode of this pattern:
jobs are accepted, acknowledged with a task id, and never run.

Named `task_queue` rather than `queue`: a module named `queue.py` here would
shadow the Python standard-library `queue` module for everything on this
directory's `sys.path`, Celery included.

Two required fields worth understanding:
  - `broker_url` carries the work to the workers.
  - `backend_url` carries results back. Both are required; a broker with no
    backend accepts jobs and then has nowhere to put the answer.
"""

import pathlib

from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig

QUEUE_NAME = "score_jobs"


def build_config(queue_root: str) -> TaskProcessorConfig:
    """Filesystem broker in, file result backend out.

    Celery's filesystem transport needs its folders to exist before a worker
    starts, so they are created here rather than assumed. That transport is what
    lets this example run with no Redis and no external infrastructure; it polls a
    directory, so point `broker_url` at redis:// for anything multi-node.
    """
    root = pathlib.Path(queue_root)
    folders = {name: root / name for name in ("in", "out", "processed", "results")}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)

    return TaskProcessorConfig(
        queue_name=QUEUE_NAME,
        adapter_config=CeleryAdapterConfig(
            broker_url="filesystem://",
            backend_url=f"file://{folders['results']}",
            broker_transport_options={
                "data_folder_in": str(folders["in"]),
                "data_folder_out": str(folders["out"]),
                "processed_folder": str(folders["processed"]),
                "store_processed": True,
            },
        ),
        # A task that fails 3 times stops retrying and lands in the DLQ, where it
        # can be inspected instead of silently disappearing.
        max_retries=3,
        failed_task_queue_name=f"{QUEUE_NAME}_dlq",
    )
