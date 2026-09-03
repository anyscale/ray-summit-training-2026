"""The poller: adapt an external queue to Serve, and own what Serve would not.

Work arrives from SQS rather than from a queue Serve manages. That is the right
choice when the queue already exists and other producers write to it, and it is
the trade: three things Ray Serve's task-consumer API would handle become yours.

  - Backpressure. The model handle raises BackPressureError once its queue is
    full (see image_model.py). Catching it, leaving the message on the queue, and
    backing off exponentially is what gives the autoscaler time to add replicas
    instead of being buried.
  - At-least-once delivery. A message is deleted only AFTER its result lands.
    Crash mid-job and SQS redelivers it once the visibility timeout expires.
  - Duplicate suppression. Redelivery is normal, so in-flight message ids are
    tracked and a repeat is skipped rather than processed twice.

Prerequisites: a live SQS queue and an S3 bucket, plus credentials from the
instance role or the standard AWS_* environment variables. This is a reference
implementation, not a workspace demo.
"""

import asyncio
import logging
import os
from io import BytesIO
from typing import Dict

import boto3
from ray import serve
from ray.serve.exceptions import BackPressureError
from ray.serve.handle import DeploymentHandle, DeploymentResponse

logger = logging.getLogger("ray.serve")

# Credentials and resource names come from the environment, never from source. On
# Anyscale the instance role usually supplies the credentials, so only the queue
# and bucket names need setting.
SQS_QUEUE_NAME = os.environ.get("SQS_QUEUE_NAME", "")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2})
class QueuePoller:
    """Pulls from SQS and forwards to the model.

    One replica by design: several pollers on one queue would multiply the
    duplicate handling for no throughput gain, because the model pool is what
    scales, not the poller.
    """

    def __init__(self, model_handle: DeploymentHandle, queue_name: str, bucket: str) -> None:
        self.model_handle = model_handle
        self.bucket = bucket
        # message_id -> task, so a redelivered message in flight is skipped.
        self.processing_requests: Dict[str, asyncio.Task] = {}
        # Consecutive backpressure hits, which sets the backoff.
        self.backpressure_counter = 0

        session = boto3.Session()   # credentials from the instance role or AWS_* env
        self.s3_client = session.client("s3")
        self.queue = session.resource("sqs", region_name=AWS_REGION).get_queue_by_name(
            QueueName=queue_name
        )

        self.shutdown_event = asyncio.Event()
        # Safe in a replica constructor: Serve runs __init__ on the replica's main
        # user-code thread, inside its event loop, never in a threadpool worker.
        self.loop = asyncio.get_running_loop()
        self.loop.create_task(self.run())   # the poll loop outlives this call

    async def run(self) -> None:
        """Poll SQS and forward each message to the model."""
        while not self.shutdown_event.is_set():
            # Long polling: one call waits up to 2s for up to 10 messages, which
            # costs far fewer API calls than spinning on an empty queue.
            messages = await self.loop.run_in_executor(
                None,
                lambda: self.queue.receive_messages(
                    MaxNumberOfMessages=10, WaitTimeSeconds=2
                ),
            )
            for message in messages:
                if message.message_id in self.processing_requests:
                    logger.info("Still processing %s, skipping.", message.message_id)
                    continue

                response = self.model_handle.remote(message.body)
                self.processing_requests[message.message_id] = self.loop.create_task(
                    self.process_finished_request(response, message)
                )

            if self.backpressure_counter == 0:
                await asyncio.sleep(0.1)
            else:
                # 2s, 4s, 8s, capped at 10s. Slowing the poller is what gives the
                # autoscaler room to add replicas.
                backoff_s = min(10, 2**self.backpressure_counter)
                logger.info("Model overloaded, polling again in %ss.", backoff_s)
                await asyncio.sleep(backoff_s)

    async def process_finished_request(
        self, response: DeploymentResponse, queue_message
    ) -> None:
        """Await one result, upload it, and only then delete the message."""
        try:
            image = await response
            filename = await self.loop.run_in_executor(
                None, self._upload_to_s3, image, queue_message.message_id
            )
            logger.info("Uploaded %s to S3.", filename)
        except BackPressureError as exc:
            # Do NOT delete: the message stays on the queue and SQS redelivers it.
            self.backpressure_counter += 1
            logger.info("(%s) %s", queue_message.message_id, exc)
        else:
            self.backpressure_counter = 0
            queue_message.delete()   # only now is the work durably done
            logger.info("Message %s deleted from queue.", queue_message.message_id)
        finally:
            del self.processing_requests[queue_message.message_id]

    def _upload_to_s3(self, image, message_id: str) -> str:
        """Blocking helper: encode the image and put it in the bucket."""
        stream = BytesIO()
        image.save(stream, "PNG")
        stream.seek(0)
        filename = f"image_{message_id}.png"
        self.s3_client.upload_fileobj(
            Fileobj=stream,
            Bucket=self.bucket,
            Key=filename,
            ExtraArgs={"ContentType": "image/png"},
        )
        return filename

    async def __del__(self) -> None:
        """Drain in-flight work before exiting, so no result is lost on a redeploy.

        Serve awaits an `async def __del__` during graceful shutdown, which is what
        makes this drain possible; a plain destructor could not wait on anything.
        """
        self.shutdown_event.set()
        while self.processing_requests:
            logger.info(
                "Processing %d requests, waiting to shut down.",
                len(self.processing_requests),
            )
            await asyncio.sleep(2)
