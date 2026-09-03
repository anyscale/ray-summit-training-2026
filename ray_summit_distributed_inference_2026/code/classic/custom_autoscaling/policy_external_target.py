"""A class-based policy, for when the target comes from outside the cluster.

Use a class instead of a function when the policy needs setup a function cannot
hold: a persistent connection, a background poller, a cache. The instance lives
only on the controller and is never serialized, so it may hold non-picklable
state such as an open socket or an asyncio task. Only `policy_kwargs` has to be
JSON-serializable.

This one reads a replica count from a JSON file, standing in for whatever external
control plane owns the decision.

Contract for any policy (see README.md for the full context reference):
  - Signature is `(AutoscalingContext) -> (target_replicas, policy_state)`, here
    as `__call__`.
  - You return the RAW target. Serve applies min/max and the delays on top.
  - It runs on the controller every 0.1s, so it must stay fast.

`policy_kwargs` requires Ray 2.56 or newer. Use it with:

    policy:
      policy_function: policy_external_target:ExternalTargetPolicy
      policy_kwargs:
        file_path: /mnt/cluster_storage/custom_autoscaling/target.json
        poll_interval_s: 5.0
"""

import json
import logging
import pathlib
from typing import Any, Dict, Tuple

from ray.serve.config import AutoscalingContext

logger = logging.getLogger("ray.serve")


class ExternalTargetPolicy:
    """Poll an external target on an interval, and return the last value seen."""

    def __init__(self, file_path: str, poll_interval_s: float = 5.0) -> None:
        self._file_path = pathlib.Path(file_path)
        self._poll_interval_s = poll_interval_s
        self._desired_replicas = 1
        self._last_poll_s = 0.0

    def _maybe_poll(self, now_s: float) -> None:
        """Read the target, but no more often than poll_interval_s.

        The rate limit is the reason this class exists. The control loop runs ten
        times a second, and touching the filesystem or an HTTP endpoint at that
        rate is exactly the stall that blocks autoscaling for every deployment.
        """
        if now_s - self._last_poll_s < self._poll_interval_s:
            return
        self._last_poll_s = now_s
        try:
            self._desired_replicas = int(json.loads(self._file_path.read_text())["replicas"])
        except Exception as exc:
            # Never raise from a policy: an exception here stalls autoscaling for
            # this deployment. Hold the last good value and say so.
            logger.warning("ExternalTargetPolicy could not read the target: %s", exc)

    def __call__(self, ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
        self._maybe_poll(ctx.current_time or 0.0)
        return self._desired_replicas, {
            "source": str(self._file_path),
            "target": self._desired_replicas,
        }
