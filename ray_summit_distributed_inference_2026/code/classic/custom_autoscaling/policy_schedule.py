"""Scale on the clock, so capacity leads the traffic instead of chasing it.

Reactive autoscaling cannot beat a cold start. If a batch window opens at 18:00
and replicas only start once requests arrive, the first requests pay the startup.
A schedule has them warm at 17:59.

This policy ignores load entirely, which is the point: it is the clearest possible
demonstration that a policy owns the decision and may disregard every metric.

Contract for any policy (see README.md for the full context reference):
  - Signature is `(AutoscalingContext) -> (target_replicas, policy_state)`.
  - You return the RAW target. Serve applies min/max and the delays on top.
  - It runs on the controller every 0.1s, so it must stay fast.

Use it with:
    policy_function: policy_schedule:policy
"""

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from ray.serve.config import AutoscalingContext


def policy(ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
    """Replica count by hour of day.

    UTC on purpose. Reading the controller's local timezone makes the same config
    behave differently depending on where the cluster happens to run.
    """
    hour = datetime.now(timezone.utc).hour
    if 9 <= hour < 17:
        return 4, {"window": "business hours"}
    if 17 <= hour < 20:
        return 8, {"window": "evening batch"}
    return 1, {"window": "off peak"}
