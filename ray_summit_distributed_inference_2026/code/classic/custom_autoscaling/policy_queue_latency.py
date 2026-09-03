"""Scale on the queued fraction, which separates "busy" from "backed up".

A replica saturated but keeping up is fine. A replica whose queue is growing is
not. The default policy cannot tell them apart because it sums both into one
number; `total_queued_requests` is the half that matters.

This is also the clearest use of `policy_state`, the second return value. It comes
back as `ctx.policy_state` on the next tick, which is the only way a plain
function carries state between iterations. Here it holds the previous queue depth
so the policy can act on the TREND rather than on one sample.

Contract for any policy (see README.md for the full context reference):
  - Signature is `(AutoscalingContext) -> (target_replicas, policy_state)`.
  - You return the RAW target. Serve applies min/max and the delays on top.
  - It runs on the controller every 0.1s, so it must stay fast.

Use it with:
    policy_function: policy_queue_latency:policy
"""

from typing import Any, Dict, Tuple

from ray.serve.config import AutoscalingContext

# Queued requests per replica above which the deployment is judged to be behind.
QUEUED_PER_REPLICA_LIMIT = 5


def policy(ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
    """Add capacity only when the queue is both deep AND still growing.

    Requiring both conditions is what stops a burst from triggering a scale-up
    that is already unnecessary by the time the replicas arrive: a deep queue that
    is draining needs no help.
    """
    # Cached properties that may wrap a lazy callable, so read each one once.
    total = ctx.total_num_requests
    queued = ctx.total_queued_requests or 0.0
    current = max(ctx.current_num_replicas, 1)

    previous_queued = float(ctx.policy_state.get("queued", 0.0))
    growing = queued > previous_queued
    queued_per_replica = queued / current

    if queued_per_replica > QUEUED_PER_REPLICA_LIMIT and growing:
        # Scale in proportion to how far behind it is, not by a fixed step.
        target = current + max(1, int(queued_per_replica // QUEUED_PER_REPLICA_LIMIT))
    elif queued == 0 and total / current < 1:
        target = current - 1        # nothing queued and replicas idle: give one back
    else:
        target = current

    return target, {"queued": queued, "growing": growing, "total": total}
