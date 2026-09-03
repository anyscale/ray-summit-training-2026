"""Scale on the replicas' own CPU and memory, not on queue depth.

Queue depth measures concurrency. It cannot tell a replica holding ten cheap
requests comfortably from a replica at one request and nearly out of memory. Those
need opposite decisions, and only the replica knows which it is.

`ctx.aggregated_metrics` holds whatever the deployment's
`record_autoscaling_stats()` returned, keyed by metric name then replica id. The
metric NAMES are the contract with the deployment: `cpu_percent` and
`memory_percent` here, matching app.py. Rename one on either side and this policy
silently reads an empty dict.

Contract for any policy (see README.md for the full context reference):
  - Signature is `(AutoscalingContext) -> (target_replicas, policy_state)`.
  - You return the RAW target. Serve applies min/max and the delays on top.
  - It runs on the controller every 0.1s, so it must stay fast.

Use it with:
    policy_function: policy_resource_pressure:policy
"""

from typing import Any, Dict, Tuple

from ray.serve.config import AutoscalingContext

# Above either of these, add a replica. Below both of the low marks, give one back.
CPU_HIGH, MEMORY_HIGH = 80.0, 85.0
CPU_LOW, MEMORY_LOW = 30.0, 40.0


def policy(ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
    """One replica step per decision, based on the worst replica's pressure.

    Taking the MAX across replicas is deliberate: one replica about to run out of
    memory justifies capacity even while the others idle, because that replica's
    requests are the ones that will fail.

    Stepping by one rather than jumping to a computed target is also deliberate.
    Resource pressure is noisy, and a policy that leaps on a single spike
    oscillates.
    """
    aggregated = ctx.aggregated_metrics or {}
    cpu_by_replica = aggregated.get("cpu_percent", {})
    memory_by_replica = aggregated.get("memory_percent", {})
    current = ctx.current_num_replicas

    # No replica has reported yet. Hold, and do NOT read the absence as idleness:
    # a deployment that just started has empty metrics, and treating that as low
    # pressure would scale it down before it ever served a request. Missing data
    # is not a signal.
    if not cpu_by_replica and not memory_by_replica:
        return current, {"reason": "no metrics reported yet", "reporting_replicas": 0}

    max_cpu = max(cpu_by_replica.values(), default=0.0)
    max_memory = max(memory_by_replica.values(), default=0.0)

    if max_cpu > CPU_HIGH or max_memory > MEMORY_HIGH:
        target, reason = min(ctx.capacity_adjusted_max_replicas, current + 1), "pressure high"
    elif max_cpu < CPU_LOW and max_memory < MEMORY_LOW:
        target, reason = max(ctx.capacity_adjusted_min_replicas, current - 1), "pressure low"
    else:
        target, reason = current, "in band"

    return target, {
        "reason": reason,
        "max_cpu_percent": round(max_cpu, 1),
        "max_memory_percent": round(max_memory, 1),
        "reporting_replicas": len(cpu_by_replica),
    }
