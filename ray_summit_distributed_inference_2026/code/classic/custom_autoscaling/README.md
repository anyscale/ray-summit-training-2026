# Custom autoscaling policies

The default policy divides in-flight requests by `target_ongoing_requests`. That is
the right signal when requests are uniform and the bottleneck is concurrency. Each
file here scales on a different signal, and each one stands alone.

| File | Signal | Why not queue depth |
|---|---|---|
| `policy_schedule.py` | Hour of day | Reactive scaling cannot beat a cold start. A known batch window wants replicas warm before the traffic, not `upscale_delay_s` after it. |
| `policy_resource_pressure.py` | Replica CPU and memory | Queue depth cannot separate ten cheap requests from one request that is nearly out of memory. Only the replica knows. |
| `policy_queue_latency.py` | Queued fraction plus its trend | Saturated but keeping up is fine; saturated and growing is not. The default sums both into one number. |
| `policy_external_target.py` | An external control plane | The decision is already made somewhere else, and the cluster's job is to honor it. |

`app.py` holds the deployment and its `record_autoscaling_stats()` hook, which is what
makes `policy_resource_pressure.py` possible at all. Swap policies from `service.yaml`:

```yaml
args:
  policy_function: policy_queue_latency:policy
```

## The policy contract

A policy is `(AutoscalingContext) -> (target_replicas, policy_state)`.

- **You return the RAW target.** Serve then applies `min_replicas`, `max_replicas`,
  `upscale_delay_s`, `downscale_delay_s`, and the damping factors on top. Never
  reimplement those inside a policy.
- **It runs on the controller** every `RAY_SERVE_CONTROL_LOOP_INTERVAL_S` (0.1s by
  default), for every deployment. Slow work here stalls the whole control loop, which
  is why `policy_external_target.py` rate-limits its own polling.
- **Never raise.** An exception stalls autoscaling for the deployment. Hold the last
  good value and log instead.
- **`policy_state` is the second return value** and comes back as `ctx.policy_state`
  next tick. For a plain function it is the only way to carry state across iterations.

## AutoscalingContext fields

Verified against the class rather than the docs, which name `total_num_queued_requests`
(the real field is `total_queued_requests`) and omit `total_running_requests`.

| Field | What it holds |
|---|---|
| `current_num_replicas`, `target_num_replicas`, `running_replicas` | Current replica state |
| `total_num_requests` | In flight plus queued |
| `total_queued_requests` | Queued only |
| `total_running_requests` | Derived: in flight, not queued |
| `aggregated_metrics[name]` | `{replica_id: value}` from `record_autoscaling_stats()` |
| `raw_metrics[name]` | `{replica_id: [timestamped values]}` behind the aggregate |
| `capacity_adjusted_min_replicas` / `_max_replicas` | Bounds after any capacity scaling |
| `policy_state` | Whatever the previous tick returned |
| `last_scale_up_time`, `last_scale_down_time`, `current_time` | Timestamps |
| `config`, `deployment_id`, `deployment_name`, `app_name` | Deployment metadata |

`total_num_requests`, `total_queued_requests`, `aggregated_metrics`, and `raw_metrics`
are cached properties that may wrap a lazy callable. Reading one resolves it, so read
each once and reuse the value.

## Collection cadence

`metrics_interval_s` on `AutoscalingConfig` is deprecated. The replica-side push is
governed by environment variables, which `service.yaml` sets:

- `RAY_SERVE_REPLICA_AUTOSCALING_METRIC_RECORD_INTERVAL_S` (0.5s default), how often a
  replica records.
- `RAY_SERVE_REPLICA_AUTOSCALING_METRIC_PUSH_INTERVAL_S` (10s default), how often it
  pushes to the controller. The default is slower than a typical `upscale_delay_s`, so
  a custom-metrics policy left at the default decides on stale readings.

This API is experimental.
