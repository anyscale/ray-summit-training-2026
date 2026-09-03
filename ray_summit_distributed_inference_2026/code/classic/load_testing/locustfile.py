"""Locust load generator for the Ranker endpoint.

Two choices in here carry the lesson.

Open loop, not closed loop. `constant_throughput(1)` caps each simulated user at
one request per second, so the offered rate equals the number of active users no
matter how slow the service gets. A closed-loop generator (`wait_time =
constant(0)`) instead fires as fast as replies come back, which couples offered
load to service latency and hides the very knee this test exists to find.
Because the wait time is one request per second, each stage's `users` value
reads directly as a target RPS.

A shape, not a fixed user count. `RankerTrafficPattern` ramps up, sustains,
spikes, and ramps back down, so one headless run shows both a scale-up and a
scale-down. A flat user count would only ever show the first.

Run against a Ranker on :8000 (see run_locust.sh):
    locust --headless --html results.html -f locustfile.py \
        --host http://localhost:8000
"""

import random

from locust import HttpUser, LoadTestShape, constant_throughput, task

# Candidates per request. The Ranker scores this many items in one forward pass,
# so it sets the per-request work and therefore where the latency knee lands.
CANDIDATES_PER_REQUEST = 32
# Drawn from a fixed pool so repeated runs offer comparable work.
ITEM_POOL = [f"item_{i}" for i in range(1_000)]


class RankerUser(HttpUser):
    """One simulated client issuing at most one ranking call per second."""

    wait_time = constant_throughput(1)

    @task
    def rank(self) -> None:
        self.client.post(
            "/rank",
            name="/rank",
            json={
                "user_id": f"u{random.randrange(10_000)}",
                "candidate_ids": random.sample(ITEM_POOL, CANDIDATES_PER_REQUEST),
            },
        )


class RankerTrafficPattern(LoadTestShape):
    """Ramp, sustain, spike, then drain.

    `users` is the offered RPS (see the open-loop note above). `spawn_rate` is
    users added per second, so 1/30 climbs by one user every 30 seconds, which is
    gradual enough that the autoscaler's upscale_delay_s has time to act.
    """

    stages = [
        {"cumulative_duration": 60, "users": 1, "spawn_rate": 1},        # warm one replica
        {"cumulative_duration": 240, "users": 20, "spawn_rate": 1 / 30}, # gradual ramp
        {"cumulative_duration": 420, "users": 40, "spawn_rate": 1 / 30}, # sustain the plateau
        {"cumulative_duration": 480, "users": 120, "spawn_rate": 4},     # spike
        {"cumulative_duration": 720, "users": 5, "spawn_rate": 4},       # drain, watch scale-down
    ]

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["cumulative_duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
