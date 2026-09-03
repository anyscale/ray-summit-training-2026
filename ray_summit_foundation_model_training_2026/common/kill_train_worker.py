"""Utilities for killing Ray Train worker actors mid-training.

Used by the notebooks in this section to demo fault tolerance live: schedule a
kill, then call ``trainer.fit()``, and watch Ray Train (and optionally torchft)
recover.

Why SIGKILL via a node-pinned Ray task instead of ``ray.kill``?
---------------------------------------------------------------
``ray.kill(actor_handle, no_restart=True)`` is a forced kill, but it requires
an ``ActorHandle``. Ray Train spawns its workers as *unnamed* actors via
``ray.remote(...)(RayTrainWorker).remote()`` (see
``ray.train.v2._internal.execution.worker_group.worker_group``), so we can't
re-acquire a handle from outside the controller process.

Instead, we use the public state API (``ray.util.state.list_actors``) to find
``RayTrainWorker`` actors and their ``(pid, node_id)``, then schedule a
``num_cpus=0`` task pinned to that node which calls
``os.kill(pid, signal.SIGKILL)``. Effects:

- Process dies abruptly (no atexit handlers run) — this is what triggers Ray
  Train's ``FailureConfig.max_failures`` retry path.
- For torchft runs, the surviving workers detect the missing peer and form a
  new quorum on the next step (no full restart).
- More representative of the real failure modes Wayve sees in production
  (OOM kills, hardware faults, preemption).
"""

from __future__ import annotations

import os
import random
import signal
import threading
import time
from typing import Optional

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors
from ray.util.state.common import ActorState

_TRAIN_WORKER_CLASS = "RayTrainWorker"


def list_workers() -> list[ActorState]:
    """Return all live ``RayTrainWorker`` actors in the cluster."""
    return list_actors(
        filters=[
            ("class_name", "=", _TRAIN_WORKER_CLASS),
            ("state", "=", "ALIVE"),
        ],
    )


def print_workers() -> None:
    """Pretty-print the current live Ray Train workers."""
    workers = list_workers()
    if not workers:
        print("[kill_train_worker] No live RayTrainWorker actors found.")
        return

    print(f"[kill_train_worker] {len(workers)} live RayTrainWorker actor(s):")
    for i, w in enumerate(workers):
        print(
            f"  [{i}] actor_id={w.actor_id}  pid={w.pid}  node_id={w.node_id}"
        )


@ray.remote(num_cpus=0)
def _sigkill_pid(pid: int) -> int:
    os.kill(pid, signal.SIGKILL)
    return pid


def kill_worker(index: Optional[int] = None) -> dict:
    """Kill a single Ray Train worker process with SIGKILL.

    Args:
        index: 0-based index into ``list_workers()``. If ``None``, picks a
            random worker. Index 0 is typically rank 0 in practice but this
            is not guaranteed — Ray Train assigns ranks after actor creation.

    Returns:
        A dict with the killed worker's ``actor_id``, ``pid``, ``node_id``.
    """
    workers = list_workers()
    if not workers:
        raise RuntimeError("No live RayTrainWorker actors found.")

    if index is None:
        target = random.choice(workers)
    else:
        if index < 0 or index >= len(workers):
            raise IndexError(
                f"index={index} out of range; {len(workers)} workers alive"
            )
        target = workers[index]

    if not target.pid or not target.node_id:
        raise RuntimeError(
            f"Target worker is missing pid/node_id (state may be "
            f"transient): {target}"
        )

    # Schedule the SIGKILL on the worker's own node so this works on
    # multi-node clusters without shell/SSH access to remote hosts.
    kill_task = _sigkill_pid.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=target.node_id,
            soft=False,
        ),
    ).remote(target.pid)
    ray.get(kill_task)

    return {
        "actor_id": target.actor_id,
        "pid": target.pid,
        "node_id": target.node_id,
    }


def kill_in(
    seconds: float,
    index: Optional[int] = None,
    *,
    wait_for_workers: float = 300.0,
    poll_interval: float = 2.0,
) -> threading.Thread:
    """Schedule a ``kill_worker`` call and return immediately.

    The kill timer starts only **after** at least one ``RayTrainWorker`` actor
    is observed alive. This matters when the cluster has to autoscale a GPU
    node up before training can start — a plain ``time.sleep(seconds)`` would
    otherwise fire before any worker exists.

    Sequence:
      1. Poll ``list_workers()`` every ``poll_interval`` seconds, for up to
         ``wait_for_workers`` seconds, until at least one worker is alive.
      2. Sleep ``seconds`` (the post-sighting kill delay).
      3. SIGKILL one worker (random by default; or ``index`` if given).

    Lets you queue up the kill *before* the blocking ``trainer.fit()`` cell:

        kill_in(30)            # 30s after the first worker appears
        trainer.fit()

    Args:
        seconds: Delay between first worker sighting and the SIGKILL.
        index: 0-based index into ``list_workers()`` at kill time; ``None``
            picks at random.
        wait_for_workers: Max seconds to wait for the first worker to appear
            (covers autoscale-up). Defaults to 5 minutes.
        poll_interval: Seconds between polls while waiting for workers.
    """

    def _go() -> None:
        deadline = time.monotonic() + wait_for_workers
        announced = False
        while True:
            try:
                workers = list_workers()
            except Exception as e:
                # State API can be transiently unavailable mid-startup; keep polling.
                workers = []
                if not announced:
                    print(
                        f"[kill_train_worker] state API not ready yet ({e!r}); "
                        f"will keep polling.",
                        flush=True,
                    )
            if workers:
                print(
                    f"[kill_train_worker] saw {len(workers)} RayTrainWorker "
                    f"actor(s); waiting {seconds:.0f}s before SIGKILL.",
                    flush=True,
                )
                break
            if time.monotonic() >= deadline:
                print(
                    f"[kill_train_worker] no RayTrainWorker actors appeared "
                    f"within {wait_for_workers:.0f}s — giving up.",
                    flush=True,
                )
                return
            if not announced:
                print(
                    f"[kill_train_worker] waiting for RayTrainWorker actors "
                    f"(up to {wait_for_workers:.0f}s — covers autoscale-up)...",
                    flush=True,
                )
                announced = True
            time.sleep(poll_interval)

        time.sleep(seconds)
        try:
            killed = kill_worker(index=index)
            print(
                f"[kill_train_worker] SIGKILLed worker "
                f"actor_id={killed['actor_id']} "
                f"pid={killed['pid']} "
                f"node_id={killed['node_id']}",
                flush=True,
            )
        except Exception as e:
            print(f"[kill_train_worker] kill failed: {e!r}", flush=True)

    t = threading.Thread(target=_go, daemon=True, name="kill_train_worker")
    t.start()
    return t
