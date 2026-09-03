"""Submit a job over HTTP and poll for it.

This is the whole client. Note what it does NOT need: no import from app.py, no
task-processor config, no broker credentials, no access to the shared mount. The
ingress owns all of that, so a client needs an address and nothing else. That is
the difference between an async-inference SERVICE and a script that happens to
share a filesystem with its workers.

The exchange is three calls:

    POST /score            -> {"task_id": ..., "status": "PENDING"}
    GET  /status/{task_id}  -> {"status": "SUCCESS", "result": {...}}
    DELETE /status/{task_id} -> withdraw a job that has not started

The POST returns in milliseconds no matter how long the job takes, which is the
point: nothing holds a connection open across the work.

Run after `serve run app:build_app`:
    python client.py
    python client.py --num-candidates 200000 --timeout-s 300
    python client.py --cancel                 # submit, then immediately withdraw
"""

import argparse
import time

import requests

BASE_URL = "http://localhost:8000"

# Celery's terminal states. Anything else means the job is still moving.
DONE_STATES = {"SUCCESS", "FAILURE", "REVOKED"}


def submit(base_url: str, user_id: str, num_candidates: int) -> str:
    """POST a job and return its task id."""
    response = requests.post(
        f"{base_url}/score",
        json={"user_id": user_id, "num_candidates": num_candidates},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["task_id"]


def poll(base_url: str, task_id: str, timeout_s: float = 120.0, interval_s: float = 1.0) -> dict:
    """Poll until the job reaches a terminal state or the timeout expires.

    Blocking in a loop is a convenience for a CLI. A real caller hands the task id
    back to its own user and lets them check later, which is what the ticket is for.
    """
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        response = requests.get(f"{base_url}/status/{task_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in DONE_STATES:
            return payload
        time.sleep(interval_s)
    return {"task_id": task_id, "status": "TIMEOUT", "result": None}


def cancel(base_url: str, task_id: str) -> dict:
    """Withdraw a job. Returns 409 if it is already running."""
    response = requests.delete(f"{base_url}/status/{task_id}", timeout=30)
    return {"http_status": response.status_code, **response.json()}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--num-candidates", type=int, default=50_000)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--cancel", action="store_true",
                        help="submit then immediately withdraw, instead of waiting")
    args = parser.parse_args()

    started = time.perf_counter()
    task_id = submit(args.base_url, args.user_id, args.num_candidates)
    # The submit latency, which is what the client actually waits for.
    print(f"submitted: {task_id}  (in {(time.perf_counter() - started) * 1000:.0f} ms)")

    if args.cancel:
        print(f"cancelled: {cancel(args.base_url, task_id)}")
        return

    outcome = poll(args.base_url, task_id, timeout_s=args.timeout_s)
    print(f"status:    {outcome['status']}")
    print(f"result:    {outcome['result']}")


if __name__ == "__main__":
    _main()
