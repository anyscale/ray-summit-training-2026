#!/usr/bin/env bash
# Drive the Ranker and write an HTML report next to this script.
#
# Deploy the pinned single-replica config first, or the reading is meaningless:
#   anyscale service deploy -f benchmark.yaml
#
# --headless runs the shape in locustfile.py to completion with no web UI, so
# this works in a job or CI. Point --host at the Service URL to test a deployed
# Service instead of a local `serve run`.
set -euo pipefail

HOST="${1:-http://localhost:8000}"

locust --headless \
    --html results.html \
    --csv ranker \
    -f locustfile.py \
    --host "$HOST"
