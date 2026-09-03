"""Typed request/response models and shared constants for the Ranker service.

These Pydantic models are the API contract. FastAPI validates incoming JSON
against `RankRequest` at the ingress boundary and serializes `RankResponse` on
the way out, so the same shapes are reused by the notebooks, the load test, and
any client.

Named `schemas` rather than `types` on purpose: a module named `types.py` here
would shadow the Python standard-library `types` module once this directory is
on `sys.path` (which `serve run app:app` does), breaking Serve itself.
"""

from pydantic import BaseModel, Field

# Dimensionality of a user feature vector returned by the online feature store.
EMBED_DIM = 64
# Size of the item embedding table. Candidate ids hash into [0, NUM_ITEMS).
NUM_ITEMS = 10_000

# Dynamic-batching defaults. Kept here so the deployment decorator, the
# `reconfigure()` path, and the notebooks all agree on one source of truth.
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_BATCH_WAIT_TIMEOUT_S = 0.01


class RankRequest(BaseModel):
    """One ranking call: score a user's candidate items."""

    user_id: str
    candidate_ids: list[str]


class ScoredCandidate(BaseModel):
    """A single candidate paired with its relevance score."""

    candidate_id: str
    score: float


class RankResponse(BaseModel):
    """The ranked result, candidates sorted by descending score."""

    user_id: str
    ranked: list[ScoredCandidate]


class BatchConfig(BaseModel):
    """Dynamic-batching knobs, passed via `user_config` and live-tunable.

    `reconfigure()` reads this to retune a running deployment without a restart.
    """

    max_batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, ge=1)
    batch_wait_timeout_s: float = Field(default=DEFAULT_BATCH_WAIT_TIMEOUT_S, ge=0.0)
