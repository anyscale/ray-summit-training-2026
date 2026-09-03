"""Mock online feature store with a batched lookup.

Stands in for a DynamoDB-style online store. The one property that matters for
teaching is the batched API: `batch_get` fetches features for many users in a
SINGLE round trip, not one round trip per user. The deployment calls this once
per request batch, so the per-request feature-fetch cost amortizes, the reason
dynamic batching helps a feature-backed ranker.

The simulated latency models a real network round trip; fold it into your
per-batch latency budget when sizing replicas.
"""

import asyncio
import zlib

import torch

from schemas import EMBED_DIM


class FeatureStore:
    """In-memory stand-in for an online feature store (e.g. DynamoDB)."""

    def __init__(self, dim: int = EMBED_DIM, latency_s: float = 0.005) -> None:
        self.dim = dim
        # One simulated round-trip latency for the whole batch, regardless of size.
        self.latency_s = latency_s

    def _features_for(self, user_id: str) -> torch.Tensor:
        """Deterministic feature vector for a user id (seeded by the id)."""
        seed = zlib.crc32(user_id.encode("utf-8"))
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(self.dim, generator=generator)

    async def batch_get(self, user_ids: list[str]) -> torch.Tensor:
        """Fetch features for many users in one round trip.

        Returns a (len(user_ids), dim) float tensor, row i for user_ids[i].
        The single `asyncio.sleep` is the whole point: batch latency is paid
        once per batch, not once per user.
        """
        await asyncio.sleep(self.latency_s)
        if not user_ids:
            return torch.empty((0, self.dim))
        return torch.stack([self._features_for(uid) for uid in user_ids])
