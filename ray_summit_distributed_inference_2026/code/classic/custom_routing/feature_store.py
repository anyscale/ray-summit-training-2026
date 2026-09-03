"""Stand-in for an online feature store.

The latency is the only interesting property. It is what a per-replica cache hit
saves, and therefore the entire reason UserAffinityRouter exists: 50 ms per miss
is worth routing around, and a router that ignores the cache pays it every time.
"""

import asyncio
import zlib

import torch

EMBED_DIM = 64


class FeatureStore:
    """One user's features per call, after a simulated network round trip."""

    def __init__(self, latency_s: float = 0.05, dim: int = EMBED_DIM) -> None:
        self.latency_s = latency_s
        self.dim = dim

    async def get(self, user_id: str) -> torch.Tensor:
        """Fetch one user's feature vector.

        CRC32 seeds the vector so the same user gets the same features on every
        replica; the built-in hash is salted per process and would not.
        """
        await asyncio.sleep(self.latency_s)
        generator = torch.Generator().manual_seed(zlib.crc32(user_id.encode("utf-8")))
        return torch.randn(self.dim, generator=generator)
