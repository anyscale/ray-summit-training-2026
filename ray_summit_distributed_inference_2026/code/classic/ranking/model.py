"""The Ranker model: a small two-tower scorer.

A two-tower ranker learns a user tower and an item tower into a shared
embedding space; the relevance score of an item for a user is the dot product
of their embeddings. It is a small, representative production model: enough to
exercise every serving pattern (composition, batching, scaling) without a GPU.

This module is deliberately tiny and CPU-only so the notebooks run end to end
on a CPU workspace. In production you would swap `load_model` for your real
checkpoint loader; nothing else in the serving graph changes.

CLI smoke test:
    python model.py            # build, save random weights, reload, score once
"""

import os
import zlib

import torch
import torch.nn as nn

from schemas import EMBED_DIM, NUM_ITEMS

# Shared embedding dimension for the two towers (the space scores are computed in).
HIDDEN_DIM = 32
# Fixed seed so save_random_weights produces a reproducible model.
_SEED = 0


def item_id_to_index(candidate_id: str) -> int:
    """Map a string candidate id to a stable item-table index.

    Uses CRC32 (not the salted built-in `hash`) so the same id maps to the same
    row on every replica and across processes, because scores must be reproducible.
    """
    return zlib.crc32(candidate_id.encode("utf-8")) % NUM_ITEMS


class TwoTowerRanker(nn.Module):
    """User tower (MLP over features) + item embedding table; score = dot product."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_items: int = NUM_ITEMS,
    ) -> None:
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.item_emb = nn.Embedding(num_items, hidden_dim)

    def embed_users(self, user_feats: torch.Tensor) -> torch.Tensor:
        """(N, embed_dim) feature vectors -> (N, hidden_dim) user embeddings."""
        return self.user_tower(user_feats)

    def embed_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """(M,) item indices -> (M, hidden_dim) item embeddings."""
        return self.item_emb(item_ids)

    def forward(self, user_feat: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score C candidates for ONE user.

        user_feat: (embed_dim,)  item_ids: (C,)  ->  (C,) scores.
        The batched serving path in ranker.py uses `embed_users`/`embed_items`
        directly to score a whole request batch in one vectorized pass; this
        single-user `forward` is the readable reference and the unit-test entry.
        """
        user_emb = self.user_tower(user_feat)  # (hidden_dim,)
        item_vecs = self.item_emb(item_ids)  # (C, hidden_dim)
        return item_vecs @ user_emb  # (C,)


def build_model() -> TwoTowerRanker:
    """Construct the model with reproducible random weights, in eval mode."""
    torch.manual_seed(_SEED)
    model = TwoTowerRanker()
    model.eval()
    return model


def save_random_weights(path: str) -> None:
    """Build a model with random weights and save its state_dict to `path`.

    Stand-in for a real training checkpoint so the serving graph has something
    to load. Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(build_model().state_dict(), path)


def load_model(path: str) -> TwoTowerRanker:
    """Load weights from `path` once per replica and return an eval-mode model."""
    model = TwoTowerRanker()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except Exception:
        print("proceeding with empty weights")
    model.eval()
    return model


if __name__ == "__main__":
    _path = "/tmp/ranker_serve/ranker.pt"
    save_random_weights(_path)
    _model = load_model(_path)
    _feat = torch.randn(EMBED_DIM)
    _ids = torch.tensor([item_id_to_index(c) for c in ["a", "b", "c"]], dtype=torch.long)
    with torch.no_grad():
        print("scores:", _model(_feat, _ids).tolist())
