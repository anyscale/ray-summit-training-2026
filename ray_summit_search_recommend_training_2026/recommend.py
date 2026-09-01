import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        # optional: small projection MLPs (kept minimal)
        self.user_proj = nn.Identity()
        self.item_proj = nn.Identity()

    def encode_users(self, user_ids: torch.LongTensor) -> torch.Tensor:
        u = self.user_proj(self.user_emb(user_ids))
        return F.normalize(u, dim=-1)

    def encode_items(self, item_ids: torch.LongTensor) -> torch.Tensor:
        v = self.item_proj(self.item_emb(item_ids))
        return F.normalize(v, dim=-1)

    def forward(self, user_ids: torch.LongTensor, pos_item_ids: torch.LongTensor):
        """
        Returns logits matrix [B,B] where diagonal is the positive pair and
        off-diagonals are in-batch negatives.
        """
        u = self.encode_users(user_ids)         # [B, D]
        v = self.encode_items(pos_item_ids)     # [B, D]
        logits = u @ v.t()                      # [B, B]
        return logits
    
class Recommend():
    def __init__(self, model_location, num_recommendations, num_users, num_items):
        self.model = TwoTower(num_users, num_items)
        self.model.load_state_dict(torch.load(model_location, weights_only=True))
        self.model.eval()
        self.num_recommendations = num_recommendations
        self.num_items = num_items
        
    @torch.no_grad()
    def recommend_topk_batch(self,
        user_ids: torch.LongTensor,     # [B]
        all_item_ids: torch.LongTensor, # [N]
    ):
        """
        Returns:
          topk_indices: [B, k]  (item ids)
          topk_scores:  [B, k]
        """
        model = self.model
        k = self.num_recommendations
        model.eval()

        # Encode
        u = model.encode_users(user_ids)     # [B, D]
        v = model.encode_items(all_item_ids) # [N, D]

        # Similarity
        scores = u @ v.t()                   # [B, N]

        # Top-k per user
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)

        # Map indices back to item ids
        topk_item_ids = all_item_ids[topk_idx]  # [B, k]

        return topk_item_ids, topk_scores
    
    def recommend_for_users(self, users):
        (items, scores) = self.recommend_topk_batch(torch.tensor(users), torch.arange(1, self.num_items))
        return items.numpy()
        
    def __call__(self, batch, column):
        users = batch[column]
        batch['recommended_items'] = self.recommend_for_users(users)
        return batch
