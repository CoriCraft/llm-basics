import torch.nn as nn
import torch
import einx


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding_weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        std = 1
        nn.init.trunc_normal_(self.embedding_weight,
                              mean=0.0, std=std, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_weight[token_ids]
