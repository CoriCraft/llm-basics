import torch
import torch.nn as nn
import math
import einx


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features),
                        device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("out_features in_features, ... in_features -> ... out_features", self.weight, x, in_features=self.in_features, out_features=self.out_features)
