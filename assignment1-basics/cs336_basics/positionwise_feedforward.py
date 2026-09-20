import torch
import torch.nn as nn
import einx
from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *, device: torch.device | None, dtype: torch.dtype | None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = torch.sigmoid(w1_x) * w1_x
        w3_x = self.w3(x)
        return self.w2(silu * w3_x)
