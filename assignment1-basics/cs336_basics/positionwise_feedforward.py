import torch
import torch.nn as nn
import einx


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None, dtype: torch.dtype | None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False,
                            device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False,
                            device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False,
                            device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einx.dot(
            "d_ff d_model , ... d_model -> ... d_ff", self.w1.weight, x)
        silu = einx.multiply(
            "... d_ff, ... d_ff -> ... d_ff", torch.sigmoid(w1_x), w1_x)
        w3_x = einx.dot(
            "d_ff d_model , ... d_model -> ... d_ff", self.w3.weight, x)
        mul = einx.multiply("... d_ff, ... d_ff -> ... d_ff", silu, w3_x)
        result = einx.dot(
            "d_model d_ff, ... d_ff -> ... d_model", self.w2.weight, mul)
        return result
