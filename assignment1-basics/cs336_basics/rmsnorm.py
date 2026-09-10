import torch
import torch.nn as nn
import einx


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(
            d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        weight = self.weight.to(torch.float32)
        mean_square = einx.mean("... [d_model]", x * x)
        r_rms = torch.rsqrt(mean_square + self.eps)
        x_norm = einx.multiply("... d_model, ...-> ... d_model", x, r_rms)
        result = einx.multiply(
            "... d_model, d_model -> ... d_model", x_norm, weight)
        return result.to(in_type)
