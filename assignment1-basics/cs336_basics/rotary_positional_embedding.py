import torch
import torch.nn as nn
import einx


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, *, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freq_indices = torch.arange(
            0, d_k, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (freq_indices / d_k))
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32)
        angles = einx.multiply("i, j -> i j", positions, freqs)
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
        return (x * cos) + (x_rot * sin)
