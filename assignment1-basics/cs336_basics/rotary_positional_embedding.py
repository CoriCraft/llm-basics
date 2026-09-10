import torch
import torch.nn as nn
import einx


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freq_indices = torch.arange(
            0, d_k, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (freq_indices / d_k))
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32)
        angels = einx.multiply("i, j -> i j", positions, freqs)
        angels = torch.repeat_interleave(angels, repeats=2, dim=-1)

        sin = torch.sin(angels)
        cos = torch.cos(angels)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _rotate_adjacent(self, x: torch.Tensor) -> torch.Tensor:
        """
        构造 [-x1, x0, -x3, x2, ...] 向量用于快速进行 2D 旋转计算：
        [x0, x1] @ [[cos, -sin], [sin, cos]] = [x0*cos - x1*sin, x0*sin + x1*cos]
        即: x * cos + [-x1, x0, ...] * sin
        """
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]
        x_rot[..., 1::2] = x[..., 0::2]
        return x_rot

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        return einx.multiply("... sequence_length d_k, sequence_length d_k -> ... sequence_length d_k",
                             x, cos) + einx.multiply("... sequence_length d_k, sequence_length d_k -> ... sequence_length d_k",
                                                     self._rotate_adjacent(x), sin)
