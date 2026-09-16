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
        # 1. 索引获取 cos 和 sin
        # 如果 token_positions 形状是 (seq_len,) -> (seq_len, d_k)
        # 如果 token_positions 形状是 (batch, seq_len) -> (batch, seq_len, d_k)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 2. 如果 cos 是 (batch, seq_len, d_k)，需要在倒数第 2 维增加 head 维度 (unsqueeze(-2))
        # 变成 (batch, 1, seq_len, d_k)，这样才能与 (batch, num_heads, seq_len, d_k) 正确广播
        if cos.ndim == x.ndim - 1:
            cos = cos.unsqueeze(-3)  # 对应 num_heads 所在的倒数第 3 维
            sin = sin.unsqueeze(-3)
        elif cos.ndim < x.ndim:
            # 如果是 1D positions 索引出来的 (seq_len, d_k)，直接让它从后往前对齐广播
            pass

        # 3. 直接使用 PyTorch 原生广播乘法（比 einx 写固定模式更具通用性且不会因维度增减崩溃）
        return (x * cos) + (self._rotate_adjacent(x) * sin)
