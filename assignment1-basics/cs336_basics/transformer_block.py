import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # 注意：Attention 前和 FFN 前需要两个独立的 Norm 层
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            theta=theta,
            max_seq_len=max_seq_len,
            **factory_kwargs
        )
        self.ln2 = RMSNorm(d_model, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN MHSA + 残差连接
        x = x + self.attn(self.ln1(x))
        # Pre-LN SwiGLU + 残差连接
        x = x + self.ffn(self.ln2(x))
        return x
