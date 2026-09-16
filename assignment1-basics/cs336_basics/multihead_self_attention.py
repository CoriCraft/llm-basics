import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rotary_positional_embedding import RotaryPositionalEmbedding


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # 单次矩阵乘法完成 Q, K, V 投影 (合并权重)
        self.qkv_project = nn.Linear(
            d_model, 3 * d_model, bias=False, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device)

        # 仅在启用 RoPE 时实例化位置编码模块
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        *batch_dims, seq_len, _ = x.shape

        # 1. 单矩阵乘法投影，并切分为 Q, K, V
        qkv = self.qkv_project(x)  # (*batch_dims, seq_len, 3 * d_model)
        # 各为 (*batch_dims, seq_len, d_model)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 2. 独立出 head 维度，并变换为 (*batch_dims, num_heads, seq_len, d_k)
        target_shape = [*batch_dims, seq_len, self.num_heads, self.d_k]
        q = q.view(target_shape).transpose(-3, -2)
        k = k.view(target_shape).transpose(-3, -2)
        v = v.view(target_shape).transpose(-3, -2)

        # 3. 如果启用了 RoPE，对 q 和 k 施加旋转位置编码
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 4. 因果掩码 (下三角为 True，不允许关注未来位置)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # 5. 点积注意力与头的合并
        result = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        result = result.transpose(-3, -2).contiguous().view(*
                                                            batch_dims, seq_len, self.d_model)

        # 6. 输出投影
        return self.out_proj(result)
