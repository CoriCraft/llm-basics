import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.linear import Linear


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)

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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        target_shape = [*batch_dims, seq_len, self.num_heads, self.d_k]
        q = q.view(target_shape).transpose(-3, -2)
        k = k.view(target_shape).transpose(-3, -2)
        v = v.view(target_shape).transpose(-3, -2)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        result = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        result = result.transpose(-3, -2).contiguous().view(*
                                                            batch_dims, seq_len, self.d_model)

        return self.o_proj(result)
