import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Int, Float

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, eps=eps, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        # 1. 词嵌入: (batch_size, sequence_length, d_model)
        h = self.token_embeddings(in_indices)

        # 2. 依次通过每一个 Transformer 块
        for layer in self.layers:
            h = layer(h)

        # 3. 最终规范化与输出词表 Logits
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits