import torch
from torch import Tensor
from jaxtyping import Float, Bool
import einx
from cs336_basics.softmax import softmax
import math


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einx.dot(
        "... queries d_k, ... keys d_k -> ... queries keys", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, -float("inf"))
    softmax_scores = softmax(scores,
                             dim=-1)
    return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", softmax_scores, V)
