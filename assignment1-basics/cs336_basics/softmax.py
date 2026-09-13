import torch
from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val, _ = torch.max(in_features, dim=dim, keepdim=True)
    sub_max_val = in_features - max_val
    exp_val = torch.exp(sub_max_val)
    sum_exp_val = torch.sum(exp_val, keepdim=True, dim=dim)
    return exp_val / sum_exp_val
