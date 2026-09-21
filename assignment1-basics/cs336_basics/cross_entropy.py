import torch
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"],
) -> torch.Tensor:
    # 1. 提取正确类别的 logit: inputs[y]
    # targets 需要匹配维度以使用 gather
    target_logits = torch.gather(
        inputs, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # 2. 内部流式计算 log-sum-exp，避免显存中保留全量 exp(inputs) 的 6GB 张量
    log_sum_exp = torch.logsumexp(inputs, dim=-1)

    # 3. 交叉熵损失 = log_sum_exp - target_logits
    loss = log_sum_exp - target_logits

    return loss.mean()