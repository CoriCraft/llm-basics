import torch
from jaxtyping import Int, Float


def cross_entropy(inputs: Float[torch.Tensor, "batch_size vocab_size"], targets: Int[torch.Tensor, "batch_size"]):
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_logits  # (..., vocab_size)

    # 2. 提取目标类别的 shifted logit: (o[y] - m)
    # 使用 torch.gather 从倒数第一个维度索引出 targets 对应的数值
    target_shifted_logits = torch.gather(
        shifted_inputs,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)  # 形状恢复为 (batch)

    # 3. 计算稳定版 Log-Sum-Exp 消除项: log(sum(exp(o_a - m)))
    log_sum_exp = torch.log(
        torch.sum(torch.exp(shifted_inputs), dim=-1))  # 形状为 (...)

    # 4. 单样本交叉熵损失: -(o[y] - m) + log(sum(exp(o - m)))
    loss = -target_shifted_logits + log_sum_exp

    # 5. 返回整个批次上所有元素的均值标量
    return loss.mean()
