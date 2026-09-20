import torch


def gradient_clipping(params: list[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    梯度裁剪，原地修改参数梯度
    Args:
        params: 模型参数列表，每个param有 .grad 属性
        max_l2_norm: 梯度最大允许L2范数 M
    """
    eps = 1e-6

    # 1. 收集所有不为None的梯度，计算全局L2范数
    grad_norms = []
    for p in params:
        if p.grad is not None:
            grad_norms.append(torch.linalg.norm(p.grad.detach()))
    if not grad_norms:
        # 没有梯度，直接返回
        return

    total_norm = torch.linalg.norm(torch.stack(grad_norms))

    # 2. 判断是否需要裁剪
    if total_norm < max_l2_norm:
        return

    # 3. 计算缩放系数，原地缩放梯度
    scale = max_l2_norm / (total_norm + eps)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)
