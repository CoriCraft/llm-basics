import math


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    带warmup的余弦退火学习率调度，LLaMA调度公式

    Args:
        t: 当前迭代步数
        alpha_max: 最大学习率
        alpha_min: 最小/最终学习率
        T_w: warmup预热步数
        T_c: 余弦退火结束步数

    Returns:
        alpha_t: 第t步学习率
    """
    if t < T_w:
        # 预热阶段线性上升
        lr = (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # 余弦退火阶段
        ratio = (t - T_w) / (T_c - T_w)
        cos_term = math.cos(ratio * math.pi)
        lr = alpha_min + 0.5 * (1.0 + cos_term) * (alpha_max - alpha_min)
    else:
        # 退火完成，维持最小学习率
        lr = alpha_min
    return lr
