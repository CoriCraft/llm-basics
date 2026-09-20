import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从词元序列中随机采样一个批次的输入与其对应的下一个词元目标。

    参数:
        x: 包含一维词元 ID 的 NumPy 数组 (或 np.memmap)。
        batch_size: 批次大小 (B)。
        context_length: 上下文序列长度 (m)。
        device: PyTorch 设备字符串 (如 'cpu', 'cuda:0', 'mps') 或 torch.device 实例。

    返回:
        (inputs, targets):
            - inputs: 形状为 (batch_size, context_length) 的 torch.LongTensor，位于指定设备。
            - targets: 形状为 (batch_size, context_length) 的 torch.LongTensor，位于指定设备。
    """
    n = len(x)

    # 任何起始索引 i 必须满足：i + context_length < n (以保证 target 能取到第 i + context_length 项)
    # 即合法的起始索引范围为 [0, n - context_length - 1]
    max_start_idx = n - context_length
    if max_start_idx <= 0:
        raise ValueError(
            f"数据集长度 ({n}) 必须大于 context_length ({context_length})"
        )

    # 随机生成 batch_size 个起始索引
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # 提取切片：输入为 x[i : i + context_length]，目标为 x[i + 1 : i + context_length + 1]
    # 使用 np.stack 组装为 (batch_size, context_length) 的 NumPy 数组
    inputs_np = np.stack([x[i : i + context_length] for i in start_indices])
    targets_np = np.stack(
        [x[i + 1 : i + 1 + context_length] for i in start_indices]
    )

    # 转换为 PyTorch 张量并转移到指定设备
    # 注意：确保张量类型为 long (int64)，以兼容嵌入层 (Embedding) 与 CrossEntropyLoss
    inputs = torch.from_numpy(inputs_np.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets_np.astype(np.int64)).to(device)

    return inputs, targets