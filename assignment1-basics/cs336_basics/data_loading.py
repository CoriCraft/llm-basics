import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从词元序列中随机采样一个批次的输入与其对应的下一个词元目标。"""
    n = len(x)
    total_len = context_length + 1

    if n < total_len:
        raise ValueError(
            f"数据集长度 ({n}) 必须大于等于 context_length + 1 ({total_len})"
        )

    # 1. 安全边界：起始点最大只能取到 n - total_len
    # randint(0, high) 生成区间为 [0, high - 1]
    start_indices = np.random.randint(0, n - context_length, size=batch_size)

    # 2. 优化 IO：只切片一次 (长度为 context_length + 1)，减少一半的 memmap 缺页中断
    chunk = np.stack([x[i : i + total_len] for i in start_indices])

    # 3. 转成 PyTorch Tensor 并直接送上 GPU
    # 注意：如果原本数据是 uint32，NumPy 转 int64 很快
    chunk_tensor = torch.from_numpy(chunk.astype(np.int64)).to(
        device, non_blocking=True
    )

    # 4. 在 GPU 显存内切出 inputs 和 targets（零拷贝，秒级完成）
    inputs = chunk_tensor[:, :-1].contiguous()
    targets = chunk_tensor[:, 1:].contiguous()

    return inputs, targets