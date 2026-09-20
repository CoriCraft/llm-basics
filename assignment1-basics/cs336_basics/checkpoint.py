import os
import typing
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """将模型权重、优化器状态以及当前迭代步数打包保存到指定路径或类文件对象中。

    参数:
        model: 需要保存状态的 PyTorch 模型。
        optimizer: 需要保存状态的优化器。
        iteration: 当前训练步数/迭代轮数。
        out: 文件路径或类文件对象（BinaryIO）。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    """从指定路径或类文件对象中加载检查点，恢复模型和优化器状态，并返回迭代步数。

    参数:
        src: 文件路径或类文件对象（BinaryIO）。
        model: 需要恢复权重的 PyTorch 模型。
        optimizer: 需要恢复状态的优化器。

    返回:
        iteration: 检查点中记录的迭代步数 (int)。
    """
    # 针对 PyTorch 2.x+ 建议或默认的 weights_only 机制，若有标量 int 需确保正常反序列化；
    # 也可以显式使用 map_location='cpu' 防止加载到不存在的 GPU 上（如有需要）。
    checkpoint = torch.load(src, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return int(checkpoint["iteration"])