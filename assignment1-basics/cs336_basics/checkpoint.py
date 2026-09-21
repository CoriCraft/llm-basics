from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    iteration: int,
    out: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    """保存模型权重、优化器状态、当前步数以及模型结构超参数。"""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "iteration": iteration,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if config is not None:
        state["config"] = config

    torch.save(state, out_path)


def load_checkpoint(
    src: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """加载模型权重；若提供了 optimizer 则同时加载优化器状态。"""
    src_path = Path(src)
    checkpoint = torch.load(src_path, map_location="cpu")

    # 兼容多种常见的键名写法
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 关键修复：仅在传入了有效 optimizer 时才恢复状态
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("iteration", 0) if isinstance(checkpoint, dict) else 0
