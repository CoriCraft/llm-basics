import argparse
import math
import os
import time
from pathlib import Path

from jaxtyping import Float, Int
import numpy as np
import torch
import torch.nn as nn

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data_loading import get_batch
from cs336_basics.transformer_lm import TransformerLM


# ==========================================
# 1. 自定义交叉熵损失函数
# ==========================================
def cross_entropy(
    inputs: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"],
) -> torch.Tensor:
    """数值稳定的手写交叉熵损失函数。"""
    # 提取目标类别的 Logits
    target_logits = torch.gather(
        inputs, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # 使用流式 logsumexp，避免显式分配巨大的 exp(inputs) 激活张量
    log_sum_exp = torch.logsumexp(inputs, dim=-1)

    loss = log_sum_exp - target_logits
    return loss.mean()


# ==========================================
# 2. 学习率调度器 (带预热的余弦退火)
# ==========================================
def get_lr(
    it: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """计算带 Warmup 的 Cosine Decay 学习率。"""
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def load_token_data(
    path: str, data_dtype: str, context_length: int, split_name: str
) -> tuple[np.ndarray, Path]:
    """Load a one-dimensional token array without materializing it in RAM."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"{split_name} 数据文件不存在: {resolved_path}\n"
            "请传入分词后的一维 token 文件（.bin/.dat 或 .npy），而非原始文本文件。"
        )

    if resolved_path.suffix.lower() in {".txt", ".json", ".jsonl", ".csv"}:
        raise ValueError(
            f"{split_name} 数据是原始文本文件: {resolved_path}\n"
            "train.py 只能读取分词后的一维 token 数据。请先将文本用你的 tokenizer "
            "编码并保存为 uint16/uint32/int64 的 .bin/.dat 或 .npy 文件。"
        )

    expected_dtype = np.dtype(data_dtype)
    if resolved_path.suffix.lower() == ".npy":
        data = np.load(resolved_path, mmap_mode="r")
        if not isinstance(data, np.memmap):
            raise ValueError(f"无法以 memmap 方式加载 {resolved_path}")
    else:
        if resolved_path.stat().st_size % expected_dtype.itemsize != 0:
            raise ValueError(
                f"{split_name} 文件大小不能被 {expected_dtype.name} 的字节数整除: "
                f"{resolved_path}。请检查 --data_dtype 是否与写入文件时一致。"
            )
        data = np.memmap(resolved_path, dtype=expected_dtype, mode="r")

    if data.ndim != 1:
        raise ValueError(
            f"{split_name} 数据必须是一维 token 数组，实际形状为"
            f" {data.shape}: {resolved_path}"
        )
    if data.dtype != expected_dtype:
        raise ValueError(
            f"{split_name} 数据 dtype 为 {data.dtype}，但"
            f" --data_dtype={data_dtype}。"
        )
    if len(data) <= context_length:
        raise ValueError(
            f"{split_name} token 数 ({len(data)}) 必须大于 context_length"
            f" ({context_length})。"
        )
    return data, resolved_path


# ==========================================
# 3. 验证集评估函数
# ==========================================
@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: np.ndarray,
    eval_iters: int,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    """在评估集上采样若干步并计算平均损失。"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)  # (batch_size, context_length, vocab_size)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ==========================================
# 4. 主训练流程
# ==========================================
def train(args):
    # ---------------- 设备设置 ----------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU", flush=True)
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS 不可用，自动回退到 CPU", flush=True)
        device = "cpu"

    if args.max_iters <= 0:
        raise ValueError("max_iters 必须为正数。")
    if args.eval_interval <= 0 or args.eval_iters <= 0:
        raise ValueError("eval_interval 和 eval_iters 必须为正数。")
    if args.warmup_iters < 0 or args.lr_decay_iters <= 0:
        raise ValueError("warmup_iters 必须非负，lr_decay_iters 必须为正数。")
    if args.decay_lr and args.warmup_iters >= args.lr_decay_iters:
        raise ValueError(
            "启用学习率衰减时，warmup_iters 必须小于 lr_decay_iters。"
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- 内存映射数据加载 ----------------
    train_data, train_path = load_token_data(
        args.train_path, args.data_dtype, args.context_length, "训练集"
    )
    if args.val_path:
        val_data, val_path = load_token_data(
            args.val_path, args.data_dtype, args.context_length, "验证集"
        )
    else:
        val_data, val_path = None, None

    print(
        "已成功内存映射数据集 | "
        f"训练集: {train_path} ({len(train_data):,} tokens) | "
        f"验证集: {val_path if val_path is not None else '未提供'} "
        f"({len(val_data) if val_data is not None else 0:,} tokens)",
        flush=True,
    )

    # ---------------- 模型架构字典构建 (最佳实践) ----------------
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "rmsnorm_eps": args.rmsnorm_eps,
    }

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.rmsnorm_eps,
        device=device,
    )

    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optim_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
    )

    start_iter = 0

    # ---------------- 从检查点恢复 ----------------
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"正在从检查点恢复训练: {args.resume_from}", flush=True)
            start_iter = load_checkpoint(
                args.resume_from, model, optimizer) + 1
            print(f"已恢复，将从步数 {start_iter} 继续", flush=True)
        else:
            print(
                f"警告: 未找到检查点文件 {args.resume_from}，将从头开始训练。",
                flush=True,
            )

    # ---------------- 训练循环 ----------------
    model.train()
    best_val_loss = float("inf")
    t0 = time.time()
    last_eval_step = start_iter

    print(
        f"开始训练 | 总步数: {args.max_iters} | 运行设备: {device}", flush=True
    )

    for it in range(start_iter, args.max_iters):
        # 1. 动态调整学习率
        lr = (
            get_lr(
                it,
                args.learning_rate,
                args.min_lr,
                args.warmup_iters,
                args.lr_decay_iters,
            )
            if args.decay_lr
            else args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 2. 获取批次数据并执行前向/反向传播
        inputs, targets = get_batch(
            train_data, args.batch_size, args.context_length, device
        )

        logits = model(inputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪防爆炸
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # 3. 评估与指标打印 (跳过第 0 步，避免冷启动漫长等待)
        is_eval_step = (it > 0 and it % args.eval_interval == 0) or (
            it == args.max_iters - 1
        )
        if is_eval_step:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            steps_done = it - last_eval_step
            last_eval_step = it

            # 评估验证集 Loss
            val_loss = None
            if val_data is not None:
                val_loss = estimate_loss(
                    model,
                    val_data,
                    args.eval_iters,
                    args.batch_size,
                    args.context_length,
                    device,
                )
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                val_info = f"val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}"
            else:
                val_info = "no val data"

            train_loss = loss.item()
            train_ppl = math.exp(
                train_loss) if train_loss < 20 else float("inf")
            tokens_per_sec = (
                steps_done * args.batch_size *
                args.context_length / max(dt, 1e-6)
            )

            print(
                f"step {it:6d} | train_loss: {train_loss:.4f} (ppl: {train_ppl:.2f}) | "
                f"{val_info} | lr: {lr:.2e} | {tokens_per_sec:.0f} tok/s",
                flush=True,
            )

            # 保存最优模型 (注入 config)
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = out_dir / "best_model.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    it,
                    best_ckpt_path,
                    config=model_config,
                )

        # 4. 定期保存快照检查点 (注入 config)
        if (
            args.save_interval > 0
            and it > 0
            and (it % args.save_interval == 0 or it == args.max_iters - 1)
        ):
            ckpt_path = out_dir / f"ckpt_iter_{it}.pt"
            save_checkpoint(
                model,
                optimizer,
                it,
                ckpt_path,
                config=model_config,
            )
            print(f"已持久化检查点至: {ckpt_path}", flush=True)

    # 保存最终训练完成状态 (注入 config)
    final_path = out_dir / "final_model.pt"
    save_checkpoint(
        model,
        optimizer,
        args.max_iters - 1,
        final_path,
        config=model_config,
    )
    print(f"训练完成！最终检查点保存在: {final_path}", flush=True)


# ==========================================
# 5. 命令行配置入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer 语言模型训练脚本")

    # I/O 与检查点
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="训练集 .bin/.dat 路径 (memmap)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="",
        help="验证集 .bin/.dat 路径 (可选)",
    )
    parser.add_argument(
        "--data_dtype",
        type=str,
        default="uint32",
        choices=["uint16", "uint32", "int64"],
        help="memmap 数组 dtype",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="检查点与日志输出目录",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="恢复训练的检查点路径 (.pt)",
    )

    # 模型结构超参数
    parser.add_argument(
        "--vocab_size", type=int, default=20000, help="词表大小"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="序列上下文窗口长度 (m)",
    )
    parser.add_argument("--d_model", type=int, default=384, help="隐藏层维度")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Transformer 块层数"
    )
    parser.add_argument("--num_heads", type=int, default=6, help="注意力头数")
    parser.add_argument(
        "--d_ff", type=int, default=1536, help="前馈全连接层隐藏维度"
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="RoPE theta 参数"
    )
    parser.add_argument(
        "--rmsnorm_eps", type=float, default=1e-5, help="RMSNorm epsilon"
    )

    # 优化器与训练超参数
    parser.add_argument(
        "--batch_size", type=int, default=64, help="批次大小 (B)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=8e-4, help="最大学习率"
    )
    parser.add_argument(
        "--min_lr", type=float, default=8e-5, help="最小衰减学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="AdamW 权重衰减系数"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float,
                        default=0.95, help="AdamW beta2")
    parser.add_argument(
        "--adam_eps", type=float, default=1e-8, help="AdamW epsilon"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="梯度范数裁剪阈值"
    )
    parser.add_argument(
        "--max_iters", type=int, default=5000, help="总训练迭代步数"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=250, help="预热步数"
    )
    parser.add_argument(
        "--lr_decay_iters", type=int, default=5000, help="余弦退火结束步数"
    )
    parser.add_argument(
        "--decay_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启学习率衰减（使用 --no-decay_lr 关闭）",
    )

    # 运行时与评估
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=250,
        help="评估与打印日志的间隔步数",
    )
    parser.add_argument(
        "--eval_iters", type=int, default=20, help="评估时随机采样的步数"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="保存检查点的间隔步数"
    )
    parser.add_argument("--seed", type=int, default=1337, help="随机种子")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
