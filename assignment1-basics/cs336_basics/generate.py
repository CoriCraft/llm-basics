import argparse
import base64
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def load_tokenizer_model(
    vocab_path: Path, merges_path: Path, special_tokens: list[str]
) -> Tokenizer:
    """加载词表和合并规则。"""
    vocab_rows = json.loads(vocab_path.read_text(encoding="utf-8"))
    merge_rows = json.loads(merges_path.read_text(encoding="utf-8"))

    vocab = {
        int(row["id"]): base64.b64decode(row["base64"]) for row in vocab_rows
    }
    merges = [
        (
            base64.b64decode(row["left_base64"]),
            base64.b64decode(row["right_base64"]),
        )
        for row in merge_rows
    ]
    return Tokenizer(vocab, merges, special_tokens)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda",
    eos_token_id: int | None = None,
) -> list[int]:
    """自回归文本生成。"""
    model.eval()
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(
        0
    )

    for _ in range(max_new_tokens):
        idx_cond = (
            idx
            if idx.size(1) <= context_length
            else idx[:, -context_length:]
        )
        logits = model(idx_cond)[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return idx[0].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Transformer 故事生成器（自动加载模型架构）"
    )

    # 必需路径
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/tinystories_v20k/best_model.pt",
        help="权重文件路径 (.pt)",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("data/vocab.json"),
        help="Tokenizer vocab 路径",
    )
    parser.add_argument(
        "--merges_path",
        type=Path,
        default=Path("data/merges.json"),
        help="Tokenizer merges 路径",
    )

    # 生成采样控制参数
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there was a little girl named Lily.",
        help="引导故事生成的开头",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="最多生成的新 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="采样温度（0为贪心，越大越发散）",
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Top-K 候选采样"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 备选架构参数（仅用于兼容未保存 config 的老旧权重文件）
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)

    args = parser.parse_args()

    # 1. 检查并解析 Checkpoint
    checkpoint_file = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_file}")

    ckpt = torch.load(checkpoint_file, map_location="cpu")

    # 2. 自动提取或回退模型结构参数
    if isinstance(ckpt, dict) and "config" in ckpt:
        print("[Info] 成功从 Checkpoint 自动识别模型架构配置！")
        cfg = ckpt["config"]
    else:
        print("[Warning] Checkpoint 未内嵌 config，回退使用命令行默认参数。")
        cfg = {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
        }

    # 3. 构建模型并加载权重
    print(f"正在加载模型并绑定到 {args.device} ...")
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        device=args.device,
    )
    load_checkpoint(checkpoint_file, model, optimizer=None)
    model.to(args.device)

    # 4. 加载分词器
    special_tokens = ["<|endoftext|>"]
    tokenizer = load_tokenizer_model(
        args.vocab_path, args.merges_path, special_tokens
    )
    eos_id = tokenizer.vocab.get(b"<|endoftext|>")

    # 5. 生成文本
    prompt_tokens = list(tokenizer.encode(args.prompt))
    print(f"\n[Prompt]: {args.prompt}")
    print("=" * 60)

    out_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        context_length=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        eos_token_id=eos_id,
    )

    story = tokenizer.decode(out_tokens)
    print(f"[Generated Story]:\n{story}")
    print("=" * 60)


if __name__ == "__main__":
    main()