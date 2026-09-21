"""Train the project BPE tokenizer and stream TinyStories into token-id files.

Example:
    uv run python -m cs336_basics.train_and_tokenize --vocab-size 10000

The token-id output is little-endian uint32 binary (compact and directly
usable by training code).  ``*.meta.json`` files record its dtype and count.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from array import array
from pathlib import Path
import struct
import time

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


ROOT = Path(__file__).resolve().parent.parent


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def text_path(name: str) -> Path:
    system_path = Path("/data") / name
    repo_path = ROOT / "data" / name
    return system_path if system_path.is_file() else repo_path


def printable(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def save_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: Path, merges_path: Path) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_rows = [
        {"id": i, "text": printable(vocab[i]), "base64": base64.b64encode(vocab[i]).decode("ascii")}
        for i in sorted(vocab)
    ]
    merge_rows = [
        {
            "rank": rank,
            "left": printable(left),
            "right": printable(right),
            "left_base64": base64.b64encode(left).decode("ascii"),
            "right_base64": base64.b64encode(right).decode("ascii"),
        }
        for rank, (left, right) in enumerate(merges)
    ]
    vocab_path.write_text(json.dumps(vocab_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    merges_path.write_text(json.dumps(merge_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(vocab_path: Path, merges_path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_rows = json.loads(vocab_path.read_text(encoding="utf-8"))
    merge_rows = json.loads(merges_path.read_text(encoding="utf-8"))
    vocab = {int(row["id"]): base64.b64decode(row["base64"]) for row in vocab_rows}
    merges = [(base64.b64decode(row["left_base64"]), base64.b64decode(row["right_base64"])) for row in merge_rows]
    return vocab, merges


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path, report_mib: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    total = input_path.stat().st_size
    processed = tokens = 0
    next_report = report_mib * 1024 * 1024
    started = time.monotonic()
    buffer = array("I")
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as source, partial.open("wb") as dest:
        for line in source:
            processed += len(line.encode("utf-8"))
            for token_id in tokenizer.encode_generator(line):
                if token_id < 0 or token_id > 0xFFFFFFFF:
                    raise ValueError(f"token id out of uint32 range: {token_id}")
                buffer.append(token_id)
                if len(buffer) >= 65536:
                    dest.write(struct.pack(f"<{len(buffer)}I", *buffer))
                    tokens += len(buffer)
                    buffer = array("I")
            if processed >= next_report:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(f"[encode] {processed / total * 100:6.2f}% | {processed / elapsed / 2**20:.2f} MiB/s | "
                      f"{tokens + len(buffer):,} tokens | rss={rss_mb():,.1f} MiB", flush=True)
                next_report = processed + report_mib * 1024 * 1024
        if buffer:
            dest.write(struct.pack(f"<{len(buffer)}I", *buffer))
            tokens += len(buffer)
    os.replace(partial, output_path)
    elapsed = max(time.monotonic() - started, 1e-6)
    print(f"[encode] 100.00% | {processed / elapsed / 2**20:.2f} MiB/s | {tokens:,} tokens | rss={rss_mb():,.1f} MiB", flush=True)
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps({"dtype": "uint32-le", "count": tokens, "source": str(input_path)}, indent=2) + "\n", encoding="utf-8"
    )
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--train-input", type=Path, default=text_path("TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid-input", type=Path, default=text_path("TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/vocab.json")
    parser.add_argument("--merges", type=Path, default=ROOT / "data/merges.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/tokenid-list")
    parser.add_argument("--progress-every-merges", type=int, default=100)
    parser.add_argument("--report-every-mib", type=int, default=64)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.special_token = args.special_token or ["<|endoftext|>"]
    if args.vocab_size < 256 + len(args.special_token):
        parser.error("--vocab-size must include 256 byte tokens and special tokens")
    for path in (args.train_input, args.valid_input):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.skip_training:
        vocab, merges = load_model(args.vocab, args.merges)
        print(f"[model] loaded vocab={len(vocab):,}, merges={len(merges):,}, rss={rss_mb():,.1f} MiB")
    else:
        print(f"[train] input={args.train_input} target_vocab={args.vocab_size:,} rss={rss_mb():,.1f} MiB", flush=True)
        started = time.monotonic()
        vocab, merges = train_bpe(args.train_input, args.vocab_size, args.special_token, args.progress_every_merges)
        print(f"[train] complete vocab={len(vocab):,} merges={len(merges):,} elapsed={time.monotonic() - started:.1f}s rss={rss_mb():,.1f} MiB", flush=True)
        save_model(vocab, merges, args.vocab, args.merges)
        print(f"[model] wrote {args.vocab} and {args.merges}", flush=True)
    tokenizer = Tokenizer(vocab, merges, args.special_token)
    for label, source in (("train", args.train_input), ("valid", args.valid_input)):
        output = args.output_dir / f"tinystories_gpt2_{label}.uint32.bin"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite")
        print(f"[encode] {source} -> {output}", flush=True)
        encode_file(tokenizer, source, output, args.report_every_mib)


if __name__ == "__main__":
    main()
