from typing import Iterable, Iterator
import json
import regex


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens or []
        self.vocab_size = len(self.vocab) - len(self.special_tokens)
        self.special_token_to_id: dict[str, int] = {
            s: self.vocab_size + i for i, s in enumerate(self.special_tokens)
        }

        # 1. 预计算反向词表和 BPE 优先级字典，避免在循环中重复构造
        self.word_to_wid: dict[bytes, int] = {
            w: wid for wid, w in self.vocab.items()}
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)}

        if self.special_tokens:
            self.delimiter_pattern = regex.compile(
                f"({'|'.join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))})"
            )
        else:
            self.delimiter_pattern = None

        self.pretoken_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def _bpe(self, piece: bytes) -> list[bytes]:
        """使用 BPE Rank 查找，仅合并当前 piece 中存在的相邻字节对"""
        parts = [bytes([b]) for b in piece]
        if len(parts) <= 1:
            return parts

        while True:
            # 找到当前所有相邻 pair 中 rank 最小（优先级最高）的一个
            min_pair = None
            min_rank = float("inf")
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            # 如果没有可以合并的 pair，说明合并完成
            if min_pair is None or min_rank == float("inf"):
                break

            # 执行该 pair 的合并
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == min_pair:
                    new_parts.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

            if len(parts) <= 1:
                break

        return parts

    def encode(self, text: str) -> list[int]:
        return list(self.encode_generator(text))

    def encode_generator(self, text: str) -> Iterator[int]:
        """流式产出 token ID，不占用额外大内存"""
        if not text:
            return

        # 1. 拆分 special tokens
        text_parts = self.delimiter_pattern.split(
            text) if self.delimiter_pattern else [text]

        for text_part in text_parts:
            if not text_part:
                continue
            if text_part in self.special_tokens:
                yield self.special_token_to_id[text_part]
                continue

            # 2. 正则 pre-token 遍历
            for match in self.pretoken_pattern.finditer(text_part):
                piece = match.group().encode("utf-8")
                # 3. 运行高效 BPE 合并并即时输出 ID
                for subword in self._bpe(piece):
                    yield self.word_to_wid[subword]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐行/逐块流式生成，稳定控制内存 < 1MB"""
        for chunk in iterable:
            yield from self.encode_generator(chunk)

    def decode(self, ids: list[int]) -> str:
        raw_text = b""
        text = ""
        special_token_range = self.vocab_size + len(self.special_tokens)
        for id in ids:
            if id >= self.vocab_size and id < special_token_range:
                text += raw_text.decode("utf-8",
                                        errors='replace') + self.special_tokens[id - self.vocab_size]
                raw_text = b""
                continue
            raw_text += self.vocab[id]
        text += raw_text.decode("utf-8",
                                errors='replace')
        return text
