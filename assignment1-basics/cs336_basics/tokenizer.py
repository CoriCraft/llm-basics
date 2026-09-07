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
        if self.special_tokens:
            self.delimiter_pattern = regex.compile(
                f"({'|'.join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))})"
            )
        else:
            self.delimiter_pattern = None
        self.pretoken_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # vocab: json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {
            int(v): k.encode("utf-8")
            for k, v in raw_vocab.items()
        }
        # merges: txt
        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()
                     and not line.startswith("#")]
        merges: list[tuple[bytes, bytes]] = [
            (line[0].encode("utf-8"), line[1].encode("utf-8"))
            for line in lines
        ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # 1. pre-tokenize
        if self.special_tokens:
            text_parts = self.delimiter_pattern.split(text)
        else:
            text_parts = [text]

        # 2. apply merges
        word_to_wid: dict[bytes, int] = {
            w: wid for wid, w in self.vocab.items()}
        tokenid_list: list[int] = []
        for text_part in text_parts:
            if not text_part:
                continue
            if text_part in self.special_tokens:
                tokenid_list.append(self.special_token_to_id[text_part])
                continue
            for match in self.pretoken_pattern.finditer(text_part):
                old_pretoken: list[bytes] = [bytes([b])
                                             for b in match.group().encode("utf-8")]
                new_pretoken: list[bytes] = []
                for merge in self.merges:
                    new_pretoken = []
                    i = 0
                    while i < len(old_pretoken):
                        if i + 1 < len(old_pretoken) and (old_pretoken[i], old_pretoken[i + 1]) == merge:
                            new_pretoken.append(merge[0] + merge[1])
                            i += 2
                        else:
                            new_pretoken.append(old_pretoken[i])
                            i += 1
                    old_pretoken = new_pretoken
                for w in new_pretoken:
                    tokenid_list.append(word_to_wid[w])
        return tokenid_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

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
