from collections import defaultdict
import os
import regex


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small!")

    # 1. 严格以 bytes 初始化词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    # 2. 预分词模式
    delimiter_pattern = regex.compile(
        "|".join(regex.escape(s) for s in special_tokens))
    pretoken_pattern = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
        for doc in delimiter_pattern.split(content):
            for match in pretoken_pattern.finditer(doc):
                pretoken_bytes = match.group().encode("utf-8")
                # 每个 pretoken 分解为独立单字节 tuple
                word_tuple = tuple(bytes([b]) for b in pretoken_bytes)
                word_counts[word_tuple] += 1

    # 3. 建立 pair 索引与倒排表
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes],
                        set[tuple[bytes, ...]]] = defaultdict(set)

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    # 4. 执行合并循环
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 频次优先，并列按字典序取大者
        max_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        # 记录 merge 与 vocab
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]

        # 仅针对包含 max_pair 的词进行增量更新
        words_to_update = list(pair_to_words[max_pair])
        del pair_counts[max_pair]
        del pair_to_words[max_pair]

        for word in words_to_update:
            count = word_counts.pop(word)

            # 移除旧 word 对其他 pair 的影响
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p != max_pair:
                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_to_words[p].discard(word)

            # 生成新 word
            new_word_list = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == max_pair:
                    new_word_list.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # 添加新 word 对应的计数与反向索引
            word_counts[new_word] = word_counts.get(new_word, 0) + count
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += count
                pair_to_words[p].add(new_word)

    return vocab, merges
