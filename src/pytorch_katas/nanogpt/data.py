"""Word-level WikiText-2 loader for the nanoGPT speedrun."""

from __future__ import annotations

import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from pytorch_katas.settings import DATA_DIR

WIKITEXT_BASE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
UNK = "<unk>"
EOS = "<eos>"


def _download(split: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{split}.txt"
    if path.exists():
        return path
    url = f"{WIKITEXT_BASE}/{split}.txt"
    urllib.request.urlretrieve(url, path)
    return path


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens.extend(tok.lower() for tok in TOKEN_RE.findall(line))
        tokens.append(EOS)
    return tokens


def encode(tokens: list[str], stoi: dict[str, int]) -> np.ndarray:
    unk_id = stoi[UNK]
    ids = [stoi.get(tok, unk_id) for tok in tokens]
    return np.asarray(ids, dtype=np.int32)


def load_wikitext(min_freq: int = 2) -> dict:
    data_dir = DATA_DIR / "nanogpt" / "wikitext-2"
    train_text = _download("train", data_dir).read_text(encoding="utf-8")
    val_text = _download("valid", data_dir).read_text(encoding="utf-8")
    train_tokens = tokenize(train_text)
    val_tokens = tokenize(val_text)

    counts = Counter(train_tokens)
    vocab = [UNK]
    vocab.extend(tok for tok, n in counts.most_common() if tok != UNK and n >= min_freq)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = vocab

    train_ids = encode(train_tokens, stoi)
    val_ids = encode(val_tokens, stoi)
    freqs = np.zeros(len(vocab), dtype=np.int64)
    for tok_id in train_ids:
        freqs[tok_id] += 1
    return {
        "train": train_ids,
        "val": val_ids,
        "stoi": stoi,
        "itos": itos,
        "freqs": freqs,
        "vocab_size": len(vocab),
    }
