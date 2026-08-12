"""Token packing: turn variable documents into dense fixed-length LM examples."""

import array
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_llm.data.pipeline import iter_corpus
from tiny_llm.data.tokenizer import BPETokenizer


def prepare_tokens(corpus: Path, tokenizer: BPETokenizer, cache_dir: Path) -> Path:
    """Tokenize once and memory-map later, avoiding a large Python object in RAM."""
    token_path = cache_dir / "tokens.npy"
    metadata = cache_dir / "tokens.json"
    if token_path.exists() and metadata.exists():
        return token_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    buffer = array.array("I")
    for text in iter_corpus(corpus):
        buffer.extend(tokenizer.encode(text))
    values = np.frombuffer(buffer, dtype=np.uint32)
    np.save(token_path, values)
    metadata.write_text(json.dumps({"tokens": len(values)}, indent=2), encoding="utf-8")
    return token_path


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path, context_length: int, validation: bool = False) -> None:
        all_tokens = np.load(path, mmap_mode="r")
        split = int(len(all_tokens) * 0.98)
        self.tokens = all_tokens[split:] if validation else all_tokens[:split]
        self.context = context_length

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.context)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.context
        # Copy detaches tensors from the read-only mmap and avoids PyTorch warnings.
        chunk = np.array(self.tokens[start : start + self.context + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])
