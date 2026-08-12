"""A thin, serializable wrapper around Hugging Face's fast BPE tokenizer."""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id("<bos>")  # type: ignore[return-value]

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id("<eos>")  # type: ignore[return-value]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: Path) -> None:
        self._tokenizer.save(str(path), pretty=True)


def train_tokenizer(
    texts: Iterable[str], output: Path, vocab_size: int, min_frequency: int
) -> BPETokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    wrapped = BPETokenizer(tokenizer)
    wrapped.save(output)
    return wrapped
