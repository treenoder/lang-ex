"""Download, cache, normalize, and snapshot external text datasets."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from itertools import cycle
from pathlib import Path

from datasets import load_dataset

from tiny_llm.config import PROJECT_ROOT, DataConfig, DataSourceConfig


def _cache_key(config: DataConfig) -> str:
    canonical = json.dumps(config.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


def _huggingface_documents(config: DataSourceConfig, cache_dir: Path) -> Iterable[dict]:
    """Use HF's streaming cache or Arrow cache depending on experiment settings."""
    return load_dataset(
        config.dataset,
        config.subset,
        split=config.split,
        streaming=config.streaming,
        cache_dir=str(cache_dir / "huggingface"),
    )


def _kaggle_documents(config: DataSourceConfig, cache_dir: Path) -> Iterable[dict]:
    """Load CSV/JSON/Parquet files downloaded by kagglehub's authenticated cache."""
    try:
        import kagglehub
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Kaggle support with: uv sync --extra kaggle") from exc

    downloaded = Path(
        kagglehub.dataset_download(config.dataset, output_dir=str(cache_dir / "kaggle"))
    )
    candidates = [
        *downloaded.rglob("*.parquet"),
        *downloaded.rglob("*.jsonl"),
        *downloaded.rglob("*.csv"),
    ]
    if not candidates:
        raise ValueError(
            f"No parquet, jsonl, or csv files found in Kaggle dataset {config.dataset}"
        )
    suffix = candidates[0].suffix
    loader = {".parquet": "parquet", ".jsonl": "json", ".csv": "csv"}[suffix]
    same_type = [str(path) for path in candidates if path.suffix == suffix]
    return load_dataset(loader, data_files=same_type, split="train", streaming=config.streaming)


def _normalize_row(row: dict, source: DataSourceConfig) -> str:
    if source.format == "question_answer":
        answers = row.get("answers", {})
        answer_values = answers.get("text", []) if isinstance(answers, dict) else []
        answer = str(answer_values[0]).strip() if answer_values else ""
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        if not question or not answer:
            return ""
        return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}".strip()

    value = row.get(source.text_column)
    if value is None:
        raise KeyError(
            f"Column {source.text_column!r} is absent from {source.name!r}; "
            f"columns: {list(row)}"
        )
    return str(value).strip()


def _source_texts(source: DataSourceConfig, cache_dir: Path) -> Iterator[str]:
    provider = _huggingface_documents if source.provider == "huggingface" else _kaggle_documents
    written = 0
    for row in provider(source, cache_dir):
        text = _normalize_row(row, source)
        if text:
            yield text
            written += 1
        if written >= source.max_documents:
            return


def _mixed_texts(config: DataConfig, cache_dir: Path) -> Iterator[tuple[str, str]]:
    """Round-robin sources so an early or large dataset cannot dominate the snapshot."""
    active = [(source, _source_texts(source, cache_dir)) for source in config.sources]
    while active:
        for source, documents in cycle(active.copy()):
            try:
                yield source.name, next(documents)
            except StopIteration:
                active = [(item, stream) for item, stream in active if item is not source]
                break


def prepare_corpus(config: DataConfig, cache_dir: Path | None = None) -> Path:
    """Return a stable local JSONL snapshot, downloading only when it is absent.

    A snapshot makes tokenizer training and token packing deterministic even when the
    remote dataset is streamed. The manifest prevents silently mixing configurations.
    """
    root = cache_dir or PROJECT_ROOT / "data" / "cache"
    snapshot_dir = root / "snapshots" / _cache_key(config)
    corpus_path = snapshot_dir / "corpus.jsonl"
    manifest_path = snapshot_dir / "manifest.json"
    if corpus_path.exists() and manifest_path.exists():
        return corpus_path

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    source_counts = {source.name: 0 for source in config.sources}
    # A partial .tmp is never mistaken for a valid cache after interruption.
    temporary = snapshot_dir / "corpus.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        for source_name, text in _mixed_texts(config, root):
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            source_counts[source_name] += 1
    if written < 100:
        raise ValueError(f"Dataset yielded only {written} non-empty documents")
    undersized = {name: count for name, count in source_counts.items() if count < 100}
    if undersized:
        raise ValueError(f"Data sources yielded fewer than 100 documents: {undersized}")
    temporary.replace(corpus_path)
    manifest_path.write_text(
        json.dumps(
            {"config": config.model_dump(), "documents": written, "sources": source_counts},
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus_path


def iter_corpus(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)["text"]
